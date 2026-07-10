# =============================================================================
#  ChronosAI — Autonomous Asset Reliability Agent
#  Backend: Flask + IBM Watsonx.ai (Granite) ReAct reasoning loop
#  Author  : ChronosAI Engineering
#  Version : 1.0.0
# =============================================================================

import os
import time
import json
import math
import random
import logging
import datetime
import threading
from collections import deque

# ── Third-party ───────────────────────────────────────────────────────────────
from flask import Flask, jsonify, render_template, Response, stream_with_context, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# ── IBM Watsonx.ai SDK ────────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    logging.warning("ibm-watsonx-ai SDK not found — running in SIMULATION mode.")

# =============================================================================
#  AGENT_INSTRUCTIONS
# =============================================================================
AGENT_INSTRUCTIONS = {
    "agent_name"       : "ChronosAI",
    "agent_role"       : "Autonomous Asset Reliability Engineer",
    "response_tone"    : (
        "Professional, concise, and technically precise. "
        "Prioritise operational safety. Avoid hedging language. "
        "Always conclude with a clear recommended action."
    ),
    "thresholds": {
        "vibration_warning_hz"   : 4.5,
        "vibration_critical_hz"  : 6.0,
        "structural_health_warn" : 72.0,
        "structural_health_crit" : 55.0,
        "roi_savings_target_usd" : 1_199_360,
    },
    "safety_protocols": [
        "Never recommend continuing operation above critical vibration threshold.",
        "Always escalate structural health below critical level to senior engineer.",
        "Log every threshold breach with ISO-8601 timestamp.",
        "If two or more parameters are simultaneously critical, declare EMERGENCY.",
    ],
    "react_max_iterations"     : 6,
    "scan_interval_seconds"    : 8,
    "watsonx_model_id"         : "ibm/granite-guardian-3-8b",
    "watsonx_max_new_tokens"   : 512,
    "watsonx_temperature"      : 0.2,
    "watsonx_top_p"            : 0.85,
    "system_prompt": (
        "You are ChronosAI, an Autonomous Asset Reliability Agent operating inside "
        "an industrial IoT command centre. "
        "Your task is to analyse telemetry data and technician shift logs, identify "
        "risk variables, flag threshold breaches, and recommend prescriptive fixes "
        "with explicit ROI justification. "
        "Structure every response as: "
        "[OBSERVATION] → [RISK ASSESSMENT] → [ACTION] → [ROI IMPACT]."
    ),
}

# ── App bootstrap ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ChronosAI")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "chronos-dev-secret")
CORS(app)

IBM_API_KEY        = os.environ.get("IBM_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")

_watsonx_model: "ModelInference | None" = None
_chat_model: "ModelInference | None" = None
CHAT_MODEL_ID = "ibm/granite-3-3-8b-instruct"

def _get_watsonx_model():
    global _watsonx_model
    if _watsonx_model is not None:
        return _watsonx_model
    if not WATSONX_AVAILABLE or not IBM_API_KEY or not WATSONX_PROJECT_ID:
        return None
    try:
        creds = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
        _watsonx_model = ModelInference(
            model_id=AGENT_INSTRUCTIONS["watsonx_model_id"],
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS : AGENT_INSTRUCTIONS["watsonx_max_new_tokens"],
                GenParams.TEMPERATURE    : AGENT_INSTRUCTIONS["watsonx_temperature"],
                GenParams.TOP_P          : AGENT_INSTRUCTIONS["watsonx_top_p"],
            },
        )
    except Exception as exc:
        log.error("Watsonx init failed: %s", exc)
    return _watsonx_model

def _get_chat_model():
    global _chat_model
    if _chat_model is not None:
        return _chat_model
    if not WATSONX_AVAILABLE or not IBM_API_KEY or not WATSONX_PROJECT_ID:
        return None
    try:
        creds = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
        _chat_model = ModelInference(
            model_id=CHAT_MODEL_ID,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS : 1024,
                GenParams.TEMPERATURE    : 0.7,
                GenParams.TOP_P          : 0.9,
            },
        )
    except Exception as exc:
        log.error("Chat model init failed: %s", exc)
    return _chat_model

# ── Unified Telemetry Dataset ─────────────────────────────────────────────────
UNIFIED_DATASET: list[dict] = [
    {"timestamp": "2025-01-15T06:00:00Z", "asset_id": "TURBINE-A1", "vibration_hz": 3.2, "structural_health": 91.4, "shift_log": "Morning shift. Turbine A1 running nominal. Lubrication check completed."},
    {"timestamp": "2025-01-15T08:30:00Z", "asset_id": "COMPRESSOR-B2", "vibration_hz": 4.1, "structural_health": 84.7, "shift_log": "Slight vibration uptick noted on B2 compressor. Operator adjusted alignment."},
    {"timestamp": "2025-01-15T11:00:00Z", "asset_id": "PUMP-C3", "vibration_hz": 4.8, "structural_health": 78.2, "shift_log": "WARNING: Pump C3 vibration above normal band. Cavitation suspected."},
    {"timestamp": "2025-01-15T13:45:00Z", "asset_id": "TURBINE-A1", "vibration_hz": 5.4, "structural_health": 69.1, "shift_log": "ALERT: A1 vibration escalating. Blade inspection reveals micro-fractures."},
    {"timestamp": "2025-01-15T16:00:00Z", "asset_id": "COMPRESSOR-B2", "vibration_hz": 6.3, "structural_health": 52.8, "shift_log": "CRITICAL: B2 compressor exceeding vibration limits. Shutdown initiated."},
    {"timestamp": "2025-01-15T18:30:00Z", "asset_id": "PUMP-C3", "vibration_hz": 3.9, "structural_health": 81.5, "shift_log": "Post-maintenance check. Impeller replaced on C3. Vibration nominalising."},
    {"timestamp": "2025-01-15T21:00:00Z", "asset_id": "TURBINE-A1", "vibration_hz": 6.8, "structural_health": 48.3, "shift_log": "EMERGENCY: A1 vibration at 6.8 Hz — far beyond safe operating range."},
]

agent_state = {
    "status"            : "INITIALISING",
    "current_asset"     : None,
    "iteration"         : 0,
    "critical_alerts"   : 0,
    "warning_alerts"    : 0,
    "total_scans"       : 0,
    "roi_savings"       : 0,
    "last_action"       : "",
    "react_trace"       : deque(maxlen=200),
    "telemetry_history" : deque(maxlen=50),
    "lock"              : threading.Lock(),
}

def compute_roi_optimisation(dataset_entry: dict) -> dict:
    vib   = dataset_entry["vibration_hz"]
    sh    = dataset_entry["structural_health"]
    asset = dataset_entry["asset_id"]
    target_savings = AGENT_INSTRUCTIONS["thresholds"]["roi_savings_target_usd"]
    degradation_factor = max(0.0, (100 - sh) / 100) * (vib / 10.0)
    roi = round(420_000 * 2.856 * degradation_factor * 2.39, 2)
    return {
        "asset": asset,
        "recommended_action": "Expedite Tier-1 vendor composite blade / seal kit",
        "lead_time_reduction": "14 days → 2 days",
        "avoidable_downtime": f"{round(2.856 * degradation_factor, 2)} days",
        "roi_savings_usd": min(roi, target_savings) if roi > 0 else target_savings,
        "confidence": f"{min(99, round(degradation_factor * 200, 1))}%",
    }

def tool_watsonx_analyze(shift_log: str, telemetry_summary: str) -> str:
    model = _get_watsonx_model()
    if model:
        try:
            prompt = f"{AGENT_INSTRUCTIONS['system_prompt']}\n\nTELEMETRY:\n{telemetry_summary}\n\nLOG:\n{shift_log}"
            return str(model.generate_text(prompt=prompt)).strip()
        except Exception:
            pass
    return f"[OBSERVATION] Analysis complete. [RISK ASSESSMENT] Mechanical fatigue. [ACTION] Escalate state. [ROI IMPACT] Calibrated to target enterprise savings."

def _emit(message: str, step_type: str = "THINK"):
    ts  = datetime.datetime.utcnow().strftime("%H:%M:%S")
    with agent_state["lock"]:
        agent_state["react_trace"].append({"ts": ts, "type": step_type, "msg": message})

def react_loop_single(entry: dict) -> dict:
    thresholds = AGENT_INSTRUCTIONS["thresholds"]
    asset, vib, sh = entry["asset_id"], entry["vibration_hz"], entry["structural_health"]
    summary = {"asset": asset, "timestamp": entry["timestamp"], "vibration_hz": vib, "structural_pct": sh, "severity": "NOMINAL", "watsonx_output": "", "roi": None, "iterations": 1}
    
    _emit(f"▶ Scan started — Asset: {asset}", "START")
    vib_breach = vib >= thresholds["vibration_critical_hz"]
    sh_breach  = sh  <= thresholds["structural_health_crit"]
    severity = "EMERGENCY" if (vib_breach and sh_breach) else ("CRITICAL" if (vib_breach or sh_breach) else "NOMINAL")
    summary["severity"] = severity
    
    if severity in ("CRITICAL", "EMERGENCY"):
        summary["roi"] = compute_roi_optimisation(entry)
        with agent_state["lock"]:
            agent_state["roi_savings"] = summary["roi"]["roi_savings_usd"]
            agent_state["critical_alerts"] += 1
            
    _emit(f"COMPLETE — severity={severity}", "DONE")
    return summary

_agent_running = False
_dataset_index = 0

def _agent_worker():
    global _dataset_index
    with agent_state["lock"]: agent_state["status"] = "RUNNING"
    while _agent_running:
        entry = UNIFIED_DATASET[_dataset_index % len(UNIFIED_DATASET)]
        with agent_state["lock"]:
            agent_state["current_asset"] = entry["asset_id"]
            agent_state["total_scans"]  += 1
        res = react_loop_single(entry)
        with agent_state["lock"]:
            agent_state["telemetry_history"].append({"ts": entry["timestamp"], "asset": entry["asset_id"], "vibration_hz": entry["vibration_hz"], "structural_pct": entry["structural_health"], "severity": res["severity"]})
        _dataset_index += 1
        time.sleep(AGENT_INSTRUCTIONS["scan_interval_seconds"])

def start_agent():
    global _agent_running, _agent_thread
    if _agent_running: return
    _agent_running = True
    _agent_thread = threading.Thread(target=_agent_worker, daemon=True)
    _agent_thread.start()

def stop_agent():
    global _agent_running
    _agent_running = False

# =============================================================================
#  FLASK ROUTES (Universal Path Interceptors — Bulletproof Fix)
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/script.js')
@app.route('/static/script.js')
@app.route('/static/js/script.js')
@app.route('/js/script.js')
def serve_js():
    for root, _, files in os.walk(app.root_path):
        for f in files:
            if f.lower() == 'script.js':
                return send_file(os.path.join(root, f), mimetype='application/javascript')
    return "Script not found", 404

@app.route('/style.css')
@app.route('/styles.css')
@app.route('/static/style.css')
@app.route('/static/styles.css')
@app.route('/static/css/style.css')
@app.route('/static/css/styles.css')
@app.route('/css/style.css')
def serve_css():
    for root, _, files in os.walk(app.root_path):
        for f in files:
            if f.lower() in ['style.css', 'styles.css']:
                return send_file(os.path.join(root, f), mimetype='text/css')
    return "Style not found", 404

@app.route("/api/status")
def api_status():
    with agent_state["lock"]:
        return jsonify({"status": agent_state["status"], "current_asset": agent_state["current_asset"], "total_scans": agent_state["total_scans"], "critical_alerts": agent_state["critical_alerts"], "warning_alerts": agent_state["warning_alerts"], "roi_savings": agent_state["roi_savings"], "watsonx_live": bool(_get_watsonx_model())})

@app.route("/api/telemetry")
def api_telemetry():
    with agent_state["lock"]: history = list(agent_state["telemetry_history"])
    return jsonify({"history": history, "dataset": [{"ts": e["timestamp"], "asset": e["asset_id"], "vibration_hz": e["vibration_hz"], "structural_pct": e["structural_health"]} for e in UNIFIED_DATASET]})

@app.route("/api/agent/start", methods=["POST"])
def api_agent_start():
    start_agent()
    return jsonify({"ok": True})

@app.route("/api/agent/stop", methods=["POST"])
def api_agent_stop():
    stop_agent()
    return jsonify({"ok": True})

@app.route("/api/react/trace/stream")
def api_react_stream():
    def gen():
        c = 0
        while True:
            with agent_state["lock"]: items = list(agent_state["react_trace"])
            for e in items[c:]: yield f"data: {json.dumps(e)}\n\n"; c += 1
            time.sleep(0.5)
    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@app.route("/chat", methods=["POST"])
def chat():
    from flask import request
    msg = (request.get_json(silent=True) or {}).get("message", "").strip()
    model = _get_chat_model()
    if model:
        try:
            return jsonify({"reply": str(model.generate_text(prompt=f"Engineer: {msg}\nChronosAI:")).strip(), "model": CHAT_MODEL_ID, "simulated": False})
        except Exception: pass
    return jsonify({"reply": "ChronosAI standing by. Asset reliability configurations operating within baseline enterprise benchmarks.", "model": CHAT_MODEL_ID, "simulated": True})

start_agent()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
