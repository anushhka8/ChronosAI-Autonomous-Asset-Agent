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
#  ─────────────────────────────────────────────────────────────────────────────
#  Customise every aspect of the agent's behaviour here.
#  Changes here propagate automatically into every Watsonx.ai call and every
#  ReAct reasoning step without touching any other part of the code.
# =============================================================================
AGENT_INSTRUCTIONS = {

    # ── Identity & Tone ───────────────────────────────────────────────────────
    "agent_name"       : "ChronosAI",
    "agent_role"       : "Autonomous Asset Reliability Engineer",
    "response_tone"    : (
        "Professional, concise, and technically precise. "
        "Prioritise operational safety. Avoid hedging language. "
        "Always conclude with a clear recommended action."
    ),

    # ── Decision Thresholds ───────────────────────────────────────────────────
    "thresholds": {
        "vibration_warning_hz"   : 4.5,    # Hz — yellow alert
        "vibration_critical_hz"  : 6.0,    # Hz — red alert / stop asset
        "structural_health_warn" : 72.0,   # % — schedule inspection
        "structural_health_crit" : 55.0,   # % — immediate intervention
        "roi_savings_target_usd" : 1_199_360,  # prescriptive optimisation target
    },

    # ── Safety Protocols ─────────────────────────────────────────────────────
    "safety_protocols": [
        "Never recommend continuing operation above critical vibration threshold.",
        "Always escalate structural health below critical level to senior engineer.",
        "Log every threshold breach with ISO-8601 timestamp.",
        "If two or more parameters are simultaneously critical, declare EMERGENCY.",
    ],

    # ── Operational Logic ────────────────────────────────────────────────────
    "react_max_iterations"     : 6,      # max ReAct loop steps per scan cycle
    "scan_interval_seconds"    : 8,      # how often the background agent wakes
    "watsonx_model_id"         : "ibm/granite-guardian-3-8b",
    "watsonx_max_new_tokens"   : 512,
    "watsonx_temperature"      : 0.2,
    "watsonx_top_p"            : 0.85,

    # ── System prompt injected into every Watsonx call ────────────────────────
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
# =============================================================================
#  END AGENT_INSTRUCTIONS
# =============================================================================

# ── App bootstrap ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ChronosAI")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "chronos-dev-secret")
CORS(app)

# ── Watsonx credentials (Securely linked to environment variables) ────────────
IBM_API_KEY        = os.environ.get("IBM_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")

_watsonx_model: "ModelInference | None" = None

# ── Separate lazy-init client for the Chat model ─────────────────────────────
_chat_model: "ModelInference | None" = None
CHAT_MODEL_ID = "ibm/granite-3-3-8b-instruct"

def _get_watsonx_model():
    """Lazy-init the Watsonx model inference client (agent / ReAct use)."""
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
        log.info("Watsonx.ai model initialised: %s", AGENT_INSTRUCTIONS["watsonx_model_id"])
    except Exception as exc:
        log.error("Watsonx init failed: %s", exc)
        _watsonx_model = None
    return _watsonx_model


def _get_chat_model():
    """Lazy-init the granite-3-3-8b-instruct client for the Chat interface."""
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
        log.info("Chat model initialised: %s", CHAT_MODEL_ID)
    except Exception as exc:
        log.error("Chat model init failed: %s", exc)
        _chat_model = None
    return _chat_model


# =============================================================================
#  INTEGRATED MACHINE LEARNING DATASET
#  Single array mixing time-series telemetry + technician shift-log text
# =============================================================================
UNIFIED_DATASET: list[dict] = [
    {
        "timestamp"        : "2025-01-15T06:00:00Z",
        "asset_id"         : "TURBINE-A1",
        "vibration_hz"     : 3.2,
        "structural_health": 91.4,
        "shift_log"        : (
            "Morning shift. Turbine A1 running nominal. Lubrication check completed. "
            "No anomalies detected. Bearings inspected — within tolerance."
        ),
    },
    {
        "timestamp"        : "2025-01-15T08:30:00Z",
        "asset_id"         : "COMPRESSOR-B2",
        "vibration_hz"     : 4.1,
        "structural_health": 84.7,
        "shift_log"        : (
            "Slight vibration uptick noted on B2 compressor. Operator adjusted alignment. "
            "Seal pressure holding. Recommend follow-up in next cycle."
        ),
    },
    {
        "timestamp"        : "2025-01-15T11:00:00Z",
        "asset_id"         : "PUMP-C3",
        "vibration_hz"     : 4.8,
        "structural_health": 78.2,
        "shift_log"        : (
            "WARNING: Pump C3 vibration above normal band. Cavitation suspected. "
            "Shut valve V-14 partially. Temperature rising 3°C above baseline. "
            "Maintenance ticket raised — awaiting spare impeller."
        ),
    },
    {
        "timestamp"        : "2025-01-15T13:45:00Z",
        "asset_id"         : "TURBINE-A1",
        "vibration_hz"     : 5.4,
        "structural_health": 69.1,
        "shift_log"        : (
            "ALERT: A1 vibration escalating. Blade inspection reveals micro-fractures on "
            "trailing edge. Structural integrity dropping. Vendor supply chain delay for "
            "composite blade kit — currently 14-day lead time. Production loss risk HIGH."
        ),
    },
    {
        "timestamp"        : "2025-01-15T16:00:00Z",
        "asset_id"         : "COMPRESSOR-B2",
        "vibration_hz"     : 6.3,
        "structural_health": 52.8,
        "shift_log"        : (
            "CRITICAL: B2 compressor exceeding vibration limits. Immediate shutdown "
            "initiated per Protocol 7. Foundation bolts showing fatigue signs. "
            "Emergency vendor contact required. Estimated downtime cost: $420,000/day. "
            "Prescriptive fix via expedited vendor supply estimated savings: $1,199,360."
        ),
    },
    {
        "timestamp"        : "2025-01-15T18:30:00Z",
        "asset_id"         : "PUMP-C3",
        "vibration_hz"     : 3.9,
        "structural_health": 81.5,
        "shift_log"        : (
            "Post-maintenance check. Impeller replaced on C3. Vibration normalising. "
            "Cavitation resolved. System returning to operational envelope. "
            "Continuous monitoring active."
        ),
    },
    {
        "timestamp"        : "2025-01-15T21:00:00Z",
        "asset_id"         : "TURBINE-A1",
        "vibration_hz"     : 6.8,
        "structural_health": 48.3,
        "shift_log"        : (
            "EMERGENCY: A1 vibration at 6.8 Hz — far beyond safe operating range. "
            "Blade fracture propagating. Asset taken offline. NDT team deployed. "
            "Total structural health at 48%. Supply chain escalation to Tier-1 vendor. "
            "Board-level incident opened. ROI model confirms $1.2M loss avoidable "
            "with proactive vendor intervention 48 hours earlier."
        ),
    },
]


# =============================================================================
#  AGENT STATE  (shared across threads)
# =============================================================================
agent_state = {
    "status"            : "INITIALISING",
    "current_asset"     : None,
    "iteration"         : 0,
    "critical_alerts"   : 0,
    "warning_alerts"    : 0,
    "total_scans"       : 0,
    "roi_savings"       : 0,
    "last_action"       : "",
    "react_trace"       : deque(maxlen=200),   # ring buffer for SSE stream
    "telemetry_history" : deque(maxlen=50),
    "lock"              : threading.Lock(),
}


# =============================================================================
#  PRESCRIPTIVE OPTIMISATION MODEL
# =============================================================================
def compute_roi_optimisation(dataset_entry: dict) -> dict:
    """
    Simulated prescriptive optimisation model.
    Returns the highest-ROI vendor supply fix recommendation.
    """
    vib   = dataset_entry["vibration_hz"]
    sh    = dataset_entry["structural_health"]
    asset = dataset_entry["asset_id"]
    target_savings = AGENT_INSTRUCTIONS["thresholds"]["roi_savings_target_usd"]

    # Weighting model: higher degradation → higher avoidable loss
    degradation_factor = max(0.0, (100 - sh) / 100) * (vib / 10.0)
    base_loss_per_day  = 420_000          # USD
    avoidable_days     = 2.856            # calibrated to hit $1,199,360
    roi                = round(base_loss_per_day * avoidable_days * degradation_factor * 2.39, 2)
    roi                = min(roi, target_savings)   # cap at target

    return {
        "asset"              : asset,
        "recommended_action" : "Expedite Tier-1 vendor composite blade / seal kit",
        "lead_time_reduction": "14 days → 2 days (emergency vendor protocol)",
        "avoidable_downtime" : f"{round(avoidable_days * degradation_factor, 2)} days",
        "roi_savings_usd"    : roi if roi > 0 else target_savings,
        "confidence"         : f"{min(99, round(degradation_factor * 200, 1))}%",
    }


# =============================================================================
#  WATSONX NLP TOOL  (ReAct TOOL-USE)
# =============================================================================
def tool_watsonx_analyze(shift_log: str, telemetry_summary: str) -> str:
    """
    ReAct tool: call Watsonx Granite to extract risk variables from shift-log text
    and correlate with numeric telemetry.
    Falls back to a deterministic heuristic when credentials are absent.
    """
    system_p = AGENT_INSTRUCTIONS["system_prompt"]
    prompt   = (
        f"{system_p}\n\n"
        f"TELEMETRY SNAPSHOT:\n{telemetry_summary}\n\n"
        f"TECHNICIAN SHIFT LOG:\n{shift_log}\n\n"
        f"Task: Identify all risk variables, severity level (LOW/MEDIUM/HIGH/CRITICAL), "
        f"root-cause hypothesis, and the single highest-ROI prescriptive action. "
        f"Respond ONLY in the structured format: "
        f"[OBSERVATION] ... [RISK ASSESSMENT] ... [ACTION] ... [ROI IMPACT] ..."
    )

    model = _get_watsonx_model()
    if model:
        try:
            result = model.generate_text(prompt=prompt)
            return result.strip() if isinstance(result, str) else str(result)
        except Exception as exc:
            log.error("Watsonx generate_text failed: %s", exc)

    # ── Simulation fallback ───────────────────────────────────────────────────
    keywords = shift_log.lower()
    if "emergency" in keywords or "critical" in keywords:
        severity = "CRITICAL"
    elif "alert" in keywords or "warning" in keywords:
        severity = "HIGH"
    elif "recommend" in keywords or "slight" in keywords:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return (
        f"[OBSERVATION] Telemetry and log analysis complete. {telemetry_summary}. "
        f"Log indicates {severity}-level operational concern.\n"
        f"[RISK ASSESSMENT] Severity: {severity}. Root cause: mechanical degradation "
        f"pattern consistent with bearing/blade fatigue.\n"
        f"[ACTION] {severity} protocol activated. "
        f"{'Immediate shutdown and vendor escalation required.' if severity == 'CRITICAL' else 'Schedule inspection within 24 hours.'}\n"
        f"[ROI IMPACT] Proactive intervention prevents unplanned downtime — "
        f"estimated savings align with $1,199,360 enterprise optimisation target.\n"
        f"[SIMULATION MODE — connect IBM Watsonx credentials for live NLP analysis]"
    )


# =============================================================================
#  ReAct REASONING LOOP
# =============================================================================
def _emit(message: str, step_type: str = "THINK"):
    """Push a reasoning trace entry into the ring buffer."""
    ts  = datetime.datetime.utcnow().strftime("%H:%M:%S")
    entry = {"ts": ts, "type": step_type, "msg": message}
    with agent_state["lock"]:
        agent_state["react_trace"].append(entry)


def react_loop_single(entry: dict) -> dict:
    """
    Execute one full ReAct cycle for a single dataset entry.
    Returns a summary dict with all findings.
    """
    thresholds = AGENT_INSTRUCTIONS["thresholds"]
    max_iter   = AGENT_INSTRUCTIONS["react_max_iterations"]
    asset      = entry["asset_id"]
    vib        = entry["vibration_hz"]
    sh         = entry["structural_health"]
    ts         = entry["timestamp"]
    log_text   = entry["shift_log"]

    summary = {
        "asset"          : asset,
        "timestamp"      : ts,
        "vibration_hz"   : vib,
        "structural_pct" : sh,
        "severity"       : "NOMINAL",
        "watsonx_output" : "",
        "roi"            : None,
        "iterations"     : 0,
    }

    _emit(f"▶ Scan started — Asset: {asset} | Vib: {vib} Hz | SH: {sh}%", "START")

    for iteration in range(1, max_iter + 1):
        summary["iterations"] = iteration
        _emit(f"[Iteration {iteration}] THINK: Evaluating telemetry for {asset}…", "THINK")

        # ── OBSERVATION ──────────────────────────────────────────────────────
        _emit(
            f"[Iteration {iteration}] OBSERVE: vib={vib} Hz | "
            f"struct_health={sh}% | log_len={len(log_text)} chars",
            "OBSERVE",
        )

        # ── THRESHOLD CHECK ───────────────────────────────────────────────────
        vib_breach = vib >= thresholds["vibration_critical_hz"]
        sh_breach  = sh  <= thresholds["structural_health_crit"]
        vib_warn   = (not vib_breach) and vib >= thresholds["vibration_warning_hz"]
        sh_warn    = (not sh_breach)  and sh  <= thresholds["structural_health_warn"]

        if vib_breach and sh_breach:
            severity = "EMERGENCY"
        elif vib_breach or sh_breach:
            severity = "CRITICAL"
        elif vib_warn or sh_warn:
            severity = "WARNING"
        else:
            severity = "NOMINAL"

        summary["severity"] = severity
        _emit(f"[Iteration {iteration}] DECIDE: Severity classified as → {severity}", "DECIDE")

        if severity in ("NOMINAL",) and iteration == 1:
            _emit(f"[Iteration {iteration}] ACTION: No threshold breach — continuing scan.", "ACTION")
            break

        # ── TOOL CALL: Watsonx NLP ────────────────────────────────────────────
        _emit(
            f"[Iteration {iteration}] ACTION: Invoking watsonx NLP tool on shift log…",
            "TOOL",
        )
        telemetry_summary = (
            f"Asset={asset}, Vibration={vib} Hz, Structural Health={sh}%, "
            f"Timestamp={ts}, Severity={severity}"
        )
        nlp_result = tool_watsonx_analyze(log_text, telemetry_summary)
        summary["watsonx_output"] = nlp_result
        _emit(f"[Iteration {iteration}] WATSONX RESPONSE:\n{nlp_result}", "WATSONX")

        # ── ROI PRESCRIPTIVE MODEL ────────────────────────────────────────────
        if severity in ("CRITICAL", "EMERGENCY"):
            _emit(
                f"[Iteration {iteration}] ACTION: Running prescriptive ROI optimisation model…",
                "ACTION",
            )
            roi_result = compute_roi_optimisation(entry)
            summary["roi"] = roi_result
            _emit(
                f"[Iteration {iteration}] ROI RESULT: "
                f"${roi_result['roi_savings_usd']:,.0f} savings identified — "
                f"{roi_result['recommended_action']}",
                "ROI",
            )

            # Update global savings counter
            with agent_state["lock"]:
                agent_state["roi_savings"] = roi_result["roi_savings_usd"]

        # ── UPDATE ALERT COUNTERS ─────────────────────────────────────────────
        with agent_state["lock"]:
            if severity == "EMERGENCY":
                agent_state["critical_alerts"] += 1
                agent_state["warning_alerts"]  += 1
                for proto in AGENT_INSTRUCTIONS["safety_protocols"]:
                    _emit(f"[SAFETY] {proto}", "SAFETY")
            elif severity == "CRITICAL":
                agent_state["critical_alerts"] += 1
            elif severity == "WARNING":
                agent_state["warning_alerts"] += 1

        _emit(
            f"[Iteration {iteration}] COMPLETE — severity={severity}, "
            f"iterations_used={iteration}/{max_iter}",
            "DONE",
        )
        break   # one pass sufficient once Watsonx tool called

    return summary


# =============================================================================
#  BACKGROUND AGENT THREAD
# =============================================================================
_agent_running = False
_agent_thread: threading.Thread | None = None
_dataset_index = 0


def _agent_worker():
    """Continuously cycle through UNIFIED_DATASET, running ReAct on each entry."""
    global _dataset_index
    log.info("Background agent worker started.")

    with agent_state["lock"]:
        agent_state["status"] = "RUNNING"

    while _agent_running:
        entry = UNIFIED_DATASET[_dataset_index % len(UNIFIED_DATASET)]

        with agent_state["lock"]:
            agent_state["current_asset"] = entry["asset_id"]
            agent_state["total_scans"]  += 1
            agent_state["iteration"]     = agent_state["total_scans"]

        result = react_loop_single(entry)

        # Store telemetry snapshot for frontend charts
        with agent_state["lock"]:
            agent_state["telemetry_history"].append({
                "ts"              : entry["timestamp"],
                "asset"           : entry["asset_id"],
                "vibration_hz"    : entry["vibration_hz"],
                "structural_pct"  : entry["structural_health"],
                "severity"        : result["severity"],
            })
            agent_state["last_action"] = (
                f"Processed {entry['asset_id']} — severity: {result['severity']}"
            )

        _dataset_index += 1
        time.sleep(AGENT_INSTRUCTIONS["scan_interval_seconds"])

    with agent_state["lock"]:
        agent_state["status"] = "STOPPED"
    log.info("Background agent worker stopped.")


def start_agent():
    global _agent_running, _agent_thread
    if _agent_running:
        return
    _agent_running = True
    _agent_thread  = threading.Thread(target=_agent_worker, daemon=True)
    _agent_thread.start()
    log.info("Agent thread launched.")


def stop_agent():
    global _agent_running
    _agent_running = False


# =============================================================================
#  FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ── DYNAMIC AUTOMATIC STATIC FILE ASSET ROUTING PATH FIXERS ───────────────────
@app.route('/script.js')
def serve_js():
    for location in ['static', '', 'templates', 'static/js']:
        file_path = os.path.join(app.root_path, location, 'script.js')
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/javascript')
    return "File not found", 404


@app.route('/style.css')
def serve_css():
    for location in ['static', '', 'templates', 'static/css']:
        file_path = os.path.join(app.root_path, location, 'style.css')
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='text/css')
    return "File not found", 404
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/api/status")
def api_status():
    """Return agent health and current state snapshot."""
    with agent_state["lock"]:
        snap = {
            "status"         : agent_state["status"],
            "current_asset"  : agent_state["current_asset"],
            "total_scans"    : agent_state["total_scans"],
            "critical_alerts": agent_state["critical_alerts"],
            "warning_alerts" : agent_state["warning_alerts"],
            "roi_savings"    : agent_state["roi_savings"],
            "last_action"    : agent_state["last_action"],
            "watsonx_live"   : bool(_get_watsonx_model()),
            "model_id"       : AGENT_INSTRUCTIONS["watsonx_model_id"],
            "agent_name"     : AGENT_INSTRUCTIONS["agent_name"],
        }
    return jsonify(snap)


@app.route("/api/telemetry")
def api_telemetry():
    """Return the telemetry history for chart rendering."""
    with agent_state["lock"]:
        history = list(agent_state["telemetry_history"])
    # Also include the static dataset for an initial rich chart
    static = [
        {
            "ts"             : e["timestamp"],
            "asset"          : e["asset_id"],
            "vibration_hz"   : e["vibration_hz"],
            "structural_pct" : e["structural_health"],
        }
        for e in UNIFIED_DATASET
    ]
    return jsonify({"history": history, "dataset": static})


@app.route("/api/dataset")
def api_dataset():
    """Return the full unified dataset."""
    return jsonify(UNIFIED_DATASET)


@app.route("/api/agent/start", methods=["POST"])
def api_agent_start():
    start_agent()
    return jsonify({"ok": True, "message": "Agent started."})


@app.route("/api/agent/stop", methods=["POST"])
def api_agent_stop():
    stop_agent()
    return jsonify({"ok": True, "message": "Agent stop signal sent."})


@app.route("/api/react/trace/stream")
def api_react_stream():
    """
    Server-Sent Events endpoint — streams ReAct trace in real time to the
    watsonx Agent Workspace terminal in the frontend.
    """
    def event_generator():
        sent_count = 0
        while True:
            with agent_state["lock"]:
                all_entries = list(agent_state["react_trace"])
            new_entries = all_entries[sent_count:]
            for entry in new_entries:
                payload = json.dumps(entry)
                yield f"data: {payload}\n\n"
                sent_count += 1
            time.sleep(0.4)

    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/thresholds")
def api_thresholds():
    """Expose current threshold config to the frontend."""
    return jsonify(AGENT_INSTRUCTIONS["thresholds"])


# =============================================================================
#  CHAT ENDPOINT  —  granite-3-3-8b-instruct
# =============================================================================
@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts { "message": "<user prompt>" } and returns
    { "reply": "<model response>", "model": "<model_id>", "simulated": bool }.

    Uses granite-3-3-8b-instruct via a dedicated lazy-init client.
    Falls back to contextual simulation when credentials are absent.
    """
    from flask import request as flask_request

    data     = flask_request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    log.info("Chat request: %s", user_msg[:120])

    model = _get_chat_model()
    if model:
        try:
            prompt = (
                "You are ChronosAI, an expert industrial reliability assistant powered by "
                "IBM Watsonx.ai Granite Guardian. You help engineers understand asset health, "
                "interpret telemetry, diagnose failures, and recommend maintenance actions.\n\n"
                f"Engineer: {user_msg}\n\n"
                "ChronosAI:"
            )
            raw       = model.generate_text(prompt=prompt)
            reply     = raw.strip() if isinstance(raw, str) else str(raw)
            simulated = False
        except Exception as exc:
            log.error("Chat model generate_text failed: %s", exc)
            reply     = _simulate_chat_reply(user_msg)
            simulated = True
    else:
        reply     = _simulate_chat_reply(user_msg)
        simulated = True

    return jsonify({
        "reply"    : reply,
        "model"    : CHAT_MODEL_ID,
        "simulated": simulated,
    })


def _simulate_chat_reply(user_msg: str) -> str:
    """
    Deterministic simulation fallback for the chat interface.
    Returns a contextually relevant response without calling Watsonx.
    """
    msg = user_msg.lower()

    if any(w in msg for w in ["vibration", "hz", "frequency"]):
        return (
            "Vibration levels are a primary leading indicator of mechanical wear. "
            "In this dataset, TURBINE-A1 reached 6.8 Hz — well above the 6.0 Hz critical threshold. "
            "Root causes typically include bearing degradation, blade imbalance, or resonance buildup. "
            "Recommended action: halt operations and conduct an NDT inspection immediately."
        )
    if any(w in msg for w in ["structural", "health", "integrity", "fracture"]):
        return (
            "Structural health below 55% indicates critical integrity risk. "
            "TURBINE-A1 dropped to 48.3% — triggering an EMERGENCY classification. "
            "This correlates with micro-fractures detected in the blade trailing edge. "
            "Prescriptive fix: expedite Tier-1 vendor composite blade kit (14-day → 2-day lead time)."
        )
    if any(w in msg for w in ["roi", "savings", "cost", "vendor", "downtime"]):
        return (
            "The prescriptive optimisation model identifies $1,199,360 in avoidable enterprise savings. "
            "This is computed as: $420,000/day loss × 2.856 avoidable days × degradation weighting. "
            "The highest-ROI action is activating the emergency vendor supply protocol to reduce "
            "lead time from 14 days to 2 days for the composite blade / seal kit."
        )
    if any(w in msg for w in ["asset", "compressor", "pump", "turbine"]):
        return (
            "ChronosAI is monitoring three assets: TURBINE-A1, COMPRESSOR-B2, and PUMP-C3. "
            "Current status — TURBINE-A1: EMERGENCY (48.3% SH, 6.8 Hz). "
            "COMPRESSOR-B2: CRITICAL (52.8% SH, 6.3 Hz). PUMP-C3: post-maintenance NOMINAL (81.5% SH). "
            "Immediate focus should be on TURBINE-A1 and COMPRESSOR-B2."
        )
    if any(w in msg for w in ["react", "reasoning", "agent", "loop", "watsonx"]):
        return (
            "The ReAct (Reasoning + Action) loop runs every 8 seconds. Each cycle: "
            "THINK → OBSERVE telemetry → DECIDE severity → call Watsonx NLP TOOL on shift logs → "
            "ACTION (alert / escalate) → run prescriptive ROI model if critical. "
            "Granite-Guardian-3-8b powers this chat; Granite-3.3-8b-Instruct drives the ReAct NLP tool."
        )
    if any(w in msg for w in ["hello", "hi", "hey", "help", "what can you"]):
        return (
            "Hello, I'm ChronosAI — your autonomous asset reliability assistant. "
            "I can help you interpret vibration telemetry, structural health indices, "
            "shift-log risk variables, ROI optimisation recommendations, and ReAct agent traces. "
            "Ask me anything about the monitored assets or the AI reasoning pipeline."
        )
    return (
        f"I've analysed your query: \"{user_msg[:80]}\". "
        "Based on current telemetry context — TURBINE-A1 is in EMERGENCY state with 48.3% structural "
        "health and 6.8 Hz vibration. I recommend reviewing the latest ReAct trace in the Agent "
        "Workspace terminal for a full reasoning breakdown. "
        "[Simulation mode — connect IBM Watsonx credentials for live Granite Guardian responses.]"
    )


# =============================================================================
#  INITIALIZATION & EXECUTION
# =============================================================================

# Kick off background ReAct loop automatically so it works under both local servers and Gunicorn/Render deployments
start_agent()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting ChronosAI on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
