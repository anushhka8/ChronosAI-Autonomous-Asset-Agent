# 🏭 ChronosAI — Autonomous Asset Reliability Agent

### IBM watsonx.ai × Granite Models × Flask

A fully-featured, responsive industrial asset reliability web application powered by IBM Granite foundation models on watsonx.ai. Features an autonomous ReAct reasoning loop chat interface, live telemetry dashboards, unified dataset log tracking, and a prescriptive maintenance return-on-investment optimization model.

---

## 🌟 Features

| Feature | Description |
| :--- | :--- |
| **AI Reliability Chat** | Conversational engineering diagnosis coach powered by Granite-3.3-8b-instruct. |
| **Telemetry Dashboard** | Real-time monitoring line graphs tracking Vibration (Hz) and Structural Health Indices (%). |
| **ReAct Workspace Logs** | Direct UI visibility into the agent's autonomous "Think-Action-Observe-Decide" loop. |
| **Prescriptive Optimization** | Automated financial models projecting downtime losses ($420k/day) and supply fix ROI. |
| **Unified Dataset Stream** | Consolidated operational log streams combining hardware thresholds and technician notes. |
| **Responsive UI** | Clean, dark-mode cybernetic dashboard theme optimized for mission-control displays. |

---

## 📁 Project Structure

```text
chronos-ai/
├── app.py              # Flask server, ReAct engine configuration, & watsonx orchestration
├── requirements.txt    # Framework and cloud connection dependencies
├── .env.example        # Environment variable layout credential template
└── templates/
    └── index.html      # Responsive frontend interactive control room dashboard


🚀 Quick Start
Step 1 — Clone & Enter Directory
Bash
git clone <your-repository-url>
cd chronos-ai


Step 2 — Create Virtual Environment
On Windows (PowerShell):
PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

On macOS / Linux:
Bash
python3 -m venv venv
source venv/bin/activate


Step 3 — Install Dependencies
Bash
pip install -r requirements.txt


Step 4 — Configure Environment Variables
Copy the example file and populate it with your private cloud keys:
Windows: copy .env.example .env
macOS/Linux: cp .env.example .env

Edit your local .env file structure:
Code snippet
IBM_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=[https://api.au-syd.dataplatform.cloud.ibm.com](https://api.au-syd.dataplatform.cloud.ibm.com)
CHAT_MODEL_ID=ibm/granite-3-3-8b-instruct

Step 5 — Run the Application
Bash
python app.py
Open your web browser and navigate to: http://127.0.0.1:5000

🔑 Getting IBM Cloud Credentials
IBM Cloud API Key: Go to IBM Cloud → Manage → IAM → API Keys. Click Create an API Key.

Watsonx Project ID: Open watsonx.ai → Manage → Projects. Select your sandbox project workspace and copy the unique Project ID string from the settings panel.

### ✅ You are officially finished!
Once Step 3 and Step 4 are complete, your GitHub link is ready to be pasted into **Slide 12** of your PPT and sent directly via the teacher's final evaluation Google Form link! You've done an incredible job putting this repo together.
