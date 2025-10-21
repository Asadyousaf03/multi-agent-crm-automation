# Multi-Agent CRM Automation System

AI-powered system using LangGraph, Gemini Flash 2.5, and Hugging Face embeddings to automate HubSpot CRM tasks (create/update contacts/deals) and send email notifications.

## Features
- Multi-agent architecture: Supervisor, HubSpot Agent, Email Agent.
- Free-tier compatible: No costs for testing.
- Error handling and logging.

## Setup

### Option 1: Google Colab (Recommended for testing)
1. Open [Multi_Agent_CRM_Automation.ipynb](Multi_Agent_CRM_Automation.ipynb) in [Colab](https://colab.research.google.com).
2. Upload `config.json` to `/content/` with valid API keys.
3. Run cells to install dependencies and execute the workflow.

### Option 2: Web UI with Flask
1. Install dependencies: `pip install -r requirements.txt`.
2. Copy `config.json.template` to `config.json` and add API keys (use Gmail App Password for `SMTP_PASS`).
3. Run the Flask server: `python multi_agent_crm_automation.py`.
4. Open `static/index.html` in a browser or serve it via a web server (e.g., `python -m http.server 8000` in the repo root).
5. Enter queries in the UI (e.g., "Create a new contact for John Doe with email john.doe@example.com and send a confirmation email").

## Configuration
- Get keys: Google AI Studio (Gemini), HubSpot Developers, Gmail App Password (SMTP).
- You can add your email or go as it is in the code while testing. 

## Testing Prompts
- Create contact: "Create a new contact for John Doe with email john.doe@example.com and send confirmation."
- Update contact: "Update contact ID 12345 with firstname Jane and send email."
- Create deal: "Create deal with dealname 'Big Sale' amount 5000 and send email."

## Logs and Errors
- Check `agent.log` for details.
- Common issues: Invalid API keys, rate limits (6s delay included).

## Result
I have saved the screenshot in images folder.

Questions?  Email at: asaduyousaf@gmail.com
