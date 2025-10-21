# Multi-Agent CRM Automation System

AI-powered system using LangGraph, Gemini Flash 1.5, and Hugging Face embeddings to automate HubSpot CRM tasks (create/update contacts/deals) and send email notifications.

## Features
- Multi-agent architecture: Supervisor, HubSpot Agent, Email Agent.
- Free-tier compatible: No costs for testing.
- Error handling and logging.

## Setup

### Option 1: Google Colab (Recommended for Testing)
1. Open [Multi_Agent_CRM_Automation.ipynb](Multi_Agent_CRM_Automation.ipynb) in [Colab](https://colab.research.google.com).
2. Upload `config.json` (use `config_template.json` with your keys and remane to `config.json`) to Colab's Files tab or use the one i ahev provided to you in Email.
3. Run cells in order: Install dependencies, main code, then input queries.

### Option 2: Local Python (.py)
1. Install dependencies: `pip install -r requirements.txt`.
2. Use `config_template` or and add your API keys, copy  `config.json` send in gmail.
3. Run: `python main.py`.

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
<image-card alt="path not found check images folder" src="images/output.png" ></image-card>

Questions?  Email at: asaduyousaf@gmail.com