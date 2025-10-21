from flask import Flask, request, jsonify
import os
import json
import logging
from typing import List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from hubspot import HubSpot
from hubspot.crm.contacts import SimplePublicObjectInputForCreate, PublicObjectSearchRequest, SimplePublicObjectInput
from hubspot.crm.deals import SimplePublicObjectInput as DealSimplePublicObjectInput
from hubspot.crm.contacts.exceptions import ApiException as ContactsApiException
from hubspot.crm.deals.exceptions import ApiException as DealsApiException
import smtplib
from email.mime.text import MIMEText
from transformers import AutoTokenizer, AutoModel
from torch import cosine_similarity
import torch
from pydantic import BaseModel, Field
import time

# Set up logging
logging.basicConfig(level=logging.INFO, filename='agent.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configurations from JSON file
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("config.json not found. Please create it with API keys.")
    raise

os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
os.environ["HUBSPOT_API_KEY"] = config["HUBSPOT_API_KEY"]
os.environ["SMTP_SERVER"] = config["SMTP_SERVER"]
os.environ["SMTP_PORT"] = config["SMTP_PORT"]
os.environ["SMTP_USER"] = config["SMTP_USER"]
os.environ["SMTP_PASS"] = config["SMTP_PASS"]
os.environ["FROM_EMAIL"] = config["FROM_EMAIL"]

# Initialize Gemini Flash 2.5 LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Initialize HubSpot client
api_client = HubSpot(access_token=os.environ["HUBSPOT_API_KEY"])

# Initialize Hugging Face embedder
embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)
embedder = AutoModel.from_pretrained(embedder_model_name)

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedder(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Classifier with weighted scoring
agent_descriptions = {
    "hubspot_agent": [
        ("create a contact", 1.0),
        ("update a contact", 1.0),
        ("create a deal", 1.0),
        ("update deal", 1.0),
        ("manage CRM operations", 0.8),
        ("add new contact", 1.0),
        ("edit contact", 1.0)
    ],
    "email_agent": [
        ("send an email", 1.0),
        ("email notification", 1.0),
        ("confirm action via email", 1.0)
    ]
}
embedded_descriptions = {
    agent: [(get_embedding(desc), weight) for desc, weight in descs]
    for agent, descs in agent_descriptions.items()
}

def classify_query(query: str) -> str:
    query_emb = get_embedding(query)
    max_score = -1
    best_agent = "supervisor"
    for agent, embeddings in embedded_descriptions.items():
        for emb, weight in embeddings:
            sim = cosine_similarity(query_emb, emb).item() * weight
            if sim > max_score:
                max_score = sim
                best_agent = agent
    logger.info(f"Query: {query}, Classified as: {best_agent}, Score: {max_score}")
    crm_keywords = ["contact", "deal", "create", "update", "crm"]
    if any(keyword in query.lower() for keyword in crm_keywords):
        best_agent = "hubspot_agent"
        logger.info(f"Overriding to hubspot_agent due to CRM keywords in query")
    return best_agent

# HubSpot Tools
@tool
def create_contact(properties: dict):
    """Create a new contact in HubSpot with the given properties."""
    logger.info(f"Creating contact with properties: {properties}")
    try:
        if 'email' in properties:
            email = properties['email']
            filter_group = {
                "filters": [{"propertyName": "email", "operator": "EQ", "value": email}]
            }
            search_request = PublicObjectSearchRequest(filter_groups=[filter_group])
            search_response = api_client.crm.contacts.search_api.do_search(public_object_search_request=search_request)
            if search_response.total > 0:
                existing_id = search_response.results[0].id
                logger.info(f"Contact already exists with ID: {existing_id}")
                return {
                    "success": False,
                    "error": f"Contact already exists with ID: {existing_id}",
                    "id": existing_id
                }

        input_obj = SimplePublicObjectInputForCreate(properties=properties)
        response = api_client.crm.contacts.basic_api.create(simple_public_object_input_for_create=input_obj)
        result = {"success": True, "id": response.id, "properties": response.properties}
        logger.info(f"Contact created successfully: {result}")
        return result
    except ContactsApiException as e:
        error_msg = f"Failed to create contact: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

@tool
def update_contact(contact_id: str, properties: dict):
    """Update an existing contact in HubSpot with the given contact_id and properties."""
    logger.info(f"Updating contact {contact_id} with properties: {properties}")
    try:
        input_obj = SimplePublicObjectInput(properties=properties)
        response = api_client.crm.contacts.basic_api.update(
            contact_id=contact_id,
            simple_public_object_input=input_obj
        )
        result = {"success": True, "id": response.id, "properties": response.properties}
        logger.info(f"Contact updated successfully: {result}")
        return result
    except ContactsApiException as e:
        error_msg = f"Failed to update contact: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

@tool
def create_deal(properties: dict):
    """Create a new deal in HubSpot with the given properties."""
    logger.info(f"Creating deal with properties: {properties}")
    try:
        input_obj = DealSimplePublicObjectInput(properties=properties)
        response = api_client.crm.deals.basic_api.create(simple_public_object_input=input_obj)
        result = {"success": True, "id": response.id, "properties": response.properties}
        logger.info(f"Deal created successfully: {result}")
        return result
    except DealsApiException as e:
        error_msg = f"Failed to create deal: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

@tool
def update_deal(deal_id: str, properties: dict):
    """Update an existing deal in HubSpot with the given deal_id and properties."""
    logger.info(f"Updating deal {deal_id} with properties: {properties}")
    try:
        input_obj = DealSimplePublicObjectInput(properties=properties)
        response = api_client.crm.deals.basic_api.update(
            deal_id=deal_id,
            simple_public_object_input=input_obj
        )
        result = {"success": True, "id": response.id, "properties": response.properties}
        logger.info(f"Deal updated successfully: {result}")
        return result
    except DealsApiException as e:
        error_msg = f"Failed to update deal: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

hubspot_tools = [create_contact, update_contact, create_deal, update_deal]

# Email Tool
@tool
def send_email(to_email: str, subject: str, body: str):
    """Send an email notification."""
    logger.info(f"Sending email to {to_email} with subject: {subject}")
    logger.info(f"SMTP Config: server={os.environ['SMTP_SERVER']}, port={os.environ['SMTP_PORT']}, user={os.environ['SMTP_USER']}")
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = os.environ["FROM_EMAIL"]
    msg['To'] = to_email
    try:
        server = smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"]))
        server.starttls()
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.sendmail(os.environ["FROM_EMAIL"], to_email, msg.as_string())
        server.quit()
        result = {"success": True, "message": f"Email sent to {to_email}"}
        logger.info(f"Email sent successfully: {result}")
        return result
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

email_tools = [send_email]

# Routing Schemas for Supervisor
class TransferToHubspot(BaseModel):
    """Delegate task to HubSpot Agent."""
    task_description: str = Field(description="Clear task for HubSpot Agent, e.g., 'Create contact with properties: {'firstname': 'Asadullah', 'email': 'asadullahyousaf786@gmail.com'}'")

class TransferToEmail(BaseModel):
    """Delegate task to Email Agent."""
    task_description: str = Field(description="Clear task for Email Agent, e.g., 'Send confirmation email to asadullahyousaf786@gmail.com with subject \"Contact Created\" and body containing contact ID'")

# Agent Node Factory
from langchain_core.prompts import ChatPromptTemplate

def create_agent_node(llm, tools, system_prompt):
    def agent_node(state: MessagesState):
        try:
            task = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    task = msg.tool_calls[0]["args"].get("task_description", "")
                    break

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt + f"\nCurrent task: {task}"),
                *state["messages"],
            ])
            bound_llm = llm.bind_tools(tools)
            response = bound_llm.invoke(prompt.format_messages())
            logger.info(f"Agent ({tools[0].name if tools else 'supervisor'}): {response.content}")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Agent node error: {str(e)}")
            return {"messages": [AIMessage(content=f"Error: {str(e)}")]}
    return agent_node

# HubSpot Agent
hubspot_system_prompt = (
    "You are the HubSpot Agent. Focus exclusively on CRM tasks (create/update contacts or deals) using the provided task description. "
    "Call the appropriate tool (e.g., create_contact, update_contact) and return a summary of the result (e.g., 'Contact created with ID: xxx'). "
    "Do not attempt email-related tasks, as these are handled by the Email Agent. "
    "Handle errors and include them in the summary without calling additional tools."
)
hubspot_agent = create_agent_node(llm, hubspot_tools, hubspot_system_prompt)
hubspot_tool_node = ToolNode(tools=hubspot_tools)

# Email Agent
email_system_prompt = (
    "You are the Email Agent. Use the current task to generate email content and call send_email. "
    "After receiving tool results, output a summary (e.g., 'Email sent successfully'). Do not call more tools if task is complete. "
    "Handle errors and include them in the summary."
)
email_agent = create_agent_node(llm, email_tools, email_system_prompt)
email_tool_node = ToolNode(tools=email_tools)

# Supervisor (Orchestrator)
supervisor_system_prompt = (
    "You are the Global Orchestrator Agent. Analyze the query and history to delegate tasks:\n"
    "- Call TransferToHubspot for CRM operations (create/update contacts/deals).\n"
    "- Call TransferToEmail for email notifications, only after a successful CRM operation.\n"
    "Steps:\n"
    "1. Review history for completed tasks (look for ToolMessages with 'success': true).\n"
    "2. If CRM needed and not done, call TransferToHubspot with task_description.\n"
    "3. If CRM succeeded (e.g., contact ID in history), call TransferToEmail with task including ID and details.\n"
    "4. If both CRM and email complete, output 'Final summary: [brief overall summary]' without calling tools.\n"
    "5. Handle errors by including them in next task or final summary. Do not repeat completed tasks.\n"
    "Provide reasoning in your response, then call the tool if needed."
)
supervisor_bound_llm = llm.bind_tools([TransferToHubspot, TransferToEmail])

def supervisor_agent(state: MessagesState):
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=supervisor_system_prompt),
            *state["messages"],
        ])
        response = supervisor_bound_llm.invoke(prompt.format_messages())
        logger.info(f"Supervisor: {response.content}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Supervisor error: {str(e)}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

# Graph Definition
graph = StateGraph(state_schema=MessagesState)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("hubspot_agent", hubspot_agent)
graph.add_node("hubspot_tools", hubspot_tool_node)
graph.add_node("email_agent", email_agent)
graph.add_node("email_tools", email_tool_node)

# Supervisor Router
def supervisor_router(state: MessagesState):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and '"success": false' in msg.content:
                return END
            if isinstance(msg, AIMessage) and "Final summary" in msg.content:
                return END
        return "supervisor"

    tool_call_name = last_msg.tool_calls[0]["name"]
    if tool_call_name == "TransferToHubspot":
        return "hubspot_agent"
    elif tool_call_name == "TransferToEmail":
        return "email_agent"
    return END

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", supervisor_router, {"hubspot_agent": "hubspot_agent", "email_agent": "email_agent", END: END, "supervisor": "supervisor"})
graph.add_conditional_edges("hubspot_agent", lambda s: agent_router(s, "hubspot"), {"hubspot_tools": "hubspot_tools", "supervisor": "supervisor"})
graph.add_edge("hubspot_tools", "hubspot_agent")
graph.add_conditional_edges("email_agent", lambda s: agent_router(s, "email"), {"email_tools": "email_tools", "supervisor": "supervisor"})
graph.add_edge("email_tools", "email_agent")

# Compile
app = graph.compile()

# Flask App
flask_app = Flask(__name__)

@flask_app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        initial_agent = classify_query(user_query)
        logger.info(f"Embedding-based initial classification: {initial_agent}")

        output = []
        for event in app.stream({"messages": [HumanMessage(content=user_query)]}, config={"recursion_limit": 20}):
            for key, value in event.items():
                for msg in value["messages"]:
                    entry = {"agent": key}
                    if isinstance(msg, AIMessage):
                        entry["message"] = msg.content
                        if msg.tool_calls:
                            entry["toolCalls"] = msg.tool_calls
                    elif isinstance(msg, ToolMessage):
                        entry["toolResult"] = msg.content
                    elif isinstance(msg, HumanMessage):
                        entry["message"] = msg.content
                    output.append(entry)
                time.sleep(0.1)  # Reduced rate limiting for API

        return jsonify({"output": output})
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    flask_app.run(debug=True, host='0.0.0.0', port=5000)