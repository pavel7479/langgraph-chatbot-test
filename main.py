

import os
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define the tool
@tool
def get_current_time() -> dict:
    """Return the current UTC time in ISO‑8601 format. Example → {"utc": "2025‑05‑21T06:42:00Z"}"""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat()}

# Input state format
class GraphState(TypedDict):
    messages: Annotated[List[dict], "Messages"]

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=api_key,
)

# Agent with tool
agent_executor = create_react_agent(llm, tools=[get_current_time])

# Stateless graph
builder = StateGraph(GraphState)
builder.add_node("agent", agent_executor)
builder.set_entry_point("agent")
builder.set_finish_point("agent")

graph = builder.compile()

# For local testing
# if __name__ == "__main__":
#     res = graph.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
#     print(res["messages"][-1]["content"])


