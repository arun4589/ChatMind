from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_ollama import ChatOllama
from langchain_experimental.tools import PythonREPLTool

from langchain_community.utilities import SerpAPIWrapper
from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
import datetime
import os
import tempfile
import streamlit as st

llm = ChatOllama(model="mistral")
#-------1 web search ------
search = SerpAPIWrapper(serpapi_api_key='9caac077d709388f110d0b41985b203e57cc165c236194d63285955e0c4f60e3')
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Use this tool to search the web for current events or factual questions."
)



# ---- 3. Tool: Python REPL ----
python_tool = Tool(
    name="Python Interpreter",
    func=PythonREPLTool().run,
    description="Executes Python code. Use for coding or data-related questions."
)


# ---- 6. Agent Initialization ----
tools = [ search_tool,python_tool]

tool_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True
)

# ---- 7. Agent Interface Function ----
def answer_with_tools(user_query):
    return tool_agent.invoke({"input": user_query})

