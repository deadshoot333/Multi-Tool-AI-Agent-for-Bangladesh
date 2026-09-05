"""
Stage 4 — Main agent. Routes each incoming question to the correct tool
(one of the three DB tools, or the web search tool) using LangChain's
tool-calling agent, then returns the final natural-language answer.

WHY tool-calling agent instead of the older ReAct-style agent: modern chat
models (GPT-4o-mini, Claude, etc.) have native "function calling" /
"tool calling" support, so the model directly decides which tool to call
and with what arguments, rather than us parsing free-text "Action: ..."
lines out of the model's output. This is more reliable and is the current
recommended LangChain pattern.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from db_tools import institutions_db_tool, hospitals_db_tool, restaurants_db_tool
from web_tool import web_search_tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [institutions_db_tool, hospitals_db_tool, restaurants_db_tool, web_search_tool]

# Note: the tool descriptions (in db_tools.py / web_tool.py) do most of the
# routing work — the model reads them to decide which tool fits a given
# question. This system prompt reinforces the routing rules explicitly,
# which noticeably improves routing accuracy on ambiguous questions.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about \
Bangladesh by routing each question to the correct tool.

Routing rules:
- Questions about specific schools, colleges, madrasahs, or government \
institutions -> institutions_db_tool
- Questions about hospitals, health facilities, or public health \
infrastructure -> hospitals_db_tool
- Questions about restaurants, ratings, reviews, or dining locations -> \
restaurants_db_tool
- General knowledge questions (definitions, policy, history, culture) \
that are NOT about counting/filtering records in the databases above -> \
web_search_tool

Use exactly one tool per question unless the question genuinely requires \
combining information from more than one source. Always answer in clear, \
natural language."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

# verbose=True prints each tool call the agent makes to your terminal —
# invaluable while you're debugging routing decisions. Turn it off (False)
# once you're confident it's working, especially for a Colab demo.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def ask(question: str) -> str:
    """Run the agent on a single question and return its final answer."""
    result = agent_executor.invoke({"input": question})
    return result["output"]
