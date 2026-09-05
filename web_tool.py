"""
Stage 3 — Web search tool for general-knowledge queries that the DB tools
can't answer: definitions, government policy, healthcare regulation,
cultural or historical context.

Uses Tavily because it has a clean, purpose-built LangChain integration and
a free tier that's generous enough for a student project. Get a free API
key at https://tavily.com and set it as the TAVILY_API_KEY environment
variable.
"""

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(
    max_results=3,
    name="web_search_tool",
    description=(
        "Use this tool ONLY for general knowledge questions that are NOT "
        "answerable by counting, filtering, or listing records in the "
        "institutions, hospitals, or restaurants databases — for example, "
        "definitions, government policy questions, healthcare regulations, "
        "or cultural/historical context about Bangladesh. Do not use this "
        "tool for questions about specific institutions, hospitals, or "
        "restaurants in the databases."
    ),
)
