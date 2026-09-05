"""
Stage 2 — LangChain tools for querying the three SQLite databases built in
Stage 1 (institutions.db, hospitals.db, restaurants.db).

Each tool follows the same "text-to-SQL" pattern:
  1. An LLM turns the natural-language question into a SQL query, using the
     table's real schema as context (so it can't invent columns that don't
     exist).
  2. The SQL query runs read-only against the SQLite file.
  3. A second LLM call turns the raw result rows into a natural-language
     answer, so the calling agent gets prose back, not a table of numbers.

WHY `SQLDatabase` (from langchain_community) instead of hand-rolling this:
it already knows how to introspect a SQLite file's schema and safely execute
queries against it, so we don't have to write our own schema-string builder
or connection management for each of the three databases.
"""

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# A shared LLM for SQL generation + answer formatting. This doesn't need to
# be your most powerful/expensive model — SQL generation and summarizing a
# handful of rows are both fairly simple tasks.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ANSWER_PROMPT = PromptTemplate.from_template(
    """Given the following user question, the SQL query that was run, and
the SQL result, answer the user's question in clear, natural language.
Do not mention SQL, databases, or table names in your answer — respond as
if you already knew the facts.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:"""
)


def _build_db_chain(db_path: str):
    """
    Builds a reusable question -> SQL -> execute -> natural-language-answer
    chain for a single SQLite database file.
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    # RunnablePassthrough.assign lets us build up a dict of
    # {question, query, result} step by step, which the ANSWER_PROMPT
    # template above expects.
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=lambda x: execute_query.invoke({"query": x["query"]})
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# One chain per database. Paths assume the .db files created by Stage 1's
# build_databases.py live in the same working directory as this file.
# If you organize your repo with a /data folder, update these paths.
# ---------------------------------------------------------------------------
_institutions_chain = _build_db_chain("institutions.db")
_hospitals_chain = _build_db_chain("hospitals.db")
_restaurants_chain = _build_db_chain("restaurants.db")


# ---------------------------------------------------------------------------
# The actual LangChain tools. The @tool decorator turns each function into
# a BaseTool the agent can call — the docstring becomes the tool's
# description, which is what the routing LLM reads to decide when to use it.
# This is why the docstrings are written as instructions to the router, not
# just documentation for a human reader.
# ---------------------------------------------------------------------------

@tool
def institutions_db_tool(question: str) -> str:
    """Use this tool for questions about Bangladeshi educational or
    government institutions — schools, colleges, madrasahs, their type,
    location (division/district), management type (government vs
    non-government), or affiliation/MPO status. Example questions:
    'how many colleges are in Barguna district', 'list government schools
    in Dhaka division'."""
    return _institutions_chain.invoke({"question": question})


@tool
def hospitals_db_tool(question: str) -> str:
    """Use this tool for questions about Bangladeshi hospitals and health
    facilities — facility type (e.g. Upazila Health Complex, Medical
    College Hospital), which agency runs it, location, or whether it's
    private or public. Example questions: 'how many hospitals are in
    Dhaka district', 'list private hospitals in Chattogram'."""
    return _hospitals_chain.invoke({"question": question})


@tool
def restaurants_db_tool(question: str) -> str:
    """Use this tool for questions about Bangladeshi restaurants — their
    rating, number of reviews, approximate price tier (affluence score),
    or location/address. Example questions: 'what's the highest-rated
    restaurant in Pirojpur', 'restaurants with more than 100 reviews'."""
    return _restaurants_chain.invoke({"question": question})
