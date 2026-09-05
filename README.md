# Multi-Tool AI Agent for Bangladesh

A LangChain agent that answers questions about Bangladeshi institutions,
hospitals, and restaurants by querying purpose-built SQLite databases, and
falls back to web search for general-knowledge questions.

## Project structure

```
.
├── build_databases.py   # Stage 1: converts the 3 HF datasets into SQLite DBs
├── db_tools.py           # Stage 2: institutions/hospitals/restaurants tools
├── web_tool.py            # Stage 3: web search tool for general knowledge
├── agent.py               # Stage 4: routing agent (tool-calling agent)
├── main.py                # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your API keys as environment variables:
   ```
   export OPENAI_API_KEY="your-key-here"
   export TAVILY_API_KEY="your-key-here"
   ```
   (Get a free Tavily key at https://tavily.com)

3. Build the databases (only needs to be run once):
   ```
   python build_databases.py
   ```
   This downloads the three datasets from HuggingFace and writes
   `institutions.db`, `hospitals.db`, and `restaurants.db` to the current
   directory.

4. Run the agent:
   ```
   python main.py
   ```

## Example questions

- "How many hospitals are in Dhaka district?" → routed to `hospitals_db_tool`
- "List government colleges in Barisal division" → routed to `institutions_db_tool`
- "What's the highest-rated restaurant in Pirojpur?" → routed to `restaurants_db_tool`
- "What is the role of DGHS in Bangladesh?" → routed to `web_search_tool`

## Data sources

- Institutional Information of Bangladesh — https://huggingface.co/datasets/Mahadih534/Institutional-Information-of-Bangladesh
- All Bangladeshi Hospitals — https://huggingface.co/datasets/Mahadih534/all-bangladeshi-hospitals
- Bangladeshi Restaurant Data — https://huggingface.co/datasets/Mahadih534/Bangladeshi-Restaurant-Data

## Notes on the data

The hospitals dataset does not include bed counts or doctor counts — it's
a facility directory (name, type, agency, location, public/private status).
The restaurants dataset does not include a cuisine field — it has rating,
review count, an approximate price-tier ("affluence") score, and location.
The tools and schemas here are built around what's actually in the data.
