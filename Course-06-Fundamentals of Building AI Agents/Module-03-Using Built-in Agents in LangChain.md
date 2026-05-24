# Course 06 — Module 03: Using Built-in Agents in LangChain

> A complete reference covering the Pandas DataFrame agent for data visualization and AI-powered SQL agents for natural language database access.

---

## Table of Contents

1. [Natural Language Data Analytics — Overview](#natural-language-data-analytics--overview)
2. [The LangChain Pandas DataFrame Agent](#the-langchain-pandas-dataframe-agent)
   - [What Makes It Different](#what-makes-it-different)
   - [Setup and Implementation](#setup-and-implementation)
   - [Querying and Visualizing Data](#querying-and-visualizing-data)
   - [Best Practices](#best-practices)
3. [AI-Powered SQL Agents](#ai-powered-sql-agents)
   - [Benefits and Capabilities](#benefits-and-capabilities)
   - [Limitations](#limitations)
   - [SQL Agent Query Flow](#sql-agent-query-flow)
   - [Implementation with LangChain](#implementation-with-langchain)
4. [Natural Language Interfaces (NLI) for Data Systems](#natural-language-interfaces-nli-for-data-systems)
   - [How NLIs Work](#how-nlis-work)
   - [Types of NLIs](#types-of-nlis)
   - [Approaches to Building NLIs](#approaches-to-building-nlis)
   - [Challenges](#challenges)
5. [Summary](#summary)

---

## Natural Language Data Analytics — Overview

Historically, data analysis required SQL, Python, R, or specialized tools like Tableau — locking insights behind a wall of technical expertise. **LLMs are democratizing analytics** by enabling natural language queries that produce both results and visualizations.

**How AI-powered analytics works:**

```
User natural language query
        |
AI-Driven Query Formulation
(identify entities, map to schema, determine intent, generate code)
        |
Database / DataFrame Data Extraction
        |
Data Analysis Process
(clean, preprocess, aggregate, identify patterns)
        |
Insight Synthesis
(interpret results, identify key findings)
        |
Presentation
(visualizations + natural language summaries)
```

**LLM capabilities that enable this:**

| Capability | Description |
|---|---|
| **Natural language understanding** | Convert questions into structured queries |
| **Code generation** | Produce Python/SQL from plain English |
| **Data reasoning** | Choose appropriate analysis for data type |
| **Chart selection** | Pick the right visualization format |
| **Insight generation** | Summarize and explain results in plain language |

---

## The LangChain Pandas DataFrame Agent

### What Makes It Different

The `create_pandas_dataframe_agent` differs from general agents in three key ways:

| Feature | Standard Agent | Pandas DataFrame Agent |
|---|---|---|
| **Configuration** | Manual setup | Pre-configured functions and prompts |
| **Data** | General access | Operates on your specific DataFrame |
| **Output** | Text responses | Values, summaries, **or visualizations** |

> **Important:** This agent is ideal for **exploration and rapid prototyping** — not recommended for production environments unless comprehensive safeguards are in place. Available in the `langchain-experimental` package.

---

### Setup and Implementation

```python
import pandas as pd
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Model
from langchain_ibm import WatsonxLLM
from langchain_experimental.agents import create_pandas_dataframe_agent

# Step 1: Load your DataFrame
df = pd.read_csv("student_alcohol_consumption.csv")
print(df.head())  # Preview first 5 rows

# Step 2: Set up IBM WatsonX credentials
credentials = {
    "url": "YOUR_WATSONX_URL",
    "apikey": "YOUR_API_KEY"
}

model_id = "meta-llama/llama-3-70b-instruct"
params = {
    GenParams.MAX_NEW_TOKENS: 1000,
    # other generation parameters
}

# Step 3: Initialize the watsonx model
watsonx_model = Model(
    model_id=model_id,
    credentials=credentials,
    params=params,
    project_id="YOUR_PROJECT_ID",
    space_id="YOUR_SPACE_ID"
)

# Step 4: Wrap in WatsonxLLM for LangChain integration
llm = WatsonxLLM(watsonx_model=watsonx_model)

# Step 5: Create the Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,                    # see details while code runs
    return_intermediate_steps=True   # view generated code (great for debugging)
)
```

> Always check the latest LangChain documentation — specific syntax can evolve with product updates.

---

### Querying and Visualizing Data

**Simple count query:**
```python
result = agent.invoke("How many rows are in this file?")
# Agent returns: "There are 395 rows."
# Generated code (viewable via intermediate_steps): len(df)
```

**Filtered count query:**
```python
result = agent.invoke("How many students are 18 years old?")
# Agent returns: "There are 82 students who are 18 years old."
# Generated code: len(df[df['age'] == 18])
```

**Viewing generated code:**
```python
result = agent.invoke("How many students are 18 years old?")
intermediate = result["intermediate_steps"]
# Shows the exact Python code the LLM generated — filters, aggregations, etc.
```

**Generating visualizations:**
```python
result = agent.invoke("Plot the gender count with bars.")
# Agent generates and executes matplotlib/seaborn code automatically
# Note: LLM understood "gender" maps to the "sex" column in the dataset
```

> **Key insight:** The LLM understands semantic meaning — asking about "gender" correctly maps to a column named "sex" in the dataset.

---

### Best Practices

| Practice | Why It Matters |
|---|---|
| **Use sandboxed environments** | Prevent unintended modifications to live data and avoid prompt injection risks |
| **Design clear, specific prompts** | Avoid ambiguous responses |
| **Validate with human expertise** | Confirm the LLM analyzed the correct data and returned correct results |
| **Iteratively refine prompts** | Improve accuracy through testing and adjustment |

---

## AI-Powered SQL Agents

### Benefits and Capabilities

AI-powered SQL agents bridge the gap between **natural language and SQL** — enabling anyone to query databases without knowing SQL syntax.

| Capability | Description |
|---|---|
| **Schema reading** | Reads and understands database schemas — only retrieves relevant table schemas (efficient) |
| **Multi-step querying** | Handles queries that require multiple SQL statements to answer fully |
| **Auto-retry on error** | If a query fails, captures the error, analyzes the traceback, and retries with a corrected query |
| **Natural language input** | Users ask in plain English; agent translates to SQL |

---

### Limitations

| Limitation | Detail |
|---|---|
| **Interpretive inaccuracies** | AI may sometimes misinterpret queries — especially ambiguous ones |
| **Complex query challenges** | Very complex queries may require manual adjustments |
| **Continuous validation needed** | Continuous testing is essential for reliability |

---

### SQL Agent Query Flow

```
User asks a natural language question
        |
SQL Agent receives the question
        |
LLM interprets and generates SQL query
        |
Database connector sends SQL to the database
        |
Database processes SQL query
        |
Raw data returned to database connector
        |
Data passed back to LLM
        |
LLM parses, processes, formats → clear natural language response
        |
User receives the answer
```

---

### Implementation with LangChain

#### Setup

```bash
# Create virtual environment
virtualenv my_env

# Install required libraries
pip install ibm-watsonx-ai langchain mysql-connector-python
```

#### Load the LLM

```python
from ibm_watsonx_ai.foundation_models import Model
from langchain_ibm import WatsonxLLM

credentials = {"url": "YOUR_URL", "apikey": "YOUR_KEY"}

model_id = "ibm/granite-13b-instruct-v2"
params = {
    "MAX_NEW_TOKENS": 500,
    "TEMPERATURE": 0.1    # low = more deterministic SQL output
}

watsonx_model = Model(
    model_id=model_id,
    credentials=credentials,
    params=params,
    project_id="skills-network",
    space_id=None
)

llm = WatsonxLLM(watsonx_model=watsonx_model)
```

#### Connect to MySQL Database

```python
from langchain_community.utilities import SQLDatabase

# Connection parameters
mysql_username = "root"
mysql_password = "password"
mysql_host = "127.0.0.1"
mysql_port = 3306
database_name = "Chinook"

# Build connection URI
db_uri = f"mysql+mysqlconnector://{mysql_username}:{mysql_password}@{mysql_host}:{mysql_port}/{database_name}"

# Connect using LangChain's SQLDatabase
db = SQLDatabase.from_uri(db_uri)
```

#### Create and Run the SQL Agent

```python
from langchain_community.agent_toolkits import create_sql_agent

# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,             # shows LLM reasoning + generated SQL
    agent_type="zero-shot-react-description"
)

# Run a natural language query
result = agent.invoke("How many Albums are listed in the database?")
# Agent generates SQL: SELECT COUNT(*) FROM Album
# Returns: "There are 347 albums in the database."
```

> With `verbose=True`, you see the complete **thought process** — all actions, the actual SQL queries generated, and the final answer.

---

## Natural Language Interfaces (NLI) for Data Systems

### How NLIs Work

NLIs transform human language into structured database queries through a multi-step pipeline:

| Step | What Happens |
|---|---|
| **1. User Input** | Everyday language, may be ambiguous or incomplete |
| **2. Query Formulation** | AI identifies entities, maps to schema, determines intent, generates SQL/Python |
| **3. Data Extraction** | Connects to data source, executes query, retrieves raw data |
| **4. Data Analysis** | Cleans, aggregates, identifies patterns |
| **5. Insight Synthesis** | Interprets results, identifies key findings |
| **6. Presentation** | Charts + natural language summaries |

### Types of NLIs

| Type | Description | Strengths | Limitations |
|---|---|---|---|
| **One-shot query systems** | Handle individual, standalone queries — no context between interactions | Simple, fast, easy to optimize | No memory, can't handle follow-ups |
| **Conversational interfaces** | Maintain context across multiple turns — dialogue-based exploration | Follow-up questions, iterative exploration, disambiguation | More complex, higher latency, requires dialogue state tracking |

> **Conversational interfaces** are gaining popularity for their ability to explore data incrementally through natural dialogue, clarify ambiguous queries, and maintain context across multiple turns.

---

### Key Technologies Powering NLIs

| Technology | Role |
|---|---|
| **Foundation LLMs** (GPT, BERT) | Understand user intent, handle paraphrasing, generate explanations |
| **Semantic parsing** | Extract entities, map natural language to schema elements, identify query operations |
| **Named entity recognition (NER)** | Identify products, regions, metrics in queries |
| **SQL generation** | Build syntactically correct SQL — handles joins, nested conditions, dialects |
| **Dialogue management** | Track conversation state, handle ambiguity, manage multi-turn flow |

**Dialogue management components:**

| Component | Description |
|---|---|
| **State tracking** | Track current state of data exploration across prior queries |
| **Decision making** | Choose external knowledge source, generate structured queries |
| **NL response generation** | Respond based on identified intents, extracted entities, context, and query results |

---

### Approaches to Building NLIs

| Approach | How It Works | Strengths | Weaknesses |
|---|---|---|---|
| **Rule-based** | Ontologies, knowledge graphs, grammar-based query interpretation | Strong semantic understanding, domain adaptation | Brittle with linguistic variations |
| **ML/Deep learning** | Text-to-SQL using deep learning, word embeddings, pre-trained LMs | Robust to paraphrasing, handles variations | Needs large training data, struggles with complex queries |
| **Hybrid** | Combines rule-based + ML — DL for NLU, ontologies for domain knowledge | Balances accuracy, robustness, and adaptability | More complex to build |

---

### Challenges

| Challenge | Detail |
|---|---|
| **Ambiguity** | "How are sales this year?" — total? by region? by product? by month? |
| **Schema mapping** | Natural language terms don't always match database column names |
| **Query complexity** | Nested conditions, multi-table joins, window functions, temporal operations |
| **Domain variation** | Finance, healthcare, retail — each has unique vocabulary and schema conventions |
| **Security & governance** | Must respect user permissions, privacy regulations, audit requirements |

**NL to SQL benchmarks:**

| Benchmark | Description |
|---|---|
| **WikiSQL** | NL questions + SQL pairs from Wikipedia tables |
| **Spider** | Cross-domain, complex SQL with joins and nested queries |
| **SParC** | Context-dependent multi-turn (follow-up questions) |
| **CoSQL** | Dialogue version — simulates real database querying conversations |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Pandas DataFrame Agent** | Pre-configured LangChain agent — accepts natural language, generates and executes Python code against a DataFrame |
| **`create_pandas_dataframe_agent`** | Creates the agent by passing LLM + DataFrame |
| **`return_intermediate_steps`** | Shows the generated Python code — essential for debugging |
| **Visualization** | Just ask in plain English — agent generates matplotlib/seaborn code automatically |
| **Pandas agent limitation** | Not for production without safeguards; use sandboxed environments |
| **SQL Agent** | Translates natural language to SQL, executes against database, returns natural language answer |
| **`create_sql_agent`** | Creates SQL agent — pass LLM, db connection, verbose, agent_type |
| **SQL agent capabilities** | Schema reading, multi-step querying, auto-retry on error |
| **SQL agent limitations** | Can misinterpret queries; complex queries may need manual adjustment |
| **One-shot NLI** | Single queries, no context — fast and simple |
| **Conversational NLI** | Multi-turn, maintains context — enables iterative exploration |
| **3 NLI approaches** | Rule-based, ML/deep learning, hybrid |
| **Key NLI challenge** | Ambiguity + schema mapping + query complexity |
| **`verbose=True`** | Shows complete LLM reasoning trace + generated SQL/Python — essential for debugging |

---

*Notes based on: Course 06 Module 03 — Using Built-in Agents in LangChain (Fundamentals of Building AI Agents)*
