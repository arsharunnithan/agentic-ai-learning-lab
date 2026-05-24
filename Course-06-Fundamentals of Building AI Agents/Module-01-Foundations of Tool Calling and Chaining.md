# Course 06 — Fundamentals of Building AI Agents
## Module 01 — Foundations of Tool Calling and Chaining

> A complete reference covering AI agents, compound AI systems, tool calling, LangChain tool design, and the ReAct agent framework.

---

## Table of Contents

1. [From Models to Compound AI Systems](#from-models-to-compound-ai-systems)
2. [What are AI Agents?](#what-are-ai-agents)
3. [The ReAct Framework](#the-react-framework)
4. [AI System Design Paradigms](#ai-system-design-paradigms)
5. [When to Use (and Not Use) Agents](#when-to-use-and-not-use-agents)
6. [Tool Calling](#tool-calling)
7. [Tools in LangChain](#tools-in-langchain)
8. [Building Agents in LangChain](#building-agents-in-langchain)
9. [Building Agents with LangGraph](#building-agents-with-langgraph)
10. [Popular Built-in LangChain Tools](#popular-built-in-langchain-tools)
11. [Summary](#summary)

---

## From Models to Compound AI Systems

**The problem with standalone models:**
- Limited by the data they were trained on
- Hard to adapt without expensive fine-tuning
- Cannot access real-time or private data

**Example:** Ask an LLM "How many vacation days do I have left?" — it will guess incorrectly because it has no access to your database.

**The solution — Compound AI Systems:**

Give the model access to tools and external systems. The same query now:
```
User query
    |
LLM creates a search query
    |
Database is searched
    |
Answer returned to LLM
    |
LLM generates: "Maya, you have 10 days left."
```

**Key insight:** Compound AI systems are modular — combine models, verifiers, databases, APIs, and other tools. This is faster and easier to adapt than fine-tuning a model.

> **RAG** is one of the most popular compound AI systems. Most RAG systems have **programmatic control logic** — a fixed path defined by a human.

---

## What are AI Agents?

When the **control logic of a compound AI system is handled by an LLM** rather than hard-coded programmatic rules, you have an **agent**.

**Three core capabilities of LLM agents:**

| Capability | Description |
|---|---|
| **Reason** | LLM plans and reasons about each step — breaks down complex problems |
| **Act** | Uses external **tools** — search, databases, calculators, APIs, other LLMs |
| **Memory** | Stores inner reasoning logs and conversation history for personalization |

**The autonomy spectrum:**

```
Programmatic (think fast)                    Agentic (think slow)
     |-------------------------------------------|
Fixed path, narrow tasks              Dynamic planning, complex tasks
Efficient, predictable                Flexible, handles ambiguity
```

**When to go agentic:** Complex tasks with many possible paths (e.g. "How many 2-oz sunscreen bottles should I pack for Florida?") — requires checking vacation days, weather forecast, sun exposure guidelines, and doing math.

**When to stay programmatic:** Narrow, well-defined problems where every query follows the same path (e.g. a vacation policy lookup).

---

## The ReAct Framework

**ReAct** (Reasoning + Acting) is the most popular way to configure LLM agents.

```
User Query
    |
LLM Reasons (think slow, plan the work)
    |
LLM decides on an Action (which tool to call, with what inputs)
    |
Tool is called → returns result
    |
LLM Observes the result
    |
Is the answer complete? → Yes → Final Answer
                       → No  → Adjust plan, try again
```

**ReAct loop in detail:**
1. **Thought** — LLM reasons about the problem step by step
2. **Action** — selects a tool and parameters
3. **Observation** — receives tool output
4. **Repeat** until final answer is reached

> If a tool returns an error or wrong answer, the agent observes this and adjusts — it doesn't just give up.

---

## AI System Design Paradigms

| Paradigm | Process | Best For | Pros | Cons |
|---|---|---|---|---|
| **Single LLM** | Input → LLM → Output | Summarization, classification, translation | Simple, fast, low cost | No memory, not adaptable |
| **Structured Workflow** | Predefined multi-step logic | Compliance, document pipelines, batch processing | Predictable, auditable, reliable | Rigid, hard to adapt |
| **Autonomous Agent** | Plan → Act → Observe → repeat | Complex, open-ended, adaptive tasks | Flexible, handles unforeseen situations | Unpredictable, costly, harder to debug |

**Hybrid architectures** are common in practice — combining workflow reliability with agent flexibility.

**Emerging standards:**
- **MCP** (Model Context Protocol) — Anthropic
- **ACP** (Agent Communication Protocol) — IBM

---

## When to Use (and Not Use) Agents

### Four-Criteria Decision Framework

**1. Is the task ambiguous or predictable?**
- Ambiguous (unclear solution path, exploration needed) → Agent
- Predictable (clear rules, repeatable structure) → Workflow

**2. Is the value worth the cost?**
- Agents can consume **10–100× more tokens** than workflows
- High-ROI strategic tasks → Agent
- Basic customer support → Workflow

**3. Does the agent meet minimum capabilities?**
Test the agent on 3–5 key skills before deploying. If it fails — scale back.

**4. What happens if the agent makes a mistake?**
- Reversible, manageable risk → Agent may be appropriate
- Zero-error systems → Do NOT use agents

### When NOT to Use Agents

| Scenario | Reason |
|---|---|
| High-volume, low-margin tasks (basic chat support) | Too expensive |
| Real-time applications (instant fraud detection) | Too slow |
| Zero-error systems (medical, security decisions) | Too risky |
| Heavily regulated industries | Need deterministic outcomes |

### Risk Management

| Risk Level | Strategy |
|---|---|
| High-stakes, hard to notice | Human review + multiple validation layers |
| High-stakes, visible | Automated checks + oversight mechanisms |
| Low-stakes | Monitor with user feedback + lightweight validation |

### Phased Deployment

1. **Proof of Concept** — low-risk, reversible tasks
2. **Pilot Program** — moderate-risk under supervision
3. **Production Scaling** — only after demonstrating safety and performance

### Effective Agent Architecture: 3 Key Components

| Component | Description |
|---|---|
| **Environment** | The digital space where the agent operates |
| **Tools** | The interfaces the agent uses to act or observe |
| **System Prompts** | Rules, goals, and behaviors guiding the agent |

> Start with read-only tool access. Add human approvals for critical steps. Enable comprehensive logging.

---

## Tool Calling

**Tools** are external functions that allow LLMs to interact with the real world — moving from text generation to real-world action.

### Why LLMs Need Tools

Without tools, LLMs are **pattern recognition machines**:
- Don't know real-time facts
- Can't access APIs
- Can't do precise math

> Example: Ask an LLM "What is 371 × 492?" — it may hallucinate **158,213**. With a calculator tool, it returns the correct answer: **182,532**.

### What Tools Enable

| Capability | Example |
|---|---|
| Real-time data retrieval | Weather, stock prices, news |
| RAG with private data | Company documents, personal databases |
| Multimodal processing | Images, audio, video |
| Extended memory | Context beyond the context window |
| External system interaction | APIs, emails, code execution |
| Precise computation | Math, logic, SQL queries |

### Traditional vs Embedded Tool Calling

**Traditional Tool Calling:**
```
Client App → [message + tool definitions] → LLM
LLM → [tool call recommendation] → Client App
Client App → calls the tool
Tool result → LLM
LLM → final answer
```
- Risk: LLM can hallucinate or make up incorrect tool calls

**Embedded Tool Calling:**
```
Client App → message → Library → [message + tool definitions] → LLM
LLM → [tool call] → Library (executes tool, retries if needed)
Library → final answer → Client App
```
- The **library** handles tool execution and retries — prevents hallucination

### Tool Calling Workflow (LangChain)

```
1. Define the tool (name, description, parameters)
2. User sends question
3. LLM evaluates available tools
4. LLM generates structured tool call (JSON) — specifying tool name + parameters
5. External system executes the tool
6. Result passed back to LLM
7. LLM generates final natural language response
```

> **LLMs do not execute tools directly** — they generate a structured representation (JSON) indicating which tool to call and with what parameters. The actual execution happens externally.

> **Function calling = Tool calling** — same concept, different terminology. OpenAI uses "function calling"; Anthropic and LangChain use "tool calling".

---

## Tools in LangChain

### Tool Schema Components

| Component | Description |
|---|---|
| **Name** | Unique identifier — use intuitive names like `add_numbers` |
| **Description** | Brief explanation of purpose and when to use it — the LLM reads this to decide which tool to call |
| **Parameters** | Expected inputs with types and descriptions |

### Creating Tools

**Method 1: Tool Class**
```python
from langchain.tools import Tool

def add_numbers(inputs: str) -> str:
    """Add numbers from a string input. Example: '10, 20, 30' -> 60"""
    digits = [int(x) for x in inputs.split(',') if x.strip().isdigit()]
    return str(sum(digits))

add_tool = Tool(
    name="AddNumbers",
    func=add_numbers,
    description="Adds numbers provided as a comma-separated string. Input: '10, 20, 30'"
)

# Direct invocation
result = add_tool.invoke("10, 20, 30")  # Returns: 60
```

**Method 2: @tool Decorator (preferred — cleaner syntax)**
```python
from langchain.tools import tool
from typing import List

@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Add a list of numbers, optionally summing their absolute values.
    
    Parameters:
        numbers: List of floats to sum
        absolute: If True, sum absolute values. Default: False
    
    Returns:
        The sum as a float
    
    Example:
        Input: {"numbers": [-10, -20, -30], "absolute": False} -> -60.0
        Input: {"numbers": [-10, -20, -30], "absolute": True}  -> 60.0
    """
    if absolute:
        return sum(abs(n) for n in numbers)
    return sum(numbers)
```

> The `@tool` decorator supports **multiple typed inputs** and **structured JSON inputs** — preferred for modern LangChain applications.

### Tool Attributes

| Attribute | Description |
|---|---|
| `name` | Derived from function name |
| `description` | Extracted from docstring — **critical for LLM tool selection** |
| `args` | Expected input schema — names, types, and defaults |

### Structured vs Simple Tools

| | Simple Tool | Structured Tool |
|---|---|---|
| **Input** | Single string | Multiple typed inputs (List, float, bool, etc.) |
| **Agent type** | `zero-shot-react-description` | `structured-chat-zero-shot-react-description` |
| **Flexibility** | Basic | High — supports complex inputs |

---

## Building Agents in LangChain

### Agent Architecture Components

| Component | Role |
|---|---|
| **LLM** | Core intelligence — interprets input, decides what to do next |
| **Tools** | External functions the LLM can call |
| **Memory** | Short-term (RAM), structured (SQL), semantic (VectorDB) |
| **Action** | Structured tool call generated by the LLM |
| **Environment** | Everything outside — OS, internet, APIs, physical devices |

### Agent Reasoning Loop

```
User query received
    |
LLM reasons about the problem (Thought)
    |
LLM decides which tool to call (Action + Action Input)
    |
Tool is executed → result returned (Observation)
    |
LLM checks if the answer is complete
    → Yes → Final Answer
    → No  → Loop back to Thought
```

### Zero-Shot ReAct Agent

```python
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=[add_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,               # print reasoning step-by-step
    handle_parsing_errors=True  # recover from malformed LLM outputs
)

result = agent.run("What is the total GDP of the US ($27.72T), Canada ($2.14T), and Mexico ($1.79T)?")
# Agent: Thought → Action (AddTool) → Observation → Final Answer: $31.55 trillion
```

### Structured Chat ReAct Agent

```python
agent = initialize_agent(
    tools=[add_numbers_with_options],
    llm=llm,
    agent="structured-chat-zero-shot-react-description"
)

result = agent.invoke({"input": "Sum -10, -20, -30 using absolute values"})
# Supports typed/structured tool inputs
```

### Agent Type Reference

| Agent Type | Input Support | Best For |
|---|---|---|
| `zero-shot-react-description` | Plain strings | Simple tasks, text-based tools |
| `structured-chat-zero-shot-react-description` | Typed inputs, JSON | Complex tools with multiple parameters |
| `openai-functions` | Structured outputs | GPT models with function calling |

---

## Building Agents with LangGraph

LangGraph is becoming the **preferred approach** over `initialize_agent` for robust, multi-step agent workflows — offering more flexibility and control.

```python
from langgraph.prebuilt import create_react_agent

# Create the agent with custom prompt
agent = create_react_agent(
    llm,
    tools=[add_tool, multiply_tool, divide_tool, subtract_tool],
    state_modifier="You are a helpful mathematical assistant."
)

# Invoke with message format
response = agent.invoke({
    "messages": [("human", "What is 7 × 3?")]
})

# Extract final answer
final_answer = response["messages"][-1].content
```

### Multi-Tool Math Toolkit Example

```python
@tool
def add_numbers(inputs: str) -> str:
    """Add numbers from comma-separated string."""
    ...

@tool
def multiply_numbers(inputs: str) -> str:
    """Multiply numbers from comma-separated string."""
    ...

@tool
def divide_numbers(inputs: str) -> str:
    """Divide first number by second number."""
    ...

@tool
def subtract_numbers(inputs: str) -> str:
    """Subtract second number from first number."""
    ...

tools = [add_numbers, multiply_numbers, divide_numbers, subtract_numbers]
agent = create_react_agent(llm, tools)
```

### Combining Tools: Multi-Step Query Example

```python
# Add Wikipedia search to the math toolkit
from langchain_community.tools import WikipediaQueryRun

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information. Input: search query string."""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)

tools = [add_numbers, multiply_numbers, ..., search_wikipedia]
agent = create_react_agent(llm, tools)

# Combined query: information retrieval + math
response = agent.invoke({
    "messages": [("human", "What is the population of Canada? Multiply it by 0.75.")]
})
# Agent: 1) Search Wikipedia for Canada population
#         2) Apply multiplication tool
#         3) Return natural language answer
```

---

## Popular Built-in LangChain Tools

### Search Tools

| Tool | Function |
|---|---|
| **SerpAPI** | Web search — returns answers |
| **Google Search** | Returns URLs, snippets, titles |
| **Tavily Search** | AI-optimized search for agents — returns URLs, content, images, answers |
| **Wikipedia** | Knowledge base search — summaries and articles |

### Code & Data Analysis

| Tool | Function |
|---|---|
| **Python REPL** | Executes Python code — calculations, data analysis, automation |
| **Pandas DataFrame** | Interact with and analyze tabular data |
| **SQL Database Toolkit** | Query and manipulate SQL databases via natural language |
| **LLMMathChain** | Translates math problems to Python and evaluates them |
| **JSON Toolkit** | Interact with large JSON/dictionary objects |

### Web Browsing & Interaction

| Tool | Function |
|---|---|
| **Requests Toolkit** | HTTP requests — interact with web APIs |
| **PlayWright Browser** | Browser automation — navigate and interact with web pages |
| **ArXiv** | Search and retrieve scientific papers |

### Productivity & Collaboration

| Tool | Function |
|---|---|
| **Gmail Toolkit** | Read, send, manage emails |
| **Office365 Toolkit** | Interact with Microsoft 365 (Outlook, OneDrive) |
| **Slack Toolkit** | Send/read Slack messages |
| **Github Toolkit** | Manage repos, issues, pull requests |
| **Google Calendar** | Create, read, update calendar events |

### File & Document Processing

| Tool | Function |
|---|---|
| **File System** | Read, write, manage local files |
| **Google Drive** | Access and manage cloud files |
| **VectorStoreQA** | Query documents in vector databases |
| **Document Loaders** | Extract content from PDF, DOCX, etc. |

### Financial & Business

| Tool | Function |
|---|---|
| **Yahoo Finance** | Financial news and market info |
| **Polygon IO** | Real-time and historical market data |
| **Stripe** | Payment processing and subscriptions |

### AI & ML Integration

| Tool | Function |
|---|---|
| **DALL-E Image Generator** | Generate images from text descriptions |
| **HuggingFace Hub Tools** | Access ML models on HuggingFace |
| **Google Imagen** | Image generation via Vertex AI |

> Some tools are **free**, others require **paid API keys**. Always check the official LangChain documentation before integrating.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Compound AI system** | Multiple components around a model — tools, verifiers, databases |
| **Agent** | Compound AI system where the LLM controls the logic (not hardcoded rules) |
| **3 agent capabilities** | Reason (plan), Act (use tools), Memory (store context) |
| **ReAct** | Reasoning + Acting loop — Thought → Action → Observation → repeat |
| **Autonomy spectrum** | Programmatic (fast, narrow) ↔ Agentic (flexible, complex) |
| **4-criteria framework** | Task ambiguity, cost value, capability test, failure impact |
| **Don't use agents** | Real-time, zero-error, high-volume, heavily regulated systems |
| **Tool calling** | LLM generates structured JSON → external system executes → result back to LLM |
| **LLMs don't execute tools** | They generate the call; execution is external |
| **Function = Tool calling** | Same concept; OpenAI says "function", others say "tool" |
| **Embedded tool calling** | Library handles execution + retries — prevents hallucination |
| **Tool schema** | Name + description (critical for LLM selection) + parameters |
| **`@tool` decorator** | Preferred syntax — supports structured, typed, multi-input tools |
| **zero-shot-react** | Simple string tools; `structured-chat-zero-shot-react` for typed inputs |
| **LangGraph** | Preferred over `initialize_agent` — more flexible, customizable |
| **Toolkit** | Collection of related tools (e.g. Gmail Toolkit, SQL Database Toolkit) |

---

*Notes based on: Course 06 Module 01 — Foundations of Tool Calling and Chaining (Fundamentals of Building AI Agents)*
