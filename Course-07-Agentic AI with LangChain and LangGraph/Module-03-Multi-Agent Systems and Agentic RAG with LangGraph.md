# Course 07 — Module 03: Multi-Agent Systems and Agentic RAG with LangGraph

> A complete reference covering multi-agent system fundamentals, agentic RAG, governance, and building multi-agent workflows with LangGraph.

---

## Table of Contents

1. [Introduction to Multi-Agent Systems](#introduction-to-multi-agent-systems)
2. [Agent Specialization Design Principles](#agent-specialization-design-principles)
3. [Collaboration Patterns](#collaboration-patterns)
4. [Communication Protocols](#communication-protocols)
5. [Orchestration Frameworks](#orchestration-frameworks)
6. [Multi-Agent LLM Systems — Why & When](#multi-agent-llm-systems--why--when)
7. [Agentic RAG](#agentic-rag)
8. [Governance of Agentic AI](#governance-of-agentic-ai)
9. [Building Multi-Agent Systems with LangGraph](#building-multi-agent-systems-with-langgraph)
10. [Summary](#summary)

---

## Introduction to Multi-Agent Systems

A **Multi-Agent System (MAS)** consists of multiple autonomous agents that interact within a shared environment to achieve individual or collective goals. Each agent operates independently — perceiving its surroundings, making decisions, and taking actions.

> **Analogy:** A team of chefs in a kitchen, each specializing in a different dish, collaborating to prepare a complete meal — together tackling tasks too complex for any one chef alone.

### Core Components

| Component | Description | Analogy |
|---|---|---|
| **Agents** | Autonomous units with specific capabilities and goals | Warehouse robots |
| **Environment** | The context within which agents operate and interact | The warehouse floor and shelves |
| **Communication Protocols** | Standards enabling agents to share information and coordinate | Signals robots send to each other |

### Key Advantages

| Advantage | Description |
|---|---|
| **Scalability** | Add or remove agents without disrupting the system |
| **Flexibility** | Agents adapt to changes in environment or tasks |
| **Robustness** | System continues functioning even if individual agents fail |

### Challenges

| Challenge | Description |
|---|---|
| **Coordination complexity** | Ensuring all agents work harmoniously can be intricate |
| **Communication overhead** | Frequent inter-agent interactions can strain system resources |
| **Security concerns** | Protecting the entire system from any malicious agent |

---

## Agent Specialization Design Principles

When designing multi-agent systems, four core principles govern agent specialization:

| Principle | Description | Example |
|---|---|---|
| **Capability boundaries** | Each agent has a well-defined, focused scope | A summarizer agent shouldn't query databases — that's the retriever's job |
| **Expertise depth vs breadth** | Balance specialized agents with generalist coordinator agents | Coordinators route tasks and monitor overall progress |
| **Interface standardization** | Agents communicate through structured inputs/outputs (e.g. JSON schemas) | Enables orchestration through LangGraph, CrewAI, BeeAI, AutoGen |
| **Clear handoff patterns** | Agents gracefully pass tasks outside their expertise to appropriate agents | Document reader → summarization agent |

### Real-World Example — Research Assistant System

```
User Query
    |
Retriever Agent → pulls relevant documents from various sources
    |
Summarizer Agent → condenses documents, extracts key insights
    |
Critique Agent → evaluates for biases or gaps
    |
Compiler Agent → generates comprehensive final report
```

**Applications:** Legal tech · Healthcare · Enterprise knowledge management

---

## Collaboration Patterns

### Pipeline Pattern (Sequential)

Agents perform sequential handoffs — each agent's output becomes the next agent's input.

```
Research Agent → Editor Agent → Publisher Agent
```

### Hub-and-Spoke Pattern

A central coordinator dispatches tasks to specialist agents.

```
Content Manager Agent
    ├── Writer Agent
    ├── Fact Checker Agent
    └── SEO Optimizer Agent
```

### Parallel with Aggregation

Multiple agents perform tasks simultaneously, then a compiler integrates results.

```
Technical Writing Agent ─┐
SEO Analysis Agent ────── → Compiler Agent → Final Output
Fact-Checking Agent ───┘
```

### Interactive Dialogue

Agents exchange messages to clarify and refine — a requirements agent queries a data agent, which asks a filter agent for more details.

---

## Communication Protocols

| Protocol | Developer | Purpose |
|---|---|---|
| **MCP** (Model Context Protocol) | Anthropic (open standard) | Standardizes how AI models access and share context with external tools and data sources — JSON-RPC based universal connector |
| **ACP** (Agent Communication Protocol) | IBM | Standardized method for AI agents to communicate and collaborate — enables secure, scalable inter-agent coordination |

---

## Orchestration Frameworks

| Framework | Developer | Key Characteristics |
|---|---|---|
| **LangGraph** | LangChain Inc. | Graph-based workflows; explicit control over agent interactions; shared state; conditional routing; visual workflow representation |
| **CrewAI** | Open-source | Structured multi-agent workflows; strict interface contracts; typed data models (Pydantic); high-fidelity data passing |
| **AutoGen** | Microsoft | Conversational interfaces; agents self-organize and negotiate task ownership; adaptive collaboration |
| **IBM BeeAI** | IBM | Enterprise-grade; modular multi-agent orchestration; reliability and scalability; uses ACP for communication |

---

## Multi-Agent LLM Systems — Why & When

### Why Single LLM Agents Fall Short

| Problem | Description |
|---|---|
| **Context overload** | One agent juggling retrieval, analysis, writing, and critique loses track of details |
| **Role confusion** | Switching between creative writing and critical review causes inconsistent output |
| **Debugging difficulty** | Hard to identify which reasoning step caused an error when all logic runs in one model |
| **Quality dilution** | "Good enough" at many tasks but excels at none |

### How Multi-Agent LLM Systems Help

- Clear responsibilities per subtask
- Targeted prompt engineering per agent
- Modular debugging and quality control
- Scalable — add or update agents independently

### Real-World Examples

**Automated Market Research Report:**
```
Research Agent (data collection)
    → Data Analysis Agent (trends, anomalies)
    → Writing Agent (structured report)
    → Critique Agent (logic, completeness)
    → Editor Agent (grammar, style)
```

**Customer Support Automation:**
```
Intent Detection Agent (classify request)
    → Knowledge Retrieval Agent (fetch FAQ/history)
    → Response Generation Agent (personalized reply)
    → Escalation Agent (human handoff with summary)
```

**Legal Contract Review:**
```
Clause Extraction Agent
    → Compliance Agent (check regulations)
    → Risk Analysis Agent (flag ambiguous terms)
    → Summary Agent (executive summary)
    → Report Generator Agent (formatted legal memo)
```

### Implementation Considerations

| Challenge | Consideration |
|---|---|
| **Context management** | Share relevant info without overwhelming agents |
| **Granularity** | Balance too few (generalist) vs too many (overhead) agents |
| **Communication costs** | Balance thorough exchange with latency and compute efficiency |
| **Error handling** | Define fallback/retry mechanisms when agents fail |

---

## Agentic RAG

### Standard RAG vs Agentic RAG

**Standard RAG:**
```
User Query → Vector DB → Retrieved Context → Prompt → LLM → Response
(LLM called once, solely for generation)
```

**Agentic RAG:**
```
User Query → Agent (LLM-powered decision maker)
    |
    ├── Internal Documentation DB (policies, procedures, guidelines)
    ├── General Knowledge DB (industry standards, best practices)
    └── Fail-safe (out-of-scope queries)
    |
Retrieved Context → Prompt → LLM → Response
```

### What the Agent Adds

The LLM agent goes beyond response generation to **actively make decisions** that improve relevance and accuracy:

| Decision | Example |
|---|---|
| **Which database to query** | Company policy question → Internal docs; Industry standard question → General knowledge |
| **How to respond** | Answer with text, generate a chart, or provide a code snippet |
| **When to route to fail-safe** | Out-of-scope question → "Sorry, I don't have that information" |

### Routing Logic Example

```
"What's the company's policy on remote work?" → Internal Documentation DB
"What are industry standards for remote work in tech?" → General Knowledge DB
"Who won the World Series in 2015?" → Fail-safe (not in any database)
```

> The agent leverages the LLM's **language understanding** to interpret the query's context — not making a random guess.

### Agentic RAG Applications

Customer support · Legal tech (internal briefs + public case databases) · Healthcare · Enterprise knowledge management

---

## Governance of Agentic AI

### Why Agentic AI Requires Governance

Agentic AI differs from traditional AI in four key ways — all amplifying risk:

| Characteristic | Risk Amplification |
|---|---|
| **Underspecification** | Broad goals, no explicit instructions — unpredictable paths |
| **Long-term planning** | Decisions build on prior ones — errors compound |
| **Goal directedness** | Works toward a goal rather than responding to inputs |
| **Directedness of impact** | Can operate with no human in the loop |

> **Key principle:** Autonomy = Increased Risk. As autonomy increases, so do risks like misinformation, decision-making errors, and security vulnerabilities.

### Governance Framework

#### Technical Safeguards

| Layer | Safeguard | Purpose |
|---|---|---|
| **Model layer** | Input/output guardrails | Detect bad actors, policy violations, ethical misalignment |
| **Orchestration layer** | Infinite loop detection | Prevent costly failures, maintain user experience |
| **Tool layer** | Role-based access control | Limit each tool to specific agents — prevent out-of-scope actions |

#### Process Controls

| Control | Description |
|---|---|
| **Interruptibility** | Can we pause or shut down specific requests or the entire system? |
| **Human-in-the-loop** | When does AI require human approval? Can the agent wait for input? |
| **Confidential data treatment** | PII detection and masking to prevent sensitive information disclosure |
| **Risk-based permissions** | What actions should AI never take autonomously? |
| **Auditability** | Can we trace back how the AI arrived at a decision? |
| **Monitoring and evaluation** | Continuous oversight of AI performance |
| **Accountability** | Who takes responsibility when AI decisions lead to harm? |

#### Testing and Monitoring

| Practice | Purpose |
|---|---|
| **Red teaming** | Expose vulnerabilities before deployment |
| **Automated evaluations** | Detect hallucinations and compliance violations in production |
| **Continuous monitoring** | Understand system behavior under the hood — observability solutions |

---

## Building Multi-Agent Systems with LangGraph

### Shared State Definition

All agents read from and write to a shared state — the memory of the entire workflow.

```python
from typing import TypedDict, List, Optional

class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[dict]
    processed_data: Optional[dict]
    chart_config: Optional[dict]
    report: Optional[str]
    errors: List[str]
    next_action: str  # controls routing between agents
```

### Agent Nodes

Each agent is a function that receives the shared state, performs its task, and returns the updated state.

```python
def data_collector_agent(state: SalesReportState) -> SalesReportState:
    # Collect raw data based on state["request"]
    state["raw_data"] = {"q1": 100, "q2": 150}  # example
    state["next_action"] = "process"
    return state

def data_processor_agent(state: SalesReportState) -> SalesReportState:
    # Process raw_data → processed_data
    state["next_action"] = "visualize"
    return state

def chart_generator_agent(state: SalesReportState) -> SalesReportState:
    # Generate chart configuration from processed_data
    state["next_action"] = "report"
    return state

def report_generator_agent(state: SalesReportState) -> SalesReportState:
    # Generate final report text
    state["next_action"] = "complete"
    return state

def error_handler_agent(state: SalesReportState) -> SalesReportState:
    # Handle errors, set report to error message
    state["next_action"] = "complete"
    return state
```

### Routing Logic

```python
def route_next_step(state: SalesReportState) -> str:
    routing = {
        "collect":    "data_collector",
        "process":    "data_processor",
        "visualize":  "chart_generator",
        "report":     "report_generator",
        "error":      "error_handler",
        "complete":   "END"
    }
    return routing.get(state.get("next_action", "collect"), "END")
```

### Building and Compiling the Graph

```python
from langgraph.graph import StateGraph, END

def create_sales_report_workflow():
    workflow = StateGraph(SalesReportState)

    # Add agent nodes
    workflow.add_node("data_collector",  data_collector_agent)
    workflow.add_node("data_processor",  data_processor_agent)
    workflow.add_node("chart_generator", chart_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("error_handler",   error_handler_agent)

    # Conditional edges — route based on next_action in state
    workflow.add_conditional_edges("data_collector",  route_next_step,
        {"data_processor": "data_processor", "error_handler": "error_handler", END: END})
    workflow.add_conditional_edges("data_processor",  route_next_step,
        {"chart_generator": "chart_generator", "error_handler": "error_handler", END: END})
    workflow.add_conditional_edges("chart_generator", route_next_step,
        {"report_generator": "report_generator", "error_handler": "error_handler", END: END})
    workflow.add_conditional_edges("report_generator", route_next_step,
        {"error_handler": "error_handler", END: END})
    workflow.add_conditional_edges("error_handler", route_next_step, {END: END})

    workflow.set_entry_point("data_collector")
    return workflow.compile()
```

### Running the Workflow

```python
app = create_sales_report_workflow()

initial_state = SalesReportState(
    request="Q1-Q2 2024 Sales Analysis",
    raw_data=None,
    processed_data=None,
    chart_config=None,
    report=None,
    errors=[],
    next_action="collect"
)

final_state = app.invoke(initial_state)
print(final_state["report"])
```

### Workflow Graph Structure

```
data_collector
    ↓ (success)          ↘ (error)
data_processor          error_handler → END
    ↓ (success)          ↗ (error)
chart_generator
    ↓ (success)          ↗ (error)
report_generator
    ↓ (complete)
   END
```

**Key design patterns illustrated:**
- Each agent reads state → performs task → updates `next_action` → returns state
- Routing function maps `next_action` values to node names
- Error handler is reachable from any node
- `END` terminates the workflow
- Shared state is accessible and modifiable by all agents

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **MAS** | Multiple autonomous agents interacting in a shared environment toward collective goals |
| **3 MAS components** | Agents, environment, communication protocols |
| **Capability boundaries** | Each agent has a focused scope — don't mix roles |
| **Interface standardization** | JSON schemas enable orchestration across frameworks |
| **Handoff patterns** | Agents gracefully pass tasks outside their expertise |
| **Pipeline pattern** | Sequential handoffs — output of one becomes input of next |
| **Hub-and-spoke** | Central coordinator dispatches to specialist agents |
| **MCP** | Anthropic's standard for LLMs to access external tools/data via JSON-RPC |
| **ACP** | IBM's standard for agent-to-agent communication |
| **LangGraph** | Graph-based orchestration; shared state; conditional routing |
| **CrewAI** | Typed Pydantic-based strict interface contracts |
| **AutoGen** | Conversational self-organizing agents |
| **IBM BeeAI** | Enterprise-grade; uses ACP |
| **Agentic RAG** | Agent decides which database to query, how to respond, and when to fail safely |
| **Standard RAG** | LLM called once for generation only |
| **Agentic RAG benefit** | More intelligent routing → more relevant and accurate responses |
| **Autonomy = Risk** | As autonomy increases, so do risks — governance is essential |
| **4 governance layers** | Model layer, orchestration layer, tool layer, process controls |
| **Shared state** | TypedDict accessible/modifiable by all LangGraph agents |
| **`next_action`** | State field that drives routing between agents |
| **Error handler** | Centralized error management — reachable from any agent node |

---

*Notes based on: Course 07 Module 03 — Multi-Agent Systems and Agentic RAG with LangGraph (Agentic AI with LangChain and LangGraph)*
