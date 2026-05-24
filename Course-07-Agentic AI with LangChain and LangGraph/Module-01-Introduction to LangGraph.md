# Course 07 — Agentic AI with LangChain and LangGraph
## Module 01 — Introduction to LangGraph

> A complete reference covering Generative vs Agentic AI, AI Agents vs Agentic AI, LangGraph core concepts, LangGraph vs LangChain, and building your first LangGraph workflow.

---

## Table of Contents

1. [Generative AI vs Agentic AI](#generative-ai-vs-agentic-ai)
2. [AI Agents vs Agentic AI](#ai-agents-vs-agentic-ai)
3. [Core Components of LangGraph](#core-components-of-langgraph)
4. [LangGraph vs LangChain](#langgraph-vs-langchain)
5. [Building a LangGraph Workflow](#building-a-langgraph-workflow)
6. [LangGraph Architecture Best Practices](#langgraph-architecture-best-practices)
7. [Summary](#summary)

---

## Generative AI vs Agentic AI

| Aspect | Generative AI | Agentic AI |
|---|---|---|
| **Behavior** | Reactive — waits for a prompt, generates content | Proactive — pursues goals through a series of actions |
| **Trigger** | User prompt | User prompt (but then acts autonomously) |
| **Output** | Text, images, code, audio | Actions, decisions, coordinated results |
| **Human involvement** | Every step requires human direction | Minimal human intervention — loops autonomously |
| **Foundation** | LLMs (chatbots), diffusion models (images/audio) | LLMs as the reasoning engine |
| **Work ends at** | Generation — no further steps without input | Continued until goal is achieved |

### How Agentic AI Works

```
Perceive environment
        |
Decide action
        |
Execute action
        |
Learn from output
        |
(repeat with minimal human intervention)
```

### Chain-of-Thought Reasoning

Agentic systems use **chain-of-thought reasoning** — the LLM breaks down a complex task into smaller logical steps, effectively "thinking out loud":

> **Example — Conference planning agent:**
> 1. Understand conference requirements (size, duration, budget)
> 2. Research venues matching parameters
> 3. Check availability for qualifying venues
> 4. Evaluate costs and negotiate
> 5. Finalize and coordinate

Gen AI is the **cognitive engine** driving an agent's decision-making.

### Real-World Examples

| Type | Example |
|---|---|
| **Generative AI** | YouTuber using a chatbot to review scripts, suggest thumbnails, generate music — human reviews at each step |
| **Agentic AI** | Personal shopping agent — hunts availability across platforms, monitors prices, handles checkout, coordinates delivery — largely autonomous |

> **The future:** Intelligent collaborators that know **when to generate** (explore options) and **when to act** (commit to a course of action).

---

## AI Agents vs Agentic AI

### AI Agents — Single Entity

AI Agents are **autonomous software entities** designed for **goal-directed task execution** within specific digital environments.

| Capability | Description |
|---|---|
| **Autonomy** | Functions with minimal human intervention — perceives inputs, reasons, executes in real-time |
| **Task-specificity** | Optimized for narrow, well-defined tasks (email filtering, DB querying, customer support) |
| **Reactivity** | Responds to inputs from users, APIs, or software in real time |

### Agentic AI — Multi-Agent Systems

Agentic AI brings **multiple agents together into a collaborative team**:

| Feature | Description |
|---|---|
| **Task decomposition** | Goals automatically split into subtasks |
| **Inter-agent communication** | Agents share updates and results via messaging or shared memory |
| **Memory and reflection** | Agents remember past steps and learn from outcomes |
| **Orchestration** | A lead agent or system coordinates the team |

> **Example:** Planning a vacation — one agent books the flight, another finds hotels, a third checks visa requirements, a coordinator ensures everything matches preferences.

### Architectural Differences

| Feature | AI Agent | Agentic AI |
|---|---|---|
| **Design** | One agent, one task | Multiple agents with distinct roles |
| **Communication** | No coordination | Constant inter-agent communication |
| **Memory** | Stateless or minimal | Persistent memory across tasks and sessions |
| **Reasoning** | Linear (step A → B) | Iterative planning and re-planning |
| **Scalability** | Limited to task size | Scales to multi-agent, multi-stage problems |
| **Applications** | Chatbots, virtual assistants | Supply chain coordination, enterprise optimization |

### Memory Types in Agentic AI

| Memory Type | Description |
|---|---|
| **Episodic memory** | Task-specific history — recall prior actions and feedback |
| **Semantic memory** | Long-term structured facts and domain knowledge |
| **Vector memory** | Similarity-based retrieval (RAG) |

### Frameworks for Building Agentic AI

**LangChain** · **LangGraph** · **IBM Bee** · **CrewAI** · **AutoGen**

---

## Core Components of LangGraph

**LangGraph** is an advanced framework within the LangChain ecosystem for building **stateful, multi-agent applications**. It models agent workflows as **graphs**.

### Three Core Primitives

| Component | Description |
|---|---|
| **Nodes** | Individual steps or functions that do the actual computation — process the current state |
| **Edges** | Define how execution flows from one node to the next |
| **State** | Shared data structure (memory) that persists and evolves across all nodes |

### Key Capabilities

| Capability | Description |
|---|---|
| **Looping and branching** | Agents make dynamic decisions — workflows can revisit nodes |
| **State persistence** | Maintains context over long interactions |
| **Human-in-the-loop** | Pause execution for human review or approval |
| **Time travel** | Rewind to previous states for debugging |

### Why Not Just Use For Loops?

| | Traditional Loops | LangGraph |
|---|---|---|
| **State management** | No persistent state | Explicit state maintained across nodes |
| **Conditional transitions** | Static if/else | Dynamic runtime branching |
| **Modularity** | Monolithic | Each node independently developed and tested |
| **Observability** | Hard to trace | Clear execution path — integrates with Mermaid diagrams |

> **Customer support example:** A `while` loop keeps asking for valid input but has no memory. A LangGraph workflow can branch, loop, pause for human input, and resume — all while retaining full conversational memory.

---

## LangGraph vs LangChain

### What is LangChain?

LangChain provides an **abstraction layer** for chaining LLM operations into sequential pipelines using a **DAG (Directed Acyclic Graph)** — tasks always move forward, never loop back.

**Example LangChain workflow (retrieve → summarize → answer):**
```
Document Loader (retrieve) | Text Splitter
        |
Chain + Prompt + LLM (summarize)
        |
Chain + Memory + Prompt + LLM (answer)
```

### What is LangGraph?

LangGraph is a **graph-based orchestration framework** for stateful, multi-agent workflows. It supports **loops**, **branches**, and **dynamic transitions** — the next step can depend on evolving conditions.

**Example LangGraph workflow (task management assistant):**
```
Nodes: process_input | add_task | complete_task | summarize
Edges: conditional routing based on user intent
State: shared task list accessible by all nodes
```

### Direct Comparison

| Feature | LangChain | LangGraph |
|---|---|---|
| **Primary focus** | Chaining LLM operations into applications | Multi-agent systems and stateful workflows |
| **Structure** | Chain / DAG — linear, moves forward only | Graph — allows loops, revisiting states |
| **State management** | Limited — passes info forward, no persistent global state | Robust — shared global state all nodes access and modify |
| **Components** | Prompt, LLM, Memory, Agent | Nodes, Edges, State |
| **Best for** | Sequential tasks: RAG, chatbots, summarization | Complex, interactive, adaptive systems |
| **Loops** | Not natively supported | Core feature |
| **Human-in-the-loop** | Limited | First-class built-in support |
| **Debugging** | Harder — opaque state passing | Clear — time travel, LangSmith integration |
| **Learning curve** | Lower | Higher — explicit node/edge/state definitions |

### When to Use Which

| Use Case | Use LangChain | Use LangGraph |
|---|---|---|
| Workflow complexity | Linear, clearly defined | Complex, conditional branching |
| Development speed | Quick prototype/MVP | Production-grade reliability |
| Memory needs | Stateless or current conversation only | Long-term across sessions |
| Interaction style | Simple LLM tool use | Multi-turn, human-in-the-loop |
| System design | Document Q&A, summarization | Multi-agent, retries, approvals |

> **Note:** LangChain is deprecating its legacy agent framework in favor of LangGraph. LangGraph manages the agent's iterative cycles and tracks the scratchpad as messages within its state.

---

## Building a LangGraph Workflow

### Complete Example — Counter with Conditional Loop

**Goal:** Start at n=1, increment and generate a random letter, print results, stop when n ≥ 13.

#### Step 1 — Define State

```python
from typing import TypedDict

class ChainState(TypedDict):
    n: int
    letter: str
```

> State is commonly defined with `TypedDict` but can be lists, nested structures, or message sequences. It holds all graph inputs, intermediate values, and outputs.

#### Step 2 — Define Nodes

```python
import random
import string

def add(state: ChainState) -> ChainState:
    """Increment n and generate a random letter."""
    new_letter = random.choice(string.ascii_lowercase)
    return {**state, "n": state["n"] + 1, "letter": new_letter}

def print_out(state: ChainState) -> ChainState:
    """Print current state — side effect only, no modification."""
    print(f"n={state['n']}, letter={state['letter']}")
    return state  # return unchanged state
```

> Nodes are functions that process state. Some modify state, others are used purely for side effects (like printing). The returned keys and values are what gets updated in state.

#### Step 3 — Define Conditional Edge Logic

```python
def stop_condition(state: ChainState) -> bool:
    """Return True to stop, False to continue."""
    return state["n"] >= 13
```

#### Step 4 — Build the Graph

```python
from langgraph.graph import StateGraph, END

# Create state graph
graph = StateGraph(ChainState)

# Add nodes
graph.add_node("add", add)
graph.add_node("print_out", print_out)

# Add regular edge: add → print_out
graph.add_edge("add", "print_out")

# Add conditional edge: print_out → (end or add)
graph.add_conditional_edges(
    "print_out",         # source node
    stop_condition,      # function that evaluates state
    {
        True: END,       # if stop_condition returns True → end
        False: "add"     # if False → loop back to add
    }
)

# Set entry point
graph.set_entry_point("add")

# Compile into a runnable app
app = graph.compile()
```

#### Step 5 — Run the Workflow

```python
result = app.invoke({"n": 1, "letter": ""})
print(result)  # Final state after workflow completes
```

**Execution flow:**
```
Start: {n: 1, letter: ""}
→ add: {n: 2, letter: "k"}
→ print_out: prints "n=2, letter=k"
→ stop_condition(n=2) → False → loop back to add
→ add: {n: 3, letter: "m"}
→ ...
→ add: {n: 13, letter: "z"}
→ print_out: prints "n=13, letter=z"
→ stop_condition(n=13) → True → END
```

### Key Methods Reference

| Method | Description |
|---|---|
| `StateGraph(StateClass)` | Create a new state graph with your state schema |
| `add_node(name, function)` | Add a node — name can differ from function name |
| `add_edge(from, to)` | Add a fixed edge between two nodes |
| `add_conditional_edges(from, func, mapping)` | Add dynamic routing based on state |
| `set_entry_point(name)` | Set the first node to execute |
| `compile()` | Build the runnable application |
| `app.invoke(initial_state)` | Run the workflow with an initial state dict |

### Visualization

LangGraph graphs can be visualized using **Mermaid diagrams**, showing nodes and edges clearly for debugging and documentation.

---

## LangGraph Architecture Best Practices

### State Design

```python
# Good: descriptive names, flat structure
class SupportAgentState(TypedDict):
    user_input: str
    agent_response: str
    issue_type: str
    retry_count: int

# Also good: document processing
class DocumentProcessingState(TypedDict):
    file_path: str
    text_content: str
    summary: str
    analysis_results: dict
```

- Use **descriptive names** like `user_query`, `agent_response`
- Keep structures **flat** — avoid deeply nested state
- Define state schema **explicitly** before building the graph

### Node Types

| Node Type | Purpose |
|---|---|
| **Processing nodes** | Data transformation or computation |
| **Validation nodes** | Check conditions or data integrity |
| **Integration nodes** | Interface with external systems (APIs, databases) |
| **Decision nodes** | Direct workflow paths based on conditions |

**Node design rule:** Each node should perform a **single, clear task**.

### Edge Patterns

```python
# Conditional routing example
def route_decision(state):
    if state["retry_count"] > 2:
        return "human_review"
    elif state["issue_type"] == "resolved":
        return "end_interaction"
    else:
        return "continue_processing"
```

### Error Handling

- Include error-specific state fields
- Create dedicated error-handling nodes
- Implement graceful fallbacks
- Use **Retry nodes** for transient failures
- Use **Error nodes** to route to human intervention after repeated failures

### Common Mistakes to Avoid

| Mistake | Better Approach |
|---|---|
| Oversized nodes handling multiple tasks | Modular nodes with single responsibilities |
| Deeply nested or unclear states | Flat, explicitly defined state schemas |
| Ignoring error conditions | Plan error handling early in design |
| Adding all complexity upfront | Start simple, add complexity incrementally |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Generative AI** | Reactive — generates content from prompts; stops at generation |
| **Agentic AI** | Proactive — pursues goals through perceive → decide → act → learn loops |
| **Chain-of-thought** | LLM breaks complex tasks into smaller logical steps — powers agent reasoning |
| **AI Agent** | Single entity, one task, reactive, stateless or minimal memory |
| **Agentic AI (multi-agent)** | Multiple agents, coordinated, persistent memory, iterative planning |
| **LangGraph** | Framework for stateful multi-agent applications using graph structure |
| **Node** | Function that processes the current state |
| **Edge** | Defines execution flow between nodes |
| **State** | Shared, persistent memory across all nodes — defined with `TypedDict` |
| **Conditional edge** | Routes to different nodes based on state evaluation at runtime |
| **LangChain** | Linear DAG-based chaining — best for sequential pipelines |
| **LangGraph** | Graph-based — supports loops, branches, persistent state, human-in-the-loop |
| **When to use LangGraph** | Complex, adaptive, multi-agent, stateful workflows |
| **When to use LangChain** | Simple, sequential, stateless or light-memory tasks |
| **`add_conditional_edges`** | Dynamic routing — maps function output values to destination nodes |
| **`compile()`** | Builds the runnable LangGraph application from the defined graph |
| **Time travel** | LangGraph feature to rewind to previous states for debugging |

---

*Notes based on: Course 07 Module 01 — Introduction to LangGraph (Agentic AI with LangChain and LangGraph)*
