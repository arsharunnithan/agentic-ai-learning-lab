# Course 08 — Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI
## Module 01 — Agentic Frameworks and LangGraph Design Patterns for Effective AI Systems

> A complete reference covering agentic AI fundamentals, multi-agent framework comparison, and the four core LangGraph design patterns.

---

## Table of Contents

1. [Agentic AI & Multi-Agent Systems](#agentic-ai--multi-agent-systems)
2. [Framework Comparison](#framework-comparison)
3. [Design Pattern 1 — Sequential (Prompt Chaining)](#design-pattern-1--sequential-prompt-chaining)
4. [Design Pattern 2 — Routing](#design-pattern-2--routing)
5. [Design Pattern 3 — Parallelization](#design-pattern-3--parallelization)
6. [Design Pattern 4 — Orchestrator](#design-pattern-4--orchestrator)
7. [Design Pattern 5 — Evaluator-Optimizer](#design-pattern-5--evaluator-optimizer)
8. [Summary](#summary)

---

## Agentic AI & Multi-Agent Systems

### What Makes a System "Agentic"?

Agentic AI systems are **autonomous systems that make decisions and take action to achieve goals** — proactive problem solvers built to navigate complex environments.

| Capability | Description |
|---|---|
| **Multi-step reasoning** | Break down complex tasks into manageable steps |
| **Decision-making** | Choose the best course of action at each step |
| **Tool use** | Call external tools, APIs, or other agents to extend capabilities |
| **Memory** | Retain context from past interactions |
| **Goal-oriented behavior** | All decisions and actions serve a defined objective |

> **Analogy:** A calculator is reactive — it only responds to inputs. Agentic AI is like a research assistant who understands your goal, breaks it into steps, uses tools to gather information, and keeps working until the problem is solved.

### Why Use Frameworks Instead of Building from Scratch?

Without frameworks you're stuck:
- Managing complex message protocols manually
- Syncing state across agents yourself
- Writing custom coordination logic
- Handling errors distributed across agents
- Building monitoring and debugging from scratch

**Frameworks solve all of this** — communication protocols built-in, automatic state management, predefined coordination patterns, standardized error handling, integrated observability.

### Multi-Agent System Benefits

| Benefit | Description |
|---|---|
| **Specialization** | Each agent designed to excel at a specific task |
| **Parallel processing** | Multiple agents work simultaneously |
| **Fault tolerance** | System continues functioning if individual agents fail |
| **Scalability** | Add more agents as complexity grows |
| **Modularity** | Easy to update or replace individual components |

---

## Framework Comparison

### The Five Major Frameworks

| Framework | Origin | Core Approach | Best For |
|---|---|---|---|
| **CrewAI** | Open-source | Role-based team simulation — agents with distinct roles collaborate as a crew | Content pipelines, automated reporting, rapid prototyping |
| **LangGraph** | LangChain Inc. | Directed graph workflows — nodes (steps) and edges (transitions) with fine-grained control | Complex structured workflows, document automation, decision trees |
| **AutoGen (AG2)** | Microsoft | Dialogue-driven — agents communicate via conversations, human-in-the-loop | Educational platforms, conversational coding, technical support |
| **Pydantic AI** | Open-source | Schema-enforced outputs — type validation on every step | APIs, enterprise data pipelines, production systems requiring reliability |
| **BeeAI** | IBM | Enterprise-grade — modular, scalable, tool-integrated, production-ready | Enterprise automation, scalable deployments |

### Framework Selection Guide

| Need | Use |
|---|---|
| Quick prototyping, easy onboarding | **CrewAI** or **AutoGen** |
| Structured workflows, state management, error recovery | **LangGraph** |
| Type-safe outputs, enterprise APIs | **Pydantic AI** |
| Enterprise-grade deployment, scalability | **BeeAI** |

### Framework Deep Dives

#### CrewAI

Define agents with **roles, goals, and backstories** → assign tasks → combine into a Crew → execute.

```python
# Standard CrewAI pipeline
researcher = Agent(role="Researcher", goal="Find information", backstory="...")
writer = Agent(role="Writer", goal="Write content", backstory="...")

research_task = Task(description="Research topic X", agent=researcher, expected_output="...")
write_task = Task(description="Write article", agent=writer, expected_output="...")

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)
crew.kickoff()
```

**Strengths:** Role-based team simulations, structured coordination, clean collaboration
**Weaknesses:** Limited flexibility, difficult debugging

#### LangGraph

Directed graph where agents are **nodes** and transitions are **edges** — full control over information flow.

- Direct access to inputs and messages sent to each LLM
- Conditional routing via router functions
- More verbose code but enables complex interactions

**Strengths:** Fine-grained control, better debugging, advanced memory, error recovery, part of LangChain ecosystem
**Weaknesses:** More verbose code than other frameworks

#### AutoGen

**Dialogue-driven** — agents communicate via structured conversations with defined personas.

```python
# AutoGen study assistant — three agents
student_agent = ConversableAgent(name="Student", system_message="You ask questions.", llm_config=llm_config)
concept_agent = ConversableAgent(name="Analyst", system_message="You analyze concepts.", llm_config=llm_config)
tips_agent = ConversableAgent(name="Advisor", system_message="You give study tips.", llm_config=llm_config)

group_chat = GroupChat(agents=[student_agent, concept_agent, tips_agent],
                       max_round=6,
                       speaker_selection_method="round_robin")
manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)
```

**Strengths:** Intuitive chat interface, built-in code execution, great for human-in-the-loop
**Weaknesses:** Less suited for highly structured, non-conversational workflows

#### BeeAI

Modular multi-agent workflows with **tool integration**, supporting both sequential and parallel execution.

```python
# BeeAI multi-agent report system
researcher = Agent(goal="Research history", tools=[wikipedia_tool])
forecaster = Agent(goal="Get weather", tools=[weather_tool])
synthesizer = Agent(goal="Combine and summarize outputs")

workflow = Workflow(agents=[researcher, forecaster, synthesizer])
result = workflow.run(location="Tokyo")
```

**Strengths:** Integrates Ollama, OpenAI, WatsonX.ai, Grok; LangChain tool plugins via MCP; memory, structured outputs, state persistence; emitters for workflow tracking; clean exception handling

---

## Design Pattern 1 — Sequential (Prompt Chaining)

The simplest pattern — one LLM's output becomes the next LLM's input.

**Why it works:** Agents handle specialized tasks better; breaks complex problems into manageable steps.

```
Input → Agent 1 (specialist) → Output 1 → Agent 2 (specialist) → Final Output
```

### Cover Letter Generator Example

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

def generate_resume_summary(state: ChainState) -> ChainState:
    prompt = f"Summarize key qualifications for this job: {state['job_description']}"
    result = llm.invoke(prompt)
    return {**state, "resume_summary": result.content}

def generate_cover_letter(state: ChainState) -> ChainState:
    prompt = f"Write a cover letter using:\nJob: {state['job_description']}\nSummary: {state['resume_summary']}"
    result = llm.invoke(prompt)
    return {**state, "cover_letter": result.content}

# Build graph
workflow = StateGraph(ChainState)
workflow.add_node("resume_summary", generate_resume_summary)
workflow.add_node("cover_letter", generate_cover_letter)
workflow.add_edge("resume_summary", "cover_letter")
workflow.add_edge("cover_letter", END)
workflow.set_entry_point("resume_summary")
app = workflow.compile()
```

**State progression:**

| Field | After Agent 1 | After Agent 2 |
|---|---|---|
| `job_description` | "Software Engineer..." | "Software Engineer..." |
| `resume_summary` | "5 years Python, ML..." | "5 years Python, ML..." |
| `cover_letter` | — | "Dear Hiring Manager..." |

---

## Design Pattern 2 — Routing

A **router agent** analyzes the input, determines the task type, and directs execution to the correct specialist agent.

```
Input → Router (classify intent) → [Summarize Agent OR Translate Agent] → Output
```

### Implementation

```python
from pydantic import BaseModel
from typing import Literal

class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class RouteDecision(BaseModel):
    task_type: Literal["summarize", "translate"]

# Router node — uses structured output to classify intent
def router_node(state: RouterState) -> RouterState:
    structured_llm = llm.with_structured_output(RouteDecision)
    prompt = f"Should this be summarized or translated? Input: {state['user_input']}"
    result = structured_llm.invoke(prompt)
    return {**state, "task_type": result.task_type}

# Conditional edge function
def route_decision(state: RouterState) -> str:
    return state["task_type"]  # returns "summarize" or "translate"

def summarize_node(state: RouterState) -> RouterState:
    result = llm.invoke(f"Summarize: {state['user_input']}")
    return {**state, "output": result.content}

def translate_node(state: RouterState) -> RouterState:
    result = llm.invoke(f"Translate to English: {state['user_input']}")
    return {**state, "output": result.content}

# Build graph
workflow = StateGraph(RouterState)
workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_decision,
    {"summarize": "summarize", "translate": "translate"})
workflow.add_edge("summarize", END)
workflow.add_edge("translate", END)
app = workflow.compile()
```

---

## Design Pattern 3 — Parallelization

Multiple independent LLM tasks run **simultaneously**, then an aggregator merges the results.

```
           ┌── French Agent ──┐
Input ─────┼── Spanish Agent ─┼──► Aggregator → Combined Output
           └── Japanese Agent ┘
```

> **Analogy:** A kitchen with multiple chefs — chopping vegetables, boiling pasta, baking bread — while the head chef combines everything into a meal.

### Multi-Language Translation Example

```python
class TranslationState(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

def translate_french(state: TranslationState) -> TranslationState:
    result = llm.invoke(f"Translate to French: {state['text']}")
    return {**state, "french": result.content}

def translate_spanish(state: TranslationState) -> TranslationState:
    result = llm.invoke(f"Translate to Spanish: {state['text']}")
    return {**state, "spanish": result.content}

def translate_japanese(state: TranslationState) -> TranslationState:
    result = llm.invoke(f"Translate to Japanese: {state['text']}")
    return {**state, "japanese": result.content}

def aggregator(state: TranslationState) -> TranslationState:
    combined = f"French: {state['french']}\nSpanish: {state['spanish']}\nJapanese: {state['japanese']}"
    return {**state, "combined_output": combined}

# Build parallel graph
workflow = StateGraph(TranslationState)
workflow.add_node("french", translate_french)
workflow.add_node("spanish", translate_spanish)
workflow.add_node("japanese", translate_japanese)
workflow.add_node("aggregator", aggregator)

# All three run from START in parallel
workflow.set_entry_point("french")  # LangGraph runs all START-connected nodes in parallel
workflow.add_edge(START, "french")
workflow.add_edge(START, "spanish")
workflow.add_edge(START, "japanese")

# All feed into aggregator
workflow.add_edge("french", "aggregator")
workflow.add_edge("spanish", "aggregator")
workflow.add_edge("japanese", "aggregator")
workflow.add_edge("aggregator", END)
app = workflow.compile()
```

---

## Design Pattern 4 — Orchestrator

The most dynamic pattern — the orchestrator **doesn't know in advance** how many workers are needed. It analyzes the request at runtime, spins up the appropriate number of parallel workers, and a synthesizer merges all outputs.

```
Input → Orchestrator (plans tasks) → Assign Workers (uses Send)
    → [Worker 1 | Worker 2 | ... | Worker N] (parallel)
    → Synthesizer → Final Output
```

> **Key difference from parallelization:** The number of workers is **dynamic** — determined at runtime based on complexity. Uses LangGraph's `Send` to dispatch tasks to workers.

### Two State Containers

| State | Purpose |
|---|---|
| **Shared State** | Global workflow context — accessible by all nodes |
| **Worker State** | Task-specific details for each individual worker |

Workers read from worker state but can append results to the shared state via `operator.add`.

### Meal Planning Orchestrator Example

```python
import operator
from typing import TypedDict, Annotated, List
from pydantic import BaseModel
from langgraph.types import Send

class Dish(BaseModel):
    name: str
    ingredients: str
    location: str  # cuisine type

class OrchestratorState(TypedDict):
    meals: str                                              # user input
    sections: List[Dish]                                    # orchestrator output
    completed_menu: Annotated[List[str], operator.add]      # parallel worker outputs (auto-appended)
    final_meal_guide: str                                   # synthesizer output

class WorkerState(TypedDict):
    section: Dish                   # task-specific dish
    completed_menu: List[str]       # shared reference

def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    # LLM breaks meal request into structured dish objects
    result = planner_pipe.invoke({"meals": state["meals"]})
    return {**state, "sections": result.dishes}

def assign_workers(state: OrchestratorState):
    # Send each dish to a separate worker in parallel
    return [Send("chef_worker", {"section": dish, "completed_menu": []})
            for dish in state["sections"]]

def chef_worker(worker_state: WorkerState) -> dict:
    dish = worker_state["section"]
    prompt = f"You are a {dish.location} chef. Create a detailed recipe for {dish.name} using {dish.ingredients}."
    result = chef_pipe.invoke(prompt)
    return {"completed_menu": [result.content]}  # operator.add appends to shared list

def synthesizer_node(state: OrchestratorState) -> OrchestratorState:
    combined = "\n---\n".join(state["completed_menu"])
    return {**state, "final_meal_guide": combined}

# Build graph
workflow = StateGraph(OrchestratorState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("chef_worker", chef_worker)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges("orchestrator", assign_workers, ["chef_worker"])
workflow.add_edge("chef_worker", "synthesizer")
workflow.add_edge("synthesizer", END)
app = workflow.compile()
```

**`operator.add`** — automatically appends each worker's output to the shared `completed_menu` list without race conditions.

---

## Design Pattern 5 — Evaluator-Optimizer

The LLM generates an output, an evaluator critiques it, feedback is passed back to the generator, and the loop continues until the output meets the target criteria or the iteration limit is reached.

```
Input → Generator → Evaluator → [Accepted → END]
                         ↓
                    [Rejected → feedback → Generator → ...]
```

> The key difference from Reflexion: the **target criteria is defined upfront** (e.g. a specific risk grade), and the loop stops precisely when that criteria is met.

### Investment Advisor Example

**Personas:**
- **Kathy Wood** — generator, high-risk innovation strategies (initial plan)
- **Ray Dalio** — revised generator, refines based on Warren Buffett's feedback
- **Warren Buffett** — evaluator, conservative value-investing assessment

```python
class InvestmentState(TypedDict):
    investor_profile: str
    investment_plan: str
    target_grade: str       # e.g. "moderate"
    current_grade: str
    feedback: str
    iteration: int

def grade_risk(state: InvestmentState) -> InvestmentState:
    """Determine target risk grade from investor profile."""
    result = grade_pipe.invoke({"profile": state["investor_profile"]})
    return {**state, "target_grade": result}

def generator_node(state: InvestmentState) -> InvestmentState:
    """Kathy Wood for initial plan, Ray Dalio for revisions."""
    if state.get("feedback"):
        result = ray_dalio_pipe.invoke({
            "profile": state["investor_profile"],
            "feedback": state["feedback"],
            "grade": state["current_grade"]
        })
    else:
        result = kathy_wood_pipe.invoke({"profile": state["investor_profile"]})
    return {**state, "investment_plan": result.content}

def evaluator_node(state: InvestmentState) -> InvestmentState:
    """Warren Buffett evaluates the plan."""
    result = buffett_evaluator_pipe.invoke({
        "plan": state["investment_plan"],
        "profile": state["investor_profile"]
    })
    return {**state,
            "current_grade": result.grade,
            "feedback": result.feedback,
            "iteration": state["iteration"] + 1}

def route_investment(state: InvestmentState) -> str:
    """Accept if grades match or iteration limit reached."""
    if state["current_grade"] == state["target_grade"] or state["iteration"] >= 5:
        return "accepted"
    return "rejected"

# Build graph
workflow = StateGraph(InvestmentState)
workflow.add_node("grade_risk", grade_risk)
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("grade_risk")
workflow.add_edge("grade_risk", "generator")
workflow.add_edge("generator", "evaluator")
workflow.add_conditional_edges("evaluator", route_investment,
    {"accepted": END, "rejected": "generator"})
app = workflow.compile()
```

**State variables tracked:**

| Variable | Purpose |
|---|---|
| `investor_profile` | User input — risk preferences and goals |
| `target_grade` | Desired risk level (set by grading node) |
| `current_grade` | Evaluator's current assessment |
| `investment_plan` | Generated strategy (updated each iteration) |
| `feedback` | Warren Buffett's critique for refinement |
| `iteration` | Counter — prevents infinite loops |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Agentic AI** | Autonomous systems that reason, plan, act, and use tools toward a goal |
| **Multi-agent benefits** | Specialization, parallelism, fault tolerance, scalability, modularity |
| **Why frameworks** | Handle memory, coordination, state, error handling — so you focus on the problem |
| **CrewAI** | Role-based crews — agents with roles, goals, backstories executing tasks |
| **LangGraph** | Graph-based workflows — fine-grained control, best for complex structured apps |
| **AutoGen** | Dialogue-driven — group chats, round-robin turns, human-in-the-loop |
| **BeeAI** | Enterprise-grade — modular, scalable, MCP integration, production-ready |
| **Sequential pattern** | Chain LLM outputs — each agent's output is the next agent's input |
| **Routing pattern** | Router classifies intent → directs to correct specialist agent |
| **Parallelization** | Multiple agents run simultaneously → aggregator merges outputs |
| **Orchestrator pattern** | Dynamic worker dispatch via `Send` — number of workers determined at runtime |
| **Worker State** | Task-specific container for each worker, separate from shared state |
| **`operator.add`** | Automatically appends worker outputs to shared list without race conditions |
| **Evaluator-Optimizer** | Generate → evaluate → feedback loop until target criteria met or iteration limit |
| **Key difference** | Orchestrator = dynamic workers; Evaluator-Optimizer = iterative refinement toward a target |

---

*Notes based on: Course 08 Module 01 — Agentic Frameworks and LangGraph Design Patterns (Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI)*
