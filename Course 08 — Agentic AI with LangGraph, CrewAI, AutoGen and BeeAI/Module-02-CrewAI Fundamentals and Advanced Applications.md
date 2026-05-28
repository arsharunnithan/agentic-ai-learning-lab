# Course 08 — Module 02: CrewAI Fundamentals and Advanced Applications

> A complete reference covering CrewAI core components, structured outputs with Pydantic, YAML configuration, CrewBase, and custom tools.

---

## Table of Contents

1. [CrewAI Core Concepts](#crewai-core-concepts)
2. [Building a CrewAI Pipeline](#building-a-crewai-pipeline)
3. [CrewOutput Object](#crewoutput-object)
4. [Structured Outputs with Pydantic](#structured-outputs-with-pydantic)
5. [YAML Configuration & CrewBase](#yaml-configuration--crewbase)
6. [Advanced Example — Meal Planning System](#advanced-example--meal-planning-system)
7. [Custom Tools in CrewAI](#custom-tools-in-crewai)
8. [Agent-Centric vs Task-Centric Tool Assignment](#agent-centric-vs-task-centric-tool-assignment)
9. [Summary](#summary)

---

## CrewAI Core Concepts

CrewAI is built around **four core concepts**: Task, Agent, Tool, and Flow.

### Task

A task is like a **director** — it has a clear vision of what needs to be accomplished.

```python
from crewai import Task

research_task = Task(
    description="Analyze the latest generative AI breakthroughs for {topic}.",
    expected_output="A detailed, insight-rich summary of recent trends.",
    agent=research_analyst  # agent assigned to this task
)
```

| Parameter | Purpose |
|---|---|
| `description` | What the agent should do — can include `{variable}` placeholders |
| `expected_output` | Defines success — specific requirements for the output |
| `agent` | Which agent handles this task |

---

### Agent

An agent is an **LLM-powered AI** guided by structured prompts that give it personality and expertise — like an actor who has skills and a role but needs direction.

```python
from crewai import Agent

research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge insights in AI and technology.",
    backstory="An expert analyst who synthesizes information from diverse sources.",
    llm=llm,
    tools=[SerperDevTool()],
    verbose=True,
    allow_delegation=False
)
```

| Parameter | Purpose |
|---|---|
| `role` | What kind of expert the agent is |
| `goal` | The objective guiding the agent's decisions |
| `backstory` | Context and behavioral instructions |
| `tools` | External capabilities the agent can use |
| `allow_delegation` | Whether the agent can delegate tasks to other agents |

---

### Tool

Tools are **functional components** that agents or tasks use to perform specific actions — like a camera for the director or a car for the actor.

- Can be assigned to **agents** (agent-centric) or **tasks** (task-centric)
- Built-in tools: `SerperDevTool` (web search), `PDFSearchTool` (RAG on PDFs)
- Custom tools defined with the `@tool` decorator

---

### Flow

Flows define **how tasks run and how agents interact**, set via the `process` parameter in the Crew object.

| Flow Type | Description |
|---|---|
| **Sequential** | Tasks run one after another — output of one becomes input for the next |
| **Hierarchical** | A manager agent dynamically assigns and oversees tasks — used when autonomy and flexibility are needed |

> Note: Sequential flow in CrewAI is similar to a **reflection pattern** (with feedback between agents), not just simple prompt chaining.

---

## Building a CrewAI Pipeline

### Full Example — Research + Writing Pipeline

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Step 1: Initialize shared LLM
llm = LLM(model="watsonx/meta-llama/llama-3-70b-instruct")

# Step 2: Define agents
research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge insights.",
    backstory="Expert analyst with years of experience synthesizing complex information.",
    llm=llm,
    tools=[SerperDevTool()],
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft well-structured and engaging content based on research findings.",
    backstory="Translates complex topics into simple language for wide audiences.",
    llm=llm,
    verbose=True
)

# Step 3: Define tasks
research_task = Task(
    description="Analyze the latest AI developments for topic: {topic}.",
    expected_output="A detailed, insight-rich summary.",
    agent=research_analyst
)

write_task = Task(
    description="Using the research findings, write an engaging blog post.",
    expected_output="An engaging four-paragraph blog post.",
    agent=writer
)

# Step 4: Assemble the crew
crew = Crew(
    agents=[research_analyst, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# Step 5: Run
result = crew.kickoff(inputs={"topic": "generative AI breakthroughs"})
```

**Execution flow:**
```
Crew initialized
    ↓
Research Analyst begins → uses SerperDevTool → synthesizes insights → task complete
    ↓
Tech Content Strategist receives research output → writes blog post → task complete
    ↓
Crew wraps up → CrewOutput returned
```

---

## CrewOutput Object

After `.kickoff()`, results are bundled into a **CrewOutput** object:

| Field | Description |
|---|---|
| `result.raw` | Unified output from all agents and tasks combined |
| `result.tasks_output` | List of individual task results |
| `result.token_usage` | Prompt, completion, and total tokens — for performance and cost tracking |
| `result.pydantic` | Pydantic object (if `output_pydantic` was set on the task) |
| `result.json_dict` | Dict output (if `output_json` was set on the task) |

```python
print(result.raw)                    # full combined output
print(result.tasks_output[0])        # research analyst's output
print(result.tasks_output[1])        # writer's output
print(result.token_usage)            # cost and performance data
```

---

## Structured Outputs with Pydantic

**Why structured outputs matter:** Free-form LLM text is hard to parse, prone to ambiguity, and risky for downstream tasks. Pydantic models enforce a schema, ensuring consistency and clean agent-to-agent handoffs.

### Pydantic Key Features

| Feature | Description |
|---|---|
| **Data validation** | Ensures inputs match expected types — raises `ValidationError` if not |
| **Automatic type conversion** | Coerces inputs into expected types when possible |
| **Nested models** | Models can include other models, lists, and optional fields |
| **Easy serialization** | `.dict()` and `.json()` for storage or API responses |

### Defining Models

```python
from pydantic import BaseModel
from typing import List

class GroceryItem(BaseModel):
    name: str
    quantity: str
    estimated_price: float
    store_category: str

class MealPlan(BaseModel):
    meal_name: str
    cooking_difficulty: str
    servings: int
    ingredients: List[GroceryItem]

class ShoppingCategory(BaseModel):
    section_name: str
    items: List[GroceryItem]
    total_estimated_cost: float

class GroceryShoppingPlan(BaseModel):
    total_budget: float
    meal_plans: List[MealPlan]
    shopping_sections: List[ShoppingCategory]
    shopping_tips: str
```

**Class hierarchy:**
```
GroceryShoppingPlan
    ├── MealPlan (list)
    │     └── GroceryItem (list of ingredients)
    └── ShoppingCategory (list)
          └── GroceryItem (list of items)
```

### Attaching Pydantic Models to Tasks

```python
meal_planning_task = Task(
    description="Find recipes matching {budget} budget and {dietary_needs} needs.",
    expected_output="A structured meal plan with ingredients.",
    agent=meal_planner,
    output_pydantic=MealPlan,         # enforces structured output as Pydantic object
    output_file="meal_plan.json"      # saves output to file
)

# Alternative — output as plain dict
shopping_task = Task(
    ...,
    output_json=GroceryShoppingPlan   # returns Python dict instead of Pydantic object
)
```

### Accessing Structured Results

```python
result = crew.kickoff(inputs={...})

# Pydantic object access
meal = result.pydantic
print(meal.meal_name)
print(meal.ingredients[0].name)

# Dict access
plan = result.json_dict
print(plan.get("total_budget"))

# Dictionary-style access (supported via __getitem__)
print(result["meal_name"])
```

### Benefits in CrewAI Workflows

| Benefit | How It Helps |
|---|---|
| **Type safety** | Validates LLM output against schema — prevents silent bugs |
| **Clear data contracts** | Formal agreement on what each task returns — easier to maintain |
| **System integration** | Push output to APIs, databases, or UIs without parsing |
| **Less post-processing** | No custom regex or parsing logic needed |
| **Consistent agent handoffs** | One agent's output cleanly becomes the next agent's input |
| **LLM behavior alignment** | Schema acts as soft prompting — constrains model output space |

---

## YAML Configuration & CrewBase

YAML separates **configuration from code** — easier to update agents and tasks without changing Python.

### Defining Agents in YAML

```yaml
# config/agents.yaml
leftover_manager:
  role: "Leftover Specialist"
  goal: "Transform leftover ingredients into creative meals."
  backstory: "A resourceful chef who hates food waste and loves creative cooking."
```

### Defining Tasks in YAML

```yaml
# config/tasks.yaml
leftover_task:
  description: "Use the leftover ingredients from {meal_plan} to suggest new meals."
  expected_output: "A list of 3 creative meals using leftover ingredients."
  agent: leftover_manager
```

### Using @CrewBase to Load YAML

```python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class LeftoverCrew:
    """Crew for managing leftovers — config loaded from YAML."""

    @agent
    def leftover_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['leftover_manager'],  # loads from YAML
            llm=self.llm
        )

    @task
    def leftover_task(self) -> Task:
        return Task(
            config=self.tasks_config['leftover_task'],      # loads from YAML
            output_json=LeftoverPlan                         # Pydantic defined in Python
        )
```

```python
# Instantiate and use
from leftover_crew import LeftoverCrew

leftovers_cb = LeftoverCrew(llm=shared_llm)
leftover_agent = leftovers_cb.leftover_manager()
leftover_task = leftovers_cb.leftover_task()
```

> **Key rule:** CrewBase automatically locates the `config/` folder — no manual loading needed. If using Jupyter Notebook, define the class in a separate `.py` file and import it.

### Combining YAML and Python Agents in One Crew

```python
complete_crew = Crew(
    agents=[meal_planner, shopping_organizer, budget_advisor, leftover_agent, summary_agent],
    tasks=[meal_task, shopping_task, budget_task, leftover_task, summary_task],
    process=Process.sequential
)

result = complete_crew.kickoff(inputs={"budget": "50", "dietary_needs": "vegetarian"})
```

---

## Advanced Example — Meal Planning System

Five specialized agents working sequentially:

| Agent | Role | Tool |
|---|---|---|
| **MealPlanner** | Find recipes matching budget and dietary needs | SerperDevTool |
| **ShoppingOrganizer** | Turn ingredients into organized shopping list | None |
| **BudgetAdvisor** | Ensure plan stays within budget | SerperDevTool |
| **LeftoverManager** | Suggest creative uses for leftover ingredients | (YAML-defined) |
| **SummaryAgent** | Compile everything into a full meal planning guide | None |

**Output formats by task:**
- Meal plan → `meal_plan.json` (via `output_pydantic=MealPlan`)
- Shopping list → `shopping_list.json` (via `output_pydantic=GroceryShoppingPlan`)
- Budget guide → `shopping_guide.md` (markdown format)
- Final report → comprehensive guide combining all outputs

---

## Custom Tools in CrewAI

Tools are functional components registered using the `@tool` decorator from `crewai.tools`.

```python
from crewai.tools import tool

@tool("AddNumbers")
def add_numbers(input: str) -> str:
    """Add all numbers found in the input string.
    
    Args:
        input: A string containing numbers to add (e.g. 'Add 7 and 8, also 9')
    Returns:
        The sum of all numbers found as a string.
    """
    import re
    numbers = [int(n) for n in re.findall(r'\d+', input)]
    return str(sum(numbers))

@tool("MultiplyNumbers")
def multiply_numbers(input: str) -> str:
    """Multiply all numbers found in the input string."""
    import re
    numbers = [int(n) for n in re.findall(r'\d+', input)]
    result = 1
    for n in numbers:
        result *= n
    return str(result)
```

```python
# Assign to agent
calculator_agent = Agent(
    role="Calculator",
    goal="Extract, add, or multiply numbers from text.",
    backstory="Expert at interpreting numeric instructions.",
    tools=[add_numbers, multiply_numbers],
    llm=llm
)

# Define and run task
calc_task = Task(
    description="Add all numbers from this: 'Add 7 and 8, also 9, don't forget 10'",
    expected_output="The sum of all numbers.",
    agent=calculator_agent
)

crew = Crew(agents=[calculator_agent], tasks=[calc_task], process=Process.sequential)
result = crew.kickoff()
# Agent parses text → picks add_numbers tool → returns 34
```

---

## Agent-Centric vs Task-Centric Tool Assignment

### Agent-Centric Approach

Tools are assigned to the **agent** — the agent **chooses** which tool to use based on the query.

```python
# Tools assigned to agent
inquiry_agent = Agent(
    role="Inquiry Specialist",
    goal="Answer customer questions using the best available tool.",
    backstory="Can access both the FAQ PDF and real-time web search.",
    tools=[pdf_search_tool, SerperDevTool()],  # agent decides which to use
    llm=llm
)

inquiry_task = Task(
    description="Answer this customer query: {question}",
    expected_output="A clear, helpful, well-formatted response.",
    agent=inquiry_agent
    # NO tools here — agent selects autonomously
)
```

**Flow:** Agent receives query → analyzes it → selects FAQ PDF or web search → returns answer.

---

### Task-Centric Approach

Tools are assigned to **individual tasks** — the agent follows **fixed instructions** for which tool to use at each step.

```python
# Agent has NO tools
customer_service_agent = Agent(
    role="Customer Service Specialist",
    goal="Provide support through a guided multi-step process.",
    backstory="Follows task instructions — does not choose tools independently.",
    llm=llm
    # no tools here
)

# Tools assigned to specific tasks
faq_search_task = Task(
    description="Search the FAQ PDF for information about: {question}",
    expected_output="Relevant FAQ content.",
    agent=customer_service_agent,
    tools=[pdf_search_tool]   # tool attached to THIS task
)

response_drafting_task = Task(
    description="Use the FAQ results to draft a friendly, helpful response.",
    expected_output="A warm, professional customer service reply.",
    agent=customer_service_agent
    # no tool needed here
)
```

**Flow:** Task 1 → forced to use PDF search → Task 2 → formats response from task 1 output.

---

### When to Use Each Approach

| Approach | Best For | Tradeoff |
|---|---|---|
| **Agent-centric** | Dynamic queries needing intelligent tool selection | Less predictable, harder to trace |
| **Task-centric** | Structured, sequential workflows with fixed steps | More predictable, easier to debug and audit |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Task** | Defines what to do — description, expected output, agent |
| **Agent** | LLM + role + goal + backstory + tools — the "who" |
| **Flow** | Sequential (linear) or Hierarchical (manager-driven) |
| **Crew** | Combines agents, tasks, tools, and flow into one system |
| **`.kickoff()`** | Runs the crew — pass `inputs={}` for variable substitution |
| **CrewOutput** | `raw`, `tasks_output`, `token_usage`, `pydantic`, `json_dict` |
| **Pydantic** | Enforces schema on LLM output — type validation, serialization, nesting |
| **`output_pydantic`** | Returns Pydantic object; `output_json` returns dict |
| **YAML config** | Separates agent/task definitions from Python code |
| **`@CrewBase`** | Loads YAML agents and tasks as methods — auto-locates `config/` folder |
| **`@tool` decorator** | Registers Python function as a CrewAI-compatible tool |
| **Agent-centric tools** | Agent chooses the tool — flexible but less predictable |
| **Task-centric tools** | Task dictates the tool — structured, traceable, easier to debug |
| **`allow_delegation`** | `False` = agent only handles its own tasks |
| **SerperDevTool** | Real-time web search (requires API key) |
| **PDFSearchTool** | RAG on PDF files using HuggingFace sentence transformers |

---

*Notes based on: Course 08 Module 02 — CrewAI Fundamentals and Advanced Applications (Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI)*
