# Course 07 — Module 02: Build Self-Improving Agents with LangGraph

> A complete reference covering the 5 types of AI agents, reflection agents, Reflexion agents, Pydantic schemas for tool calls, and ReAct agents.

---

## Table of Contents

1. [The 5 Types of AI Agents](#the-5-types-of-ai-agents)
2. [Reflection Agents](#reflection-agents)
3. [Pydantic & JSON Serialization for Tool Calls](#pydantic--json-serialization-for-tool-calls)
4. [Reflexion Agents](#reflexion-agents)
5. [Building a Reflexion Agent](#building-a-reflexion-agent)
6. [ReAct Agents](#react-agents)
7. [Summary](#summary)

---

## The 5 Types of AI Agents

Agents are classified by their level of intelligence, decision-making process, and how they interact with their environment.

### 1. Simple Reflex Agent

Follows **predefined condition-action rules** — no memory, no planning.

```
Environment → Sensors → Percepts → Condition-Action Rules → Actuators → Action → Environment
```

**Example:** Thermostat — `if temperature < 18°C then turn on heat`

| | |
|---|---|
| **Strengths** | Fast, effective in structured, predictable environments |
| **Weaknesses** | No memory, can repeatedly make same mistakes, fails in dynamic scenarios |

---

### 2. Model-Based Reflex Agent

Adds an **internal model of the world** (state component) — remembers where it's been and tracks how its own actions affect the environment.

**Example:** Robotic vacuum — remembers cleaned areas, knows that moving forward changes location.

```
Percepts → [How world evolves + What my actions do] → State → Condition-Action Rules → Action
```

| | |
|---|---|
| **Strengths** | Infers and remembers parts of environment it can't currently observe |
| **Weaknesses** | Still reactive — doesn't plan ahead |

---

### 3. Goal-Based Agent

Replaces condition-action rules with **goals** — simulates future outcomes of actions and asks "will this help me reach my goal?"

**Example:** Self-driving car with destination X:
```
State: "I'm on Main Street"
Prediction: "If I turn left, I head toward the highway"
Goal check: "Does this help me reach X?" → Yes → Turn left
```

| | |
|---|---|
| **Strengths** | Plans ahead, adapts to environment |
| **Weaknesses** | Any way of meeting the goal will do — doesn't evaluate quality of outcome |

---

### 4. Utility-Based Agent

Considers not just **if** a goal is met, but **how desirable** different outcomes are — assigns a utility (happiness/preference) score to each possible future state.

**Example:** Autonomous drone delivery
- Goal-based: "Deliver to address X" — picks any valid route
- Utility-based: "Deliver quickly, safely, with minimum energy" — simulates multiple paths, estimates duration/battery/weather, picks highest utility route

| | |
|---|---|
| **Strengths** | Ranks options, not just picks anything that satisfies the goal |
| **Weaknesses** | Requires an accurate utility function |

---

### 5. Learning Agent

The most adaptable and powerful — **learns from experience**, improves performance over time through feedback.

**Components:**

| Component | Role |
|---|---|
| **Critic** | Observes action outcomes, compares to performance standard, generates reward signal |
| **Learning element** | Updates agent's knowledge using critic's feedback — improves state-to-action mapping |
| **Problem generator** | Suggests new unexplored actions — "try a different path" |
| **Performance element** | Selects actions based on what learning element has determined to be optimal |

**Example:** AI chess bot
- Performance element plays using current strategy
- Critic observes a loss
- Learning element adjusts strategy across thousands of games
- Problem generator suggests new unexplored moves

| | |
|---|---|
| **Strengths** | Improves over time, handles novel situations |
| **Weaknesses** | Slowest, most data-intensive |

### Agent Type Comparison

| Agent Type | Key Ability | Memory | Planning |
|---|---|---|---|
| Simple Reflex | Reacts | None | None |
| Model-Based Reflex | Remembers | State tracking | None |
| Goal-Based | Aims | State + goals | Goal-directed |
| Utility-Based | Evaluates | State + utility | Optimal outcome |
| Learning | Improves | All of the above | Adaptive |

---

## Reflection Agents

**Reflection agents** iteratively improve AI outputs by critically analyzing their own performance through a feedback loop.

### Three Types

| Type | Description |
|---|---|
| **Basic Reflection Agent** | Generator + Reflector loop |
| **Reflexion Agent** | Adds external tools, citations, verifiable claims |
| **LATS** (Language Agent Tree Search) | Tree-based search over possible actions |

### Basic Reflection Agent — How It Works

```
User Query
    |
Generator → Initial response ("wear a fedora")
    |
Reflector → Critique ("fedoras are outdated")
    |
Generator → Improved response ("wear well-fitted clothes, good posture")
    |
Reflector → Critique ("good, could add personal style")
    |
Generator → Final response ("find clothes that match your personal style, be confident")
```

- The loop runs for a **set number of steps**
- Both generator and reflector accumulate access to previous outputs — **building memory** over iterations
- The reflector's critique is wrapped as a **HumanMessage** (not AIMessage) so the generator receives it as user input — maintaining the correct feedback loop

### Building a LinkedIn Post Reflection Agent

#### Setup — LLM and Chains

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import WatsonxLLM

llm = WatsonxLLM(...)  # IBM Granite model

# Generator chain
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a LinkedIn content expert. Generate engaging posts."),
    MessagesPlaceholder(variable_name="messages")  # memory across iterations
])
generate_chain = generate_prompt | llm

# Reflector chain
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional LinkedIn content strategist. Critique the following post."),
    MessagesPlaceholder(variable_name="messages")
])
reflect_chain = reflect_prompt | llm
```

#### State — MessageGraph

```python
from langgraph.graph import MessageGraph

# MessageGraph: specialized StateGraph whose state holds only an array of messages
# [HumanMessage, AIMessage, SystemMessage, ToolMessage, ...]
graph = MessageGraph()
```

> **MessageGraph** is a specialized `StateGraph` that accumulates messages. Each turn adds a HumanMessage followed by an AIMessage. LangGraph handles the merging automatically.

#### Nodes

```python
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Sequence

def generation_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    response = generate_chain.invoke({"messages": messages})
    return [AIMessage(content=response)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    critique = reflect_chain.invoke({"messages": messages})
    # Wrapped as HumanMessage — so generator receives it as user input
    return [HumanMessage(content=critique)]
```

#### Graph Construction

```python
from langgraph.graph import END

def should_continue(messages):
    """Stop after 6 messages (3 generate + 3 reflect cycles)."""
    return len(messages) > 6

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)

graph.set_entry_point("generate")
graph.add_edge("reflect", "generate")  # reflection feeds back to generator

graph.add_conditional_edges(
    "generate",
    should_continue,
    {True: END, False: "reflect"}
)

app = graph.compile()
```

#### Run

```python
result = app.invoke([HumanMessage(
    content="Write a LinkedIn post on getting a software developer job at IBM under 160 characters."
)])
# result[-1] = final refined LinkedIn post
```

---

## Pydantic & JSON Serialization for Tool Calls

**Pydantic models** ensure LLM outputs are structured, validated, and JSON-serializable — essential for feeding outputs into APIs, databases, or other agents.

### Why Pydantic?

| Feature | Benefit |
|---|---|
| **Type validation** | Ensures inputs/outputs conform to expected schema |
| **JSON serialization** | `.json()` and `.parse_raw()` simplify tool chaining |
| **Reusability** | Base classes shared across multiple tools |
| **LangChain integration** | Works natively with `bind_tools()` and `with_structured_output()` |

### Basic Schema Example

```python
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition: sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit: fahrenheit or celsius")

# Bind as tool to LLM
weather_llm = llm.bind_tools(tools=[WeatherSchema])
response = weather_llm.invoke("It's sunny and 75 degrees")
# Returns: {"condition": "sunny", "temperature": 75, "unit": "fahrenheit"}
```

### Using Literal to Restrict Values

```python
from typing import Literal

class CalculatorSchema(BaseModel):
    operation: Literal['add', 'subtract', 'multiply', 'divide'] = Field(
        description="Mathematical operation to perform"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
```

> `Literal` restricts a field to specific constant values — prevents invalid operations from reaching your system.

### Reusable Math Tool Schemas

```python
from pydantic import BaseModel
from typing import Literal

class TwoOperands(BaseModel):
    a: float
    b: float

class AddInput(TwoOperands):
    operation: Literal['add']

class SubtractInput(TwoOperands):
    operation: Literal['subtract']

class MathOutput(BaseModel):
    result: float

def add_tool(data: AddInput) -> MathOutput:
    return MathOutput(result=data.a + data.b)

def subtract_tool(data: SubtractInput) -> MathOutput:
    return MathOutput(result=data.a - data.b)

# Dynamic dispatch from JSON
def dispatch_tool(json_payload: str) -> str:
    base = SubtractInput.parse_raw(json_payload)
    if base.operation == "subtract":
        output = subtract_tool(SubtractInput.parse_raw(json_payload))
    return output.json()  # {"result": 4.0}
```

---

## Reflexion Agents

**Reflexion agents** extend basic reflection agents by:
- Using **external tools** (web search, APIs) to incorporate real-time information
- Producing responses with **citations** and **verifiable claims** — not just improved opinions
- **Self-identifying weaknesses** — flagging missing and superfluous information

### Reflexion vs Reflection

| Feature | Reflection Agent | Reflexion Agent |
|---|---|---|
| Self-critique | Yes | Yes |
| External tools | No | Yes (web search, APIs) |
| Citations | No | Yes |
| Verifiable claims | No | Yes |
| Post-training learning | No | Yes — can incorporate new information |

### Reflexion Workflow

```
User Query
    |
Responder (Generator LLM)
  → outputs structured schema: {answer, reflection, search_queries}
    |
Search Tool (Tavily)
  → returns {title, URL, content} for each query
  → result appended to response_list
    |
Revisor LLM
  → reads self-critique from responder
  → modifies response using tool outputs
  → outputs: {revised_answer, references, self_critique, next_search_queries}
    |
Search Tool again → Revisor again → ... (iterate N times)
    |
Final revised answer with citations
```

### Schema Design

```python
from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="The generated answer")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Search queries to improve the response")

class ReviseAnswer(AnswerQuestion):
    """Extends AnswerQuestion with citations."""
    references: List[str] = Field(description="Citations for the revised response")
```

---

## Building a Reflexion Agent

### Setup

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Search tool — up to 5 results per query
search_tool = TavilySearchResults(max_results=5)

# LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-nano")
```

### Responder Chain

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

responder_system = """You are Dr. Paul Saladino, specializing in controversial health approaches.
Respond in 250 words."""

responder_prompt = ChatPromptTemplate.from_messages([
    ("system", responder_system),
    MessagesPlaceholder(variable_name="messages")
])

# Bind AnswerQuestion schema as tool
responder_chain = responder_prompt | llm.bind_tools([AnswerQuestion])
```

### Tool Execution Node

```python
from langchain_core.messages import ToolMessage

def execute_tools(state):
    """Extract search queries from last AI message, run search, append results."""
    messages = state
    ai_message = messages[-1]
    tool_calls = ai_message.tool_calls[0]["args"]
    search_queries = tool_calls["search_queries"]

    tool_messages = []
    for query in search_queries:
        results = search_tool.invoke(query)
        tool_messages.append(ToolMessage(
            content=str(results),
            tool_call_id=ai_message.tool_calls[0]["id"]
        ))
    return messages + tool_messages
```

### Revisor Chain

```python
revisor_system = """You are Dr. Peter Attia, expert in longevity and evidence-based health.
Revise the response using the search results. Add citations."""

revisor_prompt = ChatPromptTemplate.from_messages([
    ("system", revisor_system),
    MessagesPlaceholder(variable_name="messages")
])

revisor_chain = revisor_prompt | llm.bind_tools([ReviseAnswer])
```

### Graph Construction

```python
from langgraph.graph import MessageGraph, END

MAX_ITERATIONS = 4

def event_loop(state):
    """Count tool messages as proxy for iterations."""
    tool_count = sum(1 for m in state if isinstance(m, ToolMessage))
    return END if tool_count >= MAX_ITERATIONS else "execute_tools"

graph = MessageGraph()
graph.add_node("respond", lambda state: [responder_chain.invoke({"messages": state})])
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", lambda state: [revisor_chain.invoke({"messages": state})])

graph.set_entry_point("respond")
graph.add_edge("respond", "execute_tools")
graph.add_edge("execute_tools", "revisor")
graph.add_conditional_edges("revisor", event_loop, {END: END, "execute_tools": "execute_tools"})

app = graph.compile()
```

### Run and Extract Results

```python
result = app.invoke([HumanMessage(
    content="I'm pre-diabetic and need to lower my blood sugar, and I have heart issues."
)])

# Initial response: general recommendations without evidence
initial_answer = result[1].tool_calls[0]["args"]["answer"]

# Final revised response: specific, citation-backed, evidence-based
final_answer = result[-1].tool_calls[0]["args"]["answer"]
final_references = result[-1].tool_calls[0]["args"]["references"]
```

---

## ReAct Agents

**ReAct** (Reasoning + Acting) agents perform **step-by-step reasoning** and use tools to answer complex queries. Every response follows a structured format:

```
Thought → Action → Action Input → Observation → (repeat) → Final Answer
```

### ReAct Flow Example

**Query:** "What's the weather in Tokyo and what should I wear?"

```
Thought: I need to look up the weather in Tokyo first.
Action: search_tool
Action Input: "Tokyo weather today"
Observation: Tokyo is 22°C with sunny skies.

Thought: Now I should recommend clothing for 22°C sunny weather.
Action: recommend_clothing
Action Input: "22°C sunny weather"
Observation: Light clothing recommended: t-shirt, shorts, sunglasses.

Final Answer: Tokyo is 22°C and sunny today. Wear light clothing: t-shirt, shorts, sunglasses.
```

### Implementation in LangGraph

#### State and Tools

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def search(query: str) -> str:
    """Search for current weather information."""
    return TavilySearchResults(max_results=3).invoke(query)

@tool
def recommend_clothing(weather: str) -> str:
    """Suggest clothing based on weather description."""
    if "rain" in weather.lower() or "wet" in weather.lower():
        return "Bring a waterproof jacket and umbrella."
    return "Light clothing recommended."

tools = [search, recommend_clothing]
tool_map = {t.name: t for t in tools}
tool_node = ToolNode(tools)
```

#### Agent and Graph

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4.1-nano")
llm_with_tools = llm.bind_tools(tools)

system_message = """You are a helpful AI assistant that thinks step-by-step and uses tools when needed.
Format: Thought → Action → Action Input → Observation → Final Answer"""

def call_model(state: AgentState):
    messages = [("system", system_message)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tools"

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

app = graph.compile()
```

#### Run

```python
result = app.invoke({
    "messages": [HumanMessage(content="What's the weather in Zurich? What should I wear?")]
})
```

**Execution trace:**
```
HumanMessage → agent (tool call: search "Zurich weather")
→ tools (search result returned)
→ agent (tool call: recommend_clothing "15°C rainy")
→ tools (clothing result: "waterproof jacket")
→ agent (Final Answer: "It's 15°C and rainy in Zurich. Bring a waterproof jacket.")
→ END (no further tool calls)
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Simple Reflex Agent** | Condition-action rules only — no memory, no planning |
| **Model-Based Reflex** | Adds internal world model and state tracking |
| **Goal-Based Agent** | Plans actions toward a defined goal |
| **Utility-Based Agent** | Picks the best outcome by evaluating utility scores |
| **Learning Agent** | Critic + learning element + problem generator = improves over time |
| **Reflection Agent** | Generator + Reflector loop — iteratively refines output |
| **MessageGraph** | Specialized StateGraph — state is a list of accumulated messages |
| **Reflection trick** | Wrap critique as `HumanMessage` so generator receives it as user input |
| **Reflexion Agent** | Reflection + external tools + citations + verifiable claims |
| **Reflexion schema** | `AnswerQuestion` (answer, reflection, search_queries) + `ReviseAnswer` (+ references) |
| **Pydantic** | Enforces structured, validated, JSON-serializable LLM outputs |
| **`Literal` type** | Restricts field to specific constant values |
| **ReAct** | Thought → Action → Action Input → Observation → Final Answer loop |
| **ReAct termination** | Loop ends when last message has no tool calls → routes to END |
| **`add_messages`** | LangGraph annotation that appends new messages to state list |
| **`ToolNode`** | Pre-built LangGraph node that executes tool calls from AIMessage |

---

*Notes based on: Course 07 Module 02 — Build Self-Improving Agents with LangGraph (Agentic AI with LangChain and LangGraph)*
