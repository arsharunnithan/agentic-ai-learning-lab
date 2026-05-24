# Course 06 — Module 02: LCEL and Manual Tool Calling in LangChain

> A complete reference covering LCEL chaining, when to call tools manually, structured outputs, and building interactive LLM agents with manual tool execution.

---

## Table of Contents

1. [LCEL — LangChain Expression Language](#lcel--langchain-expression-language)
2. [Manual Tool Calling — When and Why](#manual-tool-calling--when-and-why)
3. [Structured Outputs for Tool Calling](#structured-outputs-for-tool-calling)
4. [Building LLM Agents with Tools](#building-llm-agents-with-tools)
5. [Building Interactive LLM Agents — Manual Tool Execution](#building-interactive-llm-agents--manual-tool-execution)
6. [The ToolCallingAgent Class](#the-toolcallingagent-class)
7. [Summary](#summary)

---

## LCEL — LangChain Expression Language

**LangChain Expression Language (LCEL)** is the modern, recommended pattern for building LangChain pipelines using the **pipe operator `|`** to connect components into clean, composable workflows.

### Core Steps

```
1. Define a template with {variables} in curly braces
2. Create a PromptTemplate instance
3. Build the chain using | to connect components
4. Invoke the chain with input values
```

### Runnable Primitives

| Primitive | Description | Shorthand |
|---|---|---|
| **RunnableSequence** | Chains components sequentially — output of one feeds next | `component1 \| component2` |
| **RunnableParallel** | Runs multiple components concurrently on the same input | dict `{"key": chain}` |
| **RunnableLambda** | Wraps a Python function into a runnable pipeline component | auto-coerced from function |

### Type Coercion — Automatic Conversion

LCEL automatically converts Python types into compatible Runnable components:

| Python Type | Converts To | Behavior |
|---|---|---|
| `dict` | `RunnableParallel` | Runs all values concurrently |
| `function` | `RunnableLambda` | Transforms inputs |

### Simple Sequential Chain Example

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_prompt(inputs):
    return f"Tell me a {inputs['adjective']} joke about {inputs['content']}."

joke_chain = (
    RunnableLambda(format_prompt)  # Step 1: format prompt
    | llm                          # Step 2: generate response
    | StrOutputParser()            # Step 3: parse to plain string
)

result = joke_chain.invoke({"adjective": "funny", "content": "programmers"})
```

### Parallel Chain Example

```python
parallel_chain = {
    "summary":     summary_prompt   | llm | StrOutputParser(),
    "translation": translate_prompt | llm | StrOutputParser(),
    "sentiment":   sentiment_prompt | llm | StrOutputParser()
}

# All three run concurrently on the same input
result = parallel_chain.invoke({"text": "LangChain makes AI development easy!"})
# { "summary": "...", "translation": "...", "sentiment": "Positive" }
```

### LCEL vs LangGraph

| Use Case | Tool |
|---|---|
| Simple to moderate pipelines | LCEL |
| Parallel task execution | LCEL |
| Complex stateful workflows | LangGraph (use LCEL within nodes) |

### Key LCEL Strengths

Parallel execution · Async support · Simplified streaming · Automatic tracing · Composability · Less boilerplate

---

## Manual Tool Calling — When and Why

### The Core Tension

LLMs can suggest which tools to use and what parameters to pass — but should they execute those tools automatically?

**Automatic agent execution:**
```
User prompt → LLM suggests tool → Agent executes → Result returned
(No human intervention)
```

**Manual tool invocation:**
```
User prompt → LLM suggests tool → Human reviews → Manual execution → Result returned
(Human in the loop)
```

### Why Manual Invocation Matters

| Benefit | Detail |
|---|---|
| **Safety** | Prevent unintended actions — especially critical for financial, medical, or security systems |
| **Cost control** | Avoid unnecessary API calls that inflate costs unexpectedly |
| **Accuracy** | Validate that the right tool is being called with the right parameters |
| **Oversight** | Validate inputs and outputs — ensure operations align with intent |
| **Reduced risk** | Only safe and necessary operations are performed |

> **Example:** An LLM suggests automatically updating a sensitive financial database based on its predictions. Automatic execution risks inaccurate reporting, financial loss, or regulatory violations. Manual invocation allows review before committing.

### When to Use Manual vs Automatic

| Scenario | Approach |
|---|---|
| High-stakes, irreversible actions | Manual invocation |
| Financial or medical data modification | Manual invocation |
| Simple, well-understood, low-risk tasks | Automatic agent |
| Cost-sensitive, high-volume workflows | Manual (avoid unnecessary API calls) |
| Validated, tested pipelines with monitoring | Automatic agent with logging |

---

## Structured Outputs for Tool Calling

Structured outputs ensure LLM responses conform to a **predefined schema** — essential for database storage, API integration, and programmatic processing.

### Why Structured Outputs?

| Use Case | Why It Matters |
|---|---|
| **Database storage** | Consistent field names and types for direct DB insertion |
| **API integration** | Match exact API request/response formats |
| **UI components** | Format data for specific display elements |
| **Multi-step workflows** | Break complex tasks into structured steps |
| **Data extraction** | Pull specific fields from unstructured text |

### Schema Definition

**Method 1: JSON-like dict**
```python
schema = {
    "name": "string",
    "interest": "string"
}
```

**Method 2: Pydantic Model (preferred)**
```python
from pydantic import BaseModel, Field

class ResponseFormatter(BaseModel):
    name: str = Field(description="The person's name")
    interest: str = Field(description="The person's primary interest")
```

**Pydantic advantages:** Type validation · Clear field descriptions · Built-in documentation · LangChain integration

### Returning Structured Output

**Method 1: Tool Calling**
```python
# Bind schema as a tool to the model
model_with_schema = llm.bind_tools([ResponseFormatter])

# Invoke and extract
response = model_with_schema.invoke("I'm Lance and I like to bike.")
tool_args = response.tool_calls[0]["args"]
# {"name": "Lance", "interest": "Biking"}

# Parse to Pydantic object
result = ResponseFormatter(**tool_args)
```

**Method 2: `with_structured_output()` — Recommended**
```python
# All-in-one: binds schema, invokes, parses output
structured_llm = llm.with_structured_output(ResponseFormatter)
result = structured_llm.invoke("I'm Lance and I like to bike.")
# result.name = "Lance", result.interest = "Biking"
```

**Method 3: JSON Mode**
```python
structured_llm = llm.with_structured_output(ResponseFormatter, method="json_mode")
result = structured_llm.invoke("I'm Lance and I like to bike.")
# Returns Python dict directly
```

### `with_structured_output()` — What It Does Automatically

```
Schema provided as tool → LLM receives input → LLM generates answer string
    → Output parsed into structured schema format
```

Handles automatically: schema binding · instructing model to use the tool · parsing output back to schema

---

## Building LLM Agents with Tools

### Setting Up the Model

```python
from langchain.chat_models import init_chat_model

# Initialize the LLM
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```

### Defining Tools with @tool Decorator

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
```

> The **docstring** is what the LLM reads to decide which tool to call — make it clear and descriptive.

### Binding Tools to the Model

```python
tools = [add, subtract, multiply]

# Wraps LLM so it's aware of all tools
llm_with_tools = llm.bind_tools(tools)
```

### Tool Mapping Dictionary — Dynamic Invocation

```python
# Maps tool names (strings) to their function objects
tool_map = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply
}

# Dynamic invocation by name
tool_name = "add"
tool_args = {"a": 1, "b": 2}
result = tool_map[tool_name].invoke(tool_args)
# Returns: 3
```

> The mapping dictionary enables **dynamic function calls** — the LLM tells you which tool to call (by name), and you look it up from the map.

---

## Building Interactive LLM Agents — Manual Tool Execution

### Full Manual Tool-Calling Workflow

```
Step 1: Build chat history with user query as HumanMessage
Step 2: Invoke LLM with tools → receives AIMessage with tool_calls[]
Step 3: Extract tool name, args, and tool_call_id from AIMessage
Step 4: Append AIMessage to chat history
Step 5: Execute the tool manually using tool_map
Step 6: Wrap result in ToolMessage with matching tool_call_id
Step 7: Append ToolMessage to chat history
Step 8: Invoke LLM again with updated history → final natural language response
```

### Step-by-Step Code

```python
from langchain_core.messages import HumanMessage, ToolMessage

# Step 1: Build chat history
query = "What is 3 plus 2?"
chat_history = [HumanMessage(content=query)]

# Step 2: Invoke LLM with tools
response1 = llm_with_tools.invoke(chat_history)
# response1 is an AIMessage with tool_calls[]

# Step 3: Extract tool call details
tool_call = response1.tool_calls[0]
tool_name = tool_call["name"]       # e.g. "add"
tool_args = tool_call["args"]       # e.g. {"a": 3, "b": 2}
tool_call_id = tool_call["id"]      # unique ID linking result to request

# Step 4: Append AIMessage to history
chat_history.append(response1)

# Step 5: Execute tool manually
tool_result = tool_map[tool_name].invoke(tool_args)  # Returns: 5

# Step 6: Wrap result in ToolMessage
tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call_id   # links result back to the original call
)

# Step 7: Append ToolMessage to history
chat_history.append(tool_message)

# Step 8: Final LLM invocation
final_response = llm_with_tools.invoke(chat_history)
print(final_response.content)
# "The addition of 3 and 2 is 5."
```

### AIMessage Tool Call Structure

```python
# response1.tool_calls[0] contains:
{
    "name": "add",              # tool to call
    "args": {"a": 3, "b": 2},  # parameters as JSON
    "id": "call_abc123",        # unique ID for linking response
    "type": "tool_call"         # confirms this is a tool call, not text
}
```

> The **tool_call_id** is critical when multiple tools are called simultaneously — it links each result back to the correct tool call request.

### Chat History Message Types

| Message Type | Created By | Purpose |
|---|---|---|
| `HumanMessage` | User | User's query or input |
| `AIMessage` | LLM | Model's response (may contain tool_calls[]) |
| `ToolMessage` | System | Tool execution result, linked by tool_call_id |

---

## The ToolCallingAgent Class

Encapsulates the entire manual tool-calling workflow into a reusable class:

```python
class ToolCallingAgent:
    def __init__(self, llm, tools):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = {tool.name: tool for tool in tools}
        self.chat_history = []

    def invoke(self, user_input: str) -> str:
        # Add user message to history
        self.chat_history.append(HumanMessage(content=user_input))

        # Get LLM response (may contain tool calls)
        response = self.llm_with_tools.invoke(self.chat_history)
        self.chat_history.append(response)

        # Handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                # Execute the tool
                result = self.tool_map[tool_name].invoke(tool_args)

                # Append tool result
                self.chat_history.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                ))

            # Final LLM response with tool results
            final = self.llm_with_tools.invoke(self.chat_history)
            self.chat_history.append(final)
            return final.content

        return response.content

# Usage
agent = ToolCallingAgent(llm, tools=[add, subtract, multiply])
print(agent.invoke("What is 3 plus 2?"))
# "The addition of 3 and 2 is 5."
print(agent.invoke("1 minus 2"))
# Agent identifies intent even from imprecise input
```

**What the ToolCallingAgent handles automatically:**
- Binds tools to the LLM
- Manages full chat history
- Extracts tool names, args, and IDs from AIMessage
- Executes tools via the tool map
- Wraps results in ToolMessage with correct ID
- Handles multiple tool calls in one response
- Passes updated history back for final natural language response

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **LCEL** | Modern pipe-based syntax for composable LangChain chains |
| **`\|` operator** | Connects Runnables sequentially |
| **RunnableSequence** | Sequential execution — output feeds next component |
| **RunnableParallel** | Concurrent execution via dict syntax |
| **RunnableLambda** | Wraps Python functions into pipeline components |
| **Type coercion** | Dicts → `RunnableParallel`; functions → `RunnableLambda` (automatic) |
| **Manual tool calling** | Human reviews LLM's tool suggestion before execution — safer, more precise |
| **When to use manual** | High-stakes, irreversible, sensitive, or cost-sensitive operations |
| **Structured outputs** | LLM responses conform to a predefined schema (Pydantic or JSON) |
| **`with_structured_output()`** | Recommended all-in-one: binds schema, invokes, parses output |
| **Tool map** | Dict mapping tool name strings to function objects — enables dynamic invocation |
| **`HumanMessage`** | Wraps user input for chat history |
| **`AIMessage`** | LLM response — may contain `tool_calls[]` array |
| **`ToolMessage`** | Wraps tool result — linked to original call via `tool_call_id` |
| **`tool_call_id`** | Links each tool result back to its request — critical for multiple simultaneous calls |
| **ToolCallingAgent** | Class encapsulating the full manual tool-calling workflow end-to-end |

---

*Notes based on: Course 06 Module 02 — LCEL and Manual Tool Calling in LangChain (Fundamentals of Building AI Agents)*
