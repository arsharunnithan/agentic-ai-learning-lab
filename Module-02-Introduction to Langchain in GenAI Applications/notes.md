### 1. Introduction to LangChain

#### What is LangChain?

LangChain is an open-source framework used to build applications powered by Large Language Models (LLMs). It helps developers connect prompts, models, and external data sources into structured workflows.

---

#### Why is LangChain needed?

Calling an LLM directly works for simple tasks, but real-world applications require:

- Multi-step processing
- Context management
- Integration with external data (e.g., databases, APIs)
- Structured and reusable workflows

LangChain provides a framework to handle all of this efficiently.

---

#### Key Features

- **Modularity**: Combine components like prompts, models, and tools
- **Chaining**: Break complex tasks into smaller steps
- **Extensibility**: Easily integrate external systems
- **Vector DB Integration**: Enables semantic search and RAG systems

---

#### Common Use Cases

- Question-answering systems
- Document summarization
- Data extraction from text
- AI-powered assistants

---

#### Conclusion
LangChain is not just about calling an LLM. It is used to build complete AI systems by connecting multiple components such as prompts, models, and external data sources into a structured pipeline.
---
# LangChain Core Concepts — Summary

LangChain is an **open-source interface** that simplifies app development using LLMs, supporting use cases like NLP and data retrieval.

---

## Core Components

### 1. 🧠 Language Model
- Takes **text input → generates text output**
- Used for task completion and document summarization
- Supported providers: **IBM, OpenAI, Google, Meta**

---

### 2. 💬 Chat Model
- Designed for **conversational interactions** (responds like a human)
- Built on top of a language model (e.g., WatsonX LLM → Chat Model)

**Chat Message Types:**

| Message Type | Purpose |
|---|---|
| Human Message | User input |
| AI Message | Model-generated response |
| System Message | Instructions to guide the model |
| Function Message | Handles function call outcomes |
| Tool Message | Manages tool interactions |

> Each message has two properties: **role** (who's speaking) and **content** (what's said).

---

### 3. 📝 Prompt Templates
Translate user questions into **clear instructions** for the model.

**Types:**
- **String Prompt Template** — single-string formatting
- **Chat Prompt Template** — message lists with role + content
- **Message Prompt Templates** — AI, System, Human, Chat variants
- **Messages Placeholder** — full control over message rendering
- **Few-Shot Prompt Template** — provides examples to guide the LLM

**Example Selectors** (for Few-Shot templates):
- Semantic Similarity
- Max Marginal Relevance (for diversity)
- N-Gram Overlap (for textual similarity)

---

### 4. 📤 Output Parsers
Transform raw LLM output into **structured, usable formats**.

**Supported formats:** `JSON` · `XML` · `CSV` · `Pandas DataFrames`

> Example: *Comma Separated List Output Parser* → converts response to CSV, ready for spreadsheet analysis.

---

## TL;DR

| Component | Role |
|---|---|
| Language Model | Text in → Text out |
| Chat Model | Conversational responses |
| Prompt Templates | Structure inputs for the model |
| Output Parsers | Structure outputs from the model |

## Simple Flow of LangChain

User Input  
↓  
Prompt Template  
↓  
LLM / Chat Model  
↓  
Output Parser  
↓  
Final Structured Output  

---

## Conclusion

LangChain is a framework that organizes how we interact with LLMs. Instead of sending raw prompts, it allows us to structure inputs, manage conversations, and convert outputs into usable formats, making it easier to build real-world AI applications.
---
# LangChain: Chains & Agents for Building Applications

> A comprehensive reference note covering chains, memory, and agents in LangChain.

---

## Table of Contents

1. [What is LangChain?](#what-is-langchain)
2. [Chains in LangChain](#chains-in-langchain)
   - [Sequential Chains](#sequential-chains)
   - [Building a 3-Chain Example](#building-a-3-chain-example)
     - [Chain 1 – Location to Dish](#chain-1--location-to-dish)
     - [Chain 2 – Dish to Recipe](#chain-2--dish-to-recipe)
     - [Chain 3 – Recipe to Cooking Time](#chain-3--recipe-to-cooking-time)
   - [Combining Chains into a Sequential Chain](#combining-chains-into-a-sequential-chain)
3. [Memory in LangChain](#memory-in-langchain)
   - [How Memory Works](#how-memory-works)
   - [ChatMessageHistory Example](#chatmessagehistory-example)
4. [Agents in LangChain](#agents-in-langchain)
   - [How Agents Work](#how-agents-work)
   - [Pandas DataFrame Agent Example](#pandas-dataframe-agent-example)
5. [Summary](#summary)

---

## What is LangChain?

LangChain is a **platform embedded with APIs** designed to help developers build intelligent applications by infusing language processing capabilities. It provides a structured framework using three core building blocks:

| Tool | Purpose |
|---|---|
| **Documents** | Source material for context and retrieval |
| **Chains** | Sequenced calls to process and transform information |
| **Agents** | Dynamic systems that reason and interact with external tools |

---

## Chains in LangChain

A **chain** is a sequence of calls where the output of one step becomes the input for the next, forming a seamless pipeline of information.

### Sequential Chains

A **sequential chain** consists of basic steps where:

- Each step takes **one input** and produces **one output**
- Output from **Step N** → Input for **Step N+1**
- All individual chains are wrapped into a **unified process**

```
[User Prompt] → Chain 1 → Chain 2 → Chain 3 → [Final Output]
```

---

### Building a 3-Chain Example

**Goal:** Given a location, identify the famous dish, its recipe, and estimated cooking time.

```
Location (China) → Famous Dish (Peking Duck) → Recipe → Cooking Time
```

---

#### Chain 1 – Location to Dish

**Input:** User-specified location (e.g., `China`)  
**Output:** Famous dish from that location (e.g., `Peking Duck`), stored under key `meal`

```python
# Step 1: Define the prompt template
template = "What is a famous dish from {location}?"

# Step 2: Create a PromptTemplate object
prompt = PromptTemplate(input_variables=["location"], template=template)

# Step 3: Create the LLM chain
location_chain = LLMChain(
    llm=mixtral_llm,       # pre-instantiated chat model
    prompt=prompt,
    output_key="meal"
)
```

---

#### Chain 2 – Dish to Recipe

**Input:** `meal` (output from Chain 1)  
**Output:** Recipe for the dish, stored under key `recipe`

```python
# Step 1: Define the prompt template
template = "Give me a simple recipe for {meal}."

# Step 2: Create a PromptTemplate object
prompt = PromptTemplate(input_variables=["meal"], template=template)

# Step 3: Create the LLM chain
dish_chain = LLMChain(
    llm=mixtral_llm,
    prompt=prompt,
    output_key="recipe"
)
```

---

#### Chain 3 – Recipe to Cooking Time

**Input:** `recipe` (output from Chain 2)  
**Output:** Estimated cooking time, stored under key `time`

```python
# Step 1: Define the prompt template
template = "Estimate the cooking time for this recipe: {recipe}"

# Step 2: Create a PromptTemplate object
prompt = PromptTemplate(input_variables=["recipe"], template=template)

# Step 3: Create the LLM chain
recipe_chain = LLMChain(
    llm=mixtral_llm,
    prompt=prompt,
    output_key="time"
)
```

---

### Combining Chains into a Sequential Chain

All three chains are wrapped together into a `SequentialChain`, creating a single unified pipeline.

```python
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=["location"],
    output_variables=["meal", "recipe", "time"],
    verbose=True   # Set to True to trace the full information flow
)

# Run the chain
result = overall_chain.invoke({"location": "China"})
```

> **Tip:** Setting `verbose=True` gives a detailed view of how each input is transformed at every step through to the final output.

---

## Memory in LangChain

Memory in LangChain enables chains and agents to **read and write historical data**, preserving context across multiple interactions.

### How Memory Works

Every chain relies on two core inputs:

- **User input** – the current prompt from the user
- **Memory** – historical context from past interactions

The memory lifecycle within a chain:

```
Before execution:  Memory is READ → enhances the user's input
After execution:   Current inputs & outputs are WRITTEN back to memory
```

This ensures **continuity and context preservation** across interactions.

---

### ChatMessageHistory Example

The `ChatMessageHistory` class manages and stores conversation histories, supporting both **human messages** and **AI messages**.

```python
from langchain.memory import ChatMessageHistory

# Instantiate the history object
history = ChatMessageHistory()

# Add an AI message to memory
history.add_ai_message("Hi!")

# Add a human (user) message to memory
history.add_user_message("What is the capital of France?")

# The memory now holds:
# [AIMessage: "Hi!", HumanMessage: "What is the capital of France?"]
```

All stored messages become available context for subsequent responses, enabling coherent multi-turn conversations.

---

## Agents in LangChain

**Agents** are dynamic systems where a language model determines and sequences actions, going beyond static chains to interact with external tools and data sources.

### How Agents Work

| Aspect | Detail |
|---|---|
| **Decision-making** | The LLM reasons about what action to take next |
| **Output** | Generates text outputs to guide actions |
| **Execution** | Does **not** execute actions directly — delegates to tools |
| **Tools** | Search engines, databases, websites, APIs, etc. |

**Example flow** – User asks: *"What is the population of Italy?"*

```
User Query
    ↓
Agent (LLM reasons about options)
    ↓
Queries External Database / Tool
    ↓
Returns curated, accurate answer
```

This demonstrates the agent's ability to **autonomously combine LLM reasoning with external tool use**.

---

### Pandas DataFrame Agent Example

LangChain includes specialized agents such as the **Pandas DataFrame Agent**, which allows users to query and visualize tabular data using natural language.

```python
from langchain.agents import create_pandas_dataframe_agent

# Create the agent
agent = create_pandas_dataframe_agent(
    llm=chat_model,      # pre-instantiated LLM
    df=dataframe,        # your Pandas DataFrame
    verbose=True         # shows LLM's reasoning process
)

# Invoke a natural language query
response = agent.invoke("How many rows are in the dataframe?")

# Example output:
# → "There are 139 rows in the DataFrame."
```

**Under the hood:**
1. The LLM translates the natural language query into Python code
2. The code is executed in the background against the DataFrame
3. The result is returned as a precise, human-readable answer

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **LangChain** | API-embedded platform for building language-powered applications |
| **Chains** | Sequences of calls; output of one step feeds into the next |
| **Sequential Chain** | Wraps multiple chains into a single unified pipeline |
| **Chain Setup** | Define template → Create PromptTemplate → Create LLMChain |
| **Memory** | Reads context before execution; writes results after execution |
| **ChatMessageHistory** | Stores AI and human messages for multi-turn conversations |
| **Agents** | Dynamic systems that use LLM reasoning + external tools |
| **DataFrame Agent** | Translates natural language into executable Python/pandas code |

---

*Notes based on the LangChain Chains & Agents introductory module.*

---


# LangChain LCEL – The Modern Chaining Method

> A comprehensive reference note on LangChain Expression Language (LCEL) for building flexible, composable AI pipelines.

---

## Table of Contents

1. [What is LCEL?](#what-is-lcel)
2. [Why LCEL over Traditional LLMChain?](#why-lcel-over-traditional-llmchain)
3. [Core Steps to Build an LCEL Chain](#core-steps-to-build-an-lcel-chain)
4. [Runnable Primitives](#runnable-primitives)
   - [RunnableSequence](#runnablesequence)
   - [RunnableParallel](#runnableparallel)
   - [RunnableLambda](#runnablelambda)
5. [Type Coercion in LCEL](#type-coercion-in-lcel)
6. [LCEL in Action – Simple Chain Example](#lcel-in-action--simple-chain-example)
7. [LCEL in Action – Parallel Chain Example](#lcel-in-action--parallel-chain-example)
8. [When to Use LCEL vs LangGraph](#when-to-use-lcel-vs-langgraph)
9. [Key Strengths of LCEL](#key-strengths-of-lcel)
10. [Summary](#summary)

---

## What is LCEL?

**LangChain Expression Language (LCEL)** is the modern, recommended pattern for building LangChain applications. It uses the **pipe operator (`|`)** to connect components, creating a clean and readable flow of data from input to output.

```
Input → Component 1 | Component 2 | Component 3 → Output
```

LCEL acts as the "glue" between LangChain building blocks — prompts, LLMs, retrievers, parsers, and tools — turning them into composable, maintainable pipelines.

---

## Why LCEL over Traditional LLMChain?

LangChain has evolved significantly. LCEL is the **newer, recommended approach** over the traditional `LLMChain` pattern.

| Feature | Traditional LLMChain | LCEL |
|---|---|---|
| Readability | Moderate | High — pipe operator is intuitive |
| Composability | Limited | Excellent |
| Data flow visualization | Implicit | Explicit and clear |
| Parallel execution | Manual setup | Built-in via `RunnableParallel` |
| Async support | Limited | Native |
| Streaming | Complex | Simplified |
| Automatic tracing | No | Yes |

---

## Core Steps to Build an LCEL Chain

Every LCEL chain follows these four steps:

1. **Define a template** with variables in curly braces `{variable}`
2. **Create a `PromptTemplate` instance** from the template
3. **Build the chain** using the pipe operator `|` to connect components
4. **Invoke the chain** with input values

```python
# Step 1: Define the template
template = "Tell me a {adjective} joke about {content}."

# Step 2: Create PromptTemplate
prompt = PromptTemplate(input_variables=["adjective", "content"], template=template)

# Step 3: Build chain with pipe operator
chain = prompt | llm | StrOutputParser()

# Step 4: Invoke with inputs
response = chain.invoke({"adjective": "funny", "content": "cats"})
```

---

## Runnable Primitives

In LangChain, **Runnables** are the interfaces and building blocks that connect different components — LLMs, retrievers, tools — into a pipeline. There are two main composition primitives:

---

### RunnableSequence

Chains components **sequentially**, passing the output of one as the input to the next.

```python
from langchain_core.runnables import RunnableSequence

# Explicit way
chain = RunnableSequence(first=component_1, last=component_2)

# LCEL shorthand (pipe operator — preferred)
chain = component_1 | component_2
```

> ✅ Both are equivalent. The pipe operator is the cleaner, preferred syntax.

---

### RunnableParallel

Runs **multiple components concurrently**, each receiving the **same input** and producing independent outputs.

```python
from langchain_core.runnables import RunnableParallel

# Explicit way
parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    sentiment=sentiment_chain
)

# LCEL shorthand — using a dictionary (auto-coerced)
parallel_chain = {
    "summary": summary_chain,
    "translation": translation_chain,
    "sentiment": sentiment_chain
}
```

**Output structure:**
```python
{
    "summary": "...",
    "translation": "...",
    "sentiment": "..."
}
```

Each key holds the result of its respective chain, all processed simultaneously.

---

### RunnableLambda

Wraps a **regular Python function** and converts it into a runnable component that LangChain can work with in a pipeline.

```python
from langchain_core.runnables import RunnableLambda

def format_prompt(inputs):
    return f"Tell me a {inputs['adjective']} joke about {inputs['content']}."

# Wrap the function as a Runnable
formatted = RunnableLambda(format_prompt)

# Use in a chain
chain = formatted | llm | StrOutputParser()
```

---

## Type Coercion in LCEL

LCEL **automatically converts** regular Python objects into compatible Runnable components behind the scenes — no manual wrapping required.

| Python Type | Auto-converted To | Behavior |
|---|---|---|
| `dict` | `RunnableParallel` | Runs all values concurrently with the same input |
| `function` | `RunnableLambda` | Transforms inputs using the function |

```python
# This dictionary is automatically treated as RunnableParallel
chain = {
    "summary": prompt_1 | llm,
    "translation": prompt_2 | llm,
    "sentiment": prompt_3 | llm
} | output_parser
```

> LCEL handles all type conversion in the background, keeping your code concise and clean.

---

## LCEL in Action – Simple Chain Example

A basic sequential chain using `RunnableLambda`, the pipe operator, and `StrOutputParser`:

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Define a formatting function
def format_prompt(inputs):
    adjective = inputs["adjective"]
    content = inputs["content"]
    return f"Tell me a {adjective} joke about {content}."

# Build the chain
joke_chain = (
    RunnableLambda(format_prompt)   # Step 1: Format the prompt
    | llm                           # Step 2: Pass to LLM
    | StrOutputParser()             # Step 3: Parse output to string
)

# Invoke
result = joke_chain.invoke({"adjective": "funny", "content": "programmers"})
print(result)
```

**Data flow:**
```
{"adjective": "funny", "content": "programmers"}
    ↓ RunnableLambda (formats prompt string)
"Tell me a funny joke about programmers."
    ↓ LLM (generates response)
"Why do programmers prefer dark mode? Because light attracts bugs!"
    ↓ StrOutputParser (extracts plain string)
Final output string
```

---

## LCEL in Action – Parallel Chain Example

Processing the same input through **three tasks simultaneously**:

```python
# Three prompt templates for different tasks
summary_prompt = PromptTemplate.from_template("Summarize this: {text}")
translate_prompt = PromptTemplate.from_template("Translate to French: {text}")
sentiment_prompt = PromptTemplate.from_template("What is the sentiment of: {text}")

# Parallel chain — all three run at the same time
parallel_chain = {
    "summary":     summary_prompt   | llm | StrOutputParser(),
    "translation": translate_prompt | llm | StrOutputParser(),
    "sentiment":   sentiment_prompt | llm | StrOutputParser()
}

# Invoke with shared input
result = parallel_chain.invoke({"text": "LangChain makes AI development easy and fun!"})

# Result structure:
# {
#   "summary": "LangChain simplifies AI development.",
#   "translation": "LangChain rend le développement IA facile et amusant !",
#   "sentiment": "Positive"
# }
```

---

## When to Use LCEL vs LangGraph

| Use Case | Recommended Tool |
|---|---|
| Simple sequential pipelines | ✅ LCEL |
| Parallel task execution | ✅ LCEL |
| Prompt + LLM + parser workflows | ✅ LCEL |
| Complex multi-step workflows with branching logic | ✅ LangGraph (use LCEL within individual nodes) |
| Stateful, cyclical agent workflows | ✅ LangGraph |

> **Best practice:** Use LCEL for orchestration within nodes, and LangGraph to manage complex workflows between nodes.

---

## Key Strengths of LCEL

| Strength | Description |
|---|---|
| **Parallel execution** | Run multiple chains concurrently with `RunnableParallel` |
| **Async support** | Native `async`/`await` support for non-blocking pipelines |
| **Simplified streaming** | Stream tokens from LLMs with minimal setup |
| **Automatic tracing** | Built-in tracing for debugging and monitoring |
| **Composability** | Mix and match components freely using `|` |
| **Type coercion** | Auto-converts dicts and functions — less boilerplate |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **LCEL** | Modern LangChain pattern using the pipe operator `\|` for clean data flow |
| **Pipe operator `\|`** | Connects runnable components sequentially |
| **PromptTemplate** | Defines prompts with `{variable}` placeholders |
| **RunnableSequence** | Chains components one after another |
| **RunnableParallel** | Runs multiple components simultaneously on the same input |
| **RunnableLambda** | Wraps Python functions into runnable pipeline components |
| **Type coercion** | Dicts → `RunnableParallel`; Functions → `RunnableLambda` (auto) |
| **LCEL vs LangGraph** | LCEL for simple flows; LangGraph for complex/stateful workflows |

---

*Notes based on the LangChain LCEL Chaining Method module.*
