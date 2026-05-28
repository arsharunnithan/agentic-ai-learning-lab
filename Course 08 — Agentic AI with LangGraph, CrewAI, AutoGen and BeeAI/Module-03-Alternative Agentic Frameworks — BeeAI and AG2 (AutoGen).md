# Course 08 — Module 03: Alternative Agentic Frameworks — BeeAI and AG2 (AutoGen)

> A complete reference covering BeeAI fundamentals, agent development, and AG2/AutoGen core concepts, tools, structured outputs, and orchestration patterns.

---

## Table of Contents

1. [BeeAI — Core Capabilities](#beeai--core-capabilities)
2. [BeeAI — First Conversation & Prompt Templates](#beeai--first-conversation--prompt-templates)
3. [BeeAI — Structured Outputs & Memory](#beeai--structured-outputs--memory)
4. [BeeAI — Building Agents](#beeai--building-agents)
5. [BeeAI — Requirements System](#beeai--requirements-system)
6. [BeeAI — ReAct, Human-in-the-Loop & Custom Tools](#beeai--react-human-in-the-loop--custom-tools)
7. [BeeAI — Multi-Agent Systems](#beeai--multi-agent-systems)
8. [AG2 (AutoGen) — Core Concepts](#ag2-autogen--core-concepts)
9. [AG2 — Agent Types & Conversations](#ag2--agent-types--conversations)
10. [AG2 — Orchestration Patterns](#ag2--orchestration-patterns)
11. [AG2 — Tools & Structured Outputs](#ag2--tools--structured-outputs)
12. [AG2 — Production Best Practices](#ag2--production-best-practices)
13. [Summary](#summary)

---

## BeeAI — Core Capabilities

**BeeAI** is a cutting-edge, open-source platform for building **production-ready AI agents and multi-agent systems**, developed under the Linux Foundation AI and Data Program and backed by IBM Research.

### Four Key Architectural Advantages

| Advantage | Description |
|---|---|
| **Production-ready architecture** | Built-in caching, memory optimization, resource management, OpenTelemetry integration |
| **Provider-agnostic backend** | Supports 10+ LLM providers — OpenAI, WatsonX.ai, Grok, Ollama, Anthropic, and more |
| **Advanced agent patterns** | ReAct, systematic thinking, multi-agent coordination — proven patterns built-in |
| **Dual-language support** | Complete feature parity between Python and TypeScript |

### Why BeeAI Uses `async` / `await`

BeeAI uses Python's async syntax because LLM calls are **I/O-bound operations** — `async/await` allows multiple tasks to run simultaneously without blocking the application.

| Keyword | Purpose |
|---|---|
| `async def` | Defines a coroutine function |
| `await` | Pauses execution until the async operation completes |

This is especially important for multi-agent coordination where several agents may be calling external APIs simultaneously.

### Key Benefits

**Modularity** · **Structured outputs** · **Async execution** · **Multi-agent support** · **Standards compliance (MCP, A2A)** · **Observability (OpenTelemetry)**

---

## BeeAI — First Conversation & Prompt Templates

### Basic Conversation

```python
import asyncio
from beeai_framework.models.openai import OpenAIChatModel
from beeai_framework.models.message import SystemMessage, UserMessage

async def main():
    # Initialize LLM (IBM Granite on WatsonX)
    llm = OpenAIChatModel("ibm/granite-13b-instruct-v2")

    messages = [
        SystemMessage("You are a helpful data science assistant."),
        UserMessage("What is the difference between supervised and unsupervised learning?")
    ]

    response = await llm.create(messages=messages)
    print(response.get_text_content())

asyncio.run(main())
```

### Dynamic Prompt Templates

Templates use **mustache-style syntax** (`{{variable}}`) for reusable, consistent prompts across multiple inputs.

```python
from beeai_framework.template import SimplePromptTemplate

# Define template with placeholders
template = SimplePromptTemplate("""
Evaluate this data science project:
- Project Name: {{project_name}}
- Business Problem: {{business_problem}}
- Data Description: {{data_description}}
- Timeline: {{timeline}}
- Success Metrics: {{success_metrics}}
""")

# Render with actual data
rendered = template.render(
    project_name="Customer Churn Prediction",
    business_problem="Reduce churn by 15%",
    data_description="12 months of transaction data, 50K customers",
    timeline="8 weeks",
    success_metrics="AUC > 0.85, precision > 0.80"
)

# Send to LLM
response = await llm.create(messages=[UserMessage(rendered)])
```

> Templates ensure every project is formatted the same way before going to the LLM — reduces bias from format inconsistency.

---

## BeeAI — Structured Outputs & Memory

### Structured Outputs with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class BusinessPlan(BaseModel):
    business_name: str = Field(description="Name of the business")
    elevator_pitch: str = Field(description="One-sentence business description")
    revenue_streams: List[str] = Field(description="List of revenue sources")
    target_market: str = Field(description="Primary customer segment")

async def generate_business_plan(description: str):
    messages = [
        SystemMessage("You are a business strategy expert."),
        UserMessage(f"Create a business plan for: {description}")
    ]

    # Use create_structure instead of create
    result = await llm.create_structure(schema=BusinessPlan, messages=messages)
    return result  # Returns typed, validated BusinessPlan object

plan = asyncio.run(generate_business_plan("An AI-powered meal planning app"))
print(plan.business_name)
print(plan.revenue_streams)
```

> `create_structure` guarantees the response matches the schema exactly — no parsing required.

### Memory Management — UnconstrainedMemory

`UnconstrainedMemory` stores all messages without limits — ideal for maintaining full conversational history.

```python
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.models.message import SystemMessage, UserMessage, AssistantMessage

async def main():
    memory = UnconstrainedMemory()

    # Add individual messages (async operation)
    await memory.add(SystemMessage("You are a helpful assistant."))
    await memory.add(UserMessage("What is machine learning?"))

    # Add multiple messages at once (more efficient)
    await memory.add_many([
        UserMessage("Explain deep learning."),
        AssistantMessage("Deep learning uses neural networks with multiple layers...")
    ])

    # Check and iterate
    print(memory.is_empty())         # False
    for message in memory.messages:
        print(message.text)

    # Clear all history
    await memory.reset()
    print(memory.is_empty())         # True
```

| Method | Description |
|---|---|
| `add(message)` | Add a single message (async) |
| `add_many(messages)` | Add multiple messages at once — more efficient for bulk ops |
| `is_empty()` | Returns bool — whether any messages are stored |
| `memory.messages` | Iterable of all stored messages |
| `reset()` | Clears all stored messages |

---

## BeeAI — Building Agents

The **`RequirementAgent`** class is BeeAI's core abstraction for intelligent, controllable agents. Unlike simple chat models, agents:
- Maintain **persistent state**
- Use **external tools**
- Follow **behavioral requirements**

```python
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search import WikipediaTool

async def main():
    agent = RequirementAgent(
        llm=llm,
        memory=UnconstrainedMemory(),
        tools=[WikipediaTool()],
        instructions="You are a research assistant. Use Wikipedia to answer questions accurately."
    )

    response = await agent.run("What is the history of artificial intelligence?")
    print(response.result.text)

asyncio.run(main())
```

---

## BeeAI — Requirements System

The requirements system provides **fine-grained control** over agent behavior through execution parameters.

| Parameter | Description |
|---|---|
| `ConditionalRequirement` | Controls tool execution order and frequency |
| `AskPermissionRequirement` | Creates human approval step before calling a tool |
| `ControlTrajectoryMiddleware` | Tracks complete execution flow for debugging |
| `force_at_step` | Forces tool usage at a specific step |
| `min_invocations` / `max_invocations` | Controls tool usage frequency |
| `only_after` | Creates tool dependencies — logical execution sequences |
| `consecutive_allowed` | Controls whether the same tool can run consecutively |

```python
from beeai_framework.agents.requirement import RequirementAgent, ConditionalRequirement
from beeai_framework.tools.search import WikipediaTool
from beeai_framework.tools.think import ThinkTool

agent = RequirementAgent(
    llm=llm,
    memory=UnconstrainedMemory(),
    tools=[WikipediaTool(), ThinkTool()],
    requirements=[
        ConditionalRequirement(WikipediaTool, max_invocations=1),  # max 1 Wikipedia search
        ConditionalRequirement(ThinkTool, max_invocations=3)        # max 3 think cycles
    ]
)
```

---

## BeeAI — ReAct, Human-in-the-Loop & Custom Tools

### ReAct Agent Pattern

Forces the **Think → Act → Think → Act → Final Answer** cycle with full observability.

```python
from beeai_framework.middleware import GlobalTrajectoryMiddleware
from beeai_framework.agents.requirement import Tool

agent = RequirementAgent(
    llm=llm,
    memory=UnconstrainedMemory(),
    tools=[ThinkTool(), WikipediaTool()],
    middleware=[GlobalTrajectoryMiddleware()],   # full execution visibility
    requirements=[
        ConditionalRequirement(ThinkTool,
            force_at_step=1,           # thinking MUST start first
            force_after=Tool,          # think after EVERY tool call
            consecutive_allowed=False, # no repetitive thinking
            max_invocations=3
        ),
        ConditionalRequirement(WikipediaTool, max_invocations=2)
    ]
)
```

**Execution flow:**
```
Step 1: ThinkTool (forced) → reason about the problem
Step 2: WikipediaTool → retrieve information
Step 3: ThinkTool (forced after tool) → reflect on results
Step 4: WikipediaTool (if needed) → additional research
Step 5: ThinkTool (forced after tool) → final reflection
Step 6: Final Answer generated
```

### Human-in-the-Loop

```python
from beeai_framework.agents.requirement import AskPermissionRequirement

agent = RequirementAgent(
    llm=llm,
    memory=UnconstrainedMemory(),
    tools=[ThinkTool(), WikipediaTool()],
    requirements=[
        AskPermissionRequirement(WikipediaTool),    # human must approve before Wikipedia search
        ConditionalRequirement(ThinkTool, max_invocations=3)
    ]
)
# When the agent wants to call WikipediaTool, it pauses and asks for human approval
```

### Custom Tools

```python
from pydantic import BaseModel, Field
from beeai_framework.tools import Tool

class MathInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class AddTool(Tool):
    name = "AddNumbers"
    description = "Add two numbers together and return the result."
    input_schema = MathInput

    async def _run(self, input: MathInput) -> str:
        result = input.a + input.b
        return f"The sum of {input.a} and {input.b} is {result}"
```

> Custom tools require: **name**, **description**, **input schema** (Pydantic BaseModel), and **`_run` method** containing the logic.

---

## BeeAI — Multi-Agent Systems

BeeAI uses **`HandoffTool`** to delegate tasks from a coordinator agent to specialist agents.

```python
from beeai_framework.tools.handoff import HandoffTool

# Specialist agents
research_agent = RequirementAgent(llm=llm, tools=[WikipediaTool()],
    instructions="You specialize in research and fact-finding.")

coding_agent = RequirementAgent(llm=llm, tools=[AddTool()],
    instructions="You specialize in code and calculations.")

# Create handoff tools for each specialist
research_handoff = HandoffTool(agent=research_agent, description="Delegate research tasks here")
coding_handoff = HandoffTool(agent=coding_agent, description="Delegate coding and math tasks here")

# Coordinator agent routes to specialists
coordinator = RequirementAgent(
    llm=llm,
    memory=UnconstrainedMemory(),
    tools=[ThinkTool(), research_handoff, coding_handoff],
    instructions="Analyze queries and delegate to the right specialist."
)

response = await coordinator.run("What is the Fibonacci sequence and write code to compute it?")
```

---

## AG2 (AutoGen) — Core Concepts

**AG2** (formerly AutoGen) is an open-source framework for building intelligent AI agents that collaborate through **structured interactions and role-based tasks**.

| Feature | Description |
|---|---|
| **Provider-agnostic** | Integrates with OpenAI, Anthropic, and other LLM providers |
| **Human-in-the-loop** | Configurable oversight and intervention at any step |
| **Multi-agent orchestration** | Coordinate multiple agents for complex collaborative tasks |
| **Tools integration** | Extend agents with API calls, code execution, external systems |
| **Structured outputs** | Enforce consistent, validated response formats |

### Setup

```python
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
from autogen import LLMConfig

# Configure LLM
llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY",
    temperature=0.0
)
```

---

## AG2 — Agent Types & Conversations

### Core Concepts

| Concept | Description |
|---|---|
| **ConversableAgent** | Base class — all AG2 agents inherit from this; can send, receive, and respond to messages |
| **AssistantAgent** | Handles problem-solving and code generation |
| **UserProxyAgent** | Executes code and provides feedback — defaults to human oversight |

### Two-Agent Chat

```python
student = ConversableAgent(
    name="Student",
    system_message="You are a curious student who asks clarifying questions.",
    human_input_mode="NEVER",   # full automation
    llm_config=llm_config
)

tutor = ConversableAgent(
    name="Tutor",
    system_message="You are an expert tutor who explains concepts clearly.",
    human_input_mode="NEVER",
    llm_config=llm_config
)

result = student.initiate_chat(
    recipient=tutor,
    message="Can you explain neural networks?",
    max_turns=4,
    summary_method="reflection_with_llm"   # generates summary from full conversation
)

print(result.summary)
```

### Code Execution

```python
from autogen import AssistantAgent, UserProxyAgent, LocalCommandLineCodeExecutor

assistant = AssistantAgent(name="Coder", llm_config=llm_config)

user_proxy = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")}
)

user_proxy.initiate_chat(assistant, message="Write Python code to plot a sine wave.")
# assistant writes code → user_proxy executes it → saves sine_wave.png
```

### Human-in-the-Loop Modes

| Mode | Behavior |
|---|---|
| `"ALWAYS"` | Pauses for human input at every step |
| `"NEVER"` | Full automation — no human input |
| `"TERMINATE"` | Requests human input only when ending the conversation |

---

## AG2 — Orchestration Patterns

### 1. Two-Agent Chat
Simplest pattern — two agents in direct back-and-forth conversation.
```python
agent_a.initiate_chat(agent_b, message="...", max_turns=5)
```

### 2. Sequential Chat
Chains multiple two-agent chats — output ("carryover") of each becomes input to the next.
```python
agent_a.initiate_chats([
    {"recipient": agent_b, "message": "Draft content", "summary_method": "reflection_with_llm"},
    {"recipient": agent_c, "message": "Review draft", "carryover": "..."},
    {"recipient": agent_d, "message": "Format final version"}
])
```

### 3. Nested Chat
Encapsulates a complex multi-agent workflow under a single trigger agent — reusable and modular.
```python
curriculum_planner.register_nested_chats(
    trigger=teacher_agent,
    chat_queue=[
        {"recipient": math_planner, "message": "Plan math content"},
        {"recipient": history_planner, "message": "Plan history content"}
    ]
)
```

### 4. Group Chat

Multiple agents interact in a shared conversation space managed by `GroupChatManager`.

```python
from autogen import GroupChat, GroupChatManager

lesson_planner = ConversableAgent(name="Planner", system_message="Create lesson content.", ...)
reviewer = ConversableAgent(name="Reviewer", system_message="Review and improve lessons.", ...)
teacher = ConversableAgent(name="Teacher",
    system_message="Oversee the process. Say 'DONE' when complete.",
    is_termination_msg=lambda x: "DONE" in x.get("content", ""),
    ...)

group_chat = GroupChat(
    agents=[lesson_planner, reviewer, teacher],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"   # LLM selects next speaker
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)
teacher.initiate_chat(manager, message="Make a lesson about the moon.")
```

### Speaker Selection Patterns

| Pattern | Description | Best For |
|---|---|---|
| `AutoPattern` / `"auto"` | LLM selects next speaker based on context | Adaptive, dynamic conversations |
| `RoundRobinPattern` | Fixed rotation sequence | Structured updates, balanced participation |
| `RandomPattern` | Random agent selection | Brainstorming, varied input |
| `ManualPattern` | Human selects next speaker | Educational settings, oversight |
| `DefaultPattern` | Requires explicit agent handoffs | Strict, predictable workflows |

### Routing & Handoffs

```python
from autogen import OnCondition, OnContextCondition, ContextVariables

# LLM-based routing — evaluate message content
agent.register_handoff(
    OnCondition(target=escalation_agent, condition="Message contains 'urgent outage'")
)

# Context-based routing — check shared state
agent.register_handoff(
    OnContextCondition(target=escalation_agent,
                       condition=lambda ctx: ctx.get("issue_severity", 0) > 8)
)
```

### Guardrails

```python
from autogen import RegexGuardrail, LLMGuardrail

# Pattern matching — detect PII
RegexGuardrail(pattern=r"\d{3}-\d{2}-\d{4}", redirect_to=safety_agent)

# Semantic filtering — detect unsafe content
LLMGuardrail(prompt="Is this message harmful?", redirect_to=compliance_agent)
```

### Termination Mechanisms

| Mechanism | How |
|---|---|
| `max_turns` | Two-agent chat turn limit |
| `max_round` | Group chat round limit |
| `is_termination_msg` | Function that returns True to end chat (e.g. "DONE" in message) |
| `max_consecutive_auto_reply` | Prevents infinite agent loops |
| `human_input_mode="ALWAYS"` | Type "exit" to end session |
| `TerminateTarget` | Fallback when no further handoff is possible |

### Context Variables

```python
from autogen import ContextVariables

ctx = ContextVariables({"issue_severity": 0, "customer_tier": "premium"})
# Shared key-value store accessible across all agents and tools in the workflow
# Updated by tools via ReplyResult — persists across interactions
```

---

## AG2 — Tools & Structured Outputs

### Registering Tools

```python
from autogen import register_function

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Link tool between two agents — reasoning and execution stay separate
register_function(
    is_prime,
    caller=math_asker,    # agent that calls the tool
    executor=math_checker # agent that executes it
)
```

### Structured Outputs with Pydantic

```python
from pydantic import BaseModel
from typing import Literal

class TicketSummary(BaseModel):
    customer_name: str
    issue_type: Literal["billing", "technical", "general"]
    urgency: Literal["low", "medium", "high", "critical"]
    recommended_action: str

# Set response_format in llm_config — AG2 validates and parses automatically
llm_config = LLMConfig(
    model="gpt-4o-mini",
    response_format=TicketSummary   # enforces schema on every response
)

support_agent = ConversableAgent(
    name="SupportAgent",
    system_message="Analyze support tickets and generate structured summaries.",
    llm_config=llm_config
)
```

---

## AG2 — Production Best Practices

### Security
- Never hardcode API keys — use environment variables and secure credential stores
- Use `RegexGuardrail` for PII detection; `LLMGuardrail` for semantic safety filtering

### Reliability
- Configure fallback models in `config_list` to maintain uptime if primary model fails
- Set `temperature=0.0` for consistent structured outputs; `0.7–1.0` for creative tasks
- Implement rate limiting and robust error handling

### Agent Design
- Craft strong system messages defining each agent's **role and constraints**
- Set `max_consecutive_auto_reply` to prevent infinite loops
- Keep agents **specialized** — focused, well-defined tasks only

### Human-in-the-Loop Strategy
- Use oversight where human judgment adds most value (high-risk decisions)
- Define clear escalation criteria (risk levels, financial thresholds)
- Log all interventions for compliance and accountability

### Tool Design
- One focused purpose per tool
- Validate all inputs and outputs
- Document capabilities clearly so agents know when to use them
- Implement robust error handling for external dependencies

### Structured Output Best Practices
- Version schemas for long-term compatibility
- Include meaningful error messages for easier debugging
- Define schemas aligned with downstream system requirements

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **BeeAI** | Open-source, production-grade, enterprise-ready — Linux Foundation + IBM Research |
| **async/await** | Enables concurrent LLM calls — essential for multi-agent performance |
| **SimplePromptTemplate** | Mustache-style `{{variable}}` templates for consistent, reusable prompts |
| **`create_structure`** | Returns typed Pydantic objects — no parsing needed |
| **UnconstrainedMemory** | Stores all messages without limit — full conversational history |
| **RequirementAgent** | BeeAI's core agent class — persistent state, tools, requirements |
| **ConditionalRequirement** | Controls tool execution frequency and order |
| **AskPermissionRequirement** | Human-in-the-loop — approval required before tool call |
| **ThinkTool** | Enables explicit reasoning before acting |
| **ReAct in BeeAI** | `force_at_step=1` + `force_after=Tool` + `consecutive_allowed=False` |
| **HandoffTool** | Delegates tasks from coordinator to specialist agents |
| **AG2** | Open-source multi-agent framework — role-based, provider-agnostic |
| **ConversableAgent** | Base AG2 agent — all types inherit from this |
| **Two-agent chat** | `initiate_chat()` — simplest AG2 orchestration |
| **Sequential chat** | `initiate_chats()` with carryover — pipeline workflows |
| **Nested chat** | `register_nested_chats()` — encapsulates reusable sub-workflows |
| **Group chat** | `GroupChatManager` + speaker selection patterns |
| **ContextVariables** | Shared key-value memory across agents and tools |
| **OnCondition** | LLM-based routing; `OnContextCondition` = context-based routing |
| **Guardrails** | `RegexGuardrail` (pattern) + `LLMGuardrail` (semantic) for safety |
| **`is_termination_msg`** | Function-based chat termination condition |
| **AG2 temperature** | `0.0` for structured outputs; `0.7–1.0` for creative tasks |

---

*Notes based on: Course 08 Module 03 — Alternative Agentic Frameworks: BeeAI and AG2/AutoGen (Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI)*
