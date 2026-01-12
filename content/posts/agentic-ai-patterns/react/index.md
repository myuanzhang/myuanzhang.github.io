---
title: "ReAct: Reasoning + Acting = Smarter Agents"
date: 2025-10-05
draft: false
---

**Learn what the ReAct pattern is, how it combines reasoning and acting to solve complex tasks, and how to build a ReAct agent from scratch using LangGraph.**

## Why Pure Tool-Using Agents Fall Short

Basic tool-calling agents have a fundamental weakness: they act without thinking. When given access to tools like web search or calculators, these agents often make poor decisions—calling the wrong tool, passing malformed arguments, or getting stuck in infinite loops. The problem isn't the tools themselves; it's the lack of explicit reasoning before each action. A human wouldn't click a button without first considering why, but traditional agents do exactly that.

The ReAct pattern (Reasoning + Acting) solves this by interleaving thought and action. Before taking any action, the agent explicitly verbalizes its reasoning. After seeing the result, it reflects on what it learned before deciding what to do next. This simple shift—from impulsive action to deliberate reasoning—dramatically improves reliability on multi-step tasks. The pattern, introduced in the 2022 paper ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629), has become the foundation for most production agent systems today.

In this tutorial, you'll learn:

- How the ReAct pattern combines reasoning and action in a unified loop
- The three core components: prompt template, tool belt, and agent loop
- How to build a complete ReAct agent using LangGraph from scratch
- How to inspect and debug reasoning traces to understand agent behavior

## What Is the ReAct Pattern?

**ReAct** is an agentic pattern that synergizes two capabilities: **Reasoning** (thinking through problems step by step) and **Acting** (taking actions through tools). Unlike pure tool-calling agents that immediately execute tool calls, ReAct agents follow a structured loop: Thought → Action → Observation → repeat. Each action is preceded by explicit reasoning, and each observation informs the next thought.

The pattern was introduced by Shunyu Yao et al. in 2022 and has since become the standard architecture for language model agents. LangGraph's `create_react_agent` implements this pattern directly, providing a production-ready framework for building ReAct-style agents without manually constructing the underlying StateGraph.

The key characteristics of ReAct agents are:

- **Explicit reasoning traces**: Every action is preceded by a thought explaining why
- **Action-observation loop**: Tools produce observations that inform subsequent reasoning
- **Self-correction**: Agents can recover from errors by reasoning about failed actions
- **Interleaved reasoning and acting**: Thought and action happen in the same loop, not as separate phases
- **Tool awareness**: Agents explicitly choose which tool to use based on the current context
- **Multi-step problem solving**: Complex tasks are broken down into reasoned steps

## Key Components of ReAct Agents

A ReAct system consists of three core components that work together to create the reasoning-acting loop.

### 1. ReAct Prompt Template

The prompt template structures the LLM's output to enforce the ReAct pattern. It explicitly tells the model to generate thoughts before actions and to use specific formatting for each step.

```python
from langchain_core.prompts import ChatPromptTemplate

# The ReAct prompt structure
react_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions by reasoning and acting.

Follow this format:
Thought: your reasoning about what to do next
Action: the action to take (should be one of: {tool_names})
Action Input: the input for the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Thought: {agent_scratchpad}"""),
    ("placeholder", "{messages}"),
])
```

This prompt serves two purposes. First, it establishes the output format—Thought, Action, Action Input—that the agent must follow. Second, it provides a scratchpad where the agent's reasoning trace accumulates across iterations. Each thought-action-observation cycle gets appended to this scratchpad, giving the model full context of its reasoning history.

---

### 2. Tool Belt

The tool belt is the collection of actions the agent can take. Each tool is a Python function with a name, description, and schema—information that helps the LLM decide when and how to use it.

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information about a query."""
    # In production, this would call a real search API
    if "langgraph" in query.lower():
        return "LangGraph is a framework for building stateful agents."
    return "No relevant information found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {e}"

tools = [search, calculator]
tool_names = ", ".join([tool.name for tool in tools])
```

The tool description is particularly important—it's the only information the LLM has about what each tool does. Good descriptions are specific, describe when to use the tool, and mention any important constraints. LangGraph's `ToolNode` automatically executes these tools when called by the agent, handling parallel execution and error handling.

---

### 3. Agent Loop (StateGraph Orchestration)

The agent loop coordinates the interaction between the model and tools. LangGraph's StateGraph manages this through conditional edges that route between reasoning (calling the model) and acting (executing tools).

```python
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM made tool calls, continue to tools
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return "end"

def call_model(state: AgentState, config):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

The graph creates a cycle: agent → tools → agent → tools. After each model call, the conditional edge checks if the model generated tool calls. If yes, route to the tools node; if no, exit to END. After tools execute, the graph always routes back to the agent, giving the model a chance to reason about the observations and decide what to do next.

---

## Interaction & Data Flow

The ReAct pattern creates a reasoning-action loop where each step builds on the previous one. The model thinks, acts, observes, and repeats until it has enough information to answer.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REACT LOOP                                   │
│                    (until Final Answer)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User Question ──► Agent ──► Thought + Action ──► Tools ──┐        │
│         ▲                  │                              │         │
│         └──────────────────┴────◄─────── Observation ◄────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The messages in the state accumulate across iterations, creating a complete trace of the agent's reasoning process:

| Iteration | Message Type | Content |
|-----------|--------------|---------|
| 1 | HumanMessage | User's question |
| 1 | AIMessage | Thought + tool call |
| 1 | ToolMessage | Tool result |
| 2 | AIMessage | Thought about result + next tool call |
| 2 | ToolMessage | Second tool result |
| 3 | AIMessage | Thought + Final Answer |

This trace is valuable for debugging and understanding agent behavior. You can see exactly what the agent was thinking at each step and why it made the decisions it did.

## End-to-End Demo: Building a Research Agent

In this section, I'll walk you through building a complete ReAct agent from scratch. Our demo will create a **research agent** that can search the web, perform calculations, and answer complex multi-step questions with full reasoning transparency.

**What we're building:**

- A ReAct agent with web search and calculator tools
- Explicit reasoning traces showing thought process
- A working system that answers multi-step questions

---

### Step 1: Project Setup & Configuration

First, install the dependencies for LangGraph and LangChain:

```bash
pip install langgraph langchain-openai langchain-core
```

Create a project structure:

```bash
mkdir react-agent && cd react-agent
touch .env requirements.txt main.py
```

Configure your API keys in `.env`:

```env
OPENAI_API_KEY=your-key-here
```

For this demo, we'll use a mock search tool instead of a real search API to avoid additional setup. In production, you would integrate with a service like Tavily or SerpAPI.

---

### Step 2: Define Tools

Create the tools that our agent will have access to:

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a topic.

    Use this for factual questions, current events, or information
    not in your training data.
    """
    # Mock search results
    knowledge = {
        "langgraph": "LangGraph is a framework for building stateful, multi-actor applications with LLMs.",
        "react": "ReAct (Reasoning + Acting) is a paradigm that synergizes reasoning and acting in language models.",
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "llm": "Large Language Models are AI systems trained on vast amounts of text data.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No specific information found for '{query}'. Try a more specific query."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Use this for any calculation. Supports Python math syntax.
    Examples: '2 + 2', '10 * 5', '100 / 4'
    """
    try:
        # Safe evaluation of basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

tools = [search, calculator]
```

The tool descriptions are critical—they tell the agent when to use each tool. Notice that `search` is described for factual questions while `calculator` is for mathematical expressions. This helps the agent choose correctly.

---

### Step 3: Create the ReAct Agent

Build the complete ReAct agent using LangGraph's StateGraph:

```python
import os
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize model
model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

# Define nodes
def call_model(state: AgentState, config):
    """Call the LLM with current messages."""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Create tool node
tool_node = ToolNode(tools)

# Define routing logic
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue using tools."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

This creates a complete ReAct loop. The agent thinks, routes to tools if needed, receives observations, and repeats until it has a final answer.

---

### Step 4: Run the Agent and Inspect Traces

Execute the agent and print the full reasoning trace:

```python
def pretty_print_messages(messages):
    """Print all messages with their reasoning traces."""
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        print(f"\n{'='*60}")
        print(f"[Step {i+1}] {msg_type}")
        print('='*60)

        if isinstance(msg, HumanMessage):
            print(f"👤 User: {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"🧠 Thought: {msg.content or 'I need to use a tool.'}")
                for tool_call in msg.tool_calls:
                    print(f"🔧 Action: {tool_call['name']}")
                    print(f"   Input: {tool_call['args']}")
            else:
                print(f"✅ Final Answer: {msg.content}")

        elif isinstance(msg, ToolMessage):
            print(f"📊 Observation: {msg.content}")

if __name__ == "__main__":
    question = "What is LangGraph and how many letters are in its full name?"

    print("🚀 Starting ReAct Agent")
    print(f"📝 Question: {question}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })

    pretty_print_messages(result["messages"])
```

---

### Step 5: Complete Example Output

Running the agent produces a transparent reasoning trace:

```
🚀 Starting ReAct Agent
📝 Question: What is LangGraph and how many letters are in its full name?

============================================================
[Step 1] HumanMessage
============================================================
👤 User: What is LangGraph and how many letters are in its full name?

============================================================
[Step 2] AIMessage
============================================================
🧠 Thought: I need to search for information about LangGraph to find its full name.
🔧 Action: search
   Input: {'query': 'LangGraph full name framework'}

============================================================
[Step 3] ToolMessage
============================================================
📊 Observation: LangGraph is a framework for building stateful, multi-actor applications with LLMs.

============================================================
[Step 4] AIMessage
============================================================
🧠 Thought: The search result tells me what LangGraph is, but doesn't explicitly state a "full name" beyond "LangGraph". Let me count the letters in "LangGraph" itself.
🔧 Action: calculator
   Input: {'expression': 'len("LangGraph")'}

============================================================
[Step 5] ToolMessage
============================================================
📊 Observation: 9

============================================================
[Step 6] AIMessage
============================================================
✅ Final Answer: LangGraph is a framework for building stateful, multi-actor applications with LLMs. The name "LangGraph" has 9 letters.
```

The trace shows the agent's complete reasoning process:
1. It identifies what information it needs
2. It chooses the right tool (search)
3. It evaluates the result and realizes it needs a count
4. It uses the calculator tool
5. It combines both pieces of information into a final answer

## Results & Outcomes

After completing the demo, you have a working ReAct agent with these concrete outputs.

### Files & Artifacts Created

```
react-agent/
├── main.py                 # Complete ReAct agent implementation
├── .env                    # API configuration
└── requirements.txt        # Dependencies
```

### What You Can Do Now

**Run the agent with new questions:**

```python
questions = [
    "What is Python and what's 100 divided by 4?",
    "Search for information about LLMs and calculate 15 times 3.",
    "What is ReAct and how many seconds are in an hour?"
]

for q in questions:
    result = app.invoke({"messages": [HumanMessage(content=q)]})
    pretty_print_messages(result["messages"])
```

**Add more tools to expand capabilities:**

```python
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time for a timezone."""
    from datetime import datetime
    return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"

tools.append(get_current_time)
# Rebuild the graph with updated tools
```

**Use the prebuilt helper for faster development:**

```python
from langgraph.prebuilt import create_react_agent

# One-line ReAct agent creation
app = create_react_agent(model, tools)
```

### Problems Solved

| Before (pure tool-calling) | After (ReAct) |
|---------------------------|---------------|
| Tools called without context | Each action preceded by explicit reasoning |
| Errors cascade without recovery | Agent reflects on failures and retries |
| Multi-step tasks often fail | Complex tasks broken into reasoned steps |
| No visibility into decision process | Full trace of thoughts and actions |
| Wrong tools selected frequently | Tool choices explained in reasoning |

### Extension Ideas

**Add memory for conversational context:**

```python
from langgraph.checkpoint.memory import MemorySaver

# Add persistent memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Use with thread_id for conversation history
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [HumanMessage(content="...")]}, config)
```

**Implement human-in-the-loop for approval:**

```python
from langgraph.types import interrupt

def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])

    # Ask for human approval on tool calls
    if response.tool_calls:
        interrupt("Should I proceed with these tool calls?")

    return {"messages": [response]}
```

**Create specialized tool categories:**

```python
# Group tools by category
search_tools = [search, wikipedia]
math_tools = [calculator, unit_converter]
code_tools = [python_repl, code_executor]

# Route to appropriate category based on query
def categorize_query(query: str) -> str:
    if "calculate" in query.lower():
        return "math"
    elif "search" in query.lower():
        return "search"
    return "general"
```

### Production Considerations

For production deployments, consider these additions:

1. **Rate limiting** — Prevent tool abuse with rate limits and cost tracking
2. **Tool result validation** — Verify tool outputs before passing them back to the model
3. **Error recovery** — Implement specialized handling for common tool failures
4. **Streaming responses** — Stream intermediate reasoning steps for better UX
5. **Observability** — Log all reasoning traces for debugging and analytics

## Conclusion

ReAct is a simple but powerful pattern: make agents think before they act. By interleaving explicit reasoning with tool execution, ReAct agents make better decisions, recover from errors more gracefully, and provide transparent explanations of their behavior. The three core components—prompt template, tool belt, and agent loop—are straightforward to implement using LangGraph's StateGraph or the prebuilt `create_react_agent` helper.

Use ReAct when you need transparent reasoning, multi-step problem solving, or reliable tool use. Skip it for simple one-shot queries where the overhead of explicit reasoning isn't justified. The pattern shines in research assistants, data analysis agents, and any application where understanding *why* the agent made a decision matters as much as the decision itself.

---

**Sources:**
- [LangGraph Official Documentation - ReAct Agent](https://langchain-ai.github.io/langgraph/)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Build ReAct AI Agents with LangGraph](https://medium.com/@tahirbalarabe2/build-react-ai-agents-with-langgraph-cb9d28cc6e20)
- [Building ReAct Agents with LangGraph: A Beginner's Guide](https://machinelearningmastery.com/building-react-agents-with-langgraph-a-beginners-guide/)
- [ReAct agent from scratch with Gemini 2.5 and LangGraph](https://ai.google.dev/gemini-api/docs/langgraph-example)
- [What is a ReAct Agent? | IBM](https://www.ibm.com/think/topics/react-agent)
- [Building Production ReAct Agents From Scratch](https://www.decodingai.com/p/building-production-react-agents)
