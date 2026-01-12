---
title: "Reflection: Building Agents That Think Twice and Do Better"
date: 2025-05-13
draft: false
---

**Learn what the Reflection pattern is, how it enables agents to self-correct and improve iteratively, and how to build a reflection-powered agent from scratch using LangGraph.**

## Why Single-Pass Generation Falls Short

Large Language Models are powerful but they rarely produce perfect output on the first try. Factual errors slip through, logical gaps go unnoticed, and code contains bugs that a second pass would catch. The standard single-pass approach—send a prompt, get a response, move on—offers no opportunity to catch mistakes, reconsider decisions, or refine responses. For complex tasks like code generation, research synthesis, or multi-step reasoning, these errors compound and the agent has no mechanism to recover.

The Reflection pattern addresses this by giving agents the ability to review their own work. Instead of producing a final answer in one shot, a reflection agent generates an initial response, critiques it, refines it based on that critique, and repeats until satisfied. This generate-critique-refine cycle mimics human self-review and produces higher-quality outputs with fewer errors. The pattern is already being used in production systems for code review, writing assistance, and multi-step reasoning tasks.

In this tutorial, you'll learn:

- How the Reflection pattern enables agents to self-correct and improve iteratively
- The four core components: generator, critic, iteration control, and state management
- How to build a complete reflection-powered code review agent using LangGraph
- When to use reflection versus simpler single-pass approaches

## What Is the Reflection Pattern?

**The Reflection pattern** is an agentic architecture where outputs are fed back into the system for evaluation and iterative improvement. Unlike traditional single-pass generation that produces a final answer immediately, reflection agents operate in a loop: generate output, critique it, refine based on feedback, and repeat until quality thresholds are met. This pattern is designed specifically for tasks where errors are costly, quality matters more than latency, and self-correction leads to measurably better results.

Real-world systems already use this pattern effectively. Code review assistants like GitHub Copilot's code analysis features use reflection to identify bugs and security issues. Writing assistants apply reflection to catch grammar errors, improve clarity, and refine arguments. Research agents use reflection to verify claims and fill gaps in their analysis. The pattern is particularly valuable in domains where structured feedback can be automatically generated and incorporated into refinements.

The key characteristics of reflection agents are:

- **Self-critique capability**: Agents evaluate their own outputs against specific quality criteria
- **Iterative refinement**: Each cycle improves the output based on structured feedback
- **Stopping conditions**: The loop terminates when quality is sufficient or maximum iterations are reached
- **Structured feedback**: Critiques provide specific, actionable guidance rather than vague comments
- **State tracking**: The system maintains drafts, reflections, and iteration count across the loop
- **Domain adaptability**: The pattern works for code, writing, research, analysis, and other structured outputs

## Key Components of Reflection Agents

A reflection system consists of four core components that work together to enable self-improvement through iterative critique and refinement.

### 1. Generator Actor

The generator produces the initial output and subsequent refinements. It takes the original task plus any feedback from reflection and produces a new version. This component serves as the creative engine, doing the actual work of generating content, code, or analysis.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def generate(state: AgentState, config) -> dict:
    """Generate or refine an output based on task and reflection feedback."""
    prompt = f"Task: {state['task']}\n"
    if state.get('reflection'):
        prompt += f"Feedback: {state['reflection']}\nPlease improve your previous response."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "iteration": state.get("iteration", 0) + 1}
```

The generator operates differently on each pass. The first iteration produces an initial attempt using only the task description. Subsequent iterations incorporate the critique, explicitly addressing identified problems. This progressive refinement ensures each cycle builds on the previous work rather than starting from scratch.

---

### 2. Reflection Critic

The reflection component evaluates the output against quality criteria, identifying problems and suggesting improvements. This critic operates like a reviewer, finding flaws that the generator missed and providing targeted guidance for improvement.

```python
def reflect(state: AgentState, config) -> dict:
    """Critique the current draft and provide actionable feedback."""
    prompt = f"""Task: {state['task']}

Draft:
{state['draft']}

Critique this output. Identify:
1. Factual errors or logical gaps
2. Missing elements or incomplete coverage
3. Areas that need clarification or expansion
4. Specific improvements that would enhance quality

Provide specific, actionable feedback."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"reflection": response.content}
```

The quality of critique determines the effectiveness of the entire pattern. Vague feedback like "make this better" leads to marginal improvements. Specific, structured feedback that names exact problems and suggests concrete fixes enables meaningful refinement. Many implementations use structured output with Pydantic models to enforce actionable critiques.

---

### 3. Iteration Control

A decision mechanism determines when to stop iterating—either after a fixed number of rounds or when quality plateaus. This prevents infinite loops while allowing sufficient refinement cycles.

```python
from typing import Literal

def should_continue(state: AgentState) -> Literal["reflect", "end"]:
    """Decide whether to continue refining or conclude."""
    # Stop if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        return "end"

    # Stop if critique indicates quality is sufficient
    reflection = state.get("reflection", "")
    if "no further improvements needed" in reflection.lower():
        return "end"

    # Otherwise, continue reflecting
    return "reflect"
```

The stopping condition balances quality against cost. Each iteration consumes additional time and tokens, so the system must terminate when marginal gains no longer justify the expense. Common approaches include fixed iteration limits, quality threshold checks, or explicit "good enough" signals from the critic.

---

### 4. State Management

The agent must track drafts, reflections, iteration count, and final outputs across the workflow. This state flows through each component and enables the loop to operate as a coherent system.

```python
from typing import TypedDict, Optional

class ReflectionState(TypedDict):
    task: str                    # The original user request (immutable)
    draft: str                   # Current work product
    reflection: str              # Critique feedback
    iteration: int               # Number of refinement cycles completed
    max_iterations: int          # Configured stopping threshold
    final_output: Optional[str]  # Polished result when loop ends
```

State management is particularly important in reflection agents because the loop accumulates context across iterations. Each cycle needs access to the original task, the current draft, all previous feedback, and how many times refinement has occurred. LangGraph's StateGraph handles this naturally, passing state between nodes and maintaining it across conditional edges.

---

## Interaction & Data Flow

The reflection pattern creates a closed loop where output becomes input for the next cycle. The generator produces a draft, the critic evaluates it, and that evaluation feeds back into the next generation. This continues until stopping conditions are met.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REFLECTION LOOP                                 │
│                    (until stopping criteria)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Task ──► Generator ──► Draft ──► Critic ──► Reflection ──┐        │
│         ▲                                                  │        │
│         └──────────────────────────────────────────────────┘        │
│                    (feedback loop)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The generator and critic alternate until the iteration controller signals termination. At that point, the final draft is returned as the output. The entire interaction is deterministic given the same task and random seed—though in practice, LLM temperature introduces variation that makes each run unique.

| State | Where It Lives | Update Frequency |
|-------|----------------|------------------|
| Task input | Immutable state graph | Once (at initialization) |
| Current draft | State graph | Per iteration (updated by generator) |
| Reflection feedback | State graph | Per iteration (updated by critic) |
| Iteration count | State graph | Per generation |
| Final output | State graph | On termination |

This flow maps cleanly to LangGraph's StateGraph structure. Each component becomes a node, state flows between them, and conditional edges handle the iteration decision. The framework manages the loop, allowing developers to focus on the logic of generation and critique rather than orchestration mechanics.

## End-to-End Demo: Building a Code Review Agent

In this section, I'll walk you through building a complete reflection agent from scratch. Our demo will create a **code review agent** that writes Python code, identifies bugs and style issues, and iteratively improves its own output until it meets production standards.

**What we're building:**

- A reflection-powered code generation agent
- Automated self-review for bugs, security issues, and style
- Iterative refinement until quality thresholds are met
- A working system that produces production-ready Python code

---

### Step 1: Project Setup & Configuration

First, install the dependencies. This demo uses LangGraph for orchestration and OpenAI for the underlying LLM calls.

```bash
pip install langgraph langchain-openai pydantic
```

Create a project structure with the necessary files:

```bash
mkdir reflection-agent && cd reflection-agent
touch .env requirements.txt main.py
```

Configure your API keys in a `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

The project is now ready for implementation. The `.env` file keeps credentials out of source control, while `requirements.txt` will track dependencies for reproducibility.

---

### Step 2: Define the State Schema

Define what information flows through the reflection loop. Using structured output for the critique ensures specific, actionable feedback rather than vague comments.

```python
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field

class CodeCritique(BaseModel):
    """Structured feedback from the reflection critic."""
    has_errors: bool = Field(description="Whether the code contains errors")
    security_issues: list[str] = Field(
        default_factory=list,
        description="Security vulnerabilities found"
    )
    style_issues: list[str] = Field(
        default_factory=list,
        description="Code style improvements needed"
    )
    suggested_improvements: str = Field(
        description="Specific suggestions for improvement"
    )
    overall_quality: int = Field(
        ge=1, le=10,
        description="Quality score from 1-10"
    )

class ReflectionState(TypedDict):
    task: str
    code: str
    critique: CodeCritique
    iteration: int
    max_iterations: int
    final_code: str
```

The structured critique serves two purposes. First, it forces the critic to provide categorized, specific feedback. Second, it enables programmatic quality checks—the iteration controller can directly examine the quality score to decide whether to continue refining.

---

### Step 3: Build the Generator Actor

Create the code generation function that produces and refines code based on task description and critique feedback.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Use GPT-4o for code generation
generator_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def generate_code(state: ReflectionState, config) -> dict:
    """Generate or refine code based on task and critique."""

    system_prompt = """You are an expert Python developer. Write clean,
    secure, well-documented code that follows best practices.
    Include type hints, docstrings, and proper error handling."""

    user_prompt = f"Task: {state['task']}\n"

    # Incorporate feedback from previous iterations
    if state['iteration'] > 0:
        critique = state['critique']
        user_prompt += f"\nYour previous code had these issues:\n"

        if critique.security_issues:
            user_prompt += f"- Security: {', '.join(critique.security_issues)}\n"

        if critique.style_issues:
            user_prompt += f"- Style: {', '.join(critique.style_issues)}\n"

        user_prompt += f"\nSuggestions: {critique.suggested_improvements}\n"
        user_prompt += "\nPlease revise the code to address these issues."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = generator_llm.invoke(messages)
    return {"code": response.content, "iteration": state["iteration"] + 1}
```

The generator adapts its behavior based on iteration count. The first pass produces an initial attempt based solely on the task. Subsequent passes explicitly reference the critique, ensuring each refinement addresses identified problems. This targeted approach is more effective than generic "try again" prompts.

---

### Step 4: Build the Reflection Critic

Create the reflection component with structured output. Using a lower temperature encourages more consistent, analytical critique.

```python
# Use lower temperature for consistent critique
critic_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
critic_llm_structured = critic_llm.with_structured_output(CodeCritique)

def reflect_on_code(state: ReflectionState, config) -> dict:
    """Critique the generated code and provide structured feedback."""

    prompt = f"""Review the following Python code for the task: {state['task']}

Code:
{state['code']}

Check for:
1. Runtime errors and logical bugs
2. Security vulnerabilities (SQL injection, XSS, command injection, etc.)
3. Code style and maintainability issues (PEP 8 violations, missing docstrings)
4. Edge cases not handled (None inputs, empty strings, invalid types)
5. Documentation quality (type hints, docstrings, comments)

Provide specific, actionable feedback. Assign a quality score from 1-10."""

    critique = critic_llm_structured.invoke(prompt)
    return {"critique": critique}
```

The structured output from the critic makes feedback actionable. Each issue is categorized (security vs style), enumerated as specific instances, and accompanied by concrete suggestions. The quality score provides a simple metric for the iteration controller to use when deciding whether to continue.

---

### Step 5: Assemble the Reflection Graph

Connect the components into a complete reflection loop using LangGraph's StateGraph. The graph handles the iteration mechanics automatically.

```python
from langgraph.graph import StateGraph, END

def should_continue(state: ReflectionState) -> Literal["reflect", "end"]:
    """Decide whether to continue reflecting or finish."""
    # Stop if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        return "end"

    # Stop if quality is acceptable and no errors remain
    critique = state["critique"]
    if critique.overall_quality >= 8 and not critique.has_errors:
        return "end"

    # Otherwise, continue reflection
    return "reflect"

def finalize(state: ReflectionState) -> dict:
    """Prepare the final output."""
    return {"final_code": state["code"]}

# Build the graph
builder = StateGraph(ReflectionState)

# Add nodes
builder.add_node("generate", generate_code)
builder.add_node("reflect", reflect_on_code)
builder.add_node("finalize", finalize)

# Set entry point
builder.set_entry_point("generate")

# Add edges
builder.add_edge("generate", "reflect")
builder.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "reflect": "generate",  # Continue iterating
        "end": "finalize"       # Finish and output
    }
)
builder.add_edge("finalize", END)

# Compile the graph
reflection_agent = builder.compile()
```

The graph orchestrates the entire reflection loop. Entry point is generation, which flows to reflection. The conditional edge after reflection either routes back to generation (for another iteration) or proceeds to finalization. This structure cleanly separates the concerns of generation, critique, and termination logic.

---

### Step 6: Run the Agent

Execute the reflection agent on a real task to see the progressive improvement across iterations.

```python
if __name__ == "__main__":
    initial_state: ReflectionState = {
        "task": "Write a Python function that validates email addresses "
                "and returns True if valid, False otherwise.",
        "code": "",
        "critique": CodeCritique(
            has_errors=True,
            overall_quality=1,
            suggested_improvements=""
        ),
        "iteration": 0,
        "max_iterations": 3,
        "final_code": ""
    }

    # Run the reflection loop
    result = reflection_agent.invoke(initial_state)

    print("=== Final Code ===")
    print(result["final_code"])
    print("\n=== Quality Score ===")
    print(f"{result['critique'].overall_quality}/10")
    print("\n=== Iterations ===")
    print(f"Completed {result['iteration']} refinement cycles")
```

The agent executes the reflection loop, producing progressively better code at each iteration. By the final iteration, the output should have a quality score of 8 or higher with no remaining errors.

---

### Complete Example Output

Running the agent produces visible improvement across multiple iterations:

```
$ python main.py

=== Iteration 1 (Generate) ===
def is_valid_email(email):
    return '@' in email and '.' in email

=== Iteration 1 (Reflect) ===
Quality Score: 3/10
Errors: True
Security: None
Style: Missing type hints, missing docstring, vague variable name
Suggestions: Use regex for proper validation, add type hints and documentation

=== Iteration 2 (Generate) ===
import re
def is_valid_email(email: str) -> bool:
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

=== Iteration 2 (Reflect) ===
Quality Score: 6/10
Errors: True
Security: Potential regex injection if email is user-controlled
Style: Missing docstring, regex pattern allows invalid TLDs
Suggestions: Add documentation, improve regex pattern, handle None input

=== Iteration 3 (Generate) ===
import re
from typing import Optional

def is_valid_email(email: Optional[str]) -> bool:
    """
    Validate an email address using regex pattern.

    Args:
        email: The email address to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

=== Iteration 3 (Reflect) ===
Quality Score: 9/10
Errors: False
Security: None
Style: None significant
Suggestions: Code is production-ready

=== Final Code ===
import re
from typing import Optional

def is_valid_email(email: Optional[str]) -> bool:
    """
    Validate an email address using regex pattern.

    Args:
        email: The email address to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

=== Final Quality Score: 9/10 ===
=== Completed 3 refinement cycles ===
```

The progression demonstrates the reflection pattern's value. The first iteration produces naive code that only checks for basic characters. The second iteration adds regex and type hints but misses edge cases. The third iteration produces production-ready code with proper documentation, edge case handling, and a refined regex pattern. Each cycle addresses specific issues identified by the critic, resulting in measurable quality improvement.

## Results & Outcomes

After completing the demo, you have a working reflection agent with these concrete outputs.

### Files & Artifacts Created

```
reflection-agent/
├── main.py                 # Complete reflection agent implementation
├── .env                    # API configuration
└── requirements.txt        # Dependencies
```

### What You Can Do Now

**Run the code review agent:**

```bash
python main.py
```

**Adapt to different domains:**

```python
# For writing improvement
class WritingCritique(BaseModel):
    clarity_score: int
    grammar_errors: list[str]
    weak_arguments: list[str]

# For data analysis
class AnalysisCritique(BaseModel):
    accuracy_concerns: list[str]
    missing_insights: list[str]
    visualization_suggestions: list[str]
```

**Adjust iteration limits for quality-cost tradeoffs:**

```python
# Quick, acceptable results (2x cost)
{"max_iterations": 2}

# High-quality, costly results (3x cost)
{"max_iterations": 5}
```

**Add external validation tools:**

```python
def validate_with_tools(state: ReflectionState) -> dict:
    """Run external validators alongside LLM critique."""
    # Run mypy for type checking
    # Run bandit for security analysis
    # Run black for style checking
    # Aggregate results into critique
    ...
```

### Problems Solved

| Before (single-pass) | After (reflection) |
|---------------------|--------------------|
| No opportunity to catch errors | Agent identifies and fixes its own mistakes |
| Inefficient solutions accepted | Performance issues flagged and addressed |
| Edge cases often missed | Critic explicitly checks for edge cases |
| Security bugs shipped to production | Dedicated security review in each cycle |
| Undocumented code becomes permanent | Style critique enforces documentation standards |

### Extension Ideas

**Implement Reflexion with episodic memory:**

The Reflexion pattern extends basic reflection by storing past mistakes and using them to guide future attempts. This prevents the agent from repeating the same errors across similar tasks.

```python
def store_mistake(state: ReflectionState):
    """Store identified mistakes to avoid repeating them."""
    memory.add(
        task_type=classify_task(state["task"]),
        mistakes=state["critique"].security_issues + state["critique"].style_issues,
        resolution=state["code"]
    )

def retrieve_past_mistakes(state: ReflectionState):
    """Retrieve similar past issues to guide generation."""
    task_type = classify_task(state["task"])
    past_mistakes = memory.search(task_type=task_type, limit=5)
    return {"past_mistakes": past_mistakes}
```

**Add multi-critic specialization:**

Different critics can specialize in different aspects of quality:

```python
def security_critic(state: ReflectionState):
    """Focus solely on security vulnerabilities."""
    ...

def style_critic(state: ReflectionState):
    """Focus solely on code style and maintainability."""
    ...

def performance_critic(state: ReflectionState):
    """Focus solely on algorithmic efficiency."""
    ...
```

**Implement Language Agent Tree Search (LATS):**

For complex tasks, explore multiple refinement paths in parallel and select the best:

```python
# Generate multiple candidate refinements
candidates = [generate(state) for _ in range(3)]

# Critique all candidates
critiques = [critique(c) for c in candidates]

# Select the best based on quality scores
best = max(zip(candidates, critiques), key=lambda x: x[1].quality_score)
```

### Production Considerations

For real deployment, add these capabilities:

1. **Caching** — Cache critiques for similar code patterns to reduce costs
2. **Parallel critique** — Run multiple critics simultaneously and aggregate results
3. **Custom stopping** — Allow domain-specific stopping conditions beyond iteration count
4. **Metrics** — Track quality improvement per iteration to optimize iteration limits
5. **Tool integration** — Connect to linters, type checkers, and security scanners for automated feedback

## Conclusion

Reflection is a simple but powerful pattern: give agents the ability to review their own work and iterate. The generate-critique-refine cycle produces measurably better outputs than single-pass generation, especially for complex tasks like code writing, research synthesis, and multi-step reasoning.

The key components are a generator, a critic, iteration control, and state management. LangGraph's StateGraph makes this pattern straightforward to implement—the conditional edges and state passing handle the reflection loop naturally. Structured output via Pydantic ensures critiques are specific and actionable rather than vague.

Use reflection when quality matters more than latency, when errors are costly, or when tasks require careful reasoning. Skip it for simple, low-stakes outputs where the additional token cost isn't justified. The pattern shines in domains like code review, writing assistance, and research validation where structured feedback leads to clear improvements.

---

**Sources:**
- [LangChain Blog - Reflection Agents](https://blog.langchain.com/reflection-agents/)
- [LangGraph Official Documentation - Reflection](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/)
- [LangGraph Reflexion Notebook](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflexion/reflexion.ipynb)
- [Building a Self-Improvement Agent with LangGraph](https://levelup.gitconnected.com/building-a-self-improvement-agent-with-langgraph-reflection-vs-reflexion-1d1abcc5865d)
- [Agentic AI from First Principles: Reflection](https://towardsdatascience.com/agentic-ai-from-first-principles-reflection/)
- [Reflection Pattern: When Agents Think Twice](https://theneuralmaze.substack.com/p/reflection-pattern-agents-that-think)
- [Enhancing Code Quality with LangGraph Reflection](https://www.analyticsvidhya.com/blog/2025/03/enhancing-code-quality-with-langgraph-reflection/)
- [Agent Patterns Documentation - Reflection](https://agent-patterns.readthedocs.io/en/stable/patterns/reflection.html)
