# LangGraph Agent Gallery

A curated collection of practical examples and use cases demonstrating
the capabilities and applications of LangGraph agents — designed for
developers who are new to the framework.

---

## Examples at a Glance

### Beginner

| # | Example | Key Concept | LLM Required |
|---|---------|-------------|:---:|
| 01 | [Hello LangGraph](#01-hello-langgraph) | StateGraph, nodes, edges | No |
| 02 | [Simple Chatbot](#02-simple-chatbot) | MessagesState, conversation memory | Yes |
| 03 | [ReAct Agent with Tools](#03-react-agent-with-tools) | Tool calling, agentic loops | Yes |
| 04 | [Conditional Routing](#04-conditional-routing) | Branching, structured output | Yes |
| 05 | [Human-in-the-Loop](#05-human-in-the-loop) | interrupt(), checkpointing | Yes |

### Intermediate

| # | Example | Key Concept | LLM Required |
|---|---------|-------------|:---:|
| 06 | [Multi-Agent Supervisor](#06-multi-agent-supervisor) | Supervisor pattern, agent coordination | Yes |
| 07 | [Self-Reflection Agent](#07-self-reflection-agent) | Generate → Evaluate → Revise loop | Yes |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/ss-aman/langgraph-agent-gallery.git
cd langgraph-agent-gallery

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
# or create a .env file:
echo 'ANTHROPIC_API_KEY="your-key"' > .env
```

### 3. Run any example

```bash
# Beginner
python examples/01_hello_langgraph.py   # no API key needed
python examples/02_simple_chatbot.py
python examples/03_react_agent.py
python examples/04_conditional_routing.py
python examples/05_human_in_the_loop.py

# Intermediate
python examples/06_multi_agent_supervisor.py
python examples/07_self_reflection_agent.py
```

---

## Example Details

### 01: Hello LangGraph

**File:** `examples/01_hello_langgraph.py`

The simplest possible LangGraph program — no LLM required. A text string
flows through two nodes that modify it step by step.

```
[START] → greet_node → shout_node → [END]
```

**Teaches:** `StateGraph`, `TypedDict` state, nodes as plain functions,
`add_edge`, `compile()`, `invoke()`.

---

### 02: Simple Chatbot

**File:** `examples/02_simple_chatbot.py`

A multi-turn chatbot that remembers everything said in the conversation.
Uses LangGraph's built-in `MessagesState` so history management is automatic.

```
[START] → chatbot_node → [END]
              ↑ (history grows with each turn)
```

**Teaches:** `MessagesState`, `add_messages` reducer, `HumanMessage` /
`AIMessage`, multi-turn loops with `invoke()`.

---

### 03: ReAct Agent with Tools

**File:** `examples/03_react_agent.py`

An agent that can call Python tools (calculator, word counter, text reverser)
to answer questions it couldn't handle with reasoning alone. Demonstrates
the Reason → Act → Observe loop.

```
[START] → agent → (has tool calls?) → tool_node
              ↑____________________________|
              (loop until done → END)
```

**Teaches:** `@tool` decorator, `ToolNode`, `tools_condition`, `bind_tools()`,
agentic loops.

---

### 04: Conditional Routing

**File:** `examples/04_conditional_routing.py`

A customer-support triage bot that classifies incoming messages and routes
them to the right specialist handler — billing, technical, general, or
escalation.

```
[START] → classify → billing_handler  ┐
                   → technical_handler├→ [END]
                   → general_handler  │
                   → escalate_handler ┘
```

**Teaches:** `add_conditional_edges()`, router functions, structured LLM
output with Pydantic, branching and merging paths.

---

### 05: Human-in-the-Loop

**File:** `examples/05_human_in_the_loop.py`

A blog-post drafting workflow that pauses execution for human review after
each draft. The human can approve (publish) or request revisions (loop back).

```
[START] → draft_node → review_node ──(approve)──→ publish_node → [END]
              ↑              │
              └──(revise)────┘
```

**Teaches:** `interrupt()`, `MemorySaver` checkpointer, `Command(resume=...)`,
`thread_id` configuration, state persistence across Python calls.

---

### 06: Multi-Agent Supervisor

**File:** `examples/06_multi_agent_supervisor.py`  **Level:** Intermediate

A supervisor LLM orchestrates two specialist workers — a **Researcher** and a
**Writer** — to produce a polished report on any topic. Every worker returns
to the supervisor after finishing, and the supervisor decides the next step
using structured routing output.

```
                     ┌──────────────┐
         ┌──────────►│  researcher  │──────────────┐
         │           └──────────────┘              ▼
[START]─►supervisor◄──────────────────────── (back to supervisor)
         │           ┌──────────────┐              ▲
         ├──────────►│    writer    │──────────────┘
         │           └──────────────┘
         └──► FINISH ──► [END]
```

**Teaches:** supervisor pattern, multi-agent coordination, `Annotated` state
with `add_messages` reducer, structured routing with Pydantic, iteration guard,
agent handoff via message history.

---

### 07: Self-Reflection Agent

**File:** `examples/07_self_reflection_agent.py`  **Level:** Intermediate

A code-writing agent that generates Python code, **critiques its own output**
with a structured multi-dimension evaluator, and revises until the score meets
a quality threshold — or a maximum iteration cap is hit.

```
[START]
   │
generate_node ◄──────────────────────────────────┐
   │                                              │ (revise)
evaluate_node                                     │
   │                                              │
(quality_gate) ── score < PASS_SCORE ─────────────┘
   │
   └── score ≥ PASS_SCORE (or max iterations) ──► format_node ──► [END]
```

**Teaches:** reflection / self-critique pattern, structured multi-score
evaluation output, quality gates as conditional edges, iteration guards,
dynamic prompt construction based on prior feedback, history logging across
iterations.

---

## Core LangGraph Concepts

| Concept | Description |
|---------|-------------|
| `StateGraph` | The main class. Define nodes and edges, then `compile()`. |
| State (`TypedDict`) | Typed dict that flows through every node. |
| Node | A Python function `(state) → dict` that updates state. |
| Edge | A directed connection between two nodes. |
| Conditional Edge | Routes to different nodes based on a function's return value. |
| `MessagesState` | Built-in state with an `add_messages` reducer for chat. |
| `ToolNode` | Prebuilt node that executes LLM tool calls. |
| `interrupt()` | Pauses graph execution for human input. |
| Checkpointer | Persists state snapshots so graphs can pause and resume. |

---

## Learning Path

Work through the examples in order — each one builds on concepts from the last.

**Beginner**
1. **Example 01** — understand graphs, nodes, and edges with zero LLM distraction.
2. **Example 02** — add an LLM and learn how message history works.
3. **Example 03** — give the LLM tools and see the agentic loop in action.
4. **Example 04** — branch the graph based on LLM decisions.
5. **Example 05** — pause the graph and bring a human into the workflow.

**Intermediate**

6. **Example 06** — coordinate multiple specialised agents under a supervisor.
7. **Example 07** — make an agent critique and improve its own output iteratively.

---

## Requirements

- Python 3.11+
- `langgraph >= 0.2.0`
- `langchain-anthropic >= 0.1.0`
- `langchain-core >= 0.3.0`
- `python-dotenv >= 1.0.0`

Full list in `requirements.txt`.
