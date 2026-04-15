"""
Example 6: Multi-Agent Supervisor (Intermediate)
=================================================

Real-world tasks often need *specialised* agents working together under
coordination. This example implements the Supervisor pattern:

  - **Supervisor**  : the "manager" LLM — given the full history it decides
                      which worker to call next, or whether the job is done
  - **Researcher**  : gathers and summarises relevant information on a topic
  - **Writer**      : crafts a polished report from the research notes

The supervisor is not a separate process — it is just another node in the
graph. What makes it a "supervisor" is that every worker returns to it
after finishing, and the supervisor node uses conditional edges to decide
who goes next.

Graph structure:
                         ┌─────────────────┐
             ┌──────────►│  researcher_node │─────────────┐
             │           └─────────────────┘             │
  [START] ──►│                                            ▼
         supervisor_node ◄──────────────────────── (back to supervisor)
             │           ┌─────────────────┐             ▲
             ├──────────►│  writer_node     │─────────────┘
             │           └─────────────────┘
             │
             └──► (FINISH) ──► [END]

Key concepts introduced:
  - Multi-agent coordination    : multiple LLM-backed nodes with distinct roles
  - Supervisor pattern          : one node decides who acts next
  - Annotated state with reducer: using add_messages to accumulate agent outputs
  - Structured routing output   : supervisor outputs a typed next-step decision
  - Agent handoffs              : workers report back to supervisor via state

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/06_multi_agent_supervisor.py
"""

import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------
# We use Annotated[list, add_messages] so that every node can append
# messages without overwriting previous ones — the full history accumulates.
# The supervisor reads this history to understand what has been done so far.

class AgentState(TypedDict):
    task: str                                       # the user's original request
    messages: Annotated[list[BaseMessage], add_messages]  # full agent conversation
    research_notes: str                             # content produced by researcher
    report: str                                     # content produced by writer
    next: str                                       # supervisor's routing decision
    iterations: int                                 # guard against infinite loops


# ---------------------------------------------------------------------------
# 2. Supervisor Routing Schema
# ---------------------------------------------------------------------------
# Pydantic model constrains the supervisor's decision to valid node names,
# preventing hallucinated or malformed routing instructions.

WORKERS = ["researcher", "writer"]

class SupervisorDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description=(
            "Which agent to call next. "
            "Use 'researcher' to gather information, "
            "'writer' to produce the final report, "
            "'FINISH' when the report is complete and satisfactory."
        )
    )
    reasoning: str = Field(description="One sentence explaining this routing decision.")


supervisor_llm = llm.with_structured_output(SupervisorDecision)

MAX_ITERATIONS = 6   # safety cap — prevents runaway loops


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """
    The brain of the operation.

    Reads the full message history and decides:
      - Has enough research been gathered?  → send to writer
      - Has the writer produced a report?   → FINISH
      - Something is missing?               → route back to the right worker
    """
    print("\n[supervisor] Evaluating progress...")

    # Build a system prompt that describes available workers and current task
    system_prompt = (
        "You are a supervisor managing a two-agent research team.\n\n"
        "Agents available:\n"
        "  - researcher : searches and summarises information on a topic\n"
        "  - writer     : synthesises research into a polished report\n\n"
        "Workflow rules:\n"
        "  1. Always call 'researcher' first to gather material.\n"
        "  2. Once research is sufficient, call 'writer' to produce the report.\n"
        "  3. If the writer's report is complete and good quality, choose FINISH.\n"
        "  4. You may loop (researcher → writer → researcher …) if the report "
        "     reveals gaps that need more research — but avoid unnecessary loops.\n\n"
        f"Current task: {state['task']}"
    )

    # Pass the full conversation history so the supervisor sees all work done
    decision: SupervisorDecision = supervisor_llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    print(f"[supervisor] → next='{decision.next}' | reason: {decision.reasoning}")

    return {
        "next": decision.next,
        "iterations": state["iterations"] + 1,
    }


def researcher_node(state: AgentState) -> dict:
    """
    Gathers and organises information relevant to the task.

    Produces structured research notes that the writer can use directly.
    """
    print("[researcher] Researching topic...")

    system = (
        "You are a thorough research analyst. Given a task, produce concise but "
        "comprehensive research notes covering: key facts, context, statistics "
        "(simulated is fine), and 3-5 important points. Format with clear headings."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Research this topic thoroughly:\n{state['task']}"),
    ])

    research_notes = response.content
    print(f"[researcher] Produced {len(research_notes)} chars of notes.")

    # Append a message so the supervisor sees the research output in history
    agent_message = AIMessage(
        content=f"[RESEARCHER OUTPUT]\n{research_notes}",
        name="researcher",
    )
    return {
        "research_notes": research_notes,
        "messages": [agent_message],
    }


def writer_node(state: AgentState) -> dict:
    """
    Synthesises the research notes into a polished, reader-friendly report.
    """
    print("[writer] Writing report from research notes...")

    if not state["research_notes"]:
        # Safety fallback — should not happen with a well-prompted supervisor
        return {
            "messages": [AIMessage(
                content="[WRITER] No research notes available yet. Please run the researcher first.",
                name="writer",
            )]
        }

    system = (
        "You are a professional technical writer. Using the research notes provided, "
        "write a well-structured, engaging report (400-600 words). Include:\n"
        "  - An executive summary (2-3 sentences)\n"
        "  - Main body with 3-4 sections and subheadings\n"
        "  - A concise conclusion with key takeaways\n"
        "Write for a professional audience. Be clear and avoid fluff."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Research notes:\n{state['research_notes']}"
        )),
    ])

    report = response.content
    print(f"[writer] Report written ({len(report)} chars).")

    agent_message = AIMessage(
        content=f"[WRITER OUTPUT]\n{report}",
        name="writer",
    )
    return {
        "report": report,
        "messages": [agent_message],
    }


# ---------------------------------------------------------------------------
# 4. Router Function for Conditional Edges
# ---------------------------------------------------------------------------

def route_next(state: AgentState) -> str:
    """
    Translate the supervisor's decision into a node name (or END).

    Also enforces the MAX_ITERATIONS safety cap — if we've looped too many
    times, force a finish regardless of the supervisor's preference.
    """
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"[router] Max iterations ({MAX_ITERATIONS}) reached — forcing FINISH.")
        return END

    decision = state["next"]
    if decision == "FINISH":
        return END
    return decision  # "researcher" or "writer"


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor decides the next step
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {"researcher": "researcher", "writer": "writer", END: END},
    )

    # After every worker, control returns to the supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. Run It
# ---------------------------------------------------------------------------

def run_task(app, task: str) -> str:
    """Run a research + writing task and return the final report."""
    initial_state: AgentState = {
        "task": task,
        "messages": [HumanMessage(content=task)],
        "research_notes": "",
        "report": "",
        "next": "",
        "iterations": 0,
    }

    print(f"\n{'='*65}")
    print(f"TASK: {task}")
    print("="*65)

    final_state = app.invoke(initial_state)

    return final_state["report"] or "No report was produced."


if __name__ == "__main__":
    app = build_graph()

    tasks = [
        "Explain the key differences between LangGraph and LangChain, and when to use each.",
        "Summarise the benefits and challenges of adopting a microservices architecture.",
    ]

    for task in tasks:
        report = run_task(app, task)
        print(f"\n{'─'*65}")
        print("FINAL REPORT:")
        print("─"*65)
        print(report)
        print()

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a third worker: a "reviewer" that scores the writer's report and
    #    asks for revisions if score < 8/10.
    # 2. Give the researcher a real web-search tool (@tool + ToolNode) so it
    #    retrieves live information instead of relying on LLM knowledge.
    # 3. Use SqliteSaver as the checkpointer so you can inspect the full
    #    message history after the run:
    #      state = app.get_state({"configurable": {"thread_id": "1"}})
    #      for m in state.values["messages"]: print(m.name, ":", m.content[:100])
