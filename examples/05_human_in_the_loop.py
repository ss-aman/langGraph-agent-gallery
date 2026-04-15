"""
Example 5: Human-in-the-Loop (Approval Workflow)
=================================================

Fully autonomous agents are powerful, but sometimes you need a human to
review or approve before the agent proceeds — think content moderation,
financial transactions, or any high-stakes action.

This example builds a blog-post drafting workflow where:
  1. The agent drafts a blog post
  2. Execution PAUSES and waits for human review
  3. The human either approves or requests edits
  4. If approved → the post is "published"
     If rejected → the agent revises and asks again

Graph structure:
  [START] → draft_node → review_node (INTERRUPT) → human_decision
                                                        |
                         ┌──── "approve" ────────── publish_node → [END]
                         └──── "revise"  ────────── draft_node  (loop)

Key concepts introduced:
  - interrupt()         : pauses graph execution and surfaces data to the caller
  - MemorySaver         : in-memory checkpointer that stores graph state between
                          resume calls (enables pause/resume across Python calls)
  - thread_id           : identifies a specific conversation / workflow instance
  - Command(resume=...) : how you resume a paused graph and pass data back in
  - Checkpointing       : LangGraph saves every state snapshot so you can
                          inspect history or replay from any point

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/05_human_in_the_loop.py
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------

class BlogState(TypedDict):
    topic: str                     # the blog topic requested by the user
    draft: str                     # current draft produced by the agent
    revision_notes: str            # feedback from the human reviewer
    revision_count: int            # how many times we've revised
    status: str                    # "draft" | "approved" | "published"


# ---------------------------------------------------------------------------
# 2. Initialise LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)


# ---------------------------------------------------------------------------
# 3. Define Nodes
# ---------------------------------------------------------------------------

def draft_node(state: BlogState) -> dict:
    """
    Generate (or revise) a blog post draft.

    On the first pass, writes a fresh post.
    On subsequent passes, takes the revision_notes into account.
    """
    if state["revision_notes"]:
        print(f"\n[draft_node] Revising draft (revision #{state['revision_count'] + 1})...")
        prompt = (
            f"You previously wrote a blog post about '{state['topic']}'.\n\n"
            f"Draft:\n{state['draft']}\n\n"
            f"Human feedback:\n{state['revision_notes']}\n\n"
            "Please rewrite the post addressing all the feedback. "
            "Keep it concise (under 300 words)."
        )
    else:
        print(f"\n[draft_node] Writing first draft about: '{state['topic']}'...")
        prompt = (
            f"Write a concise, engaging blog post (under 300 words) about: {state['topic']}. "
            "Include a catchy title, a brief intro, 2-3 main points, and a conclusion."
        )

    response = llm.invoke([
        SystemMessage(content="You are a professional blog writer."),
        HumanMessage(content=prompt),
    ])

    return {
        "draft": response.content,
        "revision_count": state["revision_count"] + 1,
        "revision_notes": "",   # clear notes after revising
        "status": "draft",
    }


def review_node(state: BlogState) -> Command:
    """
    PAUSE execution and present the draft to the human for review.

    interrupt() does two things:
      1. Saves the current graph state to the checkpointer
      2. Raises an exception that pauses execution and returns control
         to the caller (app.invoke / app.stream)

    The caller then calls app.invoke(Command(resume=...)) to resume.
    """
    print("\n" + "="*60)
    print(f"[review_node] Draft ready for review (revision #{state['revision_count']})")
    print("="*60)
    print(state["draft"])
    print("="*60)

    # interrupt() pauses here and surfaces the draft to the caller.
    # The value passed to interrupt() is available in the NodeInterrupt exception.
    human_feedback = interrupt({
        "draft": state["draft"],
        "revision_count": state["revision_count"],
        "message": "Please review the draft. Type 'approve' to publish or provide revision notes.",
    })

    # Execution resumes HERE after Command(resume=...) is called.
    # human_feedback contains whatever was passed as resume=...
    print(f"\n[review_node] Human responded: '{human_feedback}'")

    if human_feedback.strip().lower() == "approve":
        return Command(goto="publish_node", update={"status": "approved"})
    else:
        return Command(
            goto="draft_node",
            update={"revision_notes": human_feedback, "status": "draft"},
        )


def publish_node(state: BlogState) -> dict:
    """Simulate publishing the approved post."""
    print("\n[publish_node] Publishing the approved post...")
    print(f"  Title extracted: {state['draft'].splitlines()[0]}")
    print(f"  Revisions needed: {state['revision_count'] - 1}")
    print("  Status: PUBLISHED")
    return {"status": "published"}


# ---------------------------------------------------------------------------
# 4. Build the Graph with a Checkpointer
# ---------------------------------------------------------------------------
# The checkpointer (MemorySaver here) is what makes pause/resume possible.
# In production you'd swap MemorySaver for a database-backed checkpointer
# (e.g. langgraph-checkpoint-postgres).

def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("draft_node", draft_node)
    graph.add_node("review_node", review_node)
    graph.add_node("publish_node", publish_node)

    graph.add_edge(START, "draft_node")
    graph.add_edge("draft_node", "review_node")
    # review_node uses Command(goto=...) so no explicit edges from it are needed
    graph.add_edge("publish_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 5. Run It — Simulating the Human-in-the-Loop Flow
# ---------------------------------------------------------------------------

def run_workflow(app, topic: str, auto_responses: list[str]) -> None:
    """
    Run the blog workflow, providing automated responses for demonstration.

    In a real app 'auto_responses' would be replaced by actual user input.
    """
    # thread_id groups all checkpoints for one workflow instance.
    # Use different thread_ids for different blog posts / users.
    config = {"configurable": {"thread_id": "blog-workflow-1"}}

    initial_state: BlogState = {
        "topic": topic,
        "draft": "",
        "revision_notes": "",
        "revision_count": 0,
        "status": "draft",
    }

    print(f"\n=== Blog Drafting Workflow: '{topic}' ===")

    # --- First run: drafts the post and pauses at review_node ---
    print("\n--- Starting workflow (will pause at review) ---")
    for event in app.stream(initial_state, config, stream_mode="updates"):
        # Each event is a dict of {node_name: state_updates}
        node_name = list(event.keys())[0]
        if node_name != "__interrupt__":
            print(f"  Completed node: '{node_name}'")

    # --- Resume loop: human provides feedback or approves ---
    for i, response in enumerate(auto_responses):
        print(f"\n--- Human response #{i+1}: '{response}' ---")
        for event in app.stream(Command(resume=response), config, stream_mode="updates"):
            node_name = list(event.keys())[0]
            if node_name != "__interrupt__":
                print(f"  Completed node: '{node_name}'")

        # Check if the workflow is done
        state = app.get_state(config)
        if state.values.get("status") == "published":
            print("\nWorkflow complete! Post has been published.")
            break
        if not state.next:
            print("\nWorkflow complete.")
            break


if __name__ == "__main__":
    app = build_graph()

    # Simulate:
    #   - First review: ask for a more upbeat tone
    #   - Second review: approve
    run_workflow(
        app=app,
        topic="Why every developer should learn LangGraph",
        auto_responses=[
            "Please make the tone more upbeat and add a concrete example in the intro.",
            "approve",
        ],
    )

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Replace auto_responses with real input(): feedback = input("Your feedback: ")
    # 2. Add a max_revisions guard in review_node to auto-publish after N rounds.
    # 3. Use SqliteSaver instead of MemorySaver to persist state between
    #    Python process restarts:
    #      from langgraph.checkpoint.sqlite import SqliteSaver
    #      checkpointer = SqliteSaver.from_conn_string("blog_workflow.db")
