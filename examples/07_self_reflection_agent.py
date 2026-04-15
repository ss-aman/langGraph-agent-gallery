"""
Example 7: Self-Reflection Agent (Intermediate)
================================================

Great outputs rarely emerge on the first attempt. This example implements
the **Generate → Evaluate → Revise** loop — a pattern used in production
systems to iteratively improve agent output until it meets a quality bar.

The agent writes Python code for a given task, then *critiques its own
output* using a structured evaluator. If the code scores below the
threshold it revises with specific feedback; once the score is high enough
(or max iterations are reached) it formats and delivers the final solution.

Graph structure:

  [START]
     │
  generate_node  ◄──────────────────────────────┐
     │                                           │ (revise — score too low)
  evaluate_node                                  │
     │                                           │
  (quality_gate) ─── score < PASS_SCORE ─────────┘
     │
     └─── score ≥ PASS_SCORE  ───►  format_node  ───► [END]
          OR max_iterations reached

Key concepts introduced:
  - Reflection / self-critique pattern : LLM evaluates its own prior output
  - Structured evaluation output       : Pydantic model gives typed scores + feedback
  - Quality gate                       : conditional edge based on a numeric score
  - Iteration guard                    : max_iterations prevents infinite loops
  - Separate state fields per stage    : clean separation of "current attempt"
                                         vs "evaluation result" vs "final output"
  - Dynamic prompt construction        : prompts change based on iteration count
                                         and previous feedback

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/07_self_reflection_agent.py
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PASS_SCORE = 8        # score out of 10 needed to exit the loop
MAX_ITERATIONS = 4    # hard cap on revision cycles

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------

class CodeState(TypedDict):
    task: str               # what code to write (user's request)
    current_code: str       # the latest generated/revised code
    evaluation: str         # evaluator's textual feedback
    score: int              # quality score 0–10 (0 = not yet scored)
    iteration: int          # current revision number (starts at 0)
    history: list[dict]     # log of every attempt + score for post-run review
    final_code: str         # the approved code surfaced to the user


# ---------------------------------------------------------------------------
# 2. Evaluation Schema
# ---------------------------------------------------------------------------
# Breaking the evaluation into explicit sub-scores forces the LLM to think
# about each dimension separately instead of giving a vague overall rating.

class CodeEvaluation(BaseModel):
    correctness_score: int = Field(
        ge=0, le=10,
        description="Does the code correctly solve the stated task? (0-10)"
    )
    style_score: int = Field(
        ge=0, le=10,
        description="Is the code clean, readable, and Pythonic? (0-10)"
    )
    robustness_score: int = Field(
        ge=0, le=10,
        description="Does the code handle edge cases and errors gracefully? (0-10)"
    )
    overall_score: int = Field(
        ge=0, le=10,
        description=(
            "Holistic quality score (0-10). "
            f"A score of {PASS_SCORE}+ means the code is ready to ship."
        )
    )
    feedback: str = Field(
        description=(
            "Specific, actionable feedback. If overall_score < "
            f"{PASS_SCORE}, explain exactly what must be fixed. "
            "Be concise — 3-5 bullet points."
        )
    )
    strengths: str = Field(
        description="What the code does well (1-2 sentences)."
    )


evaluator_llm = llm.with_structured_output(CodeEvaluation)


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------

def generate_node(state: CodeState) -> dict:
    """
    Generate (first pass) or revise (subsequent passes) the code.

    On the first iteration the prompt is a clean spec.
    On later iterations the prior code AND the evaluator's feedback are
    included so the model knows exactly what to fix.
    """
    iteration = state["iteration"]

    if iteration == 0:
        print(f"\n[generate] Writing initial code for: '{state['task']}'")
        user_prompt = (
            f"Write clean, well-documented Python code that solves this task:\n\n"
            f"{state['task']}\n\n"
            "Requirements:\n"
            "  - Include a clear docstring\n"
            "  - Handle obvious edge cases (empty input, None, type errors…)\n"
            "  - Add a brief __main__ block that demonstrates usage\n"
            "  - Keep it under 80 lines"
        )
    else:
        print(f"\n[generate] Revising code (attempt #{iteration + 1})...")
        user_prompt = (
            f"Task: {state['task']}\n\n"
            f"Your previous attempt (score {state['score']}/10):\n"
            f"```python\n{state['current_code']}\n```\n\n"
            f"Evaluator feedback:\n{state['evaluation']}\n\n"
            "Please rewrite the code addressing ALL feedback points. "
            "Keep the parts that were praised and fix the specific issues mentioned."
        )

    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert Python developer. Write clean, correct, Pythonic code. "
            "Return ONLY the Python code — no markdown fences, no extra commentary."
        )),
        HumanMessage(content=user_prompt),
    ])

    # Strip accidental markdown fences the model sometimes adds
    code = response.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    print(f"[generate] Code produced ({len(code.splitlines())} lines).")
    return {
        "current_code": code,
        "iteration": iteration + 1,
    }


def evaluate_node(state: CodeState) -> dict:
    """
    Critique the current code against correctness, style, and robustness.

    Produces a structured evaluation with sub-scores and targeted feedback.
    This node does NOT modify the code — it only reads and judges it.
    """
    print(f"[evaluate] Scoring iteration #{state['iteration']}...")

    eval_prompt = (
        f"Evaluate this Python code written for the following task:\n\n"
        f"Task: {state['task']}\n\n"
        f"Code:\n```python\n{state['current_code']}\n```\n\n"
        f"Score each dimension honestly. A perfect 10 is production-ready code."
    )

    result: CodeEvaluation = evaluator_llm.invoke([
        SystemMessage(content=(
            "You are a senior Python code reviewer. "
            "Be strict but fair. Focus on correctness first, then style, then robustness. "
            "Provide concrete, actionable feedback."
        )),
        HumanMessage(content=eval_prompt),
    ])

    print(
        f"[evaluate] Scores — correctness:{result.correctness_score} "
        f"style:{result.style_score} "
        f"robustness:{result.robustness_score} "
        f"overall:{result.overall_score}/10"
    )
    print(f"[evaluate] Feedback: {result.feedback[:120]}...")

    # Append this attempt to the history log
    history_entry = {
        "iteration": state["iteration"],
        "score": result.overall_score,
        "correctness": result.correctness_score,
        "style": result.style_score,
        "robustness": result.robustness_score,
        "feedback": result.feedback,
        "code_length": len(state["current_code"].splitlines()),
    }

    return {
        "score": result.overall_score,
        "evaluation": result.feedback,
        "history": state["history"] + [history_entry],
    }


def format_node(state: CodeState) -> dict:
    """
    Package the approved code as the final output.

    This is a lightweight "exit" node — a good place to add post-processing
    like auto-formatting (black/ruff), saving to disk, or sending a webhook.
    """
    reason = (
        f"passed quality gate (score {state['score']}/{PASS_SCORE} required)"
        if state["score"] >= PASS_SCORE
        else f"reached max iterations ({MAX_ITERATIONS})"
    )
    print(f"\n[format] Finalising code — {reason}.")

    return {"final_code": state["current_code"]}


# ---------------------------------------------------------------------------
# 4. Quality Gate (Conditional Edge Function)
# ---------------------------------------------------------------------------

def quality_gate(state: CodeState) -> str:
    """
    Decide whether to approve or send back for revision.

    Three exit conditions:
      1. Score meets the bar → approve (go to format_node)
      2. Max iterations hit  → approve anyway (prevents infinite loops)
      3. Otherwise           → revise (loop back to generate_node)
    """
    if state["score"] >= PASS_SCORE:
        print(f"\n[quality_gate] PASSED (score {state['score']} ≥ {PASS_SCORE}). Approving.")
        return "approve"

    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n[quality_gate] MAX ITERATIONS reached. Approving best effort.")
        return "approve"

    remaining = MAX_ITERATIONS - state["iteration"]
    print(f"\n[quality_gate] FAILED (score {state['score']} < {PASS_SCORE}). "
          f"Sending back for revision ({remaining} attempt(s) left).")
    return "revise"


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(CodeState)

    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("format",   format_node)

    graph.add_edge(START,      "generate")
    graph.add_edge("generate", "evaluate")

    # The quality gate routes to either format_node (done) or generate (revise)
    graph.add_conditional_edges(
        "evaluate",
        quality_gate,
        {"approve": "format", "revise": "generate"},
    )

    graph.add_edge("format", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. Run It
# ---------------------------------------------------------------------------

def print_score_progression(history: list[dict]) -> None:
    """Display a visual summary of how the score improved across iterations."""
    print("\nScore progression:")
    for entry in history:
        bar = "█" * entry["score"] + "░" * (10 - entry["score"])
        print(
            f"  Iteration {entry['iteration']:>2} │ {bar} │ {entry['score']:>2}/10 "
            f"(C:{entry['correctness']} S:{entry['style']} R:{entry['robustness']})"
        )


def solve(app, task: str) -> None:
    initial: CodeState = {
        "task": task,
        "current_code": "",
        "evaluation": "",
        "score": 0,
        "iteration": 0,
        "history": [],
        "final_code": "",
    }

    print(f"\n{'='*65}")
    print(f"TASK: {task}")
    print("="*65)

    result = app.invoke(initial)

    print_score_progression(result["history"])

    print(f"\n{'─'*65}")
    print("FINAL CODE:")
    print("─"*65)
    print(result["final_code"])
    print()


if __name__ == "__main__":
    app = build_graph()

    tasks = [
        # Task 1 — medium difficulty: tests correctness + edge cases
        (
            "Write a function `flatten(nested)` that recursively flattens "
            "an arbitrarily nested list of lists into a single flat list. "
            "Example: flatten([1, [2, [3, 4]], 5]) → [1, 2, 3, 4, 5]"
        ),
        # Task 2 — slightly harder: tests robustness + algorithm
        (
            "Write a function `lru_cache_dict(capacity)` that returns a "
            "dictionary-like object implementing an LRU (Least Recently Used) "
            "cache with a fixed capacity. It must support get(key) and "
            "put(key, value) operations, evicting the least recently used "
            "item when capacity is exceeded."
        ),
    ]

    for task in tasks:
        solve(app, task)

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a "test_node" between evaluate and format that actually *runs*
    #    the code with unittest and feeds real test failures back as evaluation.
    #
    # 2. Split the evaluator into two nodes: a fast syntax/style check (no LLM)
    #    followed by an LLM semantic evaluation — avoid LLM calls for trivial
    #    issues like missing docstrings.
    #
    # 3. Visualise the score trajectory with matplotlib:
    #      import matplotlib.pyplot as plt
    #      scores = [h["score"] for h in result["history"]]
    #      plt.plot(scores, marker="o"); plt.ylim(0, 10); plt.show()
    #
    # 4. Persist state with MemorySaver and use get_state_history() to replay
    #    any previous iteration:
    #      for snapshot in app.get_state_history(config):
    #          print(snapshot.values["iteration"], snapshot.values["score"])
