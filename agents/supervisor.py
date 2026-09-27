"""
agents/supervisor.py — the router.

The supervisor reads the request and the work done so far, and decides which
specialist runs next — or that the work is complete.

Deterministic rules protect the workflow:
  1. step cap
  2. no repeat actions
  3. no early "done"
  4. no idle repeats
  5. no unrequested Jira/email actions
  6. no unnecessary bug agent
  7. Gmail SEARCH always goes to research
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from guardrails import EMAIL_RE
from llm import get_model


log = logging.getLogger(__name__)


ROUTES = [
    "research",
    "bug",
    "jira",
    "comms",
    "done",
]

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))


# ============================================================
# SUPERVISOR PROMPT
# ============================================================

SUPERVISOR_PROMPT = """You are the supervisor of a QA assistant team.

Decide the SINGLE next step for the request based on the work already done.

Team members:

- research:
  Gather context from:
  - project documents / RAG
  - Jira SEARCH
  - Gmail SEARCH / READ

  Use research FIRST for information and retrieval questions.

- bug:
  Format findings into a clean bug report.

- jira:
  Create or update a Jira ticket.
  This is a real action and ONLY the Jira agent can do it.

- comms:
  Draft and send an email.
  This is a real action and ONLY the communication agent can do it.

- done:
  Everything requested by the user has been completed.

Routing rules:

1. Identify EVERY action requested by the user.

2. Information/search requests normally go to research.

3. Gmail search/read requests MUST go to research.

4. Gmail search examples:
   - Search my Gmail for emails about testing
   - Find my emails about login failures
   - Search Gmail for Python
   - Find emails from LinkedIn
   - Search my mail for testing
   - Retrieve emails from Gmail

5. Sending an email goes to comms.

6. Creating/updating Jira goes to jira.

7. If a bug report is requested, use bug before Jira when appropriate.

8. If quality feedback says the previous step was weak, route to that step again.

9. A Jira/comms action that already completed, was declined, or was
   blocked by a guardrail is considered handled.

10. Never perform Jira or email actions unless explicitly requested.

11. Choose done only when every requested item is handled.

Examples:

"Search my Gmail for emails about testing"
→ research

"Use Gmail to find emails related to login failures"
→ research

"Search Jira for TEST issues"
→ research

"What is regression testing?"
→ research

"Create a Jira ticket for this bug"
→ research → bug → jira

"Email the bug report to me"
→ research → bug → comms
"""


# ============================================================
# DECISION MODEL
# ============================================================

class Decision(BaseModel):
    next: Literal[
        "research",
        "bug",
        "jira",
        "comms",
        "done",
    ]

    why: str = Field(
        description="one short reason for this routing decision"
    )


# ============================================================
# REQUEST HEURISTICS
# ============================================================

# ------------------------------------------------------------
# Jira REAL-ACTION requests
# ------------------------------------------------------------

_JIRA_ASK = re.compile(
    r"\b(create|file|raise|open|log|update)\b[^.]*\b(jira|ticket)\b"
    r"|"
    r"\blog\b[^.]*\bas an? (bug|issue)\b"
    r"|"
    r"\b(update|comment on|add a comment to|close)\b"
    r"[^.]*\b[A-Z][A-Z0-9]+-\d+\b",
    re.I,
)


# ------------------------------------------------------------
# Bug-report requests
# ------------------------------------------------------------

_BUG_ASK = re.compile(
    r"\b("
    r"bug report|"
    r"format\w*|"
    r"write[- ]?up|"
    r"write (it )?up|"
    r"report it"
    r")\b",
    re.I,
)


# ------------------------------------------------------------
# Email SEND requests
# ------------------------------------------------------------

_EMAIL_ASK = re.compile(
    r"\b(email|e-mail|mail|send)\b",
    re.I,
)


# ------------------------------------------------------------
# Gmail/email SEARCH requests
#
# IMPORTANT:
# Do NOT allow arbitrary text between "find/search" and
# "email/mail/Gmail".
#
# Otherwise this request:
#
# "Find open bugs and email a summary to qa.lead@gmail.com"
#
# could incorrectly become a Gmail SEARCH request.
# ------------------------------------------------------------

_GMAIL_SEARCH_ASK = re.compile(
    r"\b("
    r"search|find|retrieve|get"
    r")\s+"
    r"(?:my\s+)?"
    r"(?:gmail|emails?|e-mails?|mail|inbox)\b"
    r"|"
    r"\blook\s+(?:for|through)\s+"
    r"(?:my\s+)?"
    r"(?:gmail|emails?|e-mails?|mail|inbox)\b",
    re.I,
)


# ------------------------------------------------------------
# Direct Gmail SEARCH phrases
# ------------------------------------------------------------

_GMAIL_SEARCH_DIRECT = re.compile(
    r"\b("
    r"gmail\s+search|"
    r"search\s+gmail|"
    r"search\s+(?:my\s+)?emails?|"
    r"find\s+(?:my\s+)?emails?|"
    r"search\s+(?:my\s+)?mail|"
    r"find\s+(?:my\s+)?mail"
    r")\b",
    re.I,
)


# ============================================================
# GMAIL SEARCH DETECTOR
# ============================================================

def is_gmail_search_request(request: str) -> bool:
    """
    Return True only when the user is asking to READ/SEARCH Gmail.

    This must NOT classify email-send requests as Gmail searches.
    """

    return bool(
        _GMAIL_SEARCH_ASK.search(request)
        or _GMAIL_SEARCH_DIRECT.search(request)
    )


# ============================================================
# EMAIL ADDRESS REGEX
# ============================================================

# Generic email-address detection.
#
# This is intentionally separate from the guardrail's EMAIL_RE.
# Supervisor only needs to know whether the request contains
# an email address.
#
# Security/domain validation remains inside guardrails.py.

_EMAIL_ADDRESS = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.I,
)


# ============================================================
# PENDING ACTIONS
# ============================================================

def pending_actions(state: dict) -> list[str]:
    """
    Return real actions explicitly requested by the user
    that have not been handled yet.
    """

    req = state.get("request", "")

    out: list[str] = []


    # --------------------------------------------------------
    # Jira WRITE action
    # --------------------------------------------------------

    if (
        _JIRA_ASK.search(req)
        and not state.get("jira_result")
    ):
        out.append("jira")


    # --------------------------------------------------------
    # Email SEND action
    #
    # Gmail SEARCH is explicitly excluded.
    # --------------------------------------------------------

    wants_email = (
        _EMAIL_ASK.search(req)
        and not is_gmail_search_request(req)
        and (
            _EMAIL_ADDRESS.search(req)
            or re.search(r"\bme\b", req, re.I)
        )
    )

    if (
        wants_email
        and not state.get("email_result")
    ):
        out.append("comms")


    return out


# ============================================================
# CHECK IF ACTION ALREADY HANDLED
# ============================================================

def _already_handled(
    state: dict,
    action: str,
) -> bool:

    result_key = {
        "jira": "jira_result",
        "comms": "email_result",
    }[action]

    return bool(
        state.get(result_key)
    )


# ============================================================
# LLM DECISION
# ============================================================

def _decide(
    model,
    messages,
) -> tuple[str, str]:

    """
    Ask the LLM for a routing decision.

    First try structured output.
    If that fails, fall back to JSON.
    """

    # --------------------------------------------------------
    # Structured output
    # --------------------------------------------------------

    try:

        decision = (
            model
            .with_structured_output(Decision)
            .invoke(messages)
        )

        return (
            decision.next,
            decision.why,
        )

    except Exception as exc:

        log.warning(
            "structured routing failed (%s), "
            "falling back to JSON",
            exc,
        )


    # --------------------------------------------------------
    # JSON fallback
    # --------------------------------------------------------

    try:

        msg = model.invoke(
            messages
            + [
                (
                    "human",
                    (
                        'Reply ONLY with JSON: '
                        '{"next": "...", "why": "..."}'
                    ),
                )
            ]
        )

        raw = (
            msg.content
            if isinstance(msg.content, str)
            else str(msg.content)
        )

        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found in model response"
            )

        data = json.loads(
            raw[start:end + 1]
        )

        return (
            str(
                data.get(
                    "next",
                    "done",
                )
            ),
            str(
                data.get(
                    "why",
                    "",
                )
            ),
        )

    except Exception as exc:

        log.warning(
            "JSON routing fallback failed: %s",
            exc,
        )

        return (
            "done",
            "could not parse decision",
        )


# ============================================================
# SUPERVISOR NODE
# ============================================================

def supervisor_node(state: dict) -> dict:
    """
    Choose the next agent and apply deterministic routing rules.
    """

    steps = (
        state.get("steps", 0)
        + 1
    )


    # ========================================================
    # RULE 1 — STEP CAP
    # ========================================================

    if steps > MAX_STEPS:

        return {
            "next_agent": "done",
            "steps": steps,
            "trail": [
                (
                    f"🧭 supervisor → done "
                    f"(step limit {MAX_STEPS} reached)"
                )
            ],
        }


    # ========================================================
    # ASK LLM
    # ========================================================

    model = get_model(
        state.get("provider")
    )

    nxt, why = _decide(
        model,
        [
            (
                "system",
                SUPERVISOR_PROMPT,
            ),
            (
                "human",
                (
                    f"Request: "
                    f"{state['request']}\n\n"
                    f"Work so far:\n"
                    f"{_work_so_far(state)}"
                ),
            ),
        ],
    )


    # ========================================================
    # VALIDATE ROUTE
    # ========================================================

    if nxt not in ROUTES:

        nxt = "done"

        why = (
            f"invalid route '{nxt}'"
        )


    request = state.get(
        "request",
        "",
    )


    pending = pending_actions(
        state
    )


    gathered = bool(
        state.get("research")
        or state.get("bug_report")
    )


    gmail_search = (
        is_gmail_search_request(
            request
        )
    )


    # ========================================================
    # SPECIAL RULE — GMAIL SEARCH → RESEARCH
    # ========================================================

    if (
        gmail_search
        and not state.get("research")
    ):

        nxt = "research"

        why = (
            "Gmail search request → "
            "research has Gmail READ/SEARCH tools"
        )


    # ========================================================
    # RULE 2 — NEVER REPEAT REAL ACTION
    # ========================================================

    elif (
        nxt in ("jira", "comms")
        and _already_handled(
            state,
            nxt,
        )
    ):

        why = (
            f"{nxt} already handled "
            "— not repeating"
        )

        nxt = (
            pending[0]
            if pending
            else "done"
        )


    # ========================================================
    # RULE 5 — REAL ACTIONS ONLY WHEN REQUESTED
    # ========================================================

    elif (
        nxt in ("jira", "comms")
        and nxt not in pending
    ):

        why = (
            f"{nxt} was not requested "
            "— skipping"
        )

        nxt = (
            pending[0]
            if pending and gathered
            else "done"
        )


    # ========================================================
    # RULE 6 — BUG ONLY WHEN NEEDED
    # ========================================================

    elif (
        nxt == "bug"
        and not _BUG_ASK.search(request)
        and "jira" not in pending
    ):

        why = (
            "no bug report was requested "
            "— skipping"
        )

        nxt = (
            pending[0]
            if pending and gathered
            else "done"
        )


    # ========================================================
    # RULE 4 — NO IDLE REPEATS
    # ========================================================

    elif (
        nxt in ("research", "bug")
        and nxt == state.get(
            "next_agent"
        )
        and state.get("quality_ok")
    ):

        why = (
            f"{nxt} just passed quality "
            "— not re-running"
        )

        nxt = (
            pending[0]
            if pending
            else "done"
        )


    # ========================================================
    # RULE 3 — DON'T FINISH WHILE ACTION IS PENDING
    # ========================================================

    elif (
        nxt == "done"
        and pending
    ):

        target = (
            pending[0]
            if gathered
            else "research"
        )

        why = (
            f"request still needs "
            f"{pending[0]} "
            "— overriding 'done'"
        )

        nxt = target


    # ========================================================
    # RETURN STATE UPDATE
    # ========================================================

    return {
        "next_agent": nxt,
        "steps": steps,
        "trail": [
            (
                f"🧭 supervisor → "
                f"{nxt} ({why})"
            )
        ],
    }


# ============================================================
# WORK SO FAR
# ============================================================

def _work_so_far(
    state: dict,
) -> str:
    """
    Summarize completed work for the supervisor.
    """

    bits: list[str] = []


    # --------------------------------------------------------
    # Research
    # --------------------------------------------------------

    if state.get("research"):

        bits.append(
            "- research: done"
        )


    # --------------------------------------------------------
    # Bug report
    # --------------------------------------------------------

    if state.get("bug_report"):

        bits.append(
            "- bug report: written"
        )


    # --------------------------------------------------------
    # Jira
    # --------------------------------------------------------

    if state.get("jira_result"):

        bits.append(
            "- jira: "
            + state["jira_result"][:120]
        )


    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if state.get("email_result"):

        bits.append(
            "- email: "
            + state["email_result"][:120]
        )


    # --------------------------------------------------------
    # Quality feedback
    # --------------------------------------------------------

    if state.get("quality_notes"):

        bits.append(
            (
                "- quality feedback on "
                f"last step "
                f"({state.get('next_agent', '?')}): "
                + state["quality_notes"][:160]
            )
        )


    return (
        "\n".join(bits)
        or "(nothing yet)"
    )