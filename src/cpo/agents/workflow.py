from __future__ import annotations

from cpo.agents.critic import (
    critique,
)

from cpo.agents.investigator import (
    INVESTIGATOR_PROMPT,
    investigate,
)

from cpo.agents.local_ollama import (
    DEFAULT_MODEL,
    generate_structured_report,
    run_tool_agent,
)

from cpo.agents.schemas import (
    InvestigationReport,
)


# ============================================================
# FULL AI CO-SCIENTIST WORKFLOW
# ============================================================


def investigate_target(
    target_id: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Complete three-stage AI co-scientist workflow:

        1. Investigator
        2. Independent Critic
        3. Investigator revision

    Parameters
    ----------
    target_id:
        Acquisition target patient, e.g. T0124.

    model:
        Ollama model, default qwen3:8b.

    Returns
    -------
    dict
        Initial investigation, critic review, and revised
        final investigation.
    """

    # ========================================================
    # ROUND 1 — INITIAL INVESTIGATION
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "ROUND 1 — INVESTIGATOR"
    )

    print(
        "=" * 120
    )

    initial_report = investigate(
        target_id=target_id,
        model=model,
    )

    # ========================================================
    # ROUND 2 — INDEPENDENT CRITIC
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "ROUND 2 — CRITIC"
    )

    print(
        "=" * 120
    )

    critic_report = critique(
        report=initial_report,
        model=model,
    )

    # ========================================================
    # ROUND 3 — INVESTIGATOR REVISION
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "ROUND 3 — INVESTIGATOR REVISION"
    )

    print(
        "=" * 120
    )

    revision_prompt = f"""
Re-investigate acquisition patient {target_id}.

Your previous report was:

========================
PREVIOUS INVESTIGATION
========================

{initial_report.model_dump_json(indent=2)}

An independent Critic Agent then reviewed your report:

========================
CRITIC REVIEW
========================

{critic_report.model_dump_json(indent=2)}

You must now produce the FINAL revised investigation.

Use the deterministic tools again where useful.

Do not automatically agree with the Critic. Resolve every
important criticism against the underlying evidence.

The final report must:
- respect chronology,
- distinguish clonal from plasmid linkage,
- treat XGBoost as ranking evidence only,
- avoid unsupported direct-transmission claims,
- acknowledge hospital/location uncertainty for
  Known_CPO_pos patient sources,
- acknowledge important alternative explanations.
"""

    revision_messages = run_tool_agent(
        system_prompt=(
            INVESTIGATOR_PROMPT
        ),
        user_prompt=(
            revision_prompt
        ),
        model=model,
        verbose=True,
    )

    final_report = generate_structured_report(
        messages=(
            revision_messages
        ),
        schema=(
            InvestigationReport
        ),
        final_instruction=(
            "Return the FINAL revised InvestigationReport. "
            "Incorporate valid Critic corrections while "
            "remaining faithful to the deterministic evidence."
        ),
        model=model,
    )

    # ========================================================
    # RETURN COMPLETE RESULT
    # ========================================================

    return {
        "target_id":
            target_id,

        "model":
            model,

        "initial_investigation":
            initial_report.model_dump(),

        "critic_review":
            critic_report.model_dump(),

        "final_investigation":
            final_report.model_dump(),
    }