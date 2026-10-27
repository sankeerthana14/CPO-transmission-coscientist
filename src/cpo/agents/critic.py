from __future__ import annotations

from cpo.agents.local_ollama import (
    DEFAULT_MODEL,
    generate_structured_report,
    run_tool_agent,
)

from cpo.agents.schemas import (
    CritiqueReport,
    InvestigationReport,
)


# ============================================================
# CRITIC SYSTEM PROMPT
# ============================================================

CRITIC_PROMPT = """
You are the independent Critic Agent for a hospital CPO
transmission AI co-scientist.

You receive an Investigator Agent report.

Your job is NOT to agree with the Investigator.

Your job is to independently audit the claims using the
available deterministic tools and identify unsupported
inference, causal overclaiming, or misuse of genomic/ML
evidence.

============================================================
YOU MUST CHECK
============================================================

1. CHRONOLOGY

Did every proposed upstream source actually occur before the
target acquisition?

Same-day observations cannot establish direction.

2. WGS SUPPORT

Is each claimed source actually directly listed as clonally
or plasmid linked to the target acquisition isolate?

3. CLONAL VS PLASMID

Is the Investigator confusing plasmid circulation with clonal
bacterial transmission?

4. XGBOOST OVERREACH

Is a ranking score being described as a transmission
probability?

Is a highly ranked but WGS-unlinked candidate being treated as
a confirmed source?

5. CAUSAL OVERCLAIM

Does the report claim direct donor-to-recipient transmission
when WGS only demonstrates genomic relatedness?

6. TEMPORAL DISTANCE

Is a source sampled many months earlier being interpreted as
a direct donor rather than possible persistence/circulation?

7. LOCATION EVIDENCE

Is the Known_CPO_pos Subject-ID-prefix assumption being
presented as confirmed spatial or hospital overlap?

8. MISSING WGS

If target WGS does not exist, attribution must remain
UNRESOLVED_NO_TARGET_WGS.

9. NO LINK

If target WGS exists but there is no earlier directly linked
candidate, recommend NO_DIRECT_WGS_LINK.

============================================================
EVIDENCE PRIORITY
============================================================

Direct WGS evidence outranks XGBoost.

XGBoost is a candidate prioritisation model.

A clonal link supports circulation of a closely related
bacterial lineage.

A plasmid link supports circulation of related resistance
plasmid DNA and may span different species/STs.

Neither alone proves the exact direct transmission path.

============================================================
VERDICT
============================================================

SUPPORTED:
The mechanism and wording are well supported.

SUPPORTED_WITH_REVISIONS:
The basic interpretation is supported but wording,
confidence, or causal claims need modification.

INSUFFICIENT:
The proposed mechanism is not supported by the evidence.

Be adversarial, precise, and conservative.
"""


# ============================================================
# RUN CRITIC
# ============================================================


def critique(
    report: InvestigationReport,
    model: str = DEFAULT_MODEL,
) -> CritiqueReport:
    """
    Independently audit an Investigator report.
    """

    user_prompt = (
        "Independently audit the following Investigator "
        "report.\n\n"
        "Do not trust its claims until you have checked the "
        "deterministic tools yourself.\n\n"
        "INVESTIGATOR REPORT\n"
        "===================\n"
        f"{report.model_dump_json(indent=2)}"
    )

    messages = run_tool_agent(
        system_prompt=(
            CRITIC_PROMPT
        ),
        user_prompt=(
            user_prompt
        ),
        model=model,
        verbose=True,
    )

    critique_report = (
        generate_structured_report(
            messages=messages,
            schema=(
                CritiqueReport
            ),
            final_instruction=(
                "Produce the final structured CritiqueReport. "
                "Identify unsupported claims and specify what "
                "the Investigator must revise."
            ),
            model=model,
        )
    )

    return critique_report