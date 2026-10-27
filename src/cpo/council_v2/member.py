from __future__ import annotations

import json

from ollama import chat

from cpo.council_v2.schemas import (
    CouncilVote,
)


def run_council_member(
    *,
    model: str,
    role_name: str,
    system_prompt: str,
    evidence_packet: dict,
    target_id: str,
    max_attempts: int = 3,
) -> CouncilVote:
    """
    Run one independent council member.

    The evidence packet is locked and identical across models.
    """

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    evidence_json = json.dumps(
        evidence_packet,
        indent=2,
        ensure_ascii=False,
    )

    base_user_prompt = f"""
Evaluate target {target_id}.

Your response target_id MUST be exactly:

{target_id}

Your council role is:

{role_name}

LOCKED EVIDENCE PACKET
======================

{evidence_json}

Return a concise structured vote.

Requirements:

- rationale: maximum two sentences
- key_evidence: maximum three short items
- do not invent identifiers
- do not use XGBoost score as transmission probability
- do not infer clonal/plasmid linkage from SHAP
- direct_transmission_proven should normally be false
"""

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        if attempt == 1:

            user_prompt = (
                base_user_prompt
            )

        else:

            user_prompt = (
                base_user_prompt
                + "\n\n"
                + "IMPORTANT RETRY INSTRUCTION:\n"
                + "Your previous response could not be "
                + "validated. Return SHORT JSON only. "
                + "Do not add commentary outside the "
                + "required schema."
            )

        response = chat(
            model=model,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

            format=(
                CouncilVote
                .model_json_schema()
            ),

            options={
                "temperature":
                    0,

                "seed":
                    42,

                # Large enough for the small schema while
                # preventing unnecessarily long answers.
                "num_predict":
                    700,
            },

            # Sequentially unload models to reduce memory use.
            keep_alive=0,
        )

        content = (
            response.message.content
            or ""
        ).strip()

        try:

            vote = (
                CouncilVote
                .model_validate_json(
                    content
                )
            )

            returned_target = (
                vote.target_id
                .strip()
                .upper()
            )

            if (
                returned_target
                != target_id
            ):

                raise ValueError(
                    "Model changed target_id "
                    f"from {target_id} to "
                    f"{vote.target_id}."
                )

            vote.target_id = (
                target_id
            )

            return vote

        except Exception as error:

            last_error = error

            print(
                f"  {model} structured-output "
                f"attempt {attempt}/{max_attempts} failed."
            )

    raise RuntimeError(
        f"{model} failed to produce a valid "
        f"CouncilVote for {target_id}: "
        f"{last_error}"
    )