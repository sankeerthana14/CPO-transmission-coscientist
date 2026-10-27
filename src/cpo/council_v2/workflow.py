from __future__ import annotations

import hashlib
import json
from datetime import (
    datetime,
    timezone,
)

from cpo.council_v2.consensus import (
    compute_consensus,
)

from cpo.council_v2.evidence import (
    build_evidence_packet_v2,
)

from cpo.council_v2.member import (
    run_council_member,
)

from cpo.council_v2.prompts import (
    LLAMA_CAUSAL_CRITIC_PROMPT,
    MEDGEMMA_BIOMEDICAL_PROMPT,
    QWEN_INVESTIGATOR_PROMPT,
)

from cpo.council_v2.validator import (
    validate_council,
)


DEFAULT_QWEN_MODEL = (
    "qwen3:8b"
)

DEFAULT_MEDGEMMA_MODEL = (
    "medgemma1.5:4b"
)

DEFAULT_LLAMA_MODEL = (
    "llama3.1:8b"
)


def _evidence_hash(
    packet: dict,
) -> str:
    """
    Prove that all council members received identical evidence.
    """

    canonical = json.dumps(
        packet,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()


def run_council_v2_target(
    target_id: str,
    *,
    qwen_model: str = (
        DEFAULT_QWEN_MODEL
    ),
    medgemma_model: str = (
        DEFAULT_MEDGEMMA_MODEL
    ),
    llama_model: str = (
        DEFAULT_LLAMA_MODEL
    ),
) -> dict:
    """
    Run Council v2 on one acquisition target.
    """

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"LLM COUNCIL V2: {target_id}"
    )

    print(
        "=" * 120
    )

    print(
        "\nBuilding compact locked evidence packet..."
    )

    (
        evidence_packet,
        reference_mechanism,
    ) = build_evidence_packet_v2(
        target_id
    )

    evidence_hash = (
        _evidence_hash(
            evidence_packet
        )
    )

    print(
        f"Evidence SHA-256: "
        f"{evidence_hash}"
    )

    member_configs = [
        {
            "model_name":
                qwen_model,

            "role":
                "investigator",

            "prompt":
                QWEN_INVESTIGATOR_PROMPT,
        },
        {
            "model_name":
                medgemma_model,

            "role":
                "biomedical_expert",

            "prompt":
                MEDGEMMA_BIOMEDICAL_PROMPT,
        },
        {
            "model_name":
                llama_model,

            "role":
                "causal_critic",

            "prompt":
                LLAMA_CAUSAL_CRITIC_PROMPT,
        },
    ]

    council_members = []

    for index, config in enumerate(
        member_configs,
        start=1,
    ):

        print(
            "\n"
            + "-" * 120
        )

        print(
            f"COUNCIL MEMBER {index}/3 — "
            f"{config['model_name']} "
            f"({config['role']})"
        )

        print(
            "-" * 120
        )

        vote = run_council_member(
            model=(
                config[
                    "model_name"
                ]
            ),

            role_name=(
                config[
                    "role"
                ]
            ),

            system_prompt=(
                config[
                    "prompt"
                ]
            ),

            evidence_packet=(
                evidence_packet
            ),

            target_id=(
                target_id
            ),
        )

        print(
            "Mechanism:",
            vote.mechanism,
        )

        print(
            "Confidence:",
            vote.confidence,
        )

        council_members.append(
            {
                "model_name":
                    config[
                        "model_name"
                    ],

                "role":
                    config[
                        "role"
                    ],

                "opinion":
                    vote.model_dump(),
            }
        )

    print(
        "\n"
        + "-" * 120
    )

    print(
        "COUNCIL CONSENSUS"
    )

    print(
        "-" * 120
    )

    consensus = (
        compute_consensus(
            council_members
        )
    )

    print(
        "Status:",
        consensus[
            "status"
        ],
    )

    print(
        "Votes:",
        consensus[
            "votes"
        ],
    )

    print(
        "Consensus mechanism:",
        consensus[
            "consensus_mechanism"
        ],
    )

    validation = (
        validate_council(
            target_id=(
                target_id
            ),

            evidence_packet=(
                evidence_packet
            ),

            reference_mechanism=(
                reference_mechanism
            ),

            council_members=(
                council_members
            ),

            consensus=(
                consensus
            ),
        )
    )

    print(
        "Validation:",
        validation[
            "status"
        ],
    )

    print(
        "WGS reference:",
        reference_mechanism,
    )

    return {
        "target_id":
            target_id,

        "experiment":
            "heterogeneous_llm_council_v2",

        "council_version":
            "2.0",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "models": {
            "investigator":
                qwen_model,

            "biomedical_expert":
                medgemma_model,

            "causal_critic":
                llama_model,
        },

        "generation_config": {
            "temperature":
                0,

            "seed":
                42,

            "tool_calls":
                False,

            "max_output_tokens":
                700,
        },

        "evidence_packet_sha256":
            evidence_hash,

        "evidence_packet":
            evidence_packet,

        # Private evaluation target. This was NOT included
        # in the LLM-visible evidence packet.
        "deterministic_wgs_reference":
            reference_mechanism,

        "council_members":
            council_members,

        "consensus":
            consensus,

        "validation":
            validation,
    }