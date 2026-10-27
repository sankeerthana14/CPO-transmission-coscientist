from __future__ import annotations

import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path

from cpo.council_v2.consensus import compute_consensus
from cpo.council_v2.evidence import build_evidence_packet_v2
from cpo.council_v2.member import run_council_member
from cpo.council_v2.prompts import (
    LLAMA_CAUSAL_CRITIC_PROMPT,
    MEDGEMMA_BIOMEDICAL_PROMPT,
    QWEN_INVESTIGATOR_PROMPT,
)
from cpo.council_v2.validator import validate_council


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "T0016",
    "T0036",
    "T0052",
    "T0058",
    "T0080",
    "T0081",
    "T0124",
    "T0125",
]


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "qwen3:8b",
    "medgemma1.5:4b",
    "llama3.1:8b",
]


# ============================================================
# ROLES
#
# IMPORTANT:
# The prompts remain FIXED to the role.
# Only the model assigned to the role changes.
# ============================================================

ROLES = [
    {
        "role": "investigator",
        "prompt": QWEN_INVESTIGATOR_PROMPT,
    },
    {
        "role": "biomedical_expert",
        "prompt": MEDGEMMA_BIOMEDICAL_PROMPT,
    },
    {
        "role": "causal_critic",
        "prompt": LLAMA_CAUSAL_CRITIC_PROMPT,
    },
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "role_ablation"
)


# ============================================================
# HELPERS
# ============================================================


def save_json(
    data,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def evidence_hash(
    packet: dict,
) -> str:
    """
    SHA-256 fingerprint proving that role assignments received
    the same evidence packet.
    """

    canonical = json.dumps(
        packet,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def short_model_name(
    model: str,
) -> str:

    mapping = {
        "qwen3:8b": "Q",
        "medgemma1.5:4b": "M",
        "llama3.1:8b": "L",
    }

    return mapping.get(
        model,
        model.replace(":", "_"),
    )


def build_assignments():
    """
    Generate all 3! = 6 model-role assignments.

    Role order:
        Investigator
        Biomedical Expert
        Causal Critic
    """

    assignments = []

    for index, model_order in enumerate(
        permutations(MODELS),
        start=1,
    ):

        mapping = {
            ROLES[i]["role"]: model_order[i]
            for i in range(3)
        }

        code = "".join(
            short_model_name(model)
            for model
            in model_order
        )

        assignments.append(
            {
                "assignment_number": index,
                "assignment_id": f"A{index}",
                "assignment_code": code,
                "models_by_role": mapping,
                "is_original_assignment": (
                    model_order
                    == tuple(MODELS)
                ),
            }
        )

    return assignments


ASSIGNMENTS = build_assignments()


def assignment_folder_name(
    assignment: dict,
) -> str:

    mapping = assignment[
        "models_by_role"
    ]

    return (
        f"{assignment['assignment_id']}_"
        f"I-{short_model_name(mapping['investigator'])}_"
        f"B-{short_model_name(mapping['biomedical_expert'])}_"
        f"C-{short_model_name(mapping['causal_critic'])}"
    )


# ============================================================
# RUN ONE TARGET
# ============================================================


def run_target(
    *,
    target_id: str,
    assignment: dict,
) -> dict:
    """
    Run one target under one role assignment.

    Scientific configuration is identical to Council v2.

    ONLY model-role assignment changes.
    """

    target_id = (
        str(target_id)
        .strip()
        .upper()
    )

    evidence_packet, reference_mechanism = (
        build_evidence_packet_v2(
            target_id
        )
    )

    packet_hash = evidence_hash(
        evidence_packet
    )

    models_by_role = assignment[
        "models_by_role"
    ]

    council_members = []

    for role_config in ROLES:

        role = role_config[
            "role"
        ]

        prompt = role_config[
            "prompt"
        ]

        model = models_by_role[
            role
        ]

        print(
            f"    {role}: {model}"
        )

        vote = run_council_member(
            model=model,
            role_name=role,
            system_prompt=prompt,
            evidence_packet=evidence_packet,
            target_id=target_id,
        )

        council_members.append(
            {
                "model_name": model,
                "role": role,
                "opinion": vote.model_dump(),
            }
        )

    consensus = compute_consensus(
        council_members
    )

    validation = validate_council(
        target_id=target_id,
        evidence_packet=evidence_packet,
        reference_mechanism=reference_mechanism,
        council_members=council_members,
        consensus=consensus,
    )

    return {
        "target_id": target_id,

        "experiment":
            "role_assignment_ablation",

        "council_version":
            "2.0-frozen",

        "assignment_id":
            assignment[
                "assignment_id"
            ],

        "assignment_code":
            assignment[
                "assignment_code"
            ],

        "is_original_assignment":
            assignment[
                "is_original_assignment"
            ],

        "models_by_role":
            models_by_role,

        "generation_config": {
            "temperature": 0,
            "seed": 42,
            "tool_calls": False,
        },

        "evidence_packet_sha256":
            packet_hash,

        "evidence_packet":
            evidence_packet,

        # NOT visible to models.
        "deterministic_wgs_reference":
            reference_mechanism,

        "council_members":
            council_members,

        "consensus":
            consensus,

        "validation":
            validation,
    }


# ============================================================
# RUN ONE ASSIGNMENT
# ============================================================


def run_assignment(
    assignment: dict,
    *,
    overwrite: bool = False,
):
    """
    Run all eight acquisition targets for one role assignment.
    """

    folder_name = assignment_folder_name(
        assignment
    )

    output_dir = (
        OUTPUT_ROOT
        / folder_name
    )

    if (
        output_dir.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"{output_dir} already exists.\n"
            "Refusing to overwrite an existing ablation run."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"{assignment['assignment_id']} "
        f"ROLE ASSIGNMENT ABLATION"
    )

    print(
        "=" * 120
    )

    mapping = assignment[
        "models_by_role"
    ]

    print(
        "\nInvestigator:      ",
        mapping[
            "investigator"
        ],
    )

    print(
        "Biomedical Expert: ",
        mapping[
            "biomedical_expert"
        ],
    )

    print(
        "Causal Critic:     ",
        mapping[
            "causal_critic"
        ],
    )

    print(
        "\nOriginal assignment:",
        assignment[
            "is_original_assignment"
        ],
    )

    reports = []

    for target_index, target_id in enumerate(
        TARGETS,
        start=1,
    ):

        print(
            "\n"
            + "-"
            * 120
        )

        print(
            f"{assignment['assignment_id']} "
            f"TARGET "
            f"{target_index}/{len(TARGETS)}: "
            f"{target_id}"
        )

        print(
            "-"
            * 120
        )

        try:

            report = run_target(
                target_id=target_id,
                assignment=assignment,
            )

            reports.append(
                report
            )

            target_path = (
                output_dir
                / (
                    f"{target_id}"
                    "_role_ablation.json"
                )
            )

            save_json(
                report,
                target_path,
            )

            print(
                "\nConsensus:",
                report[
                    "consensus"
                ][
                    "consensus_mechanism"
                ],
            )

            print(
                "Reference:",
                report[
                    "deterministic_wgs_reference"
                ],
            )

            print(
                "Validated:",
                report[
                    "validation"
                ][
                    "consensus_matches_reference"
                ],
            )

        except Exception as error:

            error_report = {
                "target_id":
                    target_id,

                "assignment_id":
                    assignment[
                        "assignment_id"
                    ],

                "assignment_code":
                    assignment[
                        "assignment_code"
                    ],

                "models_by_role":
                    mapping,

                "status":
                    "ERROR",

                "error_type":
                    type(
                        error
                    ).__name__,

                "error":
                    str(
                        error
                    ),
            }

            reports.append(
                error_report
            )

            save_json(
                error_report,

                output_dir
                / (
                    f"{target_id}"
                    "_role_ablation_error.json"
                ),
            )

            print(
                "\nERROR:",
                type(
                    error
                ).__name__,
                str(
                    error
                ),
            )

    completed = [
        report
        for report
        in reports
        if report.get(
            "status"
        )
        != "ERROR"
    ]

    errors = [
        report
        for report
        in reports
        if report.get(
            "status"
        )
        == "ERROR"
    ]

    correct = [
        report
        for report
        in completed
        if (
            report
            .get(
                "validation",
                {}
            )
            .get(
                "consensus_matches_reference"
            )
            is True
        )
    ]

    combined = {
        "experiment":
            "role_assignment_ablation",

        "council_version":
            "2.0-frozen",

        "assignment":
            assignment,

        "frozen_configuration": {
            "temperature":
                0,

            "seed":
                42,

            "evidence":
                "Council v2 compact evidence packet",

            "consensus":
                "three-member majority vote",

            "important_note":
                (
                    "Only model-role assignment changes "
                    "between ablation conditions."
                ),
        },

        "summary": {
            "total_targets":
                len(
                    TARGETS
                ),

            "completed_targets":
                len(
                    completed
                ),

            "error_targets":
                len(
                    errors
                ),

            "consensus_matching_reference":
                len(
                    correct
                ),

            "agreement_among_completed":
                (
                    len(
                        correct
                    )
                    / len(
                        completed
                    )
                    if completed
                    else None
                ),

            "end_to_end_agreement":
                (
                    len(
                        correct
                    )
                    / len(
                        TARGETS
                    )
                ),
        },

        "reports":
            reports,
    }

    combined_path = (
        output_dir
        / "all_role_ablation_reports.json"
    )

    save_json(
        combined,
        combined_path,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"{assignment['assignment_id']} COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"Completed: "
        f"{len(completed)}/{len(TARGETS)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print(
        "Consensus matching reference: "
        f"{len(correct)}/{len(completed)} "
        "completed"
    )

    print(
        "End-to-end agreement: "
        f"{len(correct)}/{len(TARGETS)}"
    )

    print(
        "\nSaved:"
    )

    print(
        combined_path
    )


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run complete 3! model-role permutation "
            "ablation using frozen Council-v2 settings."
        )
    )

    parser.add_argument(
        "--assignments",
        nargs="+",
        type=int,
        default=[
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        help=(
            "Assignment numbers to run. "
            "Default: all six."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing assignment output. "
            "Avoid for final paper experiment."
        ),
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = []

    for number in args.assignments:

        if (
            number < 1
            or number > 6
        ):

            raise ValueError(
                "Assignments must be between 1 and 6."
            )

        selected.append(
            ASSIGNMENTS[
                number - 1
            ]
        )

    print(
        "\nROLE ASSIGNMENT ABLATION"
    )

    print(
        "========================"
    )

    print(
        "\nFrozen Council-v2 configuration."
    )

    print(
        "Only model-role assignment will change.\n"
    )

    for assignment in selected:

        run_assignment(
            assignment,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()