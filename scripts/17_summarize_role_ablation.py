from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "role_ablation"
)


SUMMARY_DIR = (
    ROOT
    / "summary"
)


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


ASSIGNMENT_FOLDERS = [
    "A1_I-Q_B-M_C-L",
    "A2_I-Q_B-L_C-M",
    "A3_I-M_B-Q_C-L",
    "A4_I-M_B-L_C-Q",
    "A5_I-L_B-Q_C-M",
    "A6_I-L_B-M_C-Q",
]


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def load_assignments():

    data = {}

    for folder in ASSIGNMENT_FOLDERS:

        path = (
            ROOT
            / folder
            / "all_role_ablation_reports.json"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing ablation output:\n{path}"
            )

        data[
            folder
        ] = load_json(
            path
        )

    return data


# ============================================================
# ASSIGNMENT-LEVEL PERFORMANCE
# ============================================================


def summarize_assignments(
    data,
):

    rows = []

    for folder, result in data.items():

        assignment = result[
            "assignment"
        ]

        reports = result.get(
            "reports",
            []
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

        unanimous = sum(
            (
                report
                .get(
                    "consensus",
                    {}
                )
                .get(
                    "status"
                )
                == "UNANIMOUS"
            )
            for report
            in completed
        )

        majority = sum(
            (
                report
                .get(
                    "consensus",
                    {}
                )
                .get(
                    "status"
                )
                == "MAJORITY"
            )
            for report
            in completed
        )

        no_consensus = sum(
            (
                report
                .get(
                    "consensus",
                    {}
                )
                .get(
                    "status"
                )
                == "NO_CONSENSUS"
            )
            for report
            in completed
        )

        mapping = assignment[
            "models_by_role"
        ]

        rows.append(
            {
                "assignment_id":
                    assignment[
                        "assignment_id"
                    ],

                "assignment_code":
                    assignment[
                        "assignment_code"
                    ],

                "original_assignment":
                    assignment[
                        "is_original_assignment"
                    ],

                "investigator":
                    mapping[
                        "investigator"
                    ],

                "biomedical_expert":
                    mapping[
                        "biomedical_expert"
                    ],

                "causal_critic":
                    mapping[
                        "causal_critic"
                    ],

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

                "unanimous_consensus":
                    unanimous,

                "majority_consensus":
                    majority,

                "no_consensus":
                    no_consensus,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "assignment_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MODEL × ROLE PERFORMANCE
# ============================================================


def summarize_model_roles(
    data,
):
    """
    Measure each model's agreement with the WGS reference
    while occupying each council role.
    """

    stats = defaultdict(
        lambda: {
            "available_votes": 0,
            "matches_reference": 0,
        }
    )

    for result in data.values():

        for report in result.get(
            "reports",
            []
        ):

            if (
                report.get(
                    "status"
                )
                == "ERROR"
            ):
                continue

            reference = (
                report.get(
                    "deterministic_wgs_reference"
                )
                or report
                .get(
                    "validation",
                    {}
                )
                .get(
                    "reference_mechanism"
                )
            )

            for member in report.get(
                "council_members",
                []
            ):

                model = member.get(
                    "model_name"
                )

                role = member.get(
                    "role"
                )

                mechanism = (
                    member
                    .get(
                        "opinion",
                        {}
                    )
                    .get(
                        "mechanism"
                    )
                )

                if (
                    model is None
                    or role is None
                    or mechanism is None
                ):
                    continue

                key = (
                    model,
                    role,
                )

                stats[
                    key
                ][
                    "available_votes"
                ] += 1

                if (
                    mechanism
                    == reference
                ):

                    stats[
                        key
                    ][
                        "matches_reference"
                    ] += 1

    rows = []

    for (
        model,
        role,
    ), values in stats.items():

        votes = values[
            "available_votes"
        ]

        matches = values[
            "matches_reference"
        ]

        rows.append(
            {
                "model":
                    model,

                "role":
                    role,

                "available_votes":
                    votes,

                "matches_reference":
                    matches,

                "agreement_rate":
                    (
                        matches
                        / votes
                        if votes
                        else None
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "model",
                "role",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TARGET × ASSIGNMENT TABLE
# ============================================================


def summarize_targets(
    data,
):

    rows = []

    for folder, result in data.items():

        assignment = result[
            "assignment"
        ]

        for report in result.get(
            "reports",
            []
        ):

            target_id = report.get(
                "target_id"
            )

            if (
                report.get(
                    "status"
                )
                == "ERROR"
            ):

                rows.append(
                    {
                        "assignment_id":
                            assignment[
                                "assignment_id"
                            ],

                        "target_id":
                            target_id,

                        "status":
                            "ERROR",

                        "reference_mechanism":
                            None,

                        "consensus_mechanism":
                            None,

                        "consensus_status":
                            None,

                        "matches_reference":
                            False,
                    }
                )

                continue

            reference = (
                report.get(
                    "deterministic_wgs_reference"
                )
            )

            consensus = report.get(
                "consensus",
                {}
            )

            validation = report.get(
                "validation",
                {}
            )

            member_votes = consensus.get(
                "member_votes",
                {}
            )

            rows.append(
                {
                    "assignment_id":
                        assignment[
                            "assignment_id"
                        ],

                    "target_id":
                        target_id,

                    "status":
                        "COMPLETED",

                    "reference_mechanism":
                        reference,

                    "consensus_mechanism":
                        consensus.get(
                            "consensus_mechanism"
                        ),

                    "consensus_status":
                        consensus.get(
                            "status"
                        ),

                    "matches_reference":
                        validation.get(
                            "consensus_matches_reference"
                        ),

                    "qwen_vote":
                        member_votes.get(
                            "qwen3:8b"
                        ),

                    "medgemma_vote":
                        member_votes.get(
                            "medgemma1.5:4b"
                        ),

                    "llama_vote":
                        member_votes.get(
                            "llama3.1:8b"
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "assignment_id",
                "target_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# JSON SUMMARY
# ============================================================


def build_json_summary(
    assignment_df,
    model_role_df,
):

    original = (
        assignment_df[
            assignment_df[
                "original_assignment"
            ]
            == True
        ]
    )

    return {
        "analysis":
            "Complete 3! model-role assignment ablation",

        "unique_clinical_targets":
            8,

        "models": [
            "qwen3:8b",
            "medgemma1.5:4b",
            "llama3.1:8b",
        ],

        "roles": [
            "investigator",
            "biomedical_expert",
            "causal_critic",
        ],

        "assignments_tested":
            6,

        "important_methodological_note": (
            "The original model-role assignment was defined "
            "a priori. The remaining five permutations are "
            "sensitivity analyses and should not be used to "
            "select a new final configuration based solely "
            "on performance on these eight cases."
        ),

        "original_assignment":
            (
                original
                .to_dict(
                    orient="records"
                )[0]
                if not original.empty
                else None
            ),

        "assignment_results":
            assignment_df.to_dict(
                orient="records"
            ),

        "model_role_results":
            model_role_df.to_dict(
                orient="records"
            ),
    }


def main():

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_assignments()

    assignment_df = (
        summarize_assignments(
            data
        )
    )

    model_role_df = (
        summarize_model_roles(
            data
        )
    )

    target_df = (
        summarize_targets(
            data
        )
    )

    assignment_path = (
        SUMMARY_DIR
        / "assignment_summary.csv"
    )

    model_role_path = (
        SUMMARY_DIR
        / "model_role_summary.csv"
    )

    target_path = (
        SUMMARY_DIR
        / "target_assignment_results.csv"
    )

    json_path = (
        SUMMARY_DIR
        / "role_ablation_summary.json"
    )

    assignment_df.to_csv(
        assignment_path,
        index=False,
    )

    model_role_df.to_csv(
        model_role_path,
        index=False,
    )

    target_df.to_csv(
        target_path,
        index=False,
    )

    summary = build_json_summary(
        assignment_df,
        model_role_df,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "ROLE-ASSIGNMENT ABLATION SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        "\nAssignment-level results:"
    )

    display_columns = [
        "assignment_id",
        "investigator",
        "biomedical_expert",
        "causal_critic",
        "completed_targets",
        "error_targets",
        "consensus_matching_reference",
        "agreement_among_completed",
        "end_to_end_agreement",
    ]

    print(
        assignment_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nModel × role agreement:"
    )

    print(
        model_role_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        assignment_path
    )

    print(
        model_role_path
    )

    print(
        target_path
    )

    print(
        json_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Treat this as a model-role sensitivity analysis. "
        "Do not select a new final architecture simply "
        "because one permutation scores highest on n=8 cases."
    )


if __name__ == "__main__":
    main()