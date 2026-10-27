from __future__ import annotations

import json
from collections import Counter, defaultdict
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
    / "council_v2_repeats"
)


SUMMARY_DIR = (
    ROOT
    / "summary"
)


RUNS = [
    1,
    2,
    3,
    4,
    5,
]


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


def load_runs():
    """
    Load all five frozen Council-v2 runs.
    """

    all_runs = {}

    for run_number in RUNS:

        run_dir = (
            ROOT
            / f"run{run_number:02d}"
        )

        combined_path = (
            run_dir
            / "all_council_v2_reports.json"
        )

        if not combined_path.exists():

            raise FileNotFoundError(
                f"Missing repeat output:\n"
                f"{combined_path}"
            )

        data = load_json(
            combined_path
        )

        all_runs[
            run_number
        ] = data

    return all_runs


def report_map(
    run_data,
):
    """
    Map target_id -> report.
    """

    return {
        report.get(
            "target_id"
        ):
            report

        for report
        in run_data.get(
            "reports",
            []
        )
    }


def summarize_runs(
    all_runs,
):
    rows = []

    for run_number in RUNS:

        reports = (
            all_runs[
                run_number
            ]
            .get(
                "reports",
                []
            )
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

        unanimous = 0
        majority = 0
        no_consensus = 0

        for report in completed:

            status = (
                report
                .get(
                    "consensus",
                    {}
                )
                .get(
                    "status"
                )
            )

            if status == "UNANIMOUS":
                unanimous += 1

            elif status == "MAJORITY":
                majority += 1

            elif status == "NO_CONSENSUS":
                no_consensus += 1

        completed_n = len(
            completed
        )

        correct_n = len(
            correct
        )

        rows.append(
            {
                "run":
                    run_number,

                "total_targets":
                    len(
                        TARGETS
                    ),

                "completed_targets":
                    completed_n,

                "error_targets":
                    len(
                        errors
                    ),

                "consensus_matching_reference":
                    correct_n,

                "agreement_among_completed":
                    (
                        correct_n
                        / completed_n
                        if completed_n
                        else None
                    ),

                "end_to_end_agreement":
                    (
                        correct_n
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

    return pd.DataFrame(
        rows
    )


def summarize_targets(
    all_runs,
):
    """
    Per-target repeatability across the 5 runs.

    Repeats are robustness measurements, NOT independent
    clinical observations.
    """

    rows = []

    mapped_runs = {
        run_number:
            report_map(
                all_runs[
                    run_number
                ]
            )

        for run_number
        in RUNS
    }

    for target_id in TARGETS:

        row = {
            "target_id":
                target_id,
        }

        mechanisms = []

        reference = None

        correct_count = 0

        completed_count = 0

        error_count = 0

        unanimous_count = 0

        majority_count = 0

        no_consensus_count = 0

        for run_number in RUNS:

            report = (
                mapped_runs[
                    run_number
                ]
                .get(
                    target_id
                )
            )

            prefix = (
                f"run{run_number:02d}"
            )

            if report is None:

                row[
                    f"{prefix}_status"
                ] = "MISSING"

                row[
                    f"{prefix}_consensus"
                ] = None

                error_count += 1

                continue

            if (
                report.get(
                    "status"
                )
                == "ERROR"
            ):

                row[
                    f"{prefix}_status"
                ] = "ERROR"

                row[
                    f"{prefix}_consensus"
                ] = None

                error_count += 1

                continue

            completed_count += 1

            validation = report.get(
                "validation",
                {}
            )

            consensus = report.get(
                "consensus",
                {}
            )

            if reference is None:

                reference = (
                    report.get(
                        "deterministic_wgs_reference"
                    )
                    or validation.get(
                        "reference_mechanism"
                    )
                )

            mechanism = consensus.get(
                "consensus_mechanism"
            )

            consensus_status = (
                consensus.get(
                    "status"
                )
            )

            row[
                f"{prefix}_status"
            ] = consensus_status

            row[
                f"{prefix}_consensus"
            ] = mechanism

            row[
                f"{prefix}_matches_reference"
            ] = validation.get(
                "consensus_matches_reference"
            )

            if mechanism is not None:

                mechanisms.append(
                    mechanism
                )

            if (
                validation.get(
                    "consensus_matches_reference"
                )
                is True
            ):

                correct_count += 1

            if (
                consensus_status
                == "UNANIMOUS"
            ):

                unanimous_count += 1

            elif (
                consensus_status
                == "MAJORITY"
            ):

                majority_count += 1

            elif (
                consensus_status
                == "NO_CONSENSUS"
            ):

                no_consensus_count += 1

        mechanism_counts = Counter(
            mechanisms
        )

        if mechanism_counts:

            (
                modal_mechanism,
                modal_count,
            ) = mechanism_counts.most_common(
                1
            )[0]

            modal_fraction = (
                modal_count
                / len(
                    mechanisms
                )
            )

        else:

            modal_mechanism = None
            modal_count = 0
            modal_fraction = None

        row.update(
            {
                "reference_mechanism":
                    reference,

                "completed_runs":
                    completed_count,

                "error_runs":
                    error_count,

                "runs_matching_reference":
                    correct_count,

                "agreement_rate_completed":
                    (
                        correct_count
                        / completed_count
                        if completed_count
                        else None
                    ),

                "end_to_end_correct_rate":
                    (
                        correct_count
                        / len(
                            RUNS
                        )
                    ),

                "modal_consensus":
                    modal_mechanism,

                "modal_consensus_count":
                    modal_count,

                "modal_consensus_fraction":
                    modal_fraction,

                "unanimous_runs":
                    unanimous_count,

                "majority_runs":
                    majority_count,

                "no_consensus_runs":
                    no_consensus_count,
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_models(
    all_runs,
):
    """
    Individual-model agreement with WGS reference across
    completed case-runs.

    These are descriptive execution-level stability metrics,
    not patient-level independent sample sizes.
    """

    stats = defaultdict(
        lambda: {
            "votes":
                0,

            "matches":
                0,

            "role":
                None,
        }
    )

    for run_number in RUNS:

        reports = (
            all_runs[
                run_number
            ]
            .get(
                "reports",
                []
            )
        )

        for report in reports:

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

                opinion = member.get(
                    "opinion",
                    {}
                )

                mechanism = opinion.get(
                    "mechanism"
                )

                if (
                    model is None
                    or mechanism is None
                ):
                    continue

                stats[
                    model
                ][
                    "votes"
                ] += 1

                stats[
                    model
                ][
                    "role"
                ] = role

                if (
                    mechanism
                    == reference
                ):

                    stats[
                        model
                    ][
                        "matches"
                    ] += 1

    rows = []

    for model, values in stats.items():

        votes = values[
            "votes"
        ]

        matches = values[
            "matches"
        ]

        rows.append(
            {
                "model":
                    model,

                "role":
                    values[
                        "role"
                    ],

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
            "model"
        )
        .reset_index(
            drop=True
        )
    )


def build_json_summary(
    run_df,
    target_df,
    model_df,
):
    """
    Compact machine-readable summary.
    """

    return {
        "analysis":
            "Council-v2 inference stability",

        "runs":
            len(
                RUNS
            ),

        "unique_clinical_targets":
            len(
                TARGETS
            ),

        "important_statistical_note": (
            "The five repeated runs are repeated model "
            "executions on the same eight clinical cases. "
            "They quantify inference stability and MUST NOT "
            "be treated as 40 independent clinical cases."
        ),

        "mean_end_to_end_agreement_across_runs":
            float(
                run_df[
                    "end_to_end_agreement"
                ]
                .mean()
            ),

        "mean_agreement_among_completed_runs":
            float(
                run_df[
                    "agreement_among_completed"
                ]
                .mean()
            ),

        "targets_with_5_of_5_correct":
            target_df.loc[
                target_df[
                    "runs_matching_reference"
                ]
                == 5,
                "target_id",
            ].tolist(),

        "targets_with_any_error":
            target_df.loc[
                target_df[
                    "error_runs"
                ]
                > 0,
                "target_id",
            ].tolist(),

        "models":
            model_df.to_dict(
                orient="records"
            ),
    }


def main():

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_runs = load_runs()

    run_df = summarize_runs(
        all_runs
    )

    target_df = summarize_targets(
        all_runs
    )

    model_df = summarize_models(
        all_runs
    )

    run_path = (
        SUMMARY_DIR
        / "run_summary.csv"
    )

    target_path = (
        SUMMARY_DIR
        / "target_stability.csv"
    )

    model_path = (
        SUMMARY_DIR
        / "model_agreement.csv"
    )

    json_path = (
        SUMMARY_DIR
        / "repeatability_summary.json"
    )

    run_df.to_csv(
        run_path,
        index=False,
    )

    target_df.to_csv(
        target_path,
        index=False,
    )

    model_df.to_csv(
        model_path,
        index=False,
    )

    json_summary = (
        build_json_summary(
            run_df,
            target_df,
            model_df,
        )
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COUNCIL V2 REPEATABILITY SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "\nPer-run performance:"
    )

    print(
        run_df.to_string(
            index=False
        )
    )

    print(
        "\nPer-target stability:"
    )

    display_columns = [
        "target_id",
        "reference_mechanism",
        "completed_runs",
        "error_runs",
        "runs_matching_reference",
        "end_to_end_correct_rate",
        "modal_consensus",
        "modal_consensus_fraction",
    ]

    print(
        target_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nIndividual-model agreement:"
    )

    print(
        model_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        run_path
    )

    print(
        target_path
    )

    print(
        model_path
    )

    print(
        json_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These five runs assess LLM inference stability. "
        "They are NOT 40 independent clinical cases."
    )


if __name__ == "__main__":
    main()