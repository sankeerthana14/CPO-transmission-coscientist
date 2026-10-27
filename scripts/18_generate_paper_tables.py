from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROCESSED = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "paper"
)

TABLE_DIR = (
    OUTPUT_ROOT
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ACQ_FILE = (
    PROCESSED
    / "acquisition_cases.csv"
)

WGS_FILE = (
    PROCESSED
    / "wgs_labeled_pairs.csv"
)

PRED_FILE = (
    PROCESSED
    / "xgboost_oof_predictions.csv"
)

SHAP_FILE = (
    PROCESSED
    / "xgboost_oof_shap.csv"
)


def bool_series(
    series: pd.Series,
) -> pd.Series:
    """
    Robust CSV boolean conversion.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


def bool_value(
    value,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def source_class(
    source_type,
):

    value = (
        str(source_type)
        .strip()
        .lower()
    )

    if "environment" in value:
        return "ENVIRONMENT"

    if (
        "patient" in value
        or "known" in value
    ):
        return "PATIENT"

    return "UNKNOWN"


def derive_mechanism(
    target_row: pd.Series,
    target_wgs: pd.DataFrame,
) -> str:
    """
    Deterministic WGS-grounded target classification.
    """

    has_wgs = bool_value(
        target_row.get(
            "has_culture_wgs",
            False,
        )
    )

    if not has_wgs:

        return (
            "UNRESOLVED_NO_TARGET_WGS"
        )

    if target_wgs.empty:

        return (
            "NO_DIRECT_WGS_LINK"
        )

    days = pd.to_numeric(
        target_wgs[
            "days_before_acquisition"
        ],
        errors="coerce",
    )

    prior = target_wgs[
        days > 0
    ].copy()

    if prior.empty:

        return (
            "NO_DIRECT_WGS_LINK"
        )

    linked = bool_series(
        prior[
            "any_wgs_link"
        ]
    )

    linked_rows = prior[
        linked
    ].copy()

    if linked_rows.empty:

        return (
            "NO_DIRECT_WGS_LINK"
        )

    mechanisms = set()

    for _, row in linked_rows.iterrows():

        cls = source_class(
            row.get(
                "source_type"
            )
        )

        if bool_value(
            row.get(
                "clonal_link",
                False,
            )
        ):

            mechanisms.add(
                f"{cls}"
                "_ASSOCIATED_CLONAL"
            )

        if bool_value(
            row.get(
                "plasmid_link",
                False,
            )
        ):

            mechanisms.add(
                f"{cls}"
                "_ASSOCIATED_PLASMID"
            )

    if len(
        mechanisms
    ) == 1:

        return next(
            iter(
                mechanisms
            )
        )

    if len(
        mechanisms
    ) > 1:

        return "MIXED"

    return (
        "NO_DIRECT_WGS_LINK"
    )


def linked_source_string(
    target_wgs: pd.DataFrame,
) -> str:
    """
    Human-readable direct WGS-linked source list.
    """

    if target_wgs.empty:
        return ""

    days = pd.to_numeric(
        target_wgs[
            "days_before_acquisition"
        ],
        errors="coerce",
    )

    prior = target_wgs[
        days > 0
    ].copy()

    if prior.empty:
        return ""

    linked = bool_series(
        prior[
            "any_wgs_link"
        ]
    )

    linked_rows = (
        prior[
            linked
        ]
        .sort_values(
            "days_before_acquisition"
        )
    )

    values = []

    for _, row in linked_rows.iterrows():

        if bool_value(
            row.get(
                "clonal_link",
                False,
            )
        ):

            link_type = (
                "clonal"
            )

        elif bool_value(
            row.get(
                "plasmid_link",
                False,
            )
        ):

            link_type = (
                "plasmid"
            )

        else:

            link_type = (
                "linked"
            )

        subject = row.get(
            "source_subject_id",
            ""
        )

        sample = row.get(
            "source_sample_id",
            ""
        )

        day = row.get(
            "days_before_acquisition",
            ""
        )

        values.append(
            f"{subject}/{sample} "
            f"({int(float(day))} d; "
            f"{link_type})"
        )

    return "; ".join(
        values
    )


def get_top_candidate(
    predictions: pd.DataFrame,
    target_id: str,
):
    """
    Top augmented XGBoost candidate.
    """

    subset = predictions[
        (
            predictions[
                "target_id"
            ]
            .astype(str)
            ==
            target_id
        )
        &
        (
            predictions[
                "model_name"
            ]
            .astype(str)
            .str.upper()
            ==
            "AUGMENTED"
        )
    ].copy()

    if subset.empty:

        return {
            "source_type": "",
            "subject": "",
            "sample": "",
            "score": np.nan,
            "top3_source_types": "",
        }

    subset = subset.sort_values(
        "rank_within_target"
    )

    top = subset.iloc[0]

    top3_types = (
        subset.head(3)[
            "source_type"
        ]
        .astype(str)
        .tolist()
    )

    return {
        "source_type":
            top.get(
                "source_type",
                ""
            ),

        "subject":
            top.get(
                "source_subject_id",
                ""
            ),

        "sample":
            top.get(
                "source_sample_id",
                ""
            ),

        "score":
            top.get(
                "predicted_probability",
                np.nan,
            ),

        "top3_source_types":
            ", ".join(
                top3_types
            ),
    }


def mechanism_label(
    mechanism: str,
) -> str:

    mapping = {
        "PATIENT_ASSOCIATED_CLONAL":
            "Patient-associated clonal circulation",

        "ENVIRONMENT_ASSOCIATED_CLONAL":
            "Environment-associated clonal circulation",

        "PATIENT_ASSOCIATED_PLASMID":
            "Patient-associated plasmid circulation",

        "ENVIRONMENT_ASSOCIATED_PLASMID":
            "Environment-associated plasmid circulation",

        "NO_DIRECT_WGS_LINK":
            "No direct prior WGS-linked source",

        "UNRESOLVED_NO_TARGET_WGS":
            "Unresolved — no target WGS",

        "MIXED":
            "Mixed WGS-supported mechanisms",
    }

    return mapping.get(
        mechanism,
        mechanism,
    )


def evidence_tier(
    mechanism: str,
) -> str:

    if mechanism in {
        "PATIENT_ASSOCIATED_CLONAL",
        "ENVIRONMENT_ASSOCIATED_CLONAL",
        "PATIENT_ASSOCIATED_PLASMID",
        "ENVIRONMENT_ASSOCIATED_PLASMID",
        "MIXED",
    }:

        return (
            "Tier 1 — direct WGS-supported "
            "source category"
        )

    if (
        mechanism
        == "NO_DIRECT_WGS_LINK"
    ):

        return (
            "Tier 2 — ranked hypothesis "
            "without direct WGS linkage"
        )

    return (
        "Tier 3 — unresolved due to "
        "missing target WGS"
    )


def build_case_table():

    acquisitions = pd.read_csv(
        ACQ_FILE,
        parse_dates=[
            "acquisition_date",
        ],
    )

    wgs = pd.read_csv(
        WGS_FILE
    )

    predictions = pd.read_csv(
        PRED_FILE
    )

    rows = []

    for _, target in acquisitions.iterrows():

        target_id = str(
            target[
                "SubjectID"
            ]
        )

        target_wgs = wgs[
            wgs[
                "target_id"
            ]
            .astype(str)
            ==
            target_id
        ].copy()

        mechanism = derive_mechanism(
            target,
            target_wgs,
        )

        top = get_top_candidate(
            predictions,
            target_id,
        )

        rows.append(
            {
                "target_id":
                    target_id,

                "acquisition_date":
                    target[
                        "acquisition_date"
                    ].date(),

                "lab_cp_family":
                    target.get(
                        "lab_cp_gene"
                    ),

                "target_species_st":
                    target.get(
                        "genomics_species_st"
                    ),

                "target_cp_gene":
                    target.get(
                        "genomics_cp_gene"
                    ),

                "wgs_reference_mechanism":
                    mechanism,

                "clinical_interpretation":
                    mechanism_label(
                        mechanism
                    ),

                "evidence_tier":
                    evidence_tier(
                        mechanism
                    ),

                "direct_wgs_linked_sources":
                    linked_source_string(
                        target_wgs
                    ),

                "top_xgboost_source_type":
                    top[
                        "source_type"
                    ],

                "top_xgboost_subject":
                    top[
                        "subject"
                    ],

                "top_xgboost_sample":
                    top[
                        "sample"
                    ],

                "top_xgboost_ranking_score":
                    top[
                        "score"
                    ],

                "top3_xgboost_source_types":
                    top[
                        "top3_source_types"
                    ],

                "direct_donor_proven":
                    False,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE_DIR
        / "table_case_level_source_attribution.csv",
        index=False,
    )

    result.to_latex(
        TABLE_DIR
        / "table_case_level_source_attribution.tex",
        index=False,
        escape=True,
    )

    return result


def calculate_model_metrics():

    predictions = pd.read_csv(
        PRED_FILE
    )

    rows = []

    for model_name in [
        "CORE",
        "AUGMENTED",
    ]:

        subset = predictions[
            predictions[
                "model_name"
            ]
            .astype(str)
            .str.upper()
            ==
            model_name
        ].copy()

        y = pd.to_numeric(
            subset[
                "label_any_wgs_link"
            ],
            errors="coerce",
        ).astype(int)

        score = pd.to_numeric(
            subset[
                "predicted_probability"
            ],
            errors="coerce",
        )

        pooled_auroc = (
            roc_auc_score(
                y,
                score,
            )
        )

        pooled_auprc = (
            average_precision_score(
                y,
                score,
            )
        )

        target_ap = []

        target_auc = []

        reciprocal_ranks = []

        hit1 = []

        hit3 = []

        hit5 = []

        recall3 = []

        recall5 = []

        best_ranks = []

        for _, group in subset.groupby(
            "target_id"
        ):

            group = group.sort_values(
                "rank_within_target"
            )

            gy = pd.to_numeric(
                group[
                    "label_any_wgs_link"
                ],
                errors="coerce",
            ).astype(int)

            if gy.sum() == 0:
                continue

            gs = pd.to_numeric(
                group[
                    "predicted_probability"
                ],
                errors="coerce",
            )

            target_ap.append(
                average_precision_score(
                    gy,
                    gs,
                )
            )

            if gy.nunique() > 1:

                target_auc.append(
                    roc_auc_score(
                        gy,
                        gs,
                    )
                )

            positive_rows = group[
                gy == 1
            ]

            ranks = pd.to_numeric(
                positive_rows[
                    "rank_within_target"
                ],
                errors="coerce",
            )

            best_rank = int(
                ranks.min()
            )

            best_ranks.append(
                best_rank
            )

            reciprocal_ranks.append(
                1.0
                / best_rank
            )

            hit1.append(
                int(
                    best_rank
                    <= 1
                )
            )

            hit3.append(
                int(
                    best_rank
                    <= 3
                )
            )

            hit5.append(
                int(
                    best_rank
                    <= 5
                )
            )

            total_positive = (
                gy.sum()
            )

            recall3.append(
                (
                    pd.to_numeric(
                        group[
                            "rank_within_target"
                        ],
                        errors="coerce",
                    )
                    <= 3
                )
                .mul(
                    gy
                )
                .sum()
                / total_positive
            )

            recall5.append(
                (
                    pd.to_numeric(
                        group[
                            "rank_within_target"
                        ],
                        errors="coerce",
                    )
                    <= 5
                )
                .mul(
                    gy
                )
                .sum()
                / total_positive
            )

        rows.append(
            {
                "model":
                    model_name,

                "pooled_auroc":
                    pooled_auroc,

                "pooled_auprc":
                    pooled_auprc,

                "mean_positive_target_auroc":
                    np.mean(
                        target_auc
                    ),

                "mean_positive_target_auprc":
                    np.mean(
                        target_ap
                    ),

                "mrr":
                    np.mean(
                        reciprocal_ranks
                    ),

                "hit_at_1":
                    np.mean(
                        hit1
                    ),

                "hit_at_3":
                    np.mean(
                        hit3
                    ),

                "hit_at_5":
                    np.mean(
                        hit5
                    ),

                "recall_at_3":
                    np.mean(
                        recall3
                    ),

                "recall_at_5":
                    np.mean(
                        recall5
                    ),

                "median_best_positive_rank":
                    np.median(
                        best_ranks
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE_DIR
        / "table_xgboost_performance.csv",
        index=False,
    )

    result.to_latex(
        TABLE_DIR
        / "table_xgboost_performance.tex",
        index=False,
        float_format="%.3f",
    )

    return result


def build_shap_table():

    shap = pd.read_csv(
        SHAP_FILE
    )

    shap_columns = [
        column
        for column
        in shap.columns
        if column.startswith(
            "shap__"
        )
    ]

    rows = []

    for column in shap_columns:

        values = pd.to_numeric(
            shap[
                column
            ],
            errors="coerce",
        )

        rows.append(
            {
                "feature":
                    column.replace(
                        "shap__",
                        "",
                    ),

                "mean_absolute_shap":
                    values.abs().mean(),
            }
        )

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result.to_csv(
        TABLE_DIR
        / "table_shap_importance.csv",
        index=False,
    )

    return result


def copy_ai_summary_tables():

    repeat_dir = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "council_v2_repeats"
        / "summary"
    )

    role_dir = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "role_ablation"
        / "summary"
    )

    files = {
        repeat_dir
        / "run_summary.csv":
            "table_council_repeatability.csv",

        repeat_dir
        / "model_agreement.csv":
            "table_council_model_agreement.csv",

        role_dir
        / "assignment_summary.csv":
            "table_role_ablation.csv",

        role_dir
        / "model_role_summary.csv":
            "table_model_role_agreement.csv",
    }

    for source, destination in files.items():

        if source.exists():

            data = pd.read_csv(
                source
            )

            data.to_csv(
                TABLE_DIR
                / destination,
                index=False,
            )


def build_key_results_json(
    case_table,
    metrics,
):

    mechanisms = (
        case_table[
            "wgs_reference_mechanism"
        ]
        .value_counts()
        .to_dict()
    )

    augmented = (
        metrics[
            metrics[
                "model"
            ]
            ==
            "AUGMENTED"
        ]
        .iloc[0]
        .to_dict()
    )

    summary = {
        "n_acquisition_targets":
            int(
                len(
                    case_table
                )
            ),

        "mechanism_counts":
            mechanisms,

        "xgboost_augmented":
            {
                key:
                    (
                        float(value)
                        if isinstance(
                            value,
                            (
                                int,
                                float,
                                np.floating,
                            ),
                        )
                        else value
                    )

                for key, value
                in augmented.items()
            },
    }

    with (
        TABLE_DIR
        / "paper_key_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )


def main():

    case_table = (
        build_case_table()
    )

    metrics = (
        calculate_model_metrics()
    )

    build_shap_table()

    copy_ai_summary_tables()

    build_key_results_json(
        case_table,
        metrics,
    )

    print(
        "\nPaper tables generated in:"
    )

    print(
        TABLE_DIR
    )

    print(
        "\nCase-level results:"
    )

    print(
        case_table[
            [
                "target_id",
                "clinical_interpretation",
                "direct_wgs_linked_sources",
                "top_xgboost_source_type",
                "top_xgboost_sample",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nXGBoost metrics:"
    )

    print(
        metrics.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()