from __future__ import annotations

import numpy as np
import pandas as pd

from cpo.config import PROCESSED_DATA_DIR


# ============================================================
# MODEL FEATURE SETS
# ============================================================
#
# We deliberately define the usable features here instead of
# allowing XGBoost to consume arbitrary dataframe columns.
#
# This prevents accidental leakage from WGS linkage outputs.
# ============================================================


# ------------------------------------------------------------
# CORE FEATURE SET
# ------------------------------------------------------------
#
# Uses chronology, source type, and sample-level GeneXpert
# evidence.
#
# This does NOT use species/ST or isolate-level genomic
# matching.
# ------------------------------------------------------------

CORE_FEATURES = [
    "days_before_acquisition",
    "source_is_environment",
    "same_cp_family_gx",
    "source_gx_positive",
    "source_gx_family_count",
]


# ------------------------------------------------------------
# TYPING-AUGMENTED FEATURE SET
# ------------------------------------------------------------
#
# Adds isolate-level genomic typing information:
#
#   - isolate CP-family compatibility
#   - exact CP-gene compatibility
#   - bacterial species compatibility
#   - sequence-type compatibility
#
# These are NOT direct clonal/plasmid linkage variables.
# ------------------------------------------------------------

AUGMENTED_FEATURES = [
    *CORE_FEATURES,

    "same_cp_family_isolate",
    "same_exact_cp_gene_isolate",
    "same_species_any",
    "same_st_any",
]


# This is the main feature set we will use in Stage 6.
MODEL_FEATURES = AUGMENTED_FEATURES


# ============================================================
# COLUMNS THAT MUST NEVER BE MODEL FEATURES
# ============================================================
#
# These either contain the outcome directly or are derived
# from the WGS linkage analysis.
# ============================================================

FORBIDDEN_WGS_FEATURES = [
    "clonal_link",
    "plasmid_link",
    "any_wgs_link",

    "clonal_forward",
    "clonal_reverse",

    "plasmid_forward",
    "plasmid_reverse",

    "clonal_link_record_count",
    "plasmid_link_record_count",

    "clonal_evidence_pairs",
    "plasmid_evidence_pairs",

    "shared_plasmid_cp_genes",

    "clonal_link_status",
    "plasmid_link_status",

    "wgs_link_mechanism",
    "wgs_direction_status",

    "target_wgs_ids",
    "source_wgs_ids",

    "target_wgs_isolate_count",
    "source_wgs_isolate_count",

    "target_wgs_resolution",
    "source_wgs_resolution",

    "wgs_pair_evaluable",
]


# ============================================================
# BOOLEAN HANDLING
# ============================================================


def coerce_bool_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Robustly convert CSV-loaded boolean-like values to bool.

    Handles:
        True / False
        "True" / "False"
        1 / 0
        "1" / "0"
        yes / no
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    mapping = {
        True: True,
        False: False,

        1: True,
        0: False,

        "1": True,
        "0": False,

        "true": True,
        "false": False,

        "yes": True,
        "no": False,

        "y": True,
        "n": False,
    }

    def convert(value):

        if pd.isna(value):
            return np.nan

        if isinstance(
            value,
            str,
        ):
            value = (
                value
                .strip()
                .lower()
            )

        return mapping.get(
            value,
            np.nan,
        )

    converted = series.apply(
        convert
    )

    invalid = (
        converted.isna()
        &
        series.notna()
    )

    if invalid.any():

        bad_values = (
            series[
                invalid
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Could not convert column "
            f"{column_name!r} to boolean.\n"
            f"Unexpected values: "
            f"{bad_values}"
        )

    return (
        converted
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# FEATURE MANIFEST
# ============================================================


def build_feature_manifest() -> pd.DataFrame:
    """
    Document every candidate model feature.
    """

    rows = [
        {
            "feature":
                "days_before_acquisition",

            "core_model":
                True,

            "augmented_model":
                True,

            "category":
                "temporal",

            "description":
                (
                    "Number of calendar days between "
                    "source sampling and target acquisition."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "source_is_environment",

            "core_model":
                True,

            "augmented_model":
                True,

            "category":
                "epidemiology",

            "description":
                (
                    "1 for environmental source; "
                    "0 for Known_CPO_pos patient source."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "same_cp_family_gx",

            "core_model":
                True,

            "augmented_model":
                True,

            "category":
                "GeneXpert",

            "description":
                (
                    "Target and source GeneXpert results "
                    "share at least one CP family."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "source_gx_positive",

            "core_model":
                True,

            "augmented_model":
                True,

            "category":
                "GeneXpert",

            "description":
                (
                    "Source GeneXpert contains at least "
                    "one recognised carbapenemase family."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "source_gx_family_count",

            "core_model":
                True,

            "augmented_model":
                True,

            "category":
                "GeneXpert",

            "description":
                (
                    "Number of CP families detected in "
                    "the source GeneXpert result."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "same_cp_family_isolate",

            "core_model":
                False,

            "augmented_model":
                True,

            "category":
                "isolate_typing",

            "description":
                (
                    "Target and source cultured/genomic "
                    "isolates share a CP-gene family."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "same_exact_cp_gene_isolate",

            "core_model":
                False,

            "augmented_model":
                True,

            "category":
                "isolate_typing",

            "description":
                (
                    "Target and source isolates share an "
                    "exact CP gene such as blaOXA-181."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "same_species_any",

            "core_model":
                False,

            "augmented_model":
                True,

            "category":
                "isolate_typing",

            "description":
                (
                    "At least one source isolate has "
                    "the same bacterial species as target."
                ),

            "direct_wgs_linkage_feature":
                False,
        },

        {
            "feature":
                "same_st_any",

            "core_model":
                False,

            "augmented_model":
                True,

            "category":
                "isolate_typing",

            "description":
                (
                    "Target and source share the same "
                    "known sequence type within the same "
                    "species."
                ),

            "direct_wgs_linkage_feature":
                False,
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# DUPLICATE HANDLING
# ============================================================


def get_ml_deduplication_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Define the biological unit used for the ML dataset.

    We do not want repeated Excel isolate slots describing
    exactly the same biological candidate to overweight the
    model.

    WGS IDs are included so biologically distinct resolved
    isolates are not accidentally merged.
    """

    candidate_columns = [
        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "source_species_st",
        "source_cp_gene",

        "target_wgs_ids",
        "source_wgs_ids",
    ]

    return [
        column
        for column
        in candidate_columns
        if column
        in data.columns
    ]


def check_duplicate_label_consistency(
    data: pd.DataFrame,
    dedupe_columns: list[str],
):
    """
    Ensure duplicate biological signatures do not carry
    contradictory WGS labels.
    """

    grouped = (
        data
        .groupby(
            dedupe_columns,
            dropna=False,
        )
        .agg(
            any_link_nunique=(
                "any_wgs_link",
                "nunique",
            ),

            clonal_nunique=(
                "clonal_link",
                "nunique",
            ),

            plasmid_nunique=(
                "plasmid_link",
                "nunique",
            ),
        )
        .reset_index()
    )

    inconsistent = grouped[
        (
            grouped[
                "any_link_nunique"
            ] > 1
        )
        |
        (
            grouped[
                "clonal_nunique"
            ] > 1
        )
        |
        (
            grouped[
                "plasmid_nunique"
            ] > 1
        )
    ]

    if not inconsistent.empty:

        raise ValueError(
            "Duplicate biological candidate rows "
            "contain inconsistent WGS labels.\n\n"
            f"{inconsistent.to_string(index=False)}"
        )


# ============================================================
# GROUPED CROSS-VALIDATION PLAN
# ============================================================


def build_group_cv_plan(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct leave-one-target-patient-out folds.

    Each fold:
        test  = all candidate rows for one acquisition patient
        train = candidate rows for all other acquisition patients

    This avoids leaking candidate structure from the same
    target patient into both training and testing.
    """

    rows = []

    targets = sorted(
        dataset[
            "target_id"
        ].unique()
    )

    for fold_number, target_id in enumerate(
        targets,
        start=1,
    ):

        test = dataset[
            dataset[
                "target_id"
            ]
            == target_id
        ]

        train = dataset[
            dataset[
                "target_id"
            ]
            != target_id
        ]

        test_positive = int(
            test[
                "label_any_wgs_link"
            ].sum()
        )

        train_positive = int(
            train[
                "label_any_wgs_link"
            ].sum()
        )

        test_negative = (
            len(test)
            - test_positive
        )

        train_negative = (
            len(train)
            - train_positive
        )

        rows.append(
            {
                "fold":
                    fold_number,

                "held_out_target":
                    target_id,

                "train_rows":
                    len(train),

                "train_positive":
                    train_positive,

                "train_negative":
                    train_negative,

                "test_rows":
                    len(test),

                "test_positive":
                    test_positive,

                "test_negative":
                    test_negative,

                "train_has_both_classes":
                    (
                        train_positive > 0
                        and
                        train_negative > 0
                    ),

                "test_has_positive":
                    (
                        test_positive > 0
                    ),

                "ranking_metrics_applicable":
                    (
                        test_positive > 0
                    ),

                "fold_type":
                    (
                        "POSITIVE_TARGET"
                        if test_positive > 0
                        else "NEGATIVE_ONLY_TARGET"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET / SOURCE LABEL SUMMARY
# ============================================================


def build_target_distribution(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize positive and negative labels by target/source type.
    """

    summary = (
        dataset
        .groupby(
            [
                "target_id",
                "source_type",
            ]
        )
        .agg(
            candidate_count=(
                "ml_row_id",
                "size",
            ),

            positive_any_link=(
                "label_any_wgs_link",
                "sum",
            ),

            positive_clonal=(
                "label_clonal_link",
                "sum",
            ),

            positive_plasmid=(
                "label_plasmid_link",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "negative_unlinked"
    ] = (
        summary[
            "candidate_count"
        ]
        -
        summary[
            "positive_any_link"
        ]
    )

    summary[
        "positive_rate"
    ] = (
        summary[
            "positive_any_link"
        ]
        /
        summary[
            "candidate_count"
        ]
    )

    return summary


# ============================================================
# MASTER DATASET BUILDER
# ============================================================


def build_ml_dataset():
    """
    Build a leakage-controlled ML dataset for predicting:

        any_wgs_link =
            clonal_link OR plasmid_link

    Operational assumptions
    -----------------------
    1. Only WGS-evaluable pairs are included.
    2. Only strictly PRIOR source samples are included.
    3. Within WGS-evaluable pairs:
         absent from both link tables -> unlinked / negative.
    4. Same-day ambiguous candidates are excluded.
    5. Direct WGS linkage information is label-only and is
       never permitted into MODEL_FEATURES.
    """

    input_file = (
        PROCESSED_DATA_DIR
        / "wgs_labeled_pairs.csv"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            "\nwgs_labeled_pairs.csv was not found.\n"
            "Run:\n\n"
            "PYTHONPATH=src "
            "python scripts/06_attach_wgs_links.py"
        )

    data = pd.read_csv(
        input_file,
        parse_dates=[
            "target_acquisition_date",
            "source_date",
        ],
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "pair_id",

        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",
        "is_prior",

        "wgs_pair_evaluable",

        "any_wgs_link",
        "clonal_link",
        "plasmid_link",

        "same_cp_family_gx",
        "source_gx_positive",
        "source_gx_family_count",

        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",
    ]

    missing = [
        column
        for column
        in required_columns
        if column
        not in data.columns
    ]

    if missing:

        raise ValueError(
            "wgs_labeled_pairs.csv is missing "
            "required columns:\n"
            f"{missing}"
        )

    # ========================================================
    # BOOLEAN NORMALIZATION
    # ========================================================

    boolean_columns = [
        "is_prior",
        "wgs_pair_evaluable",

        "any_wgs_link",
        "clonal_link",
        "plasmid_link",

        "same_cp_family_gx",
        "source_gx_positive",

        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",
    ]

    for column in boolean_columns:

        data[
            column
        ] = coerce_bool_series(
            data[
                column
            ],
            column_name=column,
        )

    # ========================================================
    # STRICT ML ELIGIBILITY FILTER
    # ========================================================
    #
    # Keep only:
    #
    #   - WGS-evaluable pairs
    #   - source sampled BEFORE acquisition
    #
    # Same-day candidates are excluded from training because
    # causal direction is ambiguous.
    # ========================================================

    eligible = data[
        data[
            "wgs_pair_evaluable"
        ]
        &
        data[
            "is_prior"
        ]
    ].copy()

    if eligible.empty:

        raise ValueError(
            "No WGS-evaluable prior candidate pairs "
            "remain after filtering."
        )

    # ========================================================
    # DEDUPLICATE BIOLOGICAL CANDIDATES
    # ========================================================

    dedupe_columns = (
        get_ml_deduplication_columns(
            eligible
        )
    )

    check_duplicate_label_consistency(
        data=eligible,
        dedupe_columns=dedupe_columns,
    )

    eligible[
        "ml_duplicate_group_size"
    ] = (
        eligible
        .groupby(
            dedupe_columns,
            dropna=False,
        )[
            "pair_id"
        ]
        .transform(
            "size"
        )
    )

    duplicate_rows = eligible[
        eligible[
            "ml_duplicate_group_size"
        ] > 1
    ].copy()

    rows_before_deduplication = (
        len(
            eligible
        )
    )

    eligible = (
        eligible
        .sort_values(
            [
                "target_id",
                "source_type",
                "days_before_acquisition",
                "pair_id",
            ]
        )
        .drop_duplicates(
            subset=dedupe_columns,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    rows_after_deduplication = (
        len(
            eligible
        )
    )

    # ========================================================
    # CREATE ML LABELS
    # ========================================================

    eligible[
        "label_any_wgs_link"
    ] = (
        eligible[
            "any_wgs_link"
        ]
        .astype(int)
    )

    # Keep these two labels for diagnostics only.
    # We are NOT training separate models from them yet.

    eligible[
        "label_clonal_link"
    ] = (
        eligible[
            "clonal_link"
        ]
        .astype(int)
    )

    eligible[
        "label_plasmid_link"
    ] = (
        eligible[
            "plasmid_link"
        ]
        .astype(int)
    )

    # ========================================================
    # SOURCE TYPE ENCODING
    # ========================================================

    eligible[
        "source_is_environment"
    ] = (
        eligible[
            "source_type"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "environment"
        )
        .astype(int)
    )

    # ========================================================
    # CONVERT BOOLEAN FEATURES TO 0/1
    # ========================================================

    binary_model_features = [
        "same_cp_family_gx",
        "source_gx_positive",

        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",
    ]

    for column in binary_model_features:

        eligible[
            column
        ] = (
            eligible[
                column
            ]
            .astype(int)
        )

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    eligible[
        "days_before_acquisition"
    ] = pd.to_numeric(
        eligible[
            "days_before_acquisition"
        ],
        errors="raise",
    )

    eligible[
        "source_gx_family_count"
    ] = pd.to_numeric(
        eligible[
            "source_gx_family_count"
        ],
        errors="raise",
    )

    # ========================================================
    # SANITY CHECK: PRIOR ONLY
    # ========================================================

    if (
        eligible[
            "days_before_acquisition"
        ] <= 0
    ).any():

        raise ValueError(
            "ML dataset contains a non-prior source. "
            "Same-day and future rows must not be used."
        )

    # ========================================================
    # SANITY CHECK: NO MISSING MODEL FEATURES
    # ========================================================

    missing_feature_counts = (
        eligible[
            MODEL_FEATURES
        ]
        .isna()
        .sum()
    )

    if (
        missing_feature_counts
        > 0
    ).any():

        problem = (
            missing_feature_counts[
                missing_feature_counts > 0
            ]
        )

        raise ValueError(
            "Missing values exist in model features:\n"
            f"{problem.to_string()}"
        )

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    accidentally_allowed = (
        set(
            MODEL_FEATURES
        )
        &
        set(
            FORBIDDEN_WGS_FEATURES
        )
    )

    if accidentally_allowed:

        raise ValueError(
            "WGS leakage detected in MODEL_FEATURES:\n"
            f"{sorted(accidentally_allowed)}"
        )

    # ========================================================
    # ASSIGN STABLE ML ROW IDS
    # ========================================================

    eligible.insert(
        0,
        "ml_row_id",
        [
            f"MLPAIR_{index:06d}"
            for index in range(
                1,
                len(
                    eligible
                ) + 1,
            )
        ],
    )

    # ========================================================
    # TARGET-LEVEL COUNTS
    # ========================================================

    eligible[
        "target_candidate_count"
    ] = (
        eligible
        .groupby(
            "target_id"
        )[
            "ml_row_id"
        ]
        .transform(
            "size"
        )
    )

    eligible[
        "target_positive_count"
    ] = (
        eligible
        .groupby(
            "target_id"
        )[
            "label_any_wgs_link"
        ]
        .transform(
            "sum"
        )
    )

    # ========================================================
    # OVERALL LABEL VALIDATION
    # ========================================================

    positives = int(
        eligible[
            "label_any_wgs_link"
        ].sum()
    )

    negatives = (
        len(
            eligible
        )
        - positives
    )

    if positives == 0:

        raise ValueError(
            "ML dataset contains no positive WGS links."
        )

    if negatives == 0:

        raise ValueError(
            "ML dataset contains no unlinked negative pairs."
        )

    scale_pos_weight = (
        negatives
        /
        positives
    )

    # ========================================================
    # BUILD DIAGNOSTIC TABLES
    # ========================================================

    feature_manifest = (
        build_feature_manifest()
    )

    target_distribution = (
        build_target_distribution(
            eligible
        )
    )

    cv_plan = (
        build_group_cv_plan(
            eligible
        )
    )

    overall_summary = pd.DataFrame(
        [
            {
                "metric":
                    "input_wgs_rows",

                "value":
                    len(data),
            },

            {
                "metric":
                    "prior_wgs_evaluable_before_dedup",

                "value":
                    rows_before_deduplication,
            },

            {
                "metric":
                    "ml_rows_after_dedup",

                "value":
                    rows_after_deduplication,
            },

            {
                "metric":
                    "duplicate_rows_removed",

                "value":
                    (
                        rows_before_deduplication
                        -
                        rows_after_deduplication
                    ),
            },

            {
                "metric":
                    "positive_any_wgs_link",

                "value":
                    positives,
            },

            {
                "metric":
                    "negative_unlinked",

                "value":
                    negatives,
            },

            {
                "metric":
                    "positive_prevalence",

                "value":
                    (
                        positives
                        /
                        len(
                            eligible
                        )
                    ),
            },

            {
                "metric":
                    "overall_scale_pos_weight",

                "value":
                    scale_pos_weight,
            },

            {
                "metric":
                    "unique_target_patients",

                "value":
                    eligible[
                        "target_id"
                    ].nunique(),
            },

            {
                "metric":
                    "positive_target_patients",

                "value":
                    eligible[
                        eligible[
                            "label_any_wgs_link"
                        ]
                        == 1
                    ][
                        "target_id"
                    ].nunique(),
            },
        ]
    )

    # ========================================================
    # POSITIVE PAIR TABLE
    # ========================================================

    positive_pairs = eligible[
        eligible[
            "label_any_wgs_link"
        ]
        == 1
    ].copy()

    return {
        "dataset":
            eligible,

        "feature_manifest":
            feature_manifest,

        "target_distribution":
            target_distribution,

        "cv_plan":
            cv_plan,

        "overall_summary":
            overall_summary,

        "duplicate_rows":
            duplicate_rows,

        "positive_pairs":
            positive_pairs,

        "core_features":
            CORE_FEATURES,

        "augmented_features":
            AUGMENTED_FEATURES,

        "scale_pos_weight":
            scale_pos_weight,
    }