from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from cpo.xgboost_model import (
    load_ml_dataset,
    make_xgboost_model,
)

from cpo.ml_dataset import (
    CORE_FEATURES,
    AUGMENTED_FEATURES,
)


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "days_before_acquisition":
        "Days before acquisition",

    "source_is_environment":
        "Environmental source",

    "same_cp_family_gx":
        "Same CP family (GeneXpert)",

    "source_gx_positive":
        "Source GeneXpert positive",

    "source_gx_family_count":
        "Number of source GX CP families",

    "same_cp_family_isolate":
        "Same isolate CP family",

    "same_exact_cp_gene_isolate":
        "Same exact CP gene",

    "same_species_any":
        "Same bacterial species",

    "same_st_any":
        "Same sequence type",
}


# ============================================================
# GROUPED OOF SHAP
# ============================================================


def build_oof_shap_explanations(
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build SHAP explanations using the same leave-one-target-out
    procedure used during model evaluation.

    The held-out target is NEVER used to train the model that
    explains its candidate rows.

    This is preferable to explaining the final all-data model
    when reporting patient-specific examples.
    """

    if feature_columns is None:

        feature_columns = (
            AUGMENTED_FEATURES
        )

    data = load_ml_dataset()

    targets = sorted(
        data[
            "target_id"
        ].unique()
    )

    outputs = []

    for fold_number, held_out_target in enumerate(
        targets,
        start=1,
    ):

        print(
            f"SHAP fold {fold_number}: "
            f"holding out {held_out_target}"
        )

        # ----------------------------------------------------
        # Grouped train/test split
        # ----------------------------------------------------

        train = data[
            data[
                "target_id"
            ]
            != held_out_target
        ].copy()

        test = data[
            data[
                "target_id"
            ]
            == held_out_target
        ].copy()

        X_train = (
            train[
                feature_columns
            ]
            .astype(float)
        )

        y_train = (
            train[
                "label_any_wgs_link"
            ]
            .astype(int)
        )

        X_test = (
            test[
                feature_columns
            ]
            .astype(float)
        )

        # ----------------------------------------------------
        # Fold-specific class weighting
        # ----------------------------------------------------

        positive_count = int(
            y_train.sum()
        )

        negative_count = int(
            len(y_train)
            - positive_count
        )

        if (
            positive_count == 0
            or negative_count == 0
        ):

            raise ValueError(
                f"Fold for {held_out_target} "
                "does not contain both classes."
            )

        scale_pos_weight = (
            negative_count
            /
            positive_count
        )

        # ----------------------------------------------------
        # Fit the same model used during evaluation
        # ----------------------------------------------------

        model = make_xgboost_model(
            scale_pos_weight=(
                scale_pos_weight
            ),
            random_state=(
                42
                + fold_number
            ),
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        explainer = (
            shap.TreeExplainer(
                model
            )
        )

        explanation = (
            explainer(
                X_test
            )
        )

        shap_values = np.asarray(
            explanation.values
        )

        if (
            shap_values.ndim
            != 2
        ):

            raise ValueError(
                "Unexpected SHAP output dimensions: "
                f"{shap_values.shape}"
            )

        # ----------------------------------------------------
        # Store original candidate information
        # ----------------------------------------------------

        fold_output = (
            test.copy()
            .reset_index(
                drop=True
            )
        )

        fold_output[
            "explanation_fold"
        ] = fold_number

        fold_output[
            "held_out_target"
        ] = held_out_target

        fold_output[
            "predicted_probability"
        ] = probabilities

        fold_output[
            "rank_within_target"
        ] = (
            fold_output[
                "predicted_probability"
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )

        # ----------------------------------------------------
        # Store one SHAP column per feature
        # ----------------------------------------------------

        for feature_index, feature in enumerate(
            feature_columns
        ):

            fold_output[
                f"shap__{feature}"
            ] = (
                shap_values[
                    :,
                    feature_index
                ]
            )

        # ----------------------------------------------------
        # Strongest positive / negative feature
        # ----------------------------------------------------

        def strongest_positive(
            row: pd.Series,
        ):

            contributions = {
                feature:
                    row[
                        f"shap__{feature}"
                    ]

                for feature
                in feature_columns
            }

            positive = {
                feature:
                    value

                for feature, value
                in contributions.items()

                if value > 0
            }

            if not positive:
                return np.nan

            return max(
                positive,
                key=positive.get,
            )

        def strongest_negative(
            row: pd.Series,
        ):

            contributions = {
                feature:
                    row[
                        f"shap__{feature}"
                    ]

                for feature
                in feature_columns
            }

            negative = {
                feature:
                    value

                for feature, value
                in contributions.items()

                if value < 0
            }

            if not negative:
                return np.nan

            return min(
                negative,
                key=negative.get,
            )

        fold_output[
            "strongest_positive_feature"
        ] = fold_output.apply(
            strongest_positive,
            axis=1,
        )

        fold_output[
            "strongest_negative_feature"
        ] = fold_output.apply(
            strongest_negative,
            axis=1,
        )

        outputs.append(
            fold_output
        )

    result = pd.concat(
        outputs,
        ignore_index=True,
    )

    if len(result) != len(data):

        raise ValueError(
            "OOF SHAP output does not match "
            "the ML dataset row count."
        )

    return result


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================


def build_global_shap_summary(
    oof_shap: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate mean absolute out-of-fold SHAP importance.
    """

    if feature_columns is None:

        feature_columns = (
            AUGMENTED_FEATURES
        )

    rows = []

    for feature in feature_columns:

        shap_column = (
            f"shap__{feature}"
        )

        values = (
            oof_shap[
                shap_column
            ]
            .astype(float)
        )

        rows.append(
            {
                "feature":
                    feature,

                "feature_label":
                    FEATURE_LABELS.get(
                        feature,
                        feature,
                    ),

                "mean_abs_shap":
                    float(
                        values
                        .abs()
                        .mean()
                    ),

                "median_abs_shap":
                    float(
                        values
                        .abs()
                        .median()
                    ),

                "mean_signed_shap":
                    float(
                        values.mean()
                    ),
            }
        )

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# TOP CANDIDATES
# ============================================================


def build_patient_rankings(
    oof_shap: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Create clean top-K ranked candidate table for every target.
    """

    outputs = []

    for target_id in sorted(
        oof_shap[
            "target_id"
        ].unique()
    ):

        subset = (
            oof_shap[
                oof_shap[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                "rank_within_target"
            )
            .head(
                top_k
            )
            .copy()
        )

        outputs.append(
            subset
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ============================================================
# EXPLAIN TRUE POSITIVE PAIRS
# ============================================================


def build_positive_pair_explanations(
    oof_shap: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a concise explanation table for all true linked pairs.
    """

    if feature_columns is None:

        feature_columns = (
            AUGMENTED_FEATURES
        )

    positives = (
        oof_shap[
            oof_shap[
                "label_any_wgs_link"
            ]
            == 1
        ]
        .copy()
    )

    rows = []

    for _, row in positives.iterrows():

        shap_pairs = [
            (
                feature,
                float(
                    row[
                        f"shap__{feature}"
                    ]
                ),
            )

            for feature
            in feature_columns
        ]

        positive_drivers = sorted(
            [
                (
                    feature,
                    value,
                )

                for feature, value
                in shap_pairs

                if value > 0
            ],
            key=lambda item:
                item[1],
            reverse=True,
        )

        negative_drivers = sorted(
            [
                (
                    feature,
                    value,
                )

                for feature, value
                in shap_pairs

                if value < 0
            ],
            key=lambda item:
                item[1],
        )

        top_positive = (
            "; ".join(
                [
                    (
                        f"{FEATURE_LABELS.get(feature, feature)}"
                        f" ({value:.3f})"
                    )

                    for feature, value
                    in positive_drivers[
                        :3
                    ]
                ]
            )
            if positive_drivers
            else np.nan
        )

        top_negative = (
            "; ".join(
                [
                    (
                        f"{FEATURE_LABELS.get(feature, feature)}"
                        f" ({value:.3f})"
                    )

                    for feature, value
                    in negative_drivers[
                        :3
                    ]
                ]
            )
            if negative_drivers
            else np.nan
        )

        rows.append(
            {
                "target_id":
                    row[
                        "target_id"
                    ],

                "target_sample_id":
                    row.get(
                        "target_sample_id",
                        np.nan,
                    ),

                "source_type":
                    row[
                        "source_type"
                    ],

                "source_subject_id":
                    row[
                        "source_subject_id"
                    ],

                "source_sample_id":
                    row[
                        "source_sample_id"
                    ],

                "days_before_acquisition":
                    row[
                        "days_before_acquisition"
                    ],

                "predicted_probability":
                    row[
                        "predicted_probability"
                    ],

                "rank_within_target":
                    row[
                        "rank_within_target"
                    ],

                "clonal_link":
                    row.get(
                        "label_clonal_link",
                        np.nan,
                    ),

                "plasmid_link":
                    row.get(
                        "label_plasmid_link",
                        np.nan,
                    ),

                "top_positive_shap_drivers":
                    top_positive,

                "top_negative_shap_drivers":
                    top_negative,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "target_id",
                "rank_within_target",
            ]
        )
        .reset_index(
            drop=True
        )
    )