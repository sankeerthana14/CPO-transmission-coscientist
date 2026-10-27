from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from cpo.config import PROCESSED_DATA_DIR

from cpo.ml_dataset import (
    CORE_FEATURES,
    AUGMENTED_FEATURES,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================
#
# IMPORTANT:
#
# We deliberately use a small, conservative tree model.
#
# There are only 14 positive relationships, so aggressive
# hyperparameter tuning would be inappropriate and would very
# easily overfit this dataset.
# ============================================================


def make_xgboost_model(
    scale_pos_weight: float,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Construct a conservative XGBoost binary classifier.

    scale_pos_weight is computed separately inside each
    training fold.
    """

    return XGBClassifier(
        objective="binary:logistic",

        # Small trees because positive sample count is tiny
        max_depth=2,

        learning_rate=0.03,

        n_estimators=150,

        min_child_weight=2,

        subsample=0.8,

        colsample_bytree=0.8,

        gamma=0.1,

        reg_alpha=0.25,

        reg_lambda=5.0,

        max_delta_step=1,

        scale_pos_weight=scale_pos_weight,

        eval_metric="aucpr",

        tree_method="hist",

        random_state=random_state,

        n_jobs=4,

        verbosity=0,
    )


# ============================================================
# LOAD DATA
# ============================================================


def load_ml_dataset() -> pd.DataFrame:
    """
    Load the frozen Stage-5 ML dataset.
    """

    input_file = (
        PROCESSED_DATA_DIR
        / "ml_any_wgs_link_dataset.csv"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            "\nml_any_wgs_link_dataset.csv "
            "was not found.\n"
            "Run:\n\n"
            "PYTHONPATH=src "
            "python scripts/07_build_ml_dataset.py"
        )

    data = pd.read_csv(
        input_file
    )

    required = [
        "ml_row_id",
        "target_id",
        "label_any_wgs_link",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "ML dataset missing required columns:\n"
            f"{missing}"
        )

    return data


# ============================================================
# FOLD METRICS
# ============================================================


def calculate_fold_ranking_metrics(
    predictions: pd.DataFrame,
) -> dict:
    """
    Calculate metrics for one held-out acquisition patient.

    Ranking metrics are meaningful only if the held-out
    patient contains at least one linked source.
    """

    y_true = (
        predictions[
            "y_true"
        ]
        .astype(int)
        .to_numpy()
    )

    y_score = (
        predictions[
            "predicted_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    positive_count = int(
        y_true.sum()
    )

    negative_count = int(
        len(y_true)
        - positive_count
    )

    result = {
        "test_rows":
            len(
                predictions
            ),

        "test_positive":
            positive_count,

        "test_negative":
            negative_count,

        "average_precision":
            np.nan,

        "roc_auc":
            np.nan,

        "best_positive_rank":
            np.nan,

        "mean_positive_rank":
            np.nan,

        "mrr":
            np.nan,

        "hit_at_1":
            np.nan,

        "hit_at_3":
            np.nan,

        "hit_at_5":
            np.nan,

        "recall_at_3":
            np.nan,

        "recall_at_5":
            np.nan,

        "max_predicted_probability":
            float(
                np.max(
                    y_score
                )
            ),

        "mean_predicted_probability":
            float(
                np.mean(
                    y_score
                )
            ),
    }

    # --------------------------------------------------------
    # Negative-only held-out patient
    # --------------------------------------------------------

    if positive_count == 0:

        return result

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    result[
        "average_precision"
    ] = float(
        average_precision_score(
            y_true,
            y_score,
        )
    )

    if negative_count > 0:

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    positive_ranks = (
        predictions.loc[
            predictions[
                "y_true"
            ]
            == 1,
            "rank_within_target",
        ]
        .astype(int)
        .tolist()
    )

    best_rank = min(
        positive_ranks
    )

    result[
        "best_positive_rank"
    ] = int(
        best_rank
    )

    result[
        "mean_positive_rank"
    ] = float(
        np.mean(
            positive_ranks
        )
    )

    result[
        "mrr"
    ] = (
        1.0
        /
        best_rank
    )

    result[
        "hit_at_1"
    ] = int(
        best_rank <= 1
    )

    result[
        "hit_at_3"
    ] = int(
        best_rank <= 3
    )

    result[
        "hit_at_5"
    ] = int(
        best_rank <= 5
    )

    result[
        "recall_at_3"
    ] = (
        sum(
            rank <= 3
            for rank
            in positive_ranks
        )
        /
        positive_count
    )

    result[
        "recall_at_5"
    ] = (
        sum(
            rank <= 5
            for rank
            in positive_ranks
        )
        /
        positive_count
    )

    return result


# ============================================================
# ONE MODEL — GROUPED CV
# ============================================================


def run_grouped_cv(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> dict:
    """
    Leave-one-acquisition-patient-out cross-validation.

    Each target_id is held out completely.
    """

    missing_features = [
        column
        for column
        in feature_columns
        if column not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"{model_name} is missing features:\n"
            f"{missing_features}"
        )

    targets = sorted(
        data[
            "target_id"
        ]
        .unique()
    )

    prediction_frames = []

    fold_metrics = []

    importance_frames = []

    for fold_number, held_out_target in enumerate(
        targets,
        start=1,
    ):

        # ----------------------------------------------------
        # Split by TARGET PATIENT
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

        y_test = (
            test[
                "label_any_wgs_link"
            ]
            .astype(int)
        )

        # ----------------------------------------------------
        # Fold-specific class imbalance weighting
        # ----------------------------------------------------

        train_positive = int(
            y_train.sum()
        )

        train_negative = int(
            len(
                y_train
            )
            - train_positive
        )

        if (
            train_positive == 0
            or train_negative == 0
        ):

            raise ValueError(
                f"Fold {fold_number} does not "
                "contain both training classes."
            )

        scale_pos_weight = (
            train_negative
            /
            train_positive
        )

        # ----------------------------------------------------
        # Fit
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
        # Store held-out predictions
        # ----------------------------------------------------

        fold_predictions = (
            test.copy()
        )

        fold_predictions[
            "model_name"
        ] = model_name

        fold_predictions[
            "fold"
        ] = fold_number

        fold_predictions[
            "held_out_target"
        ] = held_out_target

        fold_predictions[
            "y_true"
        ] = (
            y_test
            .to_numpy()
        )

        fold_predictions[
            "predicted_probability"
        ] = probabilities

        # Rank 1 = highest model score
        fold_predictions[
            "rank_within_target"
        ] = (
            fold_predictions[
                "predicted_probability"
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )

        prediction_frames.append(
            fold_predictions
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        metrics = (
            calculate_fold_ranking_metrics(
                fold_predictions
            )
        )

        metrics.update(
            {
                "model_name":
                    model_name,

                "fold":
                    fold_number,

                "held_out_target":
                    held_out_target,

                "train_rows":
                    len(train),

                "train_positive":
                    train_positive,

                "train_negative":
                    train_negative,

                "scale_pos_weight":
                    scale_pos_weight,
            }
        )

        fold_metrics.append(
            metrics
        )

        # ----------------------------------------------------
        # Feature importance — gain
        # ----------------------------------------------------

        raw_importance = (
            model
            .get_booster()
            .get_score(
                importance_type="gain"
            )
        )

        importance_rows = []

        for feature in feature_columns:

            importance_rows.append(
                {
                    "model_name":
                        model_name,

                    "fold":
                        fold_number,

                    "held_out_target":
                        held_out_target,

                    "feature":
                        feature,

                    "gain":
                        float(
                            raw_importance.get(
                                feature,
                                0.0,
                            )
                        ),
                }
            )

        importance_frames.append(
            pd.DataFrame(
                importance_rows
            )
        )

    # ========================================================
    # COMBINE OOF RESULTS
    # ========================================================

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fold_metrics_df = pd.DataFrame(
        fold_metrics
    )

    feature_importance = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    # ========================================================
    # POOLED OUT-OF-FOLD METRICS
    # ========================================================

    pooled_y = (
        predictions[
            "y_true"
        ]
        .astype(int)
        .to_numpy()
    )

    pooled_score = (
        predictions[
            "predicted_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    pooled_ap = (
        average_precision_score(
            pooled_y,
            pooled_score,
        )
    )

    pooled_auc = (
        roc_auc_score(
            pooled_y,
            pooled_score,
        )
    )

    # ========================================================
    # TARGET-LEVEL RANKING SUMMARY
    # ========================================================

    positive_folds = (
        fold_metrics_df[
            fold_metrics_df[
                "test_positive"
            ] > 0
        ]
        .copy()
    )

    negative_folds = (
        fold_metrics_df[
            fold_metrics_df[
                "test_positive"
            ] == 0
        ]
        .copy()
    )

    summary = {
        "model_name":
            model_name,

        "n_features":
            len(
                feature_columns
            ),

        "pooled_oof_auprc":
            float(
                pooled_ap
            ),

        "pooled_oof_auroc":
            float(
                pooled_auc
            ),

        "mean_positive_target_auprc":
            float(
                positive_folds[
                    "average_precision"
                ].mean()
            ),

        "mean_positive_target_auroc":
            float(
                positive_folds[
                    "roc_auc"
                ].mean()
            ),

        "mean_mrr":
            float(
                positive_folds[
                    "mrr"
                ].mean()
            ),

        "hit_at_1":
            float(
                positive_folds[
                    "hit_at_1"
                ].mean()
            ),

        "hit_at_3":
            float(
                positive_folds[
                    "hit_at_3"
                ].mean()
            ),

        "hit_at_5":
            float(
                positive_folds[
                    "hit_at_5"
                ].mean()
            ),

        "mean_recall_at_3":
            float(
                positive_folds[
                    "recall_at_3"
                ].mean()
            ),

        "mean_recall_at_5":
            float(
                positive_folds[
                    "recall_at_5"
                ].mean()
            ),

        "median_best_positive_rank":
            float(
                positive_folds[
                    "best_positive_rank"
                ].median()
            ),

        "mean_max_score_negative_targets":
            float(
                negative_folds[
                    "max_predicted_probability"
                ].mean()
            ),

        "positive_test_targets":
            len(
                positive_folds
            ),

        "negative_only_test_targets":
            len(
                negative_folds
            ),
    }

    # ========================================================
    # AVERAGE FEATURE IMPORTANCE
    # ========================================================

    mean_importance = (
        feature_importance
        .groupby(
            [
                "model_name",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_gain=(
                "gain",
                "mean",
            ),

            median_gain=(
                "gain",
                "median",
            ),
        )
        .sort_values(
            "mean_gain",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return {
        "predictions":
            predictions,

        "fold_metrics":
            fold_metrics_df,

        "feature_importance_by_fold":
            feature_importance,

        "mean_feature_importance":
            mean_importance,

        "summary":
            pd.DataFrame(
                [
                    summary
                ]
            ),
    }


# ============================================================
# FIT FINAL FULL-DATA MODEL
# ============================================================


def fit_final_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> XGBClassifier:
    """
    Fit one final model on all Stage-5 rows.

    This model is NOT used to report validation performance.
    It is saved for later source-ranking / agent inference.
    """

    X = (
        data[
            feature_columns
        ]
        .astype(float)
    )

    y = (
        data[
            "label_any_wgs_link"
        ]
        .astype(int)
    )

    positives = int(
        y.sum()
    )

    negatives = (
        len(y)
        -
        positives
    )

    if (
        positives == 0
        or negatives == 0
    ):

        raise ValueError(
            "Final model requires both classes."
        )

    scale_pos_weight = (
        negatives
        /
        positives
    )

    model = make_xgboost_model(
        scale_pos_weight=(
            scale_pos_weight
        ),
        random_state=(
            random_state
        ),
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# RUN BOTH MODEL VARIANTS
# ============================================================


def train_and_evaluate_xgboost() -> dict:
    """
    Evaluate both:
        CORE
        AUGMENTED

    using leave-one-target-out cross-validation.
    """

    data = (
        load_ml_dataset()
    )

    core = run_grouped_cv(
        data=data,
        feature_columns=(
            CORE_FEATURES
        ),
        model_name=(
            "CORE"
        ),
    )

    augmented = run_grouped_cv(
        data=data,
        feature_columns=(
            AUGMENTED_FEATURES
        ),
        model_name=(
            "AUGMENTED"
        ),
    )

    summaries = pd.concat(
        [
            core[
                "summary"
            ],
            augmented[
                "summary"
            ],
        ],
        ignore_index=True,
    )

    predictions = pd.concat(
        [
            core[
                "predictions"
            ],
            augmented[
                "predictions"
            ],
        ],
        ignore_index=True,
    )

    fold_metrics = pd.concat(
        [
            core[
                "fold_metrics"
            ],
            augmented[
                "fold_metrics"
            ],
        ],
        ignore_index=True,
    )

    feature_importance = pd.concat(
        [
            core[
                "mean_feature_importance"
            ],
            augmented[
                "mean_feature_importance"
            ],
        ],
        ignore_index=True,
    )

    feature_importance_by_fold = pd.concat(
        [
            core[
                "feature_importance_by_fold"
            ],
            augmented[
                "feature_importance_by_fold"
            ],
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Fit full-data models for later agent inference
    # --------------------------------------------------------

    final_core_model = (
        fit_final_model(
            data=data,
            feature_columns=(
                CORE_FEATURES
            ),
            random_state=42,
        )
    )

    final_augmented_model = (
        fit_final_model(
            data=data,
            feature_columns=(
                AUGMENTED_FEATURES
            ),
            random_state=43,
        )
    )

    return {
        "data":
            data,

        "summaries":
            summaries,

        "predictions":
            predictions,

        "fold_metrics":
            fold_metrics,

        "feature_importance":
            feature_importance,

        "feature_importance_by_fold":
            feature_importance_by_fold,

        "final_core_model":
            final_core_model,

        "final_augmented_model":
            final_augmented_model,
    }