from pathlib import Path

import pandas as pd

from cpo.xgboost_model import (
    train_and_evaluate_xgboost,
)

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)


def main():

    print(
        "\nTraining and evaluating XGBoost..."
    )

    result = (
        train_and_evaluate_xgboost()
    )

    summaries = (
        result[
            "summaries"
        ]
    )

    predictions = (
        result[
            "predictions"
        ]
    )

    fold_metrics = (
        result[
            "fold_metrics"
        ]
    )

    feature_importance = (
        result[
            "feature_importance"
        ]
    )

    feature_importance_by_fold = (
        result[
            "feature_importance_by_fold"
        ]
    )

    core_model = (
        result[
            "final_core_model"
        ]
    )

    augmented_model = (
        result[
            "final_augmented_model"
        ]
    )

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "XGBOOST — LEAVE-ONE-TARGET-OUT RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        summaries.to_string(
            index=False
        )
    )

    # ========================================================
    # FOLD RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "PER-TARGET RESULTS"
    )

    print(
        "=" * 120
    )

    display_fold_columns = [
        "model_name",
        "held_out_target",

        "test_rows",
        "test_positive",

        "average_precision",
        "roc_auc",

        "best_positive_rank",
        "mean_positive_rank",

        "mrr",

        "hit_at_1",
        "hit_at_3",
        "hit_at_5",

        "recall_at_3",
        "recall_at_5",

        "max_predicted_probability",
    ]

    print(
        fold_metrics[
            display_fold_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # TOP-10 RANKINGS PER TARGET — AUGMENTED MODEL
    # ========================================================

    augmented_predictions = (
        predictions[
            predictions[
                "model_name"
            ]
            == "AUGMENTED"
        ]
        .copy()
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "AUGMENTED MODEL — TOP 10 CANDIDATES PER TARGET"
    )

    print(
        "=" * 120
    )

    ranking_columns = [
        "target_id",

        "rank_within_target",
        "predicted_probability",

        "y_true",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",

        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",
    ]

    top_rankings = []

    for target_id in sorted(
        augmented_predictions[
            "target_id"
        ].unique()
    ):

        subset = (
            augmented_predictions[
                augmented_predictions[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                "rank_within_target"
            )
            .head(10)
        )

        top_rankings.append(
            subset
        )

        print(
            f"\n{'=' * 28} "
            f"{target_id} "
            f"{'=' * 28}"
        )

        existing_columns = [
            column
            for column
            in ranking_columns
            if column
            in subset.columns
        ]

        print(
            subset[
                existing_columns
            ]
            .to_string(
                index=False
            )
        )

    top_rankings = pd.concat(
        top_rankings,
        ignore_index=True,
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "MEAN FEATURE IMPORTANCE (GAIN)"
    )

    print(
        "=" * 120
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    model_dir = (
        TABLE_DIR.parent
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    core_model_file = (
        model_dir
        / "xgboost_core.json"
    )

    augmented_model_file = (
        model_dir
        / "xgboost_augmented.json"
    )

    core_model.save_model(
        core_model_file
    )

    augmented_model.save_model(
        augmented_model_file
    )

    # ========================================================
    # SAVE TABLES / PREDICTIONS
    # ========================================================

    prediction_csv = (
        PROCESSED_DATA_DIR
        / "xgboost_oof_predictions.csv"
    )

    predictions.to_csv(
        prediction_csv,
        index=False,
    )

    summary_file = (
        TABLE_DIR
        / "table_06_xgboost_summary.xlsx"
    )

    summaries.to_excel(
        summary_file,
        index=False,
    )

    fold_file = (
        TABLE_DIR
        / "table_06_xgboost_fold_metrics.xlsx"
    )

    fold_metrics.to_excel(
        fold_file,
        index=False,
    )

    rankings_file = (
        TABLE_DIR
        / "table_06_xgboost_top10_rankings.xlsx"
    )

    top_rankings.to_excel(
        rankings_file,
        index=False,
    )

    importance_file = (
        TABLE_DIR
        / "table_06_xgboost_feature_importance.xlsx"
    )

    feature_importance.to_excel(
        importance_file,
        index=False,
    )

    importance_fold_file = (
        TABLE_DIR
        / "table_06_xgboost_feature_importance_by_fold.xlsx"
    )

    feature_importance_by_fold.to_excel(
        importance_fold_file,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 120
    )

    print(
        f"\nOOF predictions:\n"
        f"{prediction_csv}"
    )

    print(
        f"\nPerformance summary:\n"
        f"{summary_file}"
    )

    print(
        f"\nPer-target metrics:\n"
        f"{fold_file}"
    )

    print(
        f"\nTop-10 rankings:\n"
        f"{rankings_file}"
    )

    print(
        f"\nFeature importance:\n"
        f"{importance_file}"
    )

    print(
        f"\nCore model:\n"
        f"{core_model_file}"
    )

    print(
        f"\nAugmented model:\n"
        f"{augmented_model_file}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese probabilities should be interpreted "
        "as candidate-ranking scores, not calibrated "
        "probabilities of transmission."
        "\nWGS evidence remains the reference label."
    )


if __name__ == "__main__":
    main()