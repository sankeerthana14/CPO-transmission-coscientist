from cpo.ml_dataset import (
    build_ml_dataset,
    CORE_FEATURES,
    AUGMENTED_FEATURES,
)

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)


def main():

    print(
        "\nBuilding leakage-controlled "
        "XGBoost training dataset..."
    )

    result = (
        build_ml_dataset()
    )

    data = (
        result[
            "dataset"
        ]
    )

    feature_manifest = (
        result[
            "feature_manifest"
        ]
    )

    target_distribution = (
        result[
            "target_distribution"
        ]
    )

    cv_plan = (
        result[
            "cv_plan"
        ]
    )

    overall_summary = (
        result[
            "overall_summary"
        ]
    )

    duplicate_rows = (
        result[
            "duplicate_rows"
        ]
    )

    positive_pairs = (
        result[
            "positive_pairs"
        ]
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    positives = int(
        data[
            "label_any_wgs_link"
        ].sum()
    )

    negatives = (
        len(data)
        -
        positives
    )

    prevalence = (
        positives
        /
        len(data)
    )

    scale_pos_weight = (
        negatives
        /
        positives
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "STAGE 5 — ML DATASET"
    )

    print(
        "=" * 120
    )

    print(
        f"\nFinal ML rows: "
        f"{len(data)}"
    )

    print(
        f"Positive any-WGS-link pairs: "
        f"{positives}"
    )

    print(
        f"Negative/unlinked pairs: "
        f"{negatives}"
    )

    print(
        f"Positive prevalence: "
        f"{prevalence:.4%}"
    )

    print(
        f"Overall negative/positive ratio: "
        f"{scale_pos_weight:.2f}"
    )

    print(
        f"Unique target patients: "
        f"{data['target_id'].nunique()}"
    )

    print(
        f"Targets containing >=1 positive: "
        f"{data.loc[data['label_any_wgs_link'] == 1, 'target_id'].nunique()}"
    )

    # ========================================================
    # FEATURE SETS
    # ========================================================

    print(
        "\n"
        + "-" * 120
    )

    print(
        "CORE FEATURES"
    )

    print(
        "-" * 120
    )

    for feature in CORE_FEATURES:

        print(
            f"  - {feature}"
        )

    print(
        "\n"
        + "-" * 120
    )

    print(
        "TYPING-AUGMENTED FEATURES"
    )

    print(
        "-" * 120
    )

    for feature in AUGMENTED_FEATURES:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "LABEL DISTRIBUTION BY TARGET / SOURCE TYPE"
    )

    print(
        "=" * 120
    )

    print(
        target_distribution.to_string(
            index=False
        )
    )

    # ========================================================
    # POSITIVE PAIRS
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "POSITIVE TRAINING PAIRS"
    )

    print(
        "=" * 120
    )

    positive_display = [
        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",

        "target_species",
        "target_st",

        "source_species_primary",
        "source_st_primary",

        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        "label_clonal_link",
        "label_plasmid_link",
        "label_any_wgs_link",
    ]

    existing_columns = [
        column
        for column
        in positive_display
        if column
        in positive_pairs.columns
    ]

    print(
        positive_pairs[
            existing_columns
        ]
        .sort_values(
            [
                "target_id",
                "days_before_acquisition",
            ]
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # GROUPED CV PLAN
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "LEAVE-ONE-TARGET-OUT CV PLAN"
    )

    print(
        "=" * 120
    )

    print(
        cv_plan.to_string(
            index=False
        )
    )

    positive_folds = int(
        cv_plan[
            "test_has_positive"
        ].sum()
    )

    negative_only_folds = (
        len(
            cv_plan
        )
        -
        positive_folds
    )

    print(
        f"\nPositive-target test folds: "
        f"{positive_folds}"
    )

    print(
        f"Negative-only test folds: "
        f"{negative_only_folds}"
    )

    print(
        "\nRanking metrics such as Hit@K and MRR will "
        "only be calculated for held-out targets "
        "that contain at least one true positive."
    )

    # ========================================================
    # DUPLICATE DIAGNOSTIC
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "BIOLOGICAL DUPLICATE HANDLING"
    )

    print(
        "=" * 120
    )

    print(
        f"\nRows involved in duplicate groups "
        f"before deduplication: "
        f"{len(duplicate_rows)}"
    )

    removed_metric = (
        overall_summary[
            overall_summary[
                "metric"
            ]
            == "duplicate_rows_removed"
        ][
            "value"
        ]
        .iloc[0]
    )

    print(
        f"Rows removed from final ML table: "
        f"{int(removed_metric)}"
    )

    # ========================================================
    # SAVE MAIN DATASET
    # ========================================================

    csv_file = (
        PROCESSED_DATA_DIR
        / "ml_any_wgs_link_dataset.csv"
    )

    data.to_csv(
        csv_file,
        index=False,
    )

    excel_file = (
        TABLE_DIR
        / "table_05_ml_dataset.xlsx"
    )

    data.to_excel(
        excel_file,
        index=False,
    )

    # ========================================================
    # SAVE DIAGNOSTICS
    # ========================================================

    manifest_file = (
        TABLE_DIR
        / "table_05_ml_feature_manifest.xlsx"
    )

    feature_manifest.to_excel(
        manifest_file,
        index=False,
    )

    distribution_file = (
        TABLE_DIR
        / "table_05_ml_target_distribution.xlsx"
    )

    target_distribution.to_excel(
        distribution_file,
        index=False,
    )

    cv_file = (
        TABLE_DIR
        / "table_05_group_cv_plan.xlsx"
    )

    cv_plan.to_excel(
        cv_file,
        index=False,
    )

    overall_file = (
        TABLE_DIR
        / "table_05_ml_overall_summary.xlsx"
    )

    overall_summary.to_excel(
        overall_file,
        index=False,
    )

    duplicate_file = (
        TABLE_DIR
        / "table_05_duplicate_diagnostic.xlsx"
    )

    duplicate_rows.to_excel(
        duplicate_file,
        index=False,
    )

    positive_file = (
        TABLE_DIR
        / "table_05_positive_training_pairs.xlsx"
    )

    positive_pairs.to_excel(
        positive_file,
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
        f"\nML dataset CSV:\n"
        f"{csv_file}"
    )

    print(
        f"\nML dataset Excel:\n"
        f"{excel_file}"
    )

    print(
        f"\nFeature manifest:\n"
        f"{manifest_file}"
    )

    print(
        f"\nTarget distribution:\n"
        f"{distribution_file}"
    )

    print(
        f"\nGrouped CV plan:\n"
        f"{cv_file}"
    )

    print(
        f"\nOverall ML summary:\n"
        f"{overall_file}"
    )

    print(
        f"\nDuplicate diagnostic:\n"
        f"{duplicate_file}"
    )

    print(
        f"\nPositive training pairs:\n"
        f"{positive_file}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe target is any_wgs_link."
        "\nDirect clonal/plasmid linkage variables are "
        "LABEL INFORMATION ONLY and are excluded "
        "from MODEL_FEATURES."
        "\nSame-day candidates are excluded."
        "\nValidation must be grouped by target_id."
    )


if __name__ == "__main__":
    main()