from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)

from cpo.explainability import (
    build_oof_shap_explanations,
    build_global_shap_summary,
    build_patient_rankings,
    build_positive_pair_explanations,
)

from cpo.ml_dataset import (
    AUGMENTED_FEATURES,
)


def main():

    print(
        "\nBuilding out-of-fold XGBoost explanations..."
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    figure_dir = (
        TABLE_DIR.parent
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # BUILD OOF SHAP
    # ========================================================

    oof_shap = (
        build_oof_shap_explanations(
            feature_columns=(
                AUGMENTED_FEATURES
            )
        )
    )

    global_shap = (
        build_global_shap_summary(
            oof_shap=oof_shap,
            feature_columns=(
                AUGMENTED_FEATURES
            ),
        )
    )

    rankings = (
        build_patient_rankings(
            oof_shap=oof_shap,
            top_k=10,
        )
    )

    positive_explanations = (
        build_positive_pair_explanations(
            oof_shap=oof_shap,
            feature_columns=(
                AUGMENTED_FEATURES
            ),
        )
    )

    # ========================================================
    # GLOBAL SHAP OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "OUT-OF-FOLD SHAP FEATURE IMPORTANCE"
    )

    print(
        "=" * 120
    )

    print(
        global_shap.to_string(
            index=False
        )
    )

    # ========================================================
    # POSITIVE PAIR EXPLANATIONS
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TRUE WGS-LINKED PAIRS — MODEL EXPLANATIONS"
    )

    print(
        "=" * 120
    )

    print(
        positive_explanations.to_string(
            index=False
        )
    )

    # ========================================================
    # TOP 5 PER PATIENT
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TOP 5 MODEL-RANKED SOURCES PER TARGET"
    )

    print(
        "=" * 120
    )

    display_columns = [
        "target_id",
        "rank_within_target",
        "predicted_probability",

        "label_any_wgs_link",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",

        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        "strongest_positive_feature",
        "strongest_negative_feature",
    ]

    for target_id in sorted(
        rankings[
            "target_id"
        ].unique()
    ):

        subset = (
            rankings[
                rankings[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                "rank_within_target"
            )
            .head(5)
        )

        print(
            f"\n{'=' * 30} "
            f"{target_id} "
            f"{'=' * 30}"
        )

        existing_columns = [
            column
            for column
            in display_columns
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

    # ========================================================
    # FIGURE 1 — CORE VS AUGMENTED PERFORMANCE
    # ========================================================

    summary_file = (
        TABLE_DIR
        / "table_06_xgboost_summary.xlsx"
    )

    performance = pd.read_excel(
        summary_file
    )

    metrics = [
        "pooled_oof_auprc",
        "pooled_oof_auroc",
        "mean_mrr",
        "hit_at_3",
        "hit_at_5",
    ]

    plot_data = (
        performance[
            [
                "model_name",
                *metrics,
            ]
        ]
        .set_index(
            "model_name"
        )
        .T
    )

    ax = plot_data.plot(
        kind="bar",
        figsize=(
            10,
            6,
        ),
    )

    ax.set_title(
        "Core vs Typing-Augmented XGBoost"
    )

    ax.set_ylabel(
        "Performance"
    )

    ax.set_xlabel(
        "Metric"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.legend(
        title="Model"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    performance_figure = (
        figure_dir
        / "figure_01_core_vs_augmented.png"
    )

    plt.savefig(
        performance_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — OOF SHAP IMPORTANCE
    # ========================================================

    shap_plot = (
        global_shap
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.barh(
        shap_plot[
            "feature_label"
        ],
        shap_plot[
            "mean_abs_shap"
        ],
    )

    ax.set_xlabel(
        "Mean |SHAP value|"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Out-of-Fold Feature Importance"
    )

    plt.tight_layout()

    shap_figure = (
        figure_dir
        / "figure_02_oof_shap_importance.png"
    )

    plt.savefig(
        shap_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # SAVE DATA
    # ========================================================

    shap_csv = (
        PROCESSED_DATA_DIR
        / "xgboost_oof_shap.csv"
    )

    oof_shap.to_csv(
        shap_csv,
        index=False,
    )

    shap_summary_file = (
        TABLE_DIR
        / "table_07_oof_shap_summary.xlsx"
    )

    global_shap.to_excel(
        shap_summary_file,
        index=False,
    )

    rankings_file = (
        TABLE_DIR
        / "table_07_patient_rankings.xlsx"
    )

    rankings.to_excel(
        rankings_file,
        index=False,
    )

    explanations_file = (
        TABLE_DIR
        / "table_07_positive_pair_explanations.xlsx"
    )

    positive_explanations.to_excel(
        explanations_file,
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
        f"\nOOF SHAP dataset:\n"
        f"{shap_csv}"
    )

    print(
        f"\nSHAP summary:\n"
        f"{shap_summary_file}"
    )

    print(
        f"\nPatient rankings:\n"
        f"{rankings_file}"
    )

    print(
        f"\nPositive-pair explanations:\n"
        f"{explanations_file}"
    )

    print(
        f"\nPerformance figure:\n"
        f"{performance_figure}"
    )

    print(
        f"\nSHAP figure:\n"
        f"{shap_figure}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese SHAP values are generated out-of-fold."
        "\nThe held-out target patient was not used to "
        "train the model explaining that patient's candidates."
    )


if __name__ == "__main__":
    main()