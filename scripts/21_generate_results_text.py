from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper"
    / "tables"
)

TEXT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper"
    / "text"
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def percentage(
    numerator,
    denominator,
):

    return (
        100.0
        * numerator
        / denominator
    )


def main():

    cases = pd.read_csv(
        TABLE_DIR
        / "table_case_level_source_attribution.csv"
    )

    metrics = pd.read_csv(
        TABLE_DIR
        / "table_xgboost_performance.csv"
    )

    shap = pd.read_csv(
        TABLE_DIR
        / "table_shap_importance.csv"
    )

    mechanisms = (
        cases[
            "wgs_reference_mechanism"
        ]
        .value_counts()
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
    )

    patient_plasmid = int(
        mechanisms.get(
            "PATIENT_ASSOCIATED_PLASMID",
            0,
        )
    )

    patient_clonal = int(
        mechanisms.get(
            "PATIENT_ASSOCIATED_CLONAL",
            0,
        )
    )

    environment_clonal = int(
        mechanisms.get(
            "ENVIRONMENT_ASSOCIATED_CLONAL",
            0,
        )
    )

    no_link = int(
        mechanisms.get(
            "NO_DIRECT_WGS_LINK",
            0,
        )
    )

    no_wgs = int(
        mechanisms.get(
            "UNRESOLVED_NO_TARGET_WGS",
            0,
        )
    )

    top_feature = (
        shap.iloc[0][
            "feature"
        ]
    )

    lines = []

    lines.append(
        "# Paper-ready results summary\n"
    )

    lines.append(
        "## Clinical acquisition-source analysis\n"
    )

    lines.append(
        (
            f"Eight patients acquired a carbapenemase gene "
            f"during hospitalisation. Direct WGS-supported "
            f"circulation patterns were identified in "
            f"{patient_plasmid + patient_clonal + environment_clonal}"
            f"/8 cases "
            f"({percentage(patient_plasmid + patient_clonal + environment_clonal, 8):.1f}%). "
            f"These comprised {patient_plasmid} patient-associated "
            f"plasmid cases, {patient_clonal} patient-associated "
            f"clonal case, and {environment_clonal} "
            f"environment-associated clonal case. "
            f"{no_link}/8 targets had WGS available but no "
            f"direct prior clonal or plasmid-linked candidate, "
            f"while {no_wgs}/8 lacked target WGS and remained "
            f"unresolved.\n"
        )
    )

    lines.append(
        "## Machine-learning prioritisation\n"
    )

    lines.append(
        (
            f"The augmented XGBoost candidate-ranking model "
            f"achieved a pooled AUROC of "
            f"{augmented['pooled_auroc']:.3f} and AUPRC of "
            f"{augmented['pooled_auprc']:.3f}. "
            f"Mean reciprocal rank was "
            f"{augmented['mrr']:.3f}; Hit@1, Hit@3 and Hit@5 "
            f"were {augmented['hit_at_1']:.2f}, "
            f"{augmented['hit_at_3']:.2f}, and "
            f"{augmented['hit_at_5']:.2f}, respectively. "
            f"The median best-positive rank was "
            f"{augmented['median_best_positive_rank']:.1f}. "
            f"The most influential global SHAP feature was "
            f"`{top_feature}`.\n"
        )
    )

    repeat_path = (
        TABLE_DIR
        / "table_council_repeatability.csv"
    )

    if repeat_path.exists():

        repeats = pd.read_csv(
            repeat_path
        )

        lines.append(
            "## LLM council reproducibility\n"
        )

        lines.append(
            (
                f"Across five repeated executions of the "
                f"frozen Council-v2 configuration, mean "
                f"end-to-end agreement with the deterministic "
                f"WGS reference was "
                f"{repeats['end_to_end_agreement'].mean():.3f}, "
                f"while mean agreement among successfully "
                f"completed cases was "
                f"{repeats['agreement_among_completed'].mean():.3f}. "
                f"Repeated executions were treated as "
                f"inference-stability measurements rather "
                f"than independent clinical observations.\n"
            )
        )

    role_path = (
        TABLE_DIR
        / "table_role_ablation.csv"
    )

    if role_path.exists():

        role = pd.read_csv(
            role_path
        )

        original = role[
            role[
                "original_assignment"
            ]
            ==
            True
        ].iloc[0]

        lines.append(
            "## Model-role ablation\n"
        )

        lines.append(
            (
                f"All six possible assignments of the three "
                f"LLMs to the Investigator, Biomedical Expert "
                f"and Causal Critic roles were evaluated. "
                f"End-to-end agreement ranged from "
                f"{role['end_to_end_agreement'].min():.3f} "
                f"to "
                f"{role['end_to_end_agreement'].max():.3f}. "
                f"The prespecified Qwen3-8B Investigator / "
                f"MedGemma 1.5-4B Biomedical Expert / "
                f"Llama 3.1-8B Causal Critic configuration "
                f"achieved "
                f"{original['end_to_end_agreement']:.3f} "
                f"end-to-end agreement and was tied for the "
                f"best-performing assignment.\n"
            )
        )

    output = "\n".join(
        lines
    )

    path = (
        TEXT_DIR
        / "paper_results_summary.md"
    )

    path.write_text(
        output,
        encoding="utf-8",
    )

    print(
        output
    )

    print(
        "\nSaved to:"
    )

    print(
        path
    )


if __name__ == "__main__":
    main()