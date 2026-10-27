from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WGS_FILE = (
    PROCESSED
    / "wgs_labeled_pairs.csv"
)

PRED_FILE = (
    PROCESSED
    / "xgboost_oof_predictions.csv"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def bool_series(series):
    """
    Robustly convert CSV boolean-like values to bool.
    """

    if pd.api.types.is_bool_dtype(series):
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


def bool_value(value):
    """
    Robustly convert one boolean-like value to bool.
    """

    if isinstance(value, bool):
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


def wrap_text(
    text,
    width=35,
):
    """
    Wrap long labels so they do not overlap.
    """

    if pd.isna(text):
        return ""

    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
        )
    )


def clean_model_name(value):
    """
    Produce short model names for role-ablation plots.
    """

    value = str(value).lower()

    if "qwen" in value:
        return "Qwen"

    if "medgemma" in value:
        return "MedGemma"

    if "llama" in value:
        return "Llama"

    return str(value)


def save_figure(
    figure,
    name,
):
    """
    Save both publication-quality PNG and vector PDF.
    """

    png_path = (
        FIGURE_DIR
        / f"{name}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{name}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(figure)

    print(
        f"Saved: {png_path.name}"
    )


# ============================================================
# FIGURE 03
# DIRECT WGS SOURCE–TARGET NETWORK
# ============================================================

def plot_wgs_network():

    data = pd.read_csv(
        WGS_FILE
    )

    days = pd.to_numeric(
        data[
            "days_before_acquisition"
        ],
        errors="coerce",
    )

    linked = bool_series(
        data[
            "any_wgs_link"
        ]
    )

    data = data[
        (days > 0)
        &
        linked
    ].copy()

    if data.empty:

        print(
            "Skipping WGS network: "
            "no direct prior WGS links found."
        )

        return

    targets = sorted(
        data[
            "target_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # Build unique source keys
    # --------------------------------------------------------

    source_keys = []

    for _, row in data.iterrows():

        subject = str(
            row.get(
                "source_subject_id",
                "",
            )
        )

        sample = str(
            row.get(
                "source_sample_id",
                "",
            )
        )

        key = (
            f"{subject}/{sample}"
        )

        if key not in source_keys:

            source_keys.append(key)

    # More vertical spacing between source labels
    source_spacing = 1.25

    source_y = {
        source:
            index
            * source_spacing

        for index, source
        in enumerate(source_keys)
    }

    max_y = max(
        source_y.values()
    ) if source_y else 1

    # Spread targets over same vertical extent
    if len(targets) == 1:

        target_y = {
            targets[0]:
                max_y / 2
        }

    else:

        target_y = {
            target:
                position

            for target, position
            in zip(
                targets,
                np.linspace(
                    0,
                    max_y,
                    len(targets),
                ),
            )
        }

    # Figure height scales with number of source labels
    figure_height = max(
        7,
        len(source_keys)
        * 0.55,
    )

    fig, ax = plt.subplots(
        figsize=(
            13.5,
            figure_height,
        ),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Plot source nodes
    # --------------------------------------------------------

    patient_x = []
    patient_y = []

    env_x = []
    env_y = []

    for source in source_keys:

        source_rows = data[
            (
                data[
                    "source_subject_id"
                ]
                .astype(str)
                + "/"
                + data[
                    "source_sample_id"
                ]
                .astype(str)
            )
            ==
            source
        ]

        row = source_rows.iloc[0]

        source_type = (
            str(
                row.get(
                    "source_type",
                    "",
                )
            )
            .strip()
            .lower()
        )

        if "environment" in source_type:

            env_x.append(0)
            env_y.append(
                source_y[source]
            )

        else:

            patient_x.append(0)
            patient_y.append(
                source_y[source]
            )

    if patient_x:

        ax.scatter(
            patient_x,
            patient_y,
            marker="o",
            s=95,
            zorder=3,
            label="Patient source",
        )

    if env_x:

        ax.scatter(
            env_x,
            env_y,
            marker="s",
            s=95,
            zorder=3,
            label="Environmental source",
        )

    # --------------------------------------------------------
    # Plot target nodes
    # --------------------------------------------------------

    ax.scatter(
        [1] * len(targets),
        [
            target_y[target]
            for target
            in targets
        ],
        marker="D",
        s=110,
        zorder=3,
        label="Acquisition target",
    )

    # --------------------------------------------------------
    # Plot genomic links
    # --------------------------------------------------------

    for _, row in data.iterrows():

        source = (
            f"{row.get('source_subject_id', '')}"
            f"/"
            f"{row.get('source_sample_id', '')}"
        )

        target = str(
            row[
                "target_id"
            ]
        )

        if bool_value(
            row.get(
                "clonal_link",
                False,
            )
        ):

            linestyle = "-"
            linewidth = 1.8

        else:

            linestyle = "--"
            linewidth = 1.5

        ax.plot(
            [0, 1],
            [
                source_y[source],
                target_y[target],
            ],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.70,
            zorder=1,
        )

    # --------------------------------------------------------
    # Source labels
    # --------------------------------------------------------

    for source, y in source_y.items():

        display_source = wrap_text(
            source,
            width=30,
        )

        ax.text(
            -0.08,
            y,
            display_source,
            ha="right",
            va="center",
            fontsize=8.5,
        )

    # --------------------------------------------------------
    # Target labels
    # --------------------------------------------------------

    for target, y in target_y.items():

        ax.text(
            1.08,
            y,
            target,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Legend outside graph
    # --------------------------------------------------------

    edge_legend = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=1.8,
            label="Clonal WGS link",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1.8,
            label="Plasmid WGS link",
        ),
    ]

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    ax.legend(
        handles
        + edge_legend,
        labels
        + [
            "Clonal WGS link",
            "Plasmid WGS link",
        ],
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
    )

    # More left/right whitespace for labels
    ax.set_xlim(
        -0.55,
        1.55,
    )

    ax.set_ylim(
        -0.7,
        max_y + 0.7,
    )

    ax.set_title(
        (
            "Direct prior WGS-linked candidate sources "
            "for hospital-acquired CPO cases"
        ),
        fontsize=13,
        pad=18,
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Candidate sources",
            "Acquisition targets",
        ],
        fontsize=10,
    )

    ax.tick_params(
        axis="x",
        pad=8,
    )

    ax.set_yticks([])

    for spine in [
        "top",
        "right",
        "left",
    ]:

        ax.spines[
            spine
        ].set_visible(False)

    save_figure(
        fig,
        "figure_03_wgs_source_target_network",
    )


# ============================================================
# FIGURE 04
# CORE VS AUGMENTED XGBOOST PERFORMANCE
# ============================================================

def plot_model_performance():

    path = (
        TABLE_DIR
        / "table_xgboost_performance.csv"
    )

    if not path.exists():

        print(
            "Skipping model performance plot: "
            "table_xgboost_performance.csv not found."
        )

        return

    data = pd.read_csv(path)

    metrics = [
        (
            "pooled_auprc",
            "AUPRC",
        ),
        (
            "pooled_auroc",
            "AUROC",
        ),
        (
            "mrr",
            "MRR",
        ),
        (
            "hit_at_3",
            "Hit@3",
        ),
        (
            "hit_at_5",
            "Hit@5",
        ),
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(
            9.5,
            5.8,
        ),
        constrained_layout=True,
    )

    for model_index, model in enumerate(
        [
            "CORE",
            "AUGMENTED",
        ]
    ):

        subset = data[
            data[
                "model"
            ]
            .astype(str)
            .str.upper()
            ==
            model
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        values = [
            float(
                row[column]
            )
            for column, _
            in metrics
        ]

        offset = (
            model_index - 0.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model.title(),
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            fontsize=8.5,
            padding=4,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            label
            for _, label
            in metrics
        ],
        fontsize=10,
    )

    ax.tick_params(
        axis="x",
        pad=8,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    # Leave space above 1.0 for bar labels
    ax.set_ylim(
        0,
        1.16,
    )

    ax.set_ylabel(
        "Performance",
        fontsize=10,
        labelpad=10,
    )

    ax.set_title(
        (
            "Leave-one-target-out "
            "candidate-ranking performance"
        ),
        fontsize=13,
        pad=35,
    )

    # Legend above axes, not over bars
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.8,
    )

    save_figure(
        fig,
        "figure_04_xgboost_performance",
    )


# ============================================================
# FIGURE 05
# GLOBAL SHAP IMPORTANCE
# ============================================================

def plot_shap_importance():

    path = (
        TABLE_DIR
        / "table_shap_importance.csv"
    )

    if not path.exists():

        print(
            "Skipping SHAP plot: "
            "table_shap_importance.csv not found."
        )

        return

    data = pd.read_csv(path)

    data = (
        data.head(10)
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
        .copy()
    )

    data[
        "display_feature"
    ] = (
        data[
            "feature"
        ]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .apply(
            lambda x:
                wrap_text(
                    x,
                    width=28,
                )
        )
    )

    figure_height = max(
        5.5,
        len(data) * 0.55,
    )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            figure_height,
        ),
        constrained_layout=True,
    )

    bars = ax.barh(
        data[
            "display_feature"
        ],
        data[
            "mean_absolute_shap"
        ],
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        fontsize=8,
        padding=4,
    )

    max_value = (
        data[
            "mean_absolute_shap"
        ]
        .max()
    )

    if pd.notna(max_value):

        ax.set_xlim(
            0,
            max_value * 1.18,
        )

    ax.set_xlabel(
        "Mean absolute SHAP value",
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Feature",
        fontsize=10,
        labelpad=10,
    )

    ax.set_title(
        (
            "Global out-of-fold SHAP importance "
            "for the augmented XGBoost model"
        ),
        fontsize=13,
        pad=16,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
        pad=5,
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
    )

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    save_figure(
        fig,
        "figure_05_shap_importance",
    )


# ============================================================
# SUPPLEMENTARY TARGET-SPECIFIC RANKING FIGURES
# ============================================================

def plot_target_ranking(
    target_id,
    filename,
):

    predictions = pd.read_csv(
        PRED_FILE
    )

    wgs = pd.read_csv(
        WGS_FILE
    )

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

    subset = (
        subset.sort_values(
            "rank_within_target"
        )
        .head(10)
    )

    if subset.empty:

        print(
            f"Skipping {target_id}: "
            "no augmented-model predictions."
        )

        return

    target_wgs = wgs[
        wgs[
            "target_id"
        ]
        .astype(str)
        ==
        target_id
    ].copy()

    if not target_wgs.empty:

        linked_mask = bool_series(
            target_wgs[
                "any_wgs_link"
            ]
        )

        linked_rows = target_wgs[
            linked_mask
        ].copy()

    else:

        linked_rows = pd.DataFrame()

    link_map = {}

    for _, row in linked_rows.iterrows():

        sample = str(
            row.get(
                "source_sample_id",
                "",
            )
        )

        if bool_value(
            row.get(
                "clonal_link",
                False,
            )
        ):

            link_map[
                sample
            ] = "clonal"

        elif bool_value(
            row.get(
                "plasmid_link",
                False,
            )
        ):

            link_map[
                sample
            ] = "plasmid"

    labels = []

    for _, row in subset.iterrows():

        sample = str(
            row.get(
                "source_sample_id",
                "",
            )
        )

        subject = str(
            row.get(
                "source_subject_id",
                "",
            )
        )

        source_type = str(
            row.get(
                "source_type",
                "",
            )
        )

        label = (
            f"{source_type}: "
            f"{subject}/{sample}"
        )

        if sample in link_map:

            label += (
                f" [WGS {link_map[sample]}]"
            )

        labels.append(
            wrap_text(
                label,
                width=43,
            )
        )

    scores = pd.to_numeric(
        subset[
            "predicted_probability"
        ],
        errors="coerce",
    ).to_numpy()

    y = np.arange(
        len(subset)
    )

    figure_height = max(
        6.5,
        len(subset) * 0.62,
    )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            figure_height,
        ),
        constrained_layout=True,
    )

    bars = ax.barh(
        y,
        scores,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=8.5,
    )

    ax.tick_params(
        axis="y",
        pad=7,
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        (
            "XGBoost candidate-prioritisation score "
            "(not a calibrated transmission probability)"
        ),
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Ranked candidate source",
        fontsize=10,
        labelpad=12,
    )

    ax.set_title(
        (
            f"Top augmented-model candidate sources "
            f"for {target_id}"
        ),
        fontsize=13,
        pad=16,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        fontsize=8,
        padding=4,
    )

    if len(scores) > 0:

        max_score = np.nanmax(
            scores
        )

        if np.isfinite(
            max_score
        ):

            ax.set_xlim(
                0,
                max(
                    1.0,
                    max_score * 1.18,
                ),
            )

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    save_figure(
        fig,
        filename,
    )


# ============================================================
# SUPPLEMENTARY FIGURE
# COUNCIL REPEATABILITY
# ============================================================

def plot_repeatability():

    path = (
        TABLE_DIR
        / "table_council_repeatability.csv"
    )

    if not path.exists():

        print(
            "Skipping council repeatability plot: "
            "summary table not found."
        )

        return

    data = pd.read_csv(path)

    if data.empty:
        return

    # Use integer x positions so arbitrary run labels
    # cannot create tick-layout problems.
    x = np.arange(
        len(data)
    )

    run_labels = (
        data[
            "run"
        ]
        .astype(str)
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            8.2,
            5.3,
        ),
        constrained_layout=True,
    )

    ax.plot(
        x,
        data[
            "end_to_end_agreement"
        ],
        marker="o",
        linewidth=2,
        markersize=7,
        label="End-to-end agreement",
    )

    ax.plot(
        x,
        data[
            "agreement_among_completed"
        ],
        marker="s",
        linewidth=2,
        markersize=7,
        label="Agreement among completed cases",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        run_labels,
        fontsize=9,
    )

    ax.set_ylim(
        0,
        1.08,
    )

    ax.set_xlabel(
        "Repeated frozen Council-v2 execution",
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Agreement with deterministic WGS reference",
        fontsize=10,
        labelpad=10,
    )

    ax.set_title(
        "LLM council inference reproducibility",
        fontsize=13,
        pad=38,
    )

    # Legend above plotting area
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.8,
    )

    save_figure(
        fig,
        "figure_S1_council_repeatability",
    )


# ============================================================
# SUPPLEMENTARY FIGURE
# ROLE-ASSIGNMENT ABLATION
# ============================================================

def plot_role_ablation():

    path = (
        TABLE_DIR
        / "table_role_ablation.csv"
    )

    if not path.exists():

        print(
            "Skipping role-ablation plot: "
            "summary table not found."
        )

        return

    data = pd.read_csv(path)

    if data.empty:
        return

    display_labels = []

    for _, row in data.iterrows():

        assignment = str(
            row.get(
                "assignment_id",
                "",
            )
        )

        investigator = clean_model_name(
            row.get(
                "investigator",
                "",
            )
        )

        biomedical = clean_model_name(
            row.get(
                "biomedical_expert",
                "",
            )
        )

        critic = clean_model_name(
            row.get(
                "causal_critic",
                "",
            )
        )

        label = (
            f"{assignment}   "
            f"I: {investigator} | "
            f"B: {biomedical} | "
            f"C: {critic}"
        )

        display_labels.append(label)

    plotting = data.copy()

    plotting[
        "display_label"
    ] = display_labels

    # Reverse so A1 appears at the top if input is A1...A6
    plotting = plotting.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    figure_height = max(
        5.2,
        len(plotting) * 0.72,
    )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            figure_height,
        ),
        constrained_layout=True,
    )

    y = np.arange(
        len(plotting)
    )

    values = pd.to_numeric(
        plotting[
            "end_to_end_agreement"
        ],
        errors="coerce",
    )

    bars = ax.barh(
        y,
        values,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        plotting[
            "display_label"
        ],
        fontsize=8.5,
    )

    ax.tick_params(
        axis="y",
        pad=7,
    )

    ax.set_xlim(
        0,
        1.12,
    )

    ax.set_xlabel(
        "End-to-end agreement with deterministic WGS reference",
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Model–role assignment",
        fontsize=10,
        labelpad=10,
    )

    ax.set_title(
        "Complete model–role assignment ablation",
        fontsize=13,
        pad=16,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        fontsize=8.5,
        padding=4,
    )

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    # Footnote explains role abbreviations without a legend
    fig.text(
        0.5,
        0.005,
        (
            "I = Investigator; "
            "B = Biomedical Expert; "
            "C = Causal Critic"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    save_figure(
        fig,
        "figure_S2_role_assignment_ablation",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nGenerating publication-ready figures...\n"
    )

    plot_wgs_network()

    plot_model_performance()

    plot_shap_importance()

    plot_target_ranking(
        "T0036",
        "figure_S3_T0036_candidate_ranking",
    )

    plot_target_ranking(
        "T0124",
        "figure_S4_T0124_candidate_ranking",
    )

    plot_repeatability()

    plot_role_ablation()

    print(
        "\nFinished."
    )

    print(
        "Figures saved to:"
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()