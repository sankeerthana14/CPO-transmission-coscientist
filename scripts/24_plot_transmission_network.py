from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

NETWORK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "network"
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


EDGE_FILE = (
    NETWORK_DIR
    / "transmission_network_edges.csv"
)

HUB_FILE = (
    NETWORK_DIR
    / "network_source_hubs.csv"
)

PROFILE_FILE = (
    NETWORK_DIR
    / "network_target_source_profiles.csv"
)

PROJECTION_FILE = (
    NETWORK_DIR
    / "network_target_projection.csv"
)


# ============================================================
# HELPERS
# ============================================================

def bool_series(series):

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series
            .fillna(False)
        )

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


def load_edges():

    edges = pd.read_csv(
        EDGE_FILE
    )

    for column in [
        "top1",
        "top3",
        "top5",
        "clonal_link",
        "plasmid_link",
        "any_wgs_link",
    ]:

        if column not in edges.columns:
            edges[column] = False

        edges[column] = bool_series(
            edges[column]
        )

    return edges


def save(
    fig,
    filename,
):

    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.close(fig)

    print(
        f"Saved: {filename}"
    )


def clean_source_label(
    node,
):

    node = str(node)

    if node.startswith(
        "P:"
    ):

        return node.replace(
            "P:",
            "",
            1,
        )

    if node.startswith(
        "E:"
    ):

        return node.replace(
            "E:",
            "",
            1,
        )

    return node


# ============================================================
# FIGURE N1
# MULTILAYER PATIENT–ENVIRONMENT EVIDENCE NETWORK
# ============================================================

def plot_evidence_network():

    edges = load_edges()

    # --------------------------------------------------------
    # Main-paper network:
    # top-3 ML candidate OR direct WGS relationship.
    # --------------------------------------------------------

    evidence = edges[
        edges[
            "top3"
        ]
        |
        edges[
            "any_wgs_link"
        ]
    ].copy()

    if evidence.empty:

        print(
            "Skipping N1: no evidence edges."
        )

        return

    source_nodes = sorted(
        evidence[
            "source_node"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    targets = sorted(
        evidence[
            "target_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # Vertical placement
    # --------------------------------------------------------

    source_spacing = 1.15

    source_y = {
        node:
            index
            * source_spacing

        for index, node
        in enumerate(
            source_nodes
        )
    }

    max_y = (
        max(
            source_y.values()
        )
        if source_y
        else 1
    )

    if len(targets) == 1:

        target_y = {
            targets[0]:
                max_y
                / 2
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

    # --------------------------------------------------------
    # Larger width because right side is reserved for legend
    # --------------------------------------------------------

    fig_height = max(
        8,
        len(source_nodes)
        * 0.52,
    )

    fig, ax = plt.subplots(
        figsize=(
            15.5,
            fig_height,
        )
    )

    # Dedicated margins:
    # left = source labels
    # right = legend
    # top = title
    fig.subplots_adjust(
        left=0.18,
        right=0.74,
        top=0.90,
        bottom=0.10,
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    for _, row in evidence.iterrows():

        source = str(
            row[
                "source_node"
            ]
        )

        target = str(
            row[
                "target_id"
            ]
        )

        if row[
            "clonal_link"
        ]:

            linestyle = "-"
            linewidth = 3.0
            alpha = 0.90

        elif row[
            "plasmid_link"
        ]:

            linestyle = "--"
            linewidth = 2.6
            alpha = 0.90

        else:

            linestyle = ":"
            linewidth = 1.3
            alpha = 0.45

        ax.plot(
            [
                0,
                1,
            ],
            [
                source_y[
                    source
                ],
                target_y[
                    target
                ],
            ],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

    # --------------------------------------------------------
    # SOURCE NODES
    # --------------------------------------------------------

    patient_x = []
    patient_y = []

    env_x = []
    env_y = []

    for source in source_nodes:

        group = evidence[
            evidence[
                "source_node"
            ]
            .astype(str)
            ==
            source
        ]

        source_type = (
            str(
                group[
                    "source_type"
                ]
                .iloc[0]
            )
            .strip()
            .lower()
        )

        if (
            source_type
            == "environment"
        ):

            env_x.append(0)

            env_y.append(
                source_y[
                    source
                ]
            )

        else:

            patient_x.append(0)

            patient_y.append(
                source_y[
                    source
                ]
            )

    if patient_x:

        ax.scatter(
            patient_x,
            patient_y,
            marker="o",
            s=100,
            zorder=4,
            label="Patient source",
        )

    if env_x:

        ax.scatter(
            env_x,
            env_y,
            marker="s",
            s=100,
            zorder=4,
            label="Environmental source",
        )

    # --------------------------------------------------------
    # TARGET NODES
    # --------------------------------------------------------

    ax.scatter(
        [1] * len(targets),
        [
            target_y[
                target
            ]
            for target
            in targets
        ],
        marker="D",
        s=125,
        zorder=4,
        label="Acquisition target",
    )

    # --------------------------------------------------------
    # LABEL SOURCE NODES
    # --------------------------------------------------------

    for source in source_nodes:

        label = clean_source_label(
            source
        )

        ax.text(
            -0.055,
            source_y[
                source
            ],
            label,
            ha="right",
            va="center",
            fontsize=8.5,
        )

    # --------------------------------------------------------
    # LABEL TARGETS
    # --------------------------------------------------------

    for target in targets:

        ax.text(
            1.055,
            target_y[
                target
            ],
            target,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # LEGEND
    #
    # IMPORTANT:
    # Place it completely outside the plotting area on the
    # right side. It therefore cannot overlap the title.
    # --------------------------------------------------------

    evidence_legend = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=3,
            label="WGS clonal link",
        ),

        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=3,
            label="WGS plasmid link",
        ),

        Line2D(
            [0],
            [0],
            linestyle=":",
            linewidth=1.5,
            label="Top-3 ML-ranked candidate",
        ),
    ]

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    ax.legend(
        handles
        + evidence_legend,
        labels
        + [
            "WGS clonal link",
            "WGS plasmid link",
            "Top-3 ML-ranked candidate",
        ],
        loc="center left",
        bbox_to_anchor=(
            1.03,
            0.50,
        ),
        borderaxespad=0,
        frameon=False,
        fontsize=9.5,
        title="Network elements",
        title_fontsize=10,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        -0.48,
        1.48,
    )

    ax.set_ylim(
        -0.75,
        max_y + 0.75,
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "Candidate source nodes",
            "Acquisition targets",
        ],
        fontsize=10,
    )

    ax.tick_params(
        axis="x",
        pad=8,
    )

    ax.set_yticks([])

    ax.set_title(
        (
            "Temporally constrained patient–environment "
            "CPO transmission evidence network"
        ),
        fontsize=14,
        pad=18,
        fontweight="bold",
    )

    # Explanatory footnote
    ax.text(
        0.5,
        -0.075,
        (
            "Dotted edges indicate ML prioritisation only; "
            "solid and dashed edges indicate direct genomic support."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    for spine in [
        "left",
        "right",
        "top",
    ]:

        ax.spines[
            spine
        ].set_visible(False)

    save(
        fig,
        "figure_N1_multilayer_transmission_network",
    )


# ============================================================
# FIGURE N2
# RECURRENT SOURCES / GENOMIC HUBS
# ============================================================

def plot_source_hubs():

    if not HUB_FILE.exists():

        print(
            "Skipping N2: "
            "network_source_hubs.csv not found."
        )

        return

    hubs = pd.read_csv(
        HUB_FILE
    )

    subset = hubs[
        (
            hubs[
                "top5_target_count"
            ]
            >= 2
        )
        |
        (
            hubs[
                "wgs_linked_target_count"
            ]
            > 0
        )
    ].copy()

    if subset.empty:

        print(
            "Skipping N2: "
            "no recurrent source hubs found."
        )

        return

    subset = subset.head(
        15
    )

    subset = subset.sort_values(
        [
            "wgs_linked_target_count",
            "top5_target_count",
        ],
        ascending=True,
    )

    subset[
        "display_source"
    ] = (
        subset[
            "source_node"
        ]
        .astype(str)
        .apply(
            clean_source_label
        )
    )

    y = np.arange(
        len(subset)
    )

    height = 0.34

    fig_height = max(
        6,
        len(subset)
        * 0.52,
    )

    fig, ax = plt.subplots(
        figsize=(
            13,
            fig_height,
        )
    )

    # Space for labels + right-side legend
    fig.subplots_adjust(
        left=0.26,
        right=0.72,
        top=0.89,
        bottom=0.13,
    )

    bars_top5 = ax.barh(
        y - height / 2,
        subset[
            "top5_target_count"
        ],
        height=height,
        label=(
            "Targets where source "
            "was top-5 ranked"
        ),
    )

    bars_wgs = ax.barh(
        y + height / 2,
        subset[
            "wgs_linked_target_count"
        ],
        height=height,
        label=(
            "Targets with direct "
            "WGS linkage"
        ),
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        subset[
            "display_source"
        ],
        fontsize=9,
    )

    ax.tick_params(
        axis="y",
        pad=7,
    )

    ax.set_xlabel(
        "Number of distinct acquisition targets",
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Candidate source node",
        fontsize=10,
        labelpad=12,
    )

    ax.set_title(
        (
            "Recurrent candidate sources and genomic hubs "
            "across acquisition cases"
        ),
        fontsize=13,
        pad=18,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # RIGHT-SIDE LEGEND
    # --------------------------------------------------------

    ax.legend(
        loc="center left",
        bbox_to_anchor=(
            1.03,
            0.50,
        ),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        title="Evidence",
        title_fontsize=10,
    )

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    # Make integer scale easier to interpret
    max_value = max(
        subset[
            "top5_target_count"
        ].max(),
        subset[
            "wgs_linked_target_count"
        ].max(),
    )

    ax.set_xlim(
        0,
        max(
            1,
            max_value
            + 0.75,
        ),
    )

    save(
        fig,
        "figure_N2_recurrent_source_hubs",
    )


# ============================================================
# FIGURE N3
# ACQUISITION TARGET NETWORK BASED ON SHARED SOURCES
# ============================================================

def plot_target_projection():

    if not PROJECTION_FILE.exists():

        print(
            "Skipping N3: "
            "network_target_projection.csv not found."
        )

        return

    projection = pd.read_csv(
        PROJECTION_FILE
    )

    if projection.empty:

        print(
            "Skipping N3: "
            "no target projection relationships found."
        )

        return

    graph = nx.Graph()

    for _, row in projection.iterrows():

        shared = int(
            row[
                "shared_evidence_sources"
            ]
        )

        if shared <= 0:
            continue

        graph.add_edge(
            str(
                row[
                    "target_a"
                ]
            ),
            str(
                row[
                    "target_b"
                ]
            ),
            shared_evidence=shared,

            shared_wgs=int(
                row[
                    "shared_wgs_sources"
                ]
            ),

            shared_plasmid=int(
                row[
                    "shared_plasmid_sources"
                ]
            ),

            shared_clonal=int(
                row[
                    "shared_clonal_sources"
                ]
            ),
        )

    if (
        graph.number_of_edges()
        == 0
    ):

        print(
            "Skipping N3: "
            "projection graph contains no edges."
        )

        return

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.7,
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            7.8,
        )
    )

    # Right side reserved for legend.
    fig.subplots_adjust(
        left=0.08,
        right=0.74,
        top=0.89,
        bottom=0.08,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1750,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=10,
        font_weight="bold",
        ax=ax,
    )

    # --------------------------------------------------------
    # Separate genomic vs non-genomic relationships
    # --------------------------------------------------------

    general_edges = []

    genomic_edges = []

    for u, v, data in graph.edges(
        data=True
    ):

        if (
            data[
                "shared_wgs"
            ]
            > 0
        ):

            genomic_edges.append(
                (
                    u,
                    v,
                )
            )

        else:

            general_edges.append(
                (
                    u,
                    v,
                )
            )

    if general_edges:

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=general_edges,
            width=1.6,
            style="dotted",
            alpha=0.5,
            ax=ax,
        )

    if genomic_edges:

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=genomic_edges,
            width=3.2,
            style="solid",
            alpha=0.9,
            ax=ax,
        )

    # --------------------------------------------------------
    # EDGE LABELS
    # --------------------------------------------------------

    edge_labels = {}

    for u, v, data in graph.edges(
        data=True
    ):

        label = (
            f"{data['shared_evidence']} shared"
        )

        if (
            data[
                "shared_wgs"
            ]
            > 0
        ):

            label += (
                f"\n{data['shared_wgs']} WGS"
            )

        edge_labels[
            (
                u,
                v,
            )
        ] = label

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        ax=ax,
    )

    ax.set_title(
        (
            "Acquisition-case network based on shared "
            "patient/environment source nodes"
        ),
        fontsize=13,
        pad=18,
        fontweight="bold",
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # RIGHT-SIDE LEGEND
    # --------------------------------------------------------

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle=":",
                linewidth=2,
                label="Shared prioritised source",
            ),

            Line2D(
                [0],
                [0],
                linestyle="-",
                linewidth=3,
                label="Shared WGS-linked source",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(
            1.03,
            0.50,
        ),
        borderaxespad=0,
        frameon=False,
        fontsize=9.5,
        title="Target relationship",
        title_fontsize=10,
    )

    save(
        fig,
        "figure_N3_target_shared_source_network",
    )


# ============================================================
# FIGURE N4
# PATIENT VS ENVIRONMENT RANK SUPPORT
# ============================================================

def plot_target_source_support():

    if not PROFILE_FILE.exists():

        print(
            "Skipping N4: "
            "network_target_source_profiles.csv not found."
        )

        return

    data = pd.read_csv(
        PROFILE_FILE
    )

    if data.empty:

        print(
            "Skipping N4: "
            "target profile file is empty."
        )

        return

    data = data.sort_values(
        "target_id"
    ).reset_index(
        drop=True
    )

    y = np.arange(
        len(data)
    )

    height = 0.34

    fig_height = max(
        6,
        len(data)
        * 0.60,
    )

    fig, ax = plt.subplots(
        figsize=(
            13,
            fig_height,
        )
    )

    # Dedicated right legend area
    fig.subplots_adjust(
        left=0.15,
        right=0.72,
        top=0.89,
        bottom=0.14,
    )

    ax.barh(
        y - height / 2,
        data[
            "patient_top5_rr_support"
        ],
        height=height,
        label="Patient-source rank support",
    )

    ax.barh(
        y + height / 2,
        data[
            "environment_top5_rr_support"
        ],
        height=height,
        label="Environmental-source rank support",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        data[
            "target_id"
        ],
        fontsize=9.5,
    )

    ax.tick_params(
        axis="y",
        pad=6,
    )

    ax.set_xlabel(
        (
            "Top-5 reciprocal-rank support "
            "(relative prioritisation; not probability)"
        ),
        fontsize=10,
        labelpad=10,
    )

    ax.set_ylabel(
        "Acquisition target",
        fontsize=10,
        labelpad=10,
    )

    ax.set_title(
        (
            "Relative patient-versus-environment "
            "candidate support by acquisition target"
        ),
        fontsize=13,
        pad=18,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # RIGHT-SIDE LEGEND
    # --------------------------------------------------------

    ax.legend(
        loc="center left",
        bbox_to_anchor=(
            1.03,
            0.50,
        ),
        borderaxespad=0,
        frameon=False,
        fontsize=9.5,
        title="Candidate source class",
        title_fontsize=10,
    )

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.8,
    )

    save(
        fig,
        "figure_N4_patient_environment_rank_support",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "============================================"
    )

    print(
        "Generating analytic network figures"
    )

    print(
        "============================================\n"
    )

    plot_evidence_network()

    plot_source_hubs()

    plot_target_projection()

    plot_target_source_support()

    print(
        "\nDone."
    )

    print(
        "\nFigures saved to:"
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()