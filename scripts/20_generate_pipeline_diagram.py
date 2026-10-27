from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
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


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
):
    """
    Draw one workflow box.
    """

    box = FancyBboxPatch(
        (
            x,
            y,
        ),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.3,
        fill=False,
    )

    ax.add_patch(
        box
    )

    ax.text(
        x
        + width / 2,
        y
        + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        x
        + width / 2,
        y
        + height * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )


def arrow(
    ax,
    start,
    end,
):
    """
    Draw directed workflow arrow.
    """

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2,
        },
    )


def main():

    fig, ax = plt.subplots(
        figsize=(
            14,
            8,
        )
    )

    ax.set_xlim(
        0,
        14,
    )

    ax.set_ylim(
        0,
        8,
    )

    ax.axis(
        "off"
    )

    width = 2.3
    height = 1.15

    # --------------------------------------------------------
    # TOP ROW
    # --------------------------------------------------------

    boxes_top = [
        (
            0.4,
            6.2,
            "Raw clinical data",
            "ENV_raw, ENV_pos,\nKnown_CPO_pos + WGS workbook",
        ),
        (
            3.2,
            6.2,
            "Acquisition cases",
            "Identify 8 patients and\nacquisition dates",
        ),
        (
            6.0,
            6.2,
            "Candidate sources",
            "Enumerate earlier patient\nand environmental sources",
        ),
        (
            8.8,
            6.2,
            "Pair features",
            "Chronology, CP family,\ngene, species and ST",
        ),
        (
            11.6,
            6.2,
            "WGS linkage",
            "Attach clonal and\nplasmid evidence",
        ),
    ]

    for (
        x,
        y,
        title,
        subtitle,
    ) in boxes_top:

        add_box(
            ax,
            x,
            y,
            width,
            height,
            title,
            subtitle,
        )

    for index in range(
        len(
            boxes_top
        )
        - 1
    ):

        x1 = (
            boxes_top[
                index
            ][0]
            + width
        )

        x2 = (
            boxes_top[
                index
                + 1
            ][0]
        )

        y = (
            boxes_top[
                index
            ][1]
            + height
            / 2
        )

        arrow(
            ax,
            (
                x1,
                y,
            ),
            (
                x2,
                y,
            ),
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    boxes_bottom = [
        (
            1.8,
            3.6,
            "ML dataset",
            "999 WGS-evaluable prior\ncandidate pairs",
        ),
        (
            4.8,
            3.6,
            "XGBoost ranking",
            "Leave-one-target-out\ncandidate prioritisation",
        ),
        (
            7.8,
            3.6,
            "SHAP",
            "Explain why candidates\nwere prioritised",
        ),
        (
            10.8,
            3.6,
            "LLM council",
            "Evidence synthesis,\nconsensus and validation",
        ),
    ]

    for (
        x,
        y,
        title,
        subtitle,
    ) in boxes_bottom:

        add_box(
            ax,
            x,
            y,
            width,
            height,
            title,
            subtitle,
        )

    for index in range(
        len(
            boxes_bottom
        )
        - 1
    ):

        x1 = (
            boxes_bottom[
                index
            ][0]
            + width
        )

        x2 = (
            boxes_bottom[
                index
                + 1
            ][0]
        )

        y = (
            boxes_bottom[
                index
            ][1]
            + height
            / 2
        )

        arrow(
            ax,
            (
                x1,
                y,
            ),
            (
                x2,
                y,
            ),
        )

    # WGS -> ML
    arrow(
        ax,
        (
            12.75,
            6.2,
        ),
        (
            2.95,
            4.75,
        ),
    )

    # --------------------------------------------------------
    # FINAL INTERPRETATION
    # --------------------------------------------------------

    add_box(
        ax,
        5.85,
        1.25,
        3.2,
        1.15,
        "Clinical source interpretation",
        (
            "Patient-associated, environment-associated,\n"
            "no direct WGS link, or unresolved"
        ),
    )

    arrow(
        ax,
        (
            11.95,
            3.6,
        ),
        (
            7.45,
            2.4,
        ),
    )

    # --------------------------------------------------------
    # EVIDENCE HIERARCHY
    # --------------------------------------------------------

    ax.text(
        7,
        0.45,
        (
            "Evidence hierarchy: "
            "WGS linkage > microbiology/chronology > "
            "XGBoost prioritisation > LLM synthesis"
        ),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(
        "Evidence-grounded framework for CPO acquisition-source analysis",
        fontsize=14,
        pad=15,
    )

    fig.savefig(
        FIGURE_DIR
        / "figure_01_analysis_workflow.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR
        / "figure_01_analysis_workflow.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved workflow diagram to:"
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()