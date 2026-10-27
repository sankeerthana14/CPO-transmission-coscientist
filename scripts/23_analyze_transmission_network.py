from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import networkx as nx


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

NETWORK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "network"
)

NETWORK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EDGE_FILE = (
    NETWORK_DIR
    / "transmission_network_edges.csv"
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


def clean_edges():

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
        "same_exact_cp_gene_isolate",
        "same_species_any",
        "same_st_any",
    ]:

        if column not in edges.columns:
            edges[column] = False

        edges[column] = bool_series(
            edges[column]
        )

    for column in [
        "rank_within_target",
        "reciprocal_rank",
        "predicted_probability",
        "days_before_acquisition",
    ]:

        if column in edges.columns:

            edges[column] = pd.to_numeric(
                edges[column],
                errors="coerce",
            )

    return edges


# ============================================================
# SOURCE HUB ANALYSIS
# ============================================================

def source_hub_analysis(
    edges,
):

    rows = []

    for source_node, group in edges.groupby(
        "source_node"
    ):

        top1 = group[
            group[
                "top1"
            ]
        ]

        top3 = group[
            group[
                "top3"
            ]
        ]

        top5 = group[
            group[
                "top5"
            ]
        ]

        wgs = group[
            group[
                "any_wgs_link"
            ]
        ]

        clonal = group[
            group[
                "clonal_link"
            ]
        ]

        plasmid = group[
            group[
                "plasmid_link"
            ]
        ]

        ranking_support = (
            top5[
                "reciprocal_rank"
            ]
            .fillna(0)
            .sum()
        )

        rows.append(
            {
                "source_node":
                    source_node,

                "source_type":
                    group[
                        "source_type"
                    ]
                    .iloc[0],

                "source_subject_id":
                    group[
                        "source_subject_id"
                    ]
                    .iloc[0],

                "source_sample_ids":
                    "; ".join(
                        sorted(
                            set(
                                group[
                                    "source_sample_id"
                                ]
                                .dropna()
                                .astype(str)
                            )
                        )
                    ),

                "candidate_target_count":
                    int(
                        group[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "top1_target_count":
                    int(
                        top1[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "top3_target_count":
                    int(
                        top3[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "top5_target_count":
                    int(
                        top5[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "wgs_linked_target_count":
                    int(
                        wgs[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "clonal_target_count":
                    int(
                        clonal[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "plasmid_target_count":
                    int(
                        plasmid[
                            "target_id"
                        ]
                        .nunique()
                    ),

                "top5_reciprocal_rank_support":
                    float(
                        ranking_support
                    ),

                "best_rank":
                    float(
                        group[
                            "rank_within_target"
                        ]
                        .min()
                    ),

                "max_xgboost_score":
                    float(
                        group[
                            "predicted_probability"
                        ]
                        .max()
                    ),

                "closest_days_before_acquisition":
                    float(
                        group[
                            "days_before_acquisition"
                        ]
                        .min()
                    ),

                "exact_gene_match_target_count":
                    int(
                        group.loc[
                            group[
                                "same_exact_cp_gene_isolate"
                            ],
                            "target_id",
                        ]
                        .nunique()
                    ),

                "same_species_target_count":
                    int(
                        group.loc[
                            group[
                                "same_species_any"
                            ],
                            "target_id",
                        ]
                        .nunique()
                    ),

                "recurrent_top5_source":
                    bool(
                        top5[
                            "target_id"
                        ]
                        .nunique()
                        >= 2
                    ),

                "genomic_hub":
                    bool(
                        wgs[
                            "target_id"
                        ]
                        .nunique()
                        >= 2
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "wgs_linked_target_count",
            "top5_target_count",
            "top5_reciprocal_rank_support",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return result


# ============================================================
# TARGET-LEVEL SOURCE CATEGORY ANALYSIS
# ============================================================

def target_source_profiles(
    edges,
):

    rows = []

    for target_id, group in edges.groupby(
        "target_id"
    ):

        patient = group[
            group[
                "source_type"
            ]
            ==
            "patient"
        ]

        environment = group[
            group[
                "source_type"
            ]
            ==
            "environment"
        ]

        patient_top5 = patient[
            patient[
                "top5"
            ]
        ]

        environment_top5 = environment[
            environment[
                "top5"
            ]
        ]

        patient_wgs = patient[
            patient[
                "any_wgs_link"
            ]
        ]

        environment_wgs = environment[
            environment[
                "any_wgs_link"
            ]
        ]

        # ----------------------------------------------------
        # Ranking support.
        #
        # This is deliberately based on reciprocal rank,
        # not XGBoost probability.
        #
        # It is a RELATIVE PRIORITISATION measure,
        # not a probability of transmission.
        # ----------------------------------------------------

        patient_rr = float(
            patient_top5[
                "reciprocal_rank"
            ]
            .fillna(0)
            .sum()
        )

        environment_rr = float(
            environment_top5[
                "reciprocal_rank"
            ]
            .fillna(0)
            .sum()
        )

        # ----------------------------------------------------
        # WGS-supported source category
        # ----------------------------------------------------

        n_patient_wgs = int(
            patient_wgs.shape[0]
        )

        n_environment_wgs = int(
            environment_wgs.shape[0]
        )

        if (
            n_patient_wgs > 0
            and n_environment_wgs == 0
        ):

            wgs_category = (
                "PATIENT"
            )

        elif (
            n_environment_wgs > 0
            and n_patient_wgs == 0
        ):

            wgs_category = (
                "ENVIRONMENT"
            )

        elif (
            n_environment_wgs > 0
            and n_patient_wgs > 0
        ):

            wgs_category = (
                "MIXED"
            )

        else:

            wgs_category = (
                "NONE"
            )

        # ----------------------------------------------------
        # Ranking-favoured category
        # ----------------------------------------------------

        if patient_rr > environment_rr:

            rank_category = (
                "PATIENT"
            )

        elif environment_rr > patient_rr:

            rank_category = (
                "ENVIRONMENT"
            )

        else:

            rank_category = (
                "TIE"
            )

        # ----------------------------------------------------
        # Combined transparent interpretation
        # ----------------------------------------------------

        if wgs_category == "PATIENT":

            network_interpretation = (
                "PATIENT_WGS_SUPPORTED"
            )

        elif wgs_category == "ENVIRONMENT":

            network_interpretation = (
                "ENVIRONMENT_WGS_SUPPORTED"
            )

        elif wgs_category == "MIXED":

            network_interpretation = (
                "MIXED_WGS_SUPPORTED"
            )

        else:

            network_interpretation = (
                f"{rank_category}"
                "_RANK_FAVOURED_NO_DIRECT_WGS"
            )

        # ----------------------------------------------------
        # Direct linked source string
        # ----------------------------------------------------

        linked_sources = []

        linked = group[
            group[
                "any_wgs_link"
            ]
        ]

        for _, row in linked.sort_values(
            "days_before_acquisition"
        ).iterrows():

            if row[
                "clonal_link"
            ]:

                mechanism = (
                    "clonal"
                )

            elif row[
                "plasmid_link"
            ]:

                mechanism = (
                    "plasmid"
                )

            else:

                mechanism = (
                    "WGS"
                )

            linked_sources.append(
                (
                    f"{row['source_node']}"
                    f" ({int(row['days_before_acquisition'])} d; "
                    f"{mechanism})"
                )
            )

        rows.append(
            {
                "target_id":
                    target_id,

                "patient_candidate_count":
                    int(
                        len(patient)
                    ),

                "environment_candidate_count":
                    int(
                        len(environment)
                    ),

                "patient_top5_count":
                    int(
                        len(
                            patient_top5
                        )
                    ),

                "environment_top5_count":
                    int(
                        len(
                            environment_top5
                        )
                    ),

                "patient_top5_rr_support":
                    patient_rr,

                "environment_top5_rr_support":
                    environment_rr,

                "rank_favoured_category":
                    rank_category,

                "patient_wgs_link_count":
                    n_patient_wgs,

                "environment_wgs_link_count":
                    n_environment_wgs,

                "patient_clonal_link_count":
                    int(
                        patient[
                            "clonal_link"
                        ]
                        .sum()
                    ),

                "patient_plasmid_link_count":
                    int(
                        patient[
                            "plasmid_link"
                        ]
                        .sum()
                    ),

                "environment_clonal_link_count":
                    int(
                        environment[
                            "clonal_link"
                        ]
                        .sum()
                    ),

                "environment_plasmid_link_count":
                    int(
                        environment[
                            "plasmid_link"
                        ]
                        .sum()
                    ),

                "wgs_supported_source_category":
                    wgs_category,

                "network_interpretation":
                    network_interpretation,

                "direct_wgs_linked_sources":
                    "; ".join(
                        linked_sources
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "target_id"
    )


# ============================================================
# ACQUISITION-TARGET PROJECTION
#
# Two acquisition targets become connected if they share
# an important candidate source.
# ============================================================

def build_target_projection(
    edges,
):

    evidence = edges[
        edges[
            "top5"
        ]
        |
        edges[
            "any_wgs_link"
        ]
    ].copy()

    pair_stats = defaultdict(
        lambda: {
            "shared_evidence_sources":
                0,

            "shared_top5_sources":
                0,

            "shared_wgs_sources":
                0,

            "shared_clonal_sources":
                0,

            "shared_plasmid_sources":
                0,

            "source_nodes":
                [],
        }
    )

    for source_node, group in evidence.groupby(
        "source_node"
    ):

        target_rows = {
            str(row[
                "target_id"
            ]):
                row

            for _, row
            in group.iterrows()
        }

        targets = sorted(
            target_rows.keys()
        )

        if len(targets) < 2:
            continue

        for a, b in combinations(
            targets,
            2,
        ):

            key = (
                a,
                b,
            )

            row_a = target_rows[a]
            row_b = target_rows[b]

            pair_stats[
                key
            ][
                "shared_evidence_sources"
            ] += 1

            pair_stats[
                key
            ][
                "source_nodes"
            ].append(
                source_node
            )

            if (
                row_a[
                    "top5"
                ]
                and
                row_b[
                    "top5"
                ]
            ):

                pair_stats[
                    key
                ][
                    "shared_top5_sources"
                ] += 1

            if (
                row_a[
                    "any_wgs_link"
                ]
                and
                row_b[
                    "any_wgs_link"
                ]
            ):

                pair_stats[
                    key
                ][
                    "shared_wgs_sources"
                ] += 1

            if (
                row_a[
                    "clonal_link"
                ]
                and
                row_b[
                    "clonal_link"
                ]
            ):

                pair_stats[
                    key
                ][
                    "shared_clonal_sources"
                ] += 1

            if (
                row_a[
                    "plasmid_link"
                ]
                and
                row_b[
                    "plasmid_link"
                ]
            ):

                pair_stats[
                    key
                ][
                    "shared_plasmid_sources"
                ] += 1

    rows = []

    for (
        target_a,
        target_b,
    ), stats in pair_stats.items():

        rows.append(
            {
                "target_a":
                    target_a,

                "target_b":
                    target_b,

                **{
                    key:
                        value

                    for key, value
                    in stats.items()

                    if key
                    != "source_nodes"
                },

                "shared_source_nodes":
                    "; ".join(
                        sorted(
                            set(
                                stats[
                                    "source_nodes"
                                ]
                            )
                        )
                    ),
            }
        )

    if rows:

        result = pd.DataFrame(
            rows
        )

        result = result.sort_values(
            [
                "shared_wgs_sources",
                "shared_top5_sources",
                "shared_evidence_sources",
            ],
            ascending=False,
        )

    else:

        result = pd.DataFrame(
            columns=[
                "target_a",
                "target_b",
                "shared_evidence_sources",
                "shared_top5_sources",
                "shared_wgs_sources",
                "shared_clonal_sources",
                "shared_plasmid_sources",
                "shared_source_nodes",
            ]
        )

    return result


# ============================================================
# CONNECTED COMPONENT ANALYSIS
# ============================================================

def component_analysis(
    edges,
):

    evidence = edges[
        edges[
            "top5"
        ]
        |
        edges[
            "any_wgs_link"
        ]
    ].copy()

    graph = nx.Graph()

    for _, row in evidence.iterrows():

        graph.add_edge(
            row[
                "source_node"
            ],
            f"T:{row['target_id']}",
        )

    rows = []

    for component_id, component in enumerate(
        nx.connected_components(
            graph
        ),
        start=1,
    ):

        targets = sorted(
            node
            for node in component
            if node.startswith(
                "T:"
            )
        )

        sources = sorted(
            node
            for node in component
            if not node.startswith(
                "T:"
            )
        )

        rows.append(
            {
                "component_id":
                    component_id,

                "n_nodes":
                    len(
                        component
                    ),

                "n_targets":
                    len(
                        targets
                    ),

                "n_sources":
                    len(
                        sources
                    ),

                "targets":
                    "; ".join(
                        targets
                    ),

                "sources":
                    "; ".join(
                        sources
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "n_targets",
                "n_nodes",
            ],
            ascending=False,
        )
    )


# ============================================================
# NETWORK SUMMARY
# ============================================================

def network_summary(
    edges,
    hubs,
    projection,
):

    evidence_edges = edges[
        edges[
            "top5"
        ]
        |
        edges[
            "any_wgs_link"
        ]
    ]

    return {
        "n_candidate_edges":
            int(
                len(
                    edges
                )
            ),

        "n_evidence_edges_top5_or_wgs":
            int(
                len(
                    evidence_edges
                )
            ),

        "n_unique_source_nodes":
            int(
                edges[
                    "source_node"
                ]
                .nunique()
            ),

        "n_targets":
            int(
                edges[
                    "target_id"
                ]
                .nunique()
            ),

        "n_direct_wgs_edges":
            int(
                edges[
                    "any_wgs_link"
                ]
                .sum()
            ),

        "n_clonal_edges":
            int(
                edges[
                    "clonal_link"
                ]
                .sum()
            ),

        "n_plasmid_edges":
            int(
                edges[
                    "plasmid_link"
                ]
                .sum()
            ),

        "n_recurrent_top5_sources":
            int(
                hubs[
                    "recurrent_top5_source"
                ]
                .sum()
            ),

        "n_genomic_hub_sources":
            int(
                hubs[
                    "genomic_hub"
                ]
                .sum()
            ),

        "n_target_pairs_sharing_evidence_source":
            int(
                len(
                    projection
                )
            ),

        "n_target_pairs_sharing_wgs_source":
            int(
                (
                    projection[
                        "shared_wgs_sources"
                    ]
                    > 0
                )
                .sum()
            )
            if len(
                projection
            )
            else 0,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    edges = clean_edges()

    hubs = source_hub_analysis(
        edges
    )

    target_profiles = (
        target_source_profiles(
            edges
        )
    )

    projection = (
        build_target_projection(
            edges
        )
    )

    components = (
        component_analysis(
            edges
        )
    )

    summary = network_summary(
        edges,
        hubs,
        projection,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    hubs.to_csv(
        NETWORK_DIR
        / "network_source_hubs.csv",
        index=False,
    )

    target_profiles.to_csv(
        NETWORK_DIR
        / "network_target_source_profiles.csv",
        index=False,
    )

    projection.to_csv(
        NETWORK_DIR
        / "network_target_projection.csv",
        index=False,
    )

    components.to_csv(
        NETWORK_DIR
        / "network_connected_components.csv",
        index=False,
    )

    with (
        NETWORK_DIR
        / "network_analysis_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT THE MOST IMPORTANT RESULTS
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "NETWORK ANALYSIS SUMMARY"
    )

    print(
        "==========================================\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nTARGET-LEVEL SOURCE INTERPRETATION\n"
    )

    print(
        target_profiles[
            [
                "target_id",
                "rank_favoured_category",
                "wgs_supported_source_category",
                "network_interpretation",
                "patient_wgs_link_count",
                "environment_wgs_link_count",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTOP RECURRENT / GENOMIC SOURCE HUBS\n"
    )

    display_hubs = hubs[
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
    ].head(
        20
    )

    print(
        display_hubs[
            [
                "source_node",
                "source_type",
                "top5_target_count",
                "wgs_linked_target_count",
                "clonal_target_count",
                "plasmid_target_count",
                "top5_reciprocal_rank_support",
            ]
        ].to_string(
            index=False
        )
    )

    if len(
        projection
    ):

        print(
            "\nTARGET PAIRS SHARING SOURCES\n"
        )

        print(
            projection.head(
                20
            ).to_string(
                index=False
            )
        )

    print(
        "\nSaved network analysis outputs to:"
    )

    print(
        NETWORK_DIR
    )


if __name__ == "__main__":
    main()