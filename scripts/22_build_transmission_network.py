from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "network"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PRED_FILE = (
    PROCESSED
    / "xgboost_oof_predictions.csv"
)

WGS_FILE = (
    PROCESSED
    / "wgs_labeled_pairs.csv"
)


# ============================================================
# HELPERS
# ============================================================

def bool_series(series: pd.Series) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

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


def bool_value(value) -> bool:

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


def first_non_null(series):

    clean = series.dropna()

    if len(clean) == 0:
        return np.nan

    return clean.iloc[0]


def join_unique(series):

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = [
        value
        for value in values
        if value
        and value.lower()
        != "nan"
    ]

    return "; ".join(
        sorted(
            set(values)
        )
    )


def normalise_source_type(value):

    value = (
        str(value)
        .strip()
        .lower()
    )

    if "environment" in value:
        return "environment"

    if (
        "patient" in value
        or "known" in value
    ):
        return "patient"

    return "unknown"


def make_source_node(row):

    source_type = normalise_source_type(
        row.get(
            "source_type",
            ""
        )
    )

    subject = str(
        row.get(
            "source_subject_id",
            ""
        )
    ).strip()

    sample = str(
        row.get(
            "source_sample_id",
            ""
        )
    ).strip()

    # Patient = patient-level node.
    # Environmental source = sample-level node.
    #
    # This preserves the epidemiologically meaningful entity.
    if source_type == "patient":

        return f"P:{subject}"

    if source_type == "environment":

        return f"E:{sample}"

    return (
        f"U:{subject}:{sample}"
    )


def safe_numeric(
    frame,
    column,
):

    if column not in frame.columns:

        return pd.Series(
            np.nan,
            index=frame.index,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def ensure_boolean_column(
    frame,
    column,
):

    if column not in frame.columns:

        frame[column] = False

    frame[column] = bool_series(
        frame[column]
    )

    return frame


# ============================================================
# LOAD + MERGE
# ============================================================

def load_data():

    predictions = pd.read_csv(
        PRED_FILE
    )

    wgs = pd.read_csv(
        WGS_FILE
    )

    # --------------------------------------------------------
    # Keep the augmented XGBoost model only.
    # --------------------------------------------------------

    predictions = predictions[
        predictions[
            "model_name"
        ]
        .astype(str)
        .str.upper()
        ==
        "AUGMENTED"
    ].copy()

    # --------------------------------------------------------
    # Strictly prior candidate sources only.
    # Same-day observations remain useful descriptively,
    # but are excluded from the directed network.
    # --------------------------------------------------------

    predictions[
        "days_before_acquisition"
    ] = safe_numeric(
        predictions,
        "days_before_acquisition",
    )

    predictions = predictions[
        predictions[
            "days_before_acquisition"
        ]
        > 0
    ].copy()

    # --------------------------------------------------------
    # Find robust merge keys.
    # --------------------------------------------------------

    candidate_keys = [
        "target_id",
        "source_subject_id",
        "source_sample_id",
        "source_type",
    ]

    merge_keys = [
        column
        for column in candidate_keys
        if (
            column in predictions.columns
            and column in wgs.columns
        )
    ]

    if len(merge_keys) < 3:

        raise RuntimeError(
            "Could not identify sufficient common keys "
            "between XGBoost predictions and WGS-labelled pairs."
        )

    # --------------------------------------------------------
    # Collapse WGS duplicates at source-observation × target.
    # --------------------------------------------------------

    for column in [
        "clonal_link",
        "plasmid_link",
        "any_wgs_link",
    ]:

        wgs = ensure_boolean_column(
            wgs,
            column,
        )

    boolean_candidates = [
        "clonal_link",
        "plasmid_link",
        "any_wgs_link",
        "same_exact_cp_gene_isolate",
        "same_cp_family_isolate",
        "same_cp_family_gx",
        "same_species_any",
        "same_st_any",
    ]

    agg_dict = {}

    for column in boolean_candidates:

        if column in wgs.columns:

            if column not in {
                "clonal_link",
                "plasmid_link",
                "any_wgs_link",
            }:

                wgs = ensure_boolean_column(
                    wgs,
                    column,
                )

            agg_dict[column] = "max"

    if (
        "days_before_acquisition"
        in wgs.columns
    ):

        agg_dict[
            "days_before_acquisition"
        ] = "min"

    # Carry useful biological descriptors if present.
    descriptor_candidates = [
        "source_species",
        "source_st",
        "source_cp_gene",
        "target_species",
        "target_st",
        "target_cp_gene",
        "target_cp_family",
        "source_cp_family",
    ]

    for column in descriptor_candidates:

        if column in wgs.columns:

            agg_dict[column] = first_non_null

    wgs_agg = (
        wgs.groupby(
            merge_keys,
            dropna=False,
        )
        .agg(
            agg_dict
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Avoid duplicated column collisions.
    # --------------------------------------------------------

    overlapping = [
        column
        for column in wgs_agg.columns
        if (
            column in predictions.columns
            and column not in merge_keys
        )
    ]

    wgs_agg = wgs_agg.rename(
        columns={
            column:
                f"{column}__wgs"

            for column in overlapping
        }
    )

    merged = predictions.merge(
        wgs_agg,
        on=merge_keys,
        how="left",
    )

    # --------------------------------------------------------
    # Consolidate WGS / feature fields after merge.
    # --------------------------------------------------------

    for column in boolean_candidates:

        wgs_column = (
            f"{column}__wgs"
        )

        if wgs_column in merged.columns:

            merged[column] = bool_series(
                merged[wgs_column]
            )

        elif column in merged.columns:

            merged[column] = bool_series(
                merged[column]
            )

        else:

            merged[column] = False

    for column in descriptor_candidates:

        wgs_column = (
            f"{column}__wgs"
        )

        if (
            column not in merged.columns
            and wgs_column in merged.columns
        ):

            merged[column] = (
                merged[
                    wgs_column
                ]
            )

    # --------------------------------------------------------
    # Standardise key numeric variables.
    # --------------------------------------------------------

    merged[
        "rank_within_target"
    ] = safe_numeric(
        merged,
        "rank_within_target",
    )

    merged[
        "predicted_probability"
    ] = safe_numeric(
        merged,
        "predicted_probability",
    )

    merged[
        "days_before_acquisition"
    ] = safe_numeric(
        merged,
        "days_before_acquisition",
    )

    merged[
        "source_type"
    ] = (
        merged[
            "source_type"
        ]
        .apply(
            normalise_source_type
        )
    )

    merged[
        "source_node"
    ] = merged.apply(
        make_source_node,
        axis=1,
    )

    merged[
        "target_node"
    ] = (
        "T:"
        +
        merged[
            "target_id"
        ]
        .astype(str)
    )

    return merged


# ============================================================
# AGGREGATE SOURCE ENTITY → TARGET EDGES
# ============================================================

def build_edge_table(
    merged: pd.DataFrame,
):

    # One source patient may have multiple samples.
    # We collapse them into one source-entity → target edge
    # while retaining the strongest/closest evidence.

    group_cols = [
        "source_node",
        "target_node",
        "target_id",
        "source_type",
    ]

    agg = {
        "source_subject_id":
            first_non_null,

        "source_sample_id":
            join_unique,

        "rank_within_target":
            "min",

        "predicted_probability":
            "max",

        "days_before_acquisition":
            "min",

        "clonal_link":
            "max",

        "plasmid_link":
            "max",

        "any_wgs_link":
            "max",

        "same_exact_cp_gene_isolate":
            "max",

        "same_cp_family_isolate":
            "max",

        "same_cp_family_gx":
            "max",

        "same_species_any":
            "max",

        "same_st_any":
            "max",
    }

    # Optional biological descriptors
    for column in [
        "source_species",
        "source_st",
        "source_cp_gene",
        "target_species",
        "target_st",
        "target_cp_gene",
        "target_cp_family",
        "source_cp_family",
    ]:

        if column in merged.columns:

            agg[column] = first_non_null

    edges = (
        merged.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            agg
        )
        .reset_index()
    )

    # Count how many raw source observations contributed
    counts = (
        merged.groupby(
            group_cols,
            dropna=False,
        )
        .size()
        .reset_index(
            name="n_supporting_candidate_rows"
        )
    )

    edges = edges.merge(
        counts,
        on=group_cols,
        how="left",
    )

    # --------------------------------------------------------
    # Ranking layers
    # --------------------------------------------------------

    edges[
        "top1"
    ] = (
        edges[
            "rank_within_target"
        ]
        <= 1
    )

    edges[
        "top3"
    ] = (
        edges[
            "rank_within_target"
        ]
        <= 3
    )

    edges[
        "top5"
    ] = (
        edges[
            "rank_within_target"
        ]
        <= 5
    )

    # Reciprocal rank is a useful transparent ranking measure.
    # It is NOT a probability.
    edges[
        "reciprocal_rank"
    ] = (
        1.0
        /
        edges[
            "rank_within_target"
        ]
    )

    # --------------------------------------------------------
    # Evidence layer classification
    # --------------------------------------------------------

    def edge_layer(row):

        clonal = bool_value(
            row[
                "clonal_link"
            ]
        )

        plasmid = bool_value(
            row[
                "plasmid_link"
            ]
        )

        if clonal and plasmid:
            return "WGS_CLONAL_AND_PLASMID"

        if clonal:
            return "WGS_CLONAL"

        if plasmid:
            return "WGS_PLASMID"

        if bool_value(
            row[
                "top1"
            ]
        ):
            return "ML_TOP1"

        if bool_value(
            row[
                "top3"
            ]
        ):
            return "ML_TOP3"

        if bool_value(
            row[
                "top5"
            ]
        ):
            return "ML_TOP5"

        return "CANDIDATE"

    edges[
        "evidence_layer"
    ] = edges.apply(
        edge_layer,
        axis=1,
    )

    return edges


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(
    edges,
):

    graph = nx.DiGraph()

    # --------------------------------------------------------
    # Source nodes
    # --------------------------------------------------------

    for source_node, group in edges.groupby(
        "source_node"
    ):

        first = group.iloc[0]

        graph.add_node(
            source_node,
            node_role="source",
            source_type=str(
                first[
                    "source_type"
                ]
            ),
            subject_id=str(
                first.get(
                    "source_subject_id",
                    "",
                )
            ),
            sample_ids=str(
                first.get(
                    "source_sample_id",
                    "",
                )
            ),
        )

    # --------------------------------------------------------
    # Acquisition target nodes
    # --------------------------------------------------------

    for target_id in sorted(
        edges[
            "target_id"
        ]
        .astype(str)
        .unique()
    ):

        graph.add_node(
            f"T:{target_id}",
            node_role="acquisition_target",
            target_id=target_id,
        )

    # --------------------------------------------------------
    # Directed source → target edges
    # --------------------------------------------------------

    for _, row in edges.iterrows():

        attrs = {}

        for column in edges.columns:

            if column in {
                "source_node",
                "target_node",
            }:

                continue

            value = row[column]

            if pd.isna(value):
                continue

            if isinstance(
                value,
                (
                    np.bool_,
                    bool,
                ),
            ):

                value = int(value)

            elif isinstance(
                value,
                (
                    np.integer,
                ),
            ):

                value = int(value)

            elif isinstance(
                value,
                (
                    np.floating,
                    float,
                ),
            ):

                value = float(value)

            else:

                value = str(value)

            attrs[column] = value

        graph.add_edge(
            row[
                "source_node"
            ],
            row[
                "target_node"
            ],
            **attrs,
        )

    return graph


# ============================================================
# MAIN
# ============================================================

def main():

    merged = load_data()

    edges = build_edge_table(
        merged
    )

    graph = build_graph(
        edges
    )

    # --------------------------------------------------------
    # Save raw merged candidate evidence
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_DIR
        / "network_candidate_rows.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save aggregated analytic edges
    # --------------------------------------------------------

    edges.to_csv(
        OUTPUT_DIR
        / "transmission_network_edges.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save node table
    # --------------------------------------------------------

    node_rows = []

    for node, attrs in graph.nodes(
        data=True
    ):

        node_rows.append(
            {
                "node_id":
                    node,

                **attrs,
            }
        )

    pd.DataFrame(
        node_rows
    ).to_csv(
        OUTPUT_DIR
        / "transmission_network_nodes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # GraphML for Gephi/Cytoscape/NetworkX
    # --------------------------------------------------------

    nx.write_graphml(
        graph,
        OUTPUT_DIR
        / "transmission_network.graphml",
    )

    summary = {
        "n_nodes":
            graph.number_of_nodes(),

        "n_edges":
            graph.number_of_edges(),

        "n_patient_source_nodes":
            int(
                sum(
                    1
                    for _, attrs
                    in graph.nodes(
                        data=True
                    )
                    if (
                        attrs.get(
                            "node_role"
                        )
                        == "source"
                        and attrs.get(
                            "source_type"
                        )
                        == "patient"
                    )
                )
            ),

        "n_environment_source_nodes":
            int(
                sum(
                    1
                    for _, attrs
                    in graph.nodes(
                        data=True
                    )
                    if (
                        attrs.get(
                            "node_role"
                        )
                        == "source"
                        and attrs.get(
                            "source_type"
                        )
                        == "environment"
                    )
                )
            ),

        "n_targets":
            int(
                edges[
                    "target_id"
                ]
                .nunique()
            ),

        "n_top5_edges":
            int(
                edges[
                    "top5"
                ]
                .sum()
            ),

        "n_wgs_edges":
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
    }

    with (
        OUTPUT_DIR
        / "network_build_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "\nTransmission network built successfully."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nOutputs:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()