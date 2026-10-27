from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from cpo.config import (
    WGS_FILE,
    PROCESSED_DATA_DIR,
)

from cpo.data_loader import (
    read_sheet,
    normalize_missing,
    coerce_date,
)

from cpo.features import (
    extract_specific_cp_genes,
)


# ============================================================
# BASIC NORMALIZATION
# ============================================================


def clean_value(value):
    """
    Normalize workbook values and return stripped strings.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return np.nan

    return str(value).strip()


def normalize_text(value):
    """
    Normalize text for exact-ish comparisons.

    Used mainly for species/ST strings.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return None

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
    )


def set_to_string(
    values: set[str],
):
    """
    Stable semicolon-separated representation.
    """

    if not values:
        return np.nan

    return ";".join(
        sorted(values)
    )


def values_to_string(
    values: list[Any] | set[Any],
):
    """
    Stable string representation for output tables.
    """

    cleaned = []

    for value in values:

        value = normalize_missing(value)

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value and value not in cleaned:
            cleaned.append(value)

    if not cleaned:
        return np.nan

    return ";".join(
        sorted(cleaned)
    )


# ============================================================
# LOAD / STANDARDISE WGS ISOLATE LIST
# ============================================================


def prepare_wgs_isolates() -> pd.DataFrame:
    """
    Load WGS_isolate_list and standardise the fields we need.
    """

    wgs = read_sheet(
        filepath=WGS_FILE,
        sheet_name="WGS_isolate_list",
        required_column="WGS ID",
    )

    required = [
        "WGS ID",
        "Isolate ID (Sample ID-Isolate No)",
        "Sample ID",
        "Subject ID",
        "Type of subject",
        "Date of Sample",
        "Genomic Species/ST",
        "Genomic CP gene",
    ]

    missing = [
        column
        for column in required
        if column not in wgs.columns
    ]

    if missing:

        raise ValueError(
            "WGS_isolate_list is missing required columns:\n"
            f"{missing}"
        )

    result = pd.DataFrame()

    result["wgs_id"] = (
        wgs["WGS ID"]
        .apply(clean_value)
    )

    result["isolate_id"] = (
        wgs[
            "Isolate ID (Sample ID-Isolate No)"
        ]
        .apply(clean_value)
    )

    result["sample_id"] = (
        wgs["Sample ID"]
        .apply(clean_value)
    )

    result["subject_id"] = (
        wgs["Subject ID"]
        .apply(clean_value)
    )

    result["subject_type"] = (
        wgs["Type of subject"]
        .apply(clean_value)
    )

    result["sample_date"] = (
        wgs["Date of Sample"]
        .apply(coerce_date)
    )

    result["species_st"] = (
        wgs["Genomic Species/ST"]
        .apply(clean_value)
    )

    result["cp_gene"] = (
        wgs["Genomic CP gene"]
        .apply(clean_value)
    )

    # Optional useful metadata
    optional_mapping = {
        "Type of sample/site":
            "sample_site_type",

        "Specific site":
            "specific_site",

        "Sink/Shower drain Unique code":
            "sink_drain_code",

        "WGS_Patient or ENV ID":
            "wgs_patient_env_id",
    }

    for (
        original,
        renamed,
    ) in optional_mapping.items():

        if original in wgs.columns:

            result[
                renamed
            ] = (
                wgs[
                    original
                ]
                .apply(clean_value)
            )

        else:

            result[
                renamed
            ] = np.nan

    # Normalized comparison fields
    result[
        "_species_st_norm"
    ] = (
        result[
            "species_st"
        ]
        .apply(normalize_text)
    )

    result[
        "_cp_gene_set"
    ] = (
        result[
            "cp_gene"
        ]
        .apply(
            extract_specific_cp_genes
        )
    )

    # Remove rows with unusable WGS IDs
    result = result[
        result[
            "wgs_id"
        ].notna()
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# PREPARE LINK TABLES
# ============================================================


def prepare_link_table(
    sheet_name: str,
    link_type: str,
) -> pd.DataFrame:
    """
    Standardise Clonal_linked_pairs or Plasmid_linked_pairs.
    """

    links = read_sheet(
        filepath=WGS_FILE,
        sheet_name=sheet_name,
        required_column="Acquisition isolate (WGS ID)",
    )

    required = [
        "Acquisition isolate (WGS ID)",
        "Acquisition isolate (Isolate ID)",
        "Acquisition isolate date of sample",
        "Source isolate (WGS ID)",
        "Source isolate (Isolate ID)",
        "Source isolate date of sample",
        "Source type",
    ]

    missing = [
        column
        for column in required
        if column not in links.columns
    ]

    if missing:

        raise ValueError(
            f"{sheet_name} is missing required columns:\n"
            f"{missing}"
        )

    result = pd.DataFrame()

    result[
        "acquisition_wgs_id"
    ] = (
        links[
            "Acquisition isolate (WGS ID)"
        ]
        .apply(clean_value)
    )

    result[
        "acquisition_isolate_id"
    ] = (
        links[
            "Acquisition isolate (Isolate ID)"
        ]
        .apply(clean_value)
    )

    result[
        "acquisition_date"
    ] = (
        links[
            "Acquisition isolate date of sample"
        ]
        .apply(coerce_date)
    )

    result[
        "source_wgs_id"
    ] = (
        links[
            "Source isolate (WGS ID)"
        ]
        .apply(clean_value)
    )

    result[
        "source_isolate_id"
    ] = (
        links[
            "Source isolate (Isolate ID)"
        ]
        .apply(clean_value)
    )

    result[
        "source_date"
    ] = (
        links[
            "Source isolate date of sample"
        ]
        .apply(coerce_date)
    )

    result[
        "source_type_wgs"
    ] = (
        links[
            "Source type"
        ]
        .apply(clean_value)
    )

    result[
        "link_type"
    ] = link_type

    if (
        "CP gene of shared plasmid(s)"
        in links.columns
    ):

        result[
            "shared_plasmid_cp_gene"
        ] = (
            links[
                "CP gene of shared plasmid(s)"
            ]
            .apply(clean_value)
        )

    else:

        result[
            "shared_plasmid_cp_gene"
        ] = np.nan

    result = result[
        result[
            "acquisition_wgs_id"
        ].notna()
        &
        result[
            "source_wgs_id"
        ].notna()
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# INDEX WGS ISOLATES BY SAMPLE
# ============================================================


def build_sample_index(
    wgs_isolates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Map each Sample ID to its WGS isolates.
    """

    result = {}

    for (
        sample_id,
        group,
    ) in wgs_isolates.groupby(
        "sample_id",
        dropna=True,
    ):

        result[
            str(sample_id)
        ] = (
            group
            .copy()
            .reset_index(
                drop=True
            )
        )

    return result


# ============================================================
# RESOLVE TARGET WGS ISOLATES
# ============================================================


def resolve_target_wgs_isolates(
    row: pd.Series,
    sample_index: dict[str, pd.DataFrame],
) -> tuple[set[str], str]:
    """
    Resolve the target acquisition row to its actual WGS IDs.

    We first find all WGS isolates from target_sample_id,
    then narrow using species/ST and CP gene when possible.
    """

    sample_id = normalize_missing(
        row.get(
            "target_sample_id"
        )
    )

    if pd.isna(sample_id):

        return (
            set(),
            "NO_TARGET_SAMPLE_ID",
        )

    sample_id = str(
        sample_id
    ).strip()

    if sample_id not in sample_index:

        return (
            set(),
            "TARGET_SAMPLE_NOT_IN_WGS",
        )

    candidates = (
        sample_index[
            sample_id
        ]
        .copy()
    )

    if candidates.empty:

        return (
            set(),
            "TARGET_SAMPLE_NOT_IN_WGS",
        )

    original_count = len(
        candidates
    )

    # --------------------------------------------------------
    # Species/ST matching
    # --------------------------------------------------------

    target_species_st = (
        normalize_text(
            row.get(
                "target_species_st"
            )
        )
    )

    species_filtered = False

    if target_species_st is not None:

        matching = candidates[
            candidates[
                "_species_st_norm"
            ]
            == target_species_st
        ].copy()

        if not matching.empty:

            candidates = matching
            species_filtered = True

    # --------------------------------------------------------
    # Exact CP-gene matching
    # --------------------------------------------------------

    target_genes = (
        extract_specific_cp_genes(
            row.get(
                "target_cp_gene"
            )
        )
    )

    gene_filtered = False

    if target_genes:

        matching = candidates[
            candidates[
                "_cp_gene_set"
            ]
            .apply(
                lambda genes:
                    bool(
                        genes
                        &
                        target_genes
                    )
            )
        ].copy()

        if not matching.empty:

            candidates = matching
            gene_filtered = True

    wgs_ids = set(
        candidates[
            "wgs_id"
        ]
        .dropna()
        .astype(str)
    )

    if not wgs_ids:

        return (
            set(),
            "TARGET_WGS_UNRESOLVED",
        )

    if (
        species_filtered
        and gene_filtered
    ):

        status = (
            "TARGET_RESOLVED_SPECIES_GENE"
        )

    elif species_filtered:

        status = (
            "TARGET_RESOLVED_SPECIES"
        )

    elif gene_filtered:

        status = (
            "TARGET_RESOLVED_GENE"
        )

    elif original_count == 1:

        status = (
            "TARGET_SINGLE_WGS_ISOLATE"
        )

    else:

        status = (
            "TARGET_SAMPLE_LEVEL_MULTIPLE"
        )

    return (
        wgs_ids,
        status,
    )


# ============================================================
# RESOLVE SOURCE WGS ISOLATES
# ============================================================


def resolve_source_wgs_isolates(
    row: pd.Series,
    sample_index: dict[str, pd.DataFrame],
) -> tuple[set[str], str]:
    """
    Resolve a candidate source row to actual WGS IDs.

    Patient-source rows are isolate-specific, so we attempt
    species/ST + gene resolution.

    Environmental candidate rows are sample-level and can
    contain multiple isolates, so all WGS isolates from that
    environmental sample remain eligible.
    """

    sample_id = normalize_missing(
        row.get(
            "source_sample_id"
        )
    )

    if pd.isna(sample_id):

        return (
            set(),
            "NO_SOURCE_SAMPLE_ID",
        )

    sample_id = str(
        sample_id
    ).strip()

    if sample_id not in sample_index:

        return (
            set(),
            "SOURCE_SAMPLE_NOT_IN_WGS",
        )

    candidates = (
        sample_index[
            sample_id
        ]
        .copy()
    )

    if candidates.empty:

        return (
            set(),
            "SOURCE_SAMPLE_NOT_IN_WGS",
        )

    source_type = str(
        row.get(
            "source_type",
            "",
        )
    ).strip().lower()

    # --------------------------------------------------------
    # Environment candidate = sample-level source
    # --------------------------------------------------------

    if (
        source_type
        == "environment"
    ):

        wgs_ids = set(
            candidates[
                "wgs_id"
            ]
            .dropna()
            .astype(str)
        )

        if not wgs_ids:

            return (
                set(),
                "ENVIRONMENT_WGS_UNRESOLVED",
            )

        if len(
            wgs_ids
        ) == 1:

            status = (
                "ENVIRONMENT_SINGLE_WGS_ISOLATE"
            )

        else:

            status = (
                "ENVIRONMENT_SAMPLE_ALL_ISOLATES"
            )

        return (
            wgs_ids,
            status,
        )

    # --------------------------------------------------------
    # Patient candidate = isolate-specific source
    # --------------------------------------------------------

    original_count = len(
        candidates
    )

    source_species_st = (
        normalize_text(
            row.get(
                "source_species_st"
            )
        )
    )

    species_filtered = False

    if source_species_st is not None:

        matching = candidates[
            candidates[
                "_species_st_norm"
            ]
            == source_species_st
        ].copy()

        if not matching.empty:

            candidates = matching
            species_filtered = True

    source_genes = (
        extract_specific_cp_genes(
            row.get(
                "source_cp_gene"
            )
        )
    )

    gene_filtered = False

    if source_genes:

        matching = candidates[
            candidates[
                "_cp_gene_set"
            ]
            .apply(
                lambda genes:
                    bool(
                        genes
                        &
                        source_genes
                    )
            )
        ].copy()

        if not matching.empty:

            candidates = matching
            gene_filtered = True

    wgs_ids = set(
        candidates[
            "wgs_id"
        ]
        .dropna()
        .astype(str)
    )

    if not wgs_ids:

        return (
            set(),
            "PATIENT_SOURCE_WGS_UNRESOLVED",
        )

    if (
        species_filtered
        and gene_filtered
    ):

        status = (
            "PATIENT_SOURCE_RESOLVED_SPECIES_GENE"
        )

    elif species_filtered:

        status = (
            "PATIENT_SOURCE_RESOLVED_SPECIES"
        )

    elif gene_filtered:

        status = (
            "PATIENT_SOURCE_RESOLVED_GENE"
        )

    elif original_count == 1:

        status = (
            "PATIENT_SOURCE_SINGLE_WGS_ISOLATE"
        )

    else:

        status = (
            "PATIENT_SOURCE_SAMPLE_LEVEL_MULTIPLE"
        )

    return (
        wgs_ids,
        status,
    )


# ============================================================
# BUILD LINK LOOKUP
# ============================================================


def build_link_lookup(
    links: pd.DataFrame,
) -> dict[
    tuple[str, str],
    list[dict[str, Any]],
]:
    """
    Map:
        (acquisition_wgs_id, source_wgs_id)

    to one or more linkage records.
    """

    lookup = defaultdict(
        list
    )

    for _, row in links.iterrows():

        acquisition_id = str(
            row[
                "acquisition_wgs_id"
            ]
        ).strip()

        source_id = str(
            row[
                "source_wgs_id"
            ]
        ).strip()

        lookup[
            (
                acquisition_id,
                source_id,
            )
        ].append(
            row.to_dict()
        )

    return dict(
        lookup
    )


# ============================================================
# QUERY LINK LOOKUP
# ============================================================


def query_linkage(
    target_wgs_ids: set[str],
    source_wgs_ids: set[str],
    lookup: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    """
    Find direct WGS relationships between target and source.

    Forward:
        target appears as Acquisition isolate
        source appears as Source isolate

    Reverse:
        source appears as Acquisition isolate
        target appears as Source isolate

    Genomic relatedness itself is symmetric, but the forward/
    reverse information is retained because the WGS workbook
    includes an inferred acquisition/source orientation.
    """

    forward_records = []

    reverse_records = []

    for target_id in target_wgs_ids:

        for source_id in source_wgs_ids:

            forward_records.extend(
                lookup.get(
                    (
                        target_id,
                        source_id,
                    ),
                    [],
                )
            )

            reverse_records.extend(
                lookup.get(
                    (
                        source_id,
                        target_id,
                    ),
                    [],
                )
            )

    all_records = (
        forward_records
        +
        reverse_records
    )

    evidence_pairs = set()

    shared_plasmid_genes = set()

    for record in all_records:

        acquisition_id = (
            record[
                "acquisition_wgs_id"
            ]
        )

        source_id = (
            record[
                "source_wgs_id"
            ]
        )

        evidence_pairs.add(
            f"{acquisition_id}->{source_id}"
        )

        shared_gene = (
            normalize_missing(
                record.get(
                    "shared_plasmid_cp_gene"
                )
            )
        )

        if not pd.isna(
            shared_gene
        ):

            shared_plasmid_genes.add(
                str(
                    shared_gene
                ).strip()
            )

    return {
        "linked":
            bool(
                all_records
            ),

        "forward_linked":
            bool(
                forward_records
            ),

        "reverse_linked":
            bool(
                reverse_records
            ),

        "forward_count":
            len(
                forward_records
            ),

        "reverse_count":
            len(
                reverse_records
            ),

        "link_record_count":
            len(
                all_records
            ),

        "evidence_pairs":
            values_to_string(
                evidence_pairs
            ),

        "shared_plasmid_genes":
            values_to_string(
                shared_plasmid_genes
            ),
    }


# ============================================================
# MASTER WGS ATTACHMENT
# ============================================================


def attach_wgs_linkage() -> pd.DataFrame:
    """
    Attach direct clonal/plasmid WGS evidence to pair_features.
    """

    feature_file = (
        PROCESSED_DATA_DIR
        / "pair_features.csv"
    )

    if not feature_file.exists():

        raise FileNotFoundError(
            "\npair_features.csv was not found.\n"
            "Run:\n\n"
            "PYTHONPATH=src "
            "python scripts/04_build_pair_features.py"
        )

    features = pd.read_csv(
        feature_file,
        parse_dates=[
            "target_acquisition_date",
            "source_date",
        ],
    )

    original_row_count = len(
        features
    )

    # --------------------------------------------------------
    # Load WGS data
    # --------------------------------------------------------

    wgs_isolates = (
        prepare_wgs_isolates()
    )

    clonal_links = (
        prepare_link_table(
            sheet_name=(
                "Clonal_linked_pairs"
            ),
            link_type=(
                "clonal"
            ),
        )
    )

    plasmid_links = (
        prepare_link_table(
            sheet_name=(
                "Plasmid_linked_pairs"
            ),
            link_type=(
                "plasmid"
            ),
        )
    )

    sample_index = (
        build_sample_index(
            wgs_isolates
        )
    )

    clonal_lookup = (
        build_link_lookup(
            clonal_links
        )
    )

    plasmid_lookup = (
        build_link_lookup(
            plasmid_links
        )
    )

    # --------------------------------------------------------
    # Resolve each candidate to WGS IDs
    # --------------------------------------------------------

    target_wgs_sets = []

    target_resolution = []

    source_wgs_sets = []

    source_resolution = []

    for _, row in features.iterrows():

        (
            target_ids,
            target_status,
        ) = resolve_target_wgs_isolates(
            row=row,
            sample_index=sample_index,
        )

        (
            source_ids,
            source_status,
        ) = resolve_source_wgs_isolates(
            row=row,
            sample_index=sample_index,
        )

        target_wgs_sets.append(
            target_ids
        )

        target_resolution.append(
            target_status
        )

        source_wgs_sets.append(
            source_ids
        )

        source_resolution.append(
            source_status
        )

    features[
        "_target_wgs_ids"
    ] = target_wgs_sets

    features[
        "_source_wgs_ids"
    ] = source_wgs_sets

    features[
        "target_wgs_resolution"
    ] = target_resolution

    features[
        "source_wgs_resolution"
    ] = source_resolution

    features[
        "target_wgs_ids"
    ] = (
        features[
            "_target_wgs_ids"
        ]
        .apply(
            set_to_string
        )
    )

    features[
        "source_wgs_ids"
    ] = (
        features[
            "_source_wgs_ids"
        ]
        .apply(
            set_to_string
        )
    )

    features[
        "target_wgs_isolate_count"
    ] = (
        features[
            "_target_wgs_ids"
        ]
        .apply(
            len
        )
    )

    features[
        "source_wgs_isolate_count"
    ] = (
        features[
            "_source_wgs_ids"
        ]
        .apply(
            len
        )
    )

    features[
        "wgs_pair_evaluable"
    ] = (
        (
            features[
                "target_wgs_isolate_count"
            ] > 0
        )
        &
        (
            features[
                "source_wgs_isolate_count"
            ] > 0
        )
    )

    # ========================================================
    # QUERY CLONAL / PLASMID PAIR TABLES
    # ========================================================

    clonal_results = []

    plasmid_results = []

    for _, row in features.iterrows():

        clonal_results.append(
            query_linkage(
                target_wgs_ids=(
                    row[
                        "_target_wgs_ids"
                    ]
                ),
                source_wgs_ids=(
                    row[
                        "_source_wgs_ids"
                    ]
                ),
                lookup=(
                    clonal_lookup
                ),
            )
        )

        plasmid_results.append(
            query_linkage(
                target_wgs_ids=(
                    row[
                        "_target_wgs_ids"
                    ]
                ),
                source_wgs_ids=(
                    row[
                        "_source_wgs_ids"
                    ]
                ),
                lookup=(
                    plasmid_lookup
                ),
            )
        )

    # --------------------------------------------------------
    # Clonal evidence
    # --------------------------------------------------------

    features[
        "clonal_link"
    ] = [
        result[
            "linked"
        ]
        for result
        in clonal_results
    ]

    features[
        "clonal_forward"
    ] = [
        result[
            "forward_linked"
        ]
        for result
        in clonal_results
    ]

    features[
        "clonal_reverse"
    ] = [
        result[
            "reverse_linked"
        ]
        for result
        in clonal_results
    ]

    features[
        "clonal_link_record_count"
    ] = [
        result[
            "link_record_count"
        ]
        for result
        in clonal_results
    ]

    features[
        "clonal_evidence_pairs"
    ] = [
        result[
            "evidence_pairs"
        ]
        for result
        in clonal_results
    ]

    # --------------------------------------------------------
    # Plasmid evidence
    # --------------------------------------------------------

    features[
        "plasmid_link"
    ] = [
        result[
            "linked"
        ]
        for result
        in plasmid_results
    ]

    features[
        "plasmid_forward"
    ] = [
        result[
            "forward_linked"
        ]
        for result
        in plasmid_results
    ]

    features[
        "plasmid_reverse"
    ] = [
        result[
            "reverse_linked"
        ]
        for result
        in plasmid_results
    ]

    features[
        "plasmid_link_record_count"
    ] = [
        result[
            "link_record_count"
        ]
        for result
        in plasmid_results
    ]

    features[
        "plasmid_evidence_pairs"
    ] = [
        result[
            "evidence_pairs"
        ]
        for result
        in plasmid_results
    ]

    features[
        "shared_plasmid_cp_genes"
    ] = [
        result[
            "shared_plasmid_genes"
        ]
        for result
        in plasmid_results
    ]

    # ========================================================
    # LINK STATUS
    # ========================================================

    features[
        "any_wgs_link"
    ] = (
        features[
            "clonal_link"
        ]
        |
        features[
            "plasmid_link"
        ]
    )

    def clonal_status(
        row: pd.Series,
    ) -> str:

        if row[
            "clonal_link"
        ]:

            return (
                "LINKED"
            )

        if row[
            "wgs_pair_evaluable"
        ]:

            return (
                "NO_DIRECT_CLONAL_LINK_LISTED"
            )

        return (
            "NOT_EVALUABLE"
        )

    def plasmid_status(
        row: pd.Series,
    ) -> str:

        if row[
            "plasmid_link"
        ]:

            return (
                "LINKED"
            )

        if row[
            "wgs_pair_evaluable"
        ]:

            return (
                "NO_DIRECT_PLASMID_LINK_LISTED"
            )

        return (
            "NOT_EVALUABLE"
        )

    features[
        "clonal_link_status"
    ] = features.apply(
        clonal_status,
        axis=1,
    )

    features[
        "plasmid_link_status"
    ] = features.apply(
        plasmid_status,
        axis=1,
    )

    # ========================================================
    # OVERALL WGS MECHANISM
    # ========================================================

    def mechanism(
        row: pd.Series,
    ) -> str:

        if (
            row[
                "clonal_link"
            ]
            and
            row[
                "plasmid_link"
            ]
        ):

            return (
                "CLONAL_AND_PLASMID"
            )

        if row[
            "clonal_link"
        ]:

            return (
                "CLONAL"
            )

        if row[
            "plasmid_link"
        ]:

            return (
                "PLASMID"
            )

        if row[
            "wgs_pair_evaluable"
        ]:

            return (
                "NO_DIRECT_LINK_LISTED"
            )

        return (
            "NOT_EVALUABLE"
        )

    features[
        "wgs_link_mechanism"
    ] = features.apply(
        mechanism,
        axis=1,
    )

    # ========================================================
    # DIRECTION DIAGNOSTIC
    # ========================================================

    def direction_status(
        row: pd.Series,
    ) -> str:

        forward = bool(
            row[
                "clonal_forward"
            ]
            or
            row[
                "plasmid_forward"
            ]
        )

        reverse = bool(
            row[
                "clonal_reverse"
            ]
            or
            row[
                "plasmid_reverse"
            ]
        )

        if (
            forward
            and reverse
        ):

            return (
                "BOTH_DIRECTIONS_LISTED"
            )

        if forward:

            return (
                "TARGET_AS_ACQUISITION"
            )

        if reverse:

            return (
                "TARGET_AS_SOURCE_IN_WGS_TABLE"
            )

        return (
            "NO_LINK_DIRECTION"
        )

    features[
        "wgs_direction_status"
    ] = features.apply(
        direction_status,
        axis=1,
    )

    # ========================================================
    # REMOVE INTERNAL SET COLUMNS
    # ========================================================

    features = features.drop(
        columns=[
            "_target_wgs_ids",
            "_source_wgs_ids",
        ]
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    if len(
        features
    ) != original_row_count:

        raise ValueError(
            "Attaching WGS linkage changed "
            "candidate-pair row count."
        )

    if not features[
        "pair_id"
    ].is_unique:

        raise ValueError(
            "pair_id is not unique after WGS attachment."
        )

    return features