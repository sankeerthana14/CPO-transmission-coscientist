from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cpo.config import (
    PROCESSED_DATA_DIR,
)

from cpo.data_loader import (
    normalize_missing,
)

from cpo.candidate_sources import (
    extract_cp_families,
    families_to_string,
)


# ============================================================
# SPECIES / SEQUENCE TYPE PARSING
# ============================================================


def parse_species_st(
    value,
) -> tuple[object, object]:
    """
    Split strings such as:

        Pseudomonas aeruginosa_ST308

    into:

        species = Pseudomonas aeruginosa
        ST      = ST308

    Other examples:

        Klebsiella pneumoniae_ST39
        -> Klebsiella pneumoniae, ST39

        Klebsiella pneumoniae_STunknown_group_9
        -> Klebsiella pneumoniae, STunknown_group_9

    Missing values return:
        (NaN, NaN)
    """

    value = normalize_missing(
        value
    )

    if pd.isna(value):
        return np.nan, np.nan

    text = str(
        value
    ).strip()

    # "_ST" is the delimiter used in this dataset.
    marker = "_ST"

    position = text.rfind(
        marker
    )

    # If no ST exists, retain the species string only.
    if position == -1:

        species = (
            text
            .replace(
                "_",
                " ",
            )
            .strip()
        )

        return species, np.nan

    species = (
        text[
            :position
        ]
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    st_suffix = (
        text[
            position
            + len(marker):
        ]
        .strip()
    )

    if not st_suffix:

        return species, np.nan

    st = (
        "ST"
        + st_suffix
    )

    return species, st


def normalize_species(
    value,
):
    """
    Normalize species names for comparison.

    Original display values are preserved elsewhere.
    """

    value = normalize_missing(
        value
    )

    if pd.isna(value):
        return None

    text = (
        str(value)
        .replace(
            "_",
            " ",
        )
        .strip()
        .lower()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_st(
    value,
):
    """
    Normalize sequence types for comparison.
    """

    value = normalize_missing(
        value
    )

    if pd.isna(value):
        return None

    text = (
        str(value)
        .upper()
        .replace(
            " ",
            "",
        )
        .strip()
    )

    return text


def st_is_known(
    value,
) -> bool:
    """
    Return True only when the ST is meaningfully known.

    Examples:
        ST308              -> True
        ST39               -> True
        STunknown          -> False
        STunknown_group_9  -> False
    """

    normalized = normalize_st(
        value
    )

    if normalized is None:
        return False

    unknown_tokens = (
        "UNKNOWN",
        "NONE",
        "NAN",
        "NA",
        "ND",
    )

    return not any(
        token in normalized
        for token in unknown_tokens
    )


# ============================================================
# EXACT CP-GENE PARSING
# ============================================================


def normalize_specific_gene(
    value,
):
    """
    Normalize a specific genomic resistance-gene name.

    Examples:
        blaNDM-1   -> NDM-1
        NDM1       -> NDM-1
        blaOXA-181 -> OXA-181
        OXA48      -> OXA-48
    """

    if value is None:
        return None

    text = (
        str(value)
        .strip()
        .upper()
        .replace(
            " ",
            "",
        )
        .replace(
            "_",
            "-",
        )
    )

    if not text:
        return None

    if text.startswith(
        "BLA"
    ):
        text = text[3:]

    match = re.fullmatch(
        r"(NDM|VIM|IMP|KPC|OXA)-?(\d+)",
        text,
    )

    if match:

        return (
            f"{match.group(1)}-"
            f"{match.group(2)}"
        )

    return text


def extract_specific_cp_genes(
    value,
) -> set[str]:
    """
    Extract specific CP-gene alleles from a genomic field.

    Example:
        blaNDM-5,blaOXA-181
            ->
        {"NDM-5", "OXA-181"}
    """

    value = normalize_missing(
        value
    )

    if pd.isna(value):
        return set()

    raw_tokens = re.split(
        r"[,;/|+]+",
        str(value),
    )

    genes = set()

    for token in raw_tokens:

        normalized = (
            normalize_specific_gene(
                token
            )
        )

        if normalized:
            genes.add(
                normalized
            )

    return genes


def gene_set_to_string(
    values: set[str],
):
    """
    Convert gene set to stable string.
    """

    if not values:
        return np.nan

    return ";".join(
        sorted(values)
    )


# ============================================================
# SOURCE ISOLATE HELPERS
# ============================================================


def get_source_isolate_strings(
    row: pd.Series,
) -> list[str]:
    """
    Collect all available source species/ST strings.

    Patient source:
        source_species_st

    Environment source:
        source_species_st
        source_species_st_1
        source_species_st_2

    Duplicates are removed.
    """

    values = []

    for column in (
        "source_species_st",
        "source_species_st_1",
        "source_species_st_2",
    ):

        value = normalize_missing(
            row.get(
                column
            )
        )

        if pd.isna(value):
            continue

        value = str(
            value
        ).strip()

        if value not in values:
            values.append(
                value
            )

    return values


def get_source_isolate_pairs(
    row: pd.Series,
) -> list[tuple[object, object]]:
    """
    Return all unique source:
        (species, ST)

    pairs.
    """

    pairs = []

    for value in (
        get_source_isolate_strings(
            row
        )
    ):

        species, st = (
            parse_species_st(
                value
            )
        )

        pair = (
            species,
            st,
        )

        if pair not in pairs:
            pairs.append(
                pair
            )

    return pairs


def get_source_specific_genes(
    row: pd.Series,
) -> set[str]:
    """
    Collect specific CP genes from all source isolate fields.

    GeneXpert is intentionally NOT included here because
    GeneXpert gives sample-level evidence, while this feature
    concerns the cultured/WGS isolate itself.
    """

    genes = set()

    for column in (
        "source_cp_gene",
        "source_cp_gene_1",
        "source_cp_gene_2",
    ):

        genes |= (
            extract_specific_cp_genes(
                row.get(
                    column
                )
            )
        )

    return genes


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================


def build_pair_features() -> pd.DataFrame:
    """
    Convert candidate_pairs.csv into a biologically meaningful
    source-target feature table.

    IMPORTANT:
    No WGS clonal/plasmid linkage labels are added here.
    Those come in the NEXT stage.
    """

    candidate_file = (
        PROCESSED_DATA_DIR
        / "candidate_pairs.csv"
    )

    acquisition_file = (
        PROCESSED_DATA_DIR
        / "acquisition_cases.csv"
    )

    if not candidate_file.exists():

        raise FileNotFoundError(
            "\ncandidate_pairs.csv was not found.\n"
            "Run:\n\n"
            "PYTHONPATH=src "
            "python scripts/03_build_candidate_pairs.py"
        )

    if not acquisition_file.exists():

        raise FileNotFoundError(
            "\nacquisition_cases.csv was not found.\n"
            "Run:\n\n"
            "PYTHONPATH=src "
            "python scripts/02_build_acquisition_cases.py"
        )

    pairs = pd.read_csv(
        candidate_file,
        parse_dates=[
            "target_acquisition_date",
            "source_date",
        ],
    )

    acquisitions = pd.read_csv(
        acquisition_file,
        parse_dates=[
            "acquisition_date",
        ],
    )

    original_row_count = len(
        pairs
    )

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "pair_id",
        "target_id",
        "target_species_st",
        "target_cp_gene",
        "source_type",
        "source_subject_id",
        "source_sample_id",
        "source_species_st",
        "source_cp_gene",
        "days_before_acquisition",
        "temporal_relation",
        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_cp_family_any",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in pairs.columns
    ]

    if missing_columns:

        raise ValueError(
            "candidate_pairs.csv is missing "
            f"required columns:\n{missing_columns}"
        )

    # ========================================================
    # ATTACH TARGET SAMPLE INFORMATION
    # ========================================================
    #
    # This is critical for Stage 4 when we join against WGS.
    # ========================================================

    target_metadata = (
        acquisitions[
            [
                "SubjectID",
                "positive_sample_id",
                "sample_schedule",
                "wards",
                "has_culture_wgs",
            ]
        ]
        .rename(
            columns={
                "SubjectID":
                    "target_id",

                "positive_sample_id":
                    "target_sample_id",

                "sample_schedule":
                    "target_sample_schedule",

                "wards":
                    "target_recorded_wards",

                "has_culture_wgs":
                    "target_has_culture_wgs",
            }
        )
    )

    pairs = pairs.merge(
        target_metadata,
        on="target_id",
        how="left",
        validate="many_to_one",
    )

    if len(
        pairs
    ) != original_row_count:

        raise ValueError(
            "Row count changed when attaching "
            "target acquisition metadata."
        )

    # ========================================================
    # TARGET SPECIES / ST
    # ========================================================

    target_parsed = (
        pairs[
            "target_species_st"
        ]
        .apply(
            parse_species_st
        )
    )

    pairs[
        "target_species"
    ] = target_parsed.apply(
        lambda value:
            value[0]
    )

    pairs[
        "target_st"
    ] = target_parsed.apply(
        lambda value:
            value[1]
    )

    # ========================================================
    # PRIMARY SOURCE SPECIES / ST
    # ========================================================

    source_primary_parsed = (
        pairs[
            "source_species_st"
        ]
        .apply(
            parse_species_st
        )
    )

    pairs[
        "source_species_primary"
    ] = source_primary_parsed.apply(
        lambda value:
            value[0]
    )

    pairs[
        "source_st_primary"
    ] = source_primary_parsed.apply(
        lambda value:
            value[1]
    )

    # ========================================================
    # ENVIRONMENT SOURCE ISOLATE 1 / 2
    # ========================================================

    for slot in (
        1,
        2,
    ):

        column = (
            f"source_species_st_{slot}"
        )

        if column in pairs.columns:

            parsed = (
                pairs[
                    column
                ]
                .apply(
                    parse_species_st
                )
            )

            pairs[
                f"source_species_{slot}"
            ] = parsed.apply(
                lambda value:
                    value[0]
            )

            pairs[
                f"source_st_{slot}"
            ] = parsed.apply(
                lambda value:
                    value[1]
            )

        else:

            pairs[
                f"source_species_{slot}"
            ] = np.nan

            pairs[
                f"source_st_{slot}"
            ] = np.nan

    # ========================================================
    # SOURCE SPECIES/ST COLLECTIONS
    # ========================================================

    pairs[
        "_source_isolate_pairs"
    ] = pairs.apply(
        get_source_isolate_pairs,
        axis=1,
    )

    pairs[
        "source_species_values"
    ] = pairs[
        "_source_isolate_pairs"
    ].apply(
        lambda values:
            ";".join(
                sorted(
                    {
                        str(species)
                        for species, _
                        in values
                        if not pd.isna(
                            species
                        )
                    }
                )
            )
            if values
            else np.nan
    )

    pairs[
        "source_st_values"
    ] = pairs[
        "_source_isolate_pairs"
    ].apply(
        lambda values:
            ";".join(
                sorted(
                    {
                        str(st)
                        for _, st
                        in values
                        if not pd.isna(
                            st
                        )
                    }
                )
            )
            if values
            else np.nan
    )

    # ========================================================
    # SPECIES MATCH
    # ========================================================

    def same_species_primary(
        row: pd.Series,
    ) -> bool:

        target_species = (
            normalize_species(
                row.get(
                    "target_species"
                )
            )
        )

        source_species = (
            normalize_species(
                row.get(
                    "source_species_primary"
                )
            )
        )

        if (
            target_species is None
            or source_species is None
        ):
            return False

        return (
            target_species
            == source_species
        )

    pairs[
        "same_species_primary"
    ] = pairs.apply(
        same_species_primary,
        axis=1,
    )

    def same_species_any(
        row: pd.Series,
    ) -> bool:

        target_species = (
            normalize_species(
                row.get(
                    "target_species"
                )
            )
        )

        if target_species is None:
            return False

        source_pairs = row[
            "_source_isolate_pairs"
        ]

        for species, _ in source_pairs:

            if (
                normalize_species(
                    species
                )
                == target_species
            ):
                return True

        return False

    pairs[
        "same_species_any"
    ] = pairs.apply(
        same_species_any,
        axis=1,
    )

    # ========================================================
    # SEQUENCE-TYPE MATCH
    # ========================================================
    #
    # ST numbers are meaningful only within the same species.
    # Therefore an ST match requires:
    #
    #   same species
    #   AND
    #   both STs known
    #   AND
    #   exact ST equality.
    # ========================================================

    def same_st_primary(
        row: pd.Series,
    ) -> bool:

        target_species = (
            normalize_species(
                row.get(
                    "target_species"
                )
            )
        )

        source_species = (
            normalize_species(
                row.get(
                    "source_species_primary"
                )
            )
        )

        target_st = normalize_st(
            row.get(
                "target_st"
            )
        )

        source_st = normalize_st(
            row.get(
                "source_st_primary"
            )
        )

        if (
            target_species is None
            or source_species is None
        ):
            return False

        if (
            target_species
            != source_species
        ):
            return False

        if not st_is_known(
            target_st
        ):
            return False

        if not st_is_known(
            source_st
        ):
            return False

        return (
            target_st
            == source_st
        )

    pairs[
        "same_st_primary"
    ] = pairs.apply(
        same_st_primary,
        axis=1,
    )

    def same_st_any(
        row: pd.Series,
    ) -> bool:

        target_species = (
            normalize_species(
                row.get(
                    "target_species"
                )
            )
        )

        target_st = normalize_st(
            row.get(
                "target_st"
            )
        )

        if target_species is None:
            return False

        if not st_is_known(
            target_st
        ):
            return False

        for (
            source_species,
            source_st,
        ) in row[
            "_source_isolate_pairs"
        ]:

            if (
                normalize_species(
                    source_species
                )
                != target_species
            ):
                continue

            source_st_normalized = (
                normalize_st(
                    source_st
                )
            )

            if not st_is_known(
                source_st_normalized
            ):
                continue

            if (
                source_st_normalized
                == target_st
            ):
                return True

        return False

    pairs[
        "same_st_any"
    ] = pairs.apply(
        same_st_any,
        axis=1,
    )

    # ========================================================
    # EXACT ISOLATE CP-GENE MATCH
    # ========================================================

    pairs[
        "_target_specific_genes"
    ] = (
        pairs[
            "target_cp_gene"
        ]
        .apply(
            extract_specific_cp_genes
        )
    )

    pairs[
        "_source_specific_genes"
    ] = pairs.apply(
        get_source_specific_genes,
        axis=1,
    )

    pairs[
        "target_specific_cp_genes"
    ] = (
        pairs[
            "_target_specific_genes"
        ]
        .apply(
            gene_set_to_string
        )
    )

    pairs[
        "source_specific_cp_genes"
    ] = (
        pairs[
            "_source_specific_genes"
        ]
        .apply(
            gene_set_to_string
        )
    )

    pairs[
        "same_exact_cp_gene_isolate"
    ] = pairs.apply(
        lambda row: bool(
            row[
                "_target_specific_genes"
            ]
            &
            row[
                "_source_specific_genes"
            ]
        ),
        axis=1,
    )

    pairs[
        "matching_exact_cp_genes"
    ] = pairs.apply(
        lambda row:
            gene_set_to_string(
                row[
                    "_target_specific_genes"
                ]
                &
                row[
                    "_source_specific_genes"
                ]
            ),
        axis=1,
    )

    # ========================================================
    # SOURCE / TARGET ISOLATE AVAILABILITY
    # ========================================================

    pairs[
        "target_has_isolate"
    ] = (
        pairs[
            "target_species_st"
        ].notna()
        |
        pairs[
            "target_cp_gene"
        ].notna()
    )

    def source_has_isolate(
        row: pd.Series,
    ) -> bool:

        isolate_columns = (
            "source_species_st",
            "source_cp_gene",
            "source_species_st_1",
            "source_cp_gene_1",
            "source_species_st_2",
            "source_cp_gene_2",
        )

        return any(
            not pd.isna(
                normalize_missing(
                    row.get(
                        column
                    )
                )
            )
            for column
            in isolate_columns
        )

    pairs[
        "source_has_isolate"
    ] = pairs.apply(
        source_has_isolate,
        axis=1,
    )

    pairs[
        "has_pair_isolate_data"
    ] = (
        pairs[
            "target_has_isolate"
        ]
        &
        pairs[
            "source_has_isolate"
        ]
    )

    # ========================================================
    # SOURCE TYPE FEATURES
    # ========================================================

    pairs[
        "source_is_patient"
    ] = (
        pairs[
            "source_type"
        ]
        == "patient"
    )

    pairs[
        "source_is_environment"
    ] = (
        pairs[
            "source_type"
        ]
        == "environment"
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    pairs[
        "days_before_acquisition"
    ] = pd.to_numeric(
        pairs[
            "days_before_acquisition"
        ],
        errors="raise",
    )

    if (
        pairs[
            "days_before_acquisition"
        ]
        < 0
    ).any():

        raise ValueError(
            "Negative days_before_acquisition "
            "found in candidate feature table."
        )

    pairs[
        "is_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        > 0
    )

    pairs[
        "is_same_day"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        == 0
    )

    pairs[
        "within_7_days_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        .between(
            1,
            7,
        )
    )

    pairs[
        "within_14_days_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        .between(
            1,
            14,
        )
    )

    pairs[
        "within_30_days_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        .between(
            1,
            30,
        )
    )

    pairs[
        "within_90_days_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        .between(
            1,
            90,
        )
    )

    pairs[
        "within_180_days_prior"
    ] = (
        pairs[
            "days_before_acquisition"
        ]
        .between(
            1,
            180,
        )
    )

    pairs[
        "log1p_days_before_acquisition"
    ] = np.log1p(
        pairs[
            "days_before_acquisition"
        ]
    )

    # ========================================================
    # CP-GENE AVAILABILITY / COMPLEXITY FEATURES
    # ========================================================

    pairs[
        "source_gx_positive"
    ] = (
        pairs[
            "source_gx_result"
        ]
        .apply(
            lambda value:
                len(
                    extract_cp_families(
                        value
                    )
                )
                > 0
        )
    )

    pairs[
        "target_gx_family_count"
    ] = (
        pairs[
            "target_lab_cp_gene"
        ]
        .apply(
            lambda value:
                len(
                    extract_cp_families(
                        value
                    )
                )
        )
    )

    pairs[
        "source_gx_family_count"
    ] = (
        pairs[
            "source_gx_result"
        ]
        .apply(
            lambda value:
                len(
                    extract_cp_families(
                        value
                    )
                )
        )
    )

    # ========================================================
    # HOSPITAL-MATCH QUALITY
    # ========================================================

    pairs[
        "hospital_match_confirmed"
    ] = (
        pairs[
            "hospital_match_status"
        ]
        == "CONFIRMED"
    )

    pairs[
        "hospital_match_unconfirmed"
    ] = (
        pairs[
            "hospital_match_status"
        ]
        .astype(str)
        .str.contains(
            "UNCONFIRMED",
            na=False,
        )
    )

    # ========================================================
    # PRELIMINARY WGS ELIGIBILITY
    # ========================================================
    #
    # This does NOT mean WGS linkage exists.
    #
    # It only says both sides have isolate-level information
    # and the source is temporally prior.
    #
    # Actual WGS availability/linkage is attached in Stage 4.
    # ========================================================

    pairs[
        "preliminary_wgs_pair_eligible"
    ] = (
        pairs[
            "is_prior"
        ]
        &
        pairs[
            "has_pair_isolate_data"
        ]
    )

    # ========================================================
    # DUPLICATE BIOLOGICAL SIGNATURE DIAGNOSTIC
    # ========================================================
    #
    # We do NOT delete duplicates here.
    #
    # Two rows may represent genuinely separate isolate slots.
    # This merely flags them for inspection before WGS joining.
    # ========================================================

    duplicate_columns = [
        "target_id",
        "source_type",
        "source_subject_id",
        "source_sample_id",
        "source_species_st",
        "source_cp_gene",
    ]

    pairs[
        "duplicate_biology_signature"
    ] = pairs.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    # ========================================================
    # IMPORTANT: DO NOT CREATE A WARD-MATCH FEATURE YET
    # ========================================================
    #
    # target_recorded_wards currently comes from the ENV
    # workbook and should not yet be interpreted as a precise
    # patient movement/ward-overlap history.
    #
    # We therefore preserve target_recorded_wards for review,
    # but deliberately DO NOT construct same_ward.
    # ========================================================

    # ========================================================
    # CLEAN INTERNAL OBJECT COLUMNS
    # ========================================================

    pairs = pairs.drop(
        columns=[
            "_source_isolate_pairs",
            "_target_specific_genes",
            "_source_specific_genes",
        ]
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(
        pairs
    ) != original_row_count:

        raise ValueError(
            "Feature engineering changed the number "
            "of source-target candidate rows."
        )

    if not pairs[
        "pair_id"
    ].is_unique:

        raise ValueError(
            "pair_id is no longer unique."
        )

    expected_targets = {
        "T0016",
        "T0036",
        "T0052",
        "T0058",
        "T0080",
        "T0081",
        "T0124",
        "T0125",
    }

    observed_targets = set(
        pairs[
            "target_id"
        ].unique()
    )

    if (
        observed_targets
        != expected_targets
    ):

        raise ValueError(
            "Unexpected target-patient set.\n"
            f"Expected: {sorted(expected_targets)}\n"
            f"Observed: {sorted(observed_targets)}"
        )

    return pairs