from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from cpo.config import ENV_FILE, PROCESSED_DATA_DIR
from cpo.data_loader import read_sheet, coerce_date, normalize_missing


# ============================================================
# TEMPORARY ASSUMPTION
# ============================================================
# ENV_raw contains an explicit Hospital Code.
#
# Known_CPO_pos appears not to contain an explicit hospital
# column. Until the clinicians confirm how site/hospital is
# encoded, we temporarily use the alphabetic Subject ID prefix:
#
#     T0054 -> T
#     T0081 -> T
#
# This is NOT treated as confirmed same-hospital evidence.
# ============================================================

USE_SUBJECT_PREFIX_FOR_KNOWN_CPO = True


# ============================================================
# GENERAL HELPERS
# ============================================================

def has_value(value) -> bool:
    """Return True when a workbook value is meaningfully populated."""

    value = normalize_missing(value)

    return not pd.isna(value)


def get_subject_prefix(subject_id) -> Optional[str]:
    """
    Extract the alphabetic prefix from a subject ID.

    Examples
    --------
    T0054 -> T
    N0012 -> N
    """

    subject_id = normalize_missing(subject_id)

    if pd.isna(subject_id):
        return None

    subject_id = str(subject_id).strip()

    if not subject_id:
        return None

    match = re.match(
        r"([A-Za-z]+)",
        subject_id,
    )

    if match:
        return match.group(1)

    return subject_id[0]


def resolve_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> Optional[str]:
    """
    Return the first matching dataframe column from a list of
    possible aliases.

    Matching is case-insensitive and ignores leading/trailing
    whitespace.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    if required:

        raise ValueError(
            "Could not find required column.\n"
            f"Tried: {candidates}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


def clean_string(value):
    """
    Normalize an Excel value and return a stripped string.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return np.nan

    return str(value).strip()


# ============================================================
# CP-GENE HELPERS
# ============================================================

def extract_cp_families(value) -> set[str]:
    """
    Extract every recognised carbapenemase family appearing
    in a value.

    Examples
    --------
    "blaNDM-1"
        -> {"NDM"}

    "blaOXA-181"
        -> {"OXA-48"}

    "blaNDM-5,blaOXA-181"
        -> {"NDM", "OXA-48"}

    "IMP, VIM, NDM"
        -> {"IMP", "VIM", "NDM"}

    Notes
    -----
    OXA-181 is treated as part of the OXA-48-like family.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return set()

    text = (
        str(value)
        .upper()
        .replace(" ", "")
    )

    families: set[str] = set()

    if "NDM" in text:
        families.add("NDM")

    if "VIM" in text:
        families.add("VIM")

    if "IMP" in text:
        families.add("IMP")

    if "KPC" in text:
        families.add("KPC")

    if (
        "OXA-48" in text
        or "OXA48" in text
        or "OXA-181" in text
        or "OXA181" in text
    ):
        families.add("OXA-48")

    return families


def families_to_string(
    families: set[str],
):
    """
    Convert a CP-family set to a stable,
    human-readable string for CSV/Excel output.
    """

    if not families:
        return np.nan

    return ";".join(
        sorted(families)
    )


# ============================================================
# ENVIRONMENT POSITIVITY
# ============================================================

def gene_xpert_is_positive(
    value,
) -> bool:
    """
    Return True when GeneXpert text contains a recognised
    carbapenemase family.
    """

    value = normalize_missing(value)

    if pd.isna(value):
        return False

    text = (
        str(value)
        .strip()
        .lower()
    )

    negative_values = {
        "",
        "nd",
        "not detected",
        "negative",
        "nil",
        "none",
        "pending",
        "na",
        "n/a",
    }

    if text in negative_values:
        return False

    recognised = (
        "ndm",
        "vim",
        "imp",
        "oxa",
        "kpc",
    )

    return any(
        token in text
        for token in recognised
    )


# ============================================================
# PREPARE POSITIVE ENVIRONMENTAL SOURCES
# ============================================================

def prepare_environment_sources() -> pd.DataFrame:
    """
    Load ENV_raw and return positive environmental samples
    that can later be considered candidate sources.
    """

    env = read_sheet(
        filepath=ENV_FILE,
        sheet_name="ENV_raw",
        required_column="SubjectID",
    )

    # --------------------------------------------------------
    # Resolve important columns
    # --------------------------------------------------------

    subject_col = resolve_column(
        env,
        [
            "SubjectID",
            "Subject ID",
        ],
        required=True,
    )

    sample_col = resolve_column(
        env,
        [
            "SampleID_ENV",
            "ENV Sample ID",
            "Environmental Sample ID",
            "Sample ID ENV",
        ],
        required=True,
    )

    date_col = resolve_column(
        env,
        [
            "ENV_Date",
            "ENV Date",
            "Environmental Date",
            "Date of environmental sample",
        ],
        required=True,
    )

    hospital_col = resolve_column(
        env,
        [
            "Hospital Code",
            "Hospital",
        ],
        required=True,
    )

    ward_col = resolve_column(
        env,
        [
            "Ward",
        ],
        required=False,
    )

    bed_col = resolve_column(
        env,
        [
            "Bed/Bed range",
            "Bed",
            "Bed range",
        ],
        required=False,
    )

    room_col = resolve_column(
        env,
        [
            "Room code",
            "Room Code",
        ],
        required=False,
    )

    site_col = resolve_column(
        env,
        [
            "Site code",
            "Site Code",
        ],
        required=False,
    )

    sink_col = resolve_column(
        env,
        [
            "Sink/Shower drain Unique code",
            "Sink/Shower Drain Unique Code",
            "Sink/Shower drain code",
        ],
        required=False,
    )

    gx_col = resolve_column(
        env,
        [
            "ENVLabGXG",
            "ENV Lab GXG",
            "ENV GeneXpert",
            "Environmental GeneXpert",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Culture / genomic fields
    # --------------------------------------------------------

    g1_species_col = resolve_column(
        env,
        [
            "G1.Species",
            "G1 Species",
        ],
        required=False,
    )

    g1_gene_col = resolve_column(
        env,
        [
            "G1.Genotype",
            "G1 Genotype",
            "G1 CP gene",
        ],
        required=False,
    )

    g2_species_col = resolve_column(
        env,
        [
            "G2.Species",
            "G2 Species",
        ],
        required=False,
    )

    g2_gene_col = resolve_column(
        env,
        [
            "G2.Genotype",
            "G2 Genotype",
            "G2 CP gene",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Standardise into our own schema
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=env.index
    )

    result["source_subject_id"] = (
        env[subject_col]
        .apply(clean_string)
    )

    result["source_sample_id"] = (
        env[sample_col]
        .apply(clean_string)
    )

    result["source_date"] = (
        env[date_col]
        .apply(coerce_date)
    )

    result["source_hospital"] = (
        env[hospital_col]
        .apply(clean_string)
    )

    result["source_ward"] = (
        env[ward_col]
        .apply(clean_string)
        if ward_col
        else np.nan
    )

    result["source_bed_range"] = (
        env[bed_col]
        .apply(clean_string)
        if bed_col
        else np.nan
    )

    result["source_room_code"] = (
        env[room_col]
        .apply(clean_string)
        if room_col
        else np.nan
    )

    result["source_site_code"] = (
        env[site_col]
        .apply(clean_string)
        if site_col
        else np.nan
    )

    result["source_sink_drain_code"] = (
        env[sink_col]
        .apply(clean_string)
        if sink_col
        else np.nan
    )

    result["source_gx_result"] = (
        env[gx_col]
        .apply(clean_string)
        if gx_col
        else np.nan
    )

    result["source_species_st_1"] = (
        env[g1_species_col]
        .apply(clean_string)
        if g1_species_col
        else np.nan
    )

    result["source_cp_gene_1"] = (
        env[g1_gene_col]
        .apply(clean_string)
        if g1_gene_col
        else np.nan
    )

    result["source_species_st_2"] = (
        env[g2_species_col]
        .apply(clean_string)
        if g2_species_col
        else np.nan
    )

    result["source_cp_gene_2"] = (
        env[g2_gene_col]
        .apply(clean_string)
        if g2_gene_col
        else np.nan
    )

    # --------------------------------------------------------
    # Determine positivity
    # --------------------------------------------------------

    result["gx_positive"] = (
        result[
            "source_gx_result"
        ]
        .apply(
            gene_xpert_is_positive
        )
    )

    result["culture_positive"] = (
        result[
            [
                "source_species_st_1",
                "source_cp_gene_1",
                "source_species_st_2",
                "source_cp_gene_2",
            ]
        ]
        .notna()
        .any(
            axis=1
        )
    )

    result["environment_positive"] = (
        result["gx_positive"]
        |
        result["culture_positive"]
    )

    # --------------------------------------------------------
    # Keep positive environments with valid dates
    # --------------------------------------------------------

    result = result[
        result["environment_positive"]
        &
        result["source_date"].notna()
    ].copy()

    result["source_type"] = (
        "environment"
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# PREPARE KNOWN CPO PATIENT SOURCES
# ============================================================

def prepare_known_cpo_sources() -> pd.DataFrame:
    """
    Convert Known_CPO_pos into one row per cultured/genomic
    isolate.

    GeneXpert-only positive samples are retained even when
    no cultured isolate is available.
    """

    known = read_sheet(
        filepath=ENV_FILE,
        sheet_name="Known_CPO_pos",
        required_column="Subject ID",
    )

    subject_col = resolve_column(
        known,
        [
            "Subject ID",
            "SubjectID",
        ],
        required=True,
    )

    sample_col = resolve_column(
        known,
        [
            "Sample ID",
            "SampleID",
        ],
        required=True,
    )

    date_col = resolve_column(
        known,
        [
            "Date of sample",
            "Sample Date",
        ],
        required=True,
    )

    gx_col = resolve_column(
        known,
        [
            "Sample GX Result",
            "GeneXpert result",
            "GeneXpert Result",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Resolve possible isolate columns once
    # --------------------------------------------------------

    isolate_columns = []

    for slot in range(
        1,
        11,
    ):

        species_col = resolve_column(
            known,
            [
                f"{slot}.Genomic Species/ST",
                f"{slot}.Genomics Species/ST",
                f"{slot}.Species/ST",
            ],
            required=False,
        )

        gene_col = resolve_column(
            known,
            [
                f"{slot}.Genomic CP gene",
                f"{slot}.Genomics CP gene",
                f"{slot}.CP gene",
            ],
            required=False,
        )

        isolate_columns.append(
            (
                slot,
                species_col,
                gene_col,
            )
        )

    # --------------------------------------------------------
    # Expand samples into isolate rows
    # --------------------------------------------------------

    rows = []

    for _, row in known.iterrows():

        subject_id = clean_string(
            row[
                subject_col
            ]
        )

        sample_id = clean_string(
            row[
                sample_col
            ]
        )

        source_date = coerce_date(
            row[
                date_col
            ]
        )

        gx_result = (
            clean_string(
                row[
                    gx_col
                ]
            )
            if gx_col
            else np.nan
        )

        isolate_found = False

        for (
            slot,
            species_col,
            gene_col,
        ) in isolate_columns:

            if (
                species_col is None
                and gene_col is None
            ):
                continue

            species_st = (
                clean_string(
                    row[
                        species_col
                    ]
                )
                if species_col
                else np.nan
            )

            cp_gene = (
                clean_string(
                    row[
                        gene_col
                    ]
                )
                if gene_col
                else np.nan
            )

            if (
                pd.isna(
                    species_st
                )
                and
                pd.isna(
                    cp_gene
                )
            ):
                continue

            isolate_found = True

            rows.append(
                {
                    "source_type":
                        "patient",

                    "source_subject_id":
                        subject_id,

                    "source_sample_id":
                        sample_id,

                    "source_date":
                        source_date,

                    "source_gx_result":
                        gx_result,

                    "source_species_st":
                        species_st,

                    "source_cp_gene":
                        cp_gene,

                    "source_isolate_slot":
                        slot,

                    "source_subject_prefix":
                        get_subject_prefix(
                            subject_id
                        ),
                }
            )

        # ----------------------------------------------------
        # Preserve GeneXpert-only positive patient source
        # ----------------------------------------------------

        if (
            not isolate_found
            and
            gene_xpert_is_positive(
                gx_result
            )
        ):

            rows.append(
                {
                    "source_type":
                        "patient",

                    "source_subject_id":
                        subject_id,

                    "source_sample_id":
                        sample_id,

                    "source_date":
                        source_date,

                    "source_gx_result":
                        gx_result,

                    "source_species_st":
                        np.nan,

                    "source_cp_gene":
                        np.nan,

                    "source_isolate_slot":
                        np.nan,

                    "source_subject_prefix":
                        get_subject_prefix(
                            subject_id
                        ),
                }
            )

    sources = pd.DataFrame(
        rows
    )

    if sources.empty:

        raise ValueError(
            "No usable Known_CPO_pos "
            "source records were extracted."
        )

    sources = sources[
        sources[
            "source_date"
        ].notna()
    ].copy()

    return sources.reset_index(
        drop=True
    )


# ============================================================
# BUILD KNOWN_CPO_POS -> TARGET PAIRS
# ============================================================

def build_patient_candidate_pairs(
    cases: pd.DataFrame,
    known_sources: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate candidate Known_CPO_pos patient ->
    acquisition-patient pairs.
    """

    pairs = []

    for _, target in cases.iterrows():

        target_id = str(
            target[
                "SubjectID"
            ]
        ).strip()

        target_date = pd.to_datetime(
            target[
                "acquisition_date"
            ]
        )

        target_hospital = clean_string(
            target.get(
                "hospital",
                np.nan,
            )
        )

        target_prefix = (
            get_subject_prefix(
                target_id
            )
        )

        # ----------------------------------------------------
        # Source must exist on or before acquisition
        # ----------------------------------------------------

        eligible = known_sources[
            known_sources[
                "source_date"
            ]
            <= target_date
        ].copy()

        # Target cannot be its own source
        eligible = eligible[
            eligible[
                "source_subject_id"
            ]
            != target_id
        ].copy()

        # ----------------------------------------------------
        # Temporary site-prefix restriction
        # ----------------------------------------------------

        if (
            USE_SUBJECT_PREFIX_FOR_KNOWN_CPO
        ):

            eligible = eligible[
                eligible[
                    "source_subject_prefix"
                ]
                == target_prefix
            ].copy()

        # ----------------------------------------------------
        # Construct candidate pairs
        # ----------------------------------------------------

        for _, source in eligible.iterrows():

            days_before = int(
                (
                    target_date
                    -
                    source[
                        "source_date"
                    ]
                ).days
            )

            pairs.append(
                {
                    # TARGET
                    "target_id":
                        target_id,

                    "target_acquisition_date":
                        target_date,

                    "target_hospital":
                        target_hospital,

                    "target_lab_cp_gene":
                        target.get(
                            "lab_cp_gene",
                            np.nan,
                        ),

                    "target_species_st":
                        target.get(
                            "genomics_species_st",
                            np.nan,
                        ),

                    "target_cp_gene":
                        target.get(
                            "genomics_cp_gene",
                            np.nan,
                        ),

                    # SOURCE
                    "source_type":
                        "patient",

                    "source_subject_id":
                        source[
                            "source_subject_id"
                        ],

                    "source_sample_id":
                        source[
                            "source_sample_id"
                        ],

                    "source_date":
                        source[
                            "source_date"
                        ],

                    "source_hospital":
                        np.nan,

                    "source_ward":
                        np.nan,

                    "source_gx_result":
                        source[
                            "source_gx_result"
                        ],

                    "source_species_st":
                        source[
                            "source_species_st"
                        ],

                    "source_cp_gene":
                        source[
                            "source_cp_gene"
                        ],

                    "source_isolate_slot":
                        source.get(
                            "source_isolate_slot",
                            np.nan,
                        ),

                    # TIME
                    "days_before_acquisition":
                        days_before,

                    # Hospital is not explicitly known
                    "hospital_match_status":
                        "UNCONFIRMED_"
                        "SUBJECT_PREFIX_ONLY",
                }
            )

    return pd.DataFrame(
        pairs
    )


# ============================================================
# BUILD ENVIRONMENT -> TARGET PAIRS
# ============================================================

def build_environment_candidate_pairs(
    cases: pd.DataFrame,
    env_sources: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate positive-environment -> acquisition-patient
    candidate pairs.

    Environmental candidates are restricted using explicit
    hospital codes.
    """

    pairs = []

    for _, target in cases.iterrows():

        target_id = str(
            target[
                "SubjectID"
            ]
        ).strip()

        target_date = pd.to_datetime(
            target[
                "acquisition_date"
            ]
        )

        target_hospital = clean_string(
            target.get(
                "hospital",
                np.nan,
            )
        )

        # ----------------------------------------------------
        # Source must occur on/before acquisition date
        # ----------------------------------------------------

        eligible = env_sources[
            env_sources[
                "source_date"
            ]
            <= target_date
        ].copy()

        # ----------------------------------------------------
        # Explicit same-hospital restriction
        # ----------------------------------------------------

        if not pd.isna(
            target_hospital
        ):

            eligible = eligible[
                eligible[
                    "source_hospital"
                ]
                .astype(str)
                .str.strip()
                ==
                str(
                    target_hospital
                ).strip()
            ].copy()

        # ----------------------------------------------------
        # Construct candidate pairs
        # ----------------------------------------------------

        for _, source in eligible.iterrows():

            days_before = int(
                (
                    target_date
                    -
                    source[
                        "source_date"
                    ]
                ).days
            )

            # ------------------------------------------------
            # Primary cultured organism for display only.
            #
            # Both environmental isolate slots are retained.
            # ------------------------------------------------

            primary_species = (
                source[
                    "source_species_st_1"
                ]
                if has_value(
                    source[
                        "source_species_st_1"
                    ]
                )
                else source[
                    "source_species_st_2"
                ]
            )

            primary_gene = (
                source[
                    "source_cp_gene_1"
                ]
                if has_value(
                    source[
                        "source_cp_gene_1"
                    ]
                )
                else source[
                    "source_cp_gene_2"
                ]
            )

            pairs.append(
                {
                    # TARGET
                    "target_id":
                        target_id,

                    "target_acquisition_date":
                        target_date,

                    "target_hospital":
                        target_hospital,

                    "target_lab_cp_gene":
                        target.get(
                            "lab_cp_gene",
                            np.nan,
                        ),

                    "target_species_st":
                        target.get(
                            "genomics_species_st",
                            np.nan,
                        ),

                    "target_cp_gene":
                        target.get(
                            "genomics_cp_gene",
                            np.nan,
                        ),

                    # SOURCE
                    "source_type":
                        "environment",

                    "source_subject_id":
                        source[
                            "source_subject_id"
                        ],

                    "source_sample_id":
                        source[
                            "source_sample_id"
                        ],

                    "source_date":
                        source[
                            "source_date"
                        ],

                    "source_hospital":
                        source[
                            "source_hospital"
                        ],

                    "source_ward":
                        source[
                            "source_ward"
                        ],

                    "source_bed_range":
                        source[
                            "source_bed_range"
                        ],

                    "source_room_code":
                        source[
                            "source_room_code"
                        ],

                    "source_site_code":
                        source[
                            "source_site_code"
                        ],

                    "source_sink_drain_code":
                        source[
                            "source_sink_drain_code"
                        ],

                    "source_gx_result":
                        source[
                            "source_gx_result"
                        ],

                    "source_species_st":
                        primary_species,

                    "source_cp_gene":
                        primary_gene,

                    # Preserve all environmental isolates
                    "source_species_st_1":
                        source[
                            "source_species_st_1"
                        ],

                    "source_cp_gene_1":
                        source[
                            "source_cp_gene_1"
                        ],

                    "source_species_st_2":
                        source[
                            "source_species_st_2"
                        ],

                    "source_cp_gene_2":
                        source[
                            "source_cp_gene_2"
                        ],

                    # TIME
                    "days_before_acquisition":
                        days_before,

                    # Explicit hospital match
                    "hospital_match_status":
                        "CONFIRMED",
                }
            )

    return pd.DataFrame(
        pairs
    )


# ============================================================
# MASTER FUNCTION
# ============================================================

def build_candidate_pairs() -> pd.DataFrame:
    """
    Build all eligible candidate source -> target relationships
    for the eight acquisition patients.
    """

    acquisition_file = (
        PROCESSED_DATA_DIR
        / "acquisition_cases.csv"
    )

    if not acquisition_file.exists():

        raise FileNotFoundError(
            "\nacquisition_cases.csv was not found.\n"
            "Run this first:\n\n"
            "PYTHONPATH=src "
            "python scripts/02_build_acquisition_cases.py"
        )

    cases = pd.read_csv(
        acquisition_file,
        parse_dates=[
            "acquisition_date",
        ],
    )

    # --------------------------------------------------------
    # Load source pools
    # --------------------------------------------------------

    env_sources = (
        prepare_environment_sources()
    )

    known_sources = (
        prepare_known_cpo_sources()
    )

    # --------------------------------------------------------
    # Generate Known_CPO_pos source pairs
    # --------------------------------------------------------

    patient_pairs = (
        build_patient_candidate_pairs(
            cases=cases,
            known_sources=known_sources,
        )
    )

    # --------------------------------------------------------
    # Generate environmental source pairs
    # --------------------------------------------------------

    environment_pairs = (
        build_environment_candidate_pairs(
            cases=cases,
            env_sources=env_sources,
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_pairs = pd.concat(
        [
            patient_pairs,
            environment_pairs,
        ],
        ignore_index=True,
        sort=False,
    )

    if all_pairs.empty:

        raise ValueError(
            "No candidate source-target pairs "
            "were generated."
        )

    # --------------------------------------------------------
    # Unique pair identifier
    # --------------------------------------------------------

    all_pairs.insert(
        0,
        "pair_id",
        [
            f"PAIR_{index:06d}"
            for index in range(
                1,
                len(all_pairs) + 1,
            )
        ],
    )

    # ========================================================
    # ROBUST CP-FAMILY MATCHING
    # ========================================================
    #
    # Sample-level GeneXpert evidence is deliberately kept
    # separate from cultured/WGS-isolate evidence.
    #
    # Example:
    #
    # Source GeneXpert:
    #     NDM + OXA48
    #
    # Specific source isolate:
    #     E. coli + blaOXA-181
    #
    # Target:
    #     NDM
    #
    # Results:
    #
    #     same_cp_family_gx       = True
    #     same_cp_family_isolate  = False
    #     same_cp_family_any      = True
    # ========================================================

    # --------------------------------------------------------
    # Target GeneXpert families
    # --------------------------------------------------------

    all_pairs[
        "_target_gx_families"
    ] = (
        all_pairs[
            "target_lab_cp_gene"
        ]
        .apply(
            extract_cp_families
        )
    )

    # --------------------------------------------------------
    # Target cultured/WGS-isolate families
    # --------------------------------------------------------

    all_pairs[
        "_target_isolate_families"
    ] = (
        all_pairs[
            "target_cp_gene"
        ]
        .apply(
            extract_cp_families
        )
    )

    # --------------------------------------------------------
    # Source GeneXpert families
    # --------------------------------------------------------

    all_pairs[
        "_source_gx_families"
    ] = (
        all_pairs[
            "source_gx_result"
        ]
        .apply(
            extract_cp_families
        )
    )

    # --------------------------------------------------------
    # Source cultured/WGS-isolate families
    # --------------------------------------------------------

    def get_source_isolate_families(
        row: pd.Series,
    ) -> set[str]:

        families = set()

        families |= (
            extract_cp_families(
                row.get(
                    "source_cp_gene"
                )
            )
        )

        families |= (
            extract_cp_families(
                row.get(
                    "source_cp_gene_1"
                )
            )
        )

        families |= (
            extract_cp_families(
                row.get(
                    "source_cp_gene_2"
                )
            )
        )

        return families

    all_pairs[
        "_source_isolate_families"
    ] = all_pairs.apply(
        get_source_isolate_families,
        axis=1,
    )

    # ========================================================
    # HUMAN-READABLE FAMILY COLUMNS
    # ========================================================

    all_pairs[
        "target_gx_cp_families"
    ] = (
        all_pairs[
            "_target_gx_families"
        ]
        .apply(
            families_to_string
        )
    )

    all_pairs[
        "target_isolate_cp_families"
    ] = (
        all_pairs[
            "_target_isolate_families"
        ]
        .apply(
            families_to_string
        )
    )

    all_pairs[
        "source_gx_cp_families"
    ] = (
        all_pairs[
            "_source_gx_families"
        ]
        .apply(
            families_to_string
        )
    )

    all_pairs[
        "source_isolate_cp_families"
    ] = (
        all_pairs[
            "_source_isolate_families"
        ]
        .apply(
            families_to_string
        )
    )

    # ========================================================
    # GENEXPERT-LEVEL MATCH
    # ========================================================

    all_pairs[
        "same_cp_family_gx"
    ] = all_pairs.apply(
        lambda row: bool(
            row[
                "_target_gx_families"
            ]
            &
            row[
                "_source_gx_families"
            ]
        ),
        axis=1,
    )

    all_pairs[
        "matching_cp_families_gx"
    ] = all_pairs.apply(
        lambda row: families_to_string(
            row[
                "_target_gx_families"
            ]
            &
            row[
                "_source_gx_families"
            ]
        ),
        axis=1,
    )

    # ========================================================
    # ISOLATE/WGS-LEVEL MATCH
    # ========================================================

    all_pairs[
        "same_cp_family_isolate"
    ] = all_pairs.apply(
        lambda row: bool(
            row[
                "_target_isolate_families"
            ]
            &
            row[
                "_source_isolate_families"
            ]
        ),
        axis=1,
    )

    all_pairs[
        "matching_cp_families_isolate"
    ] = all_pairs.apply(
        lambda row: families_to_string(
            row[
                "_target_isolate_families"
            ]
            &
            row[
                "_source_isolate_families"
            ]
        ),
        axis=1,
    )

    # ========================================================
    # ANY CP-FAMILY EVIDENCE
    # ========================================================

    all_pairs[
        "_target_all_families"
    ] = all_pairs.apply(
        lambda row: (
            row[
                "_target_gx_families"
            ]
            |
            row[
                "_target_isolate_families"
            ]
        ),
        axis=1,
    )

    all_pairs[
        "_source_all_families"
    ] = all_pairs.apply(
        lambda row: (
            row[
                "_source_gx_families"
            ]
            |
            row[
                "_source_isolate_families"
            ]
        ),
        axis=1,
    )

    all_pairs[
        "same_cp_family_any"
    ] = all_pairs.apply(
        lambda row: bool(
            row[
                "_target_all_families"
            ]
            &
            row[
                "_source_all_families"
            ]
        ),
        axis=1,
    )

    all_pairs[
        "matching_cp_families_any"
    ] = all_pairs.apply(
        lambda row: families_to_string(
            row[
                "_target_all_families"
            ]
            &
            row[
                "_source_all_families"
            ]
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Backwards-compatible alias
    # --------------------------------------------------------

    all_pairs[
        "same_cp_family"
    ] = (
        all_pairs[
            "same_cp_family_any"
        ]
    )

    # --------------------------------------------------------
    # Remove internal Python-set columns
    # --------------------------------------------------------

    all_pairs = all_pairs.drop(
        columns=[
            "_target_gx_families",
            "_target_isolate_families",
            "_source_gx_families",
            "_source_isolate_families",
            "_target_all_families",
            "_source_all_families",
        ]
    )

    # ========================================================
    # TEMPORAL RELATION
    # ========================================================
    #
    # A sample obtained on the same calendar day as acquisition
    # is retained, but direction is considered ambiguous because
    # we do not have exact timestamps.
    # ========================================================

    all_pairs[
        "temporal_relation"
    ] = np.select(
        [
            all_pairs[
                "days_before_acquisition"
            ] > 0,

            all_pairs[
                "days_before_acquisition"
            ] == 0,
        ],
        [
            "PRIOR",
            "SAME_DAY_AMBIGUOUS",
        ],
        default="AFTER_ACQUISITION",
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if (
        all_pairs[
            "days_before_acquisition"
        ] < 0
    ).any():

        raise ValueError(
            "Found candidate sources sampled AFTER "
            "target acquisition. "
            "Check temporal filtering."
        )

    # ========================================================
    # SORT FOR INSPECTION
    # ========================================================

    all_pairs = (
        all_pairs
        .sort_values(
            [
                "target_id",
                "source_type",
                "days_before_acquisition",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return all_pairs