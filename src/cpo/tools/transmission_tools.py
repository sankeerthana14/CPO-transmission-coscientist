from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpo.config import (
    PROCESSED_DATA_DIR,
)


# ============================================================
# INPUT FILES
# ============================================================
#
# These are deterministic outputs already produced by the
# earlier stages of the pipeline.
#
# The LLM never reads the raw clinical Excel files directly.
# It accesses evidence only through the controlled functions
# defined in this module.
# ============================================================

ACQUISITION_FILE = (
    PROCESSED_DATA_DIR
    / "acquisition_cases.csv"
)

WGS_FILE = (
    PROCESSED_DATA_DIR
    / "wgs_labeled_pairs.csv"
)

OOF_PREDICTIONS_FILE = (
    PROCESSED_DATA_DIR
    / "xgboost_oof_predictions.csv"
)

OOF_SHAP_FILE = (
    PROCESSED_DATA_DIR
    / "xgboost_oof_shap.csv"
)


# ============================================================
# BASIC HELPERS
# ============================================================


def _require_file(
    filepath: Path,
):
    """
    Raise a clear error if a required pipeline output is
    missing.
    """

    if not filepath.exists():

        raise FileNotFoundError(
            f"Required pipeline file was not found:\n"
            f"{filepath}"
        )


def _json_safe(
    value,
):
    """
    Convert pandas/numpy values to ordinary JSON-safe Python
    values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):

        value = value.item()

    try:

        if pd.isna(
            value
        ):

            return None

    except (
        TypeError,
        ValueError,
    ):

        pass

    return value


def _records_to_json(
    dataframe: pd.DataFrame,
) -> str:
    """
    Convert dataframe records into human-readable JSON for
    the LLM.
    """

    records = []

    for record in dataframe.to_dict(
        orient="records"
    ):

        clean_record = {
            key:
                _json_safe(
                    value
                )

            for key, value
            in record.items()
        }

        records.append(
            clean_record
        )

    return json.dumps(
        records,
        indent=2,
        ensure_ascii=False,
    )


def _coerce_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    Robustly convert CSV-loaded boolean fields.

    Supports:
        True / False
        "True" / "False"
        1 / 0
        yes / no
    """

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.astype(
            bool
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


# ============================================================
# DATA LOADERS
# ============================================================


def _load_acquisitions() -> pd.DataFrame:
    """
    Load the eight acquisition cases.
    """

    _require_file(
        ACQUISITION_FILE
    )

    return pd.read_csv(
        ACQUISITION_FILE,
        parse_dates=[
            "acquisition_date",
        ],
    )


def _load_wgs_pairs() -> pd.DataFrame:
    """
    Load candidate pairs with attached WGS linkage evidence.
    """

    _require_file(
        WGS_FILE
    )

    return pd.read_csv(
        WGS_FILE,
        parse_dates=[
            "target_acquisition_date",
            "source_date",
        ],
    )


def _load_oof_predictions() -> pd.DataFrame:
    """
    Load leave-one-target-out XGBoost predictions.
    """

    _require_file(
        OOF_PREDICTIONS_FILE
    )

    return pd.read_csv(
        OOF_PREDICTIONS_FILE
    )


def _load_oof_shap() -> pd.DataFrame:
    """
    Load out-of-fold SHAP explanations.
    """

    _require_file(
        OOF_SHAP_FILE
    )

    return pd.read_csv(
        OOF_SHAP_FILE
    )


# ============================================================
# TOOL 1 — ACQUISITION CASE
# ============================================================


def get_acquisition_case(
    target_id: str,
) -> str:
    """
    Retrieve the acquisition-event information for one target
    patient.

    Use this tool first when investigating a patient.

    Args:
        target_id:
            Acquisition patient identifier such as T0058,
            T0081, or T0124.

    Returns:
        JSON containing the target's acquisition date,
        carbapenemase gene, bacterial isolate information,
        hospital, wards and positive sample ID.
    """

    acquisitions = (
        _load_acquisitions()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    subset = acquisitions[
        acquisitions[
            "SubjectID"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        target_id
    ].copy()

    if subset.empty:

        return json.dumps(
            {
                "error":
                    "Target acquisition patient not found.",

                "target_id":
                    target_id,
            },
            indent=2,
        )

    preferred_columns = [
        "SubjectID",

        "acquisition_date",

        "lab_cp_gene",

        "genomics_species_st",
        "genomics_cp_gene",

        "has_culture_wgs",

        "hospital",
        "wards",

        "positive_sample_id",
        "sample_schedule",

        "remarks",
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in subset.columns
    ]

    return _records_to_json(
        subset[
            columns
        ]
    )


# ============================================================
# TOOL 2 — XGBOOST RANKING
# ============================================================


def get_ranked_candidate_sources(
    target_id: str,
    top_k: int = 10,
) -> str:
    """
    Retrieve the augmented XGBoost source ranking for one
    acquisition patient.

    The returned XGBoost value is a candidate-ranking score.
    It MUST NOT be interpreted as a calibrated probability of
    transmission.

    The rankings are out-of-fold:
    the target patient was excluded from training when its
    candidates were ranked.

    Args:
        target_id:
            Acquisition patient identifier.

        top_k:
            Number of highest-ranked candidate sources to
            return. Defaults to 10.

    Returns:
        JSON containing ranked source candidates and their
        microbiological compatibility features.
    """

    predictions = (
        _load_oof_predictions()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    try:

        top_k = int(
            top_k
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = 10

    top_k = max(
        1,
        min(
            top_k,
            25,
        ),
    )

    subset = predictions[
        (
            predictions[
                "model_name"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            "AUGMENTED"
        )
        &
        (
            predictions[
                "target_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            target_id
        )
    ].copy()

    if subset.empty:

        return json.dumps(
            {
                "target_id":
                    target_id,

                "message":
                    (
                        "No out-of-fold augmented XGBoost "
                        "ranking is available for this target."
                    ),
            },
            indent=2,
        )

    subset = (
        subset
        .sort_values(
            "rank_within_target"
        )
        .head(
            top_k
        )
    )

    preferred_columns = [
        "target_id",

        "rank_within_target",
        "predicted_probability",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",

        "same_cp_family_gx",
        "same_cp_family_isolate",

        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        # These are included as reference labels because this
        # tool is used for retrospective investigation.
        # The agent must still treat WGS evidence as higher
        # priority than the XGBoost ranking.
        "label_any_wgs_link",
        "label_clonal_link",
        "label_plasmid_link",
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in subset.columns
    ]

    return _records_to_json(
        subset[
            columns
        ]
    )


# ============================================================
# TOOL 3 — DIRECT WGS LINKS
# ============================================================


def get_direct_wgs_links(
    target_id: str,
) -> str:
    """
    Retrieve all PRIOR candidate sources with direct clonal or
    plasmid WGS linkage to an acquisition target.

    WGS linkage takes evidential priority over XGBoost ranking.

    A clonal link supports circulation of a closely related
    bacterial lineage.

    A plasmid link supports circulation of related resistance
    plasmid DNA and may occur between different bacterial
    species or sequence types.

    Neither relationship alone proves direct donor-to-recipient
    transmission.

    Args:
        target_id:
            Acquisition patient identifier.

    Returns:
        JSON containing all prior directly WGS-linked sources.
        If no link exists, the response indicates whether the
        target lacks WGS or simply has no prior direct link.
    """

    data = (
        _load_wgs_pairs()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    target_mask = (
        data[
            "target_id"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        target_id
    )

    target_rows = data[
        target_mask
    ].copy()

    if target_rows.empty:

        return json.dumps(
            {
                "target_id":
                    target_id,

                "message":
                    "Target is absent from WGS pair data.",
            },
            indent=2,
        )

    linked = (
        _coerce_boolean_series(
            target_rows[
                "any_wgs_link"
            ]
        )
    )

    days = pd.to_numeric(
        target_rows[
            "days_before_acquisition"
        ],
        errors="coerce",
    )

    subset = target_rows[
        linked
        &
        (
            days > 0
        )
    ].copy()

    # --------------------------------------------------------
    # No direct link found
    # --------------------------------------------------------

    if subset.empty:

        wgs_count = 0

        if (
            "target_wgs_isolate_count"
            in target_rows.columns
        ):

            wgs_count = (
                pd.to_numeric(
                    target_rows[
                        "target_wgs_isolate_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .max()
            )

        if (
            wgs_count
            == 0
        ):

            message = (
                "The target acquisition patient has no "
                "usable target WGS isolate. Genomic source "
                "attribution is unresolved."
            )

            status = (
                "UNRESOLVED_NO_TARGET_WGS"
            )

        else:

            message = (
                "The target has WGS data, but none of the "
                "prior candidate sources has a direct listed "
                "clonal or plasmid WGS relationship."
            )

            status = (
                "NO_DIRECT_WGS_LINK"
            )

        return json.dumps(
            {
                "target_id":
                    target_id,

                "direct_wgs_link_count":
                    0,

                "status":
                    status,

                "message":
                    message,
            },
            indent=2,
        )

    # --------------------------------------------------------
    # Direct links exist
    # --------------------------------------------------------

    preferred_columns = [
        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "source_date",
        "days_before_acquisition",
        "temporal_relation",

        "target_species",
        "target_st",
        "target_cp_gene",

        "source_species_primary",
        "source_st_primary",

        "source_cp_gene",
        "source_gx_result",

        "same_cp_family_gx",
        "same_cp_family_isolate",

        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        "clonal_link",
        "plasmid_link",

        "shared_plasmid_cp_genes",

        "wgs_direction_status",

        "target_wgs_ids",
        "source_wgs_ids",

        "clonal_evidence_pairs",
        "plasmid_evidence_pairs",

        "hospital_match_status",
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in subset.columns
    ]

    # Clonal links first, then plasmid, then temporally closest
    if (
        "clonal_link"
        in subset.columns
    ):

        subset[
            "_clonal_sort"
        ] = _coerce_boolean_series(
            subset[
                "clonal_link"
            ]
        )

    else:

        subset[
            "_clonal_sort"
        ] = False

    if (
        "plasmid_link"
        in subset.columns
    ):

        subset[
            "_plasmid_sort"
        ] = _coerce_boolean_series(
            subset[
                "plasmid_link"
            ]
        )

    else:

        subset[
            "_plasmid_sort"
        ] = False

    subset = subset.sort_values(
        [
            "_clonal_sort",
            "_plasmid_sort",
            "days_before_acquisition",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    return _records_to_json(
        subset[
            columns
        ]
    )


# ============================================================
# TOOL 4 — DETAILED CANDIDATE EVIDENCE
# ============================================================


def get_candidate_evidence(
    target_id: str,
    source_sample_id: str,
) -> str:
    """
    Retrieve detailed evidence for one particular candidate
    source-target relationship.

    This tool is useful when the Investigator or Critic wants
    to examine a WGS-linked candidate or a highly ranked
    XGBoost candidate in detail.

    Args:
        target_id:
            Acquisition patient identifier.

        source_sample_id:
            Candidate source sample ID such as HTS-E186,
            HTS-P315 or HTS-P331.

    Returns:
        JSON containing chronology, microbiological features,
        WGS linkage and location-confidence information.
    """

    data = (
        _load_wgs_pairs()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    source_sample_id = (
        str(
            source_sample_id
        )
        .strip()
        .upper()
    )

    subset = data[
        (
            data[
                "target_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            target_id
        )
        &
        (
            data[
                "source_sample_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            source_sample_id
        )
    ].copy()

    if subset.empty:

        return json.dumps(
            {
                "target_id":
                    target_id,

                "source_sample_id":
                    source_sample_id,

                "message":
                    "Candidate source-target pair not found.",
            },
            indent=2,
        )

    preferred_columns = [
        # TARGET
        "target_id",
        "target_sample_id",
        "target_acquisition_date",

        "target_hospital",

        "target_species",
        "target_st",
        "target_cp_gene",

        "target_lab_cp_gene",

        # SOURCE
        "source_type",
        "source_subject_id",
        "source_sample_id",
        "source_date",

        "source_hospital",
        "source_ward",

        "days_before_acquisition",
        "temporal_relation",

        "source_gx_result",

        "source_species_primary",
        "source_st_primary",
        "source_cp_gene",

        # COMPATIBILITY
        "same_cp_family_gx",
        "same_cp_family_isolate",

        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        # WGS
        "wgs_pair_evaluable",

        "clonal_link",
        "plasmid_link",

        "shared_plasmid_cp_genes",

        "target_wgs_ids",
        "source_wgs_ids",

        "clonal_evidence_pairs",
        "plasmid_evidence_pairs",

        "wgs_direction_status",

        # DATA QUALITY
        "hospital_match_status",
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in subset.columns
    ]

    return _records_to_json(
        subset[
            columns
        ]
    )


# ============================================================
# TOOL 5 — SHAP / MODEL EXPLANATION
# ============================================================


def get_model_explanation(
    target_id: str,
    source_sample_id: str,
) -> str:
    """
    Retrieve the out-of-fold SHAP explanation for one ranked
    candidate source.

    SHAP explains WHY XGBoost assigned its ranking score.

    SHAP does NOT establish:
        transmission,
        biological causality,
        clonal linkage,
        plasmid linkage.

    Args:
        target_id:
            Acquisition patient identifier.

        source_sample_id:
            Candidate source sample identifier.

    Returns:
        JSON containing ranking score, target-specific rank,
        strongest positive/negative model drivers and SHAP
        contributions.
    """

    data = (
        _load_oof_shap()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    source_sample_id = (
        str(
            source_sample_id
        )
        .strip()
        .upper()
    )

    subset = data[
        (
            data[
                "target_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            target_id
        )
        &
        (
            data[
                "source_sample_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            source_sample_id
        )
    ].copy()

    if subset.empty:

        return json.dumps(
            {
                "target_id":
                    target_id,

                "source_sample_id":
                    source_sample_id,

                "message":
                    (
                        "No out-of-fold SHAP explanation "
                        "is available for this candidate."
                    ),
            },
            indent=2,
        )

    shap_columns = [
        column
        for column
        in subset.columns
        if column.startswith(
            "shap__"
        )
    ]

    preferred_columns = [
        "target_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "predicted_probability",
        "rank_within_target",

        "label_any_wgs_link",

        "strongest_positive_feature",
        "strongest_negative_feature",

        *shap_columns,
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in subset.columns
    ]

    return _records_to_json(
        subset[
            columns
        ]
    )


# ============================================================
# TOOL 6 — TARGET EVIDENCE OVERVIEW
# ============================================================


def get_target_evidence_summary(
    target_id: str,
) -> str:
    """
    Retrieve a compact deterministic overview of one target's
    candidate-source evidence.

    This is intended as a convenience tool for the local agent.

    Args:
        target_id:
            Acquisition patient identifier.

    Returns:
        JSON with candidate counts, WGS-link counts and the
        best augmented-XGBoost candidate.
    """

    wgs = (
        _load_wgs_pairs()
    )

    predictions = (
        _load_oof_predictions()
    )

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    target_wgs = wgs[
        wgs[
            "target_id"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        target_id
    ].copy()

    if target_wgs.empty:

        return json.dumps(
            {
                "target_id":
                    target_id,

                "message":
                    "Target not found.",
            },
            indent=2,
        )

    days = pd.to_numeric(
        target_wgs[
            "days_before_acquisition"
        ],
        errors="coerce",
    )

    prior = target_wgs[
        days > 0
    ].copy()

    any_link = (
        _coerce_boolean_series(
            prior[
                "any_wgs_link"
            ]
        )
        if (
            "any_wgs_link"
            in prior.columns
        )
        else pd.Series(
            False,
            index=prior.index,
        )
    )

    clonal = (
        _coerce_boolean_series(
            prior[
                "clonal_link"
            ]
        )
        if (
            "clonal_link"
            in prior.columns
        )
        else pd.Series(
            False,
            index=prior.index,
        )
    )

    plasmid = (
        _coerce_boolean_series(
            prior[
                "plasmid_link"
            ]
        )
        if (
            "plasmid_link"
            in prior.columns
        )
        else pd.Series(
            False,
            index=prior.index,
        )
    )

    target_predictions = predictions[
        (
            predictions[
                "target_id"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
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

    if not target_predictions.empty:

        best = (
            target_predictions
            .sort_values(
                "rank_within_target"
            )
            .iloc[0]
        )

        best_candidate = {
            "source_type":
                _json_safe(
                    best.get(
                        "source_type"
                    )
                ),

            "source_subject_id":
                _json_safe(
                    best.get(
                        "source_subject_id"
                    )
                ),

            "source_sample_id":
                _json_safe(
                    best.get(
                        "source_sample_id"
                    )
                ),

            "ranking_score":
                _json_safe(
                    best.get(
                        "predicted_probability"
                    )
                ),

            "rank":
                _json_safe(
                    best.get(
                        "rank_within_target"
                    )
                ),
        }

    else:

        best_candidate = None

    result = {
        "target_id":
            target_id,

        "total_candidate_pairs":
            int(
                len(
                    target_wgs
                )
            ),

        "prior_candidate_pairs":
            int(
                len(
                    prior
                )
            ),

        "prior_direct_wgs_links":
            int(
                any_link.sum()
            ),

        "prior_clonal_links":
            int(
                clonal.sum()
            ),

        "prior_plasmid_links":
            int(
                plasmid.sum()
            ),

        "best_xgboost_candidate":
            best_candidate,

        "important_note":
            (
                "The XGBoost value is a ranking score, "
                "not a calibrated probability of transmission."
            ),
    }

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# TOOL REGISTRY
# ============================================================
#
# local_ollama.py passes these Python functions directly to
# Ollama.
#
# The function name, type annotations, arguments and docstring
# are used to construct the tool definitions seen by Qwen.
# ============================================================

AVAILABLE_TOOLS = {
    "get_acquisition_case":
        get_acquisition_case,

    "get_target_evidence_summary":
        get_target_evidence_summary,

    "get_ranked_candidate_sources":
        get_ranked_candidate_sources,

    "get_direct_wgs_links":
        get_direct_wgs_links,

    "get_candidate_evidence":
        get_candidate_evidence,

    "get_model_explanation":
        get_model_explanation,
}