from __future__ import annotations

import json
import math
from typing import Any

from cpo.tools.transmission_tools import (
    get_acquisition_case,
    get_direct_wgs_links,
    get_model_explanation,
    get_ranked_candidate_sources,
)


# ============================================================
# BASIC HELPERS
# ============================================================


def _parse_json(
    value: str,
) -> Any:
    """
    Parse deterministic tool JSON.
    """

    try:
        return json.loads(value)

    except json.JSONDecodeError as error:
        raise ValueError(
            "A deterministic transmission tool returned "
            "invalid JSON."
        ) from error


def _truthy(
    value,
) -> bool:
    """
    Convert common representations to boolean.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

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


def _safe_number(
    value,
    digits: int = 6,
):
    """
    Convert numeric values to compact JSON-safe values.
    """

    if value is None:
        return None

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return value

    if math.isnan(number):
        return None

    return round(
        number,
        digits,
    )


def _source_class(
    source_type: str,
) -> str | None:
    """
    Convert source type to PATIENT / ENVIRONMENT.
    """

    value = (
        str(
            source_type
            or ""
        )
        .strip()
        .lower()
    )

    if "environment" in value:
        return "ENVIRONMENT"

    if (
        "patient" in value
        or "known" in value
    ):
        return "PATIENT"

    return None


# ============================================================
# DETERMINISTIC WGS REFERENCE
# ============================================================


def derive_reference_mechanism(
    direct_wgs,
) -> str:
    """
    Derive WGS-grounded mechanism.

    IMPORTANT:
    This classification is NOT shown to the council.
    It is retained only for evaluation/validation.
    """

    if isinstance(
        direct_wgs,
        dict,
    ):

        status = direct_wgs.get(
            "status"
        )

        if (
            status
            == "UNRESOLVED_NO_TARGET_WGS"
        ):
            return (
                "UNRESOLVED_NO_TARGET_WGS"
            )

        return (
            "NO_DIRECT_WGS_LINK"
        )

    if not isinstance(
        direct_wgs,
        list,
    ):

        return (
            "NO_DIRECT_WGS_LINK"
        )

    mechanisms = set()

    for row in direct_wgs:

        source_class = (
            _source_class(
                row.get(
                    "source_type"
                )
            )
        )

        if source_class is None:
            continue

        if _truthy(
            row.get(
                "clonal_link"
            )
        ):

            mechanisms.add(
                f"{source_class}"
                "_ASSOCIATED_CLONAL"
            )

        if _truthy(
            row.get(
                "plasmid_link"
            )
        ):

            mechanisms.add(
                f"{source_class}"
                "_ASSOCIATED_PLASMID"
            )

    if len(
        mechanisms
    ) == 0:

        return (
            "NO_DIRECT_WGS_LINK"
        )

    if len(
        mechanisms
    ) == 1:

        return next(
            iter(
                mechanisms
            )
        )

    return "MIXED"


# ============================================================
# ACQUISITION SUMMARY
# ============================================================


def _build_acquisition_summary(
    acquisition,
) -> dict:
    """
    Keep only clinically useful target information.
    """

    if not isinstance(
        acquisition,
        list,
    ) or not acquisition:

        return {}

    row = acquisition[0]

    wanted = [
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
    ]

    return {
        key:
            row.get(
                key
            )

        for key
        in wanted
        if key
        in row
    }


# ============================================================
# DECISIVE WGS SUMMARY
# ============================================================


def _summarize_wgs(
    direct_wgs,
) -> tuple[dict, list[dict]]:
    """
    Build a compact, explicit WGS summary.

    This section is intentionally placed before XGBoost and
    SHAP so small LLMs cannot miss the decisive evidence.
    """

    summary = {
        "target_wgs_available":
            True,

        "total_direct_wgs_links":
            0,

        "prior_patient_clonal_links":
            0,

        "prior_patient_plasmid_links":
            0,

        "prior_environment_clonal_links":
            0,

        "prior_environment_plasmid_links":
            0,
    }

    if isinstance(
        direct_wgs,
        dict,
    ):

        status = direct_wgs.get(
            "status"
        )

        if (
            status
            == "UNRESOLVED_NO_TARGET_WGS"
        ):

            summary[
                "target_wgs_available"
            ] = False

        summary[
            "wgs_status"
        ] = status

        return (
            summary,
            [],
        )

    if not isinstance(
        direct_wgs,
        list,
    ):

        summary[
            "wgs_status"
        ] = "UNKNOWN"

        return (
            summary,
            [],
        )

    summary[
        "wgs_status"
    ] = (
        "DIRECT_WGS_LINKS_PRESENT"
    )

    compact_links = []

    seen = set()

    for row in direct_wgs:

        source_sample_id = (
            str(
                row.get(
                    "source_sample_id",
                    ""
                )
            )
            .strip()
        )

        clonal = _truthy(
            row.get(
                "clonal_link"
            )
        )

        plasmid = _truthy(
            row.get(
                "plasmid_link"
            )
        )

        dedup_key = (
            source_sample_id,
            clonal,
            plasmid,
        )

        if dedup_key in seen:
            continue

        seen.add(
            dedup_key
        )

        source_class = (
            _source_class(
                row.get(
                    "source_type"
                )
            )
        )

        if (
            source_class
            == "PATIENT"
            and clonal
        ):

            summary[
                "prior_patient_clonal_links"
            ] += 1

        if (
            source_class
            == "PATIENT"
            and plasmid
        ):

            summary[
                "prior_patient_plasmid_links"
            ] += 1

        if (
            source_class
            == "ENVIRONMENT"
            and clonal
        ):

            summary[
                "prior_environment_clonal_links"
            ] += 1

        if (
            source_class
            == "ENVIRONMENT"
            and plasmid
        ):

            summary[
                "prior_environment_plasmid_links"
            ] += 1

        compact_links.append(
            {
                "source_type":
                    row.get(
                        "source_type"
                    ),

                "source_subject_id":
                    row.get(
                        "source_subject_id"
                    ),

                "source_sample_id":
                    row.get(
                        "source_sample_id"
                    ),

                "days_before_acquisition":
                    _safe_number(
                        row.get(
                            "days_before_acquisition"
                        ),
                        digits=1,
                    ),

                "source_species":
                    row.get(
                        "source_species_primary"
                    ),

                "source_st":
                    row.get(
                        "source_st_primary"
                    ),

                "source_cp_gene":
                    row.get(
                        "source_cp_gene"
                    ),

                "same_exact_cp_gene":
                    _truthy(
                        row.get(
                            "same_exact_cp_gene_isolate"
                        )
                    ),

                "same_species":
                    _truthy(
                        row.get(
                            "same_species_any"
                        )
                    ),

                "same_st":
                    _truthy(
                        row.get(
                            "same_st_any"
                        )
                    ),

                "clonal_link":
                    clonal,

                "plasmid_link":
                    plasmid,

                "shared_plasmid_cp_genes":
                    row.get(
                        "shared_plasmid_cp_genes"
                    ),

                "hospital_match_status":
                    row.get(
                        "hospital_match_status"
                    ),
            }
        )

    summary[
        "total_direct_wgs_links"
    ] = len(
        compact_links
    )

    return (
        summary,
        compact_links,
    )


# ============================================================
# XGBOOST SUMMARY
# ============================================================


def _unique_top_candidates(
    ranking_raw,
    top_k: int = 5,
) -> list[dict]:
    """
    Keep only top unique source samples.

    No retrospective WGS labels are included.
    """

    if not isinstance(
        ranking_raw,
        list,
    ):

        return []

    sorted_rows = sorted(
        ranking_raw,
        key=lambda row: (
            _safe_number(
                row.get(
                    "rank_within_target"
                )
            )
            or 10**9
        ),
    )

    candidates = []

    seen_samples = set()

    for row in sorted_rows:

        sample_id = (
            str(
                row.get(
                    "source_sample_id",
                    ""
                )
            )
            .strip()
        )

        if not sample_id:
            continue

        if sample_id in seen_samples:
            continue

        seen_samples.add(
            sample_id
        )

        score = row.get(
            "predicted_probability"
        )

        candidates.append(
            {
                "rank":
                    int(
                        row.get(
                            "rank_within_target"
                        )
                    ),

                "source_type":
                    row.get(
                        "source_type"
                    ),

                "source_subject_id":
                    row.get(
                        "source_subject_id"
                    ),

                "source_sample_id":
                    sample_id,

                "days_before_acquisition":
                    _safe_number(
                        row.get(
                            "days_before_acquisition"
                        ),
                        digits=1,
                    ),

                "ranking_score":
                    _safe_number(
                        score
                    ),

                "same_cp_family_gx":
                    _truthy(
                        row.get(
                            "same_cp_family_gx"
                        )
                    ),

                "same_cp_family_isolate":
                    _truthy(
                        row.get(
                            "same_cp_family_isolate"
                        )
                    ),

                "same_exact_cp_gene":
                    _truthy(
                        row.get(
                            "same_exact_cp_gene_isolate"
                        )
                    ),

                "same_species":
                    _truthy(
                        row.get(
                            "same_species_any"
                        )
                    ),

                "same_st":
                    _truthy(
                        row.get(
                            "same_st_any"
                        )
                    ),
            }
        )

        if (
            len(
                candidates
            )
            >= top_k
        ):
            break

    return candidates


# ============================================================
# SHAP SUMMARY
# ============================================================


def _compact_shap_for_candidate(
    target_id: str,
    candidate: dict,
) -> dict | None:
    """
    Retrieve only a very small SHAP summary.

    IMPORTANT:
    - predicted_probability is NOT exposed
    - label_any_wgs_link is NOT exposed
    - raw full SHAP rows are NOT exposed
    """

    sample_id = candidate.get(
        "source_sample_id"
    )

    if not sample_id:
        return None

    try:

        result = _parse_json(
            get_model_explanation(
                target_id,
                sample_id,
            )
        )

    except Exception:
        return None

    if not isinstance(
        result,
        list,
    ) or not result:

        return None

    # Choose the highest-ranked row for this sample.
    rows = sorted(
        result,
        key=lambda row: (
            _safe_number(
                row.get(
                    "rank_within_target"
                )
            )
            or 10**9
        ),
    )

    row = rows[0]

    contributions = []

    for key, value in row.items():

        if not key.startswith(
            "shap__"
        ):
            continue

        numeric_value = (
            _safe_number(
                value
            )
        )

        if not isinstance(
            numeric_value,
            (
                int,
                float,
            ),
        ):
            continue

        contributions.append(
            {
                "feature":
                    key.replace(
                        "shap__",
                        "",
                    ),

                "shap_value":
                    numeric_value,
            }
        )

    contributions = sorted(
        contributions,
        key=lambda item: abs(
            item[
                "shap_value"
            ]
        ),
        reverse=True,
    )[:3]

    return {
        "source_sample_id":
            sample_id,

        "rank":
            candidate.get(
                "rank"
            ),

        "ranking_score":
            candidate.get(
                "ranking_score"
            ),

        "strongest_positive_feature":
            row.get(
                "strongest_positive_feature"
            ),

        "strongest_negative_feature":
            row.get(
                "strongest_negative_feature"
            ),

        "top_shap_contributions":
            contributions,
    }


# ============================================================
# FINAL V2 EVIDENCE PACKET
# ============================================================


def build_evidence_packet_v2(
    target_id: str,
) -> tuple[dict, str]:
    """
    Build the compact Council-v2 evidence packet.

    Returns
    -------
    evidence_packet:
        Shown to all three council members.

    reference_mechanism:
        Deterministic WGS reference retained privately for
        validation. NOT shown to council members.
    """

    target_id = (
        str(
            target_id
        )
        .strip()
        .upper()
    )

    acquisition_raw = (
        _parse_json(
            get_acquisition_case(
                target_id
            )
        )
    )

    direct_wgs_raw = (
        _parse_json(
            get_direct_wgs_links(
                target_id
            )
        )
    )

    ranking_raw = (
        _parse_json(
            get_ranked_candidate_sources(
                target_id,
                10,
            )
        )
    )

    reference_mechanism = (
        derive_reference_mechanism(
            direct_wgs_raw
        )
    )

    (
        decisive_wgs_summary,
        linked_sources,
    ) = _summarize_wgs(
        direct_wgs_raw
    )

    top_candidates = (
        _unique_top_candidates(
            ranking_raw,
            top_k=5,
        )
    )

    shap_summaries = []

    for candidate in top_candidates:

        summary = (
            _compact_shap_for_candidate(
                target_id,
                candidate,
            )
        )

        if summary is not None:

            shap_summaries.append(
                summary
            )

    packet = {
        "target_id":
            target_id,

        "acquisition":
            _build_acquisition_summary(
                acquisition_raw
            ),

        # THIS IS INTENTIONALLY FIRST AND DECISIVE.
        "decisive_wgs_summary":
            decisive_wgs_summary,

        "direct_wgs_linked_sources":
            linked_sources,

        "xgboost_candidate_prioritisation": {
            "important_note":
                (
                    "ranking_score is only a relative "
                    "candidate-prioritisation score. It is "
                    "NOT a calibrated probability of "
                    "transmission."
                ),

            "top_5_unique_candidates":
                top_candidates,
        },

        "shap_summary": {
            "important_note":
                (
                    "SHAP explains why XGBoost prioritised a "
                    "candidate. SHAP does NOT establish "
                    "clonal linkage, plasmid linkage, "
                    "transmission, or causality."
                ),

            "candidates":
                shap_summaries,
        },

        "scientific_guardrails": [
            (
                "Only direct_wgs_linked_sources.clonal_link "
                "may establish direct clonal WGS relatedness."
            ),
            (
                "Only direct_wgs_linked_sources.plasmid_link "
                "may establish direct plasmid WGS relatedness."
            ),
            (
                "Exact CP-gene, species, ST, XGBoost rank, "
                "or SHAP values alone must never be called "
                "clonal or plasmid WGS linkage."
            ),
            (
                "Direct genomic relatedness supports "
                "circulation but does not prove direct "
                "donor-to-recipient transmission."
            ),
        ],
    }

    return (
        packet,
        reference_mechanism,
    )