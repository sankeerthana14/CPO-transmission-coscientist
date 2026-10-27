from __future__ import annotations

import json
import re


IDENTIFIER_PATTERN = re.compile(
    r"\bT\d{4}\b"
    r"|"
    r"\bHTS-[A-Za-z]\d+\b"
    r"|"
    r"\bHTS\d+\b",
    flags=re.IGNORECASE,
)


def _extract_identifiers(
    text: str,
) -> set[str]:
    """
    Extract patient/sample/WGS-like identifiers.
    """

    return {
        value.upper()
        for value
        in IDENTIFIER_PATTERN.findall(
            text
        )
    }


def _allowed_identifiers(
    evidence_packet: dict,
) -> set[str]:
    """
    Build identifier whitelist from evidence supplied to LLM.
    """

    evidence_text = json.dumps(
        evidence_packet,
        ensure_ascii=False,
    )

    return _extract_identifiers(
        evidence_text
    )


def validate_council(
    *,
    target_id: str,
    evidence_packet: dict,
    reference_mechanism: str,
    council_members: list[dict],
    consensus: dict,
) -> dict:
    """
    Validate council consensus against deterministic WGS
    reference and basic hallucination / overclaim constraints.
    """

    target_id = (
        target_id
        .strip()
        .upper()
    )

    fatal_issues = []

    warnings = []

    member_checks = []

    allowed_ids = (
        _allowed_identifiers(
            evidence_packet
        )
    )

    risky_phrases = [
        "high probability candidate",
        "most probable source",
        "confirmed donor",
        "confirmed source",
        "definitely transmitted",
        "proven transmission",
    ]

    for member in council_members:

        model_name = (
            member[
                "model_name"
            ]
        )

        role = member[
            "role"
        ]

        opinion = member[
            "opinion"
        ]

        member_warnings = []

        if (
            opinion[
                "target_id"
            ]
            .upper()
            != target_id
        ):

            member_warnings.append(
                "TARGET_ID_MISMATCH"
            )

        if opinion[
            "direct_transmission_proven"
        ]:

            member_warnings.append(
                "DIRECT_TRANSMISSION_OVERCLAIM"
            )

        opinion_text = (
            json.dumps(
                opinion,
                ensure_ascii=False,
            )
        )

        mentioned_ids = (
            _extract_identifiers(
                opinion_text
            )
        )

        unknown_ids = sorted(
            mentioned_ids
            -
            allowed_ids
        )

        if unknown_ids:

            member_warnings.append(
                "UNKNOWN_IDENTIFIERS: "
                + ", ".join(
                    unknown_ids
                )
            )

        lower_text = (
            opinion_text
            .lower()
        )

        detected_risky = [
            phrase
            for phrase
            in risky_phrases
            if phrase
            in lower_text
        ]

        if detected_risky:

            member_warnings.append(
                "RISKY_CAUSAL_LANGUAGE: "
                + ", ".join(
                    detected_risky
                )
            )

        matches_reference = (
            opinion[
                "mechanism"
            ]
            ==
            reference_mechanism
        )

        member_checks.append(
            {
                "model":
                    model_name,

                "role":
                    role,

                "mechanism":
                    opinion[
                        "mechanism"
                    ],

                "matches_wgs_reference":
                    matches_reference,

                "warnings":
                    member_warnings,
            }
        )

        warnings.extend(
            [
                (
                    f"{model_name}: "
                    f"{warning}"
                )
                for warning
                in member_warnings
            ]
        )

    consensus_mechanism = (
        consensus.get(
            "consensus_mechanism"
        )
    )

    if (
        consensus_mechanism
        is None
    ):

        fatal_issues.append(
            "NO_COUNCIL_CONSENSUS"
        )

    elif (
        consensus_mechanism
        != reference_mechanism
    ):

        fatal_issues.append(
            (
                "CONSENSUS_CONFLICTS_WITH_"
                "DETERMINISTIC_WGS_REFERENCE: "
                f"council={consensus_mechanism}; "
                f"reference={reference_mechanism}"
            )
        )

    if fatal_issues:

        validation_status = (
            "FLAGGED_FOR_EXPERT_REVIEW"
        )

        validated_mechanism = (
            None
        )

    elif warnings:

        validation_status = (
            "VALIDATED_CONSENSUS_WITH_WARNINGS"
        )

        validated_mechanism = (
            consensus_mechanism
        )

    else:

        validation_status = (
            "VALIDATED_CONSENSUS"
        )

        validated_mechanism = (
            consensus_mechanism
        )

    return {
        "status":
            validation_status,

        "validated_mechanism":
            validated_mechanism,

        "reference_mechanism":
            reference_mechanism,

        "consensus_matches_reference":
            (
                consensus_mechanism
                ==
                reference_mechanism
            ),

        "fatal_issues":
            fatal_issues,

        "warnings":
            warnings,

        "member_checks":
            member_checks,
    }