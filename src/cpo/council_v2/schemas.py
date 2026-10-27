from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Mechanism = Literal[
    "PATIENT_ASSOCIATED_CLONAL",
    "ENVIRONMENT_ASSOCIATED_CLONAL",
    "PATIENT_ASSOCIATED_PLASMID",
    "ENVIRONMENT_ASSOCIATED_PLASMID",
    "MIXED",
    "NO_DIRECT_WGS_LINK",
    "UNRESOLVED_NO_TARGET_WGS",
]


Confidence = Literal[
    "high",
    "moderate",
    "low",
    "unresolved",
]


class CouncilVote(BaseModel):
    """
    Compact structured vote returned independently by each
    council member.
    """

    target_id: str

    mechanism: Mechanism

    confidence: Confidence

    rationale: str = Field(
        description=(
            "Maximum two concise sentences explaining why "
            "this mechanism is best supported."
        ),
        max_length=800,
    )

    key_evidence: list[str] = Field(
        description=(
            "Up to three decisive observations from the "
            "provided evidence packet."
        ),
        min_length=1,
        max_length=3,
    )

    direct_transmission_proven: bool = Field(
        description=(
            "Normally false. WGS relatedness does not by "
            "itself prove direct donor-to-recipient transfer."
        )
    )