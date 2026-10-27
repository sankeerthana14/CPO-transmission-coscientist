from __future__ import annotations

from collections import Counter


def compute_consensus(
    council_members: list[dict],
) -> dict:
    """
    Compute deterministic 3-member consensus.

    3/3 -> UNANIMOUS
    2/3 -> MAJORITY
    1/1/1 -> NO_CONSENSUS
    """

    votes = [
        member[
            "opinion"
        ][
            "mechanism"
        ]
        for member
        in council_members
    ]

    counts = Counter(
        votes
    )

    if not votes:

        return {
            "status":
                "NO_OPINIONS",

            "consensus_mechanism":
                None,

            "vote_count":
                0,

            "total_members":
                0,

            "votes":
                {},

            "member_votes":
                {},
        }

    (
        top_mechanism,
        top_count,
    ) = counts.most_common(
        1
    )[0]

    total = len(
        votes
    )

    if (
        top_count
        == total
    ):

        status = (
            "UNANIMOUS"
        )

        consensus_mechanism = (
            top_mechanism
        )

    elif (
        top_count
        >= 2
    ):

        status = (
            "MAJORITY"
        )

        consensus_mechanism = (
            top_mechanism
        )

    else:

        status = (
            "NO_CONSENSUS"
        )

        consensus_mechanism = (
            None
        )

    member_votes = {
        member[
            "model_name"
        ]:
            member[
                "opinion"
            ][
                "mechanism"
            ]

        for member
        in council_members
    }

    return {
        "status":
            status,

        "consensus_mechanism":
            consensus_mechanism,

        "vote_count":
            top_count,

        "total_members":
            total,

        "votes":
            dict(
                counts
            ),

        "member_votes":
            member_votes,
    }