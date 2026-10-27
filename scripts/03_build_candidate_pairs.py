from cpo.candidate_sources import build_candidate_pairs

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)


def main():

    print(
        "\nBuilding candidate source -> target pairs..."
    )

    pairs = build_candidate_pairs()

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "CANDIDATE SOURCE -> TARGET PAIRS"
    )

    print(
        "=" * 110
    )

    print(
        f"\nTotal candidate pairs: "
        f"{len(pairs)}"
    )

    print(
        "\nSource-type counts:"
    )

    print(
        pairs[
            "source_type"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # COUNTS PER TARGET
    # ========================================================

    summary = (
        pairs
        .groupby(
            [
                "target_id",
                "source_type",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    print(
        "\n"
        + "-" * 110
    )

    print(
        "CANDIDATES PER TARGET"
    )

    print(
        "-" * 110
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # CP-FAMILY MATCH SUMMARY
    # ========================================================

    cp_summary = (
        pairs
        .groupby(
            [
                "target_id",
                "source_type",
            ]
        )
        .agg(
            candidate_count=(
                "pair_id",
                "size",
            ),

            gx_match_count=(
                "same_cp_family_gx",
                "sum",
            ),

            isolate_match_count=(
                "same_cp_family_isolate",
                "sum",
            ),

            any_cp_match_count=(
                "same_cp_family_any",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        "\n"
        + "-" * 110
    )

    print(
        "CP-FAMILY MATCHES"
    )

    print(
        "-" * 110
    )

    print(
        cp_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # TEMPORAL SUMMARY
    # ========================================================

    temporal_summary = (
        pairs
        .groupby(
            [
                "target_id",
                "temporal_relation",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    print(
        "\n"
        + "-" * 110
    )

    print(
        "TEMPORAL RELATION SUMMARY"
    )

    print(
        "-" * 110
    )

    print(
        temporal_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # COLUMNS USED FOR TERMINAL INSPECTION
    # ========================================================

    display_columns = [
        "source_type",
        "source_subject_id",
        "source_sample_id",
        "source_date",

        "days_before_acquisition",
        "temporal_relation",

        "source_gx_result",
        "source_species_st",
        "source_cp_gene",

        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_cp_family_any",

        "matching_cp_families_gx",
        "matching_cp_families_isolate",
        "matching_cp_families_any",

        "hospital_match_status",
    ]

    # ========================================================
    # SHOW TEMPORALLY CLOSEST CANDIDATES
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "10 TEMPORALLY CLOSEST CANDIDATES FOR EACH TARGET"
    )

    print(
        "=" * 110
    )

    for target_id in sorted(
        pairs[
            "target_id"
        ].unique()
    ):

        print(
            f"\n{'=' * 25} "
            f"{target_id} "
            f"{'=' * 25}"
        )

        target_pairs = (
            pairs[
                pairs[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                [
                    "days_before_acquisition",
                    "source_type",
                    "source_sample_id",
                ]
            )
            .head(10)
        )

        existing_columns = [
            column
            for column
            in display_columns
            if column
            in target_pairs.columns
        ]

        print(
            target_pairs[
                existing_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SHOW ANY CP-FAMILY MATCH
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "10 CLOSEST CANDIDATES WITH ANY CP-FAMILY MATCH"
    )

    print(
        "=" * 110
    )

    matching_any = pairs[
        pairs[
            "same_cp_family_any"
        ]
    ].copy()

    for target_id in sorted(
        pairs[
            "target_id"
        ].unique()
    ):

        subset = (
            matching_any[
                matching_any[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                [
                    "days_before_acquisition",
                    "source_type",
                    "source_sample_id",
                ]
            )
            .head(10)
        )

        print(
            f"\n--- {target_id} ---"
        )

        if subset.empty:

            print(
                "No CP-family-compatible "
                "candidate found on/before acquisition."
            )

            continue

        existing_columns = [
            column
            for column
            in display_columns
            if column
            in subset.columns
        ]

        print(
            subset[
                existing_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SHOW ISOLATE-LEVEL MATCHES
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "10 CLOSEST ISOLATE-LEVEL CP-FAMILY MATCHES"
    )

    print(
        "=" * 110
    )

    isolate_matching = pairs[
        pairs[
            "same_cp_family_isolate"
        ]
    ].copy()

    for target_id in sorted(
        pairs[
            "target_id"
        ].unique()
    ):

        subset = (
            isolate_matching[
                isolate_matching[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                [
                    "days_before_acquisition",
                    "source_type",
                    "source_sample_id",
                ]
            )
            .head(10)
        )

        print(
            f"\n--- {target_id} ---"
        )

        if subset.empty:

            print(
                "No isolate-level CP-family "
                "match found on/before acquisition."
            )

            continue

        existing_columns = [
            column
            for column
            in display_columns
            if column
            in subset.columns
        ]

        print(
            subset[
                existing_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SAME-DAY AMBIGUOUS CANDIDATES
    # ========================================================

    same_day = pairs[
        pairs[
            "temporal_relation"
        ]
        == "SAME_DAY_AMBIGUOUS"
    ].copy()

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SAME-DAY AMBIGUOUS CANDIDATES"
    )

    print(
        "=" * 110
    )

    print(
        f"\nTotal same-day ambiguous pairs: "
        f"{len(same_day)}"
    )

    if not same_day.empty:

        same_day_summary = (
            same_day
            .groupby(
                [
                    "target_id",
                    "source_type",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
            .reset_index()
        )

        print(
            "\n"
            + same_day_summary.to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE FULL PAIR DATASET
    # ========================================================

    csv_file = (
        PROCESSED_DATA_DIR
        / "candidate_pairs.csv"
    )

    pairs.to_csv(
        csv_file,
        index=False,
    )

    excel_file = (
        TABLE_DIR
        / "table_02_candidate_pairs.xlsx"
    )

    pairs.to_excel(
        excel_file,
        index=False,
    )

    # ========================================================
    # SAVE SUMMARIES
    # ========================================================

    summary_file = (
        TABLE_DIR
        / "table_02_candidate_pair_summary.xlsx"
    )

    summary.to_excel(
        summary_file,
        index=False,
    )

    cp_summary_file = (
        TABLE_DIR
        / "table_02_cp_family_summary.xlsx"
    )

    cp_summary.to_excel(
        cp_summary_file,
        index=False,
    )

    temporal_summary_file = (
        TABLE_DIR
        / "table_02_temporal_summary.xlsx"
    )

    temporal_summary.to_excel(
        temporal_summary_file,
        index=False,
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 110
    )

    print(
        f"\nCandidate pair CSV:\n"
        f"{csv_file}"
    )

    print(
        f"\nCandidate pair Excel:\n"
        f"{excel_file}"
    )

    print(
        f"\nCandidate-count summary:\n"
        f"{summary_file}"
    )

    print(
        f"\nCP-family summary:\n"
        f"{cp_summary_file}"
    )

    print(
        f"\nTemporal summary:\n"
        f"{temporal_summary_file}"
    )

    print(
        "\nIMPORTANT:"
        "\nKnown_CPO_pos patient hospital matching is currently "
        "UNCONFIRMED and uses Subject ID prefix only."
        "\nSame-day candidate sources are retained but flagged "
        "as SAME_DAY_AMBIGUOUS."
    )


if __name__ == "__main__":
    main()