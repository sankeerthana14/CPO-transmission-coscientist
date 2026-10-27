from cpo.wgs_linkage import (
    attach_wgs_linkage,
)

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)


def main():

    print(
        "\nAttaching direct WGS linkage evidence..."
    )

    data = (
        attach_wgs_linkage()
    )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "WGS-LABELED CANDIDATE PAIRS"
    )

    print(
        "=" * 120
    )

    print(
        f"\nTotal candidate pairs: "
        f"{len(data)}"
    )

    print(
        f"WGS-evaluable pairs: "
        f"{int(data['wgs_pair_evaluable'].sum())}"
    )

    print(
        f"Direct clonal links: "
        f"{int(data['clonal_link'].sum())}"
    )

    print(
        f"Direct plasmid links: "
        f"{int(data['plasmid_link'].sum())}"
    )

    print(
        f"Any WGS link: "
        f"{int(data['any_wgs_link'].sum())}"
    )

    # ========================================================
    # LINK COUNTS BY TARGET
    # ========================================================

    summary = (
        data
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

            wgs_evaluable_count=(
                "wgs_pair_evaluable",
                "sum",
            ),

            clonal_link_count=(
                "clonal_link",
                "sum",
            ),

            plasmid_link_count=(
                "plasmid_link",
                "sum",
            ),

            any_wgs_link_count=(
                "any_wgs_link",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        "\n"
        + "-" * 120
    )

    print(
        "WGS LINKS BY TARGET"
    )

    print(
        "-" * 120
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # ACTUAL LINKED CANDIDATES
    # ========================================================

    linked = data[
        data[
            "any_wgs_link"
        ]
    ].copy()

    print(
        "\n"
        + "=" * 120
    )

    print(
        "DIRECT WGS-LINKED CANDIDATE SOURCES"
    )

    print(
        "=" * 120
    )

    display_columns = [
        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",
        "temporal_relation",

        "target_species",
        "target_st",

        "source_species_primary",
        "source_st_primary",

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

    if linked.empty:

        print(
            "\nNo candidate pair matched a direct "
            "clonal/plasmid WGS pair."
        )

    else:

        for target_id in sorted(
            linked[
                "target_id"
            ].unique()
        ):

            subset = (
                linked[
                    linked[
                        "target_id"
                    ]
                    == target_id
                ]
                .sort_values(
                    [
                        "clonal_link",
                        "plasmid_link",
                        "days_before_acquisition",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                    ],
                )
            )

            print(
                f"\n{'=' * 30} "
                f"{target_id} "
                f"{'=' * 30}"
            )

            existing = [
                column
                for column
                in display_columns
                if column
                in subset.columns
            ]

            print(
                subset[
                    existing
                ]
                .to_string(
                    index=False
                )
            )

    # ========================================================
    # TARGET WGS COVERAGE
    # ========================================================

    coverage = (
        data[
            [
                "target_id",
                "target_sample_id",
                "target_wgs_isolate_count",
                "target_wgs_resolution",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "target_id"
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TARGET WGS COVERAGE"
    )

    print(
        "=" * 120
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    csv_file = (
        PROCESSED_DATA_DIR
        / "wgs_labeled_pairs.csv"
    )

    data.to_csv(
        csv_file,
        index=False,
    )

    excel_file = (
        TABLE_DIR
        / "table_04_wgs_labeled_pairs.xlsx"
    )

    data.to_excel(
        excel_file,
        index=False,
    )

    summary_file = (
        TABLE_DIR
        / "table_04_wgs_link_summary.xlsx"
    )

    summary.to_excel(
        summary_file,
        index=False,
    )

    linked_file = (
        TABLE_DIR
        / "table_04_direct_wgs_links.xlsx"
    )

    linked.to_excel(
        linked_file,
        index=False,
    )

    coverage_file = (
        TABLE_DIR
        / "table_04_target_wgs_coverage.xlsx"
    )

    coverage.to_excel(
        coverage_file,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 120
    )

    print(
        f"\nWGS-labeled CSV:\n"
        f"{csv_file}"
    )

    print(
        f"\nWGS-labeled Excel:\n"
        f"{excel_file}"
    )

    print(
        f"\nWGS summary:\n"
        f"{summary_file}"
    )

    print(
        f"\nDirect WGS links:\n"
        f"{linked_file}"
    )

    print(
        f"\nTarget WGS coverage:\n"
        f"{coverage_file}"
    )

    print(
        "\nIMPORTANT:"
        "\n'NO_DIRECT_LINK_LISTED' does NOT yet mean a "
        "validated negative ML label."
        "\nBefore XGBoost, we must confirm that absence from "
        "Clonal_linked_pairs / Plasmid_linked_pairs means "
        "the isolates were compared and failed the linkage "
        "criterion."
    )


if __name__ == "__main__":
    main()