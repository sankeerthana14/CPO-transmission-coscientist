from cpo.features import (
    build_pair_features,
)

from cpo.config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
)


def main():

    print(
        "\nBuilding source-target feature table..."
    )

    features = (
        build_pair_features()
    )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SOURCE-TARGET FEATURE TABLE"
    )

    print(
        "=" * 110
    )

    print(
        f"\nTotal candidate pairs: "
        f"{len(features)}"
    )

    print(
        f"Unique targets: "
        f"{features['target_id'].nunique()}"
    )

    print(
        f"Unique pair IDs: "
        f"{features['pair_id'].nunique()}"
    )

    # ========================================================
    # GLOBAL FEATURE COUNTS
    # ========================================================

    feature_counts = {
        "Prior sources":
            int(
                features[
                    "is_prior"
                ].sum()
            ),

        "Same-day ambiguous":
            int(
                features[
                    "is_same_day"
                ].sum()
            ),

        "Any CP-family match":
            int(
                features[
                    "same_cp_family_any"
                ].sum()
            ),

        "Isolate CP-family match":
            int(
                features[
                    "same_cp_family_isolate"
                ].sum()
            ),

        "Exact CP-gene match":
            int(
                features[
                    "same_exact_cp_gene_isolate"
                ].sum()
            ),

        "Same bacterial species":
            int(
                features[
                    "same_species_any"
                ].sum()
            ),

        "Same known ST":
            int(
                features[
                    "same_st_any"
                ].sum()
            ),

        "Pair has isolate data":
            int(
                features[
                    "has_pair_isolate_data"
                ].sum()
            ),

        "Preliminary WGS eligible":
            int(
                features[
                    "preliminary_wgs_pair_eligible"
                ].sum()
            ),

        "Duplicate biology signature":
            int(
                features[
                    "duplicate_biology_signature"
                ].sum()
            ),
    }

    print(
        "\nFEATURE COUNTS"
    )

    for (
        name,
        count,
    ) in feature_counts.items():

        print(
            f"{name:<35} "
            f"{count:>6}"
        )

    # ========================================================
    # SUMMARY BY TARGET / SOURCE TYPE
    # ========================================================

    summary = (
        features
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

            prior_count=(
                "is_prior",
                "sum",
            ),

            same_day_count=(
                "is_same_day",
                "sum",
            ),

            cp_family_any_count=(
                "same_cp_family_any",
                "sum",
            ),

            cp_family_isolate_count=(
                "same_cp_family_isolate",
                "sum",
            ),

            exact_gene_count=(
                "same_exact_cp_gene_isolate",
                "sum",
            ),

            same_species_count=(
                "same_species_any",
                "sum",
            ),

            same_st_count=(
                "same_st_any",
                "sum",
            ),

            preliminary_wgs_eligible_count=(
                "preliminary_wgs_pair_eligible",
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
        "FEATURE SUMMARY BY TARGET"
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
    # HIGH-COMPATIBILITY PREVIEW
    # ========================================================
    #
    # This is NOT a transmission label.
    #
    # It simply surfaces candidates with:
    #
    #   prior timing
    #   + same CP family
    #   + same bacterial species
    #
    # WGS still decides whether genomic linkage exists.
    # ========================================================

    compatible = features[
        features[
            "is_prior"
        ]
        &
        features[
            "same_cp_family_any"
        ]
        &
        features[
            "same_species_any"
        ]
    ].copy()

    print(
        "\n"
        + "=" * 110
    )

    print(
        "BIOLOGICALLY COMPATIBLE CANDIDATES "
        "(NOT TRANSMISSION LABELS)"
    )

    print(
        "=" * 110
    )

    display_columns = [
        "target_id",
        "target_sample_id",

        "source_type",
        "source_subject_id",
        "source_sample_id",

        "days_before_acquisition",

        "target_species",
        "target_st",

        "source_species_primary",
        "source_st_primary",

        "same_cp_family_gx",
        "same_cp_family_isolate",
        "same_exact_cp_gene_isolate",

        "same_species_any",
        "same_st_any",

        "hospital_match_status",
    ]

    for target_id in sorted(
        features[
            "target_id"
        ].unique()
    ):

        subset = (
            compatible[
                compatible[
                    "target_id"
                ]
                == target_id
            ]
            .sort_values(
                [
                    "same_st_any",
                    "same_exact_cp_gene_isolate",
                    "same_cp_family_isolate",
                    "days_before_acquisition",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
            )
            .head(15)
        )

        print(
            f"\n{'=' * 25} "
            f"{target_id} "
            f"{'=' * 25}"
        )

        if subset.empty:

            print(
                "No prior candidate with both "
                "CP-family and species compatibility."
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
    # SAME-ST CANDIDATES
    # ========================================================

    same_st = features[
        features[
            "is_prior"
        ]
        &
        features[
            "same_st_any"
        ]
    ].copy()

    print(
        "\n"
        + "=" * 110
    )

    print(
        "PRIOR SAME-SPECIES + SAME-ST CANDIDATES"
    )

    print(
        "=" * 110
    )

    if same_st.empty:

        print(
            "\nNo same-ST candidates found."
        )

    else:

        st_display = [
            "target_id",
            "target_sample_id",

            "source_type",
            "source_subject_id",
            "source_sample_id",

            "days_before_acquisition",

            "target_species",
            "target_st",

            "source_species_primary",
            "source_st_primary",

            "same_exact_cp_gene_isolate",
            "same_cp_family_isolate",

            "hospital_match_status",
        ]

        existing_columns = [
            column
            for column
            in st_display
            if column
            in same_st.columns
        ]

        print(
            same_st[
                existing_columns
            ]
            .sort_values(
                [
                    "target_id",
                    "days_before_acquisition",
                ]
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # DUPLICATE-SIGNATURE DIAGNOSTIC
    # ========================================================

    duplicates = features[
        features[
            "duplicate_biology_signature"
        ]
    ].copy()

    print(
        "\n"
        + "=" * 110
    )

    print(
        "DUPLICATE BIOLOGICAL SIGNATURE DIAGNOSTIC"
    )

    print(
        "=" * 110
    )

    print(
        f"\nRows flagged: "
        f"{len(duplicates)}"
    )

    print(
        "These are NOT automatically removed; "
        "they may represent different isolate slots."
    )

    # ========================================================
    # SAVE FEATURE DATASET
    # ========================================================

    csv_file = (
        PROCESSED_DATA_DIR
        / "pair_features.csv"
    )

    features.to_csv(
        csv_file,
        index=False,
    )

    excel_file = (
        TABLE_DIR
        / "table_03_pair_features.xlsx"
    )

    features.to_excel(
        excel_file,
        index=False,
    )

    summary_file = (
        TABLE_DIR
        / "table_03_feature_summary.xlsx"
    )

    summary.to_excel(
        summary_file,
        index=False,
    )

    # ========================================================
    # SAVE SAME-ST DIAGNOSTIC
    # ========================================================

    same_st_file = (
        TABLE_DIR
        / "table_03_same_st_candidates.xlsx"
    )

    same_st.to_excel(
        same_st_file,
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
        f"\nFeature CSV:\n"
        f"{csv_file}"
    )

    print(
        f"\nFeature Excel:\n"
        f"{excel_file}"
    )

    print(
        f"\nFeature summary:\n"
        f"{summary_file}"
    )

    print(
        f"\nSame-ST diagnostic:\n"
        f"{same_st_file}"
    )

    print(
        "\nIMPORTANT:"
        "\nNo clonal or plasmid WGS linkage has been "
        "used yet."
        "\nThese are candidate-pair FEATURES only."
    )


if __name__ == "__main__":
    main()