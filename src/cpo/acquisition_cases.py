import pandas as pd

from cpo.config import (
    ENV_FILE,
    TARGET_PATIENTS,
)

from cpo.data_loader import (
    read_sheet,
    coerce_date,
    normalize_missing,
    unique_non_missing,
)


def build_acquisition_cases() -> pd.DataFrame:
    """
    Build one clean row for each of the 8 patients who acquired
    a carbapenemase-producing gene during hospitalisation.
    """

    # ---------------------------------------------------------
    # 1. LOAD ENV_pos
    # ---------------------------------------------------------

    env = read_sheet(
        filepath=ENV_FILE,
        sheet_name="ENV_pos",
        required_column="SubjectID",
    )

    # ---------------------------------------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = [
        "SubjectID",
        "Date of positive sample",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in env.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in ENV_pos: {missing_columns}\n\n"
            f"Available columns:\n{list(env.columns)}"
        )

    # ---------------------------------------------------------
    # 3. KEEP ONLY THE 8 ACQUISITION PATIENTS
    # ---------------------------------------------------------

    target_rows = env[
        env["SubjectID"].isin(TARGET_PATIENTS)
    ].copy()

    if target_rows.empty:
        raise ValueError(
            "None of the 8 target patients were found in ENV_pos."
        )

    # ---------------------------------------------------------
    # 4. PARSE ACQUISITION DATE
    # ---------------------------------------------------------

    target_rows["acquisition_date"] = (
        target_rows["Date of positive sample"]
        .apply(coerce_date)
    )

    # Keep only rows that actually contain a positive sample date
    positive_rows = target_rows[
        target_rows["acquisition_date"].notna()
    ].copy()

    if positive_rows.empty:
        raise ValueError(
            "The 8 target patients were found, but no positive sample dates "
            "could be parsed."
        )

    # ---------------------------------------------------------
    # 5. FIND FIRST POSITIVE SAMPLE FOR EACH PATIENT
    # ---------------------------------------------------------

    positive_rows = positive_rows.sort_values(
        [
            "SubjectID",
            "acquisition_date",
        ]
    )

    first_acquisition = (
        positive_rows
        .groupby(
            "SubjectID",
            as_index=False,
        )
        .first()
    )

    # ---------------------------------------------------------
    # 6. ADD HOSPITAL / WARD INFORMATION
    # ---------------------------------------------------------

    aggregations = {}

    if "Hospital Code" in target_rows.columns:
        aggregations["hospital"] = (
            "Hospital Code",
            unique_non_missing,
        )

    if "Ward" in target_rows.columns:
        aggregations["wards"] = (
            "Ward",
            unique_non_missing,
        )

    if aggregations:
        location_df = (
            target_rows
            .groupby("SubjectID")
            .agg(**aggregations)
            .reset_index()
        )

        first_acquisition = first_acquisition.merge(
            location_df,
            on="SubjectID",
            how="left",
        )

    # ---------------------------------------------------------
    # 7. STANDARDISE THE IMPORTANT BIOLOGY COLUMNS
    # ---------------------------------------------------------

    column_mapping = {
        "LabGXG": "lab_cp_gene",
        "Genomics Species_ST": "genomics_species_st",
        "Genomics_CPgene": "genomics_cp_gene",
        "SampleID": "positive_sample_id",
        "Sample_schedule": "sample_schedule",
        "Remarks": "remarks",
    }

    for original_column, new_column in column_mapping.items():

        if original_column in first_acquisition.columns:
            first_acquisition[new_column] = (
                first_acquisition[original_column]
                .apply(normalize_missing)
            )
        else:
            first_acquisition[new_column] = pd.NA

    # ---------------------------------------------------------
    # 8. FLAG WHETHER CULTURE/WGS INFORMATION EXISTS
    # ---------------------------------------------------------

    first_acquisition["has_culture_wgs"] = (
        first_acquisition[
            "genomics_species_st"
        ].notna()
    )

    # ---------------------------------------------------------
    # 9. BUILD FINAL OUTPUT TABLE
    # ---------------------------------------------------------

    output_columns = [
        "SubjectID",
        "acquisition_date",
        "lab_cp_gene",
        "genomics_species_st",
        "genomics_cp_gene",
        "has_culture_wgs",
    ]

    optional_columns = [
        "hospital",
        "wards",
        "positive_sample_id",
        "sample_schedule",
        "remarks",
    ]

    for column in optional_columns:
        if column in first_acquisition.columns:
            output_columns.append(column)

    cases = first_acquisition[
        output_columns
    ].copy()

    # ---------------------------------------------------------
    # 10. SORT CHRONOLOGICALLY
    # ---------------------------------------------------------

    cases = cases.sort_values(
        "acquisition_date"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 11. VALIDATION
    # ---------------------------------------------------------

    expected_patients = set(TARGET_PATIENTS)
    observed_patients = set(cases["SubjectID"])

    missing_patients = (
        expected_patients
        - observed_patients
    )

    unexpected_patients = (
        observed_patients
        - expected_patients
    )

    if missing_patients:
        raise ValueError(
            "Missing acquisition patients: "
            f"{sorted(missing_patients)}"
        )

    if unexpected_patients:
        raise ValueError(
            "Unexpected patients found: "
            f"{sorted(unexpected_patients)}"
        )

    if len(cases) != 8:
        raise ValueError(
            f"Expected exactly 8 acquisition patients, "
            f"but found {len(cases)}."
        )

    return cases