from cpo.acquisition_cases import build_acquisition_cases
from cpo.config import TABLE_DIR, PROCESSED_DATA_DIR


def main():
    # ---------------------------------------------------------
    # BUILD THE 8-PATIENT ACQUISITION TABLE
    # ---------------------------------------------------------

    cases = build_acquisition_cases()

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("CPO ACQUISITION CASES")
    print("=" * 100)

    print(
        cases.to_string(
            index=False,
        )
    )

    print("\n" + "-" * 100)

    print(
        f"Number of acquisition patients: {len(cases)}"
    )

    print(
        "Patients with culture/WGS: "
        f"{int(cases['has_culture_wgs'].sum())}"
    )

    print("-" * 100)

    # ---------------------------------------------------------
    # SAVE AS CSV
    # ---------------------------------------------------------

    csv_file = (
        PROCESSED_DATA_DIR
        / "acquisition_cases.csv"
    )

    cases.to_csv(
        csv_file,
        index=False,
    )

    # ---------------------------------------------------------
    # SAVE AS EXCEL
    # ---------------------------------------------------------

    excel_file = (
        TABLE_DIR
        / "table_01_acquisition_cases.xlsx"
    )

    cases.to_excel(
        excel_file,
        index=False,
    )

    # ---------------------------------------------------------
    # CONFIRM OUTPUT
    # ---------------------------------------------------------

    print("\nSaved files:")

    print(
        f"CSV:   {csv_file}"
    )

    print(
        f"Excel: {excel_file}"
    )


if __name__ == "__main__":
    main()