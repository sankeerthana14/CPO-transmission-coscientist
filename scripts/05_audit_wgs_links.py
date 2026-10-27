from cpo.config import WGS_FILE
from cpo.data_loader import read_sheet


SHEETS = [
    "WGS_isolate_list",
    "Clonal_linked_pairs",
    "Plasmid_linked_pairs",
]


def main():

    print("\n" + "=" * 120)
    print("WGS WORKBOOK AUDIT")
    print("=" * 120)

    for sheet_name in SHEETS:

        print("\n")
        print("=" * 120)
        print(sheet_name)
        print("=" * 120)

        # We don't yet know the exact first required column,
        # so load the sheet normally.
        df = read_sheet(
            filepath=WGS_FILE,
            sheet_name=sheet_name,
        )

        print(
            f"\nRows: {len(df)}"
        )

        print(
            f"Columns: {len(df.columns)}"
        )

        print("\nCOLUMN NAMES:")

        for i, column in enumerate(
            df.columns,
            start=1,
        ):

            print(
                f"{i:>3}. {column}"
            )

        print(
            "\nFIRST 5 ROWS:"
        )

        print(
            df.head(5).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()