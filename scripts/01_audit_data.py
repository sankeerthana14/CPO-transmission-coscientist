import sys

from openpyxl import load_workbook

from cpo.config import (
    ENV_FILE,
    WGS_FILE,
)


def print_workbook(filepath):

    print("\n" + "=" * 80)
    print(filepath.name)
    print("=" * 80)

    workbook = load_workbook(
        filepath,
        read_only=True,
        data_only=True,
    )

    for sheet in workbook.sheetnames:

        worksheet = workbook[sheet]

        print(
            f"{sheet:<30} "
            f"rows={worksheet.max_row:<7} "
            f"cols={worksheet.max_column}"
        )


def main():

    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f"ENV file not found:\n{ENV_FILE}"
        )

    if not WGS_FILE.exists():
        raise FileNotFoundError(
            f"WGS file not found:\n{WGS_FILE}"
        )

    print_workbook(ENV_FILE)
    print_workbook(WGS_FILE)


if __name__ == "__main__":
    main()