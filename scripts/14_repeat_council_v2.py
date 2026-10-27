from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpo.council_v2.workflow import (
    DEFAULT_LLAMA_MODEL,
    DEFAULT_MEDGEMMA_MODEL,
    DEFAULT_QWEN_MODEL,
    run_council_v2_target,
)


TARGETS = [
    "T0016",
    "T0036",
    "T0052",
    "T0058",
    "T0080",
    "T0081",
    "T0124",
    "T0125",
]


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "council_v2_repeats"
)


def save_json(
    data,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def run_repeat(
    run_number: int,
    *,
    qwen_model: str,
    medgemma_model: str,
    llama_model: str,
    overwrite: bool = False,
):
    """
    Run one full frozen Council-v2 repeat over all 8 targets.
    """

    run_name = f"run{run_number:02d}"

    output_dir = (
        OUTPUT_ROOT
        / run_name
    )

    if (
        output_dir.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"{output_dir} already exists. "
            "Refusing to overwrite a previous experimental run."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"FROZEN COUNCIL V2 REPEAT: {run_name}"
    )

    print(
        "=" * 120
    )

    for index, target_id in enumerate(
        TARGETS,
        start=1,
    ):

        print(
            "\n"
            + "#"
            * 120
        )

        print(
            f"{run_name.upper()} — "
            f"TARGET {index}/{len(TARGETS)}: "
            f"{target_id}"
        )

        print(
            "#"
            * 120
        )

        try:

            report = run_council_v2_target(
                target_id,

                qwen_model=qwen_model,
                medgemma_model=medgemma_model,
                llama_model=llama_model,
            )

            report[
                "repeat_run"
            ] = run_number

            reports.append(
                report
            )

            target_path = (
                output_dir
                / (
                    f"{target_id}"
                    "_council_v2_report.json"
                )
            )

            save_json(
                report,
                target_path,
            )

            print(
                f"\nSaved: {target_path}"
            )

        except Exception as error:

            error_report = {
                "target_id":
                    target_id,

                "repeat_run":
                    run_number,

                "status":
                    "ERROR",

                "error_type":
                    type(
                        error
                    ).__name__,

                "error":
                    str(
                        error
                    ),
            }

            reports.append(
                error_report
            )

            error_path = (
                output_dir
                / (
                    f"{target_id}"
                    "_council_v2_error.json"
                )
            )

            save_json(
                error_report,
                error_path,
            )

            print(
                f"\nERROR for {target_id}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    combined = {
        "experiment":
            "heterogeneous_llm_council_v2_repeat",

        "council_version":
            "2.0",

        "repeat_run":
            run_number,

        "models": {
            "investigator":
                qwen_model,

            "biomedical_expert":
                medgemma_model,

            "causal_critic":
                llama_model,
        },

        "frozen_configuration": {
            "temperature":
                0,

            "seed":
                42,

            "tool_calls":
                False,

            "note":
                (
                    "Scientific configuration identical to "
                    "Council v2. Repeat performed solely to "
                    "assess inference stability."
                ),
        },

        "targets":
            TARGETS,

        "reports":
            reports,
    }

    combined_path = (
        output_dir
        / "all_council_v2_reports.json"
    )

    save_json(
        combined,
        combined_path,
    )

    successful = [
        report
        for report
        in reports
        if report.get(
            "status"
        )
        != "ERROR"
    ]

    errors = [
        report
        for report
        in reports
        if report.get(
            "status"
        )
        == "ERROR"
    ]

    matches = [
        report
        for report
        in successful
        if (
            report
            .get(
                "validation",
                {}
            )
            .get(
                "consensus_matches_reference"
            )
            is True
        )
    ]

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"{run_name.upper()} COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"Successful targets: "
        f"{len(successful)}/{len(TARGETS)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print(
        "Consensus matching WGS reference: "
        f"{len(matches)}/{len(successful)} completed"
    )

    print(
        f"\nCombined report:\n{combined_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Repeat the frozen Council-v2 configuration "
            "for inference-stability analysis."
        )
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[
            2,
            3,
            4,
            5,
        ],
        help=(
            "Repeat indices to run. "
            "Default: 2 3 4 5"
        ),
    )

    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL,
    )

    parser.add_argument(
        "--medgemma-model",
        default=DEFAULT_MEDGEMMA_MODEL,
    )

    parser.add_argument(
        "--llama-model",
        default=DEFAULT_LLAMA_MODEL,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing repeat folder. "
            "Avoid using this for the paper experiment."
        ),
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for run_number in args.runs:

        if run_number == 1:

            raise ValueError(
                "Run 1 already exists and must not be rerun. "
                "Use the original Council-v2 output."
            )

        run_repeat(
            run_number,

            qwen_model=args.qwen_model,
            medgemma_model=args.medgemma_model,
            llama_model=args.llama_model,

            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()