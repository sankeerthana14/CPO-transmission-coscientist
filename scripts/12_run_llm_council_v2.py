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
    Path(
        __file__
    )
    .resolve()
    .parents[1]
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "council_v2"
)


def save_json(
    data,
    path: Path,
):
    """
    Save formatted JSON.
    """

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


def run_one(
    target_id: str,
    *,
    qwen_model: str,
    medgemma_model: str,
    llama_model: str,
):
    """
    Run and save Council-v2 output for one target.
    """

    report = (
        run_council_v2_target(
            target_id,

            qwen_model=(
                qwen_model
            ),

            medgemma_model=(
                medgemma_model
            ),

            llama_model=(
                llama_model
            ),
        )
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{target_id}"
            "_council_v2_report.json"
        )
    )

    save_json(
        report,
        output_path,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    return report


def main():
    parser = (
        argparse.ArgumentParser(
            description=(
                "Run heterogeneous "
                "LLM Council v2."
            )
        )
    )

    parser.add_argument(
        "target_id",
        nargs="?",
        choices=TARGETS,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--qwen-model",
        default=(
            DEFAULT_QWEN_MODEL
        ),
    )

    parser.add_argument(
        "--medgemma-model",
        default=(
            DEFAULT_MEDGEMMA_MODEL
        ),
    )

    parser.add_argument(
        "--llama-model",
        default=(
            DEFAULT_LLAMA_MODEL
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        not args.all
        and not args.target_id
    ):

        parser.error(
            "Provide target_id or --all."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = (
        TARGETS
        if args.all
        else [
            args.target_id
        ]
    )

    reports = []

    for index, target_id in enumerate(
        targets,
        start=1,
    ):

        print(
            "\n"
            + "#"
            * 120
        )

        print(
            f"COUNCIL V2 TARGET "
            f"{index}/{len(targets)}: "
            f"{target_id}"
        )

        print(
            "#"
            * 120
        )

        try:

            report = run_one(
                target_id,

                qwen_model=(
                    args.qwen_model
                ),

                medgemma_model=(
                    args.medgemma_model
                ),

                llama_model=(
                    args.llama_model
                ),
            )

            reports.append(
                report
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                f"{target_id}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            error_report = {
                "target_id":
                    target_id,

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

            save_json(
                error_report,

                OUTPUT_DIR
                / (
                    f"{target_id}"
                    "_council_v2_error.json"
                ),
            )

    if args.all:

        combined_path = (
            OUTPUT_DIR
            / "all_council_v2_reports.json"
        )

        combined = {
            "experiment":
                "heterogeneous_llm_council_v2",

            "council_version":
                "2.0",

            "models": {
                "investigator":
                    args.qwen_model,

                "biomedical_expert":
                    args.medgemma_model,

                "causal_critic":
                    args.llama_model,
            },

            "targets":
                TARGETS,

            "reports":
                reports,
        }

        save_json(
            combined,
            combined_path,
        )

        print(
            "\n"
            + "=" * 120
        )

        print(
            "ALL-PATIENT COUNCIL V2 RUN COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            "\nCombined report:"
        )

        print(
            combined_path
        )


if __name__ == "__main__":
    main()