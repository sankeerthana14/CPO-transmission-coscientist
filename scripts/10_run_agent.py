from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpo.agents.workflow import (
    investigate_target,
)

from cpo.config import (
    TABLE_DIR,
)


# ============================================================
# ACQUISITION TARGETS
# ============================================================

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


# ============================================================
# RUN ONE TARGET
# ============================================================


def run_one(
    target_id: str,
    model: str,
) -> dict:
    """
    Run the complete local AI co-scientist workflow for one
    acquisition patient:

        Investigator
            ->
        Critic
            ->
        Final Investigator revision

    The LLM runs locally through Ollama.
    """

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"LOCAL AI CO-SCIENTIST: {target_id}"
    )

    print(
        "=" * 120
    )

    print(
        f"\nLocal model: {model}"
    )

    print(
        "\nRunning Investigator -> Critic -> Revision..."
    )

    # --------------------------------------------------------
    # Execute agent workflow
    # --------------------------------------------------------

    result = investigate_target(
        target_id=target_id,
        model=model,
    )

    # ========================================================
    # INITIAL INVESTIGATION
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "INITIAL INVESTIGATION"
    )

    print(
        "=" * 120
    )

    print(
        json.dumps(
            result[
                "initial_investigation"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # CRITIC REVIEW
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "CRITIC REVIEW"
    )

    print(
        "=" * 120
    )

    print(
        json.dumps(
            result[
                "critic_review"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # FINAL INVESTIGATION
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL INVESTIGATION"
    )

    print(
        "=" * 120
    )

    print(
        json.dumps(
            result[
                "final_investigation"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_dir = (
        TABLE_DIR.parent
        / "reports"
        / "agents"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{target_id}_agent_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "-" * 120
    )

    print(
        "REPORT SAVED"
    )

    print(
        "-" * 120
    )

    print(
        f"\n{output_file}"
    )

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the local CPO Transmission "
            "AI Co-scientist."
        )
    )

    # --------------------------------------------------------
    # Optional target ID
    # --------------------------------------------------------

    parser.add_argument(
        "target_id",
        nargs="?",
        help=(
            "Acquisition target ID, for example T0124."
        ),
    )

    # --------------------------------------------------------
    # Run all targets
    # --------------------------------------------------------

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run the AI co-scientist for all eight "
            "acquisition patients."
        ),
    )

    # --------------------------------------------------------
    # Local Ollama model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help=(
            "Ollama model to use. "
            "Default: qwen3:8b"
        ),
    )

    args = parser.parse_args()

    model = (
        args.model
        .strip()
    )

    # ========================================================
    # RUN ALL TARGETS
    # ========================================================

    if args.all:

        print(
            "\n"
            + "=" * 120
        )

        print(
            "RUNNING ALL 8 ACQUISITION PATIENTS"
        )

        print(
            "=" * 120
        )

        print(
            f"\nModel: {model}"
        )

        all_results = []

        for index, target_id in enumerate(
            TARGETS,
            start=1,
        ):

            print(
                "\n"
                + "#" * 120
            )

            print(
                f"TARGET {index}/{len(TARGETS)}: "
                f"{target_id}"
            )

            print(
                "#" * 120
            )

            try:

                result = run_one(
                    target_id=target_id,
                    model=model,
                )

                all_results.append(
                    result
                )

            except Exception as error:

                print(
                    "\nERROR while investigating "
                    f"{target_id}:"
                )

                print(
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                # Preserve failure information but continue
                # with the remaining acquisition patients.
                all_results.append(
                    {
                        "target_id":
                            target_id,

                        "error_type":
                            type(
                                error
                            ).__name__,

                        "error":
                            str(
                                error
                            ),
                    }
                )

        # ----------------------------------------------------
        # Save combined report
        # ----------------------------------------------------

        output_dir = (
            TABLE_DIR.parent
            / "reports"
            / "agents"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined_file = (
            output_dir
            / "all_agent_reports.json"
        )

        with open(
            combined_file,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                all_results,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\n"
            + "=" * 120
        )

        print(
            "ALL-PATIENT RUN COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            f"\nCombined report:\n"
            f"{combined_file}"
        )

        return

    # ========================================================
    # RUN ONE TARGET
    # ========================================================

    if not args.target_id:

        parser.error(
            "Provide a target ID, e.g. T0124, "
            "or use --all."
        )

    target_id = (
        args.target_id
        .strip()
        .upper()
    )

    if (
        target_id
        not in TARGETS
    ):

        parser.error(
            f"Unknown target ID: {target_id}\n\n"
            f"Valid targets:\n"
            f"{', '.join(TARGETS)}"
        )

    run_one(
        target_id=target_id,
        model=model,
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()