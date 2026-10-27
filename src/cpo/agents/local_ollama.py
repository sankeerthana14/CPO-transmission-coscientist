from __future__ import annotations

from typing import Type

from ollama import ChatResponse, chat
from pydantic import BaseModel

from cpo.tools.transmission_tools import (
    AVAILABLE_TOOLS,
)


# ============================================================
# DEFAULT LOCAL MODEL
# ============================================================

DEFAULT_MODEL = "qwen3:8b"


# ============================================================
# TOOL-CALLING AGENT LOOP
# ============================================================


def run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = 12,
    verbose: bool = True,
):
    """
    Run a local Ollama tool-calling agent.

    The LLM decides which deterministic analysis tool to call.
    Tool results are returned to the LLM, which can then call
    more tools or finish its reasoning.

    Returns
    -------
    list
        Complete conversation history including tool calls
        and tool results.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    tool_functions = list(
        AVAILABLE_TOOLS.values()
    )

    for step in range(
        1,
        max_steps + 1,
    ):

        if verbose:

            print(
                f"\nAgent step {step}/{max_steps}..."
            )

        response: ChatResponse = chat(
            model=model,
            messages=messages,
            tools=tool_functions,
            think=True,
            options={
                "temperature": 0.1,
            },
        )

        # Preserve assistant response, including tool calls.
        messages.append(
            response.message
        )

        tool_calls = (
            response.message.tool_calls
            or []
        )

        # ----------------------------------------------------
        # No tool call = reasoning phase is complete
        # ----------------------------------------------------

        if not tool_calls:

            if verbose:

                print(
                    "Agent finished evidence collection."
                )

            return messages

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for tool_call in tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = dict(
                tool_call.function.arguments
                or {}
            )

            if verbose:

                print(
                    f"  Calling tool: {tool_name}"
                )

                if arguments:

                    print(
                        f"  Arguments: {arguments}"
                    )

            function = AVAILABLE_TOOLS.get(
                tool_name
            )

            if function is None:

                tool_result = (
                    f"ERROR: Unknown tool '{tool_name}'."
                )

            else:

                try:

                    tool_result = function(
                        **arguments
                    )

                except Exception as error:

                    tool_result = (
                        "ERROR executing tool "
                        f"{tool_name}: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

            # Official Ollama tool-result format.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(
                        tool_result
                    ),
                }
            )

    raise RuntimeError(
        "Agent exceeded maximum tool-calling steps. "
        "The model may be stuck repeatedly calling tools."
    )


# ============================================================
# STRUCTURED OUTPUT
# ============================================================


def generate_structured_report(
    messages,
    schema: Type[BaseModel],
    final_instruction: str,
    model: str = DEFAULT_MODEL,
):
    """
    Convert accumulated evidence/reasoning into a validated
    Pydantic structured report.

    Ollama receives the Pydantic JSON schema through `format`.
    """

    schema_json = (
        schema.model_json_schema()
    )

    structured_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                final_instruction
                + "\n\n"
                + "Return ONLY JSON matching this schema:\n"
                + str(
                    schema_json
                )
            ),
        },
    ]

    response: ChatResponse = chat(
        model=model,
        messages=structured_messages,
        format=schema_json,
        think=False,
        options={
            "temperature": 0,
        },
    )

    if not response.message.content:

        raise RuntimeError(
            "Ollama returned an empty structured response."
        )

    return schema.model_validate_json(
        response.message.content
    )