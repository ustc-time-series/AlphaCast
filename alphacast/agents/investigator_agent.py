from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Callable, Dict, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart  # type: ignore
from pydantic_ai.models.function import FunctionModel  # type: ignore

from alphacast.config import DatasetConfig, ExperimentConfig
from .prompts import get_agent_instructions


INVESTIGATOR_AGENT_PROMPT_FALLBACK = dedent(
    """
    You are InvestigatorAgent. Call `gather_forecast_inputs` once with the provided parameters and return the same structured JSON payload produced by GeneratorAgent2’s `load_prediction_context`, including the optimal baseline forecast, similarity hints, feature metadata, exogenous slices, and coverage notes.
    """
)


def create_investigator_agent(
    cfg: ExperimentConfig | None,
    dataset_lookup: Dict[str, DatasetConfig],
    briefing_lookup: Dict[str, str],
    prepare_investor_packet: Callable[[ExperimentConfig | None, DatasetConfig, Dict[str, str], int, Optional[int]], dict],
    json_default: Callable[[Any], Any],
) -> Agent:
    instructions = get_agent_instructions("InvestigatorAgent", INVESTIGATOR_AGENT_PROMPT_FALLBACK)

    def _extract_json_request(messages: list[Any]) -> dict[str, Any]:
        for message in reversed(messages):
            parts = getattr(message, "parts", [])
            for part in reversed(parts):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    try:
                        return json.loads(content)
                    except Exception:
                        continue
        return {}

    def _investigator_model(messages, agent_info) -> ModelResponse:
        payload = _extract_json_request(messages)
        dataset_name = payload.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name is required")
        ds_cfg = dataset_lookup.get(dataset_name)
        if ds_cfg is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        window_offset = payload.get("window_offset")
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
        forecast_horizon = payload.get("forecast_horizon")
        if forecast_horizon is not None:
            try:
                forecast_horizon = int(forecast_horizon)
            except Exception:
                forecast_horizon = None
        packet = prepare_investor_packet(
            cfg,
            ds_cfg,
            briefing_lookup,
            window_offset_int,
            forecast_horizon,
        )
        return ModelResponse(
            parts=[TextPart(json.dumps(packet, default=json_default))],
            model_name="function:investigator",
        )

    return Agent(
        FunctionModel(function=_investigator_model),
        instructions=instructions,
    )


__all__ = ["create_investigator_agent", "INVESTIGATOR_AGENT_PROMPT_FALLBACK"]
