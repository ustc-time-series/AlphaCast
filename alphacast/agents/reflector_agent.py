from __future__ import annotations

import json
import math
import re
from textwrap import dedent
from typing import Any, Callable, Dict, List

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart  # type: ignore
from pydantic_ai.models.function import FunctionModel  # type: ignore

from .prompts import get_agent_instructions


REFLECTOR_AGENT_PROMPT_FALLBACK = dedent(
    """
    You are ReflectorAgent. Audit each GeneratorAgent forecast by calling `assess_forecast` once and `scan_chain_of_thought` once.
    Detect unsupported numeric claims or hallucinated statistics in GeneratorAgent's reasoning and demand a rerun when issues exist.
    Output strict JSON with `approved`, `issues`, `notes`, and attach any diagnostic findings that explain the decision.
    """
)

_NUMBER_PATTERN = re.compile(
    r"(?P<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?P<percent>%?)"
)
_WINDOW_CLAIM_PATTERN = re.compile(
    r"\bhorizon\b[^\d\-]{0,12}(-?\d+(?:\.\d+)?)"
    r"|"
    r"\b(?:prediction|forecast|output)\s+(?:window|steps?|points?)\b[^\d\-]{0,12}(-?\d+(?:\.\d+)?)"
    r"|"
    r"\b(?:predict|forecast|generat(?:e|ing))\s+(-?\d+(?:\.\d+)?)\s*\b(?:steps?|points?)\b"
    r"|"
    r"\b(?:window|steps?|points?)\s+(?:of|is)[^\d\-]{0,8}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BASELINE_CLAIM_PATTERN = re.compile(
    r"\b(?:baseline)\b[^\d\-]{0,16}(-?\d+(?:\.\d+)?)"
    r"|"
    r"\breference\s+(?:(?:mean|avg|value|level|last|final)\b)[^\d\-]{0,12}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _summarize_series(values: List[float]) -> List[float]:
    clean = [_safe_float(v) for v in values]
    clean = [c for c in clean if c is not None]
    if not clean:
        return []
    clean_floats = [float(c) for c in clean]
    mean_val = sum(clean_floats) / len(clean_floats)
    return [
        clean_floats[0],
        clean_floats[-1],
        min(clean_floats),
        max(clean_floats),
        mean_val,
    ]


def _collect_numeric_context(
    predictions: List[float],
    investor_packet: Dict[str, Any],
    predicted_window: int,
    window_offset: int,
) -> List[float]:
    context: List[float] = []

    for value in predictions:
        safe_val = _safe_float(value)
        if safe_val is not None:
            context.append(safe_val)

    context.extend(
        [
            float(predicted_window),
            float(window_offset),
        ]
    )

    reference = investor_packet.get("reference_prediction")
    if isinstance(reference, list) and reference:
        context.extend(_summarize_series(reference))
        diffs: List[float] = []
        for pred_val, ref_val in zip(predictions, reference):
            pred_f = _safe_float(pred_val)
            ref_f = _safe_float(ref_val)
            if pred_f is not None and ref_f is not None:
                diffs.append(pred_f - ref_f)
        if diffs:
            context.extend(_summarize_series(diffs))

    def _walk(value: Any, seen: set[int]) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            safe_val = _safe_float(value)
            if safe_val is not None:
                context.append(safe_val)
            return
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
            try:
                _walk(value.tolist(), seen)
                return
            except Exception:
                return
        if isinstance(value, dict):
            for item in value.values():
                _walk(item, seen)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item, seen)

    _walk(investor_packet, set())

    deduped: List[float] = []
    seen_values: set[tuple[int, int]] = set()
    for val in context:
        safe_val = _safe_float(val)
        if safe_val is None:
            continue
        signature = (round(safe_val, 6), round(safe_val, 2))
        if signature in seen_values:
            continue
        seen_values.add(signature)
        deduped.append(safe_val)
        if len(deduped) >= 6000:
            break

    return deduped


def _looks_like_date_fragment(text: str, start: int, end: int) -> bool:
    before = text[start - 1 : start] if start > 0 else ""
    after = text[end : end + 1] if end < len(text) else ""
    after_next_digit = end + 1 < len(text) and text[end + 1].isdigit()
    before_prev_digit = start - 2 >= 0 and text[start - 2].isdigit()
    if (after == "-" and after_next_digit) or (before == "-" and before_prev_digit):
        return True
    if (after == ":" and after_next_digit) or (before == ":" and before_prev_digit):
        return True
    # Detect YYYY-MM-DD style sequences
    date_slice = text[start : min(len(text), start + 10)]
    if re.match(r"\d{4}-\d{2}-\d{2}", date_slice):
        return True
    return False


def _extract_numbers_from_text(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not text:
        return results
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(0)
        number = match.group("number")
        cleaned = number.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        start, end = match.span()
        if _looks_like_date_fragment(text, start, end):
            continue
        results.append(
            {
                "raw": raw,
                "value": value,
                "is_percent": bool(match.group("percent")),
                "start": start,
                "end": end,
            }
        )
    return results


def _is_supported_numeric_claim(value: float, context_numbers: List[float]) -> bool:
    if not context_numbers:
        return False
    for ctx in context_numbers:
        tolerance = max(0.02 * max(abs(value), abs(ctx), 1.0), 0.5)
        if abs(value - ctx) <= tolerance:
            return True
    return False


def scan_chain_of_thought(
    predictions: List[float],
    predicted_window: int,
    investor_packet: Dict[str, Any],
    chain_of_thought: str,
    window_offset: int,
) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        "total_numbers": 0,
        "unsupported_numbers": [],
        "window_claim_mismatches": [],
        "baseline_claim_mismatches": [],
        "context_size": 0,
        "flagged": False,
        "summary": "",
        "window_offset": window_offset,
    }

    text = chain_of_thought or ""
    stripped = text.strip()
    if not stripped:
        analysis["summary"] = "No chain-of-thought text provided; numeric audit skipped."
        return analysis

    numeric_tokens = _extract_numbers_from_text(text)
    analysis["total_numbers"] = len(numeric_tokens)

    context_numbers = _collect_numeric_context(
        predictions,
        investor_packet,
        predicted_window,
        window_offset,
    )
    analysis["context_size"] = len(context_numbers)

    if not context_numbers:
        analysis["summary"] = (
            "Unable to cross-check numeric claims because investor packet lacked numeric context."
        )
        analysis["flagged"] = True
        return analysis

    context_magnitude = max((abs(num) for num in context_numbers if num is not None), default=1.0)
    unsupported: List[Dict[str, Any]] = []
    for token in numeric_tokens:
        if token["is_percent"]:
            continue
        value = float(token["value"])
        # Skip year-like integers (1900-2099) — common in time-series reasoning
        if value == int(value) and 1900 <= int(value) <= 2099:
            continue
        if abs(value) <= max(0.05 * context_magnitude, 1.0) and context_magnitude > 10:
            # Allow small housekeeping numbers when the context scale is large.
            continue
        if _is_supported_numeric_claim(value, context_numbers):
            continue
        snippet = text[max(0, token["start"] - 40) : min(len(text), token["end"] + 40)].strip()
        unsupported.append(
            {
                "raw": token["raw"],
                "value": value,
                "snippet": snippet,
            }
        )

    analysis["unsupported_numbers"] = unsupported

    window_mismatches: List[Dict[str, Any]] = []
    for match in _WINDOW_CLAIM_PATTERN.finditer(text):
        raw_claimed = next((g for g in match.groups() if g is not None), None)
        if raw_claimed is None:
            continue
        try:
            claimed = float(raw_claimed)
        except (TypeError, ValueError):
            continue
        if abs(claimed - predicted_window) > 0.5:
            snippet = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)].strip()
            window_mismatches.append(
                {
                    "raw": match.group(0).strip(),
                    "claimed_window": claimed,
                    "expected_window": predicted_window,
                    "snippet": snippet,
                }
            )

    analysis["window_claim_mismatches"] = window_mismatches

    reference = investor_packet.get("reference_prediction")
    baseline_mismatches: List[Dict[str, Any]] = []
    baseline_stats: Dict[str, float] = {}
    if isinstance(reference, list):
        clean_reference = [_safe_float(val) for val in reference]
        clean_reference = [val for val in clean_reference if val is not None]
        if clean_reference:
            mean_val = sum(clean_reference) / len(clean_reference)
            last_val = clean_reference[-1]
            baseline_stats = {"mean": mean_val, "last": last_val}

    if baseline_stats:
        for match in _BASELINE_CLAIM_PATTERN.finditer(text):
            raw_claimed = next((g for g in match.groups() if g is not None), None)
            if raw_claimed is None:
                continue
            snippet = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)].strip()
            try:
                claimed = float(raw_claimed)
            except (TypeError, ValueError):
                continue
            reference_type = "mean"
            snippet_lower = snippet.lower()
            if "last" in snippet_lower or "final" in snippet_lower:
                reference_type = "last"
            target = baseline_stats.get(reference_type, baseline_stats["mean"])
            tolerance = max(0.05 * max(abs(target), 1.0), 0.5)
            if abs(claimed - target) > tolerance:
                baseline_mismatches.append(
                    {
                        "raw": match.group(0).strip(),
                        "claimed": claimed,
                        "reference_type": reference_type,
                        "reference_value": target,
                        "snippet": snippet,
                    }
                )

    analysis["baseline_claim_mismatches"] = baseline_mismatches

    summary_parts: List[str] = []
    if unsupported:
        summary_parts.append(f"{len(unsupported)} unsupported numeric claim(s)")
    if window_mismatches:
        summary_parts.append(f"{len(window_mismatches)} horizon mismatch(es)")
    if baseline_mismatches:
        summary_parts.append(f"{len(baseline_mismatches)} baseline contradiction(s)")

    if summary_parts:
        analysis["flagged"] = True
        analysis["summary"] = "Chain-of-thought audit: " + "; ".join(summary_parts)
    else:
        analysis["summary"] = (
            "Chain-of-thought numeric claims align with investor packet context."
        )

    return analysis


def create_reflector_agent(
    assess_forecast: Callable[[List[float], int, Dict[str, Any], str], Dict[str, Any]],
    json_default: Callable[[Any], Any],
) -> Agent:
    instructions = get_agent_instructions("ReflectorAgent", REFLECTOR_AGENT_PROMPT_FALLBACK)

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

    def _reflector_model(messages, agent_info) -> ModelResponse:
        payload = _extract_json_request(messages)
        raw_predictions = payload.get("predictions") or []
        predictions: List[float] = []
        for value in raw_predictions:
            try:
                predictions.append(float(value))
            except Exception:
                continue
        try:
            predicted_window = int(
                payload.get("predicted_window", len(predictions)) or len(predictions)
            )
        except Exception:
            predicted_window = len(predictions)
        investor_packet = payload.get("investor_packet") or {}
        chain_of_thought = payload.get("chain_of_thought") or ""
        try:
            window_offset = int(payload.get("window_offset", 0) or 0)
        except Exception:
            window_offset = 0
        report = assess_forecast(
            predictions,
            predicted_window,
            investor_packet,
            chain_of_thought,
        )
        analysis = scan_chain_of_thought(
            predictions,
            predicted_window,
            investor_packet,
            chain_of_thought,
            window_offset,
        )

        diagnostics: Dict[str, Any] = report.setdefault("diagnostics", {})
        diagnostics["chain_of_thought"] = analysis

        detected_issues: List[str] = []
        if analysis["unsupported_numbers"]:
            samples = ", ".join(
                item["raw"] for item in analysis["unsupported_numbers"][:3]
            )
            detected_issues.append(
                f"Chain-of-thought numeric claims lack grounding: {samples}"
            )
        if analysis["window_claim_mismatches"]:
            samples = ", ".join(
                item["raw"] for item in analysis["window_claim_mismatches"][:2]
            )
            detected_issues.append(
                f"Chain-of-thought horizon references contradict request: {samples}"
            )
        if analysis["baseline_claim_mismatches"]:
            samples = ", ".join(
                item["raw"] for item in analysis["baseline_claim_mismatches"][:2]
            )
            detected_issues.append(
                f"Chain-of-thought baseline stats contradict investor packet: {samples}"
            )

        if detected_issues:
            issues = report.setdefault("issues", [])
            issues.extend(detected_issues)
            report["approved"] = False

        summary_note = analysis.get("summary")
        if summary_note:
            existing_note = report.get("notes")
            report["notes"] = f"{existing_note}; {summary_note}" if existing_note else summary_note

        if payload.get("window_offset") is not None:
            report.setdefault("window_offset", window_offset)
        return ModelResponse(
            parts=[TextPart(json.dumps(report, default=json_default))],
            model_name="function:reflector",
        )

    return Agent(
        FunctionModel(function=_reflector_model),
        instructions=instructions,
    )


__all__ = ["create_reflector_agent", "REFLECTOR_AGENT_PROMPT_FALLBACK", "scan_chain_of_thought"]
