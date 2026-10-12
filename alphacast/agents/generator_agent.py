from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic_ai import Agent, RunContext  # type: ignore

from alphacast.config import DatasetConfig, ExperimentConfig
from alphacast.data_loader import TIME_COL
from .prompts import get_agent_instructions


GENERATOR_AGENT_PROMPT_FALLBACK = dedent(
    """
    You are GeneratorAgent, a world-class time-series forecasting expert operating in a multi-agent workflow.
    Each forecasting step must follow this sequence:
      1. Call `consult` exactly once to obtain the InvestigatorAgent research packet for the requested dataset/window.
      2. Examine the packet carefully. Use `reference_prediction` as the baseline, consult neighbor guidance and exogenous trends, and only adjust the baseline when evidence clearly supports a correction.
      3. Record a brief "Reflection" that confirms the prediction length equals `predicted_window`, all pending `emit_predictions` arguments are correct (including window_offset), and the forecast aligns with the baseline guidance and exogenous outlook.
      4. Call `record_chain_of_thought` exactly once with dataset_name, window_offset, and a concise summary referencing the evidence and any adjustments (or the decision to keep the baseline).
      5. Call `emit_predictions` exactly once with the prediction list and required metadata (training_csv, predicted_window, output_dir, dataset_name, frequency, window_offset, start_timestamp, selected_features, feature_weights, and optional exogenous selections).

    Use no tools other than `consult`, `record_chain_of_thought`, and `emit_predictions`.
    """
)


def create_generator_agent(
    model_name: str,
    cfg: ExperimentConfig | None,
    dataset_lookup: Dict[str, DatasetConfig],
    briefing_lookup: Dict[str, str],
    prepare_investor_packet: Callable[[ExperimentConfig | None, DatasetConfig, Dict[str, str], int, Optional[int]], dict],
    json_default: Callable[[Any], Any],
    reflector_agent: Agent,
    deterministic_run_for_dataset: Callable[[ExperimentConfig, Any], dict],
) -> Agent:
    instructions = get_agent_instructions("GeneratorAgent", GENERATOR_AGENT_PROMPT_FALLBACK)
    generator_agent = Agent(model_name, instructions=instructions)
    globals()["RunContext"] = RunContext

    investigator_cache: dict[tuple[str, int], dict[str, Any]] = {}
    chain_cache: dict[tuple[str, int], str] = {}

    def _dataset_out_dir(name: str) -> str:
        base_dir = cfg.output_dir if cfg else "outputs"
        return os.path.join(base_dir, name)

    def _append_chain_log(dataset_name: str, window_offset: int, content: str) -> str:
        ds_out_dir = _dataset_out_dir(dataset_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        path = os.path.join(ds_out_dir, "chain_of_thought.log")
        entry = {
            "window_offset": int(window_offset),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "content": content,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    @generator_agent.tool
    def consult(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: Optional[int] = None,
    ) -> dict:
        ds_cfg = dataset_lookup.get(dataset_name)
        if ds_cfg is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
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
        investigator_cache[(dataset_name, window_offset_int)] = packet
        return packet

    @generator_agent.tool
    def record_chain_of_thought(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int,
        summary: str,
    ) -> dict:
        try:
            window_offset_int = int(window_offset)
        except Exception:
            window_offset_int = int(window_offset or 0)
        path = _append_chain_log(dataset_name, window_offset_int, summary)
        chain_cache[(dataset_name, window_offset_int)] = summary
        return {"logged": True, "path": path}

    @generator_agent.tool
    def emit_predictions(
        ctx: RunContext[None],
        predictions: List[float],
        training_csv: str,
        predicted_window: int,
        output_dir: str,
        dataset_name: str,
        window_offset: Optional[int] = None,
        frequency: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        feature_weights: Optional[dict] = None,
        exogenous_vars: Optional[List[str]] = None,
        exogenous_feature_selection: Optional[dict] = None,
        exogenous_correlations: Optional[dict] = None,
    ) -> dict:
        H = int(predicted_window)
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
        n = len(predictions)
        if n == 0:
            raise ValueError("predictions must be a non-empty list of floats")
        if n != H:
            print(f"[warn] Normalizing LLM predictions length from {n} to {H} by {'trimming' if n>H else 'padding'}")
            if n > H:
                predictions = predictions[:H]
            else:
                last_val = float(predictions[-1])
                predictions = predictions + [last_val] * (H - n)
        arr = np.asarray(predictions, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("predictions must be finite numbers")

        ds_cfg = dataset_lookup.get(dataset_name)
        investor_packet = investigator_cache.get((dataset_name, window_offset_int))
        if investor_packet is None and ds_cfg is not None:
            try:
                investor_packet = prepare_investor_packet(
                    cfg,
                    ds_cfg,
                    briefing_lookup,
                    window_offset_int,
                    H,
                )
            except Exception:
                investor_packet = {}
        elif investor_packet is None:
            investor_packet = {}

        chain_text = chain_cache.get((dataset_name, window_offset_int), "")
        reflection_request = {
            "dataset_name": dataset_name,
            "window_offset": window_offset_int,
            "predictions": arr.tolist(),
            "predicted_window": H,
            "investor_packet": investor_packet or {},
            "chain_of_thought": chain_text,
        }
        reflection_result = reflector_agent.run_sync(json.dumps(reflection_request, default=json_default))
        try:
            reflection = json.loads(reflection_result.output)
        except Exception as exc:
            raise RuntimeError(f"ReflectorAgent returned invalid payload: {exc}") from exc
        if not reflection.get("approved", False):
            issues = reflection.get("issues") or []
            notes = reflection.get("notes") or ""
            joined = ", ".join(str(item) for item in issues)
            raise RuntimeError(f"ReflectorAgent rejected forecast: {joined} {notes}")
        try:
            with open(os.path.join(_dataset_out_dir(dataset_name), "reflector_report.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"window_offset": window_offset_int, **reflection}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        ds_out_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        out_csv = os.path.join(ds_out_dir, "predictions.csv")

        feat_path = os.path.join(ds_out_dir, "features.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                _feat = json.load(f)
            if isinstance(_feat, dict) and _feat:
                available_features = [str(k) for k in _feat.keys()]
                desired_count = min(3, len(available_features)) if len(available_features) >= 3 else len(available_features)

                provided = []
                if isinstance(selected_features, list):
                    provided = [str(name) for name in selected_features if str(name) in available_features]

                auto_added = False
                if len(provided) < desired_count:
                    for name in available_features:
                        if name not in provided:
                            provided.append(name)
                        if len(provided) >= desired_count:
                            break
                    auto_added = True

                cleaned_weights: dict[str, float] = {}
                if isinstance(feature_weights, dict):
                    for key, value in feature_weights.items():
                        if key in provided:
                            try:
                                cleaned_weights[key] = float(value)
                            except Exception:
                                continue

                if len(cleaned_weights) != len(provided) or sum(cleaned_weights.values()) <= 0:
                    weight = 1.0 / len(provided) if provided else 0.0
                    cleaned_weights = {name: weight for name in provided} if provided else {}
                else:
                    total = float(sum(cleaned_weights.values())) or 1.0
                    cleaned_weights = {name: float(val) / total for name, val in cleaned_weights.items()}

                if auto_added:
                    print(
                        f"[info] Auto-filled target features for dataset '{dataset_name}': {provided}"
                    )

                selected_features = provided
                feature_weights = cleaned_weights

        exo_top3_path = os.path.join(ds_out_dir, "exogenous_top3.json")
        exo_feat_path = os.path.join(ds_out_dir, "exogenous_features.json")
        exo_corr_path = os.path.join(ds_out_dir, "exogenous_correlations.json")
        if (
            getattr(cfg, "use_exogenous", False)
            and os.path.exists(exo_top3_path)
            and os.path.exists(exo_feat_path)
        ):
            with open(exo_top3_path, "r", encoding="utf-8") as f:
                _top3 = json.load(f)
            with open(exo_feat_path, "r", encoding="utf-8") as f:
                _exo_feats = json.load(f)
            _exo_corrs = {}
            if os.path.exists(exo_corr_path):
                with open(exo_corr_path, "r", encoding="utf-8") as f:
                    _exo_corrs = json.load(f)

            if not isinstance(exogenous_vars, list):
                exogenous_vars = []
            vars_clean = [str(v) for v in exogenous_vars if str(v) in _top3]
            auto_vars = False
            if len(vars_clean) != len(_top3):
                vars_clean = list(_top3)
                auto_vars = True

            dims_clean: dict[str, list[str]] = {}
            if isinstance(exogenous_feature_selection, dict):
                for var, dims in exogenous_feature_selection.items():
                    if var in _top3 and isinstance(dims, list):
                        dims_clean[var] = [str(d) for d in dims if str(d) in (_exo_feats.get(var) or {})]

            auto_dims = False
            for var in _top3:
                allowed = list((_exo_feats.get(var) or {}).keys())
                original = list(dims_clean.get(var, []))
                selected = list(dims_clean.get(var, []))
                for name in allowed:
                    if name not in selected:
                        selected.append(name)
                    if len(selected) >= min(3, len(allowed)):
                        break
                if len(selected) < min(3, len(allowed)):
                    selected = allowed[: min(3, len(allowed))]
                dims_clean[var] = selected
                if selected != original:
                    auto_dims = True

            corr_clean: dict[str, float] = {}
            if isinstance(exogenous_correlations, dict):
                for var, val in exogenous_correlations.items():
                    if var in _top3:
                        try:
                            corr_clean[var] = float(val)
                        except Exception:
                            continue
            for var in _top3:
                if var not in corr_clean:
                    base_val = (_exo_corrs or {}).get(var, 0.0)
                    try:
                        corr_clean[var] = float(base_val)
                    except Exception:
                        corr_clean[var] = 0.0

            if auto_vars or auto_dims:
                dims_preview = {var: dims_clean.get(var, []) for var in _top3}
                print(
                    f"[info] Auto-filled exogenous selections for dataset '{dataset_name}': vars={vars_clean}, dims={dims_preview}"
                )

            exogenous_vars = vars_clean
            exogenous_feature_selection = dims_clean
            exogenous_correlations = corr_clean

        features_used_note = None
        try:
            if selected_features is not None or feature_weights is not None:
                names = selected_features if isinstance(selected_features, list) else []
                top_items = []
                if isinstance(feature_weights, dict) and feature_weights:
                    items = [
                        (str(k), float(v))
                        for k, v in feature_weights.items()
                        if v is not None and np.isfinite(float(v))
                    ]
                    items.sort(key=lambda kv: kv[1], reverse=True)
                    top_items = items[:10]
                pretty_weights = ", ".join([f"{k}:{v:.3f}" for k, v in top_items]) if top_items else ""
                features_used_note = f"selected=[{', '.join([str(n) for n in names])}]" + (f"; weights={pretty_weights}" if pretty_weights else "")
        except Exception:
            features_used_note = None

        parsed_start_ts: Optional[pd.Timestamp] = None
        if start_timestamp is not None:
            try:
                parsed_start_ts = pd.to_datetime(start_timestamp)
            except Exception:
                print(
                    f"[warn] Invalid start_timestamp '{start_timestamp}' for dataset '{dataset_name}'. Falling back to automatic anchoring."
                )
                parsed_start_ts = None

        freq_clean: Optional[str] = None
        if isinstance(frequency, str):
            _f = frequency.strip()
            if _f.lower() in {"h", "d", "w", "m", "s"}:
                freq_clean = _f.upper()
            else:
                freq_clean = _f

        existing_df: Optional[pd.DataFrame] = None
        existing_unique: Optional[pd.DataFrame] = None
        if os.path.exists(out_csv):
            existing_df = pd.read_csv(out_csv)
            if len(existing_df) > 0 and "time_stamp" in existing_df.columns:
                existing_df["time_stamp"] = pd.to_datetime(existing_df["time_stamp"])  # type: ignore
                existing_unique = (
                    existing_df.sort_values("time_stamp", kind="mergesort")
                    .drop_duplicates(subset=["time_stamp"], keep="last")
                    .reset_index(drop=True)
                )
            else:
                existing_df = None
                existing_unique = None

        if not freq_clean:
            try:
                tdf_try = pd.read_csv(training_csv)
                tdf_try[TIME_COL] = pd.to_datetime(tdf_try[TIME_COL])
                tdf_try = tdf_try.sort_values(TIME_COL).reset_index(drop=True)
                freq_guess = pd.infer_freq(tdf_try[TIME_COL])
            except Exception:
                freq_guess = None
            if not freq_guess and existing_unique is not None and len(existing_unique) > 1:
                try:
                    freq_guess = pd.infer_freq(existing_unique["time_stamp"])  # type: ignore
                except Exception:
                    freq_guess = None
            if isinstance(freq_guess, str) and freq_guess.lower() in {"h", "d", "w", "m", "s"}:
                freq_clean = freq_guess.upper()
            else:
                freq_clean = freq_guess

        timestamps: List[pd.Timestamp] = []
        if parsed_start_ts is not None and freq_clean:
            try:
                offset = pd.tseries.frequencies.to_offset(freq_clean)
                timestamps = [parsed_start_ts + offset * i for i in range(H)]
            except Exception as exc:
                print(
                    f"[warn] Failed to apply provided start_timestamp for dataset '{dataset_name}': {exc}. Using automatic anchoring."
                )
                timestamps = []
                parsed_start_ts = None

        if not timestamps:
            if existing_unique is not None and len(existing_unique) > 0:
                last_ts = existing_unique["time_stamp"].iloc[-1]
                try:
                    last_ts = pd.Timestamp(last_ts)
                    if freq_clean:
                        offset = pd.tseries.frequencies.to_offset(freq_clean)
                        timestamps = [last_ts + offset * (i + 1) for i in range(H)]
                except Exception:
                    timestamps = []
            if not timestamps:
                try:
                    tdf_try = pd.read_csv(training_csv)
                    tdf_try[TIME_COL] = pd.to_datetime(tdf_try[TIME_COL])
                    tdf_try = tdf_try.sort_values(TIME_COL).reset_index(drop=True)
                    last_ts = pd.Timestamp(tdf_try[TIME_COL].iloc[-1])
                    inferred = pd.infer_freq(tdf_try[TIME_COL])
                    if isinstance(inferred, str) and inferred.lower() in {"h", "d", "w", "m", "s"}:
                        inferred = inferred.upper()
                    offset = pd.tseries.frequencies.to_offset(freq_clean or inferred or "D")
                    timestamps = [last_ts + offset * (i + 1) for i in range(H)]
                except Exception:
                    timestamps = []

        if not timestamps:
            raise RuntimeError(f"Unable to infer timestamps for dataset '{dataset_name}'.")

        if existing_df is not None and "emission_index" in existing_df.columns:
            try:
                start_sequence = int(pd.to_numeric(existing_df["emission_index"], errors="coerce").max(skipna=True) or -1) + 1
            except Exception:
                start_sequence = int(existing_df.shape[0])
        elif existing_df is not None:
            start_sequence = int(existing_df.shape[0])
        else:
            start_sequence = 0

        new_chunk = pd.DataFrame(
            {
                "time_stamp": pd.to_datetime(timestamps),
                "prediction": arr.tolist(),
                "window_offset": window_offset_int,
                "horizon_index": list(range(H)),
                "emission_index": np.arange(start_sequence, start_sequence + H, dtype=int),
            }
        )

        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_chunk], ignore_index=True)
        else:
            combined = new_chunk

        combined.to_csv(out_csv, index=False)

        meta = {
            "dataset": dataset_name,
            "chosen_model": "LLM",
            "frequency": freq_clean,
            "look_back": None,
            "predicted_window": H,
        }
        if parsed_start_ts is not None:
            meta["segment_start_timestamp"] = parsed_start_ts.isoformat()
        try:
            meta["features_used"] = {
                "selected_features": selected_features if isinstance(selected_features, list) else [],
                "feature_weights": feature_weights if isinstance(feature_weights, dict) else {},
            }
        except Exception:
            pass
        try:
            if exogenous_vars is not None or exogenous_feature_selection is not None or exogenous_correlations is not None:
                exo_sel = {
                    "exogenous_vars": exogenous_vars if isinstance(exogenous_vars, list) else [],
                    "exogenous_feature_selection": exogenous_feature_selection if isinstance(exogenous_feature_selection, dict) else {},
                    "exogenous_correlations": exogenous_correlations if isinstance(exogenous_correlations, dict) else {},
                }
                with open(os.path.join(ds_out_dir, "exogenous_selected_features.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_sel, f, indent=2)
                meta["exogenous_used"] = exo_sel
                print(f"[info] LLM reported exogenous usage for dataset '{dataset_name}': vars={exo_sel['exogenous_vars']}")
        except Exception:
            pass

        with open(os.path.join(ds_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if selected_features is not None or feature_weights is not None:
            sel = {"selected_features": selected_features, "feature_weights": feature_weights}
            with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=2)
            try:
                top_msg = features_used_note or ""
                print(f"[info] LLM reported feature usage for dataset '{dataset_name}': {top_msg}")
            except Exception:
                pass
        return {"ok": True, "predictions_saved": out_csv}

    @generator_agent.tool
    def run_pipeline(
        ctx: RunContext[None],
        training_csv: str,
        test_csv: str,
        look_back: int,
        predicted_window: int,
        output_dir: str,
        dataset_name: str,
    ) -> dict:
        ds_obj = dataset_lookup.get(dataset_name)
        if ds_obj is None:
            class _Ds:
                def __init__(self):
                    self.name = dataset_name
                    self.training_csv = training_csv
                    self.test_csv = test_csv
                    self.look_back = int(look_back)
                    self.predicted_window = int(predicted_window)
                    self.sliding_window = int(look_back)
                    self.frequency = None
                    self.checkpoints = {}
                    self.context_prompt_file = None

            ds_obj = _Ds()

        cfg_like = ExperimentConfig(datasets=[ds_obj], output_dir=output_dir)
        return deterministic_run_for_dataset(cfg_like, ds_obj)

    return generator_agent


__all__ = ["create_generator_agent", "GENERATOR_AGENT_PROMPT_FALLBACK"]
