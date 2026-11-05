from __future__ import annotations

import json
import os
import time
from pathlib import Path
from textwrap import dedent, indent
from typing import Dict, List, Optional
import tempfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alphacast.config import DatasetConfig, ExperimentConfig, load_config
from alphacast.data_loader import TIME_COL, infer_target_column
from alphacast.agents.runtime import (
    build_agent_or_none,
    clear_resume_state,
    deterministic_run_for_dataset,
    load_resume_state,
    save_resume_state,
)
from alphacast.eval import align_predictions, mae, mse, smape
from alphacast.tools.analysis import analyze_training
from alphacast.features import extract_target_features, extract_exogenous_features
from alphacast.utils.predictions import atomic_csv, discard_window, saved_window, window_rows


_CASE_BASE_FILES = (
    "memory.json", "case_base.json", "case_neighbor.json",
    "cluster_base.json", "cases_stats.json",
)


def _load_or_build_case_base(train_df, ds, output_dir, case_base_dir):
    destination = Path(output_dir) / ds.name
    # Prefer the initialization directory, then a previous experiment.
    # Do not compare training data or settings with the existing library.
    for folder in (Path(case_base_dir) / ds.name, destination):
        if not all((folder / name).is_file() for name in _CASE_BASE_FILES):
            continue
        contents = {name: json.loads((folder / name).read_text(encoding="utf-8"))
                    for name in _CASE_BASE_FILES}
        destination.mkdir(parents=True, exist_ok=True)
        if folder.resolve() != destination.resolve():
            for name, value in contents.items():
                _save_json(destination / name, value)
        return contents["memory.json"]
    return analyze_training(
        train_df, int(ds.look_back), int(ds.predicted_window), output_dir,
        ds.name, ds.sliding_window, method="weighted", num_clusters=6,
        dataset_cfg=ds,
    ).memory


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _complete_predictions(path, test_df, look_back, horizon, stride):
    frame = pd.read_csv(path)
    length = len(test_df) - look_back
    offsets = list(range(0, length, stride))
    if 'window_offset' not in frame or set(frame.window_offset) != set(offsets):
        raise ValueError('Forecast origins do not match the complete test protocol')
    chunks = []
    for offset in offsets:
        size = min(horizon, length - offset)
        timestamps = test_df[TIME_COL].iloc[offset + look_back:offset + look_back + size]
        chunks.append(window_rows(frame, offset, size, timestamps))
    clean = pd.concat(chunks, ignore_index=True)
    clean['time_stamp'] = pd.to_datetime(clean['time_stamp'], format='mixed')
    if len(clean) != len(frame):
        # Keep the original before repairing legacy duplicate submissions.
        backup = Path(str(path) + f'.before_dedup_{time.time_ns()}')
        backup.write_bytes(Path(path).read_bytes())
        atomic_csv(clean, path)
    return clean


def _load_dataset_brief(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        print(f"[warn] Failed to read dataset briefing '{path}': {exc}")
        return ""
    return text.strip()


def run_experiment(config_path: str, dataset_selectors: Optional[List[str]] = None,
                   case_base_dir: Optional[str] = None) -> None:
    # Load environment from .env if present
    load_dotenv(override=False)

    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    case_base_dir = case_base_dir or str(Path(__file__).resolve().parent / "outputs/case_bases")

    selectors = [s.strip().lower() for s in (dataset_selectors or []) if s]
    if selectors:
        alias_lookup: Dict[str, DatasetConfig] = {}
        for ds in cfg.datasets:
            for alias in ds.all_aliases():
                alias_lookup.setdefault(alias, ds)

        missing = [token for token in selectors if token not in alias_lookup]
        if missing:
            raise ValueError(f"Unknown dataset selector(s): {', '.join(missing)}")

        selected: List[DatasetConfig] = []
        seen_names = set()
        for token in selectors:
            ds = alias_lookup[token]
            if ds.name not in seen_names:
                selected.append(ds)
                seen_names.add(ds.name)

        if not selected:
            raise ValueError("No datasets resolved from provided selectors.")

        cfg.datasets = selected
        print(f"[info] Running datasets: {', '.join(ds.name for ds in cfg.datasets)}")
    
    dataset_briefings: Dict[str, str] = {
        ds.name: _load_dataset_brief(getattr(ds, "context_prompt_file", None))
        for ds in cfg.datasets
    }

    # Set up logfire
    # logfire.configure()
    # logfire.instrument_pydantic_ai()

    mode = os.getenv("ORCHESTRATION_MODE", "llm").lower()
    use_agent = mode == "llm"

    agent = build_agent_or_none(cfg, dataset_briefings, stop_after_emit=True) if use_agent else None
    if use_agent and agent is None:
        print("[warn] LLM orchestration unavailable or misconfigured; falling back to deterministic mode.")

    rows = []
    for ds in cfg.datasets:
        case_base_ready = False
        print(f"\n=== Processing dataset: {ds.name} ===")
        dataset_brief = dataset_briefings.get(ds.name, "")
        formatted_brief = ""
        if dataset_brief:
            formatted_brief = "Dataset briefing:\n" + indent(dataset_brief, "  ")
        try:
            train_df = pd.read_csv(ds.training_csv)
            train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL])
            train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
        except Exception as exc:
            print(f"[warn] Failed to load training data for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            rows.append(deterministic_run_for_dataset(cfg, ds, case_base_ready=case_base_ready))
            continue

        look_back = int(ds.look_back)
        predicted_window = int(ds.predicted_window)

        try:
            memory = _load_or_build_case_base(train_df, ds, cfg.output_dir, case_base_dir)
            case_base_ready = True
        except Exception as exc:
            print(f"[warn] Training analysis failed for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            rows.append(deterministic_run_for_dataset(cfg, ds, case_base_ready=case_base_ready))
            continue

        ds_out_dir = os.path.join(cfg.output_dir, ds.name)
        frequency = None
        if isinstance(memory, dict):
            frequency = memory.get("frequency")

        # Dynamically infer the target column for downstream feature computation
        target_col = infer_target_column(train_df, ds.name)
        y = train_df[target_col].to_numpy(dtype=float)
        sel_cfg = getattr(cfg, "feature_selection_override", None)

        features = {}
        if getattr(cfg, "use_features", True):
            try:
                features = extract_target_features(y, frequency)
                with open(os.path.join(ds_out_dir, "features.json"), "w", encoding="utf-8") as f:
                    json.dump(features, f, indent=2)
                if isinstance(sel_cfg, dict):
                    with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                        json.dump(sel_cfg, f, indent=2)
            except Exception as exc:
                features = {}
                print(f"[warn] Failed to compute target features for dataset '{ds.name}': {exc}")

        if features:
            try:
                basic_keys = [
                    "basic_count",
                    "basic_mean",
                    "basic_std",
                    "basic_min",
                    "basic_max",
                    "basic_skew",
                    "basic_kurt",
                ]
                print(f"- Frequency: {frequency} | Target feature count: {len(features)}")
                for key in basic_keys:
                    if key in features:
                        try:
                            print(f"- {key}: {float(features[key]):.6g}")
                        except Exception:
                            print(f"- {key}: {features[key]}")
                extras = [
                    k
                    for k in features.keys()
                    if k not in set(basic_keys + ["spectral_entropy", "seasonal_strength"])
                ]
                if "spectral_entropy" in features:
                    try:
                        print(f"- spectral_entropy: {float(features['spectral_entropy']):.6g}")
                    except Exception:
                        print(f"- spectral_entropy: {features['spectral_entropy']}")
                if "seasonal_strength" in features:
                    try:
                        print(f"- seasonal_strength: {float(features['seasonal_strength']):.6g}")
                    except Exception:
                        print(f"- seasonal_strength: {features['seasonal_strength']}")
                if extras:
                    extras = sorted(extras)
                    preview = ", ".join(extras[:20]) + (" ..." if len(extras) > 20 else "")
                    print(f"- Other feature keys: {preview}")
            except Exception:
                pass

        exo_features = {}
        exo_corr = {}
        exo_top3 = []
        exo_columns = {}
        if getattr(cfg, "use_exogenous", False):
            try:
                (
                    exo_features,
                    exo_corr,
                    exo_top3,
                    exo_columns,
                ) = extract_exogenous_features(
                    train_df, target_col, ds.name, frequency
                )
                with open(os.path.join(ds_out_dir, "exogenous_features.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_features, f, indent=2)
                with open(
                    os.path.join(ds_out_dir, "exogenous_correlations.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(exo_corr, f, indent=2)
                with open(os.path.join(ds_out_dir, "exogenous_top3.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_top3, f, indent=2)
                with open(os.path.join(ds_out_dir, "exogenous_columns.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_columns, f, indent=2)

                print(
                    f"- Exogenous variables enabled. Top-3: {', '.join(exo_top3) if exo_top3 else 'None'}"
                )
                if exo_corr:
                    ordered = sorted(exo_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)
                    for name, val in ordered[:10]:
                        try:
                            print(f"  · corr({name}) = {float(val):.4f}")
                        except Exception:
                            print(f"  · corr({name}) = {val}")
            except Exception as exc:
                exo_features, exo_corr, exo_top3, exo_columns = {}, {}, [], {}
                print(f"[warn] Exogenous variable processing failed for dataset '{ds.name}': {exc}")

        try:
            test_df = pd.read_csv(ds.test_csv)
            test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
            test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)
        except Exception as exc:
            print(f"[warn] Failed to load test data for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            rows.append(deterministic_run_for_dataset(cfg, ds, case_base_ready=case_base_ready))
            continue

        target_df = test_df.iloc[look_back:].reset_index(drop=True)

        out_csv = os.path.join(ds_out_dir, "predictions.csv")
        agent_success = False
        resume_required = False

        if agent is not None:
            old_metrics = Path(ds_out_dir) / 'evaluation_metrics.json'
            if old_metrics.exists():
                old_metrics.replace(Path(str(old_metrics) + f'.previous_{time.time_ns()}'))
            _save_json(Path(ds_out_dir) / 'evaluation_status.json',
                       dict(status='running', dataset=ds.name))
            resume_state = load_resume_state(ds_out_dir)
            if resume_state is None and os.path.exists(out_csv):
                os.remove(out_csv)

            total_len = len(target_df)
            if total_len == 0:
                print(
                    f"[info] Dataset '{ds.name}' has no forecast horizon after applying look_back; skipping LLM orchestration."
                )
                agent_success = True
            else:
                horizon = predicted_window
                stride = ds.sliding_window
                training_literal = json.dumps(ds.training_csv)
                output_literal = json.dumps(cfg.output_dir)
                dataset_literal = json.dumps(ds.name)
                max_net_failures = 3
                max_other_failures = 3

                def _is_network_error(ex: Exception) -> bool:
                    txt = str(ex).lower()
                    return any(
                        marker in txt
                        for marker in [
                            "timeout",
                            "timed out",
                            "connection",
                            "network",
                            "temporarily unavailable",
                            "connection reset",
                            "dns",
                            "host unreachable",
                            "429",
                            "502",
                            "503",
                            "504",
                        ]
                    )

                planned = {off + j for off in range(0, total_len, stride)
                           for j in range(min(horizon, total_len - off))}
                total_needed = len(planned)
                current_len = 0
                current_collected = 0
                step_index = 0

                resume_state_valid = False
                if resume_state is not None:
                    try:
                        same_total = int(resume_state.get("total_len", total_len)) == total_len
                        same_stride = int(resume_state.get("stride", stride)) == stride
                        same_horizon = int(resume_state.get("horizon", horizon)) == horizon
                        if same_total and same_stride and same_horizon:
                            resume_state_valid = True
                        else:
                            print(
                                f"[warn] Ignoring stored resume state for dataset '{ds.name}' due to configuration mismatch; starting fresh."
                            )
                    except Exception:
                        pass

                if resume_state_valid:
                    # Scan from the first origin; complete saved windows are reused.
                    # This repairs stale counters without repeating successful API calls.
                    current_len = 0
                    step_index = 0
                    current_collected = 0
                    if os.path.exists(out_csv):
                        try:
                            _existing = pd.read_csv(out_csv)
                            if "time_stamp" in _existing.columns and len(_existing) > 0:
                                _existing["time_stamp"] = pd.to_datetime(_existing["time_stamp"], format='mixed')
                                existing_unique = (
                                    _existing.sort_values("time_stamp", kind="mergesort")
                                    .drop_duplicates(subset=["time_stamp"], keep="last")
                                )
                                unique_len = len(existing_unique)
                            else:
                                unique_len = len(_existing)
                            current_collected = min(unique_len, current_collected or unique_len)
                        except Exception:
                            pass
                    current_collected = min(current_collected, total_needed)
                    print(
                        f"[info] Resuming LLM orchestration for dataset '{ds.name}' from step {step_index} with {current_collected}/{total_needed} predictions already collected."
                    )
                else:
                    if resume_state is not None:
                        clear_resume_state(ds_out_dir)
                    current_len = 0
                    current_collected = 0
                    step_index = 0
                    if os.path.exists(out_csv):
                        os.remove(out_csv)

                llm_failed = False
                early_stop_due_to_bounds = False

                while current_len < total_len:
                    step_horizon = min(horizon, total_len - current_len)
                    window_offset = current_len

                    prompt_sections: List[str] = []
                    if formatted_brief:
                        prompt_sections.append(formatted_brief)
                    prompt_sections.append(
                        dedent(
                            f"""
                            You are forecasting the dataset {ds.name} (step {step_index}).

                            Step configuration:
                              - window_offset: {window_offset}
                              - look_back length: {look_back}
                              - remaining targets: {total_needed - current_collected}
                              - forecast horizon for this step: {step_horizon}

                            Required actions:
                              1. Call tool.consult exactly once with dataset_name={dataset_literal}, window_offset={window_offset}, forecast_horizon={step_horizon} to fetch the InvestigatorAgent packet.
                              2. Analyse the packet: anchor on `reference_prediction`, compare it to neighbor hints and exogenous trends, and decide whether a careful adjustment is justified. Capture the main signals in a short internal plan.
                              3. Before emitting predictions, write a brief "Reflection" confirming the prediction list will have length {step_horizon}, that every argument you will pass to tool.emit_predictions (dataset_name, output_dir, predicted_window, window_offset, start_timestamp, selected_features, feature_weights, exogenous selections) is correct, and that the forecast stays consistent with the baseline guidance and exogenous outlook.
                              4. Log the reasoning by calling tool.record_chain_of_thought exactly once with dataset_name={dataset_literal}, window_offset={window_offset}, and a concise reasoning summary referencing the evidence and any adjustments (or the decision to keep the baseline).
                              5. Call tool.emit_predictions exactly once with:
                                   - predictions: a list of {step_horizon} floats,
                                   - training_csv: {training_literal},
                                   - predicted_window: {step_horizon},
                                   - output_dir: {output_literal},
                                   - dataset_name: {dataset_literal},
                                   - window_offset: {window_offset},
                                   - frequency: reuse the 'frequency' field from the context (use null if missing),
                                   - start_timestamp: reuse 'prediction_start_timestamp' from the context when provided,
                                   - selected_features: choose ≥3 feature names from the provided dictionary when available (use [] if none),
                                   - feature_weights: non-negative weights for those features that sum to 1.0 (use {{}} if none),
                                   - exogenous_vars / exogenous_feature_selection / exogenous_correlations: when exogenous context is present, echo the listed variables, pick ≥3 dimension names per variable, and report the provided correlations.

                            Rules:
                              - Only use consult, record_chain_of_thought, and emit_predictions.
                              - Treat this step independently; rely only on the context you just loaded.
                            """
                        ).strip()
                    )
                    prompt = "\n\n".join(prompt_sections)

                    net_failures = 0
                    other_failures = 0
                    expected_times = test_df[TIME_COL].iloc[
                        window_offset + look_back:window_offset + look_back + step_horizon
                    ]
                    while True:
                        # A valid committed window survives errors after writing and process restarts.
                        if saved_window(out_csv, window_offset, step_horizon, expected_times) is not None:
                            break
                        discard_window(out_csv, window_offset)
                        try:
                            agent.run_sync(prompt)
                            if saved_window(out_csv, window_offset, step_horizon, expected_times) is None:
                                raise ValueError("Agent ended without a complete valid forecast")
                            break
                        except Exception as exc:
                            # Includes the local ForecastCommitted stop signal; verify the file first.
                            if saved_window(out_csv, window_offset, step_horizon, expected_times) is not None:
                                print(f"[info] Dataset '{ds.name}' step {step_index}: committed forecast recovered.")
                                break
                            discard_window(out_csv, window_offset)
                            if _is_network_error(exc) and net_failures < max_net_failures:
                                net_failures += 1
                                print(f"[warn] Network error for '{ds.name}' step {step_index}; retry {net_failures}/{max_net_failures}.")
                                time.sleep(1.0)
                                continue
                            if not _is_network_error(exc) and other_failures < max_other_failures:
                                other_failures += 1
                                print(f"[warn] Invalid/incomplete forecast for '{ds.name}' step {step_index}; retry {other_failures}/{max_other_failures}: {exc}")
                                time.sleep(1.0)
                                continue
                            print(f"[warn] LLM orchestration failed for '{ds.name}' step {step_index}: {exc}")
                            llm_failed = True
                            resume_required = True
                            _save_json(Path(ds_out_dir) / 'evaluation_status.json',
                                       dict(status='incomplete', dataset=ds.name,
                                            window_offset=window_offset, error=str(exc)))
                            break

                    if llm_failed:
                        break

                    if not os.path.exists(out_csv):
                        print(
                            f"[warn] LLM did not emit predictions for dataset '{ds.name}' (step {step_index})."
                        )
                        llm_failed = True
                        resume_required = True
                        break

                    pred_df_full = pd.read_csv(out_csv)
                    if "time_stamp" not in pred_df_full.columns or len(pred_df_full) == 0:
                        print(
                            f"[warn] Predictions file for dataset '{ds.name}' is empty or missing 'time_stamp' column after step {step_index}."
                        )
                        llm_failed = True
                        resume_required = True
                        break

                    pred_df_full["time_stamp"] = pd.to_datetime(pred_df_full["time_stamp"], format='mixed')
                    unique_pred_df = (
                        pred_df_full.sort_values("time_stamp", kind="mergesort")
                        .drop_duplicates(subset=["time_stamp"], keep="last")
                        .reset_index(drop=True)
                    )
                    unique_count = len(unique_pred_df)
                    if unique_count > total_needed:
                        current_collected = int(total_needed)
                        early_stop_due_to_bounds = True
                        print(
                            f"[info] Dataset '{ds.name}': unique forecast coverage reached {total_needed}; proceeding to evaluation with full emission history retained."
                        )
                        break
                        
                    base_path = Path(ds_out_dir) / "basemodel_results.json"
                    if base_path.exists():
                        basemodel_results = json.loads(base_path.read_text()) or []
                        if basemodel_results and isinstance(basemodel_results[-1], dict):
                            basemodel_results[-1].setdefault(
                                'start_timestamp', expected_times.iloc[0].isoformat()
                            )
                            _save_json(base_path, basemodel_results)

                    new_collected = unique_count
                    # Per-origin validation above is authoritative. A reused or overlapping
                    # window may legitimately add no new unique timestamps.
                    current_collected = new_collected
                    current_len += stride
                    step_index += 1

                    save_resume_state(
                        ds_out_dir,
                        {
                            "current_collected": int(current_collected),
                            "current_len": int(current_len),
                            "step_index": int(step_index),
                            "total_needed": int(total_needed),
                            "total_len": int(total_len),
                            "stride": int(stride),
                            "horizon": int(horizon),
                        },
                    )
                    print(
                        f"[info] Dataset '{ds.name}': collected {current_collected}/{total_needed} predictions via LLM."
                    )

                if not llm_failed and (current_len >= total_len or early_stop_due_to_bounds):
                    try:
                        pred_df_full = _complete_predictions(
                            out_csv, test_df, look_back, horizon, stride
                        )
                        y_true, y_pred = align_predictions(target_df, pred_df_full, ds.name)
                        if len(y_true) != len(pred_df_full) or not len(y_true):
                            raise ValueError("Not all emitted rows match finite test targets")
                        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                            raise ValueError("Non-finite evaluation arrays")
                    except Exception as exc:
                        _save_json(Path(ds_out_dir) / 'evaluation_status.json',
                                   dict(status='verification_failed', dataset=ds.name, error=str(exc)))
                        print(f"[warn] Dataset '{ds.name}' evaluation rejected: {exc}")
                        continue
                    evaluation = dict(dataset=ds.name, MSE=mse(y_true, y_pred),
                                      MAE=mae(y_true, y_pred), sMAPE=smape(y_true, y_pred),
                                      model='LLM', forecast_windows=len(range(0, total_len, stride)),
                                      evaluated_points=len(y_true),
                                      unique_evaluated_points=int(pred_df_full.time_stamp.nunique()),
                                      look_back=look_back, horizon=horizon, stride=stride)
                    _save_json(Path(ds_out_dir) / 'evaluation_metrics.json', evaluation)
                    _save_json(Path(ds_out_dir) / 'evaluation_status.json',
                               dict(status='completed', dataset=ds.name, metrics=evaluation))
                    clear_resume_state(ds_out_dir)
                    rows.append(
                        {
                            "dataset": ds.name,
                            "MSE": mse(y_true, y_pred),
                            "MAE": mae(y_true, y_pred),
                            "sMAPE": smape(y_true, y_pred),
                            "model": "LLM",
                        }
                    )
                    agent_success = True

                    try:
                        sel_path = os.path.join(ds_out_dir, "selected_features.json")
                        if os.path.exists(sel_path):
                            with open(sel_path, "r", encoding="utf-8") as f:
                                sel = json.load(f)
                            sf = sel.get("selected_features")
                            fw = sel.get("feature_weights")
                            if isinstance(sf, list):
                                print(f"=== LLM feature usage | dataset: {ds.name} ===")
                                print(f"- selected_features: {', '.join(sf)}")
                            if isinstance(fw, dict) and fw:
                                try:
                                    ordered = [
                                        (str(k), float(v))
                                        for k, v in fw.items()
                                        if v is not None and np.isfinite(float(v))
                                    ]
                                    ordered.sort(key=lambda kv: kv[1], reverse=True)
                                    preview = ", ".join([f"{k}:{v:.3f}" for k, v in ordered[:10]])
                                    if preview:
                                        suffix = " ..." if len(ordered) > 10 else ""
                                        print(f"- feature_weights: {preview}{suffix}")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    try:
                        if getattr(cfg, "use_exogenous", False):
                            exo_sel_path = os.path.join(ds_out_dir, "exogenous_selected_features.json")
                            if os.path.exists(exo_sel_path):
                                with open(exo_sel_path, "r", encoding="utf-8") as f:
                                    exo_sel = json.load(f)
                                vars_used = exo_sel.get("exogenous_vars") or []
                                dims_map = exo_sel.get("exogenous_feature_selection", {})
                                cors_map = exo_sel.get("exogenous_correlations", {})
                                print(f"=== LLM exogenous usage | dataset: {ds.name} ===")
                                print(f"- variables: {', '.join(vars_used) if vars_used else 'None'}")
                                if isinstance(dims_map, dict):
                                    for var, dims in dims_map.items():
                                        if isinstance(dims, list):
                                            try:
                                                corr_val = cors_map.get(var)
                                                if corr_val is not None:
                                                    print(f"- {var}: r={float(corr_val):.4f}; dimensions -> {', '.join(dims)}")
                                                else:
                                                    print(f"- {var}: dimensions -> {', '.join(dims)}")
                                            except Exception:
                                                print(f"- {var}: dimensions -> {', '.join(dims)}")
                    except Exception:
                        pass

                elif llm_failed and resume_required:
                    save_resume_state(
                        ds_out_dir,
                        {
                            "current_collected": int(current_collected),
                            "current_len": int(current_len),
                            "step_index": int(step_index),
                            "total_needed": int(total_needed),
                            "total_len": int(total_len),
                            "stride": int(stride),
                            "horizon": int(horizon),
                        },
                    )
                    print(
                        f"[info] Saved partial LLM results for dataset '{ds.name}'. Re-run the experiment to resume forecasting from step {step_index}."
                    )

        if agent_success:
            continue

        if agent is not None and resume_required:
            continue

        rows.append(deterministic_run_for_dataset(cfg, ds, case_base_ready=case_base_ready))

    summary = pd.DataFrame(rows)
    columns = ['dataset', 'MSE', 'MAE', 'sMAPE', 'model']
    summary = summary.reindex(columns=columns)
    atomic_csv(summary, Path(cfg.output_dir) / 'metrics.csv')
    _save_json(Path(cfg.output_dir) / 'metrics.json', summary.to_dict(orient='records'))
    print("\n=== Experiment Summary ===")
    print(summary.to_string(index=False))
    if use_agent and sum(row.get('model') == 'LLM' for row in rows) != len(cfg.datasets):
        raise RuntimeError('Incomplete LLM evaluation; inspect dataset evaluation_status.json files')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run time series agent experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--case-base-dir", type=str,
        default=str(Path(__file__).resolve().parent / "outputs/case_bases"),
        help="Reuse initialized libraries here; automatically build missing libraries.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Dataset name or alias to run (repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets defined in config (default).",
    )

    args, unknown = parser.parse_known_args()

    selector_tokens: List[str] = []
    if not args.all:
        selector_tokens.extend(args.datasets)

        for token in unknown:
            if token.startswith("--") and len(token) > 2:
                selector_tokens.append(token[2:])
            else:
                raise ValueError(f"Unrecognized argument '{token}'. Use --dataset or known aliases.")

    run_experiment(args.config, selector_tokens if selector_tokens else None, args.case_base_dir)
