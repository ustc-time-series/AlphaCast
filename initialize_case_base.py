from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
DEEP_MODELS = ("Autoformer", "DLinear", "TimesNet", "PatchTST", "iTransformer")
DATASETS = {
    **{name: (f"EPF/{name}/EPF_{name}", 168, 24, 48, "h")
       for name in ("NP", "PJM", "BE", "FR", "DE")},
    "ETTh": ("ETT/ETTh1/ETT_ETTh1", 96, 96, 48, "h"),
    "ETTm": ("ETT/ETTm1/ETT_ETTm1", 96, 96, 96, "15min"),
    "WP": ("POWER_NEW/windy_power/windy_power", 96, 96, 96, "15min"),
    "SP": ("POWER_NEW/sunny_power/sunny_power", 96, 96, 96, "15min"),
    "MOPEX": ("MOPEX/mopex", 96, 96, 48, "D"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dataset", nargs="+", help="Dataset names; default: NP")
    group.add_argument("--all-datasets", action="store_true")
    p.add_argument("--config", type=Path, help="Optional existing AlphaCast YAML; overrides built-in dataset settings")
    p.add_argument("--data-root", type=Path, help="Defaults to project datasets or sibling AlphaCast++/datasets")
    p.add_argument("--checkpoints-root", type=Path, help="Directory containing DATASET/MODEL/checkpoint.pth")
    p.add_argument("--foundation-root", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/case_bases")
    p.add_argument("--models", nargs="+", help="Subset of source candidates; aliases SNaive/ARIMA/CES are accepted")
    p.add_argument("--device", default="cpu", help="cpu (default) or physical cuda:N; set before importing torch")
    p.add_argument("--case-stride", type=int, help="Override training case sampling stride only")
    p.add_argument("--num-clusters", type=int, default=6)
    p.add_argument("--method", choices=["weighted", "voting"], default="weighted")
    p.add_argument("--dry-run", action="store_true", help="Check resources and print settings without forecasting")
    p.add_argument("--overwrite", action="store_true", help="Replace existing dataset case-base directory")
    return p.parse_args()


def resource_root(override, relative):
    if override is not None:
        return override.expanduser().resolve()
    for base in (ROOT, ROOT.parent / "AlphaCast++"):
        path = base / relative
        if path.is_dir():
            return path.resolve()
    return ROOT / relative


def main():
    args = parse_args()
    if args.case_stride is not None and args.case_stride <= 0:
        raise ValueError("--case-stride must be positive")
    if args.num_clusters <= 0:
        raise ValueError("--num-clusters must be positive")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device.startswith("cuda:") and args.device[5:].isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device[5:]
    else:
        raise ValueError("--device must be cpu or cuda:N")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import pandas as pd
    import yaml
    from alphacast.config import DatasetConfig
    from alphacast.data_loader import TIME_COL, infer_target_column
    from alphacast.tools import analysis

    data_root = resource_root(args.data_root, "datasets")
    checkpoints_root = resource_root(args.checkpoints_root, "checkpoints/deep_baselines")
    foundation_root = resource_root(args.foundation_root, "foundation_models")
    if args.config:
        config_path = args.config.expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text())
        entries = raw["datasets"]
        if isinstance(entries, dict):
            entries = [dict(value, name=value.get("name", name)) for name, value in entries.items()]
        # Read only fields understood by this checkout, including ++ configs.
        fields = DatasetConfig.__dataclass_fields__
        configs = [DatasetConfig(**{k: v for k, v in entry.items() if k in fields}) for entry in entries]
        for ds in configs:
            for key in ("training_csv", "test_csv", "context_prompt_file"):
                value = getattr(ds, key)
                if value:
                    path = Path(os.path.expandvars(value)).expanduser()
                    setattr(ds, key, str((config_path.parent / path).resolve()))
            ds.checkpoints = {name: str((config_path.parent / Path(os.path.expandvars(path)).expanduser()).resolve())
                              for name, path in ds.checkpoints.items() if path}
    else:
        configs = [DatasetConfig(name=name, training_csv=str(data_root / f"{stem}_train.csv"),
                   test_csv=str(data_root / f"{stem}_test.csv"), look_back=length,
                   predicted_window=horizon, sliding_window=stride, frequency=freq,
                   checkpoints={model: str(checkpoints_root / name / model / "checkpoint.pth")
                                for model in DEEP_MODELS})
                   for name, (stem, length, horizon, stride, freq) in DATASETS.items()]
    selected = args.dataset or ([ds.name for ds in configs] if args.all_datasets else ["NP"])
    lookup = {alias.lower(): ds for ds in configs for alias in [ds.name, *ds.aliases]}
    try:
        datasets = list({lookup[name.lower()].name: lookup[name.lower()] for name in selected}.values())
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {exc.args[0]}") from exc
    source_factory = analysis.get_default_models
    aliases = {"snaive": "seasonalnaive", "arima": "autoarima", "ces": "autoces"}
    requested = {aliases.get(name.lower(), name.lower()) for name in args.models} if args.models else None
    available = {m.alias.lower() for m in source_factory()}
    if requested and requested - available:
        raise ValueError(f"Unknown candidates: {sorted(requested - available)}; available: {sorted(available)}")

    def models_for(ds):
        models = [m for m in source_factory() if requested is None or m.alias.lower() in requested]
        for model in models:
            if model.alias in ("Sundial", "Chronos"):
                folder = "sundial-base-128m" if model.alias == "Sundial" else "chronos-bolt-base"
                model.local_dir = ds.checkpoints.get(model.alias, str(foundation_root / folder))
                if not Path(model.local_dir, "config.json").is_file():
                    raise FileNotFoundError(f"Missing {model.alias} local weights: {model.local_dir}")
            if model.alias in DEEP_MODELS and not Path(ds.checkpoints.get(model.alias, "")).is_file():
                raise FileNotFoundError(f"Missing {ds.name}/{model.alias} checkpoint: {ds.checkpoints.get(model.alias)}")
        return models

    plans = []
    for ds in datasets:
        evaluation_stride = ds.sliding_window
        if args.case_stride is not None:
            ds.sliding_window = args.case_stride
        if min(ds.look_back, ds.predicted_window, ds.sliding_window) <= 0:
            raise ValueError(f"Invalid window parameters for {ds.name}")
        # Deliberately never open test/validation data when building cases.
        frame = pd.read_csv(ds.training_csv)
        frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], format="mixed")
        frame = frame.sort_values(TIME_COL).reset_index(drop=True)
        values = frame[infer_target_column(frame, ds.name)].to_numpy(float)
        if frame[TIME_COL].duplicated().any() or frame[TIME_COL].isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Invalid training timestamps/target for {ds.name}")
        count = (len(frame) - ds.look_back - ds.predicted_window) // ds.sliding_window + 1
        if count <= 0:
            raise ValueError(f"Not enough training data for {ds.name}")
        models = models_for(ds)
        destination = args.output_dir.resolve() / ds.name
        if destination.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"{destination} already exists; choose a new output directory or --overwrite")
        setting = dict(dataset=ds.name, training_csv=ds.training_csv, test_csv=ds.test_csv,
                       training_rows=len(frame), look_back=ds.look_back, horizon=ds.predicted_window,
                       case_stride=ds.sliding_window, evaluation_stride=evaluation_stride,
                       forecast_windows=count, frequency=ds.frequency, models=[m.alias for m in models],
                       checkpoints=ds.checkpoints, device=args.device, method=args.method,
                       num_clusters=args.num_clusters, source="alphacast.tools.analysis.analyze_training",
                       training_only=True, error_metric="MSE on original target values")
        print(json.dumps(setting, ensure_ascii=False, indent=2), flush=True)
        plans.append((ds, frame, models, setting))
    if args.dry_run:
        return
    if args.device != "cpu":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")

    original_evaluate = analysis.evaluate_models_on_window
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ds, frame, models, setting in plans:
        errors = []
        def evaluate(candidates, x, fut, ts_x, ts_fut, season_length):
            best_name, best_error = "SeasonalNaive", float("inf")
            for model in candidates:
                name, error = original_evaluate([model], x, fut, ts_x, ts_fut, season_length)
                if not np.isfinite(error):
                    raise ValueError(f"Non-finite error for {ds.name}/{model.alias}")
                errors.append(dict(window_index=len(errors) // len(candidates),
                                   start_timestamp=ts_x.iloc[0].isoformat(), model=model.alias, MSE=error))
                if error < best_error:
                    best_name, best_error = name, error
            print(f"[{ds.name}] cases {len(errors) // len(candidates)}/{setting['forecast_windows']}", flush=True)
            return best_name, best_error
        # Build in a temporary directory; a failed candidate never publishes a partial case base.
        with tempfile.TemporaryDirectory(prefix=".case_base_", dir=args.output_dir) as temporary:
            try:
                analysis.get_default_models = lambda: models
                analysis.evaluate_models_on_window = evaluate
                result = analysis.analyze_training(frame, ds.look_back, ds.predicted_window,
                    temporary, ds.name, ds.sliding_window, method=args.method,
                    num_clusters=args.num_clusters, dataset_cfg=ds)
                if len(result.case_base) != setting["forecast_windows"]:
                    raise ValueError("Case count differs from the planned windows")
            finally:
                analysis.get_default_models = source_factory
                analysis.evaluate_models_on_window = original_evaluate
                analysis.configure_deep_learning_runtime(None, None)
            folder = Path(temporary) / ds.name
            pd.DataFrame(errors).to_csv(folder / "model_errors.csv", index=False)
            (folder / "setting.json").write_text(json.dumps(setting, ensure_ascii=False, indent=2))
            destination = args.output_dir.resolve() / ds.name
            if destination.exists():
                import time
                destination.rename(destination.with_name(ds.name + f".previous_{time.time_ns()}"))
            folder.rename(destination)
            print(f"[{ds.name}] saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
