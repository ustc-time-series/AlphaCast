"""Atomic, origin-aware prediction storage shared by tools and the runner."""
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd


class ForecastCommitted(RuntimeError):
    """Local control signal: an approved forecast was saved; stop this agent run."""


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if 'time_stamp' in frame:
        # CSV reads yield strings, while new chunks contain Timestamps. Normalize
        # before writing so midnight-only data cannot mix date and datetime forms.
        frame = frame.copy()
        frame['time_stamp'] = pd.to_datetime(frame['time_stamp'], format='mixed')
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def window_rows(frame, offset, horizon, timestamps):
    required = {'window_offset', 'horizon_index', 'time_stamp', 'prediction'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Prediction schema missing {required - set(frame.columns)}')
    rows = frame[frame.window_offset == offset].copy()
    if 'emission_index' in rows:
        rows = rows.sort_values('emission_index', kind='stable')
    # Choose one complete submission, never stitch a partial retry into an older one.
    if rows.empty or not (rows.horizon_index == 0).any():
        raise ValueError(f'No forecast starts at origin {offset}')
    submission = (rows.horizon_index == 0).cumsum()
    rows = rows[submission == submission.max()]
    if rows.horizon_index.tolist() != list(range(horizon)):
        raise ValueError(f'Incomplete forecast at origin {offset}')
    actual_times = pd.DatetimeIndex(pd.to_datetime(rows.time_stamp, format='mixed'))
    if not actual_times.equals(pd.DatetimeIndex(timestamps)):
        raise ValueError(f'Wrong timestamps at origin {offset}')
    values = pd.to_numeric(rows.prediction, errors='coerce').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite forecast at origin {offset}')
    return rows


def saved_window(path, offset, horizon, timestamps):
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    try:
        return window_rows(frame, offset, horizon, timestamps)
    except ValueError:
        return None


def discard_window(path, offset):
    if not Path(path).exists():
        return
    frame = pd.read_csv(path)
    if 'window_offset' not in frame:
        raise ValueError('Cannot safely remove a window without window_offset')
    if (frame.window_offset == offset).any():
        atomic_csv(frame[frame.window_offset != offset], path)


def commit_window(path, chunk):
    """First complete submission wins. Called without awaits in a single writer."""
    offset = int(chunk.window_offset.iloc[0])
    timestamps = pd.to_datetime(chunk.time_stamp)
    if saved_window(path, offset, len(chunk), timestamps) is not None:
        return False
    existing = pd.read_csv(path) if Path(path).exists() else None
    if existing is not None:
        if 'window_offset' not in existing:
            raise ValueError('Cannot safely replace an origin without window_offset')
        existing = existing[existing.window_offset != offset]
        if not existing.empty:
            chunk = pd.concat([existing, chunk], ignore_index=True)
    atomic_csv(chunk, path)
    return True
