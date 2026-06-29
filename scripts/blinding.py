#!/usr/bin/env python3
"""Canonical ROI blinding and singles-driven combinatorial background helper.

Supports legacy /events/<event_key> groups and Hong Cai flat /normal_hits/data tables.
The script writes selected singles, optional sampled synthetic pairs, and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TRIGGER_BACKGROUND = {"background", "nonbeam", "non-beam", "offbeam", "off-beam", "cosmic", "0", "false"}
TRIGGER_SIGNAL = {"signal", "beam", "onbeam", "on-beam", "1", "true"}


@dataclass
class Summary:
    mode: str
    input_format: str
    n_events_total: int
    n_events_selected: int
    n_single_selected: int
    n_pairs_sampled: int
    n_pairs_accepted: int
    pair_acceptance: float
    roi_keep_fraction: float
    n_double_expected: float


def decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def trigger_allowed(beam_type: object, mode: str) -> bool:
    if mode in {"nop", "signal+background"}:
        return True
    tag = decode_scalar(beam_type).strip().lower()
    if mode == "background":
        return tag in TRIGGER_BACKGROUND
    if mode == "signal":
        return tag in TRIGGER_SIGNAL
    raise ValueError(f"unknown mode: {mode}")


def load_legacy_events(handle: h5py.File, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    rows = []
    event_ids = []
    if "events" not in handle:
        raise ValueError("legacy format requires /events")
    for event_index, event_key in enumerate(handle["events"].keys()):
        group = handle["events"][event_key]
        beam_type = group.attrs.get("beam_type", group.attrs.get("trigger_type", "signal"))
        if not trigger_allowed(beam_type, mode):
            continue
        x = np.asarray(group["x"][:], dtype=float)
        y = np.asarray(group["y"][:], dtype=float)
        z = np.asarray(group["z"][:], dtype=float)
        labels = np.asarray(group.get("labels", np.arange(len(x)))[:])
        n = min(len(x), len(y), len(z), len(labels))
        for i in range(n):
            rows.append((x[i], y[i], z[i], labels[i], event_index, decode_scalar(beam_type)))
        event_ids.append(event_index)
    return structured_hits(rows), np.asarray(event_ids), "legacy-events"


def load_flat_table(handle: h5py.File, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if "normal_hits" not in handle or "data" not in handle["normal_hits"]:
        raise ValueError("flat format requires /normal_hits/data")
    data = handle["normal_hits/data"][:]
    names = set(data.dtype.names or [])
    required = {"x", "y", "z"}
    missing = required - names
    if missing:
        raise ValueError(f"/normal_hits/data missing required fields: {sorted(missing)}")
    event_field = "event_id" if "event_id" in names else None
    cluster_field = "cluster_id" if "cluster_id" in names else None
    beam_field = "beam_type" if "beam_type" in names else None
    rows = []
    selected_events = set()
    for i, row in enumerate(data):
        beam_type = row[beam_field] if beam_field else "signal"
        if not trigger_allowed(beam_type, mode):
            continue
        event_id = int(row[event_field]) if event_field else i
        cluster_id = int(row[cluster_field]) if cluster_field else i
        rows.append((float(row["x"]), float(row["y"]), float(row["z"]), cluster_id, event_id, decode_scalar(beam_type)))
        selected_events.add(event_id)
    return structured_hits(rows), np.asarray(sorted(selected_events)), "hong-cai-flat"


def structured_hits(rows: Iterable[tuple[float, float, float, int, int, str]]) -> np.ndarray:
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("cluster_id", "i8"), ("event_id", "i8"), ("beam_type", "U32")]
    return np.asarray(list(rows), dtype=dtype)


def load_hits(path: Path, mode: str) -> tuple[np.ndarray, np.ndarray, str, int]:
    import h5py

    with h5py.File(path, "r") as handle:
        if "normal_hits" in handle and "data" in handle["normal_hits"]:
            hits, selected_events, fmt = load_flat_table(handle, mode)
            total_events = len(np.unique(handle["normal_hits/data"]["event_id"])) if "event_id" in (handle["normal_hits/data"].dtype.names or []) else len(handle["normal_hits/data"])
            return hits, selected_events, fmt, int(total_events)
        hits, selected_events, fmt = load_legacy_events(handle, mode)
        return hits, selected_events, fmt, len(handle["events"].keys())


def pair_angles(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    dz = b["z"] - a["z"]
    flip = dz < 0
    dx = np.where(flip, -dx, dx)
    dy = np.where(flip, -dy, dy)
    dz = np.abs(dz)
    return np.arctan2(dx, dz), np.arctan2(dy, dz)


def roi_keep(theta_zx: np.ndarray, theta_zy: np.ndarray, center_zx: float, center_zy: float, radius: float, axis_symmetric: bool) -> np.ndarray:
    centers = [(center_zx, center_zy)]
    if axis_symmetric:
        centers.append((-center_zx, -center_zy))
    masked = np.zeros_like(theta_zx, dtype=bool)
    for cx, cy in centers:
        masked |= np.hypot(theta_zx - cx, theta_zy - cy) <= radius
    return ~masked


def sample_pairs(hits: np.ndarray, n_pairs: int, min_distance: float, apply_roi: bool, args: argparse.Namespace) -> tuple[np.ndarray, float, float]:
    if len(hits) < 2 or n_pairs <= 0:
        return np.empty(0, dtype=[("theta_zx", "f8"), ("theta_zy", "f8"), ("distance", "f8")]), 0.0, 1.0
    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, len(hits), size=n_pairs)
    j = rng.integers(0, len(hits), size=n_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, len(hits), size=int(np.sum(same)))
        same = i == j
    a, b = hits[i], hits[j]
    distance = np.sqrt((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2 + (b["z"] - a["z"]) ** 2)
    pair_mask = distance >= min_distance
    theta_zx, theta_zy = pair_angles(a, b)
    roi_mask = roi_keep(theta_zx, theta_zy, args.center_zx, args.center_zy, args.theta_radius, args.axis_symmetric) if apply_roi else np.ones(n_pairs, dtype=bool)
    keep = pair_mask & roi_mask
    dtype = [("theta_zx", "f8"), ("theta_zy", "f8"), ("distance", "f8")]
    pairs = np.empty(int(np.sum(keep)), dtype=dtype)
    pairs["theta_zx"] = theta_zx[keep]
    pairs["theta_zy"] = theta_zy[keep]
    pairs["distance"] = distance[keep]
    pair_acceptance = float(np.mean(pair_mask))
    roi_keep_fraction = float(np.sum(pair_mask & roi_mask) / max(np.sum(pair_mask), 1))
    return pairs, pair_acceptance, roi_keep_fraction


def write_outputs(output_dir: Path, hits: np.ndarray, pairs: np.ndarray, summary: Summary) -> None:
    import h5py

    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "blinded_singles_and_pairs.h5", "w") as handle:
        handle.create_dataset("singles", data=hits)
        handle.create_dataset("synthetic_pairs", data=pairs)
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply canonical trigger/ROI blinding and estimate singles-driven accidental doublets.")
    parser.add_argument("--input", required=True, type=Path, help="Input HDF5 file in legacy /events or flat /normal_hits/data format.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for summary JSON and optional HDF5 outputs.")
    parser.add_argument("--mode", choices=["nop", "background", "signal", "signal+background"], default="signal", help="Trigger selection and ROI policy.")
    parser.add_argument("--n-pairs", type=int, default=100000, help="Number of random single-single pairs to sample for shape and acceptance.")
    parser.add_argument("--min-distance", type=float, default=0.0, help="Minimum 3D separation required for accepted synthetic pairs.")
    parser.add_argument("--center-zx", type=float, default=0.0, help="Circular ROI center in theta_zx radians.")
    parser.add_argument("--center-zy", type=float, default=0.0, help="Circular ROI center in theta_zy radians.")
    parser.add_argument("--theta-radius", type=float, default=0.10, help="Circular ROI radius in radians.")
    parser.add_argument("--axis-symmetric", action=argparse.BooleanOptionalAction, default=True, help="Also exclude the opposite angular direction.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for synthetic pair sampling.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global np
    import numpy as np

    hits, selected_events, fmt, total_events = load_hits(args.input, args.mode)
    apply_roi = args.mode in {"signal", "signal+background"}
    pairs, pair_acceptance, roi_keep_fraction = sample_pairs(hits, args.n_pairs, args.min_distance, apply_roi, args)
    n_events = max(len(selected_events), 1)
    expected = (len(hits) ** 2) / (2.0 * n_events) * pair_acceptance * roi_keep_fraction
    summary = Summary(args.mode, fmt, total_events, len(selected_events), len(hits), args.n_pairs, len(pairs), pair_acceptance, roi_keep_fraction, expected)
    write_outputs(args.output_dir, hits, pairs, summary)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
