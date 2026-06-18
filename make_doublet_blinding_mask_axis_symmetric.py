#!/usr/bin/env python3
"""
Build a deterministic candidate-level blinding mask for 2x2 double-blip HDF5 files.

This script mirrors the HDF5 structure used in CombinatoryBg_v5_blip_rates_and_stats.ipynb:

    /events/<event_key>/labels
    /events/<event_key>/x
    /events/<event_key>/y
    /events/<event_key>/z

A candidate is an event with exactly two accepted fiducial cluster labels after applying
    labels >= 0
and the default 2x2 fiducial geometry cuts used in the notebook:
    -51.85 <= y <= 51.85 cm
    12.68 <= |z| <= 54.32 cm

The script does NOT remove candidates from the input file. It writes a separate mask file.

Blinding policy implemented here:
  1. Hard-mask all doublet candidates inside a predefined MCP-like angular ROI.
  2. Randomly mask additional candidates outside the ROI until a fixed total hidden
     fraction is reached.
  3. The random top-up is deterministic, using a salted cryptographic hash of
     (event_key, candidate_id).

By default the output is a PUBLIC mask file that only contains candidate IDs and blind_mask.
It intentionally does not reveal which candidates are ROI vs random top-up.
Use --private-out only for coordinator/debugging output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


VERSION = "0.2.0"


@dataclass(frozen=True)
class FiducialConfig:
    y_abs_max_cm: float = 51.85
    z_inner_abs_cm: float = 12.68
    z_outer_abs_cm: float = 54.32


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    event_index: int
    event_key: str
    label0: int
    label1: int
    npts0: int
    npts1: int
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    dx: float
    dy: float
    dz: float
    dist_cm: float
    theta_z_rad: float
    theta_zx_rad: float
    theta_zy_rad: float
    theta_transverse_rad: float


def fiducial_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: FiducialConfig) -> np.ndarray:
    """Fiducial geometry mask matching the notebook's accepted cluster selection."""
    return (
        (y >= -cfg.y_abs_max_cm) & (y <= cfg.y_abs_max_cm) &
        (((z >= cfg.z_inner_abs_cm) & (z <= cfg.z_outer_abs_cm)) |
         ((z >= -cfg.z_outer_abs_cm) & (z <= -cfg.z_inner_abs_cm)))
    )


def _read_string_or_bytes_key(key) -> str:
    if isinstance(key, bytes):
        return key.decode("utf-8", errors="replace")
    return str(key)


def iter_event_keys(events_group: h5py.Group) -> List[str]:
    """Return event keys in deterministic order."""
    keys = [_read_string_or_bytes_key(k) for k in events_group.keys()]

    # HDF5 group order is often insertion order, but sort naturally when possible.
    # This keeps candidate_id stable across systems for numeric event names.
    def sort_key(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    return sorted(keys, key=sort_key)


def extract_doublet_candidates(
    h5_path: str | os.PathLike,
    *,
    min_dist_cm: float = 10.0,
    use_centroid: bool = True,
    sort_by_z: bool = True,
    fiducial: FiducialConfig = FiducialConfig(),
) -> Tuple[List[Candidate], Dict[str, int]]:
    """
    Read /events/<key> groups and build one candidate per exactly-two-cluster event.

    Returns
    -------
    candidates, stats
        stats includes total events, multiplicity counters, and rejected-by-distance count.
    """
    candidates: List[Candidate] = []
    stats: Dict[str, int] = {
        "n_events": 0,
        "n_events_no_accepted_clusters": 0,
        "n_events_one_cluster": 0,
        "n_events_two_clusters_before_dist": 0,
        "n_events_two_clusters_after_dist": 0,
        "n_events_three_or_more_clusters": 0,
        "n_rejected_min_dist": 0,
    }

    with h5py.File(h5_path, "r") as f:
        if "events" not in f:
            raise KeyError("Input HDF5 has no /events group.")
        events = f["events"]
        keys = iter_event_keys(events)
        stats["n_events"] = len(keys)

        for event_index, key in enumerate(keys):
            g = events[key]
            for required in ("labels", "x", "y", "z"):
                if required not in g:
                    raise KeyError(f"Event {key!r} is missing dataset {required!r}.")

            labels = np.asarray(g["labels"][:])
            x = np.asarray(g["x"][:], dtype=float)
            y = np.asarray(g["y"][:], dtype=float)
            z = np.asarray(g["z"][:], dtype=float)

            geom_mask = fiducial_mask(x, y, z, fiducial)
            idx = np.where(geom_mask & (labels >= 0))[0]
            if idx.size == 0:
                stats["n_events_no_accepted_clusters"] += 1
                continue

            labs = labels[idx]
            uniq = np.unique(labs)
            if uniq.size == 1:
                stats["n_events_one_cluster"] += 1
                continue
            if uniq.size >= 3:
                stats["n_events_three_or_more_clusters"] += 1
                continue
            if uniq.size != 2:
                # Mostly here for completeness.
                continue

            stats["n_events_two_clusters_before_dist"] += 1

            lab0, lab1 = int(uniq[0]), int(uniq[1])
            c0 = idx[labs == uniq[0]]
            c1 = idx[labs == uniq[1]]
            if c0.size == 0 or c1.size == 0:
                continue

            if use_centroid:
                p0 = np.array([x[c0].mean(), y[c0].mean(), z[c0].mean()], dtype=float)
                p1 = np.array([x[c1].mean(), y[c1].mean(), z[c1].mean()], dtype=float)
            else:
                p0 = np.array([x[c0[0]], y[c0[0]], z[c0[0]]], dtype=float)
                p1 = np.array([x[c1[0]], y[c1[0]], z[c1[0]]], dtype=float)

            if sort_by_z and (p1[2] < p0[2]):
                p0, p1 = p1, p0
                lab0, lab1 = lab1, lab0
                c0, c1 = c1, c0

            d = p1 - p0
            dist = float(np.linalg.norm(d))
            if not np.isfinite(dist) or dist <= 0:
                stats["n_rejected_min_dist"] += 1
                continue
            if dist < float(min_dist_cm):
                stats["n_rejected_min_dist"] += 1
                continue

            dx, dy, dz = map(float, d)
            theta_z = float(math.acos(float(np.clip(dz / dist, -1.0, 1.0))))
            theta_zx = float(math.atan2(dx, dz))
            theta_zy = float(math.atan2(dy, dz))
            theta_transverse = float(math.sqrt(theta_zx * theta_zx + theta_zy * theta_zy))

            stats["n_events_two_clusters_after_dist"] += 1
            candidates.append(Candidate(
                candidate_id=len(candidates),
                event_index=event_index,
                event_key=str(key),
                label0=lab0,
                label1=lab1,
                npts0=int(c0.size),
                npts1=int(c1.size),
                x0=float(p0[0]), y0=float(p0[1]), z0=float(p0[2]),
                x1=float(p1[0]), y1=float(p1[1]), z1=float(p1[2]),
                dx=dx, dy=dy, dz=dz,
                dist_cm=dist,
                theta_z_rad=theta_z,
                theta_zx_rad=theta_zx,
                theta_zy_rad=theta_zy,
                theta_transverse_rad=theta_transverse,
            ))

    return candidates, stats


def hash_to_unit_interval(event_key: str, candidate_id: int, salt: str) -> float:
    """Deterministic pseudo-random score in [0, 1) from event key, candidate id, and salt."""
    payload = f"{salt}|{event_key}|{candidate_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


def rounding_to_int(x: float, mode: str) -> int:
    if mode == "floor":
        return int(math.floor(x))
    if mode == "ceil":
        return int(math.ceil(x))
    if mode == "round":
        return int(round(x))
    raise ValueError(f"Unknown rounding mode: {mode}")


def compute_roi_mask(
    candidates: List[Candidate],
    *,
    roi_mode: str,
    theta_cone_deg: float,
    theta_zx_deg: float,
    theta_zy_deg: float,
    theta_z_deg: float,
    center_zx_deg: float,
    center_zy_deg: float,
    axis_symmetric: bool = True,
) -> np.ndarray:
    """Return boolean array: candidate is in the hard signal-region ROI.

    By default this treats the MCP-like direction as an *axis*, not an oriented
    arrow: a doublet aligned with +z or -z is considered signal-like. This is
    intentionally robust against arbitrary cluster ordering. Use
    axis_symmetric=False only for a directed +z-only ROI.
    """
    n = len(candidates)
    in_roi = np.zeros(n, dtype=bool)
    if n == 0:
        return in_roi

    if axis_symmetric:
        # Recompute the angular variables from the displacement vector after
        # canonically orienting each candidate so dz >= 0. This makes the ROI
        # independent of which cluster happened to be stored first and catches
        # both +z and -z aligned doublets.
        dx = np.array([c.dx for c in candidates], dtype=float)
        dy = np.array([c.dy for c in candidates], dtype=float)
        dz = np.array([c.dz for c in candidates], dtype=float)
        dist = np.array([c.dist_cm for c in candidates], dtype=float)

        flip = dz < 0.0
        dx = np.where(flip, -dx, dx)
        dy = np.where(flip, -dy, dy)
        dz = np.abs(dz)

        theta_z = np.arccos(np.clip(dz / dist, -1.0, 1.0))
        theta_zx = np.arctan2(dx, dz)
        theta_zy = np.arctan2(dy, dz)
    else:
        theta_z = np.array([c.theta_z_rad for c in candidates], dtype=float)
        theta_zx = np.array([c.theta_zx_rad for c in candidates], dtype=float)
        theta_zy = np.array([c.theta_zy_rad for c in candidates], dtype=float)

    center_zx = math.radians(center_zx_deg)
    center_zy = math.radians(center_zy_deg)
    d_zx = theta_zx - center_zx
    d_zy = theta_zy - center_zy

    if roi_mode == "cone":
        radius = np.sqrt(d_zx * d_zx + d_zy * d_zy)
        in_roi = radius <= math.radians(theta_cone_deg)
    elif roi_mode == "box":
        in_roi = (np.abs(d_zx) <= math.radians(theta_zx_deg)) & (np.abs(d_zy) <= math.radians(theta_zy_deg))
    elif roi_mode == "theta_z":
        # Useful for quick checks, but cone/box in theta_zx/theta_zy is usually cleaner.
        in_roi = theta_z <= math.radians(theta_z_deg)
    else:
        raise ValueError(f"Unknown roi_mode: {roi_mode}")

    return np.asarray(in_roi, dtype=bool)


def compute_blinding_masks(
    candidates: List[Candidate],
    *,
    in_roi: np.ndarray,
    hidden_fraction: float,
    target_rounding: str,
    salt: str,
    allow_roi_overflow: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Return blind_mask, random_topup_mask, visible_mask, hash_scores, n_target_hidden.

    The random top-up selects candidates outside the ROI with the smallest hash scores,
    so the exact target hidden count is reached deterministically.
    """
    n = len(candidates)
    if not (0.0 <= hidden_fraction <= 1.0):
        raise ValueError("hidden_fraction must be in [0, 1].")
    if len(in_roi) != n:
        raise ValueError("in_roi has wrong length.")

    n_target_hidden = rounding_to_int(hidden_fraction * n, target_rounding)
    n_target_hidden = max(0, min(n, n_target_hidden))

    n_roi = int(np.sum(in_roi))
    if n_roi > n_target_hidden and not allow_roi_overflow:
        raise RuntimeError(
            "Blinding target exceeded: ROI candidates alone are more than the fixed hidden target. "
            "No public mask was written. Either increase --hidden-fraction, narrow the ROI, "
            "or rerun with --allow-roi-overflow if this is an intentional coordinator-only check."
        )

    hash_scores = np.array(
        [hash_to_unit_interval(c.event_key, c.candidate_id, salt) for c in candidates],
        dtype=np.float64,
    )

    random_topup = np.zeros(n, dtype=bool)
    n_topup = max(0, n_target_hidden - n_roi)
    if n_topup > 0:
        outside = np.where(~in_roi)[0]
        order = outside[np.argsort(hash_scores[outside], kind="mergesort")]
        random_topup[order[:n_topup]] = True

    blind_mask = np.asarray(in_roi | random_topup, dtype=bool)
    visible_mask = ~blind_mask
    return blind_mask, random_topup, visible_mask, hash_scores, n_target_hidden


def write_public_mask_h5(
    out_path: str | os.PathLike,
    *,
    input_path: str | os.PathLike,
    candidates: List[Candidate],
    blind_mask: np.ndarray,
    visible_mask: np.ndarray,
    config: Dict,
    stats: Dict[str, int],
    n_target_hidden: int,
    overwrite: bool = False,
) -> None:
    """Write public blinding mask: no ROI/random split and no angular diagnostics."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(candidates)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    numeric_dtype = np.dtype([
        ("candidate_id", "<i8"),
        ("event_index", "<i8"),
        ("blind_mask", "?"),
        ("visible", "?"),
    ])
    numeric = np.empty(n, dtype=numeric_dtype)
    event_keys = np.empty(n, dtype=object)

    for i, c in enumerate(candidates):
        numeric[i] = (c.candidate_id, c.event_index, bool(blind_mask[i]), bool(visible_mask[i]))
        event_keys[i] = c.event_key

    with h5py.File(out_path, "w") as f:
        g = f.create_group("blinding")
        g.attrs["schema"] = "doublet_candidate_public_mask_v1"
        g.attrs["script_version"] = VERSION
        g.attrs["input_file"] = str(input_path)
        g.attrs["n_candidates"] = n
        g.attrs["n_target_hidden"] = int(n_target_hidden)
        g.attrs["n_visible"] = int(np.sum(visible_mask))
        g.attrs["hidden_fraction_target"] = float(config["hidden_fraction"])
        g.attrs["config_json"] = json.dumps(config, sort_keys=True)
        g.attrs["stats_json"] = json.dumps(stats, sort_keys=True)

        g.create_dataset("event_key", data=event_keys, dtype=string_dtype)
        g.create_dataset("mask", data=numeric, compression="gzip", shuffle=True)


def write_private_table_h5(
    out_path: str | os.PathLike,
    *,
    input_path: str | os.PathLike,
    candidates: List[Candidate],
    in_roi: np.ndarray,
    random_topup: np.ndarray,
    blind_mask: np.ndarray,
    visible_mask: np.ndarray,
    hash_scores: np.ndarray,
    config: Dict,
    stats: Dict[str, int],
    n_target_hidden: int,
    overwrite: bool = False,
) -> None:
    """Write full coordinator/debug table. This reveals ROI counts; do not use as analyst-facing output."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Private output file exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(candidates)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    numeric_dtype = np.dtype([
        ("candidate_id", "<i8"),
        ("event_index", "<i8"),
        ("label0", "<i4"),
        ("label1", "<i4"),
        ("npts0", "<i4"),
        ("npts1", "<i4"),
        ("x0", "<f8"), ("y0", "<f8"), ("z0", "<f8"),
        ("x1", "<f8"), ("y1", "<f8"), ("z1", "<f8"),
        ("dx", "<f8"), ("dy", "<f8"), ("dz", "<f8"),
        ("dist_cm", "<f8"),
        ("theta_z_rad", "<f8"),
        ("theta_zx_rad", "<f8"),
        ("theta_zy_rad", "<f8"),
        ("theta_transverse_rad", "<f8"),
        ("hash_score", "<f8"),
        ("in_roi", "?"),
        ("random_topup", "?"),
        ("blind_mask", "?"),
        ("visible", "?"),
    ])
    numeric = np.empty(n, dtype=numeric_dtype)
    event_keys = np.empty(n, dtype=object)

    for i, c in enumerate(candidates):
        numeric[i] = (
            c.candidate_id, c.event_index,
            c.label0, c.label1, c.npts0, c.npts1,
            c.x0, c.y0, c.z0, c.x1, c.y1, c.z1,
            c.dx, c.dy, c.dz, c.dist_cm,
            c.theta_z_rad, c.theta_zx_rad, c.theta_zy_rad, c.theta_transverse_rad,
            float(hash_scores[i]),
            bool(in_roi[i]), bool(random_topup[i]), bool(blind_mask[i]), bool(visible_mask[i]),
        )
        event_keys[i] = c.event_key

    with h5py.File(out_path, "w") as f:
        g = f.create_group("blinding")
        g.attrs["schema"] = "doublet_candidate_private_table_v1"
        g.attrs["script_version"] = VERSION
        g.attrs["input_file"] = str(input_path)
        g.attrs["n_candidates"] = n
        g.attrs["n_target_hidden"] = int(n_target_hidden)
        g.attrs["n_roi"] = int(np.sum(in_roi))
        g.attrs["n_random_topup"] = int(np.sum(random_topup))
        g.attrs["n_hidden"] = int(np.sum(blind_mask))
        g.attrs["n_visible"] = int(np.sum(visible_mask))
        g.attrs["config_json"] = json.dumps(config, sort_keys=True)
        g.attrs["stats_json"] = json.dumps(stats, sort_keys=True)

        g.create_dataset("event_key", data=event_keys, dtype=string_dtype)
        g.create_dataset("candidates", data=numeric, compression="gzip", shuffle=True)


def write_csv(
    out_path: str | os.PathLike,
    *,
    candidates: List[Candidate],
    blind_mask: np.ndarray,
    visible_mask: np.ndarray,
    include_private: bool = False,
    in_roi: Optional[np.ndarray] = None,
    random_topup: Optional[np.ndarray] = None,
    hash_scores: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> None:
    """Optional CSV export. Public by default; private columns only with include_private=True."""
    import csv

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"CSV file exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    public_fields = ["candidate_id", "event_index", "event_key", "blind_mask", "visible"]
    private_fields = [
        "in_roi", "random_topup", "hash_score",
        "label0", "label1", "npts0", "npts1",
        "x0", "y0", "z0", "x1", "y1", "z1",
        "dx", "dy", "dz", "dist_cm",
        "theta_z_rad", "theta_zx_rad", "theta_zy_rad", "theta_transverse_rad",
    ]
    fields = public_fields + (private_fields if include_private else [])

    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for i, c in enumerate(candidates):
            row = {
                "candidate_id": c.candidate_id,
                "event_index": c.event_index,
                "event_key": c.event_key,
                "blind_mask": int(bool(blind_mask[i])),
                "visible": int(bool(visible_mask[i])),
            }
            if include_private:
                row.update({
                    "in_roi": int(bool(in_roi[i])) if in_roi is not None else "",
                    "random_topup": int(bool(random_topup[i])) if random_topup is not None else "",
                    "hash_score": float(hash_scores[i]) if hash_scores is not None else "",
                    **asdict(c),
                })
                # asdict(c) also contains public keys; public values above are fine either way.
            writer.writerow(row)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a deterministic blinding mask for exactly-two-cluster double-blip candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_h5", help="Input clustering HDF5 file with /events/<key> groups.")
    p.add_argument("-o", "--out", required=True, help="Public output HDF5 mask file.")
    p.add_argument("--private-out", default=None,
                   help="Optional private coordinator/debug HDF5 table. Reveals ROI/random split.")
    p.add_argument("--csv", default=None,
                   help="Optional public CSV mask export. Use --csv-private to include private columns.")
    p.add_argument("--csv-private", action="store_true",
                   help="Include ROI/random/angle columns in CSV. Coordinator/debug only.")

    p.add_argument("--min-dist-cm", type=float, default=10.0,
                   help="Minimum 3D centroid/point separation for a doublet candidate.")
    p.add_argument("--use-centroid", dest="use_centroid", action="store_true", default=True,
                   help="Use cluster centroids, matching the newer notebook summary path.")
    p.add_argument("--first-hit", dest="use_centroid", action="store_false",
                   help="Use first point in each cluster, matching older notebook behavior.")
    p.add_argument("--sort-by-z", dest="sort_by_z", action="store_true", default=True,
                   help="Order the two blips so z1 >= z0 before computing angles.")
    p.add_argument("--no-sort-by-z", dest="sort_by_z", action="store_false")
    p.add_argument("--axis-symmetric-roi", dest="axis_symmetric_roi", action="store_true", default=True,
                   help="Treat the ROI as an unoriented axis: +z and -z aligned doublets are both hidden.")
    p.add_argument("--directed-roi", dest="axis_symmetric_roi", action="store_false",
                   help="Use the signed candidate direction as stored. Mostly for debugging old behavior.")

    p.add_argument("--y-abs-max-cm", type=float, default=51.85)
    p.add_argument("--z-inner-abs-cm", type=float, default=12.68)
    p.add_argument("--z-outer-abs-cm", type=float, default=54.32)

    p.add_argument("--roi-mode", choices=["cone", "box", "theta_z"], default="cone",
                   help="Hard signal-region definition.")
    p.add_argument("--theta-cone-deg", type=float, default=5.0,
                   help="For --roi-mode cone: radius in the (theta_zx, theta_zy) plane.")
    p.add_argument("--theta-zx-deg", type=float, default=5.0,
                   help="For --roi-mode box: half-width in theta_zx.")
    p.add_argument("--theta-zy-deg", type=float, default=5.0,
                   help="For --roi-mode box: half-width in theta_zy.")
    p.add_argument("--theta-z-deg", type=float, default=5.0,
                   help="For --roi-mode theta_z: theta_z upper bound.")
    p.add_argument("--center-zx-deg", type=float, default=0.0,
                   help="ROI center in theta_zx. Useful if expected MCP direction is not exactly z-axis.")
    p.add_argument("--center-zy-deg", type=float, default=0.0,
                   help="ROI center in theta_zy. Useful if expected MCP direction is not exactly z-axis.")

    p.add_argument("--hidden-fraction", type=float, default=0.25,
                   help="Fixed total fraction of final doublet candidates to hide.")
    p.add_argument("--target-rounding", choices=["round", "floor", "ceil"], default="round",
                   help="Integer rounding rule for hidden_fraction * N_candidates.")
    p.add_argument("--salt", default=None,
                   help="Secret salt for deterministic random top-up. Prefer --salt-file to avoid shell history.")
    p.add_argument("--salt-file", default=None,
                   help="Path to text file containing the secret salt.")
    p.add_argument("--allow-roi-overflow", action="store_true",
                   help="If ROI exceeds hidden target, still write a mask with all ROI hidden. Coordinator-only.")
    p.add_argument("--reveal-counts", action="store_true",
                   help="Print ROI/random split. Coordinator-only; do not use in blinded analysis logs.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return p.parse_args(argv)


def get_salt(args: argparse.Namespace) -> str:
    if args.salt_file:
        salt = Path(args.salt_file).read_text().strip()
        if not salt:
            raise ValueError("Salt file is empty.")
        return salt
    if args.salt:
        return str(args.salt)

    # This is intentionally allowed for quick dry tests, but it should not be used for a real blind.
    print(
        "WARNING: no --salt or --salt-file supplied. Using a non-secret default salt. "
        "Do not use this for the real blind.",
        file=sys.stderr,
    )
    return "CHANGE_ME_NOT_A_SECRET_SALT"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    salt = get_salt(args)
    salt_sha256 = hashlib.sha256(salt.encode("utf-8")).hexdigest()

    fid = FiducialConfig(
        y_abs_max_cm=args.y_abs_max_cm,
        z_inner_abs_cm=args.z_inner_abs_cm,
        z_outer_abs_cm=args.z_outer_abs_cm,
    )

    config = {
        "script_version": VERSION,
        "min_dist_cm": float(args.min_dist_cm),
        "use_centroid": bool(args.use_centroid),
        "sort_by_z": bool(args.sort_by_z),
        "axis_symmetric_roi": bool(args.axis_symmetric_roi),
        "fiducial": asdict(fid),
        "roi_mode": args.roi_mode,
        "theta_cone_deg": float(args.theta_cone_deg),
        "theta_zx_deg": float(args.theta_zx_deg),
        "theta_zy_deg": float(args.theta_zy_deg),
        "theta_z_deg": float(args.theta_z_deg),
        "center_zx_deg": float(args.center_zx_deg),
        "center_zy_deg": float(args.center_zy_deg),
        "hidden_fraction": float(args.hidden_fraction),
        "target_rounding": args.target_rounding,
        "salt_sha256": salt_sha256,
        "public_output_hides_roi_split": True,
    }

    candidates, stats = extract_doublet_candidates(
        args.input_h5,
        min_dist_cm=args.min_dist_cm,
        use_centroid=args.use_centroid,
        sort_by_z=args.sort_by_z,
        fiducial=fid,
    )

    in_roi = compute_roi_mask(
        candidates,
        roi_mode=args.roi_mode,
        theta_cone_deg=args.theta_cone_deg,
        theta_zx_deg=args.theta_zx_deg,
        theta_zy_deg=args.theta_zy_deg,
        theta_z_deg=args.theta_z_deg,
        center_zx_deg=args.center_zx_deg,
        center_zy_deg=args.center_zy_deg,
        axis_symmetric=bool(args.axis_symmetric_roi),
    )

    try:
        blind_mask, random_topup, visible_mask, hash_scores, n_target_hidden = compute_blinding_masks(
            candidates,
            in_roi=in_roi,
            hidden_fraction=args.hidden_fraction,
            target_rounding=args.target_rounding,
            salt=salt,
            allow_roi_overflow=args.allow_roi_overflow,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        if args.reveal_counts:
            print(f"Coordinator detail: n_candidates={len(candidates)} n_roi={int(np.sum(in_roi))}", file=sys.stderr)
        return 2

    write_public_mask_h5(
        args.out,
        input_path=args.input_h5,
        candidates=candidates,
        blind_mask=blind_mask,
        visible_mask=visible_mask,
        config=config,
        stats=stats,
        n_target_hidden=n_target_hidden,
        overwrite=args.overwrite,
    )

    if args.private_out:
        write_private_table_h5(
            args.private_out,
            input_path=args.input_h5,
            candidates=candidates,
            in_roi=in_roi,
            random_topup=random_topup,
            blind_mask=blind_mask,
            visible_mask=visible_mask,
            hash_scores=hash_scores,
            config=config,
            stats=stats,
            n_target_hidden=n_target_hidden,
            overwrite=args.overwrite,
        )

    if args.csv:
        write_csv(
            args.csv,
            candidates=candidates,
            blind_mask=blind_mask,
            visible_mask=visible_mask,
            include_private=bool(args.csv_private),
            in_roi=in_roi,
            random_topup=random_topup,
            hash_scores=hash_scores,
            overwrite=args.overwrite,
        )

    # Public-safe summary: does not reveal ROI/random split.
    print("Blinding mask written.")
    print(f"  input events:        {stats['n_events']}")
    print(f"  doublet candidates:  {len(candidates)}")
    print(f"  target hidden:       {n_target_hidden} ({args.hidden_fraction:.3f} target fraction)")
    print(f"  visible candidates:  {int(np.sum(visible_mask))}")
    print(f"  public mask:         {args.out}")
    if args.private_out:
        print(f"  private table:       {args.private_out}")
    if args.csv:
        print(f"  csv:                 {args.csv}")

    if args.reveal_counts:
        print("Coordinator-only counts:")
        print(f"  in hard ROI:         {int(np.sum(in_roi))}")
        print(f"  random top-up:       {int(np.sum(random_topup))}")
        print(f"  total hidden:        {int(np.sum(blind_mask))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
