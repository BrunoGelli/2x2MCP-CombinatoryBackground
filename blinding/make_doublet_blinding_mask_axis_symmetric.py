#!/usr/bin/env python3
"""
Build a candidate-level hard-ROI blinding mask for 2x2 double-blip HDF5 files.

Input format
------------
This script expects the modern clustering HDF5 layout used by the current
combinatorial-background notebooks:

    /events/<event_key>/labels
    /events/<event_key>/x
    /events/<event_key>/y
    /events/<event_key>/z

A candidate is one event with exactly two accepted fiducial cluster labels after
applying:

    labels >= 0
    -y_abs_max_cm <= y <= y_abs_max_cm
    z_inner_abs_cm <= |z| <= z_outer_abs_cm

Blinding policy
---------------
Only one policy is implemented:

    hard-mask candidates inside a circular projected-angle ROI.

There is no random thinning, no salt, no sideband normalization, no private
coordinator table, and no box/theta_z ROI mode.

The circular ROI is defined in the (theta_zx, theta_zy) plane:

    sqrt((theta_zx - center_zx)^2 + (theta_zy - center_zy)^2) <= radius

By default the ROI is axis-symmetric: a doublet aligned with +z or -z is treated
as signal-like. This makes the mask robust against arbitrary cluster ordering.
Use --directed-roi only if you intentionally want a +z-only ROI.

Output format
-------------
The output HDF5 file contains:

    /blinding/event_key
    /blinding/mask

where /blinding/mask has fields:

    candidate_id, event_index, blind_mask, visible

The input HDF5 is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


VERSION = "0.3.0"


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
    """Fiducial geometry mask matching the accepted cluster selection."""
    return (
        (y >= -cfg.y_abs_max_cm)
        & (y <= cfg.y_abs_max_cm)
        & (
            ((z >= cfg.z_inner_abs_cm) & (z <= cfg.z_outer_abs_cm))
            | ((z >= -cfg.z_outer_abs_cm) & (z <= -cfg.z_inner_abs_cm))
        )
    )


def _read_string_or_bytes_key(key) -> str:
    if isinstance(key, bytes):
        return key.decode("utf-8", errors="replace")
    return str(key)


def iter_event_keys(events_group: h5py.Group) -> List[str]:
    """Return event keys in deterministic order."""
    keys = [_read_string_or_bytes_key(k) for k in events_group.keys()]

    def sort_key(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    return sorted(keys, key=sort_key)


def extract_doublet_candidates(
    h5_path: str | os.PathLike,
    *,
    events_group_path: str = "events",
    min_dist_cm: float = 10.0,
    use_centroid: bool = True,
    sort_by_z: bool = True,
    fiducial: FiducialConfig = FiducialConfig(),
) -> Tuple[List[Candidate], Dict[str, int]]:
    """Read event groups and build one candidate per exactly-two-cluster event."""
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
        if events_group_path not in f:
            raise KeyError(f"Input HDF5 has no /{events_group_path} group.")

        events = f[events_group_path]
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
            if not np.isfinite(dist) or dist <= 0 or dist < float(min_dist_cm):
                stats["n_rejected_min_dist"] += 1
                continue

            dx, dy, dz = map(float, d)
            theta_z = float(math.acos(float(np.clip(dz / dist, -1.0, 1.0))))
            theta_zx = float(math.atan2(dx, dz))
            theta_zy = float(math.atan2(dy, dz))
            theta_transverse = float(math.sqrt(theta_zx * theta_zx + theta_zy * theta_zy))

            stats["n_events_two_clusters_after_dist"] += 1
            candidates.append(
                Candidate(
                    candidate_id=len(candidates),
                    event_index=event_index,
                    event_key=str(key),
                    label0=lab0,
                    label1=lab1,
                    npts0=int(c0.size),
                    npts1=int(c1.size),
                    x0=float(p0[0]),
                    y0=float(p0[1]),
                    z0=float(p0[2]),
                    x1=float(p1[0]),
                    y1=float(p1[1]),
                    z1=float(p1[2]),
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    dist_cm=dist,
                    theta_z_rad=theta_z,
                    theta_zx_rad=theta_zx,
                    theta_zy_rad=theta_zy,
                    theta_transverse_rad=theta_transverse,
                )
            )

    return candidates, stats


def _angle_delta(a: np.ndarray, center: float) -> np.ndarray:
    """Smallest signed angular difference a - center, wrapped to [-pi, pi)."""
    return (a - center + np.pi) % (2.0 * np.pi) - np.pi


def projected_angles_for_roi(
    candidates: List[Candidate],
    *,
    axis_symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return theta_zx, theta_zy arrays using the requested ROI orientation convention."""
    if not candidates:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    dx = np.array([c.dx for c in candidates], dtype=float)
    dy = np.array([c.dy for c in candidates], dtype=float)
    dz = np.array([c.dz for c in candidates], dtype=float)

    if axis_symmetric:
        # Treat +z and -z as the same axis.  If the displacement points toward
        # -z, flip the whole vector before building projected angles.
        flip = dz < 0.0
        dx = np.where(flip, -dx, dx)
        dy = np.where(flip, -dy, dy)
        dz = np.abs(dz)

    theta_zx = np.arctan2(dx, dz)
    theta_zy = np.arctan2(dy, dz)
    return theta_zx, theta_zy


def compute_circular_roi_mask(
    candidates: List[Candidate],
    *,
    theta_radius_deg: float,
    center_zx_deg: float,
    center_zy_deg: float,
    axis_symmetric: bool = True,
) -> np.ndarray:
    """Return boolean array: candidate is inside the circular projected-angle ROI."""
    if theta_radius_deg < 0:
        raise ValueError("theta_radius_deg must be non-negative.")

    theta_zx, theta_zy = projected_angles_for_roi(candidates, axis_symmetric=axis_symmetric)
    center_zx = math.radians(center_zx_deg)
    center_zy = math.radians(center_zy_deg)

    d_zx = _angle_delta(theta_zx, center_zx)
    d_zy = _angle_delta(theta_zy, center_zy)
    radius = np.sqrt(d_zx * d_zx + d_zy * d_zy)
    return radius <= math.radians(theta_radius_deg)


def write_mask_h5(
    out_path: str | os.PathLike,
    *,
    input_path: str | os.PathLike,
    candidates: List[Candidate],
    in_roi: np.ndarray,
    config: Dict,
    stats: Dict[str, int],
    overwrite: bool = False,
) -> None:
    """Write candidate-level hard-ROI blinding mask."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(candidates)
    if len(in_roi) != n:
        raise ValueError("in_roi has wrong length.")

    visible = ~np.asarray(in_roi, dtype=bool)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    mask_dtype = np.dtype(
        [
            ("candidate_id", "<i8"),
            ("event_index", "<i8"),
            ("blind_mask", "?"),
            ("visible", "?"),
        ]
    )

    mask = np.empty(n, dtype=mask_dtype)
    event_keys = np.empty(n, dtype=object)
    for i, c in enumerate(candidates):
        mask[i] = (c.candidate_id, c.event_index, bool(in_roi[i]), bool(visible[i]))
        event_keys[i] = c.event_key

    with h5py.File(out_path, "w") as f:
        g = f.create_group("blinding")
        g.attrs["schema"] = "doublet_candidate_circular_roi_mask_v1"
        g.attrs["script_version"] = VERSION
        g.attrs["input_file"] = str(input_path)
        g.attrs["config_json"] = json.dumps(config, sort_keys=True)
        # Do not store ROI or candidate-count summaries as attributes; the mask
        # itself is the only analysis-facing bookkeeping.

        g.create_dataset("event_key", data=event_keys, dtype=string_dtype)
        g.create_dataset("mask", data=mask, compression="gzip", shuffle=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a hard circular-ROI blinding mask for exactly-two-cluster double-blip candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_h5", help="Input clustering HDF5 file with /events/<key> groups.")
    p.add_argument("-o", "--out", required=True, help="Output HDF5 mask file.")
    p.add_argument("--events-group", default="events", help="Path to the event group inside input_h5.")

    p.add_argument("--min-dist-cm", type=float, default=10.0, help="Minimum 3D separation for a doublet candidate.")
    p.add_argument("--use-centroid", dest="use_centroid", action="store_true", default=True, help="Use cluster centroids.")
    p.add_argument("--first-hit", dest="use_centroid", action="store_false", help="Use first point in each cluster.")
    p.add_argument("--sort-by-z", dest="sort_by_z", action="store_true", default=True, help="Order the two blips so z1 >= z0 before storing angles.")
    p.add_argument("--no-sort-by-z", dest="sort_by_z", action="store_false")

    p.add_argument("--axis-symmetric-roi", dest="axis_symmetric_roi", action="store_true", default=True, help="Treat +z and -z aligned doublets as the same ROI axis.")
    p.add_argument("--directed-roi", dest="axis_symmetric_roi", action="store_false", help="Use directed +z projected angles only.")

    p.add_argument("--theta-radius-deg", type=float, default=5.0, help="Circular ROI radius in the (theta_zx, theta_zy) plane.")
    p.add_argument("--center-zx-deg", type=float, default=0.0, help="ROI center in theta_zx.")
    p.add_argument("--center-zy-deg", type=float, default=0.0, help="ROI center in theta_zy.")

    p.add_argument("--y-abs-max-cm", type=float, default=51.85)
    p.add_argument("--z-inner-abs-cm", type=float, default=12.68)
    p.add_argument("--z-outer-abs-cm", type=float, default=54.32)

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    fid = FiducialConfig(
        y_abs_max_cm=args.y_abs_max_cm,
        z_inner_abs_cm=args.z_inner_abs_cm,
        z_outer_abs_cm=args.z_outer_abs_cm,
    )

    config = {
        "script_version": VERSION,
        "events_group": str(args.events_group),
        "min_dist_cm": float(args.min_dist_cm),
        "use_centroid": bool(args.use_centroid),
        "sort_by_z": bool(args.sort_by_z),
        "axis_symmetric_roi": bool(args.axis_symmetric_roi),
        "roi_shape": "circle",
        "theta_radius_deg": float(args.theta_radius_deg),
        "center_zx_deg": float(args.center_zx_deg),
        "center_zy_deg": float(args.center_zy_deg),
        "fiducial": asdict(fid),
    }

    candidates, stats = extract_doublet_candidates(
        args.input_h5,
        events_group_path=args.events_group,
        min_dist_cm=args.min_dist_cm,
        use_centroid=args.use_centroid,
        sort_by_z=args.sort_by_z,
        fiducial=fid,
    )

    in_roi = compute_circular_roi_mask(
        candidates,
        theta_radius_deg=args.theta_radius_deg,
        center_zx_deg=args.center_zx_deg,
        center_zy_deg=args.center_zy_deg,
        axis_symmetric=bool(args.axis_symmetric_roi),
    )

    write_mask_h5(
        args.out,
        input_path=args.input_h5,
        candidates=candidates,
        in_roi=in_roi,
        config=config,
        stats=stats,
        overwrite=args.overwrite,
    )

    print(f"Circular ROI blinding mask written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
