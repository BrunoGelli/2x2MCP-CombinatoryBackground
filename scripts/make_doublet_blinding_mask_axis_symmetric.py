#!/usr/bin/env python3
"""
Build a candidate-level hard-ROI blinding/selection mask for 2x2 double-blip HDF5 files.

Supported input formats
-----------------------
1) Hong-style flat blip table, preferred:

    /normal_hits/data        compound dataset with per-blip fields such as
                             x, y, z, cluster_id, Q, beam_type
    /normal_hits/ref_region  event slices with start/stop fields

   If /normal_hits/ref_region is absent, the script can group /normal_hits/data
   by an event-id column such as event_id.

2) Older event-group format:

    /events/<event_key>/labels
    /events/<event_key>/x
    /events/<event_key>/y
    /events/<event_key>/z

Candidate definition
--------------------
A candidate is one event with exactly two accepted fiducial cluster labels after:

    labels >= 0
    -y_abs_max_cm <= y <= y_abs_max_cm
    z_inner_abs_cm <= |z| <= z_outer_abs_cm
    pair separation >= min_dist_cm

For each candidate, the two cluster positions are centroids by default.

Modes
-----
The output mask has blind_mask=True for candidates hidden from the analysis and
visible=True for candidates allowed through.

    nop                : allow all doublet candidates; ignore trigger type and ROI
    background         : allow only non-beam/background events; ignore ROI
    signal             : allow only beam/signal events outside the circular ROI
    signal+background  : allow any trigger type outside the circular ROI

Here "enforce ROI" means the circular ROI is hidden/cut.

ROI
---
The ROI is a circle in projected-angle space:

    sqrt((theta_zx - center_zx)^2 + (theta_zy - center_zy)^2) <= theta_radius

By default the ROI is axis-symmetric, so +z and -z aligned doublets are treated
as the same direction. Use --directed-roi only for debugging a directed +z-only
selection.

Output
------
The output HDF5 file contains:

    /blinding/event_key
    /blinding/mask

where /blinding/mask has fields:

    candidate_id, event_index, trigger_type, blind_mask, visible

The input HDF5 is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


VERSION = "0.4.0"


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
    trigger_type: int
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


def _sort_key(s: str):
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def _path_exists(h5: h5py.File, path: str) -> bool:
    try:
        h5[path]
        return True
    except KeyError:
        return False


def _names_from_dataset(ds: h5py.Dataset) -> Tuple[str, ...]:
    names = ds.dtype.names
    if names is None:
        raise TypeError(f"Dataset {ds.name} is not a compound/structured dataset.")
    return tuple(names)


def _resolve_field(
    names: Sequence[str],
    requested: str,
    aliases: Sequence[str],
    *,
    required: bool = True,
    role: str = "field",
) -> Optional[str]:
    names_set = set(names)

    if requested and requested != "auto":
        if requested in names_set:
            return requested
        if required:
            raise KeyError(f"Requested {role} {requested!r} not present. Available fields: {list(names)}")
        return None

    for alias in aliases:
        if alias in names_set:
            return alias

    if required:
        raise KeyError(f"Could not auto-detect {role}. Tried aliases {list(aliases)}. Available fields: {list(names)}")
    return None


def _first_int_from_struct_array(arr: np.ndarray, field: Optional[str]) -> Optional[int]:
    if field is None or arr is None or len(arr) == 0:
        return None
    vals = np.asarray(arr[field])
    if vals.size == 0:
        return None
    vals = vals[np.isfinite(vals)] if np.issubdtype(vals.dtype, np.number) else vals
    if vals.size == 0:
        return None
    try:
        return int(vals[0])
    except Exception:
        return None


def _event_key_from_sources(
    *,
    event_index: int,
    hits: np.ndarray,
    ref_row,
    event_id_field_hits: Optional[str],
    event_id_field_ref: Optional[str],
) -> str:
    if event_id_field_ref is not None and ref_row is not None:
        try:
            return _read_string_or_bytes_key(ref_row[event_id_field_ref])
        except Exception:
            pass

    if event_id_field_hits is not None and hits is not None and len(hits) > 0:
        vals = np.asarray(hits[event_id_field_hits])
        if vals.size:
            # If the slice has multiple values, use the first; ref_region defines the event window.
            return _read_string_or_bytes_key(vals[0])

    return str(event_index)


def build_candidate_from_hits(
    hits: np.ndarray,
    *,
    event_index: int,
    event_key: str,
    trigger_type: int,
    x_field: str,
    y_field: str,
    z_field: str,
    label_field: str,
    min_dist_cm: float,
    use_centroid: bool,
    sort_by_z: bool,
    fiducial: FiducialConfig,
    candidate_id: int,
) -> Optional[Candidate]:
    """Build a Candidate from one event's per-hit/blip rows, or return None."""
    if hits is None or len(hits) == 0:
        return None

    x = np.asarray(hits[x_field], dtype=float)
    y = np.asarray(hits[y_field], dtype=float)
    z = np.asarray(hits[z_field], dtype=float)
    labels = np.asarray(hits[label_field])

    geom_mask = fiducial_mask(x, y, z, fiducial)
    idx = np.where(geom_mask & (labels >= 0))[0]
    if idx.size == 0:
        return None

    labs = labels[idx]
    uniq = np.unique(labs)
    if uniq.size != 2:
        return None

    lab0, lab1 = int(uniq[0]), int(uniq[1])
    c0 = idx[labs == uniq[0]]
    c1 = idx[labs == uniq[1]]
    if c0.size == 0 or c1.size == 0:
        return None

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
    if not np.isfinite(dist) or dist <= 0.0 or dist < float(min_dist_cm):
        return None

    dx, dy, dz = map(float, d)
    theta_z = float(math.acos(float(np.clip(dz / dist, -1.0, 1.0))))
    theta_zx = float(math.atan2(dx, dz))
    theta_zy = float(math.atan2(dy, dz))
    theta_transverse = float(math.sqrt(theta_zx * theta_zx + theta_zy * theta_zy))

    return Candidate(
        candidate_id=candidate_id,
        event_index=event_index,
        event_key=str(event_key),
        trigger_type=int(trigger_type),
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


def detect_input_format(
    h5_path: str | os.PathLike,
    *,
    requested: str,
    hits_dataset_path: str,
    ref_region_path: str,
    events_group_path: str,
) -> str:
    if requested != "auto":
        return requested

    with h5py.File(h5_path, "r") as f:
        if _path_exists(f, hits_dataset_path):
            return "normal-hits"
        if _path_exists(f, events_group_path):
            return "event-groups"

    raise KeyError(
        "Could not auto-detect input format. Expected either "
        f"{hits_dataset_path!r} or {events_group_path!r}."
    )


def extract_candidates_from_normal_hits(
    h5_path: str | os.PathLike,
    *,
    hits_dataset_path: str,
    ref_region_path: str,
    x_field: str,
    y_field: str,
    z_field: str,
    label_field: str,
    event_id_field: str,
    trigger_type_field: str,
    default_trigger_type: Optional[int],
    min_dist_cm: float,
    use_centroid: bool,
    sort_by_z: bool,
    fiducial: FiducialConfig,
) -> Tuple[List[Candidate], Dict[str, object]]:
    """Extract exactly-two-cluster candidates from Hong-style normal_hits data."""
    candidates: List[Candidate] = []

    with h5py.File(h5_path, "r") as f:
        if hits_dataset_path not in f:
            raise KeyError(f"Input HDF5 has no dataset {hits_dataset_path!r}.")

        hits_ds = f[hits_dataset_path]
        hit_names = _names_from_dataset(hits_ds)

        x_field = _resolve_field(hit_names, x_field, ["x", "X"], role="x field")
        y_field = _resolve_field(hit_names, y_field, ["y", "Y"], role="y field")
        z_field = _resolve_field(hit_names, z_field, ["z", "Z"], role="z field")
        label_field = _resolve_field(
            hit_names,
            label_field,
            ["cluster_id", "cluster", "label", "labels", "dbscan_label"],
            role="cluster/label field",
        )
        event_id_field_hits = _resolve_field(
            hit_names,
            event_id_field,
            ["event_id", "event", "evt_id", "event_idx", "event_index", "entry", "spill_id"],
            required=False,
            role="event id field",
        )
        trigger_field_hits = _resolve_field(
            hit_names,
            trigger_type_field,
            ["beam_type", "trigger_type", "is_beam", "beam_related", "beam", "trigger"],
            required=False,
            role="trigger/beam-type field",
        )

        has_ref = ref_region_path in f
        ref_names: Tuple[str, ...] = ()
        event_id_field_ref: Optional[str] = None
        trigger_field_ref: Optional[str] = None

        if has_ref:
            ref_ds = f[ref_region_path]
            ref_names = _names_from_dataset(ref_ds)
            if "start" not in ref_names or "stop" not in ref_names:
                raise KeyError(f"{ref_region_path!r} must have 'start' and 'stop' fields.")
            event_id_field_ref = _resolve_field(
                ref_names,
                event_id_field,
                ["event_id", "event", "evt_id", "event_idx", "event_index", "entry", "spill_id"],
                required=False,
                role="event id field in ref_region",
            )
            trigger_field_ref = _resolve_field(
                ref_names,
                trigger_type_field,
                ["beam_type", "trigger_type", "is_beam", "beam_related", "beam", "trigger"],
                required=False,
                role="trigger/beam-type field in ref_region",
            )

            for event_index, rr in enumerate(ref_ds):
                start = int(rr["start"])
                stop = int(rr["stop"])
                if stop <= start:
                    continue

                hits = hits_ds[start:stop]
                event_key = _event_key_from_sources(
                    event_index=event_index,
                    hits=hits,
                    ref_row=rr,
                    event_id_field_hits=event_id_field_hits,
                    event_id_field_ref=event_id_field_ref,
                )

                trigger_type = _first_int_from_struct_array(hits, trigger_field_hits)
                if trigger_type is None:
                    trigger_type = _first_int_from_struct_array(np.asarray([rr], dtype=ref_ds.dtype), trigger_field_ref)
                if trigger_type is None:
                    trigger_type = default_trigger_type if default_trigger_type is not None else -1

                cand = build_candidate_from_hits(
                    hits,
                    event_index=event_index,
                    event_key=event_key,
                    trigger_type=int(trigger_type),
                    x_field=x_field,
                    y_field=y_field,
                    z_field=z_field,
                    label_field=label_field,
                    min_dist_cm=min_dist_cm,
                    use_centroid=use_centroid,
                    sort_by_z=sort_by_z,
                    fiducial=fiducial,
                    candidate_id=len(candidates),
                )
                if cand is not None:
                    candidates.append(cand)

        else:
            if event_id_field_hits is None:
                raise KeyError(
                    f"No {ref_region_path!r} dataset found and no event-id field could be auto-detected in "
                    f"{hits_dataset_path!r}. Use --event-id-field."
                )

            event_ids = np.asarray(hits_ds[event_id_field_hits])
            unique_events = np.unique(event_ids)

            for event_index, event_value in enumerate(unique_events):
                row_idx = np.where(event_ids == event_value)[0]
                if row_idx.size == 0:
                    continue
                hits = hits_ds[row_idx]
                event_key = _read_string_or_bytes_key(event_value)

                trigger_type = _first_int_from_struct_array(hits, trigger_field_hits)
                if trigger_type is None:
                    trigger_type = default_trigger_type if default_trigger_type is not None else -1

                cand = build_candidate_from_hits(
                    hits,
                    event_index=event_index,
                    event_key=event_key,
                    trigger_type=int(trigger_type),
                    x_field=x_field,
                    y_field=y_field,
                    z_field=z_field,
                    label_field=label_field,
                    min_dist_cm=min_dist_cm,
                    use_centroid=use_centroid,
                    sort_by_z=sort_by_z,
                    fiducial=fiducial,
                    candidate_id=len(candidates),
                )
                if cand is not None:
                    candidates.append(cand)

    info = {
        "input_format": "normal-hits",
        "hits_dataset": hits_dataset_path,
        "ref_region_dataset": ref_region_path if has_ref else None,
        "fields": {
            "x": x_field,
            "y": y_field,
            "z": z_field,
            "label": label_field,
            "event_id_hits": event_id_field_hits,
            "trigger_type_hits": trigger_field_hits,
            "event_id_ref": event_id_field_ref,
            "trigger_type_ref": trigger_field_ref,
        },
    }
    return candidates, info


def iter_event_group_keys(events_group: h5py.Group) -> List[str]:
    keys = [_read_string_or_bytes_key(k) for k in events_group.keys()]
    return sorted(keys, key=_sort_key)


def extract_candidates_from_event_groups(
    h5_path: str | os.PathLike,
    *,
    events_group_path: str,
    default_trigger_type: Optional[int],
    min_dist_cm: float,
    use_centroid: bool,
    sort_by_z: bool,
    fiducial: FiducialConfig,
) -> Tuple[List[Candidate], Dict[str, object]]:
    """Extract candidates from the older /events/<event_key> layout."""
    candidates: List[Candidate] = []

    with h5py.File(h5_path, "r") as f:
        if events_group_path not in f:
            raise KeyError(f"Input HDF5 has no group {events_group_path!r}.")
        events = f[events_group_path]

        for event_index, key in enumerate(iter_event_group_keys(events)):
            g = events[key]
            for required in ("labels", "x", "y", "z"):
                if required not in g:
                    raise KeyError(f"Event {key!r} is missing dataset {required!r}.")

            dtype = [
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("cluster_id", "i8"),
            ]
            n = len(g["x"])
            hits = np.empty(n, dtype=dtype)
            hits["x"] = np.asarray(g["x"][:], dtype=float)
            hits["y"] = np.asarray(g["y"][:], dtype=float)
            hits["z"] = np.asarray(g["z"][:], dtype=float)
            hits["cluster_id"] = np.asarray(g["labels"][:], dtype=int)

            trigger_type = int(default_trigger_type) if default_trigger_type is not None else -1

            cand = build_candidate_from_hits(
                hits,
                event_index=event_index,
                event_key=str(key),
                trigger_type=trigger_type,
                x_field="x",
                y_field="y",
                z_field="z",
                label_field="cluster_id",
                min_dist_cm=min_dist_cm,
                use_centroid=use_centroid,
                sort_by_z=sort_by_z,
                fiducial=fiducial,
                candidate_id=len(candidates),
            )
            if cand is not None:
                candidates.append(cand)

    info = {
        "input_format": "event-groups",
        "events_group": events_group_path,
        "fields": {
            "x": "x",
            "y": "y",
            "z": "z",
            "label": "labels",
            "event_id": "event_key",
            "trigger_type": "default" if default_trigger_type is not None else None,
        },
    }
    return candidates, info


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


def compute_visible_mask(
    candidates: List[Candidate],
    *,
    mode: str,
    in_roi: np.ndarray,
    signal_trigger_value: int,
    background_trigger_value: int,
) -> np.ndarray:
    """Compute the analysis-visible candidates for the requested mode."""
    n = len(candidates)
    if len(in_roi) != n:
        raise ValueError("in_roi has wrong length.")

    if mode == "nop":
        return np.ones(n, dtype=bool)

    trigger = np.array([c.trigger_type for c in candidates], dtype=int)

    if mode == "background":
        return trigger == int(background_trigger_value)

    if mode == "signal":
        return (trigger == int(signal_trigger_value)) & (~in_roi)

    if mode == "signal+background":
        return ~in_roi

    raise ValueError(f"Unknown mode: {mode!r}")


def validate_trigger_availability(
    candidates: List[Candidate],
    *,
    mode: str,
    default_trigger_type: Optional[int],
) -> None:
    """Fail early if trigger-dependent modes are requested without trigger info."""
    if mode not in {"background", "signal"}:
        return
    if default_trigger_type is not None:
        return
    if not candidates:
        return
    trigger = np.array([c.trigger_type for c in candidates], dtype=int)
    if np.all(trigger == -1):
        raise ValueError(
            f"Mode {mode!r} needs a trigger/beam-type field, but no trigger field was found. "
            "Use --trigger-type-field, or provide --default-trigger-type for old files."
        )


def write_mask_h5(
    out_path: str | os.PathLike,
    *,
    input_path: str | os.PathLike,
    candidates: List[Candidate],
    visible: np.ndarray,
    in_roi: np.ndarray,
    config: Dict,
    overwrite: bool = False,
) -> None:
    """Write candidate-level blinding/selection mask."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(candidates)
    visible = np.asarray(visible, dtype=bool)
    if len(visible) != n:
        raise ValueError("visible has wrong length.")
    if len(in_roi) != n:
        raise ValueError("in_roi has wrong length.")

    blind_mask = ~visible
    string_dtype = h5py.string_dtype(encoding="utf-8")
    mask_dtype = np.dtype(
        [
            ("candidate_id", "<i8"),
            ("event_index", "<i8"),
            ("trigger_type", "<i8"),
            ("blind_mask", "?"),
            ("visible", "?"),
        ]
    )

    mask = np.empty(n, dtype=mask_dtype)
    event_keys = np.empty(n, dtype=object)
    for i, c in enumerate(candidates):
        mask[i] = (c.candidate_id, c.event_index, c.trigger_type, bool(blind_mask[i]), bool(visible[i]))
        event_keys[i] = c.event_key

    with h5py.File(out_path, "w") as f:
        g = f.create_group("blinding")
        g.attrs["schema"] = "doublet_candidate_trigger_aware_circular_roi_mask_v1"
        g.attrs["script_version"] = VERSION
        g.attrs["input_file"] = str(input_path)
        g.attrs["config_json"] = json.dumps(config, sort_keys=True)

        # The main public analysis interface:
        g.create_dataset("event_key", data=event_keys, dtype=string_dtype)
        g.create_dataset("mask", data=mask, compression="gzip", shuffle=True)

        # Optional machine-friendly candidate geometry table. This is not needed
        # for normal mask application, but it helps downstream code apply the same
        # ROI convention to singles-driven MC without reimplementing parsing.
        geom_dtype = np.dtype(
            [
                ("candidate_id", "<i8"),
                ("dx", "<f8"),
                ("dy", "<f8"),
                ("dz", "<f8"),
                ("dist_cm", "<f8"),
                ("theta_zx_rad", "<f8"),
                ("theta_zy_rad", "<f8"),
                ("theta_transverse_rad", "<f8"),
                ("in_roi", "?"),
            ]
        )
        geom = np.empty(n, dtype=geom_dtype)
        for i, c in enumerate(candidates):
            geom[i] = (
                c.candidate_id,
                c.dx,
                c.dy,
                c.dz,
                c.dist_cm,
                c.theta_zx_rad,
                c.theta_zy_rad,
                c.theta_transverse_rad,
                bool(in_roi[i]),
            )
        g.create_dataset("candidate_geometry", data=geom, compression="gzip", shuffle=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a trigger-aware hard circular-ROI mask for exactly-two-cluster double-blip candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_h5", help="Input clustering HDF5 file.")
    p.add_argument("-o", "--out", required=True, help="Output HDF5 mask file.")
    p.add_argument(
        "--mode",
        choices=["nop", "background", "signal", "signal+background"],
        default="signal",
        help=(
            "nop: allow all; background: allow trigger==background value only and ignore ROI; "
            "signal: allow trigger==signal value outside ROI; signal+background: allow any trigger outside ROI."
        ),
    )

    p.add_argument(
        "--input-format",
        choices=["auto", "normal-hits", "event-groups"],
        default="auto",
        help="Input schema. 'normal-hits' is Hong's flat blip-table format.",
    )
    p.add_argument("--hits-dataset", default="normal_hits/data", help="Hong-format per-blip dataset path.")
    p.add_argument("--ref-region-dataset", default="normal_hits/ref_region", help="Hong-format event slice dataset path.")
    p.add_argument("--events-group", default="events", help="Old-format event group path.")

    p.add_argument("--x-field", default="auto")
    p.add_argument("--y-field", default="auto")
    p.add_argument("--z-field", default="auto")
    p.add_argument("--label-field", default="auto", help="Cluster label field, e.g. cluster_id.")
    p.add_argument("--event-id-field", default="auto", help="Event id field. Used if present; otherwise ref-region index is used.")
    p.add_argument("--trigger-type-field", default="auto", help="Trigger/beam-type field, e.g. beam_type.")
    p.add_argument("--signal-trigger-value", type=int, default=1, help="Value identifying beam/signal-related events.")
    p.add_argument("--background-trigger-value", type=int, default=0, help="Value identifying non-beam/background events.")
    p.add_argument(
        "--default-trigger-type",
        type=int,
        default=None,
        help="Use this trigger type when the input has no trigger field. Mainly for old-format debug tests.",
    )

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

    input_format = detect_input_format(
        args.input_h5,
        requested=args.input_format,
        hits_dataset_path=args.hits_dataset,
        ref_region_path=args.ref_region_dataset,
        events_group_path=args.events_group,
    )

    if input_format == "normal-hits":
        candidates, source_info = extract_candidates_from_normal_hits(
            args.input_h5,
            hits_dataset_path=args.hits_dataset,
            ref_region_path=args.ref_region_dataset,
            x_field=args.x_field,
            y_field=args.y_field,
            z_field=args.z_field,
            label_field=args.label_field,
            event_id_field=args.event_id_field,
            trigger_type_field=args.trigger_type_field,
            default_trigger_type=args.default_trigger_type,
            min_dist_cm=args.min_dist_cm,
            use_centroid=args.use_centroid,
            sort_by_z=args.sort_by_z,
            fiducial=fid,
        )
    elif input_format == "event-groups":
        candidates, source_info = extract_candidates_from_event_groups(
            args.input_h5,
            events_group_path=args.events_group,
            default_trigger_type=args.default_trigger_type,
            min_dist_cm=args.min_dist_cm,
            use_centroid=args.use_centroid,
            sort_by_z=args.sort_by_z,
            fiducial=fid,
        )
    else:
        raise ValueError(f"Unsupported input format: {input_format!r}")

    validate_trigger_availability(
        candidates,
        mode=args.mode,
        default_trigger_type=args.default_trigger_type,
    )

    in_roi = compute_circular_roi_mask(
        candidates,
        theta_radius_deg=args.theta_radius_deg,
        center_zx_deg=args.center_zx_deg,
        center_zy_deg=args.center_zy_deg,
        axis_symmetric=bool(args.axis_symmetric_roi),
    )

    visible = compute_visible_mask(
        candidates,
        mode=args.mode,
        in_roi=in_roi,
        signal_trigger_value=args.signal_trigger_value,
        background_trigger_value=args.background_trigger_value,
    )

    config = {
        "script_version": VERSION,
        "mode": args.mode,
        "input_format": input_format,
        "source_info": source_info,
        "min_dist_cm": float(args.min_dist_cm),
        "use_centroid": bool(args.use_centroid),
        "sort_by_z": bool(args.sort_by_z),
        "axis_symmetric_roi": bool(args.axis_symmetric_roi),
        "roi_shape": "circle",
        "theta_radius_deg": float(args.theta_radius_deg),
        "center_zx_deg": float(args.center_zx_deg),
        "center_zy_deg": float(args.center_zy_deg),
        "signal_trigger_value": int(args.signal_trigger_value),
        "background_trigger_value": int(args.background_trigger_value),
        "default_trigger_type": args.default_trigger_type,
        "fiducial": asdict(fid),
    }

    write_mask_h5(
        args.out,
        input_path=args.input_h5,
        candidates=candidates,
        visible=visible,
        in_roi=in_roi,
        config=config,
        overwrite=args.overwrite,
    )

    print(f"Trigger-aware circular ROI mask written: {args.out}")
    print(f"  mode: {args.mode}")
    print(f"  input format: {input_format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
