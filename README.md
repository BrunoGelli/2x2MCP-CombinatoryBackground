# 2x2 MCP combinatorial-background analysis

This repository contains the canonical, ROI-blinded workflow for estimating the
accidental/combinatorial double-blip background in the DUNE ND-LAr 2x2
demonstrator millicharged-particle (MCP) search.

## Physics purpose

MCP-like candidates are pairs of low-energy blips whose displacement is aligned
with the beam direction. The dominant accidental background is estimated from
data by randomly pairing observed single-blip candidates to build a synthetic
double-blip angular distribution.

The workflow is intentionally small:

- `scripts/blinding.py` applies trigger/ROI selection and produces selected
  singles, synthetic random pairs, and a normalization summary.
- `notebooks/combinatorial_background_roi_blinded.ipynb` is the collaborator
  notebook for running the script and plotting the blinded synthetic-pair shape.
- `docs/data_format.md` documents accepted HDF5 input layouts.

## Blinding policy

The signal-like angular region is excluded with a hard circular region of
interest (ROI) in projected angular space:

```text
sqrt((theta_zx - center_zx)^2 + (theta_zy - center_zy)^2) <= theta_radius
```

By default the ROI is axis-symmetric: the opposite direction is excluded too, so
+z- and -z-aligned doublets are treated equivalently. Disable that only for a
specific cross-check with `--no-axis-symmetric`.

There is no random thinning, no salt file, no sideband normalization, no box ROI,
and no private coordinator table in this workflow.

## Selection modes

`--mode` controls trigger selection and whether the circular ROI is applied:

| Mode | Trigger selection | ROI cut |
| --- | --- | --- |
| `nop` | all trigger types | no |
| `background` | non-beam/background triggers only | no |
| `signal` | beam/signal triggers only | yes |
| `signal+background` | all trigger types | yes |

Recognized trigger labels include common beam/non-beam strings such as `beam`,
`signal`, `onbeam`, `background`, `nonbeam`, `offbeam`, and `cosmic`.

## Background model and normalization

The singles-driven Monte Carlo estimates the doublet **shape** and pair-level
acceptance. It includes the minimum-distance cut and, for signal-like modes, the
ROI keep fraction. It is not normalized to the observed doublet count and is not
sideband-normalized.

The expected accidental doublet rate is computed from the number of selected
single blips:

```text
N_double_expected ≈ N_single^2 / (2 N_events) × pair_acceptance × ROI_keep_fraction
```

Here `N_events` is the number of selected events in the input sample. The output
`summary.json` records every factor used in the estimate.

## Supported input formats

Two HDF5 layouts are supported.

### Legacy event-group format

```text
/events/<event_key>/x
/events/<event_key>/y
/events/<event_key>/z
/events/<event_key>/labels
```

Optional event-group attributes `beam_type` or `trigger_type` are used for mode
selection. If they are absent, events are treated as signal/beam triggers.

### Hong Cai flat blip-table format

```text
/normal_hits/data
/normal_hits/ref_region
```

The `/normal_hits/data` structured table must include `x`, `y`, and `z` fields.
Recommended fields include `cluster_id`, `Q`, `event_id`, and `beam_type`. See
`docs/data_format.md` for details.

## Install

Use a fresh Python environment if possible:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the blinding script

Example signal-region blinded run:

```bash
python scripts/blinding.py \
  --input data/example_blips.h5 \
  --output-dir outputs/signal_roi_blinded \
  --mode signal \
  --n-pairs 100000 \
  --min-distance 2.0 \
  --center-zx 0.0 \
  --center-zy 0.0 \
  --theta-radius 0.10
```

Background-trigger control sample:

```bash
python scripts/blinding.py \
  --input data/example_blips.h5 \
  --output-dir outputs/background_control \
  --mode background \
  --n-pairs 100000 \
  --min-distance 2.0
```

No-operation parsing/check mode:

```bash
python scripts/blinding.py \
  --input data/example_blips.h5 \
  --output-dir outputs/nop_check \
  --mode nop \
  --n-pairs 1000
```

## Run the notebook

Open the notebook and edit the configuration cell:

```bash
jupyter notebook notebooks/combinatorial_background_roi_blinded.ipynb
```

Set `INPUT_H5`, `OUTPUT_DIR`, ROI settings, and `RUN_BLINDING = True` when you
are ready to generate outputs. The notebook is intentionally lightweight and
should not be used for large production MC scans; run those through
`scripts/blinding.py` directly.

## Outputs

Each script run writes:

- `summary.json`: selected event/single counts, sampled-pair counts,
  `pair_acceptance`, `ROI_keep_fraction`, and `N_double_expected`.
- `blinded_singles_and_pairs.h5`: selected singles and sampled synthetic pairs
  with `theta_zx`, `theta_zy`, and pair distance.

## Intentionally not done anymore

The cleaned workflow deliberately removes old exploratory machinery:

- random thinning and salt-file blinding,
- sideband or observed-doublet normalization,
- box-shaped ROI cuts,
- Ar39 multiplicity explanation studies,
- old geometry-toy and voxel-overlay plots,
- private coordinator tables or hidden unblinding state.

Those studies may be useful historical context, but they are not part of the
canonical collaborator workflow in this repository.
