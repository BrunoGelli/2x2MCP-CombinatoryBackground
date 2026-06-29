# 2x2 MCP combinatorial-background workflow

This repository contains the canonical collaborator workflow for estimating the accidental/combinatorial double-blip background in the DUNE ND-LAr 2x2 demonstrator millicharged-particle (MCP) search.

The repo has intentionally been reduced to one workflow:

- `notebooks/combinatorial_background_roi_blinded.ipynb` — analysis notebook for the ROI-blinded combinatorial-background study.
- `scripts/make_doublet_blinding_mask_axis_symmetric.py` — deterministic HDF5 mask builder for double-blip candidates.
- `docs/data_format.md` — concise notes on the supported HDF5 layouts.
- `requirements.txt` — lightweight Python dependencies for running the script and notebook.

## Physics purpose

MCP-like candidates in the 2x2 search are pairs of isolated blips aligned with the beam direction. The dominant accidental background is estimated from data by pairing single-blip events randomly to construct a synthetic double-blip distribution. This singles-driven Monte Carlo estimates the doublet shape and pair-level acceptance from data rather than normalizing to an observed doublet control count.

## Blinding policy

The signal-like region is excluded with a hard circular ROI in projected angular space:

```text
sqrt((theta_zx - center_zx)^2 + (theta_zy - center_zy)^2) <= theta_radius
```

The ROI is axis-symmetric by default, so +z- and -z-aligned doublets are treated equivalently. Use the default axis-symmetric mode for analysis. The directed +z-only option exists only for debugging.

There is no random thinning, no salt file, no private coordinator table, no sideband normalization, and no box-shaped ROI in the canonical workflow.

## Analysis modes

The blinding script supports four modes:

| Mode | Trigger selection | ROI cut |
| --- | --- | --- |
| `nop` | allow all trigger types | no ROI cut |
| `background` | allow only non-beam/background trigger events | no ROI cut |
| `signal` | allow beam/signal trigger events | exclude circular ROI |
| `signal+background` | allow all trigger types | exclude circular ROI |

The output has `blind_mask=True` for candidates hidden from the analysis and `visible=True` for candidates allowed through.

## Background model and normalization

The expected accidental doublet count is normalized from the number of single blips:

```text
N_double_expected ≈ N_single^2 / (2 N_events) × pair_acceptance × ROI_keep_fraction
```

The singles-driven MC estimates the doublet shape and the relevant pair-level acceptance, including the minimum-distance cut and the ROI keep fraction when the selected mode applies the ROI exclusion. Do not normalize this MC to the observed doublet count or to sidebands.

## Supported input formats

Two HDF5 layouts are supported.

1. Legacy event-group format:

   ```text
   /events/<event_key>/x
   /events/<event_key>/y
   /events/<event_key>/z
   /events/<event_key>/labels
   ```

2. Hong Cai flat blip-table format:

   ```text
   /normal_hits/data
   /normal_hits/ref_region
   ```

   The flat table should include fields such as `x`, `y`, `z`, `cluster_id`, `Q`, `event_id`, and `beam_type`. The script auto-detects common field names; override them with `--x-field`, `--label-field`, `--event-id-field`, or `--trigger-type-field` if needed.

See `docs/data_format.md` for a short format reference.

## Setup

Create an environment with the lightweight dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the blinding script

Example for the preferred Hong flat-table format in signal mode:

```bash
python scripts/make_doublet_blinding_mask_axis_symmetric.py input_blips.h5 \
  --out outputs/signal_blinding_mask.h5 \
  --input-format normal-hits \
  --mode signal \
  --theta-radius-deg 5.0 \
  --center-zx-deg 0.0 \
  --center-zy-deg 0.0 \
  --min-dist-cm 10.0 \
  --overwrite
```

Example for the legacy event-group format when no trigger field exists:

```bash
python scripts/make_doublet_blinding_mask_axis_symmetric.py legacy_events.h5 \
  --out outputs/legacy_signal_mask.h5 \
  --input-format event-groups \
  --mode signal \
  --default-trigger-type 1 \
  --overwrite
```

Useful switches:

- `--mode {nop,background,signal,signal+background}` selects trigger and ROI behavior.
- `--axis-symmetric-roi` is the default and should be used for the analysis.
- `--directed-roi` disables axis symmetry for debugging only.
- `--signal-trigger-value` and `--background-trigger-value` define the trigger labels, defaulting to `1` and `0`.
- `--hits-dataset`, `--ref-region-dataset`, and `--events-group` override HDF5 paths.

## Run the notebook

Start Jupyter from the repository root after installing requirements:

```bash
jupyter notebook notebooks/combinatorial_background_roi_blinded.ipynb
```

Update the input-file and output-directory configuration cells at the top of the notebook for your local HDF5 files. Run only the lightweight setup/inspection cells first. Do not run heavy MC production cells unless you have intentionally chosen a cheap configuration or are ready for the full production runtime.

## Outputs to expect

The blinding script writes a separate HDF5 mask file and never modifies the input HDF5. The mask file contains:

```text
/blinding/event_key
/blinding/mask
```

`/blinding/mask` includes candidate identifiers, event indices, trigger type, `blind_mask`, and `visible`. The file also stores metadata describing the mode, ROI settings, fiducial cuts, and candidate geometry needed for checks.

The notebook produces analysis tables and diagnostic plots for the ROI-blinded singles-driven doublet estimate. Write generated artifacts to an ignored or local output directory rather than committing large production plots.

## What is intentionally not done anymore

The cleaned workflow deliberately removes earlier exploratory ideas and obsolete scripts, including:

- random thinning of candidates outside the ROI,
- salted random masks or private coordinator salt files,
- sideband normalization of the singles-driven MC to observed doublets,
- box-shaped ROI cuts,
- Ar39 multiplicity explanation studies,
- old geometry toy overlays,
- exploratory `CombinatoryBg_vXX` notebooks and archived plot directories.

Keep new development focused on the canonical script and notebook unless collaborators agree to add a documented extension.
