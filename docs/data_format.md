# Supported input data formats

The canonical workflow accepts two HDF5 layouts.

## Legacy event-group format

Each event is stored under `/events/<event_key>/` with at least:

- `x`: blip/hit x positions in cm
- `y`: blip/hit y positions in cm
- `z`: blip/hit z positions in cm
- `labels`: cluster labels; negative labels are ignored

## Hong Cai flat blip-table format

The preferred flat-table format stores hits in:

- `/normal_hits/data`: compound dataset with fields such as `x`, `y`, `z`, `cluster_id`, `Q`, `event_id`, and `beam_type`
- `/normal_hits/ref_region`: event slices into `/normal_hits/data`

The blinding script auto-detects common field names. Use the `--*-field` options when a file uses non-standard names.
