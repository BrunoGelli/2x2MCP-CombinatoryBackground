# Supported input data formats

## Legacy event-group format

```text
/events/<event_key>/x
/events/<event_key>/y
/events/<event_key>/z
/events/<event_key>/labels
```

Optional group attributes `beam_type` or `trigger_type` are used for mode selection.
If neither exists, events are treated as beam/signal triggers.

## Hong Cai flat blip-table format

```text
/normal_hits/data
/normal_hits/ref_region
```

The `/normal_hits/data` structured table must contain `x`, `y`, and `z` fields.
Recommended fields are `cluster_id`, `Q`, `event_id`, and `beam_type`.
`event_id` is used for the normalization denominator and `beam_type` is used for
mode selection when present.
