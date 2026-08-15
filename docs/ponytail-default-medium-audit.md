# Ponytail medium default dogfood audit (2026-08-14)

## Verdict

**Default-on is mechanically ready; effectiveness is mixed / NOT-YET.** `fak agent` now selects
`ponytail:medium` unless the operator explicitly passes `--work-profile standard`. The owned loop
mediates that selection through `BuildOwnedSystemBlock` before the system message reaches the
planner, and the run report records `work_profile` plus `work_profile_witness`.

## Dogfood witness

`TestRunReportCapturesDefaultWorkProfileWitness` drives the real owned `Run` loop with a capture
planner. It proves all three effects in one path:

1. an unset profile resolves to `ponytail:native:medium`;
2. the planner receives the fak-mediated medium fragment, not the historical static prompt alone;
3. the report captures the canonical selection and fragment digest.

The explicit `standard` path remains an off switch. The machine-readable witness is
[`_witnesses/ponytail-default-medium-dogfood-2026-08-14.json`](_witnesses/ponytail-default-medium-dogfood-2026-08-14.json).

## Effectiveness audit

The latest live provider witness remains
[`_witnesses/armbench-ponytail-gates-live-final-2026-08-14.json`](_witnesses/armbench-ponytail-gates-live-final-2026-08-14.json):

| arm | behavior | correctness | robustness |
|---|---:|---:|---:|
| baseline | 1/3 | 3/5 | 16/16 |
| Ponytail | 2/3 | 4/5 | 15/16 |

That is directionally positive for behavior and correctness, but not a clean win: Ponytail lost one
robustness case and retained failures in behavior and correctness. This audit therefore does **not**
claim broad effectiveness. It justifies default dogfood because the setting is reversible,
observable, and preserves the profile's correctness carve-outs.

A fresh OpenAI live attempt on 2026-08-14 failed with HTTP 401 because the configured environment
credential was a placeholder. No result from that attempt is counted. The captured provider audit is
single-trial and used the pinned Ponytail skill arm rather than this exact native-medium fragment;
those limits stay explicit in the JSON witness.
