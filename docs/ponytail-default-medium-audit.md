---
title: "Ponytail medium default dogfood audit (updated 2026-08-24)"
description: "Default-on is mechanically ready and its exact fragment is now benchmark-addressable; effectiveness is mixed / NOT-YET."
---
# Ponytail medium default dogfood audit (updated 2026-08-24)

## Verdict

**Default-on is mechanically ready and its exact fragment is now benchmark-addressable;
effectiveness is mixed / NOT-YET.** `fak agent` selects `ponytail:medium` unless the operator
explicitly passes `--work-profile standard`. The owned loop mediates that selection through
`BuildOwnedSystemBlock` before the system message reaches the planner, and the run report records
`work_profile` plus `work_profile_witness`.

## Dogfood witness

`TestRunReportCapturesDefaultWorkProfileWitness` drives the real owned `Run` loop with a capture
planner. It proves all three effects in one path:

1. an unset profile resolves to `ponytail:native:medium`;
2. the planner receives the fak-mediated medium fragment, not the historical static prompt alone;
3. the report captures the canonical selection and fragment digest.

The explicit `standard` path remains an off switch. The machine-readable witness is
[`_witnesses/ponytail-default-medium-dogfood-2026-08-14.json`](_witnesses/ponytail-default-medium-dogfood-2026-08-14.json).

## Exact-default gate arm

`fak armbench ponytail-gates` now keeps its baseline, pinned upstream Caveman, and pinned upstream
Ponytail comparators and adds `native_medium` as a fourth, runner-local arm. The new arm obtains its
system prompt from `syspromptmmu.DescribeWorkProfile`, passes the returned segment directly to the
provider invocation, and refuses unknown arms. It also refuses to run if the resolved canonical
profile or its pinned fragment digest drifts:

- canonical profile: `ponytail:native:medium`
- fragment witness: `sha256:fb03503c5d7d3e75155796ddf34c426ba5126f86c10d2b929c1f1796328fb7ed`

The no-spend receipt is
[`_witnesses/ponytail-default-medium-gates-dryrun-2026-08-24.json`](_witnesses/ponytail-default-medium-gates-dryrun-2026-08-24.json).
It records all four arm identities, passes the unchanged pinned `node:test` regression fixture
(4/4), and marks every provider-backed category for every arm `not_run`; consequently
`overall_pass` is honestly `false`. This receipt proves inventory, canonical rendering provenance,
digest pinning, and fail-closed dry-run accounting. It does **not** prove model quality.

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
single-trial and used the pinned Ponytail skill arm rather than the exact native-medium fragment.
The runner can now measure that fragment directly, but no new live `native_medium` result is claimed
by the 2026-08-24 no-spend receipt.
