# Work-accounting coverage

`fak info --work-coverage` reports the authoritative registry of shipped mechanisms whose effect must appear in WORK DONE, be explicitly non-additive, be intentionally excluded, or remain honestly unavailable. Use `--json` for `fak.info.work-accounting-coverage/1`.

```console
fak info --work-coverage
fak info --work-coverage --json
```

Every declaration names its authoritative producer and one status:

- `accounted`: mapped to a stable work-source ID and exclusivity group;
- `overlapping`: visible, but not independently summable; the reason and group are mandatory;
- `not_yet_measurable`: shipped behavior without a trustworthy live effect counter; the reason is mandatory and UI/query must say unavailable, not zero;
- `intentionally_excluded`: visible operational work that must not be converted into savings, such as safety interventions.

The registry currently accounts provider prompt caching, response/vDSO memoization, inline tool serving, compaction, stale-read context elision, native schema/tool filtering, and KV-prefix reuse. Cold-tool defer is explicitly overlapping. Cold-tool defer and model routing publish observed decisions plus optional paired-run calibrated deltas as explicitly overlapping counterfactuals; they are never added to provider/cache totals. Safety BLOCK/FIX/DEFER interventions remain visible in Safety but intentionally excluded from token/call savings.

`internal/workaccount.Validate` is the enforcement seam: an accounted mechanism without units, source mapping, and exclusivity group fails; unavailable/excluded entries without reasons fail; unknown statuses and duplicate IDs fail. Adding a measurable mechanism therefore requires a WORK DONE mapping or a typed, reasoned disposition in the same change.
