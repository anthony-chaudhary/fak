# Verified default plus control versus scratch construction

This paired study tests one narrow product theory:

> Operators prefer adopting a useful verified harness and changing it through visible control seams over reconstructing the equivalent harness from scratch.

It does **not** count agent dogfood, maintainer runs, surveys without completed tasks, or the older generator-versus-handwritten creation benchmark as human preference evidence.

## Same task, two arms

Each independent participant completes the exact bytes in `task-card.md` twice on one machine under the same declared network condition:

1. **default-control** — receive a verified default lock and use `harness inspect`, `override` or `derive`, `preview`, and final verification;
2. **scratch** — receive the behavioral requirements but no product manifest, lock, generated files, or task answer; construct the equivalent harness product themselves and verify it.

Randomize `pair_order` before either clock. Balance `default-first` and `scratch-first` across the first two complete pairs. Use fresh directories and caches. Never drop the second arm after seeing the first result.

## What gets measured

Every arm preserves a raw receipt containing:

- exact start/stop timestamps and elapsed seconds;
- every command, error, and help request;
- success and independent verification result;
- post-task confidence from 1 (would not operate this) to 5 (would operate this unaided);
- binary version and commit identity;
- digest of the completed artifact;
- for default-control, the imported lock ID plus captured inspect and preview evidence; and
- after arm two, explicit preference (`default-control`, `scratch`, or `none`) and a short reason.

The operator archives privacy-reviewed receipts under `receipts/`, computes each exact file digest, and appends only `{source,digest}` to `study.json`. Evaluation re-reads the raw bytes and fails closed on edited/missing receipts, duplicate participants, incomplete pairs, contradictory clocks/order, missing provenance, or success without verification.

```text
fak harness study control --input docs/benchmarks/harness-control-study/study.json
```

Exit `3` means the study is still `not_yet`; exit `0` means the declared minimum pair count is admissible and both arms have verification evidence. `measured` does not itself mean the preference hypothesis won: report the arm metrics and preference counts as observed.

## Human handoff

1. Record random privacy-safe `person-…`, `pair-…`, and machine slugs. Do not record names, handles, emails, employers, or account identifiers.
2. Confirm the participant has not worked on fak internals. Record broad experience outside the repository if needed.
3. Hash `task-card.md`; it must equal `task_digest` in `study.json` before either arm starts.
4. Build the study binary once from the pinned commit and record both `fak version` and commit. Do not use an older PATH binary—a guarded agent pilot hit exactly this ambiguity.
5. Copy only the assigned directory under `materials/` into a fresh arm directory. Give its `arm-card.md`, `task-card.md`, and the pinned binary; the default-control bundle includes its verified starting product, while the scratch bundle intentionally contains no product material. Record every additional hint as a help request.
6. Stop at the task card boundary, preserve failed attempts, and independently replay the stated verification commands.
7. Ask preference only after both arms. A preference without two completed artifacts is inadmissible.
8. Privacy-review and archive the raw receipts, then rerun the evaluator.

The first release target is two complete independent pairs for a directional read. A promotional claim requires a larger preregistered sample and confidence interval; this repository must not turn a two-person pilot into a market-wide statement.
