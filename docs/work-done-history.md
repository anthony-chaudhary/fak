# Work-done history

WORK DONE history is opt-in, privacy-safe, and baseline/workload aware. A history file stores at most 256 `fak.info.work-done-history/1` records. Records contain query timestamps/windows, evidence-bearing metrics, stable source enums, baseline identity, and SHA-256 workload/run identities. Raw workload keys, run keys, prompts, tool arguments, and filesystem paths are never persisted.

Compare and retain a session total:

```console
fak info --gateway-url http://127.0.0.1:PORT --work-done-json \
  --work-done-history "$HOME/.cache/fak/work-done.jsonl" \
  --workload-key coding-default --run-key launch-42
```

Use the same history options without `--work-done-json` to add a historical cue to the live TUI. Keys should describe a stable workload cohort, not contain prompt text or paths; they are hashed before storage.

The query export contains the exact prior/current records used and one comparison. Comparison is refused or separated when workload identity changes, baseline ID/fingerprint changes, a window reset occurs, required evidence is unavailable, history is sparse, or either window is shorter than one second. No percentage is inferred in those cases.

Compatible records report token and call deltas as `improved`, `regressed`, or `steady`. Attribution is `fak_mechanism_change` when source effects changed and `total_changed_same_source_mix` otherwise. Workload and baseline discontinuities are named explicitly rather than drawn as a continuous trend.

The ledger is rewritten with owner-only permissions and tail retention. This history is an operational convenience, not a billing record; preserve the baseline, source, and evidence contracts documented in [`work-done-baselines.md`](work-done-baselines.md), [`work-done-sources.md`](work-done-sources.md), and [`work-done-query.md`](work-done-query.md).
