# Codex UserPromptSubmit live dogfood — 2026-08-26

## Verdict

The repo's live Codex session exercised the guarded opt-in path successfully, but the installed-copy activation lagged committed source. The guarded hook appended four allow rows during real top-100 issue work. Direct probes of the installed binary returned success but added no permissive or hardened rows, and its source-supported `--usage-summary` flag was rejected. The activation defect is filed as #9361.

## Operating envelope

- Repository: `fak` shared `main` checkout.
- Workload: live coordination, implementation, tests, commits, and issue closure for the fixed top-100 backlog snapshot; not a synthetic fixture.
- Concurrency: one prompt at each UserPromptSubmit boundary.
- Duration: four observed guarded prompt submissions between `2026-08-27T01:26:02Z` and `2026-08-27T01:32:02Z` (UTC).
- Privacy: ledger rows contain only schema, UTC timestamp, ISO week, mode, and outcome; no prompt text, path, session ID, or host identity is reproduced here.

## Observed receipt

The live ledger at the configured Codex home contained these aggregate facts:

| Schema | Week | Mode | Outcome | Rows |
|---|---|---|---|---:|
| `fak.sessions.codex_submit_usage/1` | `2026-W35` | `guarded` | `allow` | 4 |

Every observed live row was an allow. No deny or error outcome was present, and no private prompt payload was recorded.

## Mode probes and defects

| Probe | Exact result | Interpretation | Defect |
|---|---|---|---|
| Active repo `UserPromptSubmit` hook during guarded work | Four new `guarded/allow` ledger rows | The guarded opt-in path was activated and persisted bounded usage evidence. | None surfaced in the guarded path. |
| Installed `fak sessions codex-loop-hook` with neither guard nor hardened opt-in | Exit 0, byte-silent, no new ledger row | Permissive behavior itself remained fail-open and silent, but the running binary did not contain the committed usage ledger behavior. | #9361 |
| Installed `fak sessions codex-loop-hook --hardened` | Exit 0, byte-silent, no new ledger row | The hardened invocation completed, but installed-copy activation prevented mode accounting from being witnessed. | #9361 |
| Installed `fak sessions codex-loop-hook --usage-summary` | `flag provided but not defined: -usage-summary` | The installed binary predated the committed summary surface; source presence did not prove running activation. | #9361 |

Marker key `codex-userpromptsubmit-installed-usage-summary-activation` makes #9361 the deduplicated tracker for the only defect surfaced. No additional distinct defect was observed.

## Conclusion

The real repo run meets #9241's one-prompt/one-turn minimum and proves the guarded mode is in use rather than merely documented. It also caught an activation-depth failure: source and tests supported the usage summary, while the installed command serving the live probes did not. The dogfood issue can close with this readout; #9361 owns activation and the missing permissive/hardened installed-copy witness.
