# Guarded-cohort regrowth witness — 2026-08-06 (#5254)

Captured output of `fak session compact-audit` over the local Codex rollout corpus
(`~/.codex/sessions`, whole corpus, no `--since`), run three ways back to back so the
three cohorts describe one snapshot of a live directory. Scrubbed: aggregate-only, no
filesystem paths, no session cwd, no prompt or tool-output bodies.

```
fak session compact-audit                --json --scrub --aggregate-only   # whole corpus
fak session compact-audit --guarded-only --json --scrub --aggregate-only   # fak-routed
fak session compact-audit --guarded-only --since 2026-07-28 \
                                         --json --scrub --aggregate-only   # post-port
```

This artifact is DoD item 3 of [#5254](https://github.com/anthony-chaudhary/fak/issues/5254),
run against the retargeted witness the 2026-07-19 decision note
(`docs/notes/COMPACTION-REPEATED-TOOL-RESULT-DEDUP-2026-07-19.md`) specified after it
measured that ~95% of the audited corpus never crossed fak's wire. It reports **two
results the DoD did not anticipate**, and neither is the reduction the DoD asked for.

## The filter works, and the mixed-provenance problem is confirmed at scale

| quantity | value |
|---|---|
| rollouts in the corpus | 3,205 (3.53 GB) |
| guard witness ledger records | 158 |
| rollouts matched to the ledger (`guarded_sessions`) | **122** |
| rollouts NOT matched (`unguarded_sessions`) | 3,084 |
| fak-routed share of the corpus | **3.8%** |

The 2026-07-19 note measured 120 / 2,448 (4.9%) on a `--since 2026-06-15` slice. This
run measures 122 / 3,206 (3.8%) on the whole corpus. The join is stable; the corpus
grew and the guarded numerator did not. An unscoped before/after over this corpus is
therefore measuring ~96% traffic no gateway-side transform could touch.

## Result 1: the corpus-wide headline reproduces

| quantity | issue (2026-07-18) | this run (2026-08-06) |
|---|---|---|
| post-fire windows with telemetry | 1,510 | 1,587 |
| windows carrying `REPEATED_TOOL_RESULT` | 296 (19.6%) | 300 (**18.90%**) |
| `tool_result/shell_command` bytes | 1,032,156,451 (66.4%) | 1,163,678,420 (**67.6%**) |
| duplicated inside that class | 12,458,846 B / 1,587 rows | 12,668,851 B / 1,609 rows |
| fast vs slow cohort median tool calls | 170 vs 172 | 170 vs 174 |

The class is not shrinking on the box as a whole, and the cohort line still holds: fast
rebounds are not caused by more tool calls.

## Result 2: the routed cohort barely carries the class — and always did

| cohort | rollouts | windows | `REPEATED_TOOL_RESULT` windows | share |
|---|---|---|---|---|
| whole corpus (mixed provenance) | 3,205 | 1,587 | 300 | 18.90% |
| **fak-guarded** | 122 | 107 | **3** | **2.80%** |
| bare (whole − guarded) | 3,083 | 1,480 | 297 | 20.07% |

`tool_result/shell_command`, same three cohorts:

| cohort | rows | bytes | dup_rows | dup_bytes | dup share of class |
|---|---|---|---|---|---|
| whole corpus | 248,209 | 1,163,678,420 | 1,609 | 12,668,851 | 1.089% |
| **fak-guarded** | 14,446 | 76,434,188 | 28 | **258,259** | **0.338%** |
| bare | 233,763 | 1,087,244,232 | 1,581 | 12,410,592 | 1.141% |

Across all `tool_result/*`: guarded 258,259 dup bytes of 77,599,550; bare 12,415,181 of
1,116,063,613.

**This is not a before/after of the dedup port.** The port landed 2026-07-27
(`2ab350829c`). The guard ledger's `guarded_at` stamps are 63 on 07-14, 54 on 07-15,
30 on 07-16, 5 on 07-17, 2 on 07-18, then 1 on 08-04, 2 on 08-05, 1 on 08-06 —
**154 of 158 records (97.5%) predate the port.** The guarded cohort measured above is
essentially a *pre-port* population, and it already carries a 7.2× lower
`REPEATED_TOOL_RESULT` share than the bare one.

## The confounder, stated rather than priced in

The guarded cohort is not a matched control, and the difference above must not be read
as a fak effect:

| quantity | whole corpus | fak-guarded |
|---|---|---|
| windows that rebounded to >=200k | 757 | **0** |
| windows censored (next fire / rollout end) | 830 | **107** |
| fast-cohort (<=30 min) windows | 513 | **0** |
| median growth tokens in-window | 82,717 (slow) / 215,942 (fast) | 72,127 |
| median pre-fire resident tokens | 237,820 | 92,903 |

Every guarded window is censored: none ran long enough to rebound. `REPEATED_TOOL_RESULT`
is a within-window property, so a cohort of uniformly shorter, lower-growth windows has
structurally fewer chances to re-append a byte-identical span. A 2.80%-vs-20.07% gap on
an unmatched cohort with 0/107 rebounds is consistent with *either* a real fak effect or
pure observation-length skew, and this run cannot separate them.

## Post-port cohort: measurable population is EXHAUSTED, not clean

`--guarded-only --since 2026-07-28` (the first full day after the port):

| quantity | value |
|---|---|
| guarded rollouts since the port | **2** (4,115,466 B) |
| unguarded rollouts since the port | 158 |
| compaction fires across those 2 | **0** |
| post-fire windows | **0** — the result carries no `regrowth` block at all |

There is no post-port guarded window in existence on this host. DoD item 3's
"materially reduced on new sessions" is therefore **not measurable today**, and the
blocker has moved: it is no longer a missing capability (`--guarded-only` now exists and
is exercised above), it is a missing population. The next checkable step is arithmetic,
not engineering — accumulate fak-guarded Codex sessions that actually fire compaction,
then re-run these same three sweeps.

## What this licenses a reader to claim

- **Licensed:** the corpus-wide class share reproduces; the fak-routed cohort is 3.8% of
  the corpus; a scoped sweep is now runnable and self-describing (`provenance` rides in
  the JSON and survives `--scrub`).
- **NOT licensed:** that the #5254 dedup port reduced anything measured here. No
  post-port guarded window exists. Any reduction claim from these numbers would be
  attributing a pre-port cohort gap to a post-port change.
