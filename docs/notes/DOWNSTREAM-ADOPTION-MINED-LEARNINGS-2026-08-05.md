# What a downstream contract deployment taught fak — mined, verified, and mostly refuted

**Date:** 2026-08-05 · **Companion to:**
[`DOWNSTREAM-REVIEW-VERIFIED-2026-08-05.md`](DOWNSTREAM-REVIEW-VERIFIED-2026-08-05.md) and the
[review it corrects](DOWNSTREAM-FAK-ADOPTION-REVIEW-2026-08-05.md)

**In one line:** a private downstream deployment of fak was mined for mechanisms fak should import,
and two thirds of the candidate imports turned out to be things fak already ships under a different
name.

## 1. What was reviewed

A private, contract-built deployment adopted fak's substrate wholesale — `fak guard`,
`fak worktree worker prepare`, `fak orient`, `fak sweep`, `fak index work`, `fak intent`, and the
DOS kernel — then ran roughly 30 concurrent agents against **one shared checkout** and instrumented
the cost. In several places it extended a fak mechanism or solved the same problem differently.

Only the mechanism, the reasoning, the failure mode, and the git/gate/CLI engineering are carried
across. The client, its domain, its data sources, its hosts, its people, and its repo path appear
nowhere. § 6 states the filter that was applied.

## 2. Coverage and confidence — read this before any finding below

Seven areas were mined; **125 candidate mechanisms** came back. Each was then checked against fak,
and every surviving gap claim was handed to an adversarial pass instructed to refute it.

| Stage | Result |
|---|---|
| Mined | 7 of 7 areas, 125 candidates |
| Verified against fak | **3 of 7 areas** (`cli-surface`, `disambiguation`, `evidence-and-results`) |
| Never verified | **4 of 7 areas** (`doc-discipline`, `fleet-economics`, `loops-and-hooks`, `repo-gates`) |
| Refuted where checked | **27 of 41**, and the load-bearing gap claim in **3 of 3** areas |

⛔ **The refute rate is the finding.** In all three areas that reached the adversarial pass, the
headline gap claim was false, and the reason was the same every time: the miner searched for the
*downstream's* vocabulary and missed fak's word for the same thing. That is the identical failure
mode recorded in the companion note against a separate, earlier review. Two independent passes on
the same day, the same error, so it is a property of the method rather than of one reviewer.

⇒ **fak's substrate is further ahead of its own documentation than it is of its downstream.** The
cheapest fix in this note is not a mechanism. It is a name.

⚠️ **The four unverified areas are excluded from every finding below.** Their agents died — three on
upstream errors, one stalled on all six attempts. Read that as *not yet*, not as *nothing there*.
Their mined candidates are unchecked against fak and must not be cited as gaps.

## 3. What survived, ranked

Fourteen of forty-one survived. These are the ones worth acting on. Claims marked ✅ were
re-derived independently for this note; the rest carry the verification pass's citation.

| # | Learning | State at HEAD | Change |
|---|---|---|---|
| 1 | A verb-count KPI must count *every* dispatch, not one switch | ✅ `internal/heavinessscore/heavinessscore.go:163` anchors on `dispatchSwitchHeader = "switch os.Args[1]"`, but `cmd/fak/main.go` has two switches: `:63` and `:629` (`switch name`, inside `dispatchPrimaryVerb`) | Import `internal/devindex`, which already parses both forms, instead of re-parsing. `AppealWired` (`:267`) shares the blind spot |
| 2 | Exit codes are the contract every wrapper reads | ✅ `cmd/fak/flagutil.go:11-13` states "`-h/--help` is NOT a usage error (rc 0)"; `cmd/fak/help_test.go:256` pins `runCommitCommand(--help) = 2` — the front door contradicts its own written rule | Write the policy into `AGENTS.md`, add a verdict-carrying error type, and derive one `-h` test from `devindex.Verbs()` |
| 3 | A raw vendor command an agent may type is a law you have not mechanized | `internal/gitgate` is exactly this for `git`: deny-as-value at the call boundary, laundering-proof through pipes, `$(...)` and `bash -c` | Extend the pattern from `git` to `dos`; ship `fak lanes` over `dos lease-lane` |
| 4 | A lease acquire returns a *receipt*, not an outcome | `tools/dos_fleet_lease.py` `do_acquire` journals a real kernel acquire, then returns `EXIT_DENIED` on a mis-pick **without calling `kernel_release`** | Release the auto-picked lane before denying; stop discarding stderr in `kernel_acquire`; split `REASON_HELD_REMOTE` into held-vs-busy |
| 5 | Put the non-vacuous control *inside the census*, not only inside the benchmark | Doctrine is house practice for measurements (`cmd/deletioncert/main.go:126` aborts on "witness is vacuous"; `cmd/ctxplanbench/main.go:236`), but no grep- or pathspec-based audit carries a known-positive | Apply it to the censuses that sweep ~900 packages and 1,664 tracked `.md` files |
| 6 | "The anchor file exists" is not "the anchor page mentions this row" | ✅ `tools/concept_disambiguation_scorecard.py` `kpi_anchored` calls only `exists(anchor)`. Of **403** rows anchoring `docs/fak/concept-glossary.md`, **167** have their canonical name nowhere in that file and ~263 have no matching `###` heading. **All 403 grade `crystal`** | Require the anchor page to carry *this* row's heading or id |
| 7 | A citation scan must exclude the map and its own generated pages | `_DOC_DIRS = ('docs',)` with no exclusion, so `grounded_tokens` harvests every heading under `docs/`. Five rows ground themselves against a README that `internal/conceptcatalog/freshness.go:20` generates *from those same rows* | Exclude generated markdown. The Go side already excludes its data dir |
| 8 | A portfolio sum hides a per-metric regression | `tools/scorecard_control_pane.py` has a hard per-metric **grade** ratchet, but the **raw** axis is still `if total_delta > 0` over the portfolio | Promote the raw axis to per-metric, staged behind the demotion knob that already exists |
| 9 | A number without its operating point is not a result | `internal/provenance/runmanifest.go` already requires `Cost`, `Tier`, `Normalization`, `CacheState`, `Tolerance`, `Baseline` | Add the missing quartet to the same struct: aggregation rule, scoring threshold, retained-item cap, reference-set hash |
| 10 | An outcome owes an artifact, and the debt is paid at the commit | `tools/githooks/pre-commit` already blocks on `INDEX_SYNC` and `PROVENANCE_LABEL` | Ship `fak closeout` as an extension of that block list, not as a stop-time report |

Four narrower survivors: per-lane contention telemetry folded out of the lane journal; a named
`Planned` verb tier; `fak mcp probe --strict` issuing `tools/list` so a third-party server's tools
are inventoried; and a `RESTATED` status, absent from every fak status vocabulary.

## 4. Defects this surfaced in fak

Each was found by reading fak, not the downstream. The first six were re-derived for this note.

1. ✅ **`internal/heavinessscore` undercounts CLI verbs.** It bounds the dispatch block to
   `switch os.Args[1]`, missing the 43 verbs dispatched from the second switch at
   `cmd/fak/main.go:629`. The KPI grades against `verbSoftLine=90` / `verbHardCeiling=200` on a
   short count.
2. ✅ **`fak commit --help` exits 2**, contradicting `cmd/fak/flagutil.go:11-13`. The behaviour is
   pinned by a test, so the test and the convention disagree in writing.
3. ✅ **`kpi_anchored` grades on file existence alone**, so 403 rows grade `crystal` while 167 of
   them name a concept the anchor page never mentions.
4. ✅ **Five catalog rows ground themselves** against a generated README built from those rows.
5. ✅ **Two disjoint sets are both called "the closed vocabulary."** `internal/abi/reasons.go`
   carries 19 tokens; `dos.toml` carries 75 `[reasons.*]`; the overlap is **zero**. `docs/FAQ.md`
   and `AGENTS.md` each claim the phrase for their own set, so `dos check-reason DEFAULT_DENY`
   reads `UNCLASSIFIED`.
6. ✅ **`docs/explainers/README.md` does not exist**, though it is referenced as a front door.
7. **`tools/dos_fleet_lease.py` `do_acquire` leaks an auto-picked lane** — it journals the acquire
   and then denies without releasing.
8. **`kernel_acquire` discards stderr**, so a kernel diagnostic parses as `{}` and every distinct
   failure collapses into one return-code branch.
9. **`foreign_record_live` is a bare clock overlay** with no positive corroboration, and nothing
   reaps the dos lane journal. The design to copy is in-tree at
   `internal/procguard/deadowner.go`, which keys the reap on owner liveness rather than idleness.
10. **`tools/docs_scorecard.py --scope all` walks untracked scratch**, reporting
    `reachability_pct: 7.4` while descending build-run directories. The scorecard skill forbids
    exactly this.
11. **`GENERATED_DOCS` in `tools/docs_scorecard.py` is a one-entry name set**, so roughly 20
    generated docs are scored as hand-written prose. The correct shape, with the reasoning inline,
    is two files away in `tools/scrub_hardware_names.py`.
12. **Two silent-swallow sites.** `internal/devindex/freshness.go:230` swallows an unreadable
    subtree; `tools/check_index_sync.py:175` prints `clean` with no denominator.
13. **`internal/devindex.OrphanNotes()` carries a false "can never disagree" comment.** It walks
    the working tree while the gate folds over `git ls-files`. The behaviour is safe; the comment
    is wrong.
14. **Two vacuity floors are missing**: `tools/coverage_report` returns 100% on an empty
    denominator, and `internal/agentsindex/resident_drift_test.go` skips on a marker-free
    `CLAUDE.md`, which is the state that file is in.
15. **`tools/bench_catalog.py` has neither `--check` nor a `new` skeleton verb**, and four
    manifests under `experiments/` sit off-schema.

## 5. Why 27 of 41 were dropped — fak's word for the same thing

This is the useful half of a refuted finding. Each row below is a mechanism a reviewer asked fak to
build, next to the shipped thing they did not find.

- **Manifest↔dispatch equivalence, both directions.** `verbs_freshness_test.go` `TestVerbManifest`
  fails on a documented-but-undispatched verb; `gate_verbtier.go` emits `VERB_UNTIERED` for the
  converse, reusing `devindex.DispatchVerbs` so there is no second authority to drift.
- **`cmd/fak/servewiring.go` is the generated-table-scored-against-source pattern, shipped.** Each
  row binds feature to flag to config field to call site, under a closed verdict enum, re-read from
  source each run, with an unlisted config field reported as `UNAUDITED`. Three separate proposals
  collapse into "port this row shape."
- **`internal/gitgate` already mechanizes the managed-command law** for the highest-risk vendor
  command, refusing force-push, amend, `add -A`, `--no-verify` and `reset --hard` even through a
  pipe or `bash -c`.
- **`internal/leaseref` is stronger than an untrusted parked-ref ledger**: deterministic normalized
  keys, an advisory TTL, and a CAS that re-reads and reports the winner as `INTENT_COLLISION`
  naming holder, session, target and age.
- **`internal/laneadmit` makes the lane *name* a refusal axis**, and `dos-plan-price` refuses a
  colliding partition before any agent launches.
- **`CONCEPT_ADMISSION` and `CONCEPT_FRESHNESS` are registered block-mode pre-commit gates.**
  `CheckFresh` regenerates the index in scratch and byte-compares — a generated crosswalk, better
  than the hand-counted one the downstream watched rot.
- **Content-beats-name classification** is already doctrine: `tools/bench_provenance.py` is an
  explicit priority ladder ending in fail-closed `unknown`, built after a name classifier tagged 51
  of 55 runs unknown.
- **The claim register ships and is CI-gated**: 52 rows, stable ids, a closed status enum,
  retraction reason and `superseded_by`, checked inside `make ci`.
- **Symlink-walk immunity by construction.** `ReadTrackedTree` runs `git ls-files -z` once; git
  stores a symlink as a blob, so a tracked-path walker cannot descend one.
- **"Has teeth" is fak's name for the must-NOT-fire control**, and it is doctrine rather than
  per-gate practice: 11 files carry the naming and 236 carry an explicit vacuity guard.
- **Adversarial re-derivation is house discipline with published what-broke tables** in `STATUS.md`
  and `CLAIMS.md`, including a `REFUTED -> FIXED` table naming a `t.Errorf` that should have been
  `t.Fatalf`.
- **Unwitnessed-completion detection** already demotes a subagent result that claims a ship but
  carries no artifact witness, because a cut-off child can return an empty success the parent folds
  clean.

## 6. What was filtered out

Applied as a hard filter, and scanned mechanically over this file before it landed: client identity
and every compound identifier built from it, the private org and repo path, issue URLs and bare
issue numbers from that repo, people and email addresses, host and rig names, network topology and
cloud vendors, the product domain and its entire sensing vocabulary, the client's data-plane
vendors, tables and columns, and absolute cost totals tied to the client's account.

The scan flagged one token, `crystal`, which is fak's own concept-verdict value and unrelated to
the downstream's use. Nothing in this note depends on anything filtered. Every measured figure
above is from **fak's own tree**.

## 7. Next checkable steps

1. **Re-run the four dead areas** (`doc-discipline`, `fleet-economics`, `loops-and-hooks`,
   `repo-gates`) through verify and refute. Their 84 candidates are unchecked, and on this pass's
   evidence roughly two thirds will be things fak already ships.
2. **File defects 1–6.** All six are re-derived and each is a small, local fix.
3. **Name the things fak already has.** The 3-of-3 refute rate says the gap is discoverability. A
   vocabulary crosswalk from a common description to fak's identifier would have prevented every
   refuted finding in § 5.
4. **Land the coverage axis** from the companion note. It remains the highest-value single change
   across both reviews.
