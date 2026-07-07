# Benchmark Authority — redesign: record → generated view → freshness gate

**Status:** proposal + complete backfill. The record now carries **every** authority claim (Quick-Reference + tombstoned), and the generator, generated view, and freshness gate are shipped. What remains is the operator's flip.
**Owner decision required:** approve the migration path in §7, then flip `BENCHMARK-AUTHORITY.md` to the generated view.

---

## TL;DR

`BENCHMARK-AUTHORITY.md` is unreadable for a human operator *and* for an agent because it makes one markdown table do three incompatible jobs at once. The fix is the pattern this repo **already uses** for the support-maturity matrix: keep the numbers in a **structured record**, **generate** the human view from it, and a **freshness gate** stops the two from drifting.

This change ships that pattern as a runnable proof:

| Piece | File | What it is |
|---|---|---|
| **Record** | `docs/benchmarks/registry.jsonl` | one JSON object per claim, fixed field grammar |
| **View** (generator) | `tools/benchmark_authority.py` | renders a scannable, tiered markdown from the record |
| **Generated view** | `docs/benchmarks/AUTHORITY-GENERATED-SAMPLE.md` | generated output over all 52 authority claims — the "after" |
| **Gate** | `tools/benchmark_authority.py --check` + `benchmark_authority_test.py` | re-render and diff; CI-wired in the `Makefile` |

It does **not** touch the canonical `BENCHMARK-AUTHORITY.md` yet — that migration is the operator's approve-and-flip step (§7).

---

## 1 · The problem, precisely

`BENCHMARK-AUTHORITY.md` is a single markdown table with 40+ rows that is simultaneously:

1. **The record** — the single source of truth; "any number claimed elsewhere must trace back to an entry here."
2. **The human reading surface** — what an operator opens to find "the number to quote."
3. **The machine parse target** — `claim_repro_scorecard.py`, `bench_provenance.py`, `docs_scorecard.py`, `fresh_status.py`, `bench_dx_scorecard.py` all read it.

These three jobs pull in opposite directions, and the table loses all three:

- **Cells are 200–450 words.** The `Artifact` column carries the number *and* the methodology *and* the retraction *and* three honesty fences *and* the reproduce command, mashed into one table cell. The compaction row is ~450 words with a nested retraction inside a cell. You cannot scan a table whose cells are essays.
- **No tiering.** The ~10 canonical/live numbers an operator actually quotes are interleaved with ~30 supporting, gated, pending, stale, and retracted rows. Nothing separates "quote this" from "do not cite — withdrawn."
- **Status is inline and inconsistent.** `STALE`, `RETRACTED`, `GATED`, `PENDING` appear as ad-hoc prose (`~~strikethrough~~`, `❌`, `⚠️`) inside cells, not as a field you can filter on.
- **Fences are prose, not data.** The honesty caveats — the most valuable part of the doc — are buried mid-cell where neither a human skim nor a machine can enumerate them.

The result: the operator can't find the headline; the agent gets one 450-word line per `|`; and every edit is a hand-merge into an essay, which is exactly how the doc accreted a double-counted retraction and a citation to a non-existent commit in the first place.

## 2 · Why you can't fix this by editing the doc

The obvious move — "add a Quick Reference of headline numbers at the top" — **violates the doc's own first rule.** A hand-copied second table of numbers is a duplicate, and "any number claimed elsewhere must trace back" means duplicates drift. Hand-maintenance *is* the disease; more hand-maintenance is not the cure.

The numbers must live in **one** place, and every view must be a **pure function** of that place.

## 3 · The fix (already proven in this repo)

Row `support-maturity-matrix` in the authority is *itself* generated: `docs/HARDWARE-MATRIX.md`'s support-maturity block is produced by `fak support-maturity-scorecard --write-doc`, and a stale cell reds `--check-doc` (CI gate `TestSupportMaturityMatrixDocFresh`). The maturity grid is a **record** (`internal/covmatrix`), the doc block is a **generated view**, and a **freshness gate** keeps them honest.

Apply the same three-part shape to the benchmark authority:

```
 docs/benchmarks/registry.jsonl        tools/benchmark_authority.py        --check (CI)
 ───────────────────────────────       ────────────────────────────        ────────────
        THE RECORD              ──►          THE VIEW(S)           ──guard──►  FRESHNESS
 one object per claim,               scannable tiered markdown,               re-render,
 fixed field grammar,                machine parse targets                    diff, exit 1
 fences as a LIST                    (both pure functions)                    on drift
```

- **Record** — structured data, so `status` is an enum you can filter, `provenance` is `bench_provenance`'s closed vocabulary, and `fences` is a list you can enumerate.
- **View** — the generator emits a human view (tiered: headline table → by-status index → per-claim cards) and can emit machine views too (see §6). Every view derives from the record; none is hand-copied.
- **Gate** — `--check` re-renders and diffs; wired into the `Makefile` next to the other `--check-doc` gates. The doc **cannot** drift from the record.

## 4 · The record schema

One JSON object per line in `docs/benchmarks/registry.jsonl`:

| field | type | meaning |
|---|---|---|
| `id` | slug | stable anchor; how other rows cross-reference (`superseded_by`) |
| `claim` | string | the human title |
| `headline` | string | the quotable number (or `GATED …` / `PENDING …`) |
| `status` | enum | `canonical` \| `live` \| `gated` \| `pending` \| `stale` \| `retracted` |
| `tier` | int | 1 = headline, 2 = supporting, 3 = historical/dead |
| `provenance` | enum | `measured` \| `modeled` \| `functional` \| `unknown` (`bench_provenance.TAGS`) |
| `model` | string | model / geometry / harness |
| `baseline` | string | what the number is measured against |
| `commit` | string | private-lineage result commit (provenance, not a public reproduce handle) |
| `artifact` | string | the tracked JSON/`.md` witness — the verifiable anchor |
| `reproduce` | string | the command an outsider re-runs |
| `issue` | int? | tracking issue |
| `fences` | list | honesty caveats, one per item — the load-bearing field |
| `superseded_by` | slug? | for `stale` rows: the id that replaced them |
| `retraction` | string? | for withdrawn claims: the recorded reason |

The enums are the whole point: `status` lets a view show "quote these" separately from "withdrawn," and `provenance` keeps a **modeled** floor from ever being rendered as a **measured** number — the exact defect `check_provenance_labels.py` exists to catch.

## 5 · Before / after (the compaction row)

**Before** — one table cell, ~450 words, number + retraction + three fences + reproduce all fused (excerpt):

> `| **fak compaction trims ~a third…** | **~107K tokens shed per fire… ⚠️ A prior version claimed a per-session "~15%→~75%" share — RETRACTED as a double-counting artifact (see fences).** | … | _this commit_ | docs/nightrun/cache-savings.jsonl … **RETRACTION/PROVENANCE:** the compaction_shed_tokens field is CUMULATIVE… Fences: (1) cite shed per fire…; (2) valuing the shed…; (3) fak-AUTHORED… Reproduce: fak cachevalue report… |`

**After** — a record (fences as a list, retraction as its own field) rendered to a compact card: a one-line headline in the quote table, then a card with `status`, `provenance`, `model`/`baseline`/`commit`/`artifact`/`reproduce` as a tight definition list, a `⚠️ retraction` line, and three `fences` bullets. See `AUTHORITY-GENERATED-SAMPLE.md` → *fak compaction trims…*. Same information, zero loss, scannable in seconds — and the retraction is now a structured field a linter can assert on, not prose to be re-buried on the next edit.

## 6 · Existing gates become validators of the record

Nothing is thrown away — the current tooling *strengthens* under this design:

- **`claim_repro_scorecard.py`** (every row must carry a resolvable artifact/`Reproduce:`) → validates `artifact` + `reproduce` per record object, instead of regexing prose. Cleaner input, same guarantee.
- **`bench_provenance.py`** → the classifier that validates each row's `provenance` tag against its own language; the generator already imports it and renders its `summary_line`.
- **`check_provenance_labels.py`** (a MODELED number must never be called "measured") → becomes a field invariant: `provenance == "modeled"` ⇒ the headline/fences must not assert "measured." Trivial to assert on structured data.
- **A future machine view** (e.g. `registry → BENCHMARK-AUTHORITY.json`) gives every downstream consumer a typed feed instead of a markdown-table scrape.

## 7 · Migration path (incremental, never a big bang)

The canonical doc stays valid at every step:

1. **✅ done (this change)** Ship the record + generator + generated view + freshness gate, and **backfill every authority claim** into `registry.jsonl` (adversarially verified; completeness critic confirmed 0 drops). Canonical `BENCHMARK-AUTHORITY.md` untouched. Operator reviews the "after."
2. **Flip:** replace the hand-maintained table in `BENCHMARK-AUTHORITY.md` with the generated view (either a `<!-- GENERATED -->` block, or make the whole file generated with a hand-authored preamble, like `HARDWARE-MATRIX.md`). Point the freshness gate at the real file. Before flipping, diff the generated view against the current doc to confirm no number moved.
3. **Retire** the ad-hoc prose-status conventions; `status` is now the single source for stale/retracted/gated. Enrich thin-fence rows from their detailed `##` sections (the backfill captured table-cell fences; a few headline rows carry additional caveats in their prose sections).

Rollback at any step is `git revert` of that step alone — the record and the doc are separate files.

## 8 · What ships in this change

- `docs/benchmarks/registry.jsonl` — **all 52 authority claims** (every Quick-Reference row + the tombstoned block), each transcribed faithfully into the field grammar. The backfill ran as an adversarially-verified workflow: extractors over table slices → a per-row skeptic that re-checks every field against the source bytes → a completeness critic. Result: **0 dropped rows, 1 provenance correction** (the fleet-5×200-7B "8.2 min" was an arithmetic projection off measured rates, so its `measured`→`modeled` — exactly the mislabel the honesty discipline exists to catch).
- `tools/benchmark_authority.py` — the generator (`--write` / `--check` / `--stdout`), pure stdlib + `bench_provenance`.
- `docs/benchmarks/AUTHORITY-GENERATED-SAMPLE.md` — the generated "after," committed as the reviewable artifact.
- `tools/benchmark_authority_test.py` — 16 stdlib tests (record validation, deterministic render, shipped-doc freshness).
- `Makefile` — test + `--check` wired next to the other doc-freshness gates.

## 9 · Open decisions for the operator

1. **Registry format** — JSONL (chosen here: one object per line, git-diff-friendly, trivially streamable) vs TOML (`dos.toml`-adjacent) vs one file per claim under `registry/`. JSONL is the lightest to start; a per-claim-file layout scales review better at 40+ rows.
2. **Flip vs keep-alongside** — make `BENCHMARK-AUTHORITY.md` fully generated (step 3), or keep it hand-authored and treat the generated view as the operator's index? Fully generated kills drift permanently; keeping both re-introduces the duplication risk §2 warns about.
3. **Where the machine view lives** — emit `BENCHMARK-AUTHORITY.json` for downstream consumers now, or defer until a consumer needs it?
