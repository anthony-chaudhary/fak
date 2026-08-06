---
title: "Triage: query-card freshness/supersession rung (#3163) — what shipped, what remains"
description: "Classifies #3163's horizon from repo evidence: the FRESH / SUPERSEDED_BY / STALE rung on FeatureCard already shipped in internal/selfquery/freshness.go and is emitted on the --json and MCP surfaces; the residual is a cmd-lane render gap (the human table never prints the rung) plus two never-started supersession signals. Also disambiguates the two freshness axes the issue itself confuses."
---

# Triage: query-card freshness/supersession rung (#3163)

*2026-08-06 — docs-lane triage, generation stream unclassified on intake (triage-only frame).*

[#3163](https://github.com/anthony-chaudhary/fak/issues/3163) asks for an
advisory **currency** rung on a `fak feature query` result card: a `FeatureCard`
proves *which bytes* it carries (`Witness`) but said nothing about whether those
bytes are still current, so a note dated 2026-06-25 ranked identically to the
2026-07-06 note that superseded it.

This note classifies the horizon from repo evidence *before* any further
implementation — the proof bar for an unclassified generation issue — and exists
because the shipped rung was, until now, **documented nowhere the query surface
could retrieve it**. That is not incidental: #3163's own `## Dogfood` section
complains that querying *"freshness staleness supersession temporal validity of
a query result card superseded by newer note"* returns `fak_index_freshness` —
**the wrong axis**. It returned the wrong axis because nothing in the doc map
described the right one. This note is the missing card.

## What already shipped

The load-bearing ask — proposal items 1 (dated-note signal) and 2 (the rung
itself) — is in the tree across two commits, both under #3163:

| Commit | Date | What |
|---|---|---|
| `74ca356fd5` | 2026-07-08 | advisory dated-note supersession rung on feature-query cards |
| `903dc2769c` | 2026-07-08 | stamp `STALE` rung for a deleted cited file |

**The field.** `FeatureCard.Freshness` (`internal/selfquery/selfquery.go:62`,
JSON `freshness,omitempty`). Empty means *not applicable / unknown*.

**The rung vocabulary** (`internal/selfquery/freshness.go:46-61`):

- `FRESH` — the newest card among a set of same-topic dated notes. Set **only**
  when at least one strictly-older sibling exists, so a lone note is never
  decorated.
- `SUPERSEDED_BY:<name>` — the suffix is the **card `Name`** of the superseding
  sibling (e.g. `SUPERSEDED_BY:doc:Borrow scout: lmnr (Laminar) → fak — deep
  re-study (2026-07-10)`), resolvable through the same match `findCard` uses.
  Note the deviation from the issue text, which proposed `SUPERSEDED_BY:<slug>`:
  a resolvable card name is strictly more useful than a slug, and is what
  shipped.
- `STALE` — the card's cited artifact (`DetailRef`) no longer exists on disk.

**How each is computed.**

- *Supersession* — `freshnessByKey` (`freshness.go:141`) buckets cards by a
  normalized, date-stripped topic key derived purely from the `DetailRef` path;
  lexical order over the ISO date token equals chronological order, so no time
  parsing is involved. Admission is precision-first (`noteInfo`,
  `freshness.go:81`): a `.md` file under `docs/notes/` carrying an ISO date, or
  nothing. It is computed over the **full candidate set**, not the post-`--limit`
  result, so an older note is still marked when its superseding sibling ranked
  out of the top-N. If every note in a topic shares the newest date the order is
  ambiguous and **no** rung is claimed.
- *Staleness* — `stalenessByKey` (`freshness.go:197`) re-checks existence via
  `os.Stat` against the **current** tree on every `Query()`, so it tracks a
  deletion at the granularity of when the agent asks, not calendar age. Fences:
  `citedRepoPath` (`freshness.go:225`) rejects URLs, cap refs (`kind:name`),
  bare names and `..` escapes; `pathMissingUnderRoot` (`freshness.go:244`)
  additionally requires the **parent directory to still exist**, separating a
  genuine single-file removal from a synthetic ref that was never a real repo
  location.
- *Merge order* — `Catalog.Query` (`selfquery.go:195-201`) overlays staleness on
  supersession, so **`STALE` wins** when both apply: a removed artifact's
  supersession is moot.
- *Cross-repo* — `MultiCatalog.freshnessRungs` (`internal/selfquery/multiroot.go:174`)
  computes both axes **per source root** under a `root`-qualified key, so a
  note supersedes only its own checkout's siblings and two repos carrying a
  same-named card cannot cross-contaminate rungs.

**The fences the issue demanded are held.** `applyFreshness`
(`freshness.go:257`) writes the advisory field and nothing else — it never
re-orders and never drops a card, so a superseded or stale card still returns,
hedged. `Cards()` on both the single-root and multi-root paths leaves `Freshness`
empty, so `SummaryDigest` stays stable; the rung is a `Query()`-path concern only.

**Tests** (`internal/selfquery/freshness_test.go`, green at `go test
./internal/selfquery/ -count=1`): `TestNoteInfoAdmitsOnlyDatedNotes`,
`TestFreshnessByKeyLogic`, `TestQueryStampsSupersessionRung`,
`TestFreshnessDoesNotChangeRanking`, `TestCitedRepoPathAdmitsOnlyRepoFileRefs`,
`TestStalenessByKeyMarksDeletedCitedFile`, `TestQueryStampsStaleRungAfterDeletion`,
`TestQueryDetailCardCarriesFreshnessRung`, `TestMultiRootFreshnessStaysScopedPerRoot`.

## The two freshness axes, disambiguated

The issue's dogfood miss is a naming collision worth pinning, because both
surfaces answer to the word "freshness" and they are not the same question:

| | `fak index freshness` / `fak_index_freshness` | `FeatureCard.Freshness` (this rung) |
|---|---|---|
| Question | does the **index** agree with the **tree**? | is this **card's content** still current? |
| Findings | undeclared leaf, dead INDEX.md link, unknown verb, orphan note, dead llms.txt link | `FRESH`, `SUPERSEDED_BY:<name>`, `STALE` |
| Unit | the repo's self-index | one query-result card |
| Home | `internal/devindex/freshness.go` | `internal/selfquery/freshness.go` |
| Effect | a gate can red the build on drift | advisory only — never re-orders, never drops |

They complement each other; neither replaces the other. A tree can be perfectly
index-fresh and still hand an agent a superseded note.

## How to observe it today

The rung reaches the `--json` and MCP (`fak_feature_query`) surfaces:

```console
$ fak feature query "field borrow query quality" --limit 40 --json
...
"name":      "doc:Borrow scout: lmnr (Laminar) → fak (2026-07-09)",
"freshness": "SUPERSEDED_BY:doc:Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)",
```

## What genuinely remains

1. **The human table never prints the rung** — the one clause of #3163's own
   *"First checkable step"* (*"compute **and print** `SUPERSEDED_BY`"*) that is
   still unmet on the operator surface. `cmd/fak/feature.go:93` renders five
   columns (`Name`, `Kind`, `Effect`, `Source`, `Summary`) and drops
   `c.Freshness` entirely, so the same card that reports
   `SUPERSEDED_BY:…` in `--json` shows nothing in the default render. Witnessed
   above: identical query, one surface carries the rung and the other does not.
   This is a `cmd`-lane change (a column plus a render test), **not** docs — a
   fix committed under a `(fak docs)` ship stamp would not diff-witness.
2. **The explicit `supersedes:` front-matter edge** — proposal item 1's third
   signal, a hand-declared supersession the dated-note heuristic cannot see.
   `internal/selfquery/freshness.go:40-43` already records this as a known
   follow-on and shapes `noteInfo`/`freshnessByKey` so it can layer on without
   disturbing the dated-note path. It shares an edge extractor with the
   graph-query sibling under epic #1494, which is why it is a foundation item
   rather than a one-file change.
3. **`Resolved by <sha>` as a supersession signal** — proposal item 1's second
   signal. Never started; needs git plumbing on the query path that the current
   tier has deliberately avoided.

Two clauses are met in substance but not to the letter, and should not be
re-opened as gaps:

- **Proposal item 3 ("bind the code-token half to `dos_recall`")** shipped as an
  in-process `os.Stat` existence re-check over the card's cited `DetailRef`, not
  as a call into `dos_recall`, and it re-checks the cited **artifact** rather
  than a code **token**. The issue's stated intent — *reuse the existing
  re-verification instead of a new oracle* — is honored in discipline (read-time
  re-check against the tree *now*) while staying inside tier-1 purity: no
  subprocess, no network, no git.
- **The push / INVALIDATE half** — invalidating a card that is superseded-in-fact
  but still on disk — was deliberately carved out to child ticket
  [#3326](https://github.com/anthony-chaudhary/fak/issues/3326) and is not a
  #3163 residual.

## Classification

- **Stream: `gen/now`** (`Generation G0 - Now / Immediate`). The evidence
  supports naming it rather than parking it in `needs-triage`: the load-bearing
  rung has *already landed* with witnesses and no dependency on a future
  architecture bet, which is the `gen/now` definition in
  [`docs/generation.md`](../generation.md); and the residual that still belongs
  to *this* ticket's own acceptance (item 1 above) is a few lines of render code
  in the `cmd` lane. The genuinely later-horizon work is already carved out of
  this issue — the push/INVALIDATE half into #3326, the explicit-edge extractor
  into the graph-query sibling under epic #1494 — so #3163 does not carry a mixed
  horizon the way, for instance, [#2928 did](DISPATCH-DURABLE-BOARD-2928-TRIAGE-2026-07-15.md).
- **Smallest honest next step:** add the rung to the `cmd/fak/feature.go` render
  (a sixth column, or a suffix on superseded/stale rows) with an exact-output
  render test, committed in the `cmd` lane. That closes #3163's "first checkable
  step" on the last surface where it is still unmet.

## Generation close evidence

- **Promotion evidence:** the rung is shipped, fenced, and test-covered
  (`internal/selfquery/freshness.go`, nine named tests, `74ca356fd5` +
  `903dc2769c`) and is live on the `--json` / `fak_feature_query` surfaces — the
  witnessed capture above shows a real card carrying
  `SUPERSEDED_BY:<resolvable card name>`. That retires the issue's founding
  blocker ("query result cards carry no per-card freshness rung at all") for
  every machine-read consumer.
- **Demotion / retirement evidence:** proposal item 1's `Resolved by <sha>` and
  `supersedes:` signals are demoted out of the near term — neither is started,
  the explicit-edge one is gated behind an extractor owned by a sibling issue,
  and the code comment that names it as a follow-on has stood since 2026-07-08
  without a consumer asking for it. The literal *"bind to `dos_recall`"* framing
  of item 3 is **retired**, not deferred: the tier-1 `os.Stat` re-check satisfies
  the intent and calling out to `dos_recall` from the query hot path would cost
  purity for no added signal.
- **Invalidating assumption:** this triage assumes the rung's *consumers are
  machines* — MCP clients and `--json` readers — so shipping it on those surfaces
  is the substantive win and the missing human column is cosmetic. If the primary
  consumer turns out to be an **operator reading the default table**, then #3163's
  headline ask has been unmet on the surface that matters since 2026-07-08, item 1
  above is not cosmetic, and this note's `gen/now`-and-nearly-done classification
  understates the remaining work.
