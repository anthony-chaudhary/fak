---
title: "Triage: query-card freshness/supersession rung (#3163) — what shipped, what remains"
description: "Classifies #3163's horizon from repo evidence and records its close: the FRESH / SUPERSEDED_BY / STALE rung on FeatureCard shipped in internal/selfquery/freshness.go, and the last residual — a cmd-lane render gap where the human table dropped the rung — closed with c92b3ba341 (#5803), so the issue's own first checkable step (compute AND print SUPERSEDED_BY) is now met on every surface. Leftovers are the two never-started supersession signals (supersedes: front-matter edge, Resolved by <sha>), the first still gated behind the edge extractor of #3161, which is closed against a commit that does not implement it. Also disambiguates the two freshness axes the issue itself confuses."
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

The rung reaches **all three** surfaces — `--json`, MCP (`fak_feature_query`),
and (since `c92b3ba341`) the default operator table. Same query, both renders:

```console
$ fak feature query "Borrow scout lmnr Laminar" --limit 2 --json
"name":      "doc:Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)",
"freshness": "FRESH",
"name":      "doc:Borrow scout: lmnr (Laminar) → fak (2026-07-09)",
"freshness": "SUPERSEDED_BY:doc:Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)",

$ fak feature query "Borrow scout lmnr Laminar" --limit 2   # trailing column, rows elided mid-line
…  read  devindex  - auto-indexed dated note.   FRESH
…  read  devindex  …(Apache-2.0 `5f14c5c`), rea…  SUPERSEDED_BY:doc:Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)
```

`STALE` is pinned on the same render by the exact-output test
`TestFeatureQueryTextPrintsFreshnessRungs` (`cmd/fak/feature_test.go`), since it
needs a deleted cited artifact to fire.

## What genuinely remains

The render residual this note originally carried is **closed**:
`c92b3ba341` (#5803) extracted `writeFeatureRows` and added `c.Freshness` as a
sixth column plus an exact-output render test. With it, #3163's own *"First
checkable step"* — *"compute **and print** `SUPERSEDED_BY`"* — is met on every
surface, which is what closes the issue. Two never-started signals remain, and
both were already demoted out of this ticket:

1. **The explicit `supersedes:` front-matter edge** — proposal item 1's third
   signal, a hand-declared supersession the dated-note heuristic cannot see.
   `internal/selfquery/freshness.go:40-43` already records this as a known
   follow-on and shapes `noteInfo`/`freshnessByKey` so it can layer on without
   disturbing the dated-note path. It shares an edge extractor with the
   graph-query sibling #3161, which is why it is a foundation item rather than a
   one-file change. **That dependency is not satisfied:** #3161 is *closed*, but
   its recorded resolving commit `cae824a` adds
   `.claude/skills/field-borrow/SKILL.md` and a borrow note and touches no
   `internal/selfquery` code — no edge extractor exists in the tree (nothing
   under `internal/selfquery` reads `see_also:`, `[[wikilinks]]`, or
   `supersedes:`). The blocker is a mis-bound close, not a shipped foundation.
   Also worth noting: only one note in `docs/notes/**` writes a `supersedes:`
   key at all, and its value is `none` — so the extractor currently has **zero**
   producers to read, which is the honest reason this stayed demoted.
2. **`Resolved by <sha>` as a supersession signal** — proposal item 1's second
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
  [`docs/generation.md`](../generation.md); and the residual that still belonged
  to *this* ticket's own acceptance — the render column — was a few lines in the
  `cmd` lane and has since landed. The genuinely later-horizon work is already
  carved out of this issue — the push/INVALIDATE half into #3326, the
  explicit-edge extractor into the graph-query sibling under epic #1494 — so
  #3163 does not carry a mixed horizon the way, for instance,
  [#2928 did](DISPATCH-DURABLE-BOARD-2928-TRIAGE-2026-07-15.md).
- **Smallest honest next step (taken, `c92b3ba341`):** add the rung to the
  `cmd/fak/feature.go` render with an exact-output render test, committed in the
  `cmd` lane. That closed #3163's "first checkable step" on the last surface
  where it was unmet, and **#3163 closes on it**.
- **Next step for the leftovers:** re-open #3161 (its own close comment invites
  it: *"Reopen if this does not fully resolve it"*) so the edge extractor has a
  live owner; the `supersedes:` rung then layers on at
  `internal/selfquery/freshness.go:40-43` — but only once some note actually
  writes the key.

## Generation close evidence

- **Promotion evidence:** the rung is shipped, fenced, and test-covered
  (`internal/selfquery/freshness.go`, nine named tests, `74ca356fd5` +
  `903dc2769c`) and is live on the `--json`, `fak_feature_query`, **and default
  operator-table** surfaces (`c92b3ba341`, #5803, one exact-output render test) —
  the witnessed capture above shows a real card carrying
  `SUPERSEDED_BY:<resolvable card name>` in *both* renders of the same query.
  That retires the issue's founding blocker ("query result cards carry no
  per-card freshness rung at all") for every consumer, machine **and** human.
- **Demotion / retirement evidence:** proposal item 1's `Resolved by <sha>` and
  `supersedes:` signals are demoted out of the near term — neither is started,
  the explicit-edge one is gated behind an extractor owned by a sibling issue,
  and the code comment that names it as a follow-on has stood since 2026-07-08
  without a consumer asking for it. The literal *"bind to `dos_recall`"* framing
  of item 3 is **retired**, not deferred: the tier-1 `os.Stat` re-check satisfies
  the intent and calling out to `dos_recall` from the query hot path would cost
  purity for no added signal.
- **Assumption that was invalidating, now moot:** this triage originally assumed
  the rung's *consumers are machines* — MCP clients and `--json` readers — making
  the missing human column cosmetic, and named the operator-reader case as the
  branch that would invalidate the "nearly done" call. Rather than defend the
  assumption, the branch was **removed**: `c92b3ba341` shipped the column, so
  both readings now land on the same verdict. Cost of settling it that way: one
  render function and a 16-line test.
- **Still-invalidating assumption:** the close assumes supersession derived from
  **dated-note filename ordering** is enough signal to call the currency axis
  answered. It only fires for `.md` files under `docs/notes/` carrying an ISO
  date whose date-stripped slugs are *byte-equal*. A superseding note that
  renames its topic slug, lives outside `docs/notes/`, or supersedes across
  topics is invisible to the rung and will silently return unhedged. If the
  corpus turns out to supersede mostly by rewrite-under-a-new-title rather than
  by re-dating the same slug, then the shipped signal covers the rare case and
  the two demoted signals (`supersedes:` edge, `Resolved by <sha>`) are the real
  ones — and re-opening this axis under epic #1494 is warranted. Nothing in the
  repo measures that ratio today; that measurement is the cheapest next probe.
