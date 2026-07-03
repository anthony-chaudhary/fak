# Dedup earlier and more often — the structural root of fresh duplication (2026-07-03)

The prior note (`SLOP-DEBT-SUBCATEGORIES-AND-SYSTEMIC-APPROACH-2026-07-03.md`)
answered *how much* duplication we carry and *of what kind*. This note answers
the question underneath it: **why does fresh duplication keep getting born, and
how do we catch it earlier and more often instead of a cycle later as debt?**

## The one-sentence diagnosis

fak has a world-class **post-hoc** clone detector and **zero authoring-time
dedup query.** Every dedup signal we own fires *after* the duplicate already
exists — as a CI scorecard number, a batch review, a nightly loop. The author
(agent or human) has no cheap way to ask *"does something like this already
exist?"* at the one moment that would prevent the clone: **before writing it.**

That is the generator. A detector that runs a cycle late converts duplication
into *debt to retire*; a query that runs at authoring time prevents the
duplication from being *created*. The goal — "add dedup checks, query, and
related earlier and more often" — is exactly this shift from late-detect to
early-query.

## Three surfaces, one shape of gap

| Surface | What already exists | Why fresh dup is still born |
|---|---|---|
| **Code** | `kpi_duplication` (`tools/code_slop_scorecard.py:638`) — a real normalized-token clone engine, whitespace/comment/rename-invariant. But it lives inside a Python **batch** scorecard that grades the whole tree a cycle after code lands. | No forward query: "here is a block I am about to write — does a token-similar block already exist in the tracked tree?" The engine can only run over *everything at once*, as a grade. |
| **Issues** | `fak issue contract` computes `duplicate_key_groups` **within one candidate batch**; `--dedupe-checked` is a producer **assertion** it checked against existing open issues (`cmd/fak/issue_contract.go:141`). | fak never performs the "search existing open issues for one like this" step — it trusts a bool the producer sets. The check is self-certified, not witnessed, and only covers a batch, not the live backlog. |
| **Guards** | The scorecard is a CI ratchet; the trunk guard refuses off-trunk commits. | No guard runs a dedup query *at the moment of authoring or filing*. The checks are optional (`/slop-score` on a loop) and late (CI), so nothing makes "did you check for an existing one?" a precondition of the write. |

The through-line: the checks **exist but fire too late (post-hoc) and too rarely
(opt-in loop / CI gate)**. Making them fire *earlier* (authoring time) and *more
often* (a query verb + a durable guard) is the whole fix.

## The reusable primitives are already in the tree

Nothing here needs a new detector — two engines already exist, each right for a
different surface:

1. **Code — the normalized Go token-window engine** (`go_tokens` +
   window-keying, `code_slop_scorecard.py:399`/`:638`). Structural: it matches a
   copy-pasted block that survived a reformat or a variable rename, and skips
   data/declaration regions via the logic-token gate. This is the right engine
   for *code* dedup, because code clones are structural, not lexical. Its one
   limit for early-query use: it is **Python-only and batch-shaped**. The spine
   is to port the forward half (tokenize one block → slide window → report
   matching tracked sites) into an importable Go package so a verb *and later a
   guard* can call it.

2. **Prose — `internal/simhash`** (`Embed` / `Cosine` / `Index.TopK`). Word +
   char-n-gram feature-hash cosine — near-duplicate detection over short prose
   that survives typos and rewording. This is the right engine for *issues*
   (near-duplicate titles/bodies), and it is **already importable Go**, already
   used by `fak traj similar`. The issue-dedup query is a thin new consumer of
   it, not a new engine.

Picking the engine by surface (structural for code, semantic for prose) is the
load-bearing design call — forcing one engine onto both is how you get a code
query that misses renamed clones or an issue query that misses reworded dupes.

## The prescription: earlier, and durable

**Earlier (authoring-time query).** A `fak dup query --file X` (and
`--stdin`/`--block`) verb that answers, for one candidate Go block, the k
tracked sites whose normalized token windows overlap it — worst (most-overlap)
first. Run it *before* committing a new helper. Same primitive, inverted from
"grade the whole tree" to "check this one thing against the tree." The issue
sibling: `fak dup issues --title T --body B` over the live backlog via
`internal/simhash`, so "is this already filed?" is a real search fak performs,
not a bool the producer asserts.

**More often + durable (a guard, not a loop).** A query an author *may* run is
still opt-in. The durable form is a guard that runs it *at* the authoring/filing
boundary:
- a **pre-commit / pre-ship advisory** that runs `dup query` over the added Go
  hunks and warns (never hard-blocks — false positives on idioms must not wedge
  the trunk) when a new block strongly overlaps an existing tracked site;
- turning `issue contract --dedupe-checked` from a **trusted bool into a
  witnessed check**: fak runs the simhash search over the live backlog and fills
  the evidence, so the armor gate passes on a *performed* search, not a claim.

**The debt sub-categories connect here.** The prior note's `dup_extractable`
(≥MIN_SPAN, ≥3 sites, ≥1 non-test site) is exactly the threshold the *query*
should warn on — so the early query and the late scorecard share one definition
of "a clone worth preventing," and driving the query's warnings to zero drives
`dup_extractable` to zero by construction. One definition, two moments: query at
write time, grade at CI time.

## Recommended sequence (spine first, then fan out)

1. **Ship the code dedup-query spine** — port the forward token-window engine
   into `internal/clonescan` (importable Go, lane-disjoint, always buildable on
   the shared trunk) with a `Query(candidate, tree) []Match` entry point and
   tests. This is the keystone: every downstream guard and loop becomes possible
   once the query exists as a library. *(This note's companion commit.)*
2. **Wire `fak dup query`** onto the engine (a `cmd/fak` verb — dispatch when the
   cmd lane is green).
3. **Add the issue-dedup query** (`fak dup issues`) as a `simhash` consumer over
   the live backlog, and promote `--dedupe-checked` to a witnessed search.
4. **Add the authoring-time guard** — advisory pre-ship dup warning over added
   hunks; never a hard block.
5. **File the generator-extraction tickets** (the prior note's verified
   generators: `ParseLedger` ×5, the render structs, the `cmd/*/main.go`
   scaffold) so the highest-fan-out clones die at the source.

The through-line: today every dedup signal is a *grade you read after the fact*;
the spine turns the same primitive into a *question you ask before you write* —
which is the only form of a dedup check that prevents duplication instead of
merely counting it.
