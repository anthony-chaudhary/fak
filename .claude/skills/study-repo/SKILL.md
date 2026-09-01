---
name: study-repo
description: High-priority deep study of external code and proposals for fak. Invoke proactively whenever a repository, package, PR, issue, release, paper-with-code, or implementation is relevant—not only on explicit study requests. Acquire into scratch and pin revisions; mine code, tests, docs, history, releases, open and closed issues, PRs, discussions, roadmaps, and license/provenance; date every observation; directly port or adapt implementation when licensing permits; and explore both shipped mechanisms and the transferable spirit of proposed or incomplete ideas. Extract many source-anchored candidates, then. Use when this named workflow matches the task.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Grep, Glob, Write, Edit, WebFetch, WebSearch, Agent, mcp__fak__fak_feature_query, mcp__fak__fak_capabilities, mcp__fak__fak_index_docs, mcp__fak__fak_index_leaves, mcp__fak__fak_index_verbs, mcp__fak__fak_index_claims
argument-hint: "<repo-url | local-path | pkg | paper-url | 'look at X'>  [--deep(default)|--quick]  [--file(default)|--draft]  [--subtree <path>]"
metadata:
  opencode: claude-only
---

# study-repo — turn "look at <repo>" into scoped, witnessed, FILED borrows

## Invocation input contract — bind the target before doing anything else

On explicit invocation, the repository target is normally carried in the **same user
message as the skill token**, for example:

```text
$study-repo https://github.com/juliusbrussee/caveman
$study-repo C:\src\project --quick
```

Treat the text after `$study-repo` as this run's argument string. Before asking any
question or emitting a readiness message:

1. Re-read the triggering user message verbatim and extract the first repository URL,
   local path, package/paper identifier, PR/commit URL, or bare project name after the
   skill token. Parse `--quick`, `--draft`, and `--subtree <path>` as modifiers; they are
   not part of the target.
2. If the client supplied the skill instructions in a follow-up message, retain the target
   from the immediately preceding invocation message. A skill expansion does **not** erase
   arguments already present in conversation context.
3. Ignore client-status debris such as `Conversation interrupted`, `Something went wrong?`,
   or `/feedback` text appended after the invocation. It is neither a target nor a reason
   to ask again when a valid target appeared earlier in that message.
4. Echo the resolved target in one short line and immediately acquire it. **Never respond
   with “provide the repository/source” when a URL, path, or name is already present in the
   triggering or immediately preceding invocation message.** Ask only when no plausible
   target exists anywhere in that invocation context.

Example: for `$study-repo https://github.com/juliusbrussee/caveman` followed by an
expanded `<skill>...</skill>` payload, resolve the target to
`https://github.com/juliusbrussee/caveman` and start the deep pass; do not treat the skill
payload as a new targetless request.

## Why this skill exists

Someone drops a repo — "look at this" — and the silent failure mode fires: an agent
skims the README, believes the marketing, and either files one giant "adopt everything
from repo X" ticket no one can ship, or writes a tidy candidate table into a note and
**files nothing at all**. Five things go wrong. It borrowed from the *pitch* instead of
the *code that implements it* (a fabricated "repo X does Y" is the exact citation failure
`dos_citation_resolve` exists to catch). It read **shallowly** — one skim, a couple of
obvious borrows, and stopped, leaving the real value unmined. It never asked whether
**fak already has it**. It produced a **monolith** or a **transcript-only** result —
research that never became dispatchable work. And — the subtlest failure — it
read just enough to plant a flag and **dismissed by ego**: it decided "ours is better" or
"we already have that" from the pitch, killing a borrow at the coarse *capability* level
("we have graph memory too") without ablating to the specific *axis* the other repo wins on
("…but not the bi-temporal edge invalidation") and without ever reconstructing *why their
users made them build it that way*. Ego is cheaper than ablation, and it silently discards
the borrows most worth taking — a repo you concluded had "nothing we don't already do" is
almost always a repo you read too coarsely and judged too proudly.

The law, in one line: **the default answer to "look at this repo" is _acquire it, read
the code DEEP, decompose, and FILE_ — pin the source, ground every borrow at a real
`path:line@sha`, read broadly enough that a completeness critic finds nothing left
unopened, witness each against fak, decide borrow-vs-integrate on the license, and file
many small independently-shippable gh issues (never one monolith, never zero).** Deep and
filed are the *defaults*; `--quick` and `--draft` are the opt-outs, not the reverse.

Its companion law: **judge a borrow on _their_ tradeoffs, not fak's ego** — reconstruct who
they built it for, and ablate a coarse "we already have that" down to the axis the borrow
targets before you drop it. "Ours is better" is a conclusion you *earn by ablation*, stated
as a named tradeoff, never the opening posture. Humility here is not "borrow everything" —
you still will not — it is comparing at a fine enough grain that the dismissal, when it
comes, is honest. Steps 2, 3, and 6 own the mechanism; this is just the rule they enforce.

This skill owns the **acquisition + exploration + scoping + filing** front-half. For a
single already-named capability it reuses [`field-borrow`](../field-borrow/SKILL.md)'s
witness discipline rather than re-implementing it.

### How this differs from its siblings (do not duplicate them)

| Skill / tool | Starts from | Mechanism | License-aware? | Files issues? |
|---|---|---|---|---|
| `fak idea-scout` | an outward **feed** | dry-run arXiv/GitHub/HN/Reddit scan + filed-stamp dedup; `--live` files | no | yes (triage queue) |
| `field-borrow` | a **named capability** | dogfood `fak_feature_query`/`fak index` → witness the gap, file epic-anchored | no | yes (grounded) |
| `sota-check` | a **kernel op** you're about to write | `fak sota` prior-art matrix, route borrow/bind/stay-minimal | per-row route | no |
| `industry-score` | the **field taxonomy** | coverage + parity-debt scorecard | no | no |
| **study-repo** (this) | **anything that names code** | acquire → read DEEP (fan-out) + reconstruct their worldview → extract → ablate to the axis → scope small → witness **on-axis** → **FILE** | **yes — borrow vs integrate** | **yes — by default** |

study-repo **feeds** field-borrow: it produces the many candidates a capability-witness
then grades. Use study-repo when the input is a *body of code*; field-borrow when the
input is already one *named capability*. If a borrow is a **compute kernel**, route it
through `sota-check` (that matrix tracks the production reference).

## What counts as "a repo" — be flexible about the input

Do not refuse because the input isn't a clean `github.com/...` URL. Map the input to an
acquisition, then run the same pass:

| Input | Acquire by |
|---|---|
| GitHub/GitLab URL | shallow clone (below) |
| Local path / existing checkout | read in place — **do not re-clone**; pin `git -C <path> rev-parse HEAD` (or note "uncommitted" + `git describe`) |
| Monorepo, only one subtree matters | clone + `git sparse-checkout set <subtree>`, or `--subtree <path>` |
| One file worth a peek | `WebFetch` the raw file — no clone (this is what `--quick` is for) |
| npm / pypi / crates package | resolve to its source repo (registry page → repository URL), then clone; else fetch the published tarball |
| Paper-with-code / arXiv | find the linked implementation repo (`paperswithcode`, the paper's "Code" link, a `WebSearch`) and study **that**; the PDF is the map, the repo is the ground truth |
| Tarball / release asset | download into scratch, unpack, treat as a checkout (no SHA → pin the release tag + URL) |
| PR / diff / a single commit | clone, `git checkout <sha>`; study the change *and* the surrounding module |
| Bare "look at X" / no URL | resolve the name (`WebSearch` for the canonical repo), confirm it's the one meant, then clone. If genuinely ambiguous, state your pick and proceed — don't stall |

The output shape flexes too: a single borrow → one leaf; a coherent track → an epic +
child leaves; a whole subsystem worth adopting → an epic. Pick the shape the work demands.

## Durable source registry — check before cloning, update when done

Use [`docs/research/monitored-repositories.json`](../../../docs/research/monitored-repositories.json) as the exact `owner/name` source ledger. Before Phase 1:

```powershell
fak study-monitor --due-days 14
```

- If the source already has a row, treat `checked_revision` and `study_note` as deduplication evidence, not as proof that the current revision has been studied.
- If it has no row, add it as `candidate` with the discovery check's date, pinned revision, push timestamp, adoption signal, and plain-language relevance.
- At the end of a completed pass, update `last_checked`, `checked_revision`, `stars_at_check`, and `last_push_at_check`; set `status` to `studied` and `study_note` to the durable note path when a note ships, or `watch` when no candidate survives.
- Land the registry update with the study note. A prose-only “we looked at it” is not durable registration.

## Durable study receipt — bounded query before acquisition, immutable receipt after decision

The prose registry is orientation, not durable decision memory. Before cloning or acquiring a source, run a bounded query using its URL, name, or capability axis:

```powershell
fak study search "<source-or-capability>" --limit 20
```

Treat a match as an explicit **extend**, **recheck**, or **supersede** decision; never silently repeat it. If the store is missing or unavailable, state `durable study memory unavailable: <error>` in the report and continue with transcript-local evidence. Do not claim that a query or write persisted.

After candidate dispositions are witnessed, write a `fak-study/1` JSON record and run `fak study add --file <record.json>`. The record must pin every source revision; carry PRESENT/PARTIAL/ABSENT, disposition, evidence paths, and any superseded record ID; and put its returned `study_...` ID in every downstream GitHub issue body. A later verified implementation outcome is a new receipt that sets the candidate outcome and supersedes the earlier ID, never an edit that promotes model output to truth.

Promotion evidence: a fresh invocation finds the source-to-candidate decision by its returned ID or bounded lexical search. Demote or retire this integration if receipts do not reduce duplicate acquisition or if unavailable storage is reported as durable. Invalidating assumption: compact lexical queries are sufficient to rediscover the relevant prior decision without autonomous crawling.

## The pass

### 1 — Acquire into scratch, and PIN the source (never the tree)

Shallow-clone into the session scratchpad, never into `C:\work\fak` (the `*scratchpad*`
gitignore is a backstop, not the plan — the clone must never be committable):

```bash
git clone --depth 1 --filter=blob:none <url> "$SCRATCH/study-<repo>"
git -C "$SCRATCH/study-<repo>" rev-parse HEAD   # PIN this SHA — the borrow's dated anchor
```

`--filter=blob:none` keeps a big repo light (blobs fault in on read); `sparse-checkout`
for a huge tree; `WebFetch` for a one-file peek. **The pinned SHA (or release tag / local
HEAD) is the falsifiable anchor** — the way `sota-check` pins a `PrimaryLink`. A borrow
with no `@sha` is a rumor. If *why they changed it* will matter, deepen the clone now
(`--depth` drops the log).

### 2 — Mine the whole evidence surface — DEEP by default

Read in this order, because it goes from claims to ground truth:

1. **README / docs** — for the *map* only (where the interesting modules live). Never
   borrow from a README bullet; it is marketing until the code confirms it.
2. **The load-bearing modules** — the actual implementation of the thing you'd borrow.
3. **The tests** — they encode intent and the edge cases the authors actually hit; a
   technique's real shape is in its test, not its docstring.
4. **Recent commits / CHANGELOG** — what they are *actively* improving is where the live,
   worth-borrowing ideas are (and what they just ripped out is a warning).
5. **Their design rationale — the user world behind the code.** Read the *why*, not just the
   *what*: issues/discussions, design docs / ADRs / RFCs, the defaults baked into their
   config, the benchmarks they hold themselves to, the constraints in their README's
   non-goals. Reconstruct **who they built this for and what they were optimizing** (latency
   over footprint? single-node simplicity over horizontal scale? a research user who
   re-runs, or an ops user who never touches it?). This lens does two jobs: it lets you tell
   a genuine *improvement* from a
   *different tradeoff for a different user* (so "we'd never do it that way" becomes a
   testable claim, not a reflex), and it is the antidote to the ego dismissal — you cannot
   honestly conclude "ours is better" about a choice whose reason you never reconstructed.

Before extraction, complete the source classes that can change the conclusion: releases/tags/
changelog plus relevant history and blame; tests/fixtures; **open and closed issues**; merged,
closed, and open PRs plus review discussion; discussions/RFCs/ADRs/roadmaps/TODOs; and exact-
revision root/per-file licenses, NOTICE/provenance, vendored/generated code, and submodules.
For every material observation record `observed_at`, `source_event_at`, source state, immutable
anchor, platform/version context, and refresh trigger. Open work is direction, not shipped proof.

**Deep is the default, and for anything past a single-file peek that means FAN OUT.**
Dispatch parallel `Explore`/`Agent` readers — one per subsystem the README map exposes
(e.g. the scheduler, the storage layer, the eval harness, the wire protocol) — each
returning its load-bearing `path:line@sha` findings. A single serial skim of a
multi-module repo is the shallow read this skill exists to kill. Then run a
**completeness-critic pass**: name the subsystems/directories you did *not* open and
justify each skip, or open it. The depth floor: you are not done reading until the critic
finds nothing material left unopened — not when you've found "a couple of borrows".

`--quick` is the deliberate escape hatch (one file, one obvious technique, a time box):
one reader, skip the fan-out. A one-file peek cannot do the step-5 worldview read, so a
`--quick` pass may only file an obvious INSPIRE borrow or **defer to a deep pass** — it may
not *earn a dismissal* (a PRESENT-on-axis drop or a DIVERGENT), which the companion law
makes conditional on the worldview read. Everything else is deep.

**A borrow must be grounded in code you read**, at `path:line@sha` in the clone — never a
paraphrase of the pitch.

### 3 — Extract candidate borrows, one technique each — ablated, not ego-scored

Before narrowing, add candidates for direct mechanisms, negative knowledge (reverts,
rejections, failures), emerging direction (issues/PRs/RFCs), and **spirit extensions**. For each
extension write `source fact -> inferred principle -> fak opportunity -> disconfirming check`.
An incomplete upstream prototype may inspire exploration but never proves shipped parity.

For each thing worth taking, write one candidate with **four** fields, not "why it beats
us":

1. **the technique, in one line** — one technique per candidate; if it says "and also", split it now.
2. **the source `path:line@sha`** — grounded in code you read, never a paraphrase of the pitch.
3. **the one AXIS it optimizes** — the *specific* property this technique is about (freshness
   of retrieval, tail latency, memory footprint, replay determinism, operator touch), stated
   narrowly enough to compare on. This is the ablation: not "graph memory" but "bi-temporal
   *edge invalidation* — a newer fact retires the old one."
4. **how fak differs on _that axis_, and their-worldview reason** — honestly one of: *fak is
   already as good here* / *fak is genuinely weaker on this axis* / *fak chose differently for
   a reason that still holds* — plus one line on **why _their_ users made them build it this
   way** (from step 2's rationale read). Do not write "beats fak"; write the comparison and
   let step 6 grade it.

Two widenings — **capture more than fak-beating borrows**:

- A **worldview finding** — a user need their design implies that fak may not be serving at
  all — is a first-class output even when there is *no line of code to copy*. It reframes
  fak's roadmap rather than adding a kernel; record it in the study note, and step 6 routes
  it through the **worldview-finding branch** — a consideration under the roadmap epic only
  if it clears the same ship-alone bar a borrow does, never an auto-decomposed leaf
  (respecting "we don't do everything").
- A **deliberate divergence worth understanding** — a place where they and fak solve the
  same problem oppositely — is worth writing down *with both rationales*, even if you file
  nothing. Understanding why they diverged is how you avoid mistaking a real improvement for
  a stylistic difference next time.

**Do not pre-dismiss here.** A "we probably already do that" hunch is not grounds to omit a
candidate — write it down and let the ablated witness in step 6 decide. A rich repo yields
*many* candidates; a list of two or three from a large tree is usually a shallow-read (or a
proud-read) tell, not a sparse repo. Steps 4–6 shrink and scope; steps 2–3's job was to give
them enough raw material.

### 4 — Choose direct port, adaptation, inspiration, or exclusion (the license gate)

Apply `/field-borrow`'s **bounded-superset portfolio rule** before treating the source's
headline winner as the only useful result. Separate the `DEFAULT` frontier from the coverage
frontier, and assign each witnessed capability `DEFAULT`, `OPTIONAL-MODULE`, `RECIPE`, `WATCH`,
or `EXCLUDE`. A technique that loses globally can still be the right optional path for a named
hardware, provider, privacy, compatibility, topology, cost, or operator-preference cohort.
Use fak's modular leaves/adapters/integrations to preserve that value without widening the core
or weakening the common-case default. Do not preserve approaches that are genuinely dominated,
unsafe, license-incompatible, outside declared scope, or unaffordable to support; date
moment-in-time findings and give `WATCH`/`EXCLUDE` rows a review or reopening condition.

Read exact-revision root/per-file licenses, NOTICE, provenance, submodules, and contribution terms.
Public visibility is not permission; repository metadata is insufficient.

- **DIRECT-PORT (preferred when compatible):** copy the smallest coherent permitted implementation
  or test, preserve notices/attribution, cite `path@sha`, then adapt names/interfaces/style.
- **ADAPT:** reuse permitted implementation when fak constraints require material changes; identify
  direct versus rewritten portions and preserve lineage/attribution.
- **INSPIRE-ONLY:** for absent, unclear, incompatible, proprietary, or behavior-only sources,
  independently implement the idea without copying expressive code/tests/comments/assets.
- **DO-NOT-USE:** exclude unsafe provenance/terms and record why.

Do not reflexively rewrite compatible licensed implementation; obligations and technical fit decide.
If obligations are ambiguous, use INSPIRE-ONLY or seek maintainer/legal review.

### 5 — Scope: many small tickets, NEVER a monolith

This is the heart of the skill, and the user's explicit worry — the failure it exists to
kill is the single "adopt repo X" mega-issue. Enforce:

- **Each borrow fak genuinely lacks becomes its own smallest independently-shippable
  ticket**, sized to a first checkable step.
- **The ship-alone test:** "can this borrow ship and prove value on its own?" If no, it is
  too big — split. If a title needs an "and also", split.
- **A coherent track → an epic + child leaves**, matching this repo's epic/leaf convention
  (`.github/issue-views.json`: `epics` decompose, they are never dispatched as a leaf).
  Never a monolith leaf standing in for a track.

Prefer **more tickets with tight scope** over fewer fat ones — a reviewer, a dispatcher,
and a fleet worker can only act on a leaf.

### 6 — Witness each borrow against fak, then FILE (the default terminal action)

For **each** surviving candidate, run [`field-borrow`](../field-borrow/SKILL.md)'s witness
step — do not restate it, use it:

- Dogfood `fak_feature_query` / `fak capabilities` / `fak index docs|leaves|verbs|claims`
  → a **first-pass PRESENT / PARTIAL / ABSENT** (refined to the axis just below — this is only
  where fak sits in the neighborhood). Guard the lexical ranker's false-ABSENT by varying the
  phrasing **and** a raw `Grep` for the obvious symbol; for a high-value borrow, witness twice
  before trusting an ABSENT.

**Witness at the AXIS, not the capability name — this is the ablation a coarse pass skips.**
Phrase the query as the *specific axis from step 3*, not the umbrella capability. A hit on
the umbrella ("yes, fak has memory / a scheduler / an eval harness") is **not** a PRESENT for
the borrow — it only means fak is in the neighborhood. You are not done until you have read
fak's actual code on that seam and compared it *on the one axis the borrow is about*. Then
classify at that grain:

- **PRESENT-on-axis** → fak already does *this specific thing* as well or better (you read the
  fak code and it covers the axis). Drop the candidate, record the card. A witnessed "we
  already had this" is a real, good result — but it must be witnessed at the axis, not
  assumed from the capability name.
- **PARTIAL/ABSENT-on-axis** → fak has the *capability* but not *this axis* (PARTIAL), or
  nothing on-point at all (ABSENT). **This is a real borrow even when the capability is
  nominally PRESENT** — it files as an *enhancement to an existing seam*, not a new capability,
  and that is the common, valuable case the coarse dismiss used to throw away. Ground it in
  the **fak seam** (`path:line` in *fak*) and **FILE it** — carrying **both** anchors (source
  `path:line@sha` + fak seam), the axis, the dogfood witness, the inspire/integrate verdict,
  and a first checkable step.
- **DIVERGENT (do-not-file, but earn it)** → fak is weaker/different on the axis, yet fak
  chose the other way *on purpose* for a reason that still holds for fak's users (from step
  2's worldview read). This is the *only* honest form of "we don't need this" — and it is
  **not** a free dismiss: you must record **the actual tradeoff and their user world** (*"they
  optimize re-index latency for a research user who re-runs; fak optimizes replay determinism
  for an audited fleet, so their invalidation model would cost us the property we sell"*). A
  DIVERGENT with no stated tradeoff is just the ego dismissal wearing a verdict — reject it
  and go ablate. When the divergence is genuinely load-bearing for fak's roadmap, file it as a
  **consideration/discussion**, not an auto-decomposed leaf.
- **WORLDVIEW-FINDING (no axis to witness)** → a step-3 worldview finding — a user need their
  design implies fak may not be serving at all — has *no source `path:line@sha` and no single
  axis*, so it cannot go through the on-axis witness; do not force it into PRESENT/PARTIAL.
  Default to **note-only** capture (step 7's candidate table), naming the user need and the
  design evidence for it (a config default, a stated non-goal, a benchmark they optimize).
  File it as a **consideration** under the roadmap epic *only* when it clears the same
  ship-alone / first-checkable-step bar a borrow does — otherwise it stays an observation, not
  an issue (respecting "we don't do everything").

**Filing is the default, not an optional epilogue.** A run that surfaced PARTIAL/ABSENT
borrows and filed *nothing* is an incomplete pass — the research died in the transcript.
File only after first-class dedupe and contract checks:

```bash
fak-dev issue dedup --repo owner/name
fak-dev issue contract --repo owner/name --issue <N>
fak dispatch issues --issues issue.json --json
```

Issue creation and detached worker launch are independent explicit gates; this skill matching is
not launch permission. Put each issue under the right epic with milestone + labels set at creation,
and use a body file rather than an inline heredoc. Use `tools/issue_lane_router.py --apply-labels`
only as a legacy/debug fallback when the first-class issue/dispatch surfaces expose a discrepancy.
**Leak-check every body**: no absolute path, host, secret, or PII from the foreign clone leaks into
a fak ticket.

Report each filed issue number back. The only sanctioned no-file outcomes: every candidate
resolved to **PRESENT-on-axis** or a **DIVERGENT with its tradeoff + worldview stated** (an
unexplained "we don't need this" is not a no-file license — it means you have not ablated
yet), or the operator passed `--draft` (write the note + candidate table, file nothing, and
say so explicitly so a human can file later).

### 7 — Register the trail (or it lived only in a transcript)

Record the pass in a dated `docs/notes/CONCEPT-STUDY-<REPO>-<date>.md`: the URL + pinned
`@sha`, what you read (including the fan-out coverage + the completeness-critic result),
and a candidate table — **borrow · source `path:line@sha` · the AXIS · their-worldview reason ·
witness on-axis (PRESENT/PARTIAL/ABSENT/DIVERGENT) · inspire|integrate · filed #** — with a
`Companions:` block linking `field-borrow` and any epic. The axis + worldview columns are what
make a later reader able to *re-check your dismissal* instead of inheriting your ego. Add its **`INDEX.md`** line in the *Notes & research* section (an unindexed
dated note is an orphan `fak index freshness` flags). Link the children from the parent
epic. Commit the note lane by explicit path (`fak commit --path
docs/notes/CONCEPT-STUDY-<REPO>-<date>.md -m "docs(notes): study <repo> (fak docs)"`; never
`git add -A`, no `Co-Authored-By`). **This registration — with the issues — is what makes
the research durable.**

## What "done" proves

- The public study records the repository/project name, public upstream URL when available,
  pinned revision, and a generic locator such as **operator-local checkout**. It never records an
  absolute user-home checkout path or operator-local infrastructure detail. Keep exact local
  locations in the private companion repository, and run `fak public-scrub audit-tree` before
  publishing.
- The source was **acquired and read at a pinned `@sha`** — every borrow cites real code,
  not a README claim — and read **deep**: the fan-out covered every load-bearing subsystem
  and the completeness critic found nothing material unopened (a sanctioned `--quick` pass
  reads one file with no fan-out — state that explicitly, the same way `--draft` is stated).
- **Issues were filed** for the PARTIAL/ABSENT borrows (their `#`s reported), each carrying
  a **source `path:line@sha`**, a **fak seam `path:line`**, a **dogfood witness**, an
  **inspire/integrate + license verdict**, and a **first checkable step** — unless every
  candidate resolved to PRESENT-on-axis or a stated DIVERGENT, or `--draft` was set (stated
  explicitly).
- Filed effects are confirmed by independent `gh issue view` read-back. Any shipped effect also
  has scoped origin ancestry, a green focused test/read-back, and `dos commit-audit` = `OK` /
  `diff-witnessed`; worker narration is never proof.
- **No monolith was filed** — the work is decomposed into small independently-shippable
  leaves (an epic + children if it is a real track).
- **Every dismissal was earned by ablation, not ego** — each dropped candidate resolved to
  PRESENT *on its axis* (fak code read and confirmed on that specific property) or to a
  DIVERGENT that names the actual tradeoff and their user world; no borrow was killed at the
  coarse capability level with "we already have that".
- Nothing fak already ships *on the borrow's axis* was re-filed (step 6's on-axis witness
  caught the true-PRESENT cases) — but an enhancement to a nominally-PRESENT capability *was*
  filed when fak was weaker on the axis.
- The instance is registered: a dated `docs/notes` entry, its `INDEX.md` line, the epic
  linking its children; `fak index freshness` reads clean for the new note.

## The anti-gaming rule (specific to this surface)

**Never file a borrow you did not locate at a real `path:line@sha` in the actually-acquired
source; never file the monolith; never let "done" mean "wrote a table and filed nothing";
and never dismiss a borrow you did not ablate.** Six failure modes this kills:

- **Borrowed the pitch.** Filing "repo X does Y" from the README when the code does not, or
  citing a repo you did not acquire. The fix is the pinned `path:line@sha` you read.
- **Read shallow.** Two borrows off a skim of a large repo, no fan-out, no completeness
  critic. The fix is step 2's depth floor — deep is the default.
- **The mega-ticket.** "Adopt everything from repo X" as one leaf. The fix is the ship-alone
  test in step 5 — decompose or file an epic + leaves.
- **Filed nothing.** A tidy candidate table and zero issues — the research died in the
  transcript. The fix is step 6: filing is the default terminal action, not an optional
  epilogue.
- **License laundering.** Copying incompatible-licensed bytes in as "inspiration". If you
  copied, it is `integrate` and the license gate applies; if you truly reimplemented
  clean-room, it is `inspire` and you still cite the source.
- **Dismissed by ego (the coarse-PRESENT / "ours is better" reflex).** Dropping a borrow at
  the capability level ("we have memory too") without reading fak's code on the axis, or
  declaring a DIVERGENT without naming the tradeoff and their user world. The tell is a
  candidate table where the whole large repo collapsed to "all PRESENT, nothing to take" —
  that is almost always a proud read, not a covered one. The fix is step 6's on-axis witness
  and step 2's worldview read: earn the dismissal at the property grain, or file the
  enhancement.

A PRESENT witnessed **on the axis** — fak's code read on that seam — is a complete, honest
pass, not a failure to find a borrow.

## Honest limits

- **The witness is lexical + a snapshot.** field-borrow's `fak_feature_query` ranker is
  substring matching (false-ABSENT risk — hence the twice-witness + raw `Grep` guard) and
  its verdict is true only as of today — re-witness before acting on an old pass.
- **A shallow/blobless clone can hide history.** `--depth 1` drops the log; if *why they
  changed it* matters, deepen the clone before trusting a "they removed this" read.
- **License reading is not legal advice.** The gate is a good-faith compatibility check
  against the source `LICENSE`; a genuinely load-bearing vendor decision wants a human.
- **Their "worldview" is your reconstruction, not their testimony.** The who-it's-for and
  why-they-chose-it you infer from their code, issues, and defaults can be wrong — so keep it
  falsifiable (cite the config default, the non-goal, the benchmark they optimize) and never
  let a *guessed* rationale harden into a reason to dismiss. A DIVERGENT resting on an
  invented motive is fiction, not a finding; if you cannot ground the rationale, ablate and
  file rather than assume.
- **Composes with, does not replace:** `field-borrow` (the per-capability witness+file —
  this skill's hand-off target), `idea-scout` (automated outward feed), `sota-check` (route
  a kernel borrow through the matrix), `industry-score` (a witnessed PRESENT/PARTIAL/ABSENT
  is a coverage row it can absorb).

## When to run this

- Someone hands you **any body of external code** — "look at X", a GitHub URL, a local
  checkout, a package, a paper-with-code — and wants it turned into action.
- **Before** filing any "we should adopt X from repo Y" issue.
- After a competitor or a paper **open-sources** something worth reading.
- On a `/loop` cadence over a watch-list of repos, to convert "we should look at that
  someday" into witnessed, scoped, license-clean, **filed** backlog.
