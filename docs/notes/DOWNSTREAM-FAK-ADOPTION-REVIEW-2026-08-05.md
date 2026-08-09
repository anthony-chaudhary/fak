# What a downstream fak adopter did differently — and what is public-safe to bring back

> ⚠️ **Partially corrected, same day, by
> [`DOWNSTREAM-REVIEW-VERIFIED-2026-08-05.md`](DOWNSTREAM-REVIEW-VERIFIED-2026-08-05.md).** That
> note ran steps 1–3 of § "Next checkable steps" below. Outcome: **§A1 and §A2's could-not-run claim
> are wrong and §A4 is overstated** — `tools/scorecard_control_pane.py` already emits `debt: None`
> for an unmeasured card, excludes it from the sum by type, prints `(N measured, M errored)`,
> hard-fails the ratchet on any errored card, and returns exit `2` when unpinned. This note assessed
> the scorecard producers and the pinned baseline but never the fold that consumes them. **§A3
> survives and sharpens** (both ratchet axes read the numerator, neither reads the domain). **§C7 is
> confirmed and half-fixed** — the multi-line lane bug is real, is in `parseLanes` rather than the
> `splitLaneTree` this note looked at, and is now fixed and pinned by a test. **§C5 is settled in
> the downstream repo's favour.** The text below is left as written, per *"retract in place and keep
> the retracted text visible"*.

**Date:** 2026-08-05 · **Subject:** a contract-delivered Go repository (referred to here as *the
downstream repo*) that adopted fak's guard, lane, worktree and shipgate concepts and then hardened
them against a ~30-concurrent-session shared checkout and a 50–150-worker agent fleet.

**Why read this.** The downstream repo is the only place fak's mechanisms have been run to
destruction by someone who was *not* fak's author, under a delivery deadline, and who then wrote
down what broke. It re-derived several of fak's ideas independently, sharpened four of them in ways
fak has not, and measured nine defects in fak/DOS that fak's own tree does not record. This note is
the public-safe extract: **mechanisms and doctrine only, no client identity, no domain, no
engagement specifics.** § E states explicitly what was filtered out and why.

**Verdict in one paragraph.** fak is ahead on breadth — the guard kernel, the concept gate, the
memory substrate, the dispatch fleet, the worktree lifecycle, ~60 skills and the scorecard ratchets
have no downstream equivalent, and the downstream repo *cites fak as prior art* for its shipgate
(re-rendering `internal/shipgate`'s non-forgeable keep-bit in shell because it had no Go home for
it). The downstream repo is ahead on **epistemics of the instruments themselves**: it consistently
asks a question fak's tooling does not, which is *"what did this gate quantify over, and can the
gate still be shown to fire at all?"* Four of its five highest-value imports are variations on that
one question. The other high-value import is a measured cost model for agent-fleet preambles, which
fak has the packages to act on (`skillfootprint`, `skillvalue`, `promptmmu`) and no measurement to
aim them with.

---

## A. Mechanisms worth importing, ranked

Each item states the mechanism, the failure it prevents, what fak has today, and the smallest
fak-shaped landing. Nothing here is client-specific.

### A1 ⭐ Every KPI is `null` when unmeasured — never `0` — with a sibling `status.<kpi>`

**Mechanism.** The downstream maintenance sensor emits one witness record per tick. Any KPI it could
not measure is `null`, never `0`, and carries a sibling field `status.<kpi>` ∈ `{ok, unavailable}`.
The loop refuses to act on a record whose statuses are not all `ok`.

**The failure.** "A loop that reads a failed measurer as `0` debt reports a perfect repo forever."
This is not hypothetical downstream: the first draft of their sensor read a wrong JSON field name,
and the only reason it was a two-minute fix rather than a silent week is that it said `null`.

**What fak has.** ~60 scorecards under `tools/*_scorecard.py` and `docs/*-SCORECARD.md`. This is
fak's densest instrument surface and therefore where a zero-for-unmeasured convention is most
expensive. I did not audit all 60; the check is one grep per scorecard for the path where a probe
fails and a numeric default is emitted.

**Landing.** A shared helper in the scorecard schema that makes `null` the only representable
"could not measure", plus a `status` map, plus one test per scorecard that kills its measurer and
asserts the output is `null`/`unavailable` rather than `0`/`ok`.

**⛔ The trap the downstream repo also documents, because it caught itself:** *the record kept the
`null`; the fold lost it.* Their genesis witness carried `"score_total": null` and the human-written
digest transcribed it as a number. The guarantee is only as strong as the last hand that copies the
value — so folds must be mechanical.

### A2 ⭐ Print the domain, not only the verdict

**Mechanism.** Every check reports how many candidates it quantified over, and an empty domain is
spelled `no-candidates`, never `clean`. Their formulation: `∀x∈∅ P(x)` is true, so **a gate whose
domain is empty is green and blind at the same time**; `0 findings` and `0 findings over 0
candidates` are different instruments and only the second can be trusted.

They pair it with two more rules worth importing verbatim:

- **Soundness before completeness.** An unsound gate is worse than none because it manufactures
  confidence. This is why `exit 0/1/2` = clean / findings / could-not-run runs through their whole
  CLI: **the third code exists so could-not-run never impersonates clean.**
- **Convert the admission into a test.** A package that documents "nothing in the tree uses this"
  turns that paragraph into a test that runs the scan and **fails the day a caller appears**. Copy
  the pattern, not the paragraph.

**What fak has.** `internal/unwiredscore` and `internal/unwitnessedclaim` are the same instinct.
The delta is that fak's live at repo scope while the downstream pattern is *per-package*, and that
fak's gates do not generally print their candidate count.

**Landing.** Add a candidate count to the scorecard/gate output contract; add `no-candidates` as a
distinct status from `clean`; adopt the per-package unwired test as a convention in `AGENTS.md`.

### A3 ⭐ `probes_trusted` — a selftest that proves each check can still fire

**Mechanism.** Before reading any number in the witness record, read `probes_trusted`. It is the
result of a selftest that **builds a document designed to trip each check, and fails if the check
stays quiet.** When it is `false`, every clean number in the record is UNTRUSTED — not clean — and
the tick routes to the supervisor instead of "fixing" anything.

This is the generalization of a rule their fleet learned the hard way: *validate every extractor
against a known-positive AND a known-negative before trusting a count.* They have two independent
incidents behind it — a completion probe that reported **3 missions done when the true count was 0**
(it globbed instead of scoping to the roster), and a verdict column whose only test was `in
[min,max]` over ranges spanning 5.26×, so it was **TRUE for every row including both outliers**: a
threshold that fails open, wearing a verdict.

**Their generalization, which is the part to import:** *a probe that can only report "more done than
reality" is as dangerous as one that reports "nothing found".*

**What fak has.** No equivalent that I found. fak's scorecards are trusted implicitly.

**Landing.** One fixture corpus per scorecard family, a runner that asserts each check fires on it,
and a `probes_trusted` boolean at the top of the board. Highest leverage of anything in this note,
because it makes every other number falsifiable.

### A4 ⭐ FLOOR vs objective in the shipgate

**Mechanism.** Their shipgate splits KPIs in two. **Floors may never regress**, no matter what
improved (link integrity, doc-contract conformance, index freshness, `probes_trusted`).
**Objectives may be maximised** (the blended score total and its parts). The selftest pins the
canonical case: *a tick that gains 3 points on one KPI while breaking 2 links is a REVERT, and the
verdict names the floor it broke.*

**The failure.** Against a single blended total, the cheapest way to reduce debt is to **delete the
citation, the link, or the page the metric counts**. Their framing: "a floor is the part of the
score you are forbidden to buy" — the same shape as a calibration metric gamed by setting
`claimed := realized`.

**What fak has.** `internal/shipgate` already carries the harder half — `improvedBit` is unexported
and set only by `Evaluate`, so a caller can only *obtain* a keep, never write one. The downstream
repo names this as the rule it is copying. What fak does not have is the floor/objective partition;
`shipgate.go` reasons about a single witness direction and `needGain`.

**Landing.** A floor set in the witness profile that is evaluated independently of the objective and
can veto a keep on its own, with the verdict naming the floor.

### A5 ⭐ Range-scoped closeout, as a verb, on the Stop hook

**Mechanism.** One verb answers a question a ratchet structurally cannot: **"the work I just did
owed which index rows, and which are still unpaid?"** Default range is `@{u}..HEAD`. Each unpaid row
arrives with `detail` (what is wrong), `pays` (the exact command or edit), and `items` (the paths or
ids) — so it is a dispatchable mission with no discovery phase.

Four properties make it work, and each is a separate import:

1. **Range scoping is the whole point.** Inherited debt swamps the row you just created, which is
   why a whole-tree ratchet is the wrong shape for *"may this session stop?"*. The report prints
   inherited debt in a separate **INHERITED** block that is never gated and never a reason to widen
   the diff.
2. **The unpaid count *is* the agent count.** Three unpaid rows means three agents.
3. **The agent may not write the verdict — re-run the verb.** "An agent asked *did your change
   help?* answers yes." Re-running reads the tree rather than the transcript, and is the only thing
   that distinguishes *paid* from *believed paid*.
4. **Exit 2 is could-not-run and is never a pass** — an empty or unresolvable range makes every row
   vacuously true.

They also carve out `unproven` as a distinct status meaning *owed, and uncheckable by this
instrument* (a row needing the network, or needing human judgement), with the rule: **do not
dispatch an agent at an unproven row** — it cannot obtain evidence, so it will produce a
confident-sounding guess. And: an unproven row silently dropped from the report is how debt becomes
invisible rather than paid.

**What fak has.** No `fak closeout`. fak has whole-tree scorecard ratchets and a Stop-hook goal
mechanism, but nothing that scopes doc/index debt to the range a session just produced.

**Landing.** `fak closeout [--range|--last N|--dirty] [--json]`, rows derived from fak's own
"keep the indexes true" mapping, bound advisory-only to the Stop hook. **Advisory is load-bearing:**
they note a blocking Stop hook could trap a headless fleet worker in a loop it cannot argue its way
out of, which is exactly fak's dispatch topology.

### A6 ⭐ Worker-preamble economics — move the rationale one hop away

**Mechanism.** Every line in a worker preamble is re-presented as `cache_read` on **every turn of
every worker**. Measured over 1,385 workers: the fixed prefix is ~33k tokens re-read ~57× per
worker = **1.88M tokens each, 39% of a worker's entire `cache_read`**, and **68.3% of a fleet wave's
bill is context against 31.3% production**.

⇒ Justification is therefore the one thing that must **not** sit in a preamble. It moves to a
sibling file that the *person editing the preamble* reads and no worker ever loads. Their split, which
transfers directly to fak's `.claude/skills/` and `tools/issue_worker_prompt.py`:

| goes in the preamble | goes in the rationale file |
|---|---|
| the imperative, in the mood of a command | why it is the imperative |
| the exact recipe / code block to run | the recipe that used to be printed and failed |
| the one-clause reason, where it changes behaviour | the date, the cost figure, the turn count |
| a named refusal and its sanctioned alternative | the measurement that ranked the refusals |
| ⛔ markers on the two or three lethal steps | the story of who died and how it was diagnosed |

**The test:** *can a worker act differently because of it, on this mission?* If not, it belongs one
hop away.

⛔ **The counterweight they also measured, and it must travel with the rule:** cutting a preamble's
*guard* section made things worse, not better — one wave's evidence workers went from **0/7 to 5/5**
delivering once a broken guard section was repaired. **Cut narrative, never constraints.**

**What fak has.** The packages to act on this (`internal/skillfootprint`, `skillvalue`, `promptmmu`,
`syspromptmmu`) and ~60 skills, but no measured prefix-cost model and no enforced preamble/rationale
split.

### A7 One canonical refusals file, prepended unconditionally to every worker prompt

**Mechanism.** All refusal and park rules live in exactly one file, prepended to every worker prompt
in every mode — not restated per mode.

**The failure, measured twice.** A rule (*"never force-overwrite a tag; this killed half of one
wave"*) lived in the read-only worker preamble. The next wave then lost **four feature workers to
that exact command**, because the feature preambles never had the line. *The knowledge existed, in
the wrong file.* A drift audit at the same time found **11 of 12** refusal topics duplicated across
two preambles and already disagreeing.

An earlier revision had tried to fix this with discipline — *"if you edit one preamble's guard rules,
diff the other"*. **That instruction is what failed**; by the next wave there were four files to keep
in sync, not two.

**Landing.** fak's skills and worker prompts have the same multi-mode shape. One prepended file, and
a test that the refusal vocabulary appears in exactly one source.

### A8 Landing is an ordinal deadline, not a preference

**The measurement** (376 transcripts across 24 waves):

| | measured |
|---|---|
| workers that landed **nothing, ever** | **26.9%** |
| median worker's **first** landing (commit \| comment \| tag) | **77% of the way through its run** |
| workers that had landed nothing at their own halfway point | **78.9%** |
| mean turn at which a refusal-death occurs | **turn 29** |
| median turn of a first landing | **turn 39** |

⭐ **Read the last two rows together: the average death lands before the average first delivery.**
That is not bad luck, it is the work order.

**The fix, and why its shape matters.** The *preference* ("publish before you polish") was already
in the preamble and produced those numbers. Preferences lose to *"let me just verify one more thing
first."* What works is **an ordinal you can count** — and the proof is that their existing stub rule
("write your report within your first three tool calls") is obeyed fleet-wide. So: first durable
statement by tool call 8, a `LANDED_SO_FAR:` field in every report template, and the self-check
*"if my next tool call were refused and I stopped there, what would exist?"*

They pair it with a depth budget from the same corpus: workers past 80 turns are **17% of the roster
and 34.7% of the bill** at **2.19×** the cost-per-worker of the 41–80 band, while **output per turn
does not rise with depth** — only the context bill does. ⇒ *if you are deep and not done, land what
you have and write `NEXT:`; a successor resumes for ~1/6 the cost. Grinding on is the expensive
option, not the diligent one.*

**Landing in fak.** `tools/issue_worker_prompt.py` and the dispatch worker preambles. This is the
single most directly transplantable item in this note.

### A9 ⛔ A guard refusal kills the report — and the result JSON looks clean

**The measurement.** Over 320 exited workers: **232 (72%) stopped at a guard refusal, at a mean of
29 turns**. The largest single family was `rm -rf` on a scratch directory — i.e. **most stopped
during *cleanup*, after the real work was done, and left the report unwritten.** In a later wave,
9 of 33 workers ended at a refusal, consuming 44% of that wave's entire spend; every one had the
sanctioned alternative printed in front of it and quit instead.

⛔⛔ **Their result JSON is indistinguishable from a clean run** — `subtype: "success"`,
`is_error: false`, `permission_denials: []`. Nothing killed them; each voluntarily ended its turn.
**Nobody downstream can tell the worker stopped early.**

Two separate mechanisms, both worth fixing:

1. **The worker treats per-tool feedback as a session stop.** fak's own refusal text says otherwise
   in its fourth line. The downstream mitigation is a rule: **never put a housekeeping step inside a
   sequence whose earlier steps are unrecoverable. No cleanup, no undo, no re-park-in-place. Your
   deliverable is never a later clause of an `&&` chain.**
2. **`&&`-coupling silently discards the deliverable.** `rm -f <file> && cat > <report>` loses the
   report when clause one is refused. The work was not undone — it was never written.

**⭐ The fak-side ask this generates.** This is a product gap in fak's guard, not just a prompting
problem: **a refusal-terminated session is not distinguishable from a clean one in the harness
result.** If `permission_denials` stays empty when the guard refused a call that ended the run, no
fleet-level instrument can ever measure this, and every downstream cost model is blind to 72% of its
worker deaths. Worth filing against `internal/guard` / the result-shape contract.

### A10 The routing contract: an index may copy a location, never a value

**Mechanism.** Their task-router file is gated by a lint that **refuses dated snapshots, copied
status, copied metrics or populations, direct issue-state links, and direct result leaves.** Project
state, command state, evidence and backlog selection each have exactly one authority, and the router
points at authorities rather than restating them.

Paired with a rule about *where a number may live at all*:

| The number is… | Put it… | What keeps it true |
|---|---|---|
| a **claim** you are making in a sentence | in prose | a KPI re-derives it from the live authority every run; a stale claim is DEBT naming the file, the line, and both numbers |
| a **capture** of what a command printed | inside a fenced block, under a line saying **when** it was captured | nothing, deliberately — it was true when captured, and it ages into evidence rather than into a lie |

⛔ **Never paste volatile output into a sentence.** Their fact-checker deliberately skips fenced
blocks, so **the fence is load-bearing punctuation, not styling.**

⭐ **And the sharpest sentence in their whole tree, which fak should adopt as doctrine:**

> **A description that counts its members is a list, and lists rot — a list of one number included.
> Re-measuring is not the fix; pinning to a sha, or deleting the number, is.**

They earned it: a paragraph warning about stale counts published a wrong pair of counts *twice*, and
the corrected pair decayed **faster than the pair it replaced** (+28 occurrences in eight working
hours). The same instinct drives their repo-layout table, which **deliberately refuses to enumerate
the package roster** — "`ls -d internal/*/` is the list, and it is the only copy that cannot go
stale" — after the hand-kept roster had silently drifted to 17 of 28.

**Relevance to fak.** fak's `INDEX.md`, `README.md`, `docs/INDEX.md` and the scorecard docs carry
many hand-maintained counts and rosters across a far larger tree. The gating idea — a table of
*{live value, the docs that restate it, the anchor that makes a line a claim, the claim's shape}*,
where adding one row checks every doc in the set from then on — is a clean fit for fak's existing
`tools/check_index_sync.py` and `tools/docnumbers`.

⛔ **One caveat they state and fak should keep:** read the live value **from the authority the binary
already uses**. A fact row holding its own copy of the number is precisely the bug the KPI exists to
catch.

### A11 A four-state commit gate — and "I refuse your diff" ≠ "I could not run"

**Mechanism.** Their pre-commit gate has four distinguishable outcomes:

| output | exit | meaning |
|---|---|---|
| `DENY <CODE>` | `1` | the gate ran and **judged your diff**. Fix your change. |
| *(silence)* | `0` | the gate ran and found nothing. |
| `GATE_STALE` | the last good gate's own | the gate could not be **rebuilt**, so the last binary that did build judged you. **Every rule still ran** — a DENY under this banner is a real judgement. |
| `GATE_UNAVAILABLE` | `0`, or **`78`** under an opt-in deny mode | **nothing judged this commit at all.** Allowed, and says so out loud rather than passing in silence. |

Exit **78** is `EX_CONFIG` — an *infrastructure* code. Read it as "fix the gate", never "fix your
change". Their bootstrap test pins both halves as a **pair**: the gate must still refuse trespass,
credentials and misplacement with a broken sibling present, *because a fix that failed open into
silence would be strictly worse than the bug it replaced.*

**The failure that produced it, which is exactly fak's topology.** Their hook was one line
(`exec go run ./tools/precommit`), so it had to compile the gate's whole import closure out of the
**shared checkout** — and **one uncompilable file anywhere in that closure denied every commit in
the repo, in every lane, with a stranger's compile error as the only message.** The fix: hash the
closure, build the gate **once** to a content-keyed path under `<git-common-dir>`, exec the binary.

**What fak has.** No `GATE_UNAVAILABLE` / `GATE_STALE` / `EX_CONFIG` vocabulary anywhere in the tree
(grepped). fak's build gates do differential attribution against base HEAD, which is the *other*
half of the problem and arguably better — but a fak agent still cannot distinguish "the gate refused
me" from "the gate could not run" by exit code.

### A12 A recorded, HEAD-keyed trunk verdict — where silence means exactly one thing

**Mechanism.** A verb runs the full suite against an **extracted HEAD** (never the working tree,
which in a shared checkout is never equal to HEAD) and stores the verdict — with the timeout it used
— at an untracked path under `<git-common-dir>`. The commit gate then consults it and reports:

| code | when |
|---|---|
| `TRUNK_RED` | last verdict RED, nothing since touched the failing package — **names the failing test and the exact `-run` line** |
| `TRUNK_VERDICT_STALE` | that verdict's packages have all been edited since |
| `TRUNK_UNKNOWN` | nothing recorded, corrupt, another history, **or a GREEN verdict that is not for HEAD** |
| *(silence)* | the suite passed against **exactly this HEAD** |

⭐ Two design choices to copy. **An old GREEN reads as UNKNOWN, not green** — silence means one thing
only. And the gate and the recorder read **one shared timeout constant**, so *they cannot disagree
about what green means*.

**Why it exists.** Their trunk was red for **262 commits over ~28 hours** because every worker's own
package tests were green and nothing told them about anyone else's.

**What fak has.** `cmd/fak/trunk_red_ledger.go` — a best-effort JSONL witness keyed by a
*convergence class* (base sha + sorted failing packages) so N clones hitting the same shared red
fold onto one row. That is genuinely good and the downstream repo has no equivalent — but it is
**reactive**: it records reds that build gates happened to discover. It never records **green**, so
fak has no way to say "this HEAD is verified green" and therefore no way to distinguish *unknown*
from *fine*. The two designs compose: keep the convergence-class fold, add the positive HEAD-keyed
verdict.

**Deliberately untracked, and worth keeping so:** "the gate that reads it runs on every commit, and
30 agents racing to commit it through a contended index is worse than the problem it solves."

### A13 The package doc as forensic record, plus a pinned binary roster

**Mechanism.** Every repo-internal tool's package doc states: why the tool exists, **the specific
wrong answer it prevents**, what was measured and when, and — the part fak lacks — **whether it is
WIRED or NOT WIRED**, with a roster test (`TestBinRosterPinsAreNeitherStaleNorUnexplained`) that
fails the day a binary appears or disappears without an explanation.

Three of their tools carry an explicit `⛔ NOT WIRED` banner naming exactly what greps to, at which
sha, and why the unwired state is the honest one. That is the "convert the admission into a test"
rule (§A2) industrialized at the binary level.

Two of their tools' package docs are worth reading in full as *examples of the genre* — one
enumerates two ways a parked-work census returns a wrong answer that **reads as good news** (an
orphan commit's diff exits non-zero with empty stdout, which a caller reads as "nothing parked"; and
"the blob differs from main" is not "unlanded", so acting on it reverts newer work), and pins both
with integration tests that build a real repository because *"a unit test over a fixed string cannot
pin either one."*

**Landing in fak.** fak has ~250 `internal/` leaves and ~700 `tools/` scripts. A roster test with a
one-line pinned explanation per binary, failing on unexplained additions, is cheap and directly
attacks fak's largest structural risk: *a binary nobody can find is a binary nobody runs.*

### A14 Generate the verb table from source, with a REFUSES column

**Mechanism.** Their verb documentation is generated **from source, never scraped from `--help`** —
because a sub-verb missing from its own help text is precisely the drift the page exists to catch —
and it carries a column for **what each verb refuses**.

**The cost that bought it:** someone needed a specific sweep, wrote a bespoke script, and only
afterwards found the verb for exactly that question — and then could not use it anyway, because of a
precondition none of the fielded artifacts met. **Both facts were discoverable only by reading
`--help` one verb at a time.**

**What fak has.** `fak index verbs` / `internal/capindex`. The novel part is the REFUSES column: for
a CLI whose entire premise is a refusal kernel, "what will this verb refuse and under what
precondition" is the single most valuable column and fak does not publish it.

### A15 Fan-out size policy: few agents, each heavy

Their AGENTS.md carries two paired sections — *which model* (default every reasoning subagent to
Opus-class at `xhigh`; never silently downgrade below the session model; when unsure whether a stage
is mechanical, it isn't) and *how many* (deliberately opposite in spirit). The second is the one fak
should import:

- **Enumerate before you fan out.** Scout inline first, then pipeline over the *known* work-list.
  "A fleet sized before the work-list exists is sized high every time, and the surplus agents mostly
  re-read what the parent already had in context."
- **Derive N from post-dedup item count, never a round number.** *"If you cannot say what the 50th
  agent reads that the 5th did not, it should not exist."*
- **Prefer depth over width when unsure** — and ⛔ *"independent means different content, not
  different indices: sharding a uniform corpus by stride gives every shard a statistically identical
  view, so N shards return one answer N times."*
- **⛔ Ultracode being on is permission to orchestrate, not an instruction to spend.**
- **⛔ But do not buy small with a silent cap.** If you drop work to stay small, log what was
  dropped. *"Small is a scope decision you disclose, not a gap you leave the reader to find."*

They also ban fan-out **from inside a fleet worker**, with a measurement: 2 of 18 workers used the
Agent tool and between them burned $58.60 against a ~$2.50 median — **46% of the entire wave's
spend**, 77% of it on a cheaper model, because **subagents ignore the model the worker was launched
with.** That last clause is a harness fact fak's dispatch fleet inherits directly.

---

## B. Doctrine worth lifting into fak's AGENTS.md

Short, high-density rules that cost nothing to adopt:

1. **SIMPLE, then CORRECT, then FAST — and the ranking is a tie-breaker between two defensible
   designs, never a licence to deliver less.** SIMPLE ranks first *because it is what makes CORRECT
   checkable*: "a design nobody can hold in their head cannot be shown correct, only hoped about."
2. **A self-report is a sentence you added to the theory, and the theory is exactly the thing whose
   truth is in question.** A recorded verdict and a re-derivation from stored evidence are different
   **kinds of object**, not two strengths of the same evidence.
3. **Types are the cheap proofs.** A constructor that can only be called with valid arguments makes
   possessing the value a proof the check ran; a two-state sum makes "we forgot to say"
   unrepresentable; a request type with *no source field at all* means a client cannot claim one.
   ⛔ The counter-example to watch: **an identity carried as a bare `string` is a join by convention,
   and conventions are not checked.**
4. **A finite witness is not a proof of equivalence.** Golden outputs are agreement-on-observations,
   not isomorphism, and enlarging the golden set does not close the gap *in kind*. Hence: **cache on
   syntax, trust on semantics.**
5. **An answer that lives only in a GitHub thread is not delivered.** A comment thread is a
   self-report surface: it cannot be re-derived, linted, or indexed. Land the artifact in the same
   breath as the comment, and let the comment **cite** it rather than **be** it.
6. **Coming back empty is a SUCCESS.** "Audited, found nothing worth changing, here is what I
   checked and the probe I validated it with" is a successful tick. **A loop that believes it must
   produce a change will produce a bad one.**
7. **Never inherit a clean verdict from a doc — including this one.** Their skill file once asserted
   a corpus was "currently clean on all six clauses"; it was false the minute it was written and its
   only effect was to tell every subsequent tick not to look. **Running the checker costs a second
   and cannot be stale; a sentence in a skill file is stale the moment somebody edits a page.**
8. **A clean verdict bounds only what that tool checks.** A link checker proves a link *resolves*;
   nothing proves the target still says what the citing doc claims.
9. **A ticket is an index too** — and the one that goes stale fastest, because it is the only one a
   landed commit can falsify *without touching a single file the commit names*.
10. **The honest stub.** A capability that is not implemented exits non-zero **naming its ticket** —
    never a silent no-op, never a stub that reads as success.
11. **A null is a result.** Post it. Their first post-change fleet wave was published as an explicit
    null, with the reasoning for why it could not yet be read as a pass.
12. **Distribution tests, not thresholds.** Their first acceptance bar would have been cleared by
    **45% of the pre-change waves it was meant to discriminate against**. ⇒ compare *distributions*
    over ≥5 samples; a single sample under the bar is a coin flip dressed as a result.
13. **Before retracting a finding, check whether the thing you alerted on changed *because of* the
    alert.** Their fleet monitor correctly flagged a problem, the orchestrator fixed it, and the
    monitor then retracted its own true finding as "a timing artifact" — crediting the system with
    discipline it acquired only mid-flight, so the next run would inherit the wrong lesson.
14. **Every probe reads a moving system: say WHEN each side of your comparison was sampled**, or the
    probe measures the gap between the two samples rather than the thing you asked. (They hit this
    twice in one session, once as a false positive and once as a false negative.)

---

## C. Defects and gaps in fak / DOS the downstream repo measured

These are the highest-value part of this review for fak specifically: independent, adversarial,
measured findings about fak's own substrate. **Each needs re-verification at fak HEAD before
action** — they were measured on the downstream repo's pinned versions, and § C7 shows why that
matters.

| # | Finding | Status against fak HEAD |
|---|---|---|
| C1 | **`arbitrate` without `acquire` is a no-op.** Against an empty WAL *every* lane arbitrates GO, so a session that only "asks the referee" gets an unconditional yes and collides anyway. | Not re-checked; consistent with fak's own memory notes. |
| C2 | **The MCP `dos_arbitrate` is pure — "state in, decision out, no I/O"** — it judges the `live_leases` array **you hand it**, which defaults to empty. On its own it answers GO no matter what siblings hold. And there is **no lease MCP tool at all**, so MCP alone can neither take a lease nor see one. | Matches the MCP server's own documented posture. ⇒ **the MCP lane surface is truth-only and must be documented as such**, or agents will keep trusting a meaningless GO. |
| C3 | **`acquire --lane X` can return `"journaled": true` for a *different*, auto-picked lane.** A caller checking only the flag holds a random lane while its own files sit unprotected, and pins a lane nobody needed. Same trap on `arbitrate`: `"outcome": "acquire"` may describe the fallback — read `auto_picked` and `reason`, never the outcome alone. | **Independently confirms** an existing fak memory note. The downstream fix is a wrapper that compares returned lane to requested lane and exits non-zero with the vendor's own reason — worth landing in fak rather than in every caller. |
| C4 | **Leases have no TTL**, so a session that ends without releasing pins its lane forever — and exclusive lanes then become unreachable fleet-wide. Their fix: a reap verb with `--stale-after` (default 2h), **dry by default**, printing per-drop evidence, requiring `--yes`, and **failing closed on an unreadable clock**. | fak's *own* `refs/fak/locks` leases already carry `TTLSeconds` (`internal/devindex/orient.go`), so this is a DOS-side gap with a proven fix shape. |
| C5 | ⛔ **Never judge a lease's liveness by its `pid`** — it is the ephemeral CLI child and **reads dead for every lease, including one taken a second ago.** Judge by age, from the heartbeat when there is one, else acquisition. | ⚠️ **This contradicts a note in fak's own memory index** that advises checking the PID before honoring a REFUSE. One of the two is wrong; reconciling them is a concrete next step. |
| C6 | **Worktree WAL split.** The journal resolves from CWD, so a worktree gets a *private* journal: measured 2 lines against the main checkout's 9,145, with `live` returning `[]` in the worktree while the main checkout showed two live leases **at the same instant**. Both directions fail silently and `"journaled": true` still comes back. ⇒ **worktree isolation protects the working tree and *disables* the lane mechanism** unless the workspace is redirected. | Directly relevant to `fak worktree worker prepare/land/reap`, which is the sanctioned isolation path. Needs a check that the lane substrate survives it. |
| C7 | **`fak orient` is documented downstream as unusable as the pre-edit check for root docs** — `lane=unknown` for multi-line lane trees, and `lease=none` even for an actively held lane. | **Half confirmed at fak HEAD.** The lease half is real: `cmd/fak/orient.go:49,92` reads `refs/fak/locks` via `internal/leaseref`, while DOS journals to `.dos/lane-journal.jsonl` — **two lease substrates in one workspace, and neither reader can see the other's holders.** The multi-line half I could **not** confirm: `internal/devindex/orient.go:124` `splitLaneTree` splits on **comma**, so the newline case needs a direct test before anyone calls it a bug. |
| C8 | **A wrapper that spawns a positional-root tool without pinning `cmd.Dir`** lets that tool walk up from the caller's CWD and stop at the first config root it finds. With shadow roots inside the repo, the walk lands on one without the caller ever leaving the tree — and **the answer it gives is not an error**: three directories inside one repo returned "22 live leases", "no live leases", "no live leases" at the same moment. | Generic. Audit fak for `exec.Command` sites that spawn root-resolving tools without an explicit `Dir`. |
| C9 | **`release` can genuinely fail while your loop records success** — the vendor emitted a traceback instead of its JSON, and a loop branching on "the command ran" logged a successful release for a lease that was still held. Only an independent count caught it (11 held vs 10 logged). ⇒ **branch on the output, then re-verify the count.** | Generic lesson for every fak wrapper over the DOS CLI. |
| C10 | **62% of the lane journal is refusal noise.** Measured 22.3 MB / 51,043 records: `REFUSE` 31,644 vs `ACQUIRE` 325. **One livelocked agent produced 91% of all fleet-wide admission refusals over 4h12m.** Every agent reads this file on every admission check, so **one spinning `--wait` loop taxes the entire fleet** with read amplification. | ⭐ **Design ask worth filing: `acquire --wait` needs exponential backoff and a refusal cap.** Also: the journal needs compaction, or admission cost grows without bound. |

---

## D. Where fak is ahead — do not regress these

Stated so the import list above is not mistaken for a scorecard. The downstream repo has no
equivalent of: the guard kernel and its policy surface, the concept-admission gate, the memory
substrate, the dispatch fleet and its issue routing, the worker-worktree lifecycle, the ~60-skill
library, the scorecard ratchet family, or the trunk-red convergence class (§A12). It explicitly
**models its repo guards on fak's** and **cites `internal/shipgate`'s non-forgeable keep-bit as the
rule it is re-rendering in shell** because it had no Go home for it.

Its contribution is *sharpening*, not breadth: it took fak's mechanisms into a hostile environment
(one shared checkout, ~30 sessions, a hard delivery date) and added the four things fak's versions
lack — **domain-printing, floors, range scoping, and measured baselines.**

---

## E. What was filtered out, and why

Applied as a hard filter, not a judgement call:

- **Client identity, the engagement, the product domain, and the vendor stack** — removed entirely.
  Nothing in this note names them, and nothing above depends on them.
- **Their internal ids** (issue numbers, `DA-nn` confusion-map ids, wave names, commit shas) —
  removed. Where a mechanism is described, it is described by shape.
- **Absolute cost totals and roster counts tied to their account** — removed. The **rates and ratios**
  are retained (cost-per-worker bands, share-of-bill, landing percentiles, refusal counts) because
  they are properties of *the agent harness*, which is fak's own domain, and because the downstream
  repo's own rule applies: ⛔ **the fleet was live when those numbers were taken, so every absolute
  total is a snapshot; compare rates, never totals.** Do not republish the absolutes.
- **Their "Go only, never Python" hard law** — noted here, not recommended. It is a real divergence:
  their repo-internal tooling is Go, run with `go run ./tools/<x>` and **never built** (a convention
  that is self-guarding, because building one writes an untracked binary into the module root that
  their own gate refuses), while fak's `tools/` is ~700 Python files. Their pre-commit gate refuses
  Python outright. Whether that shape suits fak is a scope decision well outside this review — but
  the *transferable* half is the **`cmd/` vs `tools/` boundary as a rule rather than a roster**, and
  one argument from it that is unconditionally right: **a commit gate built out of the product
  binary could not gate a commit that breaks the product binary, so it must not be a verb.**
- **Their organisation's values list** — removed. The transferable idea, kept in §B, is the *form*:
  a values list is only more than decoration when each entry is paid by a **named mechanism** rather
  than by good intentions.

---

## Next checkable steps, in order

1. **§A1 grep** — one pass over `tools/*_scorecard.py` for the failure path that emits a numeric
   default where the probe could not run. Cheapest, highest-value, and it is the precondition for
   trusting anything else on the board.
2. **§C5 reconcile** — settle the lease-liveness question (pid vs age) against DOS HEAD, and correct
   whichever of the two records is wrong.
3. **§C7 direct test** — feed `fak orient` a multi-line lane tree and read the `lane=` column; and
   decide whether the two lease substrates (`refs/fak/locks` vs `.dos/lane-journal.jsonl`) get
   unified, bridged, or documented as disjoint.
4. **§A8 transplant** — the ordinal landing deadline and `LANDED_SO_FAR:` into
   `tools/issue_worker_prompt.py`. Smallest diff with the largest measured effect available.
5. **§A9 file** — the guard-result-shape gap: a refusal-terminated session must be distinguishable
   from a clean one in the harness result, or no fleet instrument can ever see it.
