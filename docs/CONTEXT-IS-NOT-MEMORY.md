---
title: "Context is not memory: the truth-duration axis in fak"
description: "Why fak separates context from memory by how long a fact stays true, and enforces a write-time gate that defaults ephemeral facts to expire, not persist."
---

# Context is not memory — the durability axis the KV story leaves out

*Why "it's 3pm right now" and "I prefer afternoon meetings" are not the same kind of
fact, why a memory system that can't tell them apart is dangerous, and the one
write-time decision that separates them.*

## TL;DR

The agent-memory literature has spent its effort on **where a remembered value lives,
how it's named, and whether it can be safely shared** — fak's own four-layer story
(routing / addressing / fusion / semantics, see
[`MEMORY-LAYERS-EXPLAINER.md`](MEMORY-LAYERS-EXPLAINER.md)) is itself a map of that
*spatial/trust* axis. But there is a second axis, orthogonal to all of it: **how long is
this fact true, and should it therefore become memory at all?** A few have *named* this
axis (cognitive science decades ago; bitemporal databases; a 2026 agent-memory paper —
§5 is honest about all of them). What essentially nobody does is **enforce it as a
write-time gate**: classify truth-duration at the moment a value would cross into durable
store, and *refuse the promotion by default.* Production memory systems leave it to an
LLM's "seems useful later" judgment or to read-time ranking — never to a gate.

The clean statement: **context and memory are not separated by size, recency, or
where the bytes sit. They are separated by truth-duration.** *Context* is what is true
*now* and useful *now* — the current time, the file you have open, the step you're on
in a checkout flow, the mood the user is in this afternoon. *Memory* is what stays true
across the situations where it might be retrieved — a preference, an identity, a
learned skill, a relationship. The two are different **because of how long they remain
valid**, not because of how recently they arrived or how much room is left in the
window.

The operator's example is the whole thesis in one line: *"the context is that it's X
time, but I don't want that in memory in general."* "It's 3pm" must be **in context**
(the model needs it to act now) and must **never be promoted to memory** (it will be
false in an hour and actively misleading tomorrow). A memory system with no principled
answer to *what earns promotion* gets this exactly backwards: it remembers the
timestamp and forgets the preference, because its write trigger is salience or
overflow, and "it's 3pm" is salient and "I like afternoons" was never said loudly.

**This is a write-policy problem, and it is decidable at exactly one place: the moment
a value crosses from the live turn into durable store.** That moment is the
write-time admission gate fak already owns for *quarantine* — the context-MMU. The
missing rung is to make that gate classify not just *is this safe to keep* (trust) but
*how long is this true* (durability), and to default ephemeral facts to **expire, not
persist.** Forgetting the timestamp is not a failure of the memory system. It is the
memory system working.

---

## 1. The two axes are orthogonal — and the field only built one

It is worth being precise about what's new here, because fak already has a deep memory
story and this must not be a restatement of it.

The existing story (S1–S6 in `DISAGGREGATED-AGENT-MEMORY.md` (private companion — not published),
and the four-layer explainer) is about a **single value** — *one cell* — and asks:
where does it live, what's its name, can two readers share it, may a writer mutate or
evict it, who wrote it, who may act on it. Call this the **spatial / trust axis**.
Every one of those questions presumes the value *exists as something worth holding* and
asks what may be *done* to it.

The axis this doc is about is upstream of all of that. Before "where does this cell
live and who may touch it," there is: **should this become a durable cell at all, or
is it situational state that should evaporate when the situation ends?** Call this the
**temporal / durability axis**. It is a property of the *fact*, not of the *cell* —
and it is decided at *write* time, before any of the S1–S6 questions are even asked.

```
                    TEMPORAL / DURABILITY AXIS  (this doc)
                    "how long is this true — should it become memory?"
                    ephemeral ───────────────────────────► durable
                        │                                      │
        "it's 3pm" ─────┤                                      ├───── "prefers afternoons"
   "user is in checkout"┤                                      ├───── "user's name is Sam"
     "model is on step 3"┤                                     ├───── "deploys go through staging"
                        │                                      │
   ─────────────────────┼──────────────────────────────────────────────────►
                        │              SPATIAL / TRUST AXIS  (S1–S6, the KV story)
                        │              "where does the cell live, who may touch it?"
                        ▼              routing · addressing · fusion · mutation ·
                  belongs in CONTEXT,  isolation · provenance · capability · arbitration
                  must NOT promote
```

The two are independent. A *durable* fact still has to be addressed, isolated, and
attributed (the S1–S6 questions apply to it). An *ephemeral* fact is one the S1–S6
questions should **never get to ask about**, because it should never have been written
to the durable tier in the first place. The field built the horizontal axis and mostly
skipped the vertical one. **The vertical one is where "it's 3pm" lives.**

---

## 2. Why systems get this backwards — the write trigger is the wrong variable

Naive and even sophisticated memory systems decide *what to remember* using one of a
handful of triggers. None of them is durability, and that mismatch is the bug:

- **Overflow / summarization-on-full.** When the context window fills, summarize the
  oldest turns and store the summary. The trigger is *running out of room*. But
  running out of room has nothing to do with whether a fact stays true — you summarize
  whatever happened to be old, timestamp and mood and preference alike, and the summary
  launders the ephemeral into the permanent. (This is the MemGPT / virtual-context
  paging shape: it solves *space*, not *durability*.)
- **Recency.** Keep what's recent, drop what's stale. But "it's 3pm" is maximally
  recent and minimally durable; recency is *anti-correlated* with what you want for a
  timestamp. Recency is a good proxy for *relevance to the current turn* (i.e. for what
  belongs in **context**) and a terrible proxy for *what belongs in memory*.
- **Importance / salience scoring.** Score each observation and keep the high-scorers
  (the Generative Agents "memory stream" shape: importance + recency + relevance). But
  salience answers "how much does this matter *right now*," which is again a context
  question. A fire alarm is maximally salient and entirely ephemeral. A quietly stated
  lifelong preference is low-salience and maximally durable.
- **Explicit user save ("remember this").** Correct when it fires, but it offloads the
  whole classification onto the user, and the interesting failures are exactly the ones
  the user *didn't* think to flag — the assistant silently promoting a one-off remark.

The through-line: **every common write trigger is a proxy for "relevant to the present
moment," which is the definition of context — and then the system uses it to decide
membership in memory.** That category error is why you get the two signature failures:

1. **The ephemeral promoted.** A timestamp, a location, a transient mood, a one-time
   task state gets written as if it were a standing fact. The user mentions once that
   they're stressed; the assistant now treats "user is anxious" as a durable trait and
   colors every future reply. The user is in a checkout flow; the agent remembers "user
   is buying X" forever. *"It's 3pm" becomes a permanent belief about the user's day.*
2. **The durable dropped.** Meanwhile the genuinely durable fact — stated quietly,
   long ago, not salient when it was said — ages out under the recency/overflow policy,
   because nothing about the write trigger noticed it would still be true in a year.

Both failures are the *same* root cause: the system has no representation of
**truth-duration**, so it substitutes a present-moment proxy and gets a present-moment
answer to a long-horizon question.

---

## 3. The relatable examples — context-only facts vs memory-worthy facts

The distinction is intuitive once you have the right pairs in hand. In every pair, the
left item belongs in **context** (the model should know it *to act now*) and must
**not** be promoted to durable memory; the right item belongs in **memory** (retrieve
it into context when the situation calls for it).

| Situational — context only, let it expire | Durable — memory-worthy, retrieve when relevant |
|---|---|
| "It's 3:47pm." | "I prefer afternoon meetings." |
| "You're in the checkout flow right now." | "My shipping address is …" |
| "I'm in a hurry today." | "I like concise answers." |
| "I'm frustrated with this bug." | "I'm a Go developer." |
| "The terminal is showing an error." | "I run tests through WSL on this box." |
| "We're on step 3 of the wizard." | "I always want a confirmation before deletes." |
| "It's raining here now." | "I live in Seattle." |
| "The user just pasted a stack trace." | "This service deploys through staging first." |
| "The current branch is `feature/x`." | "We work directly on `main` in this repo." |
| "I'm tired, keep it short." | "Default to short answers unless I ask for detail." |

Notice the *pattern* in the pairs, because it is the actual classifier:

- **The same surface fact can be either, depending on the verb.** "It's raining" is
  context; "I live somewhere it rains" is memory. The ephemeral version is an
  **observation of the present**; the durable version is a **disposition or standing
  state**. The job of the write gate is to keep the observation and *not* mint the
  disposition unless there's evidence the disposition is real.
- **The dangerous promotions are the left column dressed as the right.** "I'm in a
  hurry today" → "user always wants speed over thoroughness." "I'm frustrated with this
  bug" → "user is generally negative." The harm isn't that the fact was wrong when
  observed — it's that an observation-of-the-present was **generalized into a
  standing-trait** with no warrant. Most real-world "creepy memory" complaints about
  consumer assistants are exactly this move.
- **The clean memory-worthy facts share a tense.** Read the right column: they're
  habitual / dispositional ("I prefer," "I always," "we work"), not punctual ("right
  now," "today," "currently"). Tense and aspect are a startlingly good *cheap* signal
  for the durability classification — punctual/progressive aspect leans ephemeral,
  habitual/stative aspect leans durable. (Not sufficient on its own, but a strong
  prior, and computable without a model call.)

A second worked example, because it's the operator's and it's the sharpest: **"it's X
time."** The agent absolutely needs the current time *in context* — to schedule, to say
"good morning," to compute "the deploy was 20 minutes ago." It must **never** write
"the time is 3pm" to memory, because (a) it's false within the hour, and (b) the next
session that retrieves it will act on a stale time as if current — the
*stale-as-current* failure, which is worse than not knowing the time at all, because the
agent doesn't know it's wrong. The correct durable residue of a thousand "it's X time"
observations is not any timestamp; it's a *derived disposition* — "the user is usually
active in the afternoon" — which is a different, genuinely durable fact that a
consolidation step could mint, and which is itself never a raw timestamp.

---

## 4. Our push: durability is a write-time classification, and the default is *expire*

Here is the part that is fak's to claim, because it follows from where fak already
stands. Three moves.

### Move 1 — Promote on durability, not on salience or overflow.

The decision "does this become memory" should be a function of **estimated
truth-duration**, computed at the write boundary, *independent of* how salient or
recent or space-pressured the value is. Concretely, every value crossing the
context→memory boundary gets a **durability class** — the same shape as a TTL
(time-to-live) on a cache entry, but semantic rather than clock-driven:

- **`turn`** — true only this turn (the open file, the current step, the pasted error).
  Lives in context, dies at turn end. *Never* eligible for memory.
- **`session`** — true for this session (the task at hand, the working branch, today's
  mood). Lives in context for the session, dies at session end.
- **`bounded`** — true until a stated expiry or a superseding event ("I'm on vacation
  until the 30th"). A durable cell *with a validity interval* — must carry its
  expiry and be re-checked, never read as timelessly true.
- **`durable`** — true across sessions until explicitly revised (preferences, identity,
  learned procedures). The only class that earns an unconditional write to long-term
  memory.

The headline policy inversion: **the default for an un-classified observation is the
shortest-lived class that fits, not the longest.** Naive systems default to *persist*
(everything is remembered unless evicted); the durability-correct default is *expire*
(nothing is promoted unless it earns `durable`). This is the single most important bit,
and it's the operator's instinct exactly: *"I don't want that in memory in general"* —
the **general** case is don't-remember; promotion is the exception that must be earned.

### Move 2 — fak already owns the place this decision must be made.

The durability class can only be assigned where the value crosses from the live turn
into durable store — at the **write-time admission gate.** That is not a new component
fak would need to invent; it is the context-MMU, which *already* runs an admit-time
verdict on every value for a *different* property (trust: admit / transform /
quarantine — `internal/ctxmmu`, `kvmmu.AdmitResult`). Durability is a second verdict
on the same gate:

- The gate already sees **what a span is** (tool result vs reasoning vs user text — the
  state machine gives it; a serving engine on an anonymous token stream cannot).
- The gate already sees **who wrote it** (provenance stamp, `internal/ifc`).
- The missing field is **how long it's true** — a `durability` tag alongside the
  existing taint/provenance tags, defaulting to the shortest class.

This is why the durability axis is *fak's* to add and not a serving engine's: the
classification needs the structure (turn boundaries, span types, principal identity)
that exists **only at the agent syscall boundary** and is erased at the token-serving
boundary. The same vantage that makes quarantine decidable makes durability decidable.
It is one more admit-time tag on a gate that already runs.

The seam is not hypothetical — it is already shaped for this. The admit-time `Verdict`
(`internal/abi/types.go`) is a discriminated union carrying a closed `Kind`, a closed
`Reason`, *and* an explicitly **OPEN `Meta map[string]string`** ("ignored if unknown").
A `durability` class is exactly an additive `Meta` tag on the verdict the gate already
returns — and because `Meta` is forward-compatible by construction (an older reader
drops an unknown key rather than breaking), the durability tag can ship without
touching the closed trainable verdict set. The shortest-lived default is itself the
fail-closed posture the ABI already takes elsewhere: an unknown verdict kind resolves
to its fail-closed `FallbackClass`, and an un-classified observation resolves to the
shortest-lived durability class. *Same gate, same fail-closed instinct, one more tag.*

### Move 3 — the ephemeral/durable split makes the *forgetting* primitive principled.

fak's sharpest primitive is **coherent middle-eviction**: remove a span from a kept
sequence, byte-identical to never having seen it (S2, `Kraw` re-rotation). Today the
*trigger* for eviction is trust (quarantine a poisoned span) or pressure (LRU-ish). The
durability axis gives eviction a **principled, non-pressure trigger**: a `turn`-class
or `session`-class span is *evicted on schedule by its own TTL*, not when the cache
happens to fill. "It's 3pm" isn't dropped because the window got tight; it's dropped
because **its truth-duration expired** — and the eviction is exact, so the surviving
context is byte-correct as if the timestamp were never there. **Forgetting-by-design,
on a clock the fact itself sets, with a bit-exact result.** That is the union of fak's
two strongest ideas — the durability classification (this doc) and the exact eraser
(S2) — and neither serving-layer caches nor naive memory stores can do it: they either
keep everything until pressure forces a blind LRU drop, or they forget approximately.

### Why the default must be *expire* — the failures aren't symmetric.

It's tempting to call this a tuning knob: set the promotion threshold somewhere
sensible and accept some error in both directions. But the two error directions have
**very different costs**, and that asymmetry is what forces the default.

- **Failing to remember a durable fact** (a false-negative promotion) is *recoverable
  and self-correcting*: the user states the preference again, or the agent asks. The
  cost is a little redundancy. The fact is still true, so a second chance to capture it
  always comes.
- **Remembering an ephemeral fact as durable** (a false-positive promotion) is
  *silent, persistent, and acts as confident truth*: the stale timestamp, the one-off
  mood frozen into a trait, the checkout-flow state that haunts every future session.
  Nobody re-states "actually I'm *not* anxious in general" because nobody knows the
  agent quietly concluded it. The fact is now false, surfaced as true, with no signal
  that it's wrong — the **stale-as-current** failure, which is strictly worse than
  absence.

A false negative costs a re-ask; a false positive costs a wrong belief that nobody
knows to correct. When the costs are that lopsided, you don't center the threshold —
you **bias hard toward the cheap error.** Defaulting to *expire* makes every
un-earned promotion a recoverable re-ask instead of a silent wrong belief. This is the
same logic as fak's fail-closed admission: an *unwitnessed* claim is refused, not
trusted, because trusting-when-wrong is the expensive direction. Durability inherits
the posture — **un-classified means ephemeral, because remembering-when-wrong is the
expensive direction.**

> **The one-line version.** *Trust* decides whether a value may enter memory; *durability*
> decides whether it should — and for how long. fak built the gate for the first; the
> second is the same gate with one more tag, and its correct default is **expire.**

---

## 5. The honest prior art — who's near this, and where the gap really is

We are not the first to notice that context and memory are different, or that time
matters. The contribution here is narrower and sharper, so it's worth being exact about
what's already known. (All of the following was cross-checked against primary sources in
a research sweep; the attributions below are the *corrected* ones — several popular
restatements get the dates or the credit wrong.)

**Cognitive science has the deepest version, and got there decades ago.** Tulving (1972)
split **episodic** memory (specific events located in time and place — inherently
contextual, time-stamped) from **semantic** memory (decontextualized, generic facts).
The crucial operation for us is the one *between* them: turning a context-rich episode
into a context-stripped fact is exactly the promotion decision — and the brain does
*not* promote everything. Complementary Learning Systems theory (McClelland, McNaughton
& O'Reilly, 1995) explains *why* there must be two systems at all: a fast, sparse
hippocampal learner for one-shot episodes and a slow, distributed neocortical learner
for generalized knowledge, because cramming fast learning into one overlapping store
causes **catastrophic interference**. Consolidation (largely during sleep replay) is not
archival copying — it is *selective transformation* that distills gist and lets verbatim
detail decay. And **forgetting is an adaptive feature** (directed/retrieval-induced
forgetting actively suppress competitors), not decay-by-neglect. The honest summary:
*the AI field imported the storage-hierarchy metaphor (working vs long-term) and skipped
the selective consolidation that is the entire point.* (One correction to the usual
retelling: "mental time travel" is Tulving 1983, not the 1972 paper; and hippocampal
*index* theory — store a sparse pointer, not the content — is Teyler & DiScenna 1986,
distinct from the pattern-separation/completion machinery often bolted onto it.)

**Databases solved the time-validity half cleanly, and the agent field is re-deriving
it.** Bitemporal modeling (Snodgrass; standardized in SQL:2011) gives every fact a
**valid-time** (when it's true in the world) distinct from a **transaction-time** (when
the system recorded it). "It's 3pm" is a fact with a ~one-hour valid-time; storing it as
a timeless assertion is simply a modeling error the database world fixed long ago. The
strongest production port into agent memory is **Zep/Graphiti** (Rasmussen et al., 2025):
a bi-temporal knowledge graph that stamps every fact edge with `(t_valid, t_invalid)`
and, on contradiction, **invalidates rather than deletes** — preserving history. But
note the precise limit: Zep models durable-fact *revision over time*; it does not gate
*promotion* — everything extracted becomes a graph fact, just a temporally-bounded one.

**The primitive we lean on already has a name — and it's from 2023, not 2026.** Zhang &
Choi, *"Mitigating Temporal Misalignment by Discarding Outdated Facts"* (EMNLP 2023,
arXiv:2305.14824), define the task of **fact duration prediction** — predicting *how long
a given fact will remain true* so a model can distinguish stable from volatile knowledge.
That is exactly the estimator the durability classifier needs, named three years before we
wrote this. **We do not claim to have invented truth-duration estimation; we build on it.**
Our contribution is the *systems* move downstream of the predictor — making that estimate
an *enforced write-time promotion gate* with expire-by-default — not the ML primitive of
estimating duration. (An earlier draft of this doc called a 2026 paper "the first to
formalize the axis"; the sweep corrected that — the axis was named in 2023, and saying
otherwise is exactly the kind of "first to" overclaim this project's honesty discipline
exists to kill.)

**The closest taxonomy and the closest mechanism — both already exist, and naming them is
what makes our claim survive.** Two results sit nearer than anything above, and the gap is
only visible once they're drawn precisely:
- **"Beyond Dialogue Time" (2026)** formalizes **ephemeral-vs-durable as a write-time
  taxonomy** — routing facts to *permanent* / *long-term* / *temporary* classes, with
  "durative" memories carrying validity windows on a semantic timeline separate from
  dialogue time. That is the same *axis* this doc draws. We cite it as confirmation the
  axis is real — prior art to build *on*, not precede.
- **Springdrift** (Brady 2026, arXiv:2604.04660), an auditable persistent agent runtime,
  is the closest *mechanism* neighbor: its Facts store is explicitly "scoped and decayed."
  But the line is sharp and load-bearing: Springdrift decays facts on an **append-only**
  JSONL log replayed chronologically — *decay-by-default on a store that never drops a
  record* — which is the opposite default from ours (**expire-by-default, refuse to
  promote**), and the decay is a read-time half-life, not a write-time admission gate.
- **Cloudflare Agent Memory** (2025) ships the closest *mainstream-vendor* write-time
  split — four first-class types (Facts / Events / Instructions / Tasks) where session
  Tasks are deprioritized after the session. But that is a content-*type* label that
  *deprioritizes*, not an estimated-truth-duration gate that *expires*.

**So where is the actual gap?** Not in "noticing time matters," not in "predicting fact
duration" (Zhang-Choi 2023), not in "naming the ephemeral/durable axis" (Beyond Dialogue
Time), not in "decaying a fact store" (Springdrift) or "typing memory by lifespan"
(Cloudflare) — all of that is covered. The gap is that across the production agent-memory
systems, the *write policy* is one of three families and **none is a principled, enforced
durability gate** (§6 has the verified roster):

1. **Capacity-driven** (MemGPT/Letta, LlamaIndex): promote on overflow — summarize the
   oldest turns when the window fills. The trigger is *"it doesn't fit,"* not *"it's
   durable."*
2. **LLM-judgment** (Mem0's ADD/UPDATE/DELETE/NOOP; Letta block edits;
   ChatGPT/Claude/Gemini auto-synthesis): an LLM decides per-fact what to keep — the
   only place "is this transient?" is even *implicitly* asked, and the systems
   themselves (Mem0's own docs) admit it misclassifies.
3. **Score-based, at *read* time** (Generative Agents' importance+recency+relevance;
   MemoryBank's Ebbinghaus decay+reinforcement): write *everything*, then approximate
   durability post-hoc via retrieval ranking or forget-by-disuse.

The one true cognitive-architecture analog of a principled split — **ACT-R**'s
base-level activation (frequency/recency, context-*independent* → durable) cleanly
separated from spreading activation (context-*dependent* → contextual) — is exactly the
distinction the LLM systems gesture at but do not implement. **The opening, then, is the
one thing none of them has: a write-time admission gate that classifies estimated
truth-duration and refuses to promote a contextual fact to durable memory — with
expire as the default.** That is a reference-monitor posture, and a reference monitor at
the agent syscall boundary is precisely what fak is.

---

## 6. SOTA, verified — the roster behind §5

Every row below was cross-checked against a primary source; verdicts are from an
adversarial fact-check pass (`CONFIRMED` = primary source agrees; `PARTLY` = true with a
correction noted). Nothing here is `REFUTED` — the prior-art shape held up — but several
attributions were *corrected*, and those corrections are the value of the pass.

### Agent / LLM memory systems — the write-policy roster

| System | Write policy (what triggers a durable write) | Handles ephemeral-vs-durable? |
|---|---|---|
| **MemGPT / Letta** (Packer et al., 2023) | Overflow: summarize main context → external store when the window fills, via LLM tool calls | No — trigger is *capacity*, not durability |
| **Letta memory blocks** | Agent rewrites labeled, size-capped blocks (`human`, `persona`) | Soft proxy (label routing); nothing stops a transient fact landing in the durable `human` block |
| **Mem0** | Two-pass: extract candidate facts → LLM picks ADD/UPDATE/DELETE/NOOP vs retrieved similars; scopes conversation/session/user | Closest in production (session vs user scope) — but scope is an LLM call its **own docs say misclassifies**; no native expiry |
| **Zep / Graphiti** (Rasmussen et al., 2025) | Extract to a **bi-temporal** graph; stamp `(t_valid, t_invalid)`; invalidate-not-delete on conflict | Models fact *revision* over time, not *promotion* — everything extracted is written |
| **Springdrift** (Brady, 2026) | Append-only JSONL Facts store, replayed chronologically; entries "scoped and **decayed**" by read-time half-life | Closest *mechanism* — but decay-by-default on a store that never drops a record; opposite default from expire-by-default, and read-time not a write gate |
| **Cloudflare Agent Memory** (2025) | Four first-class types (Facts / Events / Instructions / Tasks); session **Tasks deprioritized** after the session | Closest *mainstream-vendor* write-time split — but a content-*type* label that deprioritizes, not a truth-duration estimate that expires |
| **Generative Agents** (Park et al., 2023) | Write **everything** to a flat memory stream; LLM importance score (1–10) + recency decay + relevance gate **retrieval**, not the write | Approximated at **read** time; transient observations are still stored |
| **MemoryBank** (Zhong et al., 2023) | Write all; **Ebbinghaus** strength decays with time, reinforced on recall | Forget-by-disuse — a recalled transient fact gets reinforced *as if* durable |
| **A-MEM** (Xu et al., 2025) | Every interaction → a linked Zettelkasten note; "memory evolution" updates links | Dynamic organization, not a promotion gate; no transient filter |
| **LangGraph / LangMem** | Names the semantic/episodic/procedural taxonomy + hot-path vs background write timing | Plumbing, not policy — *what* to write to which namespace is left to the developer |
| **LlamaIndex Memory** | Token-overflow flush short-term → long-term blocks by priority | Capacity-driven, like MemGPT |
| **ChatGPT memory** (OpenAI) | Two tiers: durable **Saved memories** (explicit or model-judged, auditable) + mutable **Reference chat history** (mined, not item-auditable) | A de-facto durable/contextual split in production, but promotion to Saved is an **opaque** model judgment with no published durability criterion |
| **Claude memory** (Anthropic) | Opt-in, **project-scoped**, user-viewable/editable memory summary via LLM synthesis | User-controllable surface, but no published explicit contextual-vs-durable promotion rule |
| **ACT-R** (cognitive architecture) | Working memory = the *activated slice* of long-term memory; base-level activation (freq/recency, context-independent) vs spreading (context-dependent) | The **principled analog** of durable-vs-contextual — which the LLM systems gesture at but don't implement |

The pattern reads straight down the right column: **everyone separates a live context
from a durable store; almost no one gates the boundary on truth-duration at write time.**

### Cognitive science — the principled grounding (verified)

- **Episodic vs semantic** (Tulving 1972, `CONFIRMED` with date correction): episodic is
  spatio-temporally indexed and context-rich; semantic is decontextualized gist. *The
  context-stripping between them is the promotion operation.* ("Mental time travel" is
  the 1983 elaboration, not 1972 — don't misattribute it.)
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995,
  `CONFIRMED`): two systems are *necessary* — fast sparse hippocampal episodic capture +
  slow distributed neocortical generalization — because one overlapping store suffers
  catastrophic interference. The 2016 update (Kumaran, Hassabis & McClelland) refines it:
  *schema-consistent* new facts can integrate fast — which maps to "a fact consistent
  with known durable preferences can be promoted cheaply."
- **Working memory** (Baddeley & Hitch 1974; episodic buffer 2000, `CONFIRMED`): a small,
  transient, capacity-limited workspace that *binds* streams — the cognitive twin of the
  context window, explicitly **not** the durable store.
- **Bitemporal data modeling** (Snodgrass; SQL:2011, `CONFIRMED`): valid-time vs
  transaction-time; a contradicting update *closes the old interval* rather than
  overwriting. The mature, boring, correct prior art for "true now, not true forever."
- **Fact duration prediction** (Zhang & Choi, EMNLP 2023, arXiv:2305.14824, `CONFIRMED`):
  defines the *task* of predicting how long a fact stays true, to discard outdated facts
  and improve calibration under temporal misalignment. This is the **estimator primitive**
  the durability classifier consumes — named in 2023. fak's contribution is the systems
  move (estimate → *enforced write-time promotion gate* → expire-by-default), not the
  estimation itself; cite this as the thing we build on, never as something we precede.

### Failure modes — the verified harms of getting it wrong

The harms aren't hypothetical, and they're the strongest argument for an enforced gate
with an expire default:

- **Adversarial promotion is a real, named attack.** The **SpAIware** incident
  (Rehberger / Embrace The Red) demonstrated a single untrusted input laundered into a
  *persistent, cross-session* compromise. **OWASP** now codifies this as **Memory
  Poisoning (T1)**, defined verbatim as *"turning a transient attack into a persistent
  behavioral bias."* That is precisely the ephemeral→durable boundary, weaponized — and
  it's exactly the boundary fak's quarantine gate already guards for trust. The
  durability tag closes the *benign* version of the same hole.
- **Stale-as-current is measurable.** The **STALE** benchmark finds even the best model
  is only **~55%** accurate at knowing when its own stored memories have gone invalid —
  because retrieval scores semantic similarity *blind to time*, so a fact "true once"
  resurfaces as "true now." That is the timestamp/location-leak failure with a number on
  it.
- **Over-remembering is self-defeating, not just creepy.** Unbounded retention degrades
  retrieval (memory rot / context pollution), collides with the right-to-be-forgotten
  (GDPR Art. 17 — and "unlearning" is often reversible obfuscation, per CMU work), and
  in sensitive settings causes real harm (durably storing a one-off emotional disclosure
  → the assistant treats a passing state as a standing trait; mental-health researchers
  have disabled memory entirely over exactly this). **Forgetting the ephemeral is a
  correctness requirement, not a nicety.**

### The field is converging on this *now* — and naming the exact gap

Two 2025–2026 signals say the durability axis has moved from "nobody's looking" to "the
named frontier," which sharpens rather than weakens the case — the contribution is the
*enforced write-time gate*, not the observation:

- **A provider shipped the mechanism without the discipline.** Anthropic's context-editing
  + memory tool (Sept 2025) has been publicly critiqued as **"a garbage collector without
  write barriers"** — which is precisely this doc's argument said from the outside. A GC
  reclaims space; a *write barrier* is the check that runs at the moment of a write to keep
  the heap coherent. Shipping eviction (the GC) without an admission gate (the write
  barrier) is exactly "promote freely, clean up later" — the failure §2 anatomizes. **The
  durability gate *is* the write barrier**: the check that runs at the promotion moment, not
  the sweep that runs after.
- **A survey named consolidation the #1 open problem.** The 2026 *"Memory for Autonomous
  LLM Agents"* survey states the thesis almost verbatim — *"Forgetting is not a bug; it is
  a feature… current systems handle it crudely… no validation that safety-critical records
  survive"* — and ranks principled consolidation / learning-to-forget as the top frontier.
  Independent convergence on the problem statement; the open lane is the *enforced*
  mechanism, which is where fak's reference-monitor posture is the natural home.

The positioning consequence (see `DIRECTION-ADVANTAGES-2026-06-19.md` (private companion — not published)):
the "reference monitor for agent *actions*" category is now contested (Microsoft's
"Agent OS" gates effects sub-millisecond). The half no incumbent has assembled is the
**result/memory** half — typed fail-closed RESULT admission **plus** write-time durability
classification **plus** byte-exact eviction, composed in one binary. S7 is the durability
third of that surviving wedge.

### From concept to code — the buildable ladder (tracked)

Rung 1 has **landed** (`[SHIPPED]`); rungs 2–3 are the tracked follow-ons of a sequenced
epic against fak's real seams (**#82**), grounded in `internal/abi`, `internal/ctxmmu`,
`internal/recall`, and `internal/kvmmu`:

- **Rung 1 — minimal, proves the inversion. `[SHIPPED]`.** A write-time `classifyDurability`
  (a cheap lexical/tense prior — *not* a model call, *not* the Zhang-Choi estimator) stamps
  `Verdict.Meta["durability"]` in `ctxmmu.MMU.Admit`; `recall` gained a **default-expire
  promotion gate** (`PromotionWarn` default / `PromotionEnforce` opt-in) that refuses to
  promote a non-`durable` page. The bite test witnesses it end to end: `it's 3pm → turn →
  refused promotion`; `the user prefers afternoons → durable → promoted`. (#82; the
  migrated #497-#500 child references are stale/unrelated in this repo, and live rung-1
  child numbers still need remapping; `TestABIGoldenFreeze` is unmoved.)
- **Rung 2 — bitemporal, kills stale-as-current.** A `recall.Page` validity interval
  (`ValidFrom`/`ValidTo`) + an as-of read gate (`ErrExpired`) makes the `bounded` class
  the first temporally-enforced one (the Zep/Graphiti + SQL:2011 spine). (#81.)
- **Rung 3 — engine-integrated, the distinctive move.** A `Segment` TTL over the bit-exact
  `KVCache.Evict` (`Kraw` re-rotation) so a turn/session span is **forgotten on a clock the
  fact itself sets**, byte-identical to never-having-seen-it — the in-context forgetting a
  pressure-driven LRU cache structurally cannot do. (#80.)
- **Close-out.** [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) durability row + flip this doc's tag to `[SHIPPED]` for what
  landed. (#82; the migrated #503 reference is stale/unrelated in this repo.)

The honest scope is on the tickets: rung 1 ships the inversion with a lexical classifier;
`bounded` is a reserved value until rung 2 gives it a validity field; the byte-exact
guarantee on rung 3 holds only for spans no later token attended (mid-context expiry is a
coherent *compaction*, not never-saw). The seam itself costs zero ABI surface — `Meta` is
the OPEN, forward-compatible map, so none of this moves the frozen ABI golden.

### Rung-1 design contract (the ratified seam — #82)

The six decisions the classifier feature, the promotion-gate feature, and the bite test
all agree on, with the exact shipped symbols:

1. **Attach point = `abi.Verdict.Meta["durability"]`** (`internal/abi/types.go:226`, the OPEN
   map). NOT a new `VerdictKind` (durability is orthogonal to trust — a span can be Allow
   AND turn-class — so it must not collide with the most-restrictive-wins fold) and NOT a
   `ReasonCode` (that is the CLOSED refusal vocabulary; durability is not a refusal).
   Confirmed zero ABI cost: `TestABIGoldenFreeze` serializes only the closed-enum integers,
   so a runtime `Meta` stamp does not move it.
2. **Classifier signature = `classifyDurability(c *abi.ToolCall, body []byte) string`**
   (`internal/ctxmmu/mmu.go`, mirroring `ScreenBytes`). It leans on bytes (and may consult
   the tool); it does NOT yet take a turn index / session id / principal / as-of clock —
   threading those into the rank-10 `ResultAdmitter` signature is a NAMED follow-on, not
   this rung.
3. **Emitted vocabulary v1 = {turn, session, durable}** (`ctxmmu.DurabilityTurn` /
   `…Session` / `…Durable`). `bounded` is a RESERVED value the lexical prior does not emit
   (no validity-interval home until rung 2); readers degrade unknown/`bounded` fail-closed.
4. **Fail-closed default = `turn`** at both writer and reader (`ctxmmu.classifyDurability`'s
   `default:` arm and `recall.promotionClass`), mirroring `abi.FallbackDeny`. Unclassified ⇒
   ephemeral, because a false-positive promotion (a poltergeist fact recalled as current) is
   the expensive direction; a false-negative is recoverable.
5. **`recall.Page.Durability string` (json `durability,omitempty`)** (`internal/recall/recall.go:61`)
   so the disposition is auditable in `manifest.json`. JSON, not ABI — no golden touch.
6. **Two-commit honesty split.** Commit 1 = classifier + `Meta` tag + `Page.Durability`
   stamped, promotion gate in **WARN** (record the class, count a would-refuse, still
   persist) — non-behavior-changing, so every caller can be audited. Commit 2 = the
   **ENFORCE** posture (`PromotionEnforce`) where a non-`durable` benign page is not promoted.

**Classifier honesty scope:** v1 is a regex/keyword/tense prior (punctual deictics + bare
clock times ⇒ turn; habitual/stative frames ⇒ durable), explicitly **NOT** the Zhang-Choi
fact-duration estimator (§5), which has no callsite and is deferred.

**Realized posture (the WARN deliverable, honest):** `PromotionWarn` is the *default*, and
`PromotionEnforce` is **opt-in** (`Recorder.WithPromotion`). The enforce flip is gated on a
caller audit, because the existing benign-round-trip callers expect every non-quarantined
result to persist: `internal/cdb/ingest.go` (the production session-ingest path) and the
`internal/recall` round-trip tests record turn-class benign bodies and would lose them
under a global enforce default. (`recall/dream.go` is a read-side consumer of an
already-loaded image, not a `Recorder` caller, so the enforce flip does not affect it.)
The WARN audit count
(`Recorder.RefusedPromotions`) is the signal those callers are migrated; flipping the
*default* to enforce is the named follow-on, not rung 1.

**Migration trap (flagged, not conflated):** an empty `Durability` on an ALREADY-PERSISTED
page must later default to **`durable`** — it crossed the old promotion-free gate — the
OPPOSITE of the in-gate `turn` default for a live observation. That inverse default lands
with rung 2's read gate; conflating the two would silently expire the existing recall store.

### Disposition-minting gate — the generalization boundary, one level up (#1598)

Rung 1 answers *"how long is this span true"* per write. It does not answer a narrower,
downstream question a consolidation step can still get wrong: *may ONE situational
observation be generalized into a standing trait about the user at all?* "I am tired
today" is `turn`-class and correctly evicted on schedule — but nothing stopped a
summarizer from free-associating it into "user prefers short answers" and minting THAT
as `durable`, which is the ephemeral-promoted failure (§2) one level up: not a raw fact
outliving its turn, but a fabricated durable belief the raw fact never supported.

`internal/ctxmmu/disposition.go` closes that gap with a small, pure, additive type set —
`Observation` (the raw remark), `Disposition` (the standing trait a caller wants to
derive from it), `Evidence` (what backs the generalization: `EvidenceCorroboration` with
independent corroborating observations reaching `MinCorroboration`, `EvidenceUserConfirmed`,
or `EvidenceEstablishedPattern`) — and one pure decision function, `GateDisposition`, that
returns a closed, typed `DispositionOutcome` (`OutcomeMinted` /
`OutcomeRefusedUnsupported` / `OutcomeRefusedInsufficientCorroboration`). It never
silently drops and never silently promotes. The default is refuse: a single
observation with no evidence is `OutcomeRefusedUnsupported` regardless of how
durable-shaped its own tense is, because one utterance minting a cross-session belief is
the expensive direction (§4). This is a second, narrower gate that sits *above*
`classifyDurability`, not a replacement for it — a minted `Disposition` carries
`DurabilityDurable`, the same vocabulary rung 1 already shipped.

---

## See also

- [`MEMORY-LAYERS-EXPLAINER.md`](MEMORY-LAYERS-EXPLAINER.md) — the *spatial/trust* axis
  (routing / addressing / fusion / semantics). This doc is the **orthogonal**
  *temporal/durability* axis the four layers don't cover.
- `DISAGGREGATED-AGENT-MEMORY.md` (private companion — not published) — S1–S6 memory
  semantics; the durability classification here is the natural **S7** (promotion /
  truth-duration), upstream of all six.
- [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — honesty ledger; the rung-1 durability gate (classifier +
  `Verdict.Meta["durability"]` tag + `recall.Page.Durability` + default-expire promotion
  gate) is now `[SHIPPED]`; the TTL scheduler (rung 3), the `bounded` validity-window
  (rung 2), and Dream-time consolidation remain follow-on `[STUB]` rungs.

---

## Appendix: the thesis as one decision tree

For a fact arriving in a turn, the write gate asks two independent questions — and only
one path ends in durable memory:

```
   A value arrives in the turn
            │
            ▼
   ┌─────────────────────────┐   trust axis (S1–S6): SHIPPED gate
   │ Safe to keep at all?     │── no ──► QUARANTINE (held out, page-out, witness to clear)
   │ (poison / secret?)       │
   └─────────────────────────┘
            │ yes
            ▼
   ┌─────────────────────────┐   durability axis (this doc, S7): rung-1 SHIPPED gate
   │ How long is it true?     │
   └─────────────────────────┘
        │        │        │        │
      turn    session  bounded  durable
        │        │        │        │
        ▼        ▼        ▼        ▼
   live in   live in   durable    the ONLY
   context,  context,  cell WITH  class that
   evict at  evict at  a validity earns an
   turn end  session   window,    unconditional
   (TTL)     end       re-checked write to memory
        └────────┴────────┘            │
     never promoted to memory          ▼
     (default for un-classified)   long-term store
```

"It's 3pm" passes the trust gate (it's safe) and lands in **turn** on the durability
gate — used now, evicted on schedule, never written to memory. "I prefer afternoons"
passes both and is the one thing that earns the durable write. *That* is context vs
memory, made into a decision the gate can actually take.
