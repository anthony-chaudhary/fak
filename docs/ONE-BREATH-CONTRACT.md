---
title: "The in-one-breath contract: a named summary block with a countable shape"
description: "The delimiters, position, and countable rules for fak's `In one breath` block — the short, background-free summary a budget-aware loader serves when it cannot afford the whole page — and the written refusal to gate its faithfulness."
---

# The in-one-breath contract

**Audience:** anyone — human or agent — writing or gating a fak explainer page, and
anyone building a budget-aware document loader that has to serve *something* when it
cannot afford the page.

*Status:* the contract is enforced by `fak breath` (`internal/promptlint/breath`) as an
**advisory with a counted ratchet**, not a refusal. fak's doc corpus predates the
contract, so the gate's only claim is *this class is not growing*.

## The block

Every page in a tree under contract opens with one block:

```markdown
# Page title

> **In one breath:** A cache keeps an answer so it is not paid for twice. fak keeps
> that answer inside the kernel. A second question with the same start is answered
> from it.

**One line:** the precise version, where nuance is allowed.
```

**Delimiters.** The block is a Markdown blockquote whose first line begins with the
exact marker `> **In one breath:**`. The block is that line plus every immediately
following blockquote line; a bare `>` or a non-blockquote line ends it. The marker is
matched **exactly**, not tolerantly: a page that spells it differently is reported as
*missing* the block rather than quietly parsed, because the marker is also read by eye
and keyed on by the loader.

**Position.** The block must appear in the lede — **before the first `##` section
heading**. A summary a reader reaches after three sections is not a summary.

**The pairing.** A `**One line:**` paragraph must follow. The block is deliberately
cruder than the page; `**One line:**` is the precise version and is where nuance is
allowed. A crude version with no precise version standing behind it is where an
oversimplification survives, because nothing on the page contradicts it.

## The contract, in numbers

| Rule | Value | Refusal token |
|---|---|---|
| Sentences in the block | **2 to 4** | `BREATH_SENTENCE_COUNT` |
| Words per sentence | **15 or fewer** | `BREATH_SENTENCE_LENGTH` |
| Em-dashes (`—`) | **none** | `BREATH_EM_DASH` |
| Parentheses | **none** | `BREATH_PARENTHESES` |
| Acronyms | **spelled out on the spot** | `BREATH_UNEXPANDED_ACRONYM` |
| Block present | **required** | `BREATH_MISSING` |
| Block before the first `##` | **required** | `BREATH_MISPLACED` |
| `**One line:**` sibling | **required** | `BREATH_MISSING_ONE_LINE` |

These are the numbers the gate enforces, and a test
(`TestContractNumbersMatchThisPage`) reads this page and fails if they drift apart —
so the contract this page *promises* and the contract the gate *applies* cannot
diverge.

**Why an em-dash is banned.** An em-dash is how a second idea is appended to a
sentence without admitting it is one, so it defeats the one-idea-per-sentence rule the
word ceiling exists to enforce. Make it a second sentence, or move the clause down.

**Why parentheses are banned.** A parenthetical is nuance the reader is asked to hold
while finishing the sentence, and this block's whole premise is a reader who cannot
yet do that.

**What "spelled out on the spot" means mechanically.** An ALLCAPS token of two or more
letters is expanded when the block also contains that many consecutive words whose
initials spell it — "GPU" is expanded by "graphics processing unit", "TTL" by "time to
live". The window may not contain the acronym itself, so "AI is" does not expand "AI".
A token immediately followed by a lowercase file extension (`AGENTS.md`) is a filename,
not an acronym. Mixed-case coinages (`IoU`, `KVCache`) are prose and are not checked.
The allowlist is **empty by default**, and that is deliberate: the block's reader has
no background. `--allow-acronym NAME` exists for a name in capitals that is not an
acronym at all.

## Why this is a gate and not a memo

A prose contract enforced by an author's diligence decays in exactly one direction: the
next page is written in the voice of the page *body* rather than the block, no existing
check notices, and this page goes on describing a shape the tree stopped having. The
decay is invisible **precisely because the block still reads well** — a 30-word sentence
with two parentheticals is good technical writing, it is simply not the artefact this
contract asked for, and nothing but a count can tell the two apart. fak's docs are
written by many concurrent sessions, which is the condition under which "reads well but
is not the artefact asked for" is undetectable by every other gate fak has.

## What this contract does NOT gate, and why

**The gate judges simplicity and brevity. It does not judge accuracy, completeness, or
faithfulness, and it must not be extended to.**

This is not modesty; it is the measured scope decision, and it is the load-bearing part
of the design.

- The TREC PLABA track (arXiv 2507.14096, 2025) ran two years of expert manual
  evaluation over plain-language adaptations and scored four axes: accuracy,
  completeness, simplicity, brevity. Its headline is that top systems *"rivaled human
  levels of factual accuracy and completeness, but not simplicity or brevity."* The axes
  a machine already gets right are the ones a checker cannot mechanically judge; the two
  it fails are exactly the two that reduce to counting words and sentences.
- The tempting extension — compare the block against the page body and prove the simple
  version follows from the precise one — **inverts the signal**. PlainQAFact (arXiv
  2503.08890, 2025) names the reason: the *elaborative explanation phenomenon*. Good
  plain-language writing ADDS content absent from the source (a definition, background,
  an example) to make it comprehensible, so an entailment- or QA-style consistency check
  scores the **best-written** blocks as the **least faithful** ones. Their own fix
  classifies each sentence as source-simplified or elaborative before scoring it, which
  needs a trained model and a judgement this gate has no business making.

So a green `fak breath` means: the block is present, in the right place, short, its
sentences are short, and it avoids the constructions that smuggle nuance past a reader
who has none. **Whether it is TRUE is review's job.** A checker that punished
elaboration would push every block toward a lossless summary of the page, which is the
opposite of what the block is for.

`internal/promptlint/breath` pins this in code two ways: `TestNoJudgementHalfKind` fails if a
faithfulness/entailment/accuracy finding kind is ever added to the closed vocabulary,
and `TestElaborativeBlockStaysClean` is a block that says things its page body does not
and must stay green. Anyone adding an entailment check has to delete those tests and
argue past this section first.

## The ratchet, not a refusal

fak's doc corpus predates this contract, so a gate that denied on any finding would deny
on almost every page. `fak breath` therefore ships as an advisory with a **counted
ratchet**, the same contract `internal/pythongate` runs:

- findings are keyed by `KIND<TAB>path`, which is **stable under editing** — inserting a
  sentence does not renumber anything;
- the baseline stores a **count** per key, not a line number, so fixing one of two
  findings tightens the floor and adding a third is still caught;
- a baseline row that does not parse is a **hard error naming the line**, never a
  skipped line: a lenient parser turns the tool's own bug into someone else's denial;
- the only claim the gate ever makes is *this class is not growing*. It never claims the
  corpus is clean.

Run it:

```
fak breath                         # advisory census over the pages under contract
fak breath --json                  # machine-readable census + findings
fak breath --gate                  # exit 1 only on findings ABOVE the counted floor
fak breath --emit-baseline > internal/promptlint/breath/baseline.txt
```

A run that examined fewer pages than its floor reports `BREATH_SCAN_FLOOR` and exits
non-zero regardless of `--gate`. That failure is about the *tool*, not the docs: this
check reports an absence, so a scan that examined zero pages prints `clean` and is
byte-identical to a tree whose every block is perfect.

## Measured starting census

The counted floor was measured at committed seed
`622df00a9d23a9f0bf8d49a65e9c39b225b2ce9c`, over every Git-tracked Markdown page in
`docs/explainers`:

| Outcome | Pages |
|---|---:|
| Conforming block | **0** |
| Block present but failing | **0** |
| Block missing | **59** |
| **Denominator** | **59** |

The partition is exact: `0 + 0 + 59 = 59`. The corresponding 59 counted findings are
tracked in `internal/promptlint/breath/baseline.txt`; they make the initial advisory honest
without rewriting any existing page or allowing a sixtieth missing block.

## Who consumes the block

The block is the artefact a **budget-aware loader serves when it cannot afford the
page**. That is the direct tie to:

- **#3229** — the context/token budget epic. A bounded, countable per-document summary
  is a budget line item a planner can price before it spends.
- **#3535** — cut the ~10k-token turn-1 forced `AGENTS.md` read via a sectioned loader.
  A sectioned loader has to decide what to serve for a section it is skipping; a block
  with an enforced ceiling is exactly that, and the ceiling is what makes the loader's
  budget arithmetic sound rather than hopeful.

## Non-goals

- Judging whether the summary is accurate or complete. See the section above.
- Rewriting existing docs. The ratchet exists so that work can happen page by page.
- Enforcing "one idea per sentence". It is part of the written contract and it is
  review's job — the word ceiling is the countable proxy the gate applies instead.

## See also

- `internal/promptlint/breath` — the checker, and its package doc, which restates the scope
  refusal where a contributor reaching for an entailment check will hit it.
- [Repository index](../INDEX.md) — the public route back to this contract and the
  other existing document gates.
