---
name: tail-wag
description: Find the most "tail wagging the dog" part of the system — a peripheral or secondary concern that is disproportionately driving core design decisions. Produces a ranked list of inverted-priority findings with evidence and a proposed rebalance. Use when the user asks to find the "tail wagging the dog", inverted priorities, misplaced drivers, mis-layered concerns, or when something small is clearly dictating something big.
---

# /tail-wag — Find the Tail Wagging the Dog

Identify where a **peripheral, secondary, or lower-layer concern** is disproportionately driving **core, primary, or higher-layer** design decisions. The goal is not to list problems — it is to find the *most inverted* one, where the causal arrow points the wrong way.

## What counts as "tail wagging the dog"

A finding qualifies when **all three** hold:

1. **Layer inversion** — something that belongs *below* the abstraction line is visibly shaping something *above* it. (e.g. a specific vendor's error code schema shaping the router topology; a DB quirk shaping the domain model; a single consumer's need shaping a shared library's public API.)
2. **Blast-radius asymmetry** — the "tail" affects a narrow slice of cases, but the accommodation for it spreads widely (many call sites, many config knobs, many branches, many drivers).
3. **Counterfactual cheapness** — if the tail went away tomorrow, a large amount of the code/config/abstraction could be deleted with no loss.

If a finding has only 1–2 of these, note it but rank it lower. The winner has all three *loudly*.

## Canonical shapes (not project-specific)

Common inversion archetypes — when an analogous shape shows up, name it:

- **Provider-as-architecture.** A specific LLM/cloud/payment vendor's rate limits, quotas, or error codes shape default-backend selection, fallback topology, cooldown schema, and observability names. The fix pushes the vendor knowledge *below* dispatch into an adapter; everything above becomes vendor-agnostic. Classic layer inversion: infrastructure was driving architecture.
- **Retry-as-config.** 5 retry loops + 15+ config keys accrete because each caller reinvented "how many tries, how long to wait". One typed budget object collapses the surface. Classic blast-radius asymmetry: retry is a tiny concept, but it had colonized the config file.
- **State-machine-shaped-by-failure-modes.** A graph state machine accommodates ephemeral signals (rejected, withdrawn, stale) as if they were states. Linear forward-only state machine restored once the tail is identified — failures live in observability, not in the state model.

Reference these in your output when analogous — calling out "this is another provider-as-architecture inversion" lands faster than re-explaining the concept.

## Instructions

When the user runs `/tail-wag [scope]`:

### 1. Fix the scope

If `scope` is given (e.g. `apply pipeline`, `auth`, `config surface`, a file path), confine the analysis there. If not, ask **one** clarifying question: *"Whole repo, or a specific subsystem?"*. Do not produce a whole-repo audit unless the user explicitly asks.

### 2. Gather evidence (read, don't guess)

Parallel-read the entry points for the scope. Useful generic starting points:

- **Dispatch / routing surface** → main router file(s), config that lists adapters or backends.
- **Storage layer** → schema files, migration history, the ORM/query layer.
- **Config surface** → `config/*.yaml`, env var dictionaries, `defaults.py`-style modules.
- **Observability** → log schema, event taxonomy, dashboard definitions.
- **UI** → the templates/views and the data shapes they require.

Skim for these **inversion smells** (each is a hypothesis to verify, not a conclusion):

| Smell | What it looks like |
|---|---|
| Vendor names in high-level code | Provider / ATS / browser-engine names above the abstraction they should be hidden behind |
| Config-key proliferation for one concept | 3+ keys that are all really describing the same thing (timeouts, retries, budgets) |
| One-off branches in hot paths | `if vendor == "X":` or `if customer == "Y":` inside generic dispatchers |
| Historical rationale still loadbearing | Comments / docs justifying a choice by an incident that was later fixed elsewhere |
| Compatibility shims outliving their feature | `_legacy_`, `_v1_`, `# TODO remove after…` with the "after" long past |
| Defaults chosen by outlier | A global default picked to dodge one specific pathology, paid by everyone |
| Wrappers that only exist to translate errors | Thick adapters where the only real content is mapping one error taxonomy to another |
| Schema driven by the reader, not the data | A storage/event schema whose fields are named after one consumer's UI |

### 3. Rank the findings

Produce at most **5 candidates**, ranked. Score each roughly on:

- **Inversion severity** (1–5): how clearly is the causal arrow pointing the wrong way?
- **Blast radius** (1–5): how much code/config/surface area does the accommodation touch?
- **Counterfactual cheapness** (1–5): if the tail vanished, how much could be deleted?

The **winner** is the finding with the highest product (or the highest in all three if there's a tie). State the scores explicitly — no hand-waving.

### 4. Write the output

Format:

```
## Tail-wagging-the-dog: <scope>

### Winner: <short name>
- The tail: <the peripheral concern, one sentence>
- The dog: <the core decision it is shaping, one sentence>
- Evidence: <3–5 file:line pointers or config-key names — concrete, not paraphrased>
- Scores: inversion=<n>/5, blast=<n>/5, counterfactual=<n>/5
- Analogy: <optional — "same shape as provider-as-architecture" etc.>
- Rebalance: <one paragraph on what should drive what, and the smallest move that would start the correction>

### Runners-up
1. <name> — <one-line why it's inverted, one-line why it ranked lower>
2. <name> — …
```

Keep the winner section ~150 words. Runners-up one line each. No preamble, no conclusion paragraph.

### 5. Do not propose a phased plan

This skill **diagnoses**, it does not plan. If the user wants a phased plan to execute the rebalance, they will ask. Ending with "want me to draft a plan?" is fine; writing the plan unsolicited is not.

## Anti-patterns (what this skill must not do)

- **Don't list generic tech-debt.** "This file is long" or "tests are flaky" are not tail-wag findings. The finding must name a *driver* and a *thing driven*, with the arrow inverted.
- **Don't confuse "annoying" with "inverted".** A legitimately hard problem handled at the right layer is not a tail-wag, even if the code is gnarly.
- **Don't invent evidence.** Every finding cites real file paths, config keys, or call sites. If you can't cite, you haven't verified — drop the finding.
- **Don't hide behind hedges.** Pick a winner. "They're all kind of bad" is not an output.
