---
title: "Long-context defaults doctrine"
description: "How fak treats advertised context windows, effective context, minimum viable context, resident budgets, output reserve, and provenance labels when choosing context defaults."
---

# Long-context defaults doctrine

The model's advertised context window is a hard cap, not a target. fak defaults should
optimize for the smallest resident view that still satisfies the task, preserve explicit
output reserve, and label every default by the strength of its evidence.

This page is the canonical policy layer for `--compact-history-budget`,
`--ctx-view-budget`, and future automatic context-budget defaults. It sits above the
mechanism docs: the token-saving scorecard proves which levers are on, while this page says
how large those levers should be allowed to make the model-visible working set.

## Terms

| Term | Definition | Default consequence |
|---|---|---|
| **HardContextCap** | The provider/model limit on the request plus generated output. Everything sent to the model counts against it: system prompt, tools, messages, tool results, documents, images, thinking where the provider counts it, and the response budget. | Never treat the cap as a resident-context goal. A 1M-token model still gets a bounded resident view unless the task needs global coverage. |
| **MECW** | Measured Effective Context Window: the largest resident context for a specific task class, model, and prompt shape that still meets the quality SLO under a local fak witness. | `min(HardContextCap, MECW)` is the ceiling for routine defaults. Without a local witness, MECW is unknown and defaults use fallback priors. |
| **MVC** | Minimum Viable Context: the smallest set of resident evidence that still meets the task SLO. It includes the objective, live instructions, recent turns, pins, and enough supporting context for each gold fact to be usable. | Do not compress below MVC to win a token metric. A smaller window can be worse when it isolates tiny evidence from the supporting chunk that makes it interpretable. |
| **Target resident budget** | The number of tokens fak tries to materialize for the current turn after reserving output and respecting MECW/MVC. This is a target, not an entitlement. | `--ctx-view-budget` and future auto budgets should express this resident target, not the model's cap. |
| **Output reserve** | Tokens deliberately held back for the model's answer, tool-call arguments, thinking blocks, or provider `max_tokens` requirement. | Input budgeting must subtract reserve before filling context. A request that uses the whole cap as input has already failed the default. |
| **Provenance label** | The evidence class behind a budget or range: `WITNESSED`, `OBSERVED`, `MODELED`, or `FALLBACK`. | Every published default or recommendation should carry one. Upgrade the label only when the witness gets stronger. |

## Default Rules

1. **Cap is not target.** HardContextCap is only the largest legal envelope. The resident
   budget is chosen from MVC, MECW, output reserve, task class, and local witness quality.

2. **Effective context is task and model dependent.** A model can accept a long request and
   still lose accuracy as length grows, as shown by position sensitivity, multi-needle and
   non-literal retrieval failures, and task-specific benchmark divergence.

3. **Minimum context matters too.** Compression can fail in both directions: too much
   unrelated context can bury a fact, but too little supporting context can make a short
   gold span hard to retrieve or reason over. MVC protects the lower bound.

4. **Reserve output first.** Before a resident view is planned, subtract the output reserve
   from the hard cap and from any measured effective envelope. Long inputs that leave no
   answer room are invalid defaults even if the provider accepts them.

5. **Prefer witnessed defaults.** A local fak bench or production trace that measures the
   exact task/model route wins over a paper, vendor doc, or general heuristic. Until then,
   use the fallback priors below and label them `FALLBACK/MODELED`.

## Fallback Priors

These are starting priors for fak defaults when no same-task fak witness exists. They are
not universal quality claims.

| Task class | Fallback resident prior | Provenance | Notes |
|---|---:|---|---|
| Routine agent turn: coding, review, shell investigation, small issue fix | 8K-32K | `FALLBACK/MODELED` | Enough for objective, recent transcript, selected files, and compact witness notes. Prefer `--ctx-view-budget` near the low end until a local trace proves broader context helps. |
| Long document or repository question with known relevant regions | 32K-128K | `FALLBACK/MODELED` | Use when the task needs several chunks, examples, or cross-file support but not whole-corpus coverage. Retrieval/planning should still select a resident view rather than dumping everything. |
| Global-coverage task: exhaustive audit, whole-repo/map synthesis, many-shot in-context learning, multimodal corpus review | >128K only with a witness or explicit global-coverage need | `FALLBACK/MODELED` until witnessed | Acceptable when the task definition itself requires broad coverage, or a same-task witness proves quality stays within SLO at that length. |

The current shipped values are conservative resident defaults, not a claim that they are
optimal forever: `--compact-history-budget` sheds old compactible turns after roughly 48K
tokens, and `--ctx-view-budget` defaults the planned resident view to 8K. Those values
should move only with a stronger witness or a clearly labeled re-budgeting decision.

Two things about that 48K a reader will otherwise get wrong (#5430):

- **48K is the flag's default, not what a `fak manage` launch runs at.** Every `fak manage`
  launch that does not pass `--compact-history-budget` explicitly is resolved to **96K**
  (`gateway.HeadlessCompactHistoryBudget`, via `resolveGuardCompactBudget`), because every
  guard launch fronts Claude Code and therefore carries its large fixed system+tools floor —
  a floor that already sits at or above the lean 48K line. The flag's printed default is the
  pre-resolution value; the number actually in force is on the running gateway at
  `/debug/vars` under `adjudication.compaction_budget`, and `fak info`'s cache tab prints it
  next to the compaction shed bar. An explicit `--compact-history-budget` always wins,
  including an explicit `0` (off, body forwarded byte-for-byte).
- **The budget is measured against `messages[]` alone.** The cut sums estimated tokens over
  message elements only (`agent.CompactAnthropicHistoryWithOptions`); the system+tools block
  is never counted against it. A budget sized from "my whole context is N tokens" is
  therefore set too HIGH, in the safe-looking direction, and the only symptom is an
  `under_budget` bail that sheds nothing — for as long as the session runs.

## When More Than 128K Is Acceptable

Use more than 128K resident context only when at least one condition is true:

- The task is a global-coverage task where omission is more dangerous than length-induced
  quality loss.
- A same-task fak witness shows the chosen model meets the quality SLO at the proposed
  resident size.
- The operator explicitly chooses an exploratory long-context run and the output is labeled
  as exploratory, not a default recommendation.

Otherwise, use retrieval, pins, ctxplan views, or staged passes to keep the resident view
inside the measured or fallback effective envelope.

## Provenance Labels

- `WITNESSED`: a local fak test, benchmark, trace, or production run demonstrates the budget
  on the same task class and model route.
- `OBSERVED`: provider telemetry or external benchmark evidence supports the direction, but
  not the exact fak route.
- `MODELED`: arithmetic, code inspection, or benchmark-derived modeling predicts the value;
  it has not been directly run on the target route.
- `FALLBACK`: a deliberately conservative prior used because no stronger evidence exists.

Never headline a fallback as if it were measured. The correct phrasing is "fallback prior
until a local witness exists."

## Source Anchors

- [Claude context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows): provider context includes input and output; larger windows still require curation.
- [Gemini long context](https://ai.google.dev/gemini-api/docs/long-context): long context enables new workflows but has task-dependent limitations and cost tradeoffs.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172): relevant information is often used less reliably when placed in the middle of long context.
- [RULER](https://arxiv.org/abs/2404.06654): vanilla needle retrieval is too shallow; harder task categories degrade with length.
- [HELMET](https://arxiv.org/abs/2410.02694): long-context evaluation needs diverse task categories rather than one synthetic probe.
- [NoLiMa](https://arxiv.org/abs/2502.05167): non-literal retrieval can degrade sharply as context grows.
- [Context Length Alone Hurts](https://arxiv.org/abs/2510.05381): length can reduce performance even when relevant evidence is retrievable.
- [Hidden in the Haystack](https://arxiv.org/abs/2505.18148): very small gold evidence can be harder to use, which is why MVC must include supporting context.
