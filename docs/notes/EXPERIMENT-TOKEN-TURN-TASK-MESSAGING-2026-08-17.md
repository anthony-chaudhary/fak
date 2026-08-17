# Messaging experiment: from token cost to agent work

**Date:** 2026-08-17
**Status:** ACTIVE EXPERIMENT — candidate language, not approved homepage copy
**Owner surface:** positioning and benchmark vocabulary

## Decision boundary

Optimization is part of FAK's value, not the whole product.

FAK is an agent kernel that manages the whole trajectory. Its seam covers context and model routing, tool calls and capability policy, session control, operational evidence, and witnessed outcomes. Lower token or turn cost is one result the kernel can produce. Copy that leaves readers seeing only a cache, inference optimizer, or cost dashboard fails this experiment even when it is catchy.

This experiment asks whether a short economic ladder can make one part of the value legible without shrinking the product:

```text
$/GPU-hour -> $/token -> $/turn -> $/task
```

The working distinction is:

> The model's unit is the token. The agent's unit is the turn. The user's unit is the task. FAK manages the path between them.

## Smallest working spine

Use a two-line message, not a lone optimization slogan:

> **From token cost to task cost.**  
> **FAK manages the agent path in between.**

The first line earns attention. The second prevents category collapse. Any candidate used publicly must survive as a pair or leave enough room nearby to name the broader kernel.

## Candidate set

### A. Economic ladders

1. **From $/GPU to $/token to $/turn to $/task.**
2. **From token cost to task cost.**
3. **The model bills tokens. The agent spends turns. The user buys tasks.**
4. **GPUs run models. Tokens fill turns. Turns finish tasks.**
5. **Providers price tokens. FAK connects them to turns and tasks.**

These are explanatory devices. They fit a benchmark page, economics section, talk, or diagram better than a permanent product category.

### B. FAK-specific bridge lines

6. **FAK is the kernel between token spend and completed work.**
7. **FAK manages what happens between a prompt and a verified result.**
8. **FAK turns model activity into controlled agent work.**
9. **FAK manages the path from model call to tool call to task outcome.**
10. **One kernel for context, models, tools, and outcomes.**

These describe more of the product. They are less numerically sharp but resist reducing FAK to optimization.

### C. Optimization claims for supporting sections

11. **Lower effective cost per turn. Fewer turns per task.**
12. **Make each model turn carry more of the task.**
13. **Spend model calls where they move the task.**
14. **Reuse the setup. Route the call. Control the tool.**
15. **Turn token efficiency into task efficiency.**

These belong under performance, benchmark, or observability headings. They should not stand alone as the definition of FAK.

### D. Outcome lines that require a witness nearby

16. **Know what each verified task cost.**
17. **Measure the task, not only the tokens.**
18. **From generated tokens to accepted work.**
19. **See what the agent spent to get the task through the gate.**
20. **Cost per attempt. Cost per pass. No hidden failures.**

These are strongest when the report actually joins spend, attempts, and independent acceptance. Until that report exists, treat them as design targets rather than shipped capability claims.

## First-pass rubric

Score each candidate from 1 to 5 on:

- **Catch:** memorable after one reading.
- **Truth:** defensible from current FAK behavior and reports.
- **Breadth:** leaves room for the rest of the kernel: context, routing, tools, policy, operations, and evidence.
- **Difference:** sounds unlike a generic inference optimizer or observability dashboard.
- **Economic clarity:** makes the GPU/token/turn/task progression easier to understand.
- **Reduction safety:** 5 means low risk of making optimization sound like the entire product.

Truth and reduction safety are gates: a candidate scoring below 4 on either cannot become top-level copy.

## Desk experiment results

This is an internal semantic/readback test, not audience evidence. Scores test the wording against the current product contract and the failure mode named above.

| Candidate | Catch | Truth | Breadth | Difference | Econ | Reduction safety | Total / 30 | Placement verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| From $/GPU to $/token to $/turn to $/task. | 5 | 4 | 2 | 4 | 5 | 2 | 22 | Diagram only; pair with kernel line |
| From token cost to task cost. | 5 | 4 | 2 | 3 | 5 | 2 | 21 | Section hook; never alone |
| The model bills tokens. The agent spends turns. The user buys tasks. | 5 | 5 | 3 | 5 | 5 | 3 | 26 | Strong explainer lead |
| Providers price tokens. FAK connects them to turns and tasks. | 4 | 4 | 3 | 4 | 5 | 3 | 23 | Economics-page lead |
| FAK is the kernel between token spend and completed work. | 4 | 5 | 4 | 5 | 4 | 4 | 26 | Strong paired headline |
| FAK manages what happens between a prompt and a verified result. | 4 | 4 | 5 | 4 | 3 | 5 | 25 | Broad product explainer |
| FAK manages the path from model call to tool call to task outcome. | 3 | 5 | 5 | 5 | 4 | 5 | 27 | Best precise descriptor |
| One kernel for context, models, tools, and outcomes. | 4 | 5 | 5 | 4 | 2 | 5 | 25 | Product-level support line |
| Lower effective cost per turn. Fewer turns per task. | 5 | 4 | 2 | 5 | 5 | 1 | 22 | Performance subsection only |
| Reuse the setup. Route the call. Control the tool. | 5 | 5 | 5 | 5 | 2 | 5 | 27 | Best whole-kernel rhythm |
| Turn token efficiency into task efficiency. | 5 | 4 | 2 | 4 | 5 | 1 | 21 | Benchmark campaign only |
| Cost per attempt. Cost per pass. No hidden failures. | 5 | 3 | 2 | 5 | 5 | 3 | 23 | Hold for joined task report |

## Current winners

### Best economic explainer

> **The model bills tokens. The agent spends turns. The user buys tasks.**

Why it works: it assigns each unit to the layer that owns it and does not claim that any one unit defines FAK.

Required follow-up line:

> **FAK manages the path from model call to tool call to task outcome.**

### Best compact whole-product line

> **Reuse the setup. Route the call. Control the tool.**

Why it works: optimization appears as one verb beside routing and control. It reflects the current performance center and the security floor without claiming that either is the entire product.

Optional outcome tail:

> **Then show what reached the task gate.**

This tail remains experimental until the joined task-economics report exists.

### Best economics-section pair

> **From token cost to task cost.**  
> **FAK manages the agent path in between.**

Why it works: the first line is catchier; the second carries the product boundary.

## Rejected as top-level definitions

- **FAK is the turn-economics layer.** Too narrow. It mistakes one reporting/optimization seam for the product.
- **More done per dollar.** Broadly appealing but generic, and “done” needs an independent witness.
- **The unit after tokens is done.** Memorable, but skips turns and makes a heterogeneous outcome sound universal.
- **Lower cost per turn.** A useful optimization claim that says nothing about tool control, safety, operations, or task success.
- **FAK prices turns and tasks.** Overclaims the current reporting join. FAK accounts for tokens and avoided model calls, but does not yet expose one canonical cost-per-verified-task row.

## External experiment design

Do not edit the README headline from this desk score. Run a small blinded test first.

### Audiences

Recruit at least three people in each group:

1. agent/application developer;
2. inference or platform operator;
3. engineering/product buyer;
4. security or governance owner.

### Variants

Show each participant one variant, then rotate order across participants:

- **V1 economic:** “From token cost to task cost. FAK manages the agent path in between.”
- **V2 layered:** “The model bills tokens. The agent spends turns. The user buys tasks. FAK manages the path.”
- **V3 kernel:** “Reuse the setup. Route the call. Control the tool. One kernel for context, models, tools, and outcomes.”
- **Control:** current positioning: “An agent kernel in one static Go binary.”

### Readback questions

Ask without showing the copy again:

1. What do you think FAK does?
2. Is it primarily a cost optimizer, an agent runtime/kernel, a security product, or something else?
3. What would you expect it to measure?
4. What would you try first?
5. Repeat any phrase you remember.

### Promotion gates

Promote a candidate only if every gate passes:

- Kernel readback: at least 70% identify FAK as an agent kernel/runtime; the optimization-only answer stays below 30%.
- Recall: at least 60% can repeat the central phrase unaided.
- Claim safety: no more than 20% infer that FAK itself guarantees task success.
- Intent: the candidate beats the control on “I understand why I would try this” while retaining the kernel category.
- Control-plane breadth: security/governance participants still name tool control or policy after reading the adjacent support line.

Capture raw responses, variant order, audience, and date. Do not convert an internal score into a customer result.

## Reporting implications

The copy points toward a useful report hierarchy without making it the whole product:

```text
optimization diagnostics: $/token, tokens/turn, effective $/turn
agent execution:          turns/task attempt, elapsed/task
outcome economics:        $/verified task, attempts/pass
kernel operations:        reuse, routing, tool verdicts, interventions, evidence
```

FAK already exposes token/reuse and avoided-model-call accounting through `fak info` and `fak info --work-done-json`. The phrase “task cost” should graduate from experiment to product claim only when a report joins task identity, all-in spend, attempts, and independent acceptance. Until then, the economic ladder is explanatory language and a roadmap for measurement.

## Working recommendation

Keep this in the message bank now:

> **The model bills tokens. The agent spends turns. The user buys tasks.**  
> **FAK manages the path from model call to tool call to task outcome.**

Keep this for performance material:

> **Lower effective cost per turn. Fewer turns per task.**

Keep this as the whole-product rhythm:

> **Reuse the setup. Route the call. Control the tool.**

The three lines are complementary. The economic line attracts attention, the path line defines FAK, and the rhythm names multiple kinds of value so optimization remains one part of the product.
