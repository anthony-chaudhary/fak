---
title: "Shared workspace, positive state, and the negation operator"
description: "fak treats an agent's bounded context as a shared workspace to manage, not a chat log to grow: shed superseded history, construct the positive state the agent should occupy, and rewrite kernel-authored negations with the deterministic negframe operator — while the default-deny capability floor still adjudicates every effect below language."
slug: shared-workspace-negation-operator
keywords:
  - shared workspace
  - global workspace
  - positive-state construction
  - negation operator
  - negframe
  - context shedding
  - default-deny capability floor
---

# Shared workspace, positive state, and the negation operator

An AI agent does not reason over an unlimited transcript. It works inside a bounded context shared by the current task, tool results, policy guidance, and recent conversation. **fak treats that context as a shared workspace to manage**, not as a chat log to grow forever. The analogy is to global-workspace theories: information that remains available can influence many later operations. It is an engineering lens, not a claim that a language model has human consciousness or that fak reproduces a cognitive architecture.

## 1. A bounded shared workspace

Every old turn that remains in context competes with the originating query and current evidence. fak therefore keeps stable setup reusable and sheds superseded history before it consumes the horizon. Its context MMU preserves useful state while compaction carries forward the positive residue: the task, current facts, and next valid action rather than a replay of every earlier failure. See [context shedding](context-shedding.md), [context signal-to-noise](context-signal-to-noise.md), and the [query-not-chat doctrine](../query-not-chat.md).

This framing matters for security too. A stale instruction or poisoned result that remains globally available can affect many later decisions. fak's capability floor still adjudicates every tool call, but a cleaner workspace reduces the model-side work needed to recover the intended task. The structural boundary remains the one described in [default deny versus classifier](default-deny-vs-classifier.md).

## 2. Construct the positive state

A prohibition names the state to avoid. Repeating it can keep that state salient even when the useful operational content is the permitted substitute. fak's positive-state rule is therefore: **write the state the agent should occupy**. A recovery message leads with the allowed command or next step, preserves the originating query, and trails any required refusal token. Compaction similarly returns useful residual state rather than negated history.

This is not permission to hide a denial or soften policy. Machine-stable reason tokens and default-deny decisions remain byte-stable where contracts require them. Positive-state construction changes kernel-authored explanatory prose around the decision; it does not change the decision. The design doctrine is recorded in [positive-state construction](../positive-state-construction.md).

## 3. The negation operator

`internal/negframe` is fak's small, deterministic negation operator for kernel-authored text. At an emission boundary it can replace a negative frame with an affordance-first equivalent: what works, what state survives, and what the agent can do next. It preserves required tokens, fails safe when the polarity is unclear, and is idempotent. User content is outside this rewrite boundary; untrusted text is not silently reinterpreted.

The operator and context shedding solve different parts of one problem:

1. **Read the workspace:** retain the task and evidence that still matter.
2. **Write the substitute:** express the allowed positive state instead of rebroadcasting the rejected one.
3. **Enforce below language:** let the tool-call capability floor decide effects regardless of prose.

The result is a bounded context with less avoidable inversion work for the model, while security remains structural. The model-backed direction experiment in [`experiments/verbalizable-direction-qwen25-0.5b`](../../experiments/verbalizable-direction-qwen25-0.5b/README.md) observed that a steered residual direction broadcasts into later layers; it did **not** observe a decoded-token flip. That result motivates measuring shared-state effects without overstating them.

## What fak claims

- **Shipped mechanism:** context reuse and shedding, positive-residue compaction, deterministic negframe transforms, and a default-deny tool-call gate.
- **Observed model result:** a named residual direction can be read, steered, and measured at downstream layers on one kernel-served Qwen architecture.
- **Not claimed:** consciousness, a complete computational global-workspace implementation, or universal accuracy gains from positive framing. Those require broader experiments.

The practical rule is simple: keep the query, construct the permitted state, shed superseded history, and enforce effects at the kernel seam.
