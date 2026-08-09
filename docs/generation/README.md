---
title: "Implementation documentation by generation"
description: "How to tell whether a fak implementation document describes current work, a later product generation, research, or historical evidence."
---

# Implementation documentation by generation

**Primary audience:** contributors deciding whether an implementation document describes current work, a later product generation, research, or historical evidence.
**Lifecycle:** current
**Generation:** `gen/now` documentation route; it classifies material from every generation stream.
**Authority:** the stream definitions and promotion rules in the [Generation Contract](../generation.md).
**Support:** use the [contributor route](../../CONTRIBUTING.md) for development help and the [operator guides](../operator/) for deployed operation.

**Next action:** before using an implementation document as authority, find its lifecycle and generation statement, then follow the matching row below.

## Choose the material you need

| Need | Lifecycle and generation | How to use it | Where to go |
|---|---|---|---|
| Behavior implemented and maintained today | **Current**, normally `gen/now` | Treat it as implementation guidance within its stated release, backend, and mode. Confirm operational claims with the page's named proof. | Start with the [contributor route](../../CONTRIBUTING.md), then follow its subsystem or task link. |
| A planned successor or near-term foundation | **Current planning**, `gen/next` | Use it to understand work intended for near-term dogfood. Do not infer that an unshipped interface or default already exists. | Read the proposal's gate and promotion evidence, then check the linked issue or implementation witness. |
| A longer-horizon architecture option | **Research**, usually `gen/second-next` or `gen/future` | Use it as a hypothesis, simulation, compatibility study, or option contract. Its claims become implementation authority only after promotion evidence lands. | Read the option's assumptions, witness plan, and promotion trigger; the [Generation Contract](../generation.md#streams) defines both streams. |
| Why an earlier design existed or what it proved | **Archived** | Use it as dated evidence. Follow any replacement link before acting; archived text is not current guidance. | Browse the [archive](../archive/) or the [notes and research index](../../INDEX.md#notes--research-docsnotes). |

`gen/now`, `gen/next`, `gen/second-next`, and `gen/future` describe the work horizon. Lifecycle describes how a document may be used: current guidance, current planning, research, or archive. Priority is separate from both; a future study can be urgent, while a current cleanup can be small.

## Mode and authority

This page is a **navigation and classification mode**, not a product-runtime mode. Runtime behavior comes from the current implementation contract and its proof. Planning documents describe intended work; research documents preserve testable options; archived documents preserve history.

When a page covers several backends, releases, or operating modes, use only the section that names your mode. If it does not identify a mode or generation, verify the behavior in current code or a maintained guide before relying on it, and report the ambiguity through the repository's issue route in [CONTRIBUTING.md](../../CONTRIBUTING.md#reporting-issues).

## Lifecycle

Generation changes only when evidence changes the work horizon. The [Generation Contract promotion verbs](../generation.md#promotion-verbs) are authoritative:

1. **Promote** material closer to `gen/now` when its named blocker is retired by a witness such as a runnable spine, dogfood result, compatibility proof, or shipped implementation.
2. **Demote** material farther from `gen/now` when an assumption fails, a witness regresses, or the current product path no longer needs it.
3. **Retire** material to the archive when it is completed, superseded, rejected with evidence, or no longer has a current owner and witness path.

A label change alone is not evidence. Record the reason and link the witness or superseding authority. Promotion can turn research into current planning and then current implementation guidance; demotion can reverse that path. Retirement preserves the useful record while removing it from current authority.

## Scope boundary

This map classifies implementation documentation. It does not replace release-specific instructions, subsystem contracts, issue labels, or runtime feature gates. For the complete machine-oriented documentation map, use [`llms.txt`](../../llms.txt); for release status, use the [release index](../releases/).
