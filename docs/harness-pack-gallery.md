---
title: "Harness pack gallery"
description: "fak harness gallery helps you choose a harness shape before choosing adapters or editing runtime code."
---
# Harness pack gallery

`fak harness gallery` helps you choose a harness shape **before** choosing adapters or editing runtime code. Each pack translates a recognizable job into the person and problem it serves, the result that should improve, the public extension seam, capability boundaries, a small first build, and its proof.

A gallery pack is a decision scaffold, not a claim that the complete product exists. `init` writes a user-owned manifest and README that turn the decision into a build checklist.

## Choose a pack by the job

| Pack | Choose it when... | What the boundary means | First proof |
|---|---|---|---|
| `readonly-support` | answers must come from approved support material | retrieval and citations are allowed; payments, account writes, and shell access are not | a cited offline answer plus denied `refund_payment` before a model call |
| `coding-workspace` | you want a local coding loop with your own UI or workspace | UI, workspace, tools, approvals, and sessions are adapters; hidden host mutation and unrestricted shell are excluded | open a workspace, stream a turn, preview a diff, and deny an out-of-scope path |
| `cited-research` | analysis must be reproducible from frozen primary sources | source capture, citations, traces, and artifacts are required; uncited synthesis and source mutation are excluded | rerun a frozen-source task and reproduce its answer, citations, and trace |
| `incident-operations` | diagnosis must not silently become production mutation | observation, proposal, approval, execution, and rollback are separate stages | diagnose offline, deny unapproved remediation, approve one fixture action, and retain a receipt |

The **starting point** is architectural guidance: generated config plus policy keeps the stock product and changes its declared task and capability floor; harnesskit adapters replace operator- or environment-facing boundaries through public Go contracts; a skill pack plus trace/artifact sinks centers the workflow and reproducibility record; operations adapters keep authority separated and auditable.

## End to end: choose, inspect, initialize, verify

### 1. List choices in human language

```console
$ fak harness gallery list
Harness starter packs

cited-research - Cited Research Notebook
  Use when: an analyst producing a review from frozen primary sources.
  Outcome: source capture, citation requirements, and not-yet labels are part of the pack contract.
  Start from: skill pack plus trace and artifact sinks.

coding-workspace - Local Coding Workspace
  Use when: a developer replacing a hosted coding-agent shell with a local fak-native loop.
  Outcome: workspace, command, diff, approval, and session boundaries are explicit public adapters.
  Start from: public harnesskit UI plus workspace/tool adapters.

incident-operations - Incident Operations Copilot
  Use when: an on-call operator diagnosing production without granting immediate remediation power.
  Outcome: observation, proposal, approval, execution, and rollback are separate typed stages.
  Start from: policy, approval, telemetry, and secret-provider adapters.

readonly-support - Readonly Support Desk
  Use when: a support team answering from an approved knowledge base.
  Outcome: the pack requires retrieval and citations while structurally excluding write capabilities.
  Start from: generated product config plus policy manifest.

Next: inspect one pack with fak harness gallery show --id <pack>
```

Pick by the job and safety boundary, not by which name sounds most sophisticated.

### 2. Inspect what the choice commits you to

```console
$ fak harness gallery show --id readonly-support
Readonly Support Desk (readonly-support)

For: a support team answering from an approved knowledge base
Problem: draft grounded answers without refunds, account mutation, or arbitrary tools
Today: use a general coding agent and rely on instructions not to mutate customer state
Better because: the pack requires retrieval and citations while structurally excluding write capabilities

Build from: generated product config plus policy manifest
Proof to capture: offline selfcheck emits a cited answer and denies refund_payment before any model call

Needs:
  1. knowledge retrieval
  2. citation projection
  3. policy preflight

Does not get:
  1. payments write
  2. account mutation
  3. shell execution

Ten-minute path:
  1. initialize the generated product
  2. set the support system prompt and task
  3. attach readonly policy
  4. run offline selfcheck and one denied write preflight

Then extend it:
  1. add a real knowledge adapter
  2. brand the browser UI
  3. archive grounded-answer and denial receipts

Next: fak harness gallery init --id readonly-support --dir ./readonly-support-pack
```

Read **Needs** as an implementation checklist and **Does not get** as a testable capability floor. Finish the **Ten-minute path** before **Then extend it**. The proof line names the acceptance artifact; setup without that artifact is not proof.

For automation, use `fak harness gallery list --json` or `fak harness gallery show --id readonly-support --json`.

### 3. Initialize user-owned planning files

```console
$ fak harness gallery init --id readonly-support --dir ./support-pack
Initialized readonly-support in support-pack
Created:
  - harness.pack.json
  - README.md

Next:
  1. Read support-pack/README.md
  2. Edit support-pack/harness.pack.json to fit your tools and boundaries
  3. Check all built-ins with fak harness gallery selfcheck
```

`harness.pack.json` is the machine-readable contract. `README.md` explains the same choice as a build sequence. Replace the sample task and prompt, map each required capability to a real adapter, then implement the ten-minute path through the named seam.

A rerun never overwrites those edits:

```console
$ fak harness gallery init --id readonly-support --dir ./support-pack
Initialized readonly-support in support-pack
Preserved:
  - harness.pack.json
  - README.md

Next:
  1. Read support-pack/README.md
  2. Edit support-pack/harness.pack.json to fit your tools and boundaries
  3. Check all built-ins with fak harness gallery selfcheck
```

### 4. Check the built-in catalog

```console
$ fak harness gallery selfcheck
harness-gallery-selfcheck OK blueprints=4
```

This validates the four built-in blueprints, including capability contradictions and unsafe paths. It does **not** certify a custom implementation. The pack's **Proof to capture** remains the end-to-end done condition.

## What comes after the starter pack?

1. Commit the initialized pack with the product that consumes it; do not leave the decision in session scratch.
2. Build the smallest real path through the named public seam.
3. Capture the declared proof, including the expected denial—not only the successful answer.
4. Add weekend extensions only after that first path is runnable and witnessed.
5. Use [`/harness-creator`](../.claude/skills/harness-creator/SKILL.md) for the generator/rebuild/selfcheck workflow and [`pkg/harnesskit`](../pkg/harnesskit) for custom UI or adapters.

These packs are the decision-oriented first rung of the graduated gallery tracked in [#6808](https://github.com/anthony-chaudhary/fak/issues/6808); they do not substitute for its independently buildable golden products.
