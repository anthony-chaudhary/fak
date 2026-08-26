---
title: "Workflow concepts: the operator's middle layer"
description: "fak traj concepts derives workflow concepts between fleet aggregates and individual tool calls."
---
# Workflow concepts: the operator's middle layer

`fak traj concepts` derives **workflow concepts** between fleet aggregates and
individual tool calls. A concept groups sessions with similar tool sets and
adjacent transitions, then exposes:

- a stable `wf-…` selector;
- prevalence (`sessions`, `share`) and call error rate;
- an explainable medoid `signature` and ranked tool label; and
- exemplar session IDs for immediate drill-down.

```console
fak traj concepts --corpus runs.jsonl
fak traj concepts --corpus runs.jsonl --concept wf-12ab34cd
fak traj concepts --corpus runs.jsonl --json --threshold .65
```

The default algorithm is intentionally deterministic and inspectable: weighted
Jaccard over tool-name and adjacent-transition features, followed by connected
components. It is not semantic task clustering. Operators should use it to find
a recurring workflow family, inspect its exemplars, and only then steer policy,
routing, prompts, or tooling. The `--threshold` knob controls whether
nearby workflows merge; raising it produces narrower concepts.

This creates the navigation path **aggregate → workflow concept → exemplar
session → exact call** without requiring an embedding service or turning an
opaque cluster into an automatic policy decision.

