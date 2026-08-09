---
title: "Kimi K3 for Claude Code, through fak"
description: "The standalone route page for fak's first-class Moonshot Kimi K3 support, and how to serve and view it locally from the repository root."
---

# Kimi K3 for Claude Code, through fak

This directory contains the standalone marketing page for fak's first-class Moonshot Kimi K3 route.

## View it

From the repository root, serve static files and open the page:

```bash
python3 -m http.server 8080
# http://localhost:8080/docs/kimi-k3/
```

The page is deliberately dependency-free: one HTML file, one SVG social card, and existing fak brand assets. The operator authority remains [`DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md#moonshot-kimi-k3); the implementation witness is [issue #5147](https://github.com/anthony-chaudhary/fak/issues/5147).

## Claims boundary

The page describes only behavior covered by the committed Kimi K3 request/streaming tests: model recognition, `reasoning_effort: "max"`, incompatible sampling-field removal, and reasoning preservation. It makes no unmeasured speed, cost, or quality claim.
