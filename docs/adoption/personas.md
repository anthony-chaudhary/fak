---
title: "Who is fak for? A persona gallery with a quote-ready pitch each"
description: "The kinds of person (and one machine) who land on fak — the solo dev, the app developer, the backend integrator, the SRE, the security engineer, the researcher, the benchmark engineer, the decision-maker, the contributor, the coding agent — each with who they are, a one-sentence quote-ready pitch, and the first door to walk through."
slug: who-is-fak-for
keywords:
  - who is fak for
  - fak personas
  - agent kernel for developers
  - agent kernel for security
  - agent kernel for SRE
  - fak pitch by audience
  - self-select the value prop
date: 2026-07-03
---

# Who is fak for? A persona gallery

> **TL;DR:** people adopt a tool when they see themselves in the pitch. This page
> lists the kinds of person who actually land on fak and gives each one a single
> quote-ready sentence for why they specifically care, plus the one door to walk
> through first. Find your row, quote the line, follow the link.

This is dimension **E — Social proof & community** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
The roster below is the same one the
[persona-readiness scorecard](https://github.com/anthony-chaudhary/fak/blob/main/docs/persona-scorecard/README.md) grades, so the
personas here are the ones fak commits to serving, not an invented list. Each
pitch is a per-audience cut of the canonical
[pitch ladder](pitch-ladder.md); if a claim here and a claim there ever disagree,
the pitch ladder wins.

Every number is witnessed and traces to
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) and
[CLAIMS.md](../../CLAIMS.md). These are pitches written *for* each persona, not
quotes *from* one: nothing here claims a real user, a market share, an unrun
benchmark, or a novelty the 0/29 prior-art audit refutes.

## Consume and build on it

**Free-tier solo dev.** Downloads a prebuilt binary and runs it. Will not clone,
build from source, or read the docs; spends seconds, not minutes.

> Download one static binary and watch it refuse a dangerous tool call in under a
> minute — no key, no GPU, no toolchain.

Start here: [the README](../../README.md).

**App developer / vibe-coder.** Building an app or agent and wants to point an
existing harness (Claude Code, Cursor, an MCP client) at fak with no hand-wiring.

> Point the agent you already run at fak with one base-URL change, and every tool
> call it makes gets checked against a default-deny floor before it runs.

Start here: [the integration index](../integrations/README.md).

**Backend integrator.** Embeds the binary in front of a model server inside a
real service and needs a stable interface plus a way to extend it.

> Install fak in front of your model server and build on a frozen interface,
> adding a leaf instead of forking the kernel.

Start here: [the getting-started guide](../../GETTING-STARTED.md).

## Operate and trust it

**Infra / platform engineer (SRE).** Runs `fak serve` in production and wants a
container image, metrics, health checks, rate limits, and a deployment story.

> Run the same static binary a developer runs, now with `/metrics`, `/healthz`,
> rate limits, and a deploy guide for Docker, Compose, and k8s.

Start here: [the deployment guide](../fak/deployment-guide.md).

**Security engineer.** Evaluates the capability floor and the containment claims
before trusting fak in front of an agent, and reads the threat model.

> Refusing an irreversible action never depends on catching the attack: the
> capability was never granted, and every deny cites a code from a closed
> vocabulary into a tamper-evident journal.

Start here: [the threat model in SECURITY.md](../../SECURITY.md).

## Study and decide

**ML researcher.** Wants to reproduce fak's determinism and benchmark results
bit-for-bit, study the kernel from the notes, and cite the work.

> Reproduce the bit-identical determinism witness offline and cite the committed
> benchmark data, not a screenshot.

Start here: [the reproduce-offline packet](../repro-packet.md).

**Benchmark / eval engineer.** Runs fak's benchmarks and compares the numbers
against the field (vLLM, SGLang, TensorRT-LLM, llama.cpp).

> Run the fan-out benchmark yourself and read fak's honest position against the
> field: the tuned ~4.1× less work on a 50-turn × 5-agent session, never the naive
> multiplier.

Start here: [the bench plan](../bench-plan.md).

**Evaluator / decision-maker.** An engineering manager or tech lead deciding in
about ten minutes whether fak is real, honest, and worth a deeper look.

> See in ten minutes what is shipped, what is still simulated, and how fak
> compares — every headline number traced to a commit, no overclaim.

Start here: [the README](../../README.md).

## Contribute and extend

**Open-source contributor.** Wants to add a feature and ship it green without
tripping an enforced guard, with the rules stated up front.

> Add your feature as a leaf and ship it green, with the enforced rules stated up
> front so the guard teaches instead of ambushes.

Start here: [the contributor contract](../../CONTRIBUTING.md).

**AI coding agent.** An autonomous coding agent (Claude Code, Codex, Cursor, an
MCP client) that lands in the repo cold and must discover, adopt, and build on
fak with no human in the loop.

> Land in the repo cold, read `AGENTS.md` and `llms.txt`, and adopt fak without a
> person handing you the steps.

Start here: [the agent entry point AGENTS.md](../../AGENTS.md).

## The shared pitch under all of them

Every row above is the same core claim aimed at a different reader: fak treats
every agent tool call like a syscall, so the model proposes and the kernel
disposes. One static Go binary makes the same agent loop safer (a default-deny
capability floor plus result quarantine), cheaper (a witnessed ~4.1× less work
than a tuned warm-cache stack, with `max|Δ| = 0` KV eviction), and faster (a
~362 ns in-process decision). None of the primitives are new; the assembly into
one drop-in binary is the point.

## Related framing artifacts

- [The pitch ladder](pitch-ladder.md) — the canonical pitch at three zoom levels,
  the source these per-persona lines are cut from.
- [The persona-readiness scorecard](https://github.com/anthony-chaudhary/fak/blob/main/docs/persona-scorecard/README.md) — the inward
  measure of whether each persona above can actually do their first job today.
- [Objections and one-line answers](objections.md) — what to say after a pitch
  lands and the pushback starts.
