---
title: "Run fak guard as a server-side client (unattended / container)"
description: "Operator route for running `fak guard -- <harness>` server-side — a hosted Claude Code / Codex / batch harness in a container or CI runner, governed by the kernel with adjudication, a hash-chained audit journal, a cost cap, and no TTY. The complement to embed-in-your-product.md: that page governs your own direct API call; this page governs a harness someone else drives."
---

# Run `fak guard` as a server-side client (unattended / container)

**Reader:** a platform or SRE operator running a *hosted harness* — a Claude Code or Codex
worker in a container, a CI runner, or a batch agent — and you want the kernel governing it
server-side: every tool call adjudicated, a tamper-evident audit journal, a hard cost cap,
and a clean restart lifecycle, with **no interactive terminal**.

This is the "guard as a server-side client" half of the runtime disambiguation in epic
[#3256](https://github.com/anthony-chaudhary/fak/issues/3256) (workstream B, issue
[#3267](https://github.com/anthony-chaudhary/fak/issues/3267)). Name the two roles so they
stop being conflated:

| Role | Who drives the loop | fak surface | This page |
|---|---|---|---|
| **Agent application runtime** | fak drives a loop *you* POST goals to | `fak serve --native` / `fak run` | no — see [embed-in-your-product.md](integrations/embed-in-your-product.md) |
| **Governed client of someone else's harness** | the *harness* (Claude Code, Codex) drives its own loop; fak wraps and governs the child process | `fak guard -- <harness>` | **yes** |

`fak guard` is documented elsewhere as a local-dev wrapper ("wrap the agent you already run
on your laptop"). The same seam runs server-side unchanged: the flags and env below are
already shipped — this page ties them into one unattended, containerized posture.

## The unattended posture is already the default off a TTY

`fak guard` auto-detects a headless launch (no color TTY, piped, or CI) and drops every
interactive affordance without a flag:

- **Banner** resolves to the full captured report instead of the interactive icon
  animation — a captured log wants the detail (`--banner auto`, `cmd/fak/guard.go`).
- **Split UI** (`--split auto`) is a no-op for a headless/piped/CI launch — no attempt to
  open a companion pane.
- **Operator-directed Stop hook** (`--operator-directed`) keeps `enforce`/`warn` for a
  headless worker so a turn cannot silently stall on "do you want me to push?" — a question
  no one is there to answer.
- **Account rotation** (`--rotate`) defaults to `auto` (headless), `off` (interactive).

You do not opt *into* unattended mode; you opt *out* of the interactive extras, which are
already suppressed when there is no terminal. What you add server-side is authentication, a
durable audit journal, and a cost cap.

## The three server-side switches

| Requirement | Switch | Note |
|---|---|---|
| **Authentication** on an off-loopback gateway bind | `--require-key-env FAK_GATEWAY_KEY` (`cmd/fak/guard.go`) | Loopback binds are auth-exempt; a container that keeps the child↔gateway traffic on loopback rarely needs a key. Bind off-host → this is mandatory (guard refuses an off-host bind with no key). |
| **Durable, hash-chained audit journal** | `FAK_AUDIT_JOURNAL=<path>` (`cmd/fak/guard_support.go`) | Opt-in globally; **bake the env into the image** and it is on by default for that image. `--audit off` disables. The in-memory exit summary is always on; only the durable trail is gated. |
| **Cost cap** | `--budget-envelope 'spend=$25,tokens=200000,wall=2h'` (`cmd/fak/guard.go`) | A hard dollar/token/wall envelope; exhaustion drains the session to `Stopped`. `--max-duration` and `--context-budget-tokens` set single axes. |

## Supervised / restarted lifecycle

A hosted worker is expected to be restarted. Guard already ties the restart supervisor to
the budget reset:

- `--restart-on-budget` — on context-budget exhaustion, stop and relaunch the wrapped child
  under the continuation trace, writing a carryover seed and exposing it via `FAK_RESET_*`
  env vars (requires `--context-budget-tokens`). `--restart-limit N` caps relaunches.
- `--max-duration` is tracked independently and **survives a restart** — the elapsed total
  carries forward, it does not reset to zero — so a supervisor that re-execs the container
  cannot launder the wall-clock cap.

Put the container's own supervisor (k8s `restartPolicy`, a compose `restart:` policy, or
`systemd`) *outside* guard; guard owns the in-session budget/seed continuity, the
orchestrator owns process liveness.

## Health & inspection endpoint

The wrapped child talks to guard's in-process gateway, which honors `/healthz`,
`/debug/vars`, and `/metrics` (`cmd/fak/guard.go`). Use them as the container's health and
inspection surface:

- `GET /healthz` — process readiness (the same probe `fak serve` exposes).
- `GET /debug/vars` — the live session view; `startup_report` carries the full boot banner
  (`fak info --startup`).
- `GET /metrics` — Prometheus counters (cache economy, adjudication, the audit chain).
- `fak session status <id>` — the budget/pace/wall drain state for the running session.

`/metrics` and the full headless banner are the structured-log surface; add
`--debug-stats <file>` to stream per-turn stats to a mounted path.

## Reference container recipe (not yet witnessed live)

The shipped [`Dockerfile`](../Dockerfile) builds the static `fak` binary onto a distroless
base and runs `fak serve` — the *gateway* runtime. Guard-as-a-client needs one thing that
image deliberately lacks: **the harness binary must be present in the image**, because guard
`exec`s it as the child. So a guard worker image starts from a base that carries the harness
(e.g. Node for Claude Code) and mounts a volume for the audit journal:

```dockerfile
# Reference only — pending a recorded live witness (see "Status" below).
FROM node:22-bookworm-slim
COPY --from=ghcr.io/anthony-chaudhary/fak:latest /usr/local/bin/fak /usr/local/bin/fak
RUN npm install -g @anthropic-ai/claude-code   # the harness guard will exec
ENV FAK_AUDIT_JOURNAL=/audit/journal.jsonl      # durable trail on by default for this image
ENTRYPOINT ["fak", "guard", \
  "--budget-envelope", "spend=$25,tokens=200000,wall=2h", \
  "--context-budget-tokens", "200000", "--restart-on-budget", \
  "--require-key-env", "FAK_GATEWAY_KEY", \
  "--"]
CMD ["claude", "-p", "..."]
```

```yaml
# compose: mount the audit journal, pass the key, restart the worker.
services:
  guarded-worker:
    build: .
    restart: unless-stopped        # the orchestrator owns liveness; guard owns budget
    environment:
      FAK_GATEWAY_KEY: ${FAK_GATEWAY_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    volumes:
      - ./audit:/audit             # the hash-chained journal survives a restart
```

The k8s equivalent is a `Deployment` with the same env + a `PersistentVolumeClaim` mounted
at `/audit`; reuse the base and probe wiring in [`deploy/k8s/README.md`](../deploy/k8s/README.md)
(`/healthz` is the readiness probe) and swap the `serve` command for the `guard` entrypoint
above.

## Status: generation, evidence, and what is not yet proven

**Generation horizon: `gen/next`.** This is a near-term foundation for the corporate
agent-runtime program (epic #3256, milestone #17). The unattended machinery is *shipped* —
headless auto-detection, `--require-key-env`, `FAK_AUDIT_JOURNAL`, `--budget-envelope`,
`--restart-on-budget`, and the `/healthz`+`/metrics`+`/debug/vars` surface all exist today —
so the work is packaging and a default-exposure/dogfood proof, not a future architecture bet
([`docs/generation.md`](generation.md)).

- **Promotion evidence (documented mode):** every switch above cites a shipped flag/env in
  `cmd/fak/guard.go` / `cmd/fak/guard_support.go`; this page is the "documented
  containerized/unattended mode" the issue asks for.
- **Demotion / retirement evidence:** if a future change makes the durable audit journal
  on-by-default under a detected container (removing the "bake the env in" step), the audit
  row of this page retires in favor of that default. If the OpenAI-wire seat's live witness
  (`experiments/agent-live/openai-wire-seat-guard-live-witness-2026-06-29.json`) is shown to
  already cover a containerized run, the "not yet witnessed live" caveat retires.
- **Invalidating assumption:** the reference recipe above assumes guard can `exec` the
  harness from an image that carries it and that the four-check gateway-transited proof holds
  identically inside a container. That has **not** been run here. Until a container run of a
  real harness (with real credentials) records the four-check witness into
  `experiments/agent-live/`, the recipe is a documented reference, not a certified image.

**Not yet reached (the open acceptance gate):** the recorded live container witness — the
four-check gateway-transited proof captured from an unattended containerized `fak guard --
<harness>` run, landed in `experiments/agent-live/`. It needs a host that can run the harness
in a container with live credentials; producing it is the next checkable step toward closing
#3267.
