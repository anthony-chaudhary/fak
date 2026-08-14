---
title: "Run fak manage as a server-side client (unattended / container)"
description: "Operator route for running `fak manage -- <harness>` server-side — a hosted Claude Code / Codex / batch harness in a container or CI runner, governed by the kernel with adjudication, a hash-chained audit journal, a cost cap, and no TTY. The complement to embed-in-your-product.md: that page governs your own direct API call; this page governs a harness someone else drives."
---

# Run `fak manage` as a server-side client (unattended / container)

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
| **Governed client of someone else's harness** | the *harness* (Claude Code, Codex) drives its own loop; fak wraps and governs the child process | `fak manage -- <harness>` | **yes** |

`fak manage` is documented elsewhere as a local-dev wrapper ("wrap the agent you already run
on your laptop"). The same seam runs server-side unchanged: the flags and env below are
already shipped — this page ties them into one unattended, containerized posture.

## The unattended posture is already the default off a TTY

`fak manage` auto-detects a headless launch (no color TTY, piped, or CI) and drops every
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

## Reference container recipe (run live — see the witness below)

The shipped [`Dockerfile`](../Dockerfile) builds the static `fak` binary onto a distroless
base and runs `fak serve` — the *gateway* runtime. Guard-as-a-client needs one thing that
image deliberately lacks: **the harness binary must be present in the image**, because guard
`exec`s it as the child. So a guard worker image starts from a base that carries the harness
(e.g. Node for Claude Code) and mounts a volume for the audit journal:

```dockerfile
# This recipe was built and run — see "Recorded live witness" below.
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/anthony-chaudhary/fak:latest /usr/local/bin/fak /usr/local/bin/fak
RUN npm install -g @anthropic-ai/claude-code   # the harness guard will exec
ENV FAK_AUDIT_JOURNAL=/audit/journal.jsonl      # durable trail on by default for this image
ENV HOME=/home/worker
RUN mkdir -p /home/worker /audit /work && chmod 777 /home/worker /audit /work
WORKDIR /work
ENTRYPOINT ["fak", "guard", \
  "--log", "/audit/gw.log", \
  "--anthropic-oauth", \
  "--budget-envelope", "spend=$25,tokens=200000,wall=2h", \
  "--"]
CMD ["claude", "-p", "..."]
```

Run it **without `-t`** — no TTY is the posture, not a limitation:

```bash
docker run --rm -e CLAUDE_CODE_OAUTH_TOKEN -v "$PWD/audit:/audit" fak-guard-worker
```

Pass the token **by name** (`-e CLAUDE_CODE_OAUTH_TOKEN`, no `=value`) so the secret never
lands in a command line, a shell history, or `docker inspect`.

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

## Credentials in an unattended container

Export the subscription token as **`CLAUDE_CODE_OAUTH_TOKEN`** (a `claude setup-token`
value — long-lived, unlike the interactive-login token). It is first in
`resolveAnthropicOAuthToken`'s precedence, ahead of `<claude-config>/.credentials.json` and
`.oauth-token`, so a container needs no credential file and no `claude` login at all.

> **Upgrade note (#3267).** Before this fix, a container that did exactly that **hung** —
> guard's pre-spawn STALE_CRED rung inspected the credential *file*, found it absent, and
> parked for its full 24h re-login budget *before* binding the gateway or spawning the
> child. There is no interactive `claude` in a container to end that park, so the worker sat
> silent: no request, no audit rows, no failure. Guard now defers to the env token and skips
> the park, because the file it was polling is not the credential being sent upstream. If you
> are pinned to an older build, `FAK_GUARD_PARK_BUDGET=0` restores the immediate fail-loud
> refusal and is the correct setting for any unattended worker on those versions.

The park itself is still right for an *attended* headless host, where a human really can run
`claude` and self-heal the fleet — it is only meaningless when an env token already outranks
the file.

## Recorded live witness

`experiments/agent-live/guard-container-unattended-live-witness-2026-08-06.json` — the
recipe above, built and run unattended (Docker 28.3.3, `node:22-bookworm-slim`,
`claude` 2.1.223, `fak` built from `7b41840b63` + the #3267 fix), status **PARTIAL**:

| Check | Result |
|---|---|
| **2 — request transited the gateway** | **PASS** — 4 `POST /v1/messages` records in the mounted `gw.log`, user agent `claude-cli/2.1.223`, gateway bound in-process on `127.0.0.1:41415` |
| **4 — hash-chained journal** | **PASS (durable trail)** — 6 rows on the mounted volume, chain intact end to end; **0 DECIDE rows**, because no turn completed |
| **1 — real result through guard** | **BLOCKED** — every upstream call refused `403 oauth_not_allowed_for_organization` |
| **3 — no-bypass credential swap** | **BLOCKED** — the proof of the swap *is* the upstream 200, which check 1 never got |

The blocker is an **organization policy** on the host's Anthropic account ("OAuth
authentication is currently not allowed for this organization"), applied upstream — not a
fak defect. Guard resolved and sent the bearer correctly; its banner reads `upstream auth —
Claude Pro/Max subscription (… OAuth token from $CLAUDE_CODE_OAUTH_TOKEN, sent as a bearer
token)`. Re-running the same image on a host whose org permits OAuth auth (or with an
`ANTHROPIC_API_KEY` seat via `--api-key-env`) should close checks 1, 3 and the DECIDE row in
one run.

Two things the run witnessed that the acceptance list did not ask for:

- **The supervised-restart lifecycle, live.** The 6 journal rows are three
  `CHILD_CRASH` → `RESTART_HOP` pairs: guard's supervisor saw the harness child die,
  recorded it, relaunched it, and chained every hop. An unattended worker's crash/restart
  history is auditable after the fact.
- **The cost cap seeding.** `--budget-envelope 'spend=$1,tokens=200000,wall=10m'` seeded the
  session (`max_duration=10m0s` on the wall axis). Enforcement of the spend/token axes is
  *not* witnessed — no turn ever consumed any.

## Status: generation, evidence, and what is not yet proven

**Generation horizon: `gen/next`.** This is a near-term foundation for the corporate
agent-runtime program (epic #3256, milestone #17). The unattended machinery is *shipped* —
headless auto-detection, `--require-key-env`, `FAK_AUDIT_JOURNAL`, `--budget-envelope`,
`--restart-on-budget`, and the `/healthz`+`/metrics`+`/debug/vars` surface all exist today —
so the work is packaging and a default-exposure/dogfood proof, not a future architecture bet
([`docs/generation.md`](generation.md)).

- **Promotion evidence (documented mode):** every switch above cites a shipped flag/env in
  `cmd/fak/guard.go` / `cmd/fak/guard_support.go`; this page is the "documented
  containerized/unattended mode" the issue asks for. The recipe was then **built and run** —
  the witness above — promoting it from a paper reference to one that boots, binds, spawns
  the harness, transits the gateway, and writes a verified hash-chained journal. That run is
  also what surfaced the unattended-hang defect in the upgrade note: the packaging exercise
  found a real behavioral gap, not just a docs gap.
- **Demotion / retirement evidence:** if a future change makes the durable audit journal
  on-by-default under a detected container (removing the "bake the env in" step), the audit
  row of this page retires in favor of that default. The "not yet witnessed live" caveat has
  already retired against the witness above; its **PARTIAL** qualifier retires when checks 1
  and 3 close on a permitting org.
- **Invalidating assumptions:** (1) the 403 is read as an org policy rather than a revoked
  setup token — both present as 403, and the `oauth_not_allowed_for_organization` code names
  the organization, but a run with a freshly minted `claude setup-token` on a permitting org
  would settle it; (2) the restart rows were produced by a child crashing on an upstream 403,
  so they witness crash-restart chaining, **not** the `--restart-on-budget` path, which this
  run never triggered; (3) the fix defers to the env token without validating it, so an
  exported-but-expired token now surfaces as a reactive upstream 401 instead of a pre-spawn
  park — the intended trade, since the park polled a different credential entirely.

**Not yet reached (the remaining acceptance gate):** checks 1 and 3 of the four-check proof —
a real result over a 200, and the credential-swap proof that depends on it — plus a `DECIDE`
row for an adjudicated tool call inside the container. All three are gated behind the same
external blocker: this host's Anthropic account refuses OAuth auth at the organization level
(`403 oauth_not_allowed_for_organization`). The smallest next step is a re-run of the **same
image** on a permitting org or an `ANTHROPIC_API_KEY` seat; nothing in the recipe or the repo
needs to change.
