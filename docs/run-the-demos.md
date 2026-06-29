---
title: "Run the fak demos yourself | lowest-common-denominator first, then dedicated tracks"
description: "How to run the fak demos on your own machine or cloud host: start with no-key/no-model lowest-common-denominator demos, then choose security, research/science, memory/serving, adoption, or live-model tracks."
---

# Run the demos yourself

The [live demos](demos.html) run on our GCP host, but a new runner should not have to
start with hosted infrastructure, a model download, a GPU, or an API key. Start with the
lowest-common-denominator set: one Go command, deterministic output, no browser required,
no network, no model weights.

| Start-here demo | Command | Track |
|---|---|---|
| **dropindemo** — how fak wraps the agent you already run | `go run ./cmd/dropindemo -print` | adoption |
| **guarddemo** — the safety floor, side by side | `go run ./cmd/guarddemo -print` | security |
| **turntaxdemo** — forced turn tax deleted in the kernel | `go run ./cmd/turntaxdemo -print` | efficiency |
| **tokendemo** — model-context and tool-call token ledger | `go run ./cmd/tokendemo -print` | efficiency |
| **unseedemo** — poisoned KV span removed with a bit-exact witness | `go run ./cmd/unseedemo -print` | research/science |

Everything else is still runnable, but it belongs in a dedicated track instead of the
front door:

| Track | Use it when you want... | Demos |
|---|---|---|
| **Security and policy** | default-deny behavior, tool poisoning, safe escalation, policy reloads, or red-team corpora | `guarddemo`, `poisonedmcpdemo`, `examples/adjudication-demo`, `examples/agentdojo-redteam`, `examples/wire-proof`, `examples/wire-quarantine-demo`, `examples/auth-hardening`, `examples/escalation-demo`, `examples/policy-hot-reload`, `examples/trace-reset`, `examples/presets`, `examples/observability` |
| **Research/science** | context reuse, KV/cache proofs, deletion certificates, or causal invalidation | `ctxdemo`, `demorace`, `tokendemo`, `unseedemo`, `ctxplandemo`, `causalbench`, `deletioncert`, `examples/routing-bench` |
| **Memory and serving systems** | placement, CXL/pool economics, memory-query composition, or model-backed races | `hwcachedemo`, `cxlpooldemo`, `memqdemo`, `simpledemo`, `demorace` |
| **Adoption and integrations** | wrapping existing agents, MCP, OpenAI Agents, AutoGen, CrewAI, or external drivers | `dropindemo`, `a2ademo`, `examples/mcp`, `examples/mcp-client`, `examples/openai-agents-guardrail`, `examples/autogen-groupchat`, `examples/crewai-crew`, `examples/extdriver` |
| **Shared task records** | task-record interchange, verdict fixtures, and reproducible task-state payloads | `examples/shared-task-record`, `examples/shared-task-record-verdicts` |

Browser demos by track:

| Demo | Command | Track | Needs a model? |
|------|---------|-------|----------------|
| **dropindemo** — drop-in wiring gallery for existing coding agents | `go run ./cmd/dropindemo` | adoption | no — self-contained |
| **guarddemo** — the safety floor, side by side (WITHOUT fak vs WITH fak, same attack) | `go run ./cmd/guarddemo` | security | no — self-contained |
| **turntaxdemo** — turn-tax race (SOTA loop vs fak 1-shot) | `go run ./cmd/turntaxdemo` | efficiency | no — self-contained |
| **unseedemo** — Un-See It / Lobotomy Cam (poisoned result deleted from KV cache, with bit-exact witness) | `go run ./cmd/unseedemo` | research/science | no — self-contained |
| **timewolfdemo** — "what time is it, Mr. Wolf?" agent loop, kernel-gated (a smuggled destructive call refused at the floor, inside the loop) | `go run ./cmd/timewolfdemo` | agentic | no — self-contained |
| **ctxdemo** — multi-agent context-reuse proof (fak vs a tuned warm-cache SOTA baseline) | `go run ./cmd/ctxdemo` | research/science | optional (live race needs one) |
| **demorace** — reuse race (fak vs a tuned warm-cache SOTA baseline) + reuse curve | `go run ./cmd/demorace` | live model research | yes (live race) |

Headless witnesses by track:

| Demo | Command | Track | What it proves |
|------|---------|-------|----------------|
| **dropindemo** | `go run ./cmd/dropindemo -selfcheck` | adoption | provider/upstream/base-URL wiring is generated without a key or network |
| **a2ademo** | `go run ./cmd/a2ademo` | adoption | in-kernel agent-to-agent delivery, denial, session/window handoff, and pub/sub |
| **ctxplandemo** | `go run ./cmd/ctxplandemo -selfcheck` | research/science | O(1) context view planning, poison exclusion, recoverability, and scaling invariants |
| **hwcachedemo** | `go run ./cmd/hwcachedemo` | memory/serving | hardware-aware demote-not-evict placement beats blind LRU under pressure |
| **cxlpooldemo** | `go run ./cmd/cxlpooldemo` | memory/serving | coherent pooled memory can save both prefills and resident KV copies, gated by trust |
| **memqdemo** | `go run ./cmd/memqdemo` | memory/serving | memory strategies are composable queries; effects default fail-closed |
| **poisonedmcpdemo** | `go run ./cmd/poisonedmcpdemo` | security | poisoned MCP results are quarantined and unwired tools are denied by structure |
| **causalbench** | `go run ./cmd/causalbench -selfcheck` | research/science | an external write evicts exactly the dependent cached read, keeps siblings warm, and refuses stale re-admission |
| **deletioncert** | `go run ./cmd/deletioncert -selfcheck` | research/science | a selected KV span is evicted to `max|Delta|=0`, bound into a certificate, and tamper-rejected |

`guarddemo`, `turntaxdemo`, `tokendemo`, `dropindemo`, and `unseedemo` are the
lowest-common-denominator demos: no model, no GPU, no download, no provider key, no network.
They run against frozen fixtures or synthetic witnesses through the real kernel path, so they
reproduce identically on any box with Go. `ctxdemo` is also model-free in `-print` and
`-bars` modes, but it lives in the research track because its main job is explaining
multi-agent context reuse rather than onboarding.

## Adding a demo

New demos should not expand the front door by default. A new start-here demo must meet the
lowest-common-denominator bar: one Go command, deterministic output, no provider key, no
network, no model weights, no GPU, and no hosted dependency. If it needs a model, a framework,
network I/O, a live gateway, a corpus, or a specialized mental model, put it in the matching
track above instead: security/policy, research/science, memory/serving, adoption/integration,
or shared task records.

Before landing a new demo or moving one between tracks, update this guide, the demo's README,
and the relevant scorecard snapshot, then run:

```bash
python tools/demo_command_audit.py
python tools/demo_quality_scorecard.py --check-doc
python tools/demo_robustness_scorecard.py --check-doc
python tools/demo_headless_smoke.py --timeout 120
```

## 1. Local — one command

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak

# the self-contained ones — no model, no GPU, no downloads:
go run ./cmd/dropindemo            # -> http://127.0.0.1:8154
#   inspect the drop-in wiring gallery for Claude Code, Codex, OpenCode, and Aider
go run ./cmd/guarddemo             # → http://127.0.0.1:8151
#   pick a scenario → "Run both agents"  (WITHOUT fak vs WITH fak, side by side)
go run ./cmd/turntaxdemo            # → http://127.0.0.1:8150
#   pick a suite → "Replay through the kernel"
go run ./cmd/unseedemo              # → http://127.0.0.1:8156
#   press Play → watch the poisoned KV span get evicted
go run ./cmd/timewolfdemo          # → http://127.0.0.1:8155
#   pick a scenario → "Run the agent"  (the children's game as a kernel-gated agent loop)

go run ./cmd/ctxdemo               # → http://127.0.0.1:8153
go run ./cmd/demorace             # → http://127.0.0.1:8147
```

**What you'll see:** each server prints the loopback URL it bound — `guarddemo` on
`http://127.0.0.1:8151`, `turntaxdemo` on `:8150`, `dropindemo` on `:8154`,
`unseedemo` on `:8156`, `ctxdemo` on `:8153`, and `demorace` on `:8147` (as shown in
the comments above) — then waits; open that URL in a browser to drive the demo. On a
shared/busy machine, cap the demos that accept it with `-jobs 8` (absolute) or
`-budget 0.75` (a fraction of the box) so they don't starve other work.

## 2. Headless — exact accounting, no browser, no model

This section is the CI-friendly surface: every command below exits, needs no browser, and
uses no model weights or provider network. It is grouped the same way as the catalog above.

Lowest-common-denominator terminal demos:

```bash
go run ./cmd/dropindemo -print                         # adoption: one static binary, child-only env wiring
go run ./cmd/guarddemo  -print                         # security: WITHOUT fak vs WITH fak (4 breaches -> 0)
go run ./cmd/guarddemo  -print -scenario turntax-happy # security: clean control (0 breaches)
go run ./cmd/turntaxdemo -print                         # efficiency: tuned SOTA vs fak (5 forced turns -> 0)
go run ./cmd/turntaxdemo -print -suite turntax-happy    # efficiency: anti-inflation control
go run ./cmd/tokendemo   -print                         # tokens: prefiltered /bad call keeps 1,452 model-context tokens out
go run ./cmd/unseedemo   -print                         # research/science: three-act KV eviction witness
```

Research/science accounting:

```bash
go run ./cmd/ctxdemo -print          # a table of per-strategy prefill-token work
go run ./cmd/ctxdemo -print -json     # the same, as JSON
go run ./cmd/ctxdemo -bars            # the reuse axis as a SIDE-BY-SIDE bar chart (cold vs warm-cache vs fak)
go run ./cmd/ctxdemo -bars -scenario deep-research   # just one scenario
```

`turntaxdemo` is always model-free; you can hit its API directly once it's running:

```bash
curl "http://127.0.0.1:8150/api/run?suite=turntax-airline" | jq .net
# → {"turns_saved":9,"tokens_saved":11880, ...}
```

Acceptance self-checks:

```bash
go run ./cmd/dropindemo -selfcheck   # adoption wiring invariants
go run ./cmd/guarddemo  -selfcheck   # WITHOUT fak: 4 / 2 / 0 breaches · WITH fak: 0 (per scenario)
go run ./cmd/turntaxdemo -selfcheck   # turn-tax + safety-floor invariants per suite
go run ./cmd/tokendemo  -selfcheck   # token-ledger invariants per suite (incl. the clean control at 0)
go run ./cmd/unseedemo  -selfcheck   # KV eviction invariants: quarantine, evict-vs-never == 0, late-evict boundary
```

`tokendemo` and `unseedemo` are also fully headless (no browser at all): `-print` for the
terminal walkthrough, `-json` for the exact event/ledger, `-selfcheck` for the invariants.

```bash
go run ./cmd/tokendemo -print -suite prefilter-bad-calls   # win 1 (model context): 4 /bad calls refused -> 1,452 tok kept out of the model
go run ./cmd/tokendemo -print -suite reread-same-file       # win 2 (tool-side): 3 re-reads served from cache -> the tool ran 3x, not 6x
go run ./cmd/tokendemo -json                                 # the exact per-call ledger (both meters), all suites
go run ./cmd/unseedemo -print                                # the three-act KV eviction witness in the terminal
go run ./cmd/unseedemo -json                                 # the browser event log as JSON
```

Dedicated security, memory, and research witnesses:

```bash
go run ./cmd/a2ademo
go run ./cmd/ctxplandemo -selfcheck
go run ./cmd/hwcachedemo
go run ./cmd/cxlpooldemo
go run ./cmd/cxlpooldemo -profiles cmd/cxlpooldemo/calibration.example.json
go run ./cmd/memqdemo
go run ./cmd/memqdemo -report memqdemo-report.json
go run ./cmd/poisonedmcpdemo
go run ./cmd/poisonedmcpdemo -json
go run ./cmd/causalbench -selfcheck
go run ./cmd/causalbench -selfcheck -out causalbench-witness.json
go run ./cmd/deletioncert -selfcheck
go run ./cmd/deletioncert -selfcheck -out deletioncert.json
```

Or play all four in one shot — **fak in 30 seconds**, then a built-in acceptance check
that each comparison still reproduces its documented headline (a cross-platform gate, no
model, no network):

```bash
bash tools/run_comparison_demos.sh        # play all four side-by-sides, then verify
bash tools/run_comparison_demos.sh -q     # quiet: just the acceptance gate (CI-usable)
```

## 3. With a real model (the live race)

`ctxdemo` and `demorace` run a model **in-process** for the live race. Export a small one
once — CPU is enough, no GPU required:

```bash
scripts/fetch-model.sh               # downloads + exports SmolLM2-135M (~hundreds of MB)
#   writes internal/model/.cache/smollm2-135m/{config.json,manifest.json,weights.f32}
```

Now `go run ./cmd/demorace` (or `ctxdemo`) detects the model and the live race lights up.
The demos also auto-detect bigger rungs (Qwen2.5 0.5B–3B) if you export them under
`~/.cache/fak-models/hf/`; missing rungs are simply skipped. See
[Getting Started §4b](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) for the full model-export options.

## 4. In a container

The demo binaries honor the `$PORT` contract (they bind `0.0.0.0:$PORT` when `PORT` is
set, else the loopback `-addr` default), so they drop straight into any container or PaaS
that injects a port. A minimal image for the self-contained demo:

```dockerfile
# Dockerfile.turntax
FROM golang:1.26 AS build
ENV CGO_ENABLED=0
WORKDIR /src
COPY . .
RUN go build -trimpath -ldflags "-s -w" -o /out/turntaxdemo ./cmd/turntaxdemo

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /usr/local/bin
COPY --from=build /out/turntaxdemo /usr/local/bin/turntaxdemo
# the trace fixtures must sit at ./testdata/turntax relative to the working dir:
COPY testdata/turntax /usr/local/bin/testdata/turntax
ENTRYPOINT ["/usr/local/bin/turntaxdemo"]
```

```bash
docker build -f Dockerfile.turntax -t fak-turntax .
docker run --rm -p 8150:8150 -e PORT=8150 fak-turntax
```

For the model demos, add a build stage that runs `internal/model/export_oracle.py --online
--model HuggingFaceTB/SmolLM2-135M-Instruct --out internal/model/.cache/smollm2-135m` and
copy that directory into the runtime image next to the binary, preserving the
`internal/model/.cache/smollm2-135m` path (it is resolved relative to the working dir).

## 5. On your own cloud VM

Any always-on Linux VM works — there is nothing GCP-specific. The shape we run:

```bash
# on the VM (substitute YOUR_* values):
#  1. build the image (locally or with your cloud's build service) and push it to
#     YOUR_REGISTRY, OR just `go build` the binaries on the VM if it has Go.
#  2. run each demo on the host network (or publish the port), honoring $PORT:
PORT=8151 ./guarddemo &
PORT=8150 ./turntaxdemo &
PORT=8154 ./dropindemo &
PORT=8156 ./unseedemo &
PORT=8155 ./timewolfdemo &
PORT=8153 ./ctxdemo &
PORT=8147 ./demorace &
#  3. open YOUR_VM_IP:<port> in the firewall / security group for inbound TCP.
```

If your environment restricts which inbound ports are reachable (some org firewall
policies pin a fixed set), put the demos behind a reverse proxy on a port that *is* open.
The browser pages call their APIs relative to the current path, and each browser demo
accepts `-base-path` (or `FAK_DEMO_BASE_PATH`) for proxies that preserve the path prefix:

```bash
FAK_DEMO_BASE_PATH=/guarddemo PORT=8151 ./guarddemo
FAK_DEMO_BASE_PATH=/turntax   PORT=8150 ./turntaxdemo
FAK_DEMO_BASE_PATH=/dropin    PORT=8154 ./dropindemo
FAK_DEMO_BASE_PATH=/ctxdemo   PORT=8153 ./ctxdemo
FAK_DEMO_BASE_PATH=/demorace  PORT=8147 ./demorace
FAK_DEMO_BASE_PATH=/unsee     PORT=8156 ./unseedemo
FAK_DEMO_BASE_PATH=/timewolf  PORT=8155 ./timewolfdemo
```

That means an HTTPS host can mount `/guarddemo/`, `/turntax/`, `/dropin/`,
`/ctxdemo/`, and `/demorace/` without HTML rewriting; the same contract also covers
`/unsee/` and `/timewolf/`. If your proxy strips the prefix before forwarding, the demos also work with
the default root mount. SSE endpoints (`api/race`, `api/curve`) need
`proxy_buffering off` and a long `proxy_read_timeout`.

Minimal Nginx shape for a single HTTPS hostname with path-preserving mounts:

```nginx
server {
    listen 443 ssl http2;
    server_name demos.example.com;

    location /guarddemo/ {
        proxy_pass http://127.0.0.1:8151;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /turntax/ {
        proxy_pass http://127.0.0.1:8150;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /dropin/ {
        proxy_pass http://127.0.0.1:8154;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ctxdemo/ {
        proxy_pass http://127.0.0.1:8153;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 1h;
    }

    location /demorace/ {
        proxy_pass http://127.0.0.1:8147;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 1h;
    }

    location /unsee/ {
        proxy_pass http://127.0.0.1:8156;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /timewolf/ {
        proxy_pass http://127.0.0.1:8155;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

When demo docs change, run the full local demo health gate before publishing:

```bash
make demo-audit
```

That target runs the static docs/metadata checks, the content-only quality/robustness
scorecards, the demo-audit tool unit tests, every browser demo behind its base path,
and the deterministic headless witnesses. For targeted debugging, run the pieces
directly:

```bash
make demo-tool-tests                  # unit tests for the demo audit harnesses
python tools/demo_command_audit.py        # static: catches stale go run/script refs in demo docs
python tools/demo_browser_contract.py     # static: keeps browser default ports/base paths synced with run/public docs
python tools/demo_quality_scorecard.py    # static: demo-debt scorecard for runnable/reproducible/honest docs
python tools/demo_quality_scorecard.py --check-doc       # static: committed scorecard snapshot is fresh
python tools/demo_robustness_scorecard.py # static: robustness-debt scorecard for simple/fast/durable demos
python tools/demo_robustness_scorecard.py --check-doc    # static: committed robustness snapshot is fresh
python tools/demo_http_smoke.py --timeout 60  # dynamic: builds browser demos and fetches base-path page/API locally
python tools/demo_headless_smoke.py --timeout 120  # dynamic: runs deterministic headless witnesses
python tools/demo_live_links.py          # static: catches stale hosted paths in docs/demos.html
python tools/demo_live_links.py --live   # probes hosted pages/APIs plus HTTPS alternatives with short timeouts
python tools/demo_live_links.py --live --status  # concise hosted/local-only status table
python tools/demo_live_links.py --live --require-https --status  # optional hard gate for HTTPS embeddability
python tools/demo_live_links.py --live --json  # machine-readable hosted/local-only HTTP/API/HTTPS status matrix
python tools/demo_live_links.py --published  # optional: checks the HTTPS GitHub Pages copy and share image
python tools/demo_live_links.py --readiness  # optional all-up status: static + live + HTTPS + published
```

`make demo-live-status` runs the same live VM probe as `--live --status` and prints
the compact hosted/local-only HTTP/API/HTTPS table. It is intentionally optional and
networked.

`make demo-https-status` runs the stricter `--live --require-https --status` launch gate
for places that need demos embedded or loaded from an HTTPS page. The default live check
stays green when plain HTTP top-level navigation works; `demo-https-status` exits non-zero
until the hosted demo stack has TLS termination.

`make demo-readiness-status` runs the all-up deployment view: local static page, live VM,
strict HTTPS, and published GitHub Pages. Use it before a launch/share pass when you want
one command to say which external surface still needs work.

The `surfaces:` line separates local code health from deployment state:
`local` is the checked-in `docs/demos.html` static/metadata/hosted-link contract,
`hosted` is the live VM over plain HTTP, `launch` is strict HTTPS embeddability/TLS
termination, and `pages` is the published GitHub Pages copy plus remote share image.
If `local` and `hosted` are green while `launch` or `pages` are red, the demos and local
page source are healthy; the remaining work is external TLS or Pages deployment drift.

`make demo-published-check` runs the same optional published-page check. It depends on
external deployment state, so it is intentionally not part of the local green gate. If it
reports `published_deployment_drift`, the checked-in `docs/demos.html` is clean but the
GitHub Pages copy is stale; republish Pages or wait for the branch deployment to catch up,
then rerun it. Other failures are real public-page or share-card drift and should be fixed
before linking the page publicly. `make demo-published-status` prints the same published
check as a compact status table and exits non-zero on the same drift.
`make demo-scorecards` runs just the content-only quality and robustness scorecards,
including the committed scorecard snapshot freshness checks.
`make demo-tool-tests` runs just the unit tests for the demo registry, hosted-link
audit, command-reference audit, browser-contract audit, and smoke-test harnesses.
`make demo-smoke` runs the same dynamic browser check as the local/CI gate.
`make demo-headless-smoke` runs the same deterministic terminal witness set.

## Updating the demos on the long-lived serve box

Our live demos are backed by a **persistent, low-cost serve VM** (a single NVIDIA L4)
that runs the fak kernel's own forward (`fak serve --backend cuda`) behind
`/v1/messages`. Unlike the on-demand datacenter GPU serve nodes — which carry an idle gardener
that self-stops or self-deletes them after an idle window (`scripts/gcp-idle-reaper.sh`)
so a crashed control-session can't leave a GPU burning — this demo box is meant to
**stay up**, so it has no reaper.

To refresh what it serves: pull `/opt/fak`, rebuild if the kernel changed, and restart
the serve systemd unit, then re-check `/healthz` and the `/metrics`
`fak_gateway_inference_requests_total` counter. The box self-documents these exact
steps in `/opt/fak/DEMO-HOST.md`, which `scripts/gcp-realmodel-note.sh` writes (and
re-writes) onto it — so whoever lands on the box next knows what it is and how to
update it without reading the whole repo. The note carries no project id or key; the
private deployment specifics stay out of the repo (see below).

> **Note on our hosted copy.** The scripts that drive *our* specific GCP deployment are
> intentionally **not** in the repo — they embed our private project id and host details.
> Everything you need to stand up your *own* copy is above; nothing about it depends on our
> environment.

---

Next: ran a demo and want to drive the kernel yourself? Walk the
[fak tutorial](fak/tutorial.md) — zero to your first adjudicated tool call, offline.

← Back to [the live demos](demos.html) · [the showcase](showcase.html) · [docs home](./)
