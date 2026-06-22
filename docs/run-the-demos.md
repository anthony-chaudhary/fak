---
title: "Run the fak demos yourself | local, headless, Docker, or your own cloud VM"
description: "How to run the interactive fak demos (guarddemo, turntaxdemo, demorace, ctxdemo) on your own machine or your own cloud host — locally with one go run, headless with no model, in a container, or self-hosted on a VM. No private infrastructure required."
---

# Run the demos yourself

The [live demos](demos.html) run on our GCP host, but every one of them is in the public
repo and runs anywhere Go runs. This page shows four ways to run them yourself — pick the
row that fits. Nothing here needs our infrastructure; the commands are the same ones the
demo source documents at the top of each `main.go`.

The demos:

| Demo | Command | Needs a model? |
|------|---------|----------------|
| **guarddemo** — the safety floor, side by side (WITHOUT fak vs WITH fak, same attack) | `go run ./cmd/guarddemo` | no — self-contained |
| **turntaxdemo** — turn-tax race (SOTA loop vs fak 1-shot) | `go run ./cmd/turntaxdemo` | no — self-contained |
| **ctxdemo** — multi-agent context-reuse proof (fak vs a tuned warm-cache SOTA baseline) | `go run ./cmd/ctxdemo` | optional (live race needs one) |
| **demorace** — reuse race (fak vs a tuned warm-cache SOTA baseline) + reuse curve | `go run ./cmd/demorace` | yes (live race) |

The first two are **self-contained** (no model, no GPU, no downloads): they replay a frozen,
class-labeled tool-call trace through the *real* kernel, so they reproduce identically on any
box. `guarddemo` is the fastest point to grasp — the moat in one side-by-side glance.

## 1. Local — one command

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak

# the self-contained ones — no model, no GPU, no downloads:
go run ./cmd/guarddemo             # → http://127.0.0.1:8151
#   pick a scenario → "Run both agents"  (WITHOUT fak vs WITH fak, side by side)
go run ./cmd/turntaxdemo            # → http://127.0.0.1:8150
#   pick a suite → "Replay through the kernel"

go run ./cmd/ctxdemo               # → http://127.0.0.1:8153
go run ./cmd/demorace             # → http://127.0.0.1:8147
```

Each server prints the URL it bound. On a shared/busy machine, cap the cores with
`-jobs 8` (absolute) or `-budget 0.75` (a fraction of the box) so the demo doesn't starve
other work.

## 2. Headless — exact accounting, no browser, no model

`ctxdemo` can print the precise, timing-free token accounting for every scenario without a
model or a server — useful in CI or to read the numbers directly:

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

`guarddemo` and `turntaxdemo` both ship a browserless `-selfcheck` that replays every
scenario through the *real* kernel (the same code path the browser drives) and asserts the
documented invariants — CI-usable, no browser, no network:

```bash
go run ./cmd/guarddemo  -selfcheck   # WITHOUT fak: 4 / 2 / 0 breaches · WITH fak: 0 (per scenario)
go run ./cmd/turntaxdemo -selfcheck   # turn-tax + safety-floor invariants per suite
```

All three self-contained comparisons ship a terminal side-by-side: the **30-second point
with zero setup** — rendered right in the terminal, no browser, no port (honors
`NO_COLOR`). One per fak value axis: `guarddemo` the **safety** axis, `turntaxdemo` the
**efficiency** axis, `ctxdemo` the **reuse** axis:

```bash
go run ./cmd/guarddemo  -print                          # safety: WITHOUT fak vs WITH fak (4 breaches → 0)
go run ./cmd/guarddemo  -print -scenario turntax-happy   # safety: the clean control (0 breaches)
go run ./cmd/turntaxdemo -print                          # efficiency: tuned SOTA vs fak (5 forced turns → 0)
go run ./cmd/turntaxdemo -print -suite turntax-happy     # efficiency: the anti-inflation control
go run ./cmd/ctxdemo     -bars                           # reuse: tokens re-read — cold vs warm-cache vs fak
```

Or play all three in one shot — **fak in 30 seconds**, then a built-in acceptance check
that each comparison still reproduces its documented headline (a cross-platform gate, no
model, no network):

```bash
bash tools/run_comparison_demos.sh        # play all three side-by-sides, then verify
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
PORT=8153 ./ctxdemo &
PORT=8147 ./demorace &
#  3. open YOUR_VM_IP:<port> in the firewall / security group for inbound TCP.
```

If your environment restricts which inbound ports are reachable (some org firewall
policies pin a fixed set), put the demos behind a reverse proxy on a port that *is* open.
The demos' pages call absolute API paths (`/api/...`), so when proxying under a path
prefix, rewrite those to the prefixed form (e.g. nginx `sub_filter "/api/" "/demo/api/"`)
and proxy the prefixed path back to the demo. SSE endpoints (`/api/race`, `/api/curve`)
need `proxy_buffering off` and a long `proxy_read_timeout`.

> **Note on our hosted copy.** The scripts that drive *our* specific GCP deployment are
> intentionally **not** in the repo — they embed our private project id and host details.
> Everything you need to stand up your *own* copy is above; nothing about it depends on our
> environment.

---

← Back to [the live demos](demos.html) · [the showcase](showcase.html) · [docs home](./)
