---
title: "Use the always-on Mac gateway from the fak UI"
description: "Repeatable operator steps for launching Claude Code from fak console against the node-macos-a fak serve gateway."
---

# Mac Agent UI

Use this when `node-macos-a` is already running the always-on Qwen3.6 stack:

```text
Claude Code <- fak console agent -> http://node-macos-a.local:8080
                                    fak serve -> llama-server :8131
```

The UI surface is `fak console agent`. With `--gateway-url`, it does not start a
second local guard; it launches Claude Code directly against the existing `fak
serve` gateway and reads the bearer from an environment variable.

## One-command test

From the repo root, this opens interactive Claude Code against the Mac gateway:

```powershell
go run ./cmd/fak claude-mac-fak
```

If `FAK_GATEWAY_KEY` is empty, the command fetches the gateway bearer from
`user@node-macos-a.local:~/.fak-gateway-key` over SSH using
`~/.ssh/id_ed25519_prod_to_laptop`. Override that host with `FAK_MAC_SSH_HOST`.
It then runs the same `fak console agent` gateway launcher with an isolated
Claude config dir.

Useful variants:

```powershell
go run ./cmd/fak claude-mac-fak --dry-run
go run ./cmd/fak claude-mac-fak --probe
go run ./cmd/fak claude-mac-fak --probe --prompt "Reply with exactly: OK"
```

With an installed `fak` binary, the same commands shorten to:

```powershell
fak claude-mac-fak
fak claude-mac-fak --probe
```

## See what fak is doing

Once interactive Claude Code starts it owns the terminal, so fak is otherwise
invisible behind the `ANTHROPIC_BASE_URL` it set. Two surfaces fix that.

**Preflight debug panel** (on by default for the interactive launch). Before
handing the terminal to Claude Code, `claude-mac-fak` probes the gateway
(`/healthz` + `/debug/vars`) and prints what fak is about to do:

```text
fak debug · gateway http://node-macos-a.local:8080
health: ok  engine(build)=metal  planner(live)=inkernel
vdso=on  cache-hit 0.88  inflight 0  up 3h12m
model qwen3.6-27b  auth gateway-bearer
metrics: run  fak claude-mac-fak --metrics   (fetches /metrics + /debug/vars with the gateway's own bearer)
  urls: http://node-macos-a.local:8080/metrics · …/debug/vars  (open on the gateway host; off-box needs the bearer)
-> launching claude ...
```

It proves the gateway is the live in-kernel `fak serve` (a `planner=mock` line
would mean scripted, non-model responses) and aborts the interactive launch
instead of starting Claude against an unreachable gateway. Pass `--debug=false`
to skip it.

**Read the metrics without token wrangling.** `/metrics` and `/debug/vars` are
loopback-exempt: they open without a bearer from the gateway host itself, but a
bare browser click from your laptop hits the remote IP and 401s. Rather than
hand-build a `curl` with the header, `--metrics` reuses the bearer the launcher
already loaded to fetch both surfaces and print them (the token is sent, never
printed):

```powershell
fak claude-mac-fak --metrics
# == /debug/vars ==   (indented JSON diagnostics)
# == /metrics ==      (Prometheus text, verbatim — pipe into promtool/grep)
```

`--metrics` never launches Claude. A 401 prints an actionable hint (set
`FAK_GATEWAY_KEY`, or run on the gateway host where these are loopback-exempt).

**Live overlay** — run this in a second pane next to the session; it polls
`/debug/vars` and prints one fak line per tick (Ctrl-C to stop):

```powershell
go run ./cmd/fak claude-mac-fak --overlay
# submits 1240  hits 1101 (88.8%)  engine 139  inflight 1  heap 412.0M  gor 47
```

`--overlay-interval 5s` changes the refresh rate; `--overlay` never launches
Claude.

## Watch fak from Grafana

The repo ships a Prometheus + Grafana stack at
[`tools/grafana/`](../../tools/grafana/README.md) that already scrapes a
`fak serve` gateway's `/metrics`. To point it at the Mac gateway, set the
`fak_gateway` job target in `tools/grafana/prometheus.yml` to the tailnet host
(e.g. `node-macos-a.local:8080`) instead of localhost, then:

```bash
tools/grafana/up.sh      # http://localhost:3000 (the --grafana-url default)
```

If the Mac gateway runs with `--require-key-env`, add Prometheus bearer auth for
that job (see the note already in `tools/grafana/prometheus.yml`). The Grafana URL
shown in the preflight panel comes from `--grafana-url` / `FAK_MAC_GRAFANA`.

## Mac service prerequisites

The always-on Mac services must be sized for a real Claude Code first turn:

- `com.fak.qwen36-model` runs `llama-server` with `--ctx-size 65536`.
- `com.fak.serve-gateway` exports `FAK_PLANNER_TIMEOUT_S=1800` and
  `FAK_HTTP_WRITE_TIMEOUT_S=1800`.
- `~/.local/bin/fak-mac-serve-gateway` exports
  `FAK_PROVIDER_EXTRA_BODY_JSON='{ "top_k": 20, "chat_template_kwargs": { "enable_thinking": false } }'`.

Reload launchd after changing either LaunchAgent:

```bash
launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.fak.qwen36-model.plist 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.fak.qwen36-model.plist
launchctl kickstart -k "gui/$(id -u)/com.fak.serve-gateway"
```

## One-time shell setup

PowerShell from the Windows driver:

```powershell
$env:FAK_MAC_GATEWAY = "http://node-macos-a.local:8080"
$env:FAK_MAC_SSH_HOST = "user@node-macos-a.local"
$env:FAK_GATEWAY_KEY = ssh -i $env:USERPROFILE\.ssh\id_ed25519_prod_to_laptop $env:FAK_MAC_SSH_HOST 'cat ~/.fak-gateway-key'
$env:FAK_MAC_MODEL = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
$env:FAK_CLAUDE_CONFIG_DIR = Join-Path $env:TEMP "fak-claude-ui-probe"
New-Item -ItemType Directory -Force -Path $env:FAK_CLAUDE_CONFIG_DIR | Out-Null
```

Bash/zsh from a tailnet machine:

```bash
export FAK_MAC_GATEWAY="http://node-macos-a.local:8080"
export FAK_MAC_SSH_HOST="user@node-macos-a.local"
export FAK_GATEWAY_KEY="$(ssh "$FAK_MAC_SSH_HOST" 'cat ~/.fak-gateway-key')"
export FAK_MAC_MODEL="lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
export FAK_CLAUDE_CONFIG_DIR="${TMPDIR:-/tmp}/fak-claude-ui-probe"
mkdir -p "$FAK_CLAUDE_CONFIG_DIR"
```

## Verify the gateway

```powershell
curl.exe -sS -H "Authorization: Bearer $env:FAK_GATEWAY_KEY" "$env:FAK_MAC_GATEWAY/healthz"
curl.exe -sS -H "Authorization: Bearer $env:FAK_GATEWAY_KEY" "$env:FAK_MAC_GATEWAY/v1/models"
```

## Dry-run the UI launch

```powershell
go run ./cmd/fak console agent `
  --claude-config-dir $env:FAK_CLAUDE_CONFIG_DIR `
  --gateway-url $env:FAK_MAC_GATEWAY `
  --gateway-key-env FAK_GATEWAY_KEY `
  --model $env:FAK_MAC_MODEL `
  --prompt "Reply with exactly: OK" `
  --dry-run
```

The dry-run should show `provider=existing-fak-gateway`, `auth=gateway-bearer`,
`ANTHROPIC_BASE_URL=$env:FAK_MAC_GATEWAY`, a redacted `ANTHROPIC_API_KEY`,
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, and `API_TIMEOUT_MS=1800000`.

## Run a probe

```powershell
go run ./cmd/fak console agent `
  --claude-config-dir $env:FAK_CLAUDE_CONFIG_DIR `
  --gateway-url $env:FAK_MAC_GATEWAY `
  --gateway-key-env FAK_GATEWAY_KEY `
  --model $env:FAK_MAC_MODEL `
  --prompt "Reply with exactly: OK" `
  -- --output-format json
```

The first full Claude Code turn can take 10-15 minutes on the local Mac model.
A healthy run returns JSON with `"is_error": false`, `"result": "OK"`, and a low
`ttft_stream_ms` value; the total `duration_ms` is the model prefill time.

For an interactive session, omit `--prompt`:

```powershell
go run ./cmd/fak console agent `
  --claude-config-dir $env:FAK_CLAUDE_CONFIG_DIR `
  --gateway-url $env:FAK_MAC_GATEWAY `
  --gateway-key-env FAK_GATEWAY_KEY `
  --model $env:FAK_MAC_MODEL
```

## Inspect served sessions

```powershell
go run ./cmd/fak console sessions `
  --addr $env:FAK_MAC_GATEWAY `
  --key $env:FAK_GATEWAY_KEY
```

This is the repeatable check that the UI is pointed at the same always-on gateway
instead of a one-off local process.
