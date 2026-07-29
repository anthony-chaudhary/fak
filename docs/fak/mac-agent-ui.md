---
title: "Use the always-on Mac gateway from the fak UI"
description: "Launch Claude Code against a Mac running your own local model behind fak serve: what to stand up first, then the repeatable operator steps."
---

# Mac Agent UI

**The showcase, in one line:** point Claude Code at your own Mac's local open model
through a single `fak` binary, and watch fak prove the gateway is live before it hands
over the terminal. A premium cloud agent, open weights running on your own silicon, one
static binary in between. It is the whole fak thesis, runnable on a laptop.

> **`fak mac` is a launcher, not an installer.** It points Claude Code at a Mac
> gateway that is **already running**; it does not create one. If you have not stood
> that gateway up yet, start at [Stand up the Mac side](#stand-up-the-mac-side) —
> running `fak mac` first will just refuse and send you back here.

> **Audience.** Anyone driving Claude Code against a Mac `fak serve` gateway — the
> Mac itself or another machine on the same network. By the end you can open
> interactive Claude Code against your Mac with one command.

Being honest about it: the fast headless probe is a smoke test, not a full
agent session. `--probe` launches Claude Code in safe mode with tools, skills, and
session persistence disabled, so the 2026-07-04 Mac witness sent 1,889 input
tokens and returned `pong` in 13.1 seconds through the live gateway. A full
interactive first turn can still be much slower — expect 10–15 minutes of prefill
on an M3 Pro — because Claude Code includes the normal agent context and tool
surface, and the session is single-stream. The loop stays observable the entire
time, through the preflight panel, `--overlay`, and `--metrics` below.

The shape you are building:

```text
Claude Code <- fak console agent -> http://<your-mac>:8080
                                    fak serve -> local OpenAI-compatible model server
```

The UI surface is `fak console agent`. With `--gateway-url`, it does not start a
second local guard; it launches Claude Code directly against the existing `fak
serve` gateway and reads the bearer from an environment variable.

Throughout this page, `<your-mac>` is the hostname or IP of your Mac and `<you>` is
your username on it. The values compiled into `fak` as defaults
(`node-macos-a.local`, `user@node-macos-a.local`) are deliberately non-resolving
placeholders, not hosts you are expected to have — see
[scrubbing real values](scrubbing-real-values.md).

## Stand up the Mac side

Two processes on the Mac, in this order. Both are prerequisites for every command
below.

1. **A local model server** speaking the OpenAI API on loopback. Any of
   `llama-server` (llama.cpp), LM Studio, or Ollama works; you need its base URL
   (typically `http://127.0.0.1:8081/v1`) and the model id it serves. The reference
   deployment below uses `llama-server` with a Qwen3.6-27B GGUF, but nothing here is
   specific to that model.

2. **`fak serve` in front of it**, bound to an address the driving machine can
   reach — `0.0.0.0`, not loopback, if you are driving from another machine:

   ```bash
   fak serve --addr 0.0.0.0:8080 \
     --base-url http://127.0.0.1:8081/v1 \
     --model qwen3.6-27b \
     --policy examples/dev-agent-policy.json
   ```

   Full options, including how to have `fak serve` load GGUF weights itself with no
   separate model server, are in the [server quickstart](server-quickstart.md).

A local first turn is slow, so raise the timeouts before a real session:
`FAK_PLANNER_TIMEOUT_S=1800` and `FAK_HTTP_WRITE_TIMEOUT_S=1800`. To keep both
processes running across reboots as LaunchAgents, see
[Mac service prerequisites](#mac-service-prerequisites).

**Bearer token.** If you started `fak serve` with `--require-key-env`, it needs a
bearer. `fak mac` reads it from `FAK_GATEWAY_KEY`, and when that is empty it fetches
`~/.fak-gateway-key` from the Mac over SSH. Either set `FAK_GATEWAY_KEY` yourself, or
set `FAK_MAC_SSH_HOST=<you>@<your-mac>` so the fetch has somewhere to go. A gateway
started without `--require-key-env` needs no token at all.

## One-command test

Point `fak` at the gateway you just started, then launch. `fak mac` is the crisp
handle; `fak claude-mac-fak` is the equivalent long form — both route to the same
launcher, byte-for-byte:

```bash
export FAK_MAC_GATEWAY="http://<your-mac>:8080"
export FAK_MAC_SSH_HOST="<you>@<your-mac>"   # only if the gateway requires a bearer
fak mac
```

This works from the Mac itself or from any machine that can reach the gateway —
`fak mac` targets it over the network either way.

The SSH fetch is non-interactive and bounded (`BatchMode=yes`, `ConnectTimeout=5`,
one attempt), so an unreachable Mac fails quickly with the SSH reason instead of
hanging before the preflight panel. `fak mac` then runs the same `fak console agent`
gateway launcher with an isolated Claude config dir.

Useful variants:

```bash
fak mac --dry-run
fak mac --probe
fak mac --probe --prompt "Reply with exactly: OK"
```

Working from a clone instead of an installed binary? Every `fak <verb>` below is
`go run ./cmd/fak <verb>` from the repository root.

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
request tuning: provider extra body set (keys: chat_template_kwargs, top_k)
metrics: run  fak claude-mac-fak --metrics   (fetches /metrics + /debug/vars with the gateway's own bearer)
  urls: http://node-macos-a.local:8080/metrics · …/debug/vars  (open on the gateway host; off-box needs the bearer)
-> launching claude ...
```

It proves the gateway is the live in-kernel `fak serve` (a `planner=mock` line
would mean scripted, non-model responses) and aborts the interactive launch
instead of starting Claude against an unreachable gateway. Pass `--debug=false`
to skip it.

Headless `--probe` runs the same reachability gate before launching Claude Code,
but keeps stdout quiet on success so Claude's JSON output stays parseable. Probe
mode also defaults Claude Code to `--output-format json`, `--safe-mode`,
`--tools ""`, `--disable-slash-commands`, and `--no-session-persistence`; pass
explicit Claude args after `--` to override those defaults. If the gateway is
down, the probe exits before Claude Code starts.

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
fak mac --overlay
# submits 1240  hits 1101 (88.8%)  engine 139  inflight 1  heap 412.0M  gor 47
```

`--overlay-interval 5s` changes the refresh rate; `--overlay` never launches
Claude.

## Watch fak from Grafana

The repo ships a Prometheus + Grafana stack at
[`tools/grafana/`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/README.md) that already scrapes a
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

This section is the **reference deployment** — the always-on LaunchAgent setup this
page was written against. Treat it as a worked example to adapt, not a checklist you
must match: the service names, model, and paths are ours. What generalizes is the
sizing, because the always-on Mac services must be big enough for a real Claude Code
first turn:

- `com.fak.qwen36-kernel` runs `llama-server` with a 32K-or-larger context
  window, the `qwen3.6-27b` alias, Metal enabled, and the OpenAI-compatible API
  on loopback for the public gateway to proxy.
- `com.fak.serve-gateway` exports `FAK_PLANNER_TIMEOUT_S=1800` and
  `FAK_HTTP_WRITE_TIMEOUT_S=1800`.
- `~/.local/bin/fak-mac-serve-gateway` exports
  `FAK_PROVIDER_EXTRA_BODY_JSON='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'`.

For Qwen3.6, the preflight panel should show `request tuning: provider extra body
set (keys: chat_template_kwargs, top_k)`. If it prints a Qwen3.6 tuning warning,
restart the gateway with the `FAK_PROVIDER_EXTRA_BODY_JSON` value above before
running the Claude Code probe.

Reload launchd after changing either LaunchAgent:

```bash
launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.fak.qwen36-kernel.plist 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.fak.qwen36-kernel.plist
launchctl kickstart -k "gui/$(id -u)/com.fak.serve-gateway"
```

## One-time shell setup

Substitute your own host, user, and model id. `FAK_MAC_MODEL` must be the model id
your `fak serve` gateway advertises — `curl $FAK_MAC_GATEWAY/v1/models` lists them.
`-i <ssh-key>` is only needed if the Mac is not reachable with your default SSH
identity.

Bash/zsh:

```bash
export FAK_MAC_GATEWAY="http://<your-mac>:8080"
export FAK_MAC_SSH_HOST="<you>@<your-mac>"
export FAK_GATEWAY_KEY="$(ssh "$FAK_MAC_SSH_HOST" 'cat ~/.fak-gateway-key')"
export FAK_MAC_MODEL="qwen3.6-27b"
export FAK_CLAUDE_CONFIG_DIR="${TMPDIR:-/tmp}/fak-claude-ui-probe"
mkdir -p "$FAK_CLAUDE_CONFIG_DIR"
```

PowerShell, driving the Mac from Windows:

```powershell
$env:FAK_MAC_GATEWAY = "http://<your-mac>:8080"
$env:FAK_MAC_SSH_HOST = "<you>@<your-mac>"
$env:FAK_GATEWAY_KEY = ssh $env:FAK_MAC_SSH_HOST 'cat ~/.fak-gateway-key'
$env:FAK_MAC_MODEL = "qwen3.6-27b"
$env:FAK_CLAUDE_CONFIG_DIR = Join-Path $env:TEMP "fak-claude-ui-probe"
New-Item -ItemType Directory -Force -Path $env:FAK_CLAUDE_CONFIG_DIR | Out-Null
```

If the gateway was started without `--require-key-env`, skip `FAK_GATEWAY_KEY` and
`FAK_MAC_SSH_HOST` entirely — there is no bearer to fetch.

## Verify the gateway

```powershell
curl.exe -sS -H "Authorization: Bearer $env:FAK_GATEWAY_KEY" "$env:FAK_MAC_GATEWAY/healthz"
curl.exe -sS -H "Authorization: Bearer $env:FAK_GATEWAY_KEY" "$env:FAK_MAC_GATEWAY/v1/models"
```

## Dry-run the UI launch

```powershell
fak console agent `
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
fak console agent `
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
fak console agent `
  --claude-config-dir $env:FAK_CLAUDE_CONFIG_DIR `
  --gateway-url $env:FAK_MAC_GATEWAY `
  --gateway-key-env FAK_GATEWAY_KEY `
  --model $env:FAK_MAC_MODEL
```

## Inspect served sessions

```powershell
fak console sessions `
  --addr $env:FAK_MAC_GATEWAY `
  --key $env:FAK_GATEWAY_KEY
```

This is the repeatable check that the UI is pointed at the same always-on gateway
instead of a one-off local process.

## See also

- [Always-On Dogfood Server](always-on-dogfood-server.md) — how the `node-macos-a`
  gateway this UI targets is stood up, measured, and kill-switched.
- [Server quickstart](server-quickstart.md) — starting your own `fak serve` endpoint
  from scratch if you don't have an always-on gateway yet.
