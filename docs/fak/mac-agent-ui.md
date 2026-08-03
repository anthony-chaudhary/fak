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

**Pick a model that fits first.** Unified memory is the constraint — the model has to fit
alongside everything else you are running. If you are unsure, start one tier smaller than you
think: proving the loop with a small model is far faster to debug than fighting swap with a
large one.

| Your Mac | Start with | Notes |
|---|---|---|
| 16 GB | a 7B model at Q4 (~4–5 GB) | Comfortable; good for proving the loop end to end. |
| 32 GB | a 14B model at Q4 (~9 GB) | The practical sweet spot for coding-shaped work. |
| 48 GB+ | a 27B model at Q4 (~17 GB) | The reference deployment on this page. |

**Get `fak` on the Mac** if it is not there yet — no Go toolchain needed:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak version
```

If that reports `fak: command not found`, the installer fell back to `~/.local/bin`, which is
not on the default macOS PATH. Add it (zsh is the macOS default shell):
`echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh`.

Then two processes on the Mac, in this order. Both are prerequisites for every command
below.

1. **A local model server** speaking the OpenAI API on loopback. Any of
   `llama-server` (llama.cpp), LM Studio, or Ollama works; you need its base URL
   (typically `http://127.0.0.1:8081/v1`) and the model id it serves. The reference
   deployment below uses `llama-server` with a Qwen3.6-27B GGUF, but nothing here is
   specific to that model. Give it a **32K-or-larger context window** — a coding agent's
   system prompt and tool surface consume a large share of it before your first message.

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

```bash
fak claude-mac-fak --metrics
# == /debug/vars ==   (indented JSON diagnostics)
# == /metrics ==      (Prometheus text, verbatim — pipe into promtool/grep)
```

`--metrics` never launches Claude. A 401 prints an actionable hint (set
`FAK_GATEWAY_KEY`, or run on the gateway host where these are loopback-exempt).

**Live overlay** — run this in a second pane next to the session; it polls
`/debug/vars` and prints one fak line per tick (Ctrl-C to stop):

```bash
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
page was written against, in the order you have to build it: the model server first,
then the gateway in front of it, then load and verify. Treat it as a worked example to
adapt, not a checklist you must match: the service names, model, and paths are ours.
What generalizes is the sizing, because the always-on Mac services must be big enough
for a real Claude Code first turn.

Two units and no third artifact:

| Unit | What it runs | Where its contents come from |
|---|---|---|
| `com.fak.qwen36-kernel` | `llama-server` on loopback, 32K context, all layers on the GPU, serving the id `qwen3.6-27b` | the command line `tools/qwen36_node_server.py --profile mac` computes |
| `com.fak.serve-gateway` | `fak serve` in front of it, with the long-turn timeouts and the Qwen3.6 request tuning in `EnvironmentVariables` | the launchd keys in `tools/com.fak.serve-gateway.plist` |

**Provenance, so you can check every line below.** The `llama-server` argv comes from
`tools/qwen36_node_server.py`; every launchd key comes from
`tools/com.fak.serve-gateway.plist`; the fill-and-load recipe is the one
`tools/install-mac-node.sh` uses for its own templates; `tools/fak-serve-caffeinate.sh`
is the sleep-prevention wrapper, unchanged. These are templates written against those
in-repo sources, not units this page re-ran on a Mac for you — after loading, believe
`launchctl list` and the log files rather than this page.

> **Label collision — settle this before you install.** `com.fak.serve-gateway` is also
> the label `fak node install` and `tools/install-mac-node.sh` use, for a **different**
> gateway: `fak serve --provider anthropic --base-url https://api.anthropic.com`, the
> adjudication proxy described in
> [Always-On Dogfood Server](always-on-dogfood-server.md). Neither installer can produce
> the local-model unit this page needs — both pin the Anthropic upstream, and
> `fak node install --help` offers only `--addr`, `--port`, `--remote`, `--key-env`,
> `--rotate-key`, and `--uninstall`, with no way to set `--base-url`, `--model`, or the
> environment variables below. One label means one loaded unit, so pick one: run
> `fak node install --uninstall` (or `./tools/install-mac-node.sh --uninstall`) first, or
> change both the `Label` value and the filename below and use that name for the rest of
> this section.

### 1. Let the repo compute the model-server command line

Do not hand-write the `llama-server` invocation — `tools/qwen36_node_server.py` already
encodes it. `--preflight` inspects the host and prints the exact command as JSON without
starting anything:

```bash
python3 tools/qwen36_node_server.py --preflight --profile mac --bind localhost \
  --extra-arg "--alias qwen3.6-27b"
```

Its `mac` profile is `--ctx-size 32768 --n-gpu-layers 99` — that is where this page's
"32K context, all layers on the GPU" sizing comes from — with
`lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M` as the default model and `8131` as the
default port. `--extra-arg "--alias qwen3.6-27b"` pins the id `/v1/models` reports so the
gateway's `--model` and the upstream agree on one name; `tools/glm52_serve.sh` pins
`--alias` for exactly that reason. The `command_line` field of that JSON is:

```text
llama-server -hf lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
  --host 127.0.0.1 --port 8131 --ctx-size 32768 --n-gpu-layers 99 --alias qwen3.6-27b
```

Drop `--preflight` to run it in the foreground and prove the model loads before you make
it a service. The JSON also reports the resolved `llama_server` path, which the plist
needs. Note the port: this reference deployment serves on `8131`, so the gateway's
`--base-url` below is `http://127.0.0.1:8131/v1`, not the generic `8081` named earlier on
this page — match whichever you actually run.

### 2. Create `com.fak.qwen36-kernel`

Run this from the repository root. It writes the same argv as above into a LaunchAgent.
The unquoted heredoc resolves the host-specific values inline — the same substitution
`tools/install-mac-node.sh` performs with `sed` over its `__PLACEHOLDER__` templates,
done in one step because there is no template file to fill here. Nothing else in the
block expands, so what you read is what lands on disk:

```bash
REPO="$(pwd)"; LOGDIR="$REPO/tools/_watchdog"
mkdir -p "$LOGDIR" ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.fak.qwen36-kernel.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.fak.qwen36-kernel</string>

    <key>ProgramArguments</key>
    <array>
      <string>$(command -v llama-server)</string>
      <string>-hf</string><string>lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M</string>
      <string>--host</string><string>127.0.0.1</string>
      <string>--port</string><string>8131</string>
      <string>--ctx-size</string><string>32768</string>
      <string>--n-gpu-layers</string><string>99</string>
      <string>--alias</string><string>qwen3.6-27b</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$REPO</string>

    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOGDIR/launchd_qwen36_kernel.log</string>
    <key>StandardErrorPath</key>
    <string>$LOGDIR/launchd_qwen36_kernel.err</string>
  </dict>
</plist>
PLIST
```

**Ours vs. yours.** The model id, the alias, and port `8131` are ours — change them
together and the gateway's `--model` and `--base-url` in step 3 must follow. The
`llama-server` path and `$REPO` are resolved from your machine. `KeepAlive` and
`RunAtLoad` are the 24/7 daemon pair `tools/com.fak.serve-gateway.plist` uses; the
explicit `StandardOutPath` / `StandardErrorPath` are what make step 4's `tail` possible.

### 3. Create `com.fak.serve-gateway`

Same shell, same variables. This is `tools/com.fak.serve-gateway.plist` with the upstream
repointed at the local model server and the long-turn settings added to the
`EnvironmentVariables` dict that template already uses for `FAK_AUDIT_JOURNAL`:

```bash
cat > ~/Library/LaunchAgents/com.fak.serve-gateway.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.fak.serve-gateway</string>

    <key>ProgramArguments</key>
    <array>
      <string>$REPO/tools/fak-serve-caffeinate.sh</string>
      <string>$(command -v fak)</string>
      <string>serve</string>
      <string>--addr</string><string>127.0.0.1:8080</string>
      <string>--base-url</string><string>http://127.0.0.1:8131/v1</string>
      <string>--model</string><string>qwen3.6-27b</string>
      <string>--policy</string><string>$REPO/examples/dev-agent-policy.json</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
      <key>FAK_AUDIT_JOURNAL</key>
      <string>$LOGDIR/serve_audit.jsonl</string>
      <key>FAK_PLANNER_TIMEOUT_S</key>
      <string>1800</string>
      <key>FAK_HTTP_WRITE_TIMEOUT_S</key>
      <string>1800</string>
      <key>FAK_PROVIDER_EXTRA_BODY_JSON</key>
      <string>{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}</string>
    </dict>

    <key>WorkingDirectory</key>
    <string>$REPO</string>

    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOGDIR/launchd_serve.log</string>
    <key>StandardErrorPath</key>
    <string>$LOGDIR/launchd_serve.err</string>
  </dict>
</plist>
PLIST
```

`tools/fak-serve-caffeinate.sh` is the first `ProgramArguments` entry for two reasons its
own header explains: it holds macOS idle- and system-sleep assertions while the gateway
runs, and it `exec`s `fak serve` so launchd's direct child stays the real process — which
is what keeps `KeepAlive` tracking the right PID and the two log paths non-empty.

**There is no `fak-mac-serve-gateway` wrapper script.** Earlier revisions of this page
named `~/.local/bin/fak-mac-serve-gateway` as the thing that exports
`FAK_PROVIDER_EXTRA_BODY_JSON`; no such script exists in this repo and none is needed,
because launchd's `EnvironmentVariables` dict sets it directly, above. If you have a
wrapper of your own in the `ProgramArguments` chain, either place works — pick one so
there is a single source for the value.

**Driving from another machine** needs a non-loopback bind and a bearer, which
`tools/install-mac-node.sh --bind-all` handles by appending two `ProgramArguments`
strings and one environment entry. The equivalent by hand, before you load the unit:

```bash
KEY="$(openssl rand -hex 32)"          # same generator install-mac-node.sh uses
printf '%s' "$KEY" > ~/.fak-gateway-key && chmod 600 ~/.fak-gateway-key
```

Then change `--addr` to `0.0.0.0:8080`, append `--require-key-env` and
`FAK_GATEWAY_KEY` to `ProgramArguments`, and add a `FAK_GATEWAY_KEY` key to
`EnvironmentVariables` with `$KEY` as its value. Two consumers, one secret: the gateway
reads it from the launchd environment, and `fak mac` reads `~/.fak-gateway-key` over SSH
when the client's `FAK_GATEWAY_KEY` is empty. Nothing in the repo creates
`~/.fak-gateway-key` for you — that `printf` is the step that makes the SSH fetch in
[One-time shell setup](#one-time-shell-setup) resolve. Both that file and the plist under
`~/Library/LaunchAgents` now hold the bearer, so treat both as secrets. Never leave a
non-loopback `--addr` without `--require-key-env`: that is an unauthenticated kernel
reachable off-host, and both `fak guard` and `fak serve` warn about it.

### 4. Load, then verify

```bash
launchctl load -w ~/Library/LaunchAgents/com.fak.qwen36-kernel.plist
launchctl load -w ~/Library/LaunchAgents/com.fak.serve-gateway.plist

launchctl list | grep com.fak                 # both labels present
curl -sf http://127.0.0.1:8131/v1/models      # the model server, id qwen3.6-27b
curl -sf http://127.0.0.1:8080/healthz        # the gateway in front of it
```

A 27B model takes a while to load on first start; if `/v1/models` is not answering yet,
watch it come up:

```bash
tail -f tools/_watchdog/launchd_qwen36_kernel.log
tail -f tools/_watchdog/launchd_serve.log
```

For Qwen3.6, the preflight panel should then show `request tuning: provider extra body
set (keys: chat_template_kwargs, top_k)`. If it prints a Qwen3.6 tuning warning, the
`FAK_PROVIDER_EXTRA_BODY_JSON` entry did not reach the gateway process — fix it in the
plist and reload before running the Claude Code probe.

### 5. Reload after editing either unit

`launchctl load -w` / `unload -w` is the pair `tools/install-mac-node.sh` and
`fak node install` use, and it is the one to use if you created the units with the
`load -w` lines above. The bootstrap-domain form does the same job:

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

> **Driving from Windows?** Every block below is bash/zsh, because the usual place to
> run them is the Mac itself. In PowerShell, read `$VAR` as `$env:VAR`, replace the
> trailing `\` line continuations with backticks, and use `curl.exe` rather than the
> `curl` alias.

## Verify the gateway

```bash
curl -sS -H "Authorization: Bearer $FAK_GATEWAY_KEY" "$FAK_MAC_GATEWAY/healthz"
curl -sS -H "Authorization: Bearer $FAK_GATEWAY_KEY" "$FAK_MAC_GATEWAY/v1/models"
```

## Dry-run the UI launch

```bash
fak console agent \
  --claude-config-dir "$FAK_CLAUDE_CONFIG_DIR" \
  --gateway-url "$FAK_MAC_GATEWAY" \
  --gateway-key-env FAK_GATEWAY_KEY \
  --model "$FAK_MAC_MODEL" \
  --prompt "Reply with exactly: OK" \
  --dry-run
```

The dry-run should show `provider=existing-fak-gateway`, `auth=gateway-bearer`,
`ANTHROPIC_BASE_URL=$FAK_MAC_GATEWAY`, a redacted `ANTHROPIC_API_KEY`,
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, and `API_TIMEOUT_MS=1800000`.

## Run a probe

```bash
fak console agent \
  --claude-config-dir "$FAK_CLAUDE_CONFIG_DIR" \
  --gateway-url "$FAK_MAC_GATEWAY" \
  --gateway-key-env FAK_GATEWAY_KEY \
  --model "$FAK_MAC_MODEL" \
  --prompt "Reply with exactly: OK" \
  -- --output-format json
```

The first full Claude Code turn can take 10-15 minutes on the local Mac model.
A healthy run returns JSON with `"is_error": false`, `"result": "OK"`, and a low
`ttft_stream_ms` value; the total `duration_ms` is the model prefill time.

For an interactive session, omit `--prompt`:

```bash
fak console agent \
  --claude-config-dir "$FAK_CLAUDE_CONFIG_DIR" \
  --gateway-url "$FAK_MAC_GATEWAY" \
  --gateway-key-env FAK_GATEWAY_KEY \
  --model "$FAK_MAC_MODEL"
```

## Inspect served sessions

```bash
fak console sessions \
  --addr "$FAK_MAC_GATEWAY" \
  --key "$FAK_GATEWAY_KEY"
```

This is the repeatable check that the UI is pointed at the same always-on gateway
instead of a one-off local process.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `fak: command not found` after install | The installer fell back to `~/.local/bin`, which is not on the default macOS PATH — see [Stand up the Mac side](#stand-up-the-mac-side). |
| An SSH error naming a host you never configured | `FAK_GATEWAY_KEY` was empty, so the launcher tried to fetch the bearer from a remote gateway host over SSH. Export the key, or pass `--fetch-key=false` when the gateway is local. |
| The preflight panel aborts the launch | The gateway is not reachable at `FAK_MAC_GATEWAY`. Check `curl -s $FAK_MAC_GATEWAY/healthz` — it is unauthenticated, so it answers without a bearer. |
| Panel shows `planner(live)=mock` | You are pointed at a gateway with no real model behind it; responses would be scripted. Confirm `--base-url` on the `fak serve` side. |
| `401` from `/metrics` or `/v1/models` | The bearer is missing. These are loopback-exempt from the gateway host itself but not from another machine — use `fak mac --metrics`, which reuses the key the launcher already loaded. |
| First interactive turn seems hung | Expected on a large local model — prefill is slow and is not streamed. Run `fak mac --overlay` in a second pane to confirm it is progressing. |
| `address already in use` on `fak serve` | Another process holds the port; pick a different `--addr`. |

## See also

- [Always-On Dogfood Server](always-on-dogfood-server.md) — how the `node-macos-a`
  gateway this UI targets is stood up, measured, and kill-switched.
- [Server quickstart](server-quickstart.md) — starting your own `fak serve` endpoint
  from scratch if you don't have an always-on gateway yet.
- [Qwen3.6 Claude dogfood playbook](../qwen36-claude-dogfood-playbook.md) — the
  model-server layer on its own: bringing a Qwen3.6 endpoint up by hand, checking
  `/v1/models` and `/v1/chat/completions`, and where the `FAK_PROVIDER_EXTRA_BODY_JSON`
  tuning comes from.
