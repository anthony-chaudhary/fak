---
title: "fak + OpenAI Codex: MCP first, OpenAI-compatible proxy when the wire fits"
description: "Use fak with OpenAI Codex and OpenAI-compatible coding agents. Current Codex CLI/IDE users should start with fak as an MCP server; OpenAI SDKs and Chat Completions clients can repoint their base URL at fak serve."
---

# fak + OpenAI Codex

fak puts a structural policy gate in front of Codex tool use.

> TL;DR: Use `fak serve --stdio` as an MCP server for current Codex CLI and IDE sessions.

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
```

## Fastest path

Codex is OpenAI's coding agent for software development. Its current surfaces include
the Codex CLI, IDE extension, Codex app, and cloud tasks. This guide keeps those surfaces
separate from the generic OpenAI-compatible API path.

There are two useful fak entry points:

| If you run... | Use this fak path | Why |
|---|---|---|
| Current Codex CLI or IDE extension | `fak serve --stdio` as an MCP server | Codex supports MCP, and fak exposes verdict tools without changing Codex's model wire. |
| Codex CLI with an OpenAI API key and you want fak in front of the model wire | `fak codex -- <codex args...>` | One command starts `fak manage`, launches Codex, and injects per-run Codex `-c model_provider=fak` overrides for the Responses wire. |
| OpenAI SDKs, OpenAI Agents SDK, LangChain, LlamaIndex, or any Chat Completions client | `fak serve` as an OpenAI-compatible gateway | The client already calls `/v1/chat/completions`, so you repoint its base URL to fak. Endpoint-by-endpoint compatibility and current limits: [openai.md](openai.md). |

Honest wire boundary: current Codex model-provider docs are Responses-oriented. fak can
proxy to an OpenAI Responses upstream with `--provider openai-responses`. The public
gateway clients hit today are `/v1/chat/completions`, `/v1/responses`, `/v1/messages`,
`/mcp`, and `/v1/fak/*`. fak now exposes a client-facing **`/v1/responses`** inbound
route (#925): a Responses-API agent repoints its OpenAI base URL at fak and every
proposed tool call crosses the kernel's capability floor, the same as the chat wire.
It is **buffered** — a `stream:true` request is refused with a 400, so a client that
needs SSE should use MCP. For current Codex CLI/IDE sessions either path works; for
OpenAI-compatible SDKs and Chat Completions agents, use the base-URL proxy path below.

## Why this matters to Codex

Codex reads `AGENTS.md` before it works in this repo. The repo-level rules already tell
it the build, test, commit, and guardrail contract. fak adds a second layer: the kernel can
adjudicate proposed tool calls and tool results with a default-deny floor that a prompt
cannot talk around.

Use the right path for the job:

- MCP path: Codex keeps its normal model/auth path and gains fak's adjudication tools.
- Proxy path: an OpenAI-compatible client sends chat/tool traffic through fak before the
  upstream model sees it.
- Offline proof: run the preflight commands before any key, model, or GPU is involved.

Do not treat "Codex can reach `/v1/responses`" as the whole integration contract. The
guarded model-wire path also has to match Codex's host-tool dialect: exact tool names
such as `update_plan`, namespace-qualified spellings such as `functions.update_plan`,
dangerous argument fields, and Codex's behavior after a denied tool. The acceptance gate
for that surface is the
[harness integration acceptance checklist](harness-acceptance-checklist.md); the failure
mode it prevents is recorded in the
[2026-07-03 tool-dialect postmortem](../notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md).

## 60-second proof before wiring Codex

From the repository root:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

Expected shape:

- `refund_payment` is denied with `POLICY_BLOCK`.
- `search_kb` is allowed.
- `fak agent --offline` blocks the injected/destructive path while the task still books.

That proves the capability floor is structural, not a model judgment.

## Install an immutable MCP server (recommended)

Codex keeps its MCP child executable open for the life of a session. Pointing it at the
mutable worktree binary can therefore block upgrades on Windows and can silently separate
the source revision from the process Codex is actually running. Install a content-addressed,
self-contained copy instead:

```powershell
fak codex mcp install --policy C:\absolute\path\to\policy.json
fak codex mcp status
```

`install` copies the binary and policy outside the worktree, records the module revision and
SHA-256 checksum, atomically patches only `[mcp_servers.fak]`, backs up the previous entry,
and performs a bounded MCP `initialize` handshake. Each upgrade gets a new filename, so an
old Codex session may keep the previous binary open while new sessions select the new copy.
Restart or reload Codex after install/upgrade; existing sessions do not reread `config.toml`.

Useful recovery operations:

```powershell
fak codex mcp install --rollback   # restore the prior known-good entry
fak codex mcp uninstall            # remove only installer-owned config/artifacts
```

`status` reports checksum/config drift, missing or wrong-architecture artifacts, stale source
revisions, and initialize-probe failures. It never prints credentials. Use `--config` to test
against an isolated Codex config fixture; all policy and command paths written to TOML are
absolute.
## Path 1: Current Codex CLI or IDE extension via MCP

Build the binary:

```bash
go build -o fak ./cmd/fak
```

### One-command guarded Codex launcher

For Codex CLI sessions where you have an `OPENAI_API_KEY` available and want fak to
mediate Codex's model wire directly, use the launcher:

```bash
./fak codex --dry-run --split off -- exec --json "List active MCP servers only."
./fak codex -- exec --json "Summarize AGENTS.md."
```

The dry-run should print a command shaped like:

```text
fak manage --split off ... -- codex --dangerously-bypass-approvals-and-sandbox exec --json ...
```

### UserPromptSubmit modes

The capability floor is explicit: the checked-in `.codex/hooks.json` and the default installer are permissive, while guarded children and operators who deliberately select hardened mode receive submit-time enforcement. Codex still
evaluates the manifest's tiny shell selector, but an ordinary direct prompt starts no
`fak` process, inspects no session transcript, emits no context, and returns no block
envelope. The selector invokes `fak` only for a child marked `FAK_GUARD_ACTIVE=1` or an
explicit hardened environment; that scoped child receives the one-shot
guarded-workflow context used by fak orchestration.

Install or refresh that default declaration with:

```bash
fak sessions codex-hook-install
```

Operators who intentionally want cross-instance submit enforcement can choose hardened
mode in either of two reversible ways:

```bash
# Persist hardened behavior in $CODEX_HOME/hooks.json.
fak sessions codex-hook-install --hardened

# Harden only Codex processes launched from this shell.
FAK_CODEX_SUBMIT_HARDENED=1 codex
```

The direct runtime surface is `fak sessions codex-loop-hook --hardened`. In hardened
mode, an unguarded direct provider is diagnosed and blocked before the next model turn;
if that bounded diagnosis times out, the hook returns a typed hardened-timeout block
instead of allowing a turn that a later audit row could misrepresent.
`--allow-direct` or `FAK_ALLOW_DIRECT_CODEX_CONTINUE=1` remains the intentional override.
Running `fak sessions codex-hook-install` again removes installer-baked hardening and
restores the permissive default. Unsetting `FAK_CODEX_SUBMIT_HARDENED` rolls back the
shell-scoped form.

### Live hardened-hook contract witness

Codex CLI 0.144.1 was exercised end to end with the reviewed repo-level
`.codex/hooks.json`. The CLI discovered the manifest, invoked `UserPromptSubmit` with
`session_id`, and honored the `{"decision":"block","reason":...}` response before
model execution (zero input/output tokens). The privacy-preserving captured payload and
normalized event evidence are committed at
[`experiments/agent-live/codex-continuation-hook-live-witness-2026-07-11.json`](../../experiments/agent-live/codex-continuation-hook-live-witness-2026-07-11.json).
That artifact witnesses the block-envelope contract now selected by hardened mode; it
is not the shipped default posture.

### Hardened recovery and intentional direct sessions

If guard reports that Codex subscription authentication is missing, repair it through
guard. These exact authentication-management commands bypass only the provider repoint and
the subscription credential probe; ordinary and model-bearing Codex commands remain guarded.

```powershell
$env:CODEX_HOME = 'C:\Users\USER\.codex-codexFOUR'
fak guard -- codex login
fak guard -- codex login status
fak guard -- codex
```

```bash
CODEX_HOME="$HOME/.codex-codexFOUR" fak guard -- codex login
CODEX_HOME="$HOME/.codex-codexFOUR" fak guard -- codex login status
CODEX_HOME="$HOME/.codex-codexFOUR" fak guard -- codex
```

The expected status after successful ChatGPT authentication is `Logged in using ChatGPT`.
When hardened mode blocks a direct session because fak cannot enforce its next model
call, relaunch with `fak codex` (preferred) or `fak manage -- codex`.

For a deliberately direct-provider continuation while hardened mode remains selected,
pass `--allow-direct` when invoking the hook, or set
`FAK_ALLOW_DIRECT_CODEX_CONTINUE=1` in the hook environment. In PowerShell use
`$env:FAK_ALLOW_DIRECT_CODEX_CONTINUE=1`; on POSIX shells prefix the Codex command with
`FAK_ALLOW_DIRECT_CODEX_CONTINUE=1`. The hardened hook intentionally fails open on
unreadable/missing session evidence or other internal diagnostic errors. For repair
when `fak` itself is missing or broken, the exact break-glass value remains
`FAK_CODEX_RAW_RECOVERY=break-glass`; it bypasses the hook and prints a warning.

After the child exits, `fak codex` also tries to find the newest Codex session JSONL
touched during the run and writes privacy-preserving vCache artifacts: sanitized token
counters plus a `fak.vcache.score.v1` score under the user config dir
(`fak/codex-vcache/`), and the default observed-window snapshot consumed by
`fak vcache score`. Use `--vcache-out-dir DIR` to relocate those artifacts,
`--vcache-snapshot=false` to skip the default snapshot update, or
`--vcache-artifacts=false` to disable the post-run extraction.

At runtime `fak manage` rewrites the Codex child argv to include:

```text
-c model_provider=fak
-c model_providers.fak.base_url="http://127.0.0.1:<port>/v1"
-c model_providers.fak.wire_api="responses"
-c model_providers.fak.env_key="OPENAI_API_KEY"
```

That path is API-key billing today. A `codex login` ChatGPT subscription remains best
used with the MCP path below until subscription auth is wired through the guarded
Responses proxy.

Optional self-check for the MCP server:

```bash
python examples/mcp/verify.py
```

Add fak to Codex as a local MCP server:

```bash
codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json
```

Then verify Codex can see it:

```bash
codex exec --json "List the active MCP servers, then summarize AGENTS.md."
```

In the interactive Codex CLI, `/mcp` should show the `fak` server. In the IDE extension,
Codex uses the same `config.toml` MCP configuration as the CLI.

What Codex gets from this path:

| MCP tool surface | What it proves |
|---|---|
| `fak_adjudicate` | Ask the kernel for a verdict before running a call. |
| `fak_syscall` | Let the kernel adjudicate and execute a registered call. |
| `fak_admit` | Screen a tool result before it re-enters model context. |
| `fak_context_change` | Read the "what changed" feed when a shared state surface is present. |

Use this path when you are running Codex itself. It preserves Codex's current model wire and
adds fak as an explicit, inspectable tool boundary.

### Break-glass recovery for guard failures

If the guarded authentication bootstrap itself cannot start on Windows, invoke the installed
Codex launcher directly for authentication only:

```powershell
$env:CODEX_HOME = 'C:\Users\USER\.codex-codexFOUR'
& "$env:APPDATA\npm\codex.cmd" login
& "$env:APPDATA\npm\codex.cmd" login status
```

This direct launch is only an authentication-bootstrap escape hatch. Return ordinary and
model-bearing sessions to `fak guard` as soon as status reports `Logged in using ChatGPT`.
The normal response to a raw-session continuation block is to exit and restart with
`fak codex`. For one deliberately unguarded repair session, use the scoped break-glass
launcher. It starts raw Codex by default, prints an unavoidable warning, removes any inherited
loopback/guard state, and restores the normal default as soon as that child exits:

```powershell
fak guard disable --reason 'repair the guarded launcher'
```

Pass a command after `--` to repair with another harness, for example `fak guard disable
--reason 'repair Claude routing' -- claude`. This is a one-child bypass, not a persistent
machine toggle.

Each child appends a privacy-safe outcome row beside the normal fak usage journal. Inspect
weekly adoption and outcomes with `fak guard disable --usage` (add `--json` for the typed
`fak-guard-disable-usage-summary/1` fold). Rows contain only timestamp and the closed outcome
`success`, `child_nonzero`, or `launch_error`; they omit the reason, command, paths, and host
identity. `FAK_USAGE_LOG=off` disables both usage journals.

If `fak` itself is too broken to execute that command, use the raw recovery token directly.
The project hook checks this token in the shell *before invoking `fak`*, so this last-resort
path does not depend on a working `fak` binary:

```powershell
# PowerShell: launch raw Codex, then clear the process-scoped token after exiting.
$env:FAK_CODEX_RAW_RECOVERY = 'break-glass'
codex
Remove-Item Env:FAK_CODEX_RAW_RECOVERY
```

```bash
# POSIX shell: scope the token to this one raw Codex process.
FAK_CODEX_RAW_RECOVERY=break-glass codex
```

Neither path is a quiet convenience bypass. The manual path requires the exact value
`break-glass`, and every prompt prints `BREAK-GLASS raw Codex recovery active`; while either
path is active, the fak capability floor and guard audit are **not running**. Use the session
only to diagnose or repair the guarded launcher, exit it, clear any manually-set variable,
and prove normal service:

```powershell
Remove-Item Env:FAK_CODEX_RAW_RECOVERY -ErrorAction SilentlyContinue
fak codex --dry-run
fak codex
```

Do not persist the variable in a shell profile, user environment, worker manifest, or CI
secret. If `fak` is absent or too old to run the hook, no token is needed: the compatibility
path already fails open. The token is for the narrower case where the current hook is
successfully blocking raw Codex but guarded launch is unavailable.

### Long-context reset budgets

There are two different questions:

- **Can fak gate Codex tool use?** Yes, use the MCP path above.
- **Can fak automatically stop/restart a session at a 150k-token context budget?** Only
  when the model traffic also flows through the fak gateway, because MCP tool calls do not
  carry the model provider's prompt/cache token accounting.
- **Can MCP participate in a reset anyway?** Yes. An MCP client or wrapper can call
  `fak_session_reset` with the trace id, its observed `context_tokens`, and the transcript
  slice to distill. fak debits the budget, refuses unless the session is actually
  budget-drained, then returns the fresh continuation trace plus `seed_messages` to prepend
  in a new model window.

For an OpenAI-compatible client that can repoint its base URL, seed a stable served
session and context budget:

```bash
fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url "$UPSTREAM_OPENAI_COMPAT_BASE" \
  --session-id codex \
  --context-budget-tokens 150000 \
  --reset-on-budget \
  --policy examples/dev-agent-policy.json
```

Then point the client at `http://127.0.0.1:8080/v1`. With `--reset-on-budget`, when the
normalized prompt/context usage exhausts the budget the gateway mints a continuation id,
distills the refused transcript into a carryover seed, re-arms the continuation trace with
a fresh 150k budget, and retries the live request under that new trace.

Without `--reset-on-budget`, the next request returns `409` with the usual `error`
envelope plus:

- `session.continuation_id`: the fresh-window handoff id.
- `reset.action: restart_fresh_session`.
- `reset.required_actions`: dump the session image, start a fresh process, rehydrate the
  planned view, and reuse provider cache only where legal.

For `fak manage`, use the restart supervisor when the wrapped client benefits from a real
child-process boundary:

```bash
fak manage --provider openai --context-budget-tokens 150000 --restart-on-budget -- <openai-compatible-agent>
```

On budget exhaustion, guard distills the served transcript into a carryover seed, re-arms
the continuation trace, writes a seed JSON file, advances the default trace for callers
that omit `X-Trace-Id`, stops the child, and relaunches it with:

- `FAK_RESET_TRACE_ID`: the continuation trace id.
- `FAK_SESSION_ID`: the same continuation id, for wrappers that map session env to trace.
- `FAK_RESET_SEED_FILE`: the carryover seed JSON to prepend into the fresh model window.

Use `--restart-limit N` to cap relaunches and `--restart-seed-dir DIR` to choose where the
seed handoff files are written. The older `--reset-on-budget` mode remains available for
clients that want the gateway to retry in-place without killing the child process. A
generic child that ignores `FAK_RESET_SEED_FILE` still restarts under the fresh trace, but
will not automatically rehydrate its local transcript.

Current Codex CLI/IDE sessions should still use MCP first. If that Codex surface does not
honor an injected OpenAI-compatible base URL, fak can adjudicate tools but cannot
independently observe provider context usage; use `fak_session_reset` only when the Codex
side or a wrapper can report the context-token count it wants fak to debit.

Cooperative MCP reset call shape:

```json
{
  "name": "fak_session_reset",
  "arguments": {
    "trace_id": "codex",
    "context_tokens": 150001,
    "messages": [
      {"role": "system", "content": "You are working in C:\\work\\fak."},
      {"role": "user", "content": "Continue the reset implementation."}
    ]
  }
}
```

The response has `reset: true`, `from_trace_id`, `to_trace_id`, a
`reset_directive.action` of `restart_fresh_session`, and `seed_messages` when the reset
was accepted. A `reset: false` result is a normal refusal value: the session was not
budget-drained, or the gateway was not started with `--reset-on-budget`.

### Prove Codex actually used fak

The MCP server being configured is not enough evidence on its own. Prove a Codex session
called the fak server and keep the proof privacy-preserving:

```powershell
codex mcp get fak
python tools\codex_dogfood_witness.py --thread-id $env:CODEX_THREAD_ID --run-codex-exec
```

The witness writes `experiments/agent-live/codex-dogfood-<thread>.json` plus a sanitized
usage JSONL. It copies token counters, fak verdicts, MCP call metadata, and DOS hook
counts; it does not copy prompts, tool arguments, tool outputs, diffs, or model text.

When Codex was launched outside `fak codex`, turn the same Codex token counters into the
default vCache value surface by extracting a privacy-preserving cache snapshot after the
run:

```powershell
.\fak vcache codex-session-extract `
  --thread-id $env:CODEX_THREAD_ID `
  --out experiments\agent-live\vcache-codex-$env:CODEX_THREAD_ID.jsonl `
  --snapshot-out default `
  --score-out experiments\agent-live\vcache-codex-score-$env:CODEX_THREAD_ID.json
.\fak vcache score --json
```

The extractor writes only token counters and a `codex` prefix-family label. With
`--snapshot-out default`, `fak vcache score` reads that observed Codex window by default
instead of falling back to the synthetic forecast. `--score-out` writes the immediate
`fak.vcache.score.v1` verdict for the same window; provider rebate remains OBSERVED
telemetry and is still not treated as trust.

A good run has this shape:

- `status: PROVEN`
- `checks.mcp_stdio_adjudication.status: PASS`
- `checks.codex_exec_mcp_usage.status: PASS`
- `checks.vcache_telemetry_proof.status: PROVEN`
- `checks.dos_helped_session.blocked: 0`
- `checks.codex_hook_fast_path.status: PASS` with `codex_python_cli_hooks: 0`
- `summary.codex_actionability.status: PASS`, with any residual debt named as
  classes such as `HOST_SHELL_OPACITY` rather than copied commands

`checks.dos_session_audit.status` may still be `WARN`. That is useful dogfood evidence,
not a failed proof: it means DOS saw host calls whose file-tree footprint was opaque
while a lane lease was live. If `checks.codex_hook_fast_path.status` is already `PASS`,
the warning is not caused by Python hook-manifest wiring; prefer path-visible tool calls
or narrower shell commands, then rerun the witness and compare
`summary.dos.session_advisory_by_tool` and `summary.dos.unknown_tree_warning_rate`.
For the single-session witness, `summary.codex_actionability` splits actionable risk
from residual debt: delegates, stop failures, out-of-tree writes, and malformed shell
arguments are actionable; `HOST_SHELL_OPACITY` and `UNKNOWN_TREE_WARNINGS` remain
privacy-preserving upstream-footprint debt when the post-repair delegate count is zero.
This actionability block is scoped to the current Codex thread, so it can stay clean
while a later multi-session transfer audit warns about another recent session.

### Gate local Codex commands through fak

When Codex is about to run a local validation or build command, wrap it with the
same policy floor instead of treating the shell as trusted:

```powershell
python tools\codex_fak_gate.py `
  --tool run_tests `
  --redact-command `
  --command-label dogfood-witness-test `
  --out experiments\agent-live\codex-fak-gate-dogfood-witness-test-$env:CODEX_THREAD_ID.json `
  -- python tools\codex_dogfood_witness_test.py
python tools\codex_fak_gate.py `
  --tool run_tests `
  --redact-command `
  --command-label dos-recent-audit-test `
  --out experiments\agent-live\codex-fak-gate-dos-recent-audit-test-$env:CODEX_THREAD_ID.json `
  -- python tools\codex_dos_recent_audit_test.py
python tools\codex_fak_gate.py --tool go_test -- go test ./cmd/fak -run "TestRunVCache|TestReadVCacheTelemetry"
```

The wrapper calls `fak preflight` first. If the named operation is denied, the command
does not run:

```powershell
python tools\codex_fak_gate.py `
  --tool git_add `
  --expect-deny `
  --expect-reason DEFAULT_DENY `
  --redact-command `
  --command-label git-add-deny `
  --json `
  --dry-run `
  --out experiments\agent-live\codex-fak-gate-git-add-deny-$env:CODEX_THREAD_ID.json
python tools\codex_fak_gate.py `
  --tool git_commit `
  --expect-deny `
  --expect-reason DEFAULT_DENY `
  --redact-command `
  --command-label git-commit-deny `
  --json `
  --dry-run `
  --out experiments\agent-live\codex-fak-gate-git-commit-deny-$env:CODEX_THREAD_ID.json
python tools\codex_fak_gate.py `
  --tool git_push `
  --expect-deny `
  --expect-reason POLICY_BLOCK `
  --redact-command `
  --command-label git-push-deny `
  --json `
  --dry-run `
  --out experiments\agent-live\codex-fak-gate-git-push-deny-$env:CODEX_THREAD_ID.json
```

Use this for Codex's own operating loop: `run_tests` before Python test commands,
`go_test` before Go test commands, default-denied names such as `git_add` and
`git_commit` before local history mutation, and deny-listed names such as
`git_push` before any publish path. JSON reports record the verdict, command
identity, and exit code; command stdout/stderr are dropped unless
`--include-command-output` is set.

Fold the gate reports into the dogfood witness when you want one report to prove both
Codex MCP usage and local command admission. Repeat `--gate-report` for every
validation command the proof depends on:

```powershell
python tools\codex_dogfood_witness.py `
  --thread-id $env:CODEX_THREAD_ID `
  --run-codex-exec `
  --gate-report experiments\agent-live\codex-fak-gate-dogfood-witness-test-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-dos-recent-audit-test-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-add-deny-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit-deny-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push-deny-$env:CODEX_THREAD_ID.json
```

That adds `checks.local_fak_gate_reports.status: PASS` and
`summary.local_fak_gate.status: PASS` to the witness, with `summary.local_fak_gate.total`
showing how many checks passed. `DENIED_EXPECTED` reports count as passing local-gate
evidence and also increment `summary.local_fak_gate.expected_denied`.
Use `--redact-command --command-label <stable-name>` for durable reports: the command
still runs, but the report keeps only a label, executable name, argc, and SHA-256 digest
instead of the raw argv.

### Post-run DOS audit for Codex sessions

After a Codex run, fold the DOS hook stream before treating the run as clean:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --check-latest `
  --out experiments\agent-live\codex-dos-recent-audit.json
```

For a local transfer gate:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --fail-on-warn `
  --max-unknown-tree-rate 0.02 `
  --max-delegates 0 `
  --gate-report experiments\agent-live\codex-fak-gate-git-add-deny-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit-deny-$env:CODEX_THREAD_ID.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push-deny-$env:CODEX_THREAD_ID.json
```

The report copies only session filenames, thread IDs, timestamps, tool names, counts,
and latencies. It flags `tree_known=false` admission warnings, native-hook delegates,
stop blocks, and whether the cached Codex hook manifest uses the native DOS launcher
or the Python CLI hook path. A Bash-dominated report means the hook could not prove
precise file-tree footprints for the run; use narrower shell calls where the host can
derive a tree, prefer MCP/fak verdict surfaces for checkable calls, and file upstream
footprint-derivation debt when the rate stays above the
[transfer-playbook](../dos-kernel-transfer-playbook.md) threshold. `using_latest: true`
only proves package freshness; `codex_hook_fast_path.status: PASS` proves the Codex
hook manifest is actually wired to the fast path.

If `codex_hook_fast_path.status` is `WARN`, inspect the manifest repair first:

```powershell
python tools\codex_dos_hook_doctor.py --codex-home $env:USERPROFILE\.codex
```

The dry-run prints projected hook modes after apply. A projection with native
Codex hooks and zero Python Codex hooks proves the repair would clear the fast-path
warning before you write the cache.

Then apply it explicitly:

```powershell
python tools\codex_dos_hook_doctor.py --codex-home $env:USERPROFILE\.codex --apply
```

The doctor keeps Python as the delegate fallback; it only changes the first path
Codex hooks try.

After the repair, read `post_repair_observations` separately from the whole recent
window. A whole-window delegate count can include old Python-hook history; a
post-repair delegate count of `0` proves the fast-path issue is gone. If the report
still shows `shell_no_write_target_detected` under `post_repair_command_shapes`, the
remaining warning is shell opacity from read/inspect calls. Prefer host-visible
read/search tools when Codex exposes them; otherwise keep the WARN as upstream
footprint-derivation debt rather than treating it as a write-safety finding.
If the family lens shows `git_write`, the actionable gate should fail: commit,
add, push, and similar operations are opaque mutations and need an explicit
operator gate.
Supplying the three expected-deny Git gate reports proves a structured gate timestamp.
The audit then also checks the post-gate Codex window; if another thread runs opaque
`git_write` after that timestamp, the transfer gate remains WARN even though the
single-thread witness can still be clean.

For automation that should fail only on post-repair actionable risk, use
`--fail-on-actionable-warn --max-delegates 0`. Keep `--fail-on-warn` for the stricter
transfer gate that still fails on residual shell-opacity debt.

The recent-audit command is intentionally multi-session: it folds the DOS-matched
Codex threads included in `sessions_audited`, so a `git_write` family from a peer
or older audited stream can make the transfer gate fail even when the
single-thread dogfood witness is clean. Use `mutating_shell_sessions` to identify
the sanitized thread/file bucket, then keep that failing report as transfer-gate
evidence; do not fold it into `checks.local_fak_gate_reports` unless the witness
is meant to fail closed too.

After the structured Git deny probes exist, pass them back into the recent audit:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --gate-report experiments\agent-live\codex-fak-gate-git-add.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push.json `
  --fail-on-actionable-warn `
  --max-delegates 0
```

That gate passes only when the expected-deny Git probes are valid and no audited
Codex session contains a new `git_write` family after the latest probe timestamp.

To file or track that residual without leaking session content, add `--out-debt
experiments\agent-live\codex-dos-host-opacity-debt.md`. The packet copies counts and
shell shape/family categories only, including any mutating family counts.

## Path 2: OpenAI-compatible clients through `fak serve`

Start fak in front of an OpenAI-compatible upstream:

```bash
go build -o fak ./cmd/fak
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder \
  --policy examples/dev-agent-policy.json
```

Then repoint an OpenAI-compatible client:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
```

For Python SDK clients:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="fak-local",
)

response = client.chat.completions.create(
    model="qwen2.5-coder",
    messages=[{"role": "user", "content": "List the Go packages in this repo."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
```

For TypeScript SDK clients:

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "fak-local",
});
```

Use this path when a framework already lets you set an OpenAI-compatible base URL.
Good fits include:

- OpenAI Agents SDK in Chat Completions mode.
- LangChain, LlamaIndex, AutoGen, and Pydantic AI Chat Completions models.
- Vercel AI SDK OpenAI-compatible providers and similar clients.

## What the kernel blocks for coding workflows

`examples/dev-agent-policy.json` is the coding-agent floor. It allows ordinary
read/search/list flows plus build and test commands. It blocks publish and
self-modification surfaces.

| Attempt | Kernel result |
|---|---|
| Read/search/list calls | Allowed when the tool is on the allow-list or prefix allow-list. |
| `git_diff`, `git_log`, `git_status`, `go_build`, `go_test`, `run_tests` | Allowed by the dev-agent policy. |
| `git_add`, `git_commit` | Denied by the default-deny floor unless routed through a narrower release/ship gate. |
| `git_push`, `git_merge`, `git_tag` | Denied with `POLICY_BLOCK`. |
| Writes to `.git/`, `internal/kernel/`, `internal/policy/`, `VERSION`, or `dos.toml` | Denied by the self-modify floor. |
| Secret-shaped fields such as `api_key`, `token`, or `authorization` | Redacted or quarantined by result-side guards. |

Check one call without launching a model:

```bash
./fak preflight --tool git_push --args "{}" --policy examples/dev-agent-policy.json
```

## Using a Responses upstream

If your upstream model provider is the OpenAI Responses API, fak can still be useful as
the gateway's upstream client:

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai-responses \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --policy examples/dev-agent-policy.json
```

Clients still call fak's supported inbound routes. That means:

- OpenAI-compatible clients call `http://127.0.0.1:8080/v1/chat/completions`.
- Responses-API clients (Codex CLI/IDE, `terminus`) call `http://127.0.0.1:8080/v1/responses`
  — the buffered inbound Responses route (#925); use MCP instead if you need streaming.
- Anthropic-wire clients call `http://127.0.0.1:8080/v1/messages`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Codex cannot see the MCP server | Run `codex mcp --help`, re-add the server, then check `/mcp` in the Codex TUI. |
| `codex exec --json` has no fak events | The MCP server is not enabled for that Codex run, or the task did not call fak. |
| OpenAI SDK gets 404 | OpenAI-compatible clients need the `/v1` suffix: `http://127.0.0.1:8080/v1`. |
| Anthropic SDK gets 404 | Anthropic clients need the origin without `/v1`: `http://127.0.0.1:8080`. |
| Everything is denied | Load a policy with `--policy`; with no policy the floor fails closed. |
| You tried to point default Codex model traffic at fak | Use MCP instead, or use a client/framework path that explicitly speaks Chat Completions to fak. |

## Source alignment

This page was checked against the current OpenAI Codex manual on 2026-06-25:

- [Codex overview](https://developers.openai.com/codex/overview)
- [AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Non-interactive `codex exec`](https://developers.openai.com/codex/noninteractive)
- [Codex configuration](https://developers.openai.com/codex/config-basic)

fak-side references:

## Skill portability: sharing skills with OpenCode and Claude Code (#10689)

Codex skills stored under `.agents/skills/<name>/SKILL.md` conform to the [Agent Skills](https://agentskills.io) specification. These skills can be shared and imported into OpenCode:

1. **Discovery Architecture**:
   - **Codex**: Discovers skills in `.agents/skills/` (and `$CODEX_HOME/skills/`).
   - **OpenCode**: Discovers skills configured in `opencode.json` via `"skills": { "paths": [".agents/skills"] }`.
   - **Claude Code**: Uses `.claude/skills/<name>/SKILL.md` as canonical definitions.

2. **Synchronization**:
   Run `go run ./cmd/fak-project-assets sync --json` to synchronize canonical skill definitions across `.claude/skills` and `.agents/skills`. Run `go run ./cmd/fak-project-assets parity --json` to verify that all skills across harnesses are in parity (`zero_unexplained_gaps: true`).

3. **Portability Layer**:
   See the comprehensive guide in [`docs/integrations/opencode.md`](opencode.md#skill-portability-importing-skills-into-opencode-10689-10690-10691) and [`docs/integrations/project-assets.md`](project-assets.md).

- [Integration index](README.md)
- [Harness integration acceptance checklist](harness-acceptance-checklist.md)
- [MCP example](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md)
- [Policy manifest guide](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)
- [Supported APIs and protocols](../supported/apis-and-protocols.md)
- [Compatibility matrix](compatibility-matrix.md)


## Diagnose an MCP startup before Codex

When Codex reports only `connection closed: initialize response`, run the same configured
stdio child through the read-only staged probe:

```powershell
fak doctor mcp --server fak
fak doctor mcp --server fak --json
```

The command reads `[mcp_servers.fak]` from `$CODEX_HOME/config.toml` (or
`~/.codex/config.toml`), never renders its `env` table, resolves the executable and version,
checks the policy path, spawns the child, writes a real MCP `initialize`, and classifies the
first stdout frame. The stable `fak-doctor-mcp/1` JSON report names stages such as
`CONFIG_INVALID`, `EXECUTABLE_MISSING`, `POLICY_UNREADABLE`, `POLICY_MALFORMED`,
`SPAWN_FAILED`, `INITIALIZE_TIMEOUT`, `EARLY_EXIT`, and `STDOUT_CONTAMINATION`, each with a
checkable recovery action. It only probes and terminates the child; it never quarantines or
rewrites production state.

To diagnose an exact command instead of Codex config, put its arguments after the flags:

```powershell
fak doctor mcp --command C:\path\to\fak.exe serve --stdio --policy C:\path\policy.json
```

## Diagnose abrupt Codex/fak sessions (`fak-dev sessiondiag`)

When a Codex-backed session exits, freezes, or loses tool events, run the read-only diagnostic as soon as possible:

```powershell
fak-dev sessiondiag --since 24h
fak-dev sessiondiag --since 2h --json
```

The verb opens Codex's `logs_2.sqlite` in SQLite read-only/query-only mode, bounds all log queries by time, and emits counts rather than prompt or tool bodies. To keep the live diagnostic bounded, it validates that the schema is readable but reports `integrity=not_checked`; run a full integrity check only on the shutdown-time copy described below. `CORRELATED_RUNTIME_PRESSURE` means the window contains store/WAL pressure, slow structured-log writes, or in-process app-server queue loss. It deliberately reports `causality=not_established`: those signals can explain an unusable session but do not prove which process terminated. `EXPLICIT_FAILURE_EVIDENCE` is reserved for an explicit panic/fatal/fak-child-exit log record. Preserve matching Windows Application Error/WER evidence when available.

`LOG_STORE_RECLAIMABLE_PRESSURE` can be large even when `quick_check` is `ok`: deleted SQLite pages remain in the database file until compaction. Do not checkpoint, vacuum, replace, or delete the live store while any Codex process is running. The safe operator procedure is: close **all** Codex processes, copy `logs_2.sqlite` together with any `-wal` and `-shm` siblings, verify the copy, then compact the copy or run an explicit SQLite maintenance operation during that shutdown window. The diagnostic itself never performs maintenance.
