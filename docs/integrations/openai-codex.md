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
| Codex CLI with an OpenAI API key and you want fak in front of the model wire | `fak codex -- <codex args...>` | One command starts `fak guard`, launches Codex, and injects per-run Codex `-c model_provider=fak` overrides for the Responses wire. |
| OpenAI SDKs, OpenAI Agents SDK, LangChain, LlamaIndex, or any Chat Completions client | `fak serve` as an OpenAI-compatible gateway | The client already calls `/v1/chat/completions`, so you repoint its base URL to fak. |

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
fak guard --split off ... -- codex --dangerously-bypass-approvals-and-sandbox exec --json ...
```

At runtime `fak guard` rewrites the Codex child argv to include:

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

For `fak guard`, use the restart supervisor when the wrapped client benefits from a real
child-process boundary:

```bash
fak guard --provider openai --context-budget-tokens 150000 --restart-on-budget -- <openai-compatible-agent>
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

- [Integration index](README.md)
- [MCP example](../../examples/mcp/README.md)
- [Policy manifest guide](../../POLICY.md)
- [Supported APIs and protocols](../supported/apis-and-protocols.md)
- [Compatibility matrix](compatibility-matrix.md)
