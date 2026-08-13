---
title: "Native desktop command center for durable coding sessions — research and staged plan"
date: 2026-08-13
status: proposed
problem_centrality: Core
platform_order: Windows first, macOS second
---

# Native desktop command center for durable coding sessions

## Verdict

Do **not** build another chat pane and do **not** make a window, tab, or harness process the
unit of durability. Build a keyboard-first **local command center over one durable fak daemon**.
The daemon owns canonical session identity, the event journal, resource residency, capability
leases, and harness adapters. Windows/macOS clients are replaceable projections. A terminal is a
first-class surface attached to the same session, not a compatibility panel hidden behind the GUI.

The first useful spine is deliberately smaller than a desktop shell: start a session through the
daemon, attach a real terminal, kill the client and harness, reopen the client, and recover the
same session with its transcript, working-directory/revision identity, pending approval, terminal
scrollback, resource-reuse accounting, and an explicit verdict for every uncertain tool effect.
Until that path is witnessed, broad visual design, multi-agent dashboards, and cross-device sync
are follow-ons rather than the product.

## Value frame

- **For:** coding power users who run several long-lived local or proxied native harness sessions.
- **Problem:** today process death, client restart, harness replacement, and new windows conflate
  four different things—work identity, model conversation, local process, and rendered view—and
  repeat repository/tool/resource setup while making recovery ambiguous.
- **Today:** use terminal multiplexers and harness-native resume commands, or accept opaque IDE/chat
  tabs. The terminal path is fast and inspectable but fragments orchestration; GUI paths aggregate
  activity but commonly hide process/state boundaries and weaken keyboard composability.
- **Better because:** one local kernel amortizes stable resources and makes every projection
  disposable without taking the work with it, while a real PTY preserves terminal semantics.
- **Witness:** the crash/reopen spine above, plus measured warm-vs-cold attach cost and zero
  cross-workspace context/capability bleed.

The real alternatives are tmux/Windows Terminal plus native CLIs, and current agent IDEs—not a
naive application that reloads everything. The native app wins only if it is at least as quick and
legible for single-session work and materially better for recovery and multi-session attention.

## All-work checks

| Check | Design answer | Required witness |
|---|---|---|
| P1 managed context | Stable instructions, tool schemas, repo indexes, and provider prefixes are owned by typed residency scopes rather than recopied into each UI session. | Cold/warm bytes, tokens, processes, and time attributed by scope. |
| P2 net-true efficiency | Compare against a tuned persistent terminal/harness baseline, including daemon/client overhead. | `claim-check`-grade cold/warm benchmark; no headline from a cold naive baseline. |
| P3 bounded adaptation | Reuse is keyed and invalidated by repo/worktree revision, policy hash, tool schema, provider/model, and credential/capability epoch. | Mutation tests prove stale entries miss and cross-scope reads deny. |
| P4 integrated operations | Journal, recovery, attention, approval, diagnostics, and update health are normal UI projections of kernel state. | Crash matrix and operator runbook exercised from a packaged build. |

## Research method and evidence boundary

On 2026-08-13, sixteen read-only research workers were launched on disjoint questions: lifecycle,
terminal HCI, desktop stacks, coding-agent competitors, Grok Bot, resource reuse, PTY embedding,
local-first state, security, multi-agent UX, context economics, platform NFRs, architecture,
product strategy, repo gaps, and open-source prior art. Five complete reports were independently
read back and reconciled here; incomplete/guard-refused/timeout outputs were not treated as
evidence. The completed reports covered terminal ergonomics, desktop stacks, Grok Bot, resource
sharing, and PTY integration. Existing repository notes and code were separately inspected.

This is a dated design input, not a claim that all volatile vendor behavior is stable. Primary
sources should be refreshed before an implementation choice is made.

## State of the art: what survives scrutiny

### Product and interaction patterns

| System/pattern | Strong idea | Gap for this product |
|---|---|---|
| tmux / terminal multiplexers | server owns sessions; clients detach/re-attach; keyboard grammar and shell composition remain intact | weak semantic view of model/tool/approval/diff state; recovery is process-centric |
| VS Code/Cursor/Windsurf-style IDE agents | repository, editor, diff, diagnostics, and agent work share one workspace | extension host/window lifecycle often remains too close to session identity; dense multi-agent attention is immature |
| Codex/Claude/Aider-style CLIs | low-friction cwd-native operation, scriptability, inspectable process ownership | each harness carries its own lifecycle and context contract; cross-harness continuity is not canonical |
| durable workflow engines | replay, durable timers, explicit effects, worker replacement | coding interaction and PTY streams are not naturally deterministic workflows; blindly replaying tools is unsafe |
| local-first/event journals | restartable materialized views, offline reads, auditable causality | require explicit schema/version/compaction and secret-redaction contracts |
| Grok Bot | named persistent coworkers, shared per-user cloud computer, skills/routines, approvals, desktop/mobile projections | cloud-computer trust and messaging metaphor do not supply repository/worktree isolation or terminal-native coding control |
| Grok Build | resumable coding sessions, local tools/sandbox/extensions/subagents and ACP surface | a harness remains one adapter; it must not become fak's canonical session store |

Verified Grok Bot facts came from [the product page](https://x.ai/bot),
[official overview](https://docs.x.ai/grok-bot/overview), and related official documentation
reviewed 2026-08-13. The useful lesson is durable named delegation plus replaceable clients. The
wrong lesson is to collapse every bot into one ambient computer or to use a messaging timeline as
the only coding surface.

### Why the terminal still feels good

The terminal's advantage is not nostalgia or monospace styling. It is a compact control protocol:

1. **Keystroke-to-effect is local and predictable.** Human-perceived direct manipulation degrades
   around the familiar 100 ms boundary; terminal input and echo usually remain below it.
2. **One compositional grammar.** Shell commands, pipes, redirection, files, environment, exit
   codes, and scripts move between interactive and automated use without translation.
3. **Stable spatial contract.** Append-oriented output and explicit command boundaries cause less
   unsolicited reflow than card feeds, animated panes, and responsive chat layouts.
4. **Dense but inspectable state.** cwd, command, output, errors, and process ownership are visible;
   users can copy exact bytes and reproduce an operation outside the UI.
5. **Failure containment.** A client can die while tmux/server/process state continues, and the
   user understands which process was lost.
6. **Keyboard reachability.** Power users do not pay pointer travel or mode-switch cost for the
   common path.

A desktop client should therefore add semantic projections—session state, diff, approvals,
resource reuse, provenance—without replacing the terminal grammar. Required initial budgets:

- local key echo p95 <= 16 ms; ordinary command acknowledgement p95 <= 50 ms;
- no unsolicited focus changes or layout reflow in the active work surface;
- every visible action has a stable command ID, shortcut, and command-palette entry;
- 100% keyboard completion of start/attach/switch/approve/deny/diff/recover/close;
- terminal bytes remain round-trippable and selectable; raw transcript export is always available;
- cold app launch target <= 1.0 s to useful session list and warm attach <= 250 ms, measured on a
  declared reference machine rather than asserted from framework marketing.

### Desktop shell and terminal substrate

The evidence does not justify committing the product to a framework before the spine benchmark.
Use a framework bake-off with the same vertical slice. The leading hypothesis is:

- **Windows first:** WinUI 3/.NET shell for platform-grade accessibility, IME, windowing,
  deployment, and Windows App SDK terminal control integration.
- **macOS second:** native SwiftUI/AppKit shell over the same versioned daemon protocol.
- **Portable fallback candidate:** Avalonia only if the bake-off proves equivalent terminal,
  accessibility, IME, high-DPI, packaging, and startup behavior. Tauri is an acceptable prototype
  control surface, not the default terminal renderer.
- **Rejected default:** Electron/web terminal as the entire workbench. It is a useful portable
  fallback, but duplicates a browser runtime and makes native terminal/accessibility fidelity a
  continuing tax.

On Windows, ConPTY is transport, not a renderer. Prefer the Windows Terminal control when its
published integration surface and license are suitable; otherwise keep ConPTY behind an internal
`TerminalSession` seam and compare renderer candidates. On macOS, use a real PTY (`forkpty`/
`openpty`) behind the same semantic seam. The daemon must own PTY/harness lifetime so closing the
window does not implicitly terminate work. Windows pseudo-console cleanup ordering and macOS
controlling-terminal/process-group behavior require dedicated integration tests.

Primary references: [ConPTY API](https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session),
[Windows Terminal control source](https://github.com/microsoft/terminal/tree/main/src/cascadia/TerminalControl),
[Windows Terminal repository](https://github.com/microsoft/terminal),
[xterm.js repository](https://github.com/xtermjs/xterm.js),
[Ghostty repository](https://github.com/ghostty-org/ghostty), and
[WezTerm repository](https://github.com/wezterm/wezterm), all accessed 2026-08-13.

## Canonical architecture

```text
 Windows client (WinUI)       macOS client (later)        CLI / terminal client
          \                         |                         /
           \---- versioned local authenticated IPC ----------/
                                  |
                    fak desktop/session daemon
       +--------------------------+---------------------------+
       | canonical event journal + materialized session view |
       | capability/policy leases + effect/idempotency ledger |
       | attention/approval queue + diagnostics              |
       | resource residency manager + content-addressed refs |
       +--------------+----------------------+---------------+
                      |                      |
             HarnessAdapter           TerminalAdapter
        Codex/Claude/Grok Build/...   ConPTY / Unix PTY
                      |
              provider/gateway/cache
```

### Four identities that must never be conflated

| Identity | Owner | Durability | Rule |
|---|---|---|---|
| `work_session_id` | fak daemon | durable across all client/harness restarts | canonical user-visible identity |
| `projection_id` | desktop/CLI client | disposable | may reconnect from any compatible local client |
| `harness_instance_id` | harness adapter | process lifetime | replaceable; zero or one active writer lease initially |
| `model_conversation_id` | provider/harness adapter | provider-defined | optional resumable anchor, never the sole source of truth |

A fifth identity, `workspace_snapshot_id`, binds repository root, worktree/branch, HEAD and dirty
fingerprint, instruction/policy/tool-schema hashes, and credential epoch. Reuse across this key is
a miss, not a best effort.

### Durable state and projections

Use an append-only, checksummed event log with monotonic per-session sequence numbers; materialize
query views for session list, timeline, approvals, resources, and terminal metadata. SQLite/WAL is
a reasonable implementation candidate, but the contract matters more than storage technology:

- commit events before acknowledging state-changing UI commands;
- batch terminal byte chunks separately from semantic events and bound retention;
- snapshot materialized views at a versioned sequence; replay tail after snapshot;
- preserve unknown event fields and support explicit schema migration;
- redact/tokenize secrets before persistence and expose retention controls;
- make export/import a portable session image, building on `internal/sessionimage`;
- never infer success of an effect from a missing process.

### Lifecycle state machine

```text
NEW -> STARTING -> RUNNING <-> QUIESCING -> QUIESCED
                   |   ^          |             |
                   |   |          v             v
                   |   +------ RECOVERING <- ORPHANED
                   |                         /    |
                   +------> FAILED <--------+     +-> CLOSED
```

Orthogonal state is required rather than one overloaded enum:

- **projection:** detached / attached;
- **harness:** absent / starting / live / exited / unreachable;
- **conversation:** none / resumable / expired / forked;
- **workspace:** exact / advanced / dirty-diverged / missing;
- **effects:** clean / pending / uncertain / compensating;
- **resources:** cold / warming / resident / stale / evicting.

Recovery protocol:

1. acquire a per-session recovery lease and fence stale harness writers;
2. replay journal to the latest verified sequence and validate checksums;
3. re-read workspace identity and policy/capability epochs;
4. query adapter-specific harness/process/conversation state without mutating it;
5. classify every begun-but-uncommitted tool effect as `known-not-run`, `confirmed`, or
   `uncertain`; never auto-replay `uncertain` effects;
6. reattach when identity matches, otherwise start replacement and restore only typed portable
   state;
7. publish a `RecoveryCompleted` event containing losses, substitutions, stale resources, and
   required human decisions;
8. let clients rebuild projections solely from snapshot + ordered events.

## Resource residency: reuse without bleed

A dedicated daemon is valuable only when ownership is explicit. Use this scope lattice:

| Resource | Safe default scope | Key/invalidation examples |
|---|---|---|
| static binary/runtime assets | host | binary/version/platform |
| provider connection pool | credential + endpoint | credential epoch, endpoint/TLS config |
| model prefix/KV cache | provider + model + exact prefix hash | provider TTL/model/template/tool schema |
| MCP process/connection | workspace + policy + credential principal | server config/binary/policy/secret epoch |
| repository index/map | repo object identity + revision/dirty fingerprint | HEAD, tracked and admitted dirty paths |
| language server | workspace/worktree + toolchain/env | root, config, env, toolchain, watched files |
| warm harness process | workspace + adapter + policy principal | harness version, cwd, environment, lease epoch |
| conversation | work session + adapter/provider | explicit fork/resume only |
| terminal/PTTY | work session + harness instance | process group and terminal generation |

Existing issue [#3405](https://github.com/anthony-chaudhary/fak/issues/3405) already tracks shell and
per-session MCP pooling. The desktop plan consumes that leaf but adds typed residency and a user
projection of hits, misses, evictions, and stale reasons.

Measure `cold_start_ms`, `warm_attach_ms`, `resident_bytes`, `new_processes`, `instruction_bytes`,
`tool_schema_bytes`, `input_tokens`, prefix-cache read/write, index bytes read, and `reuse_reason`
per scope. A reuse claim is net-true only against a tuned persistent terminal/harness baseline.

## Security invariants

1. Local IPC authenticates the logged-in user and binds each command to session, workspace,
   projection, capability, and monotonically increasing lease epoch.
2. Only one harness writer lease exists per work session in the first spine; stale writers are
   fenced after recovery.
3. Capabilities expire or are re-adjudicated when workspace, policy, credential, adapter, or
   executable identity changes.
4. Context and resource keys are fail-closed; an unrecognized or partially computed scope cannot
   hit a broader cache.
5. Tool effects carry idempotency/effect IDs. Read-only effects may be retried under policy;
   destructive or externally visible uncertain effects require read-back or operator resolution.
6. MCP/plugin processes run under declared principals and never inherit ambient secrets merely
   because the daemon is long-lived.
7. Journal/export surfaces redact secrets and retain provenance; terminal raw bytes have explicit
   sensitivity and retention controls.
8. Signed updates and protocol version negotiation prevent a downgraded client from silently
   weakening daemon policy.
9. UI rendering treats terminal/tool/model content as untrusted data; hyperlinks, OSC commands,
   clipboard writes, and file opens pass policy.
10. Cross-workspace negative tests are release gates, not unit-test-only assumptions.

## Keyboard-first workbench

The primary object is a **session**, not a chat. Initial layout:

- left: dense session switcher grouped by workspace with state glyph, attention reason, elapsed
  time, cost/resource delta, and last verified effect;
- center: real terminal or semantic timeline/diff, toggled without changing session identity;
- right (optional): context/resources/provenance inspector, closed by default on small displays;
- bottom/command palette: one searchable grammar for session, harness, terminal, policy, and diff
  actions;
- global attention queue: approvals, uncertain effects, conflicts, crashed/orphaned sessions, and
  completed work—sorted by urgency and age, not chat recency.

Suggested commands: `session.new`, `session.switch`, `session.attach`, `session.detach`,
`session.recover`, `session.fork`, `session.close`, `view.terminal`, `view.timeline`, `view.diff`,
`attention.next`, `effect.confirm`, `effect.retry`, `effect.abandon`, `resource.explain`.
Shortcuts are user-remappable and shown from the same command registry; there is no mouse-only
operation. Multiple windows are projections over one daemon and may show the same session
read-only; writer focus/lease is explicit.

## Gap audit against current fak

| Capability | Status | Evidence | Missing seam |
|---|---|---|---|
| append-only crash journal with boot/process identity | PRESENT | `internal/sessionjournal/sessionjournal.go`; `docs/observability/durable-artifacts.md` | single canonical desktop event envelope and migrations |
| lifecycle reconciliation/action planning | PRESENT | `docs/session-lifecycle-reconciliation.md`; `cmd/fak/session_reconcile.go` | daemon-owned continuous reconciler + client subscription |
| session registry, descriptors, read/search/replay | PRESENT | `internal/sessionregistry`, `sessiondesc`, `sessionread`, `sessionsearch`, `sessionreplay` | unified query model and stable IPC schema |
| portable snapshots/images | PARTIAL | `internal/sessionimage`; `docs/notes/PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md` | adapter-neutral restore contract and desktop export/import |
| control-state pause/resume/steer | PARTIAL | `internal/sessionctl`, `sessionsteer`, `sessionsignals`; `docs/notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md` | writer fencing and generation-aware acknowledgements |
| harness profiling/resolution | PARTIAL | `internal/harnessprofile`, `harnessres`; harness/session CLI files in `cmd/fak` | versioned `HarnessAdapter` lifecycle interface |
| managed context/cache/routing | PRESENT | `internal/ctxmmu`, `ctxresidency`, `gateway`, `vdso` | desktop-readable residency attribution and strict scope keys |
| policy/adjudication | PRESENT | `internal/policy`, `internal/adjudicator` | local IPC principal/capability lease envelope |
| real PTY ownership/embedding | ABSENT as desktop substrate | current CLI/process paths and issue #3405 | `TerminalAdapter`, scrollback contract, renderer bake-off |
| long-lived local desktop daemon | ABSENT | external-process read plane is currently file/query oriented | authenticated command+event IPC, supervisor/install contract |
| packaged Windows/macOS clients | ABSENT | no desktop client tree | spine client, accessibility/IME/window lifecycle |
| multi-session semantic attention | PARTIAL | fleet/watchdog/observability commands | one session projection and attention reason taxonomy |

The repository is unusually strong below the UI: lifecycle, journal, image, policy, cache, and
read-plane leaves exist. The architectural risk is creating a parallel desktop state store instead
of composing them.

## Staged delivery plan

### Phase 0 — contract and benchmark harness (spine prerequisite)

- Define versioned IDs, event envelope, effect state, adapter interface, recovery result, and
  resource-scope key as additive leaves.
- Build a deterministic fake harness that can crash before/after effects and can expire a model
  conversation.
- Capture tuned terminal/harness baseline: cold start, warm resume, process count, memory, input
  bytes/tokens, and crash recovery semantics.
- **Exit:** protocol golden files and crash matrix run without any desktop framework.

### Phase 1 — minimal working Windows spine

- Per-user daemon starts on demand and owns one fake/real harness plus ConPTY.
- WinUI shell lists one session, opens the terminal, displays lifecycle/resource state, and sends
  commands over authenticated versioned IPC.
- Kill client: harness continues. Kill harness: daemon marks orphaned and recovers. Kill daemon:
  restart replays journal and resolves uncertain effects. Reboot: same contract.
- **Exit witness:** packaged local app performs start -> command -> crash each boundary -> reopen ->
  verified recovery; keyboard-only; raw terminal transcript and event log captured.

### Phase 2 — native harness adapters and residency

- Codex and Claude adapters first; Grok Build/ACP evaluated as a third adapter.
- Share only typed stable resources; show hit/miss/stale reason and memory/process cost.
- Implement explicit fork when conversation/workspace identities diverge.
- **Exit:** cross-adapter crash matrix and zero cross-workspace/cache/capability bleed.

### Phase 3 — power-user semantic workbench

- Timeline/diff/provenance views, attention queue, approvals and uncertain-effect resolution.
- Multi-window/read-only projection support; command registry and remapping.
- Benchmark 1/5/20/100 sessions for input latency, memory, attach, attention operations, and sleep/
  resume behavior.
- **Exit:** power users complete defined workflows no slower than tuned terminal for one session and
  faster for recovery/attention tasks.

### Phase 4 — macOS projection and optional sync

- Implement native macOS shell and Unix PTY adapter against unchanged protocol.
- Do not sync ambient secrets, raw terminal history, or capabilities by default. Cross-device sync
  requires a separate threat model and conflict contract.
- **Exit:** the same protocol/crash/accessibility suite passes on declared macOS hardware.

## Nonfunctional acceptance gates

- app/client/process/daemon/harness crash matrix; sleep, logout, reboot, update, disk-full, corrupt
  tail, expired credential, changed HEAD, deleted worktree, and provider resume expiry;
- Windows UI Automation and macOS accessibility trees expose semantic session/status/action names;
- screen reader and keyboard-only completion; IME composition; 100/125/150/200% DPI; multi-monitor
  restore with missing-monitor fallback;
- daemon idle and 20-session memory/process budgets declared before implementation and compared to
  tuned terminal baseline;
- protocol supports one previous client version during rolling update and fails closed otherwise;
- no in-place auto-retry of an uncertain destructive effect;
- journal recovery is bounded by snapshot + tail and reports corruption rather than truncating it
  silently;
- uninstall/retention/export behavior is explicit and testable.

## Kill criteria

Stop or radically rescope the desktop shell if the spine cannot meet all of these:

1. window/client death is less survivable or less legible than tmux plus the native harness;
2. single-session keyboard operation is measurably slower on the defined core workflow;
3. daemon residency costs exceed repeated setup savings at the measured normal session count;
4. exact resource/capability scoping cannot prevent cross-workspace bleed;
5. framework terminal, IME, or accessibility defects require maintaining a second fake-terminal
   semantics layer;
6. adapter recovery cannot distinguish confirmed from uncertain external effects.

## Backlog contract (dependency order)

1. **Spine ([#6569](https://github.com/anthony-chaudhary/fak/issues/6569)):** durable daemon + fake harness + terminal projection + client/harness/daemon crash
   witness on Windows.
2. Canonical session/event/effect protocol golden corpus.
3. Authenticated local IPC principal and capability leases.
4. Harness adapter lifecycle interface and Codex adapter.
5. Claude adapter; evaluate Grok Build ACP adapter independently.
6. Typed resource residency keys and cold/warm attribution.
7. Terminal adapter/renderer bake-off with ConPTY and PTY fixtures.
8. Attention queue and uncertain-effect resolution UX.
9. Accessibility/IME/DPI/multi-monitor release gate.
10. macOS native projection over unchanged protocol.

Each item is independently shippable only after its done condition includes a captured effect
witness. Broad visual polish, marketplace/plugins, mobile views, and cloud sync are intentionally
outside the first spine and must be filed—not silently narrated—if pursued.

## Source ledger

Primary references reviewed 2026-08-13:

- Microsoft, [Creating a Pseudoconsole session](https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session).
- Microsoft, [Embedding Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/embedding).
- Microsoft, [Windows App SDK support policy](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/stable-channel).
- Microsoft, [Windows Terminal](https://github.com/microsoft/terminal) (MIT).
- xterm.js, [repository](https://github.com/xtermjs/xterm.js) (MIT).
- Ghostty, [repository](https://github.com/ghostty-org/ghostty) (MIT).
- WezTerm, [repository](https://github.com/wezterm/wezterm) (MIT).
- Tauri, [process model](https://v2.tauri.app/concept/process-model/) and
  [security](https://v2.tauri.app/security/).
- Electron, [process model](https://www.electronjs.org/docs/latest/tutorial/process-model) and
  [security checklist](https://www.electronjs.org/docs/latest/tutorial/security).
- Avalonia, [supported platforms](https://docs.avaloniaui.net/docs/overview/supported-platforms).
- xAI, [Grok Bot](https://x.ai/bot), [Grok Bot overview](https://docs.x.ai/grok-bot/overview),
  [Grok Build overview](https://docs.x.ai/build/overview), and
  [Grok Build open-source announcement](https://x.ai/news/grok-build-open-source).
- Anthropic, [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
- OpenAI, [prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
- Model Context Protocol, [architecture](https://modelcontextprotocol.io/docs/learn/architecture).
- Language Server Protocol, [specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/).
- Nielsen Norman Group, [response-time limits](https://www.nngroup.com/articles/response-times-3-important-limits/).

Repository evidence is pinned by path above; use `fak version modules` when implementation claims
are made, because this shared trunk advances rapidly.




