# OpenAI Codex Desktop App for Windows (UI/UX): Architecture, Empirical Study, and fak Harness Integration

**Date:** September 5, 2026 (Pacific) / September 6, 2026 (UTC)  
**Status:** Bounded local installation audit, empirical issue analysis, and harness architectural specification.  
**Audience:** Harness architects, agent kernel contributors, and operators pairing `fak` with OpenAI Codex on Windows.  
**Authority:** Companion to [`docs/integrations/openai-codex.md`](../integrations/openai-codex.md), [`docs/supported/agent-harnesses.md`](../supported/agent-harnesses.md), and [`internal/harnessprofile/`](../../internal/harnessprofile/).

---

## 1. Executive Summary & Context

OpenAI's standalone **Codex Desktop Application for Windows** (shipped as an MSIX desktop package under the product identity `OpenAI.Codex` / `ChatGPT` with internal codenames `Codex` and `Owl`) represents OpenAI's dedicated desktop frontend for agentic software development. Rather than acting as a lightweight webview wrapper or cloud sandbox terminal, modern Codex Desktop orchestrates local command execution, subagent task trees (`spawn_agent`), multi-project workspaces, Model Context Protocol (MCP) tool routing, and visual Git diff reviews directly on the host machine.

This study conducts an in-depth empirical audit of:
1. The **local Windows installation and runtime architecture** on this workstation;
2. The **key UI/UX feature surfaces and internal capabilities** across the late-2024 to late-2026 release lines (including `0.153.x` / `0.154.x-alpha` core engines and frontier `gpt-6-astra` / `gpt-5.6` models);
3. **Common friction points, failure modes, and open issues** on the Windows platform (Win32 long-path errors, console popup flashes, conhost hangs, SQLite database bloat, MCP process proliferation, and approval deadlocks);
4. Concrete **architectural strategies for `fak` as an agent kernel and governance substrate** to bridge, protect, and accelerate Codex Desktop sessions without UI friction.

---

## 2. Local Installation & Runtime Architecture Witness

Inspection of the local Windows development workstation reveals an active, production Codex desktop environment:

### Package & Binary Metadata

| Component | Inspected Artifact & Value | Notes |
|---|---|---|
| **Package Identity** | `OpenAI.Codex_26.901.5280.0_x64__2p2nqsd0c76g0` | Windows MSIX / Centennial Desktop Bridge package. |
| **Publisher ID** | `2p2nqsd0c76g0` (`CN=50BDFD77-8903-4850-9FFE-6E8522F64D5B`) | Cryptographically verified Windows Store publisher. |
| **App Manifest Display** | `ChatGPT` (`AppID: OpenAI.Codex_2p2nqsd0c76g0!App`) | Shell integration registering protocol `codex://` and Explorer menu `OpenProjectInCodex`. |
| **Desktop Shell** | `openai-codex-electron` version `26.901.41600` (Build `7982`) | Electron 42.3.0 runtime (custom OpenAI shell `Owl` / `CodexBrowser`). |
| **Core Engine CLI** | `codex-cli 0.153.4` (`bin\27d6a192e9c98618\codex.exe`) | Compiled native Rust core (`codex-rs`). |
| **Active Running Processes** | `ChatGPT.exe` / `Codex.exe`, `codex.exe app-server`, `codex-code-mode-host.exe` | Multi-process desktop client paired with a background JSON-RPC app server. |
| **Local Communication** | Named pipes: `\\.\pipe\codex-browser-use-*`, `\\.\pipe\codex-computer-use-*` | High-bandwidth IPC for screen capture, automation, and REPL communication. |

### Filesystem Storage Layout

* **Application Binaries:** `C:\Program Files\WindowsApps\OpenAI.Codex_26.901.5280.0_x64__2p2nqsd0c76g0\app\`
* **CLI & Helper Tools:** `C:\Users\USER\AppData\Local\OpenAI\Codex\bin\27d6a192e9c98618\`
* **Virtualized Web Profile:** `C:\Users\USER\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Roaming\Codex\`
* **User Configuration & Workspace State:** `C:\Users\USER\.codex\` (linked via NTFS Junction to `.codex-codexFOUR`):
  * `config.toml`: Execution policy, model selector, active MCP server definitions (`[mcp_servers.fak]`).
  * `thread_history_1.sqlite` (~8.7 GB) & `logs_2.sqlite` (~1.8 GB): Structured session history, OTEL logs, and turn transcripts.
  * `state_5.sqlite`, `goals_1.sqlite`, `memories_1.sqlite`, `queue_1.sqlite`: Granular subagent task trees and persistent state.

### Native Addons & Bundled Subsystems

The desktop package bundles custom native C++ addons (`app/resources/native/`):
* `windows-account.node`: COM class integration (`d28d30e7-17f3-41d9-93f0-9bae6dc4884b`) for Windows account identity and shell context menus.
* `windows-updater.node`: MSIX background update coordinator.
* `computer-use-app-icons.node` & `hid-topology-watcher.node`: System UI automation and hardware macro peripherals (WorkLouder device kit `@worklouder/device-kit-oai`).
* **Bundled Plugins:** `@oai/sky` (`computer-use`), `unified-computer-use` (Playwright / CUA Node.js REPL), `codex-app-tools` (`create_thread`, `send_message_to_thread`, `fork_thread`, `handoff_thread`), `deep-research`, `latex`, and `visualize`.

---

## 3. UI/UX Feature Surface Breakdown

The modern Codex app features an extensive visual and workflow surface:

### A. Multi-Project Workspaces & Runtime Environments
* **Project Spaces:** Allows pinning and switching across multiple local repository directories simultaneously.
* **Execution Environment Toggle:** Supports switching agent execution context per project:
  * **Windows Native:** Host PowerShell 7, Windows PowerShell 5.1, or `cmd.exe`.
  * **WSL2:** Linux distribution execution inside Windows Subsystem for Linux.
  * **Container / Cloud Sandboxes:** Remote execution containers.

### B. Multi-Threaded Subagent Hierarchy (`spawn_agent`)
* **Subagent Drawer:** Visual drawer displaying hierarchical child agents categorized into **Active** and **Done** cards.
* **Autonomous Task Delegation:** Coordinator agents spawn bounded subagents (e.g., `coder`, `explorer`, `reviewer`) with isolated context windows and distinct worktree sandboxes.
* **Thread Persistence & Reconnect:** Preserves draft messages, agent state, and active terminal processes across app restarts and WebSocket reconnects.

### C. Visual Diffs & Turn Review
* **Side-by-Side & Unified Diff Views:** Built-in editor diff page with syntax highlighting and token-level intraline change tracking.
* **Pre-Apply / Inline Turn Diffs:** Diffs are streamed directly into the conversation turn, allowing operators to inspect proposed edits before or after execution.

### D. Approval Policies & Sandbox Controls
* **Permission Profiles:** Configurable via UI or `config.toml`:
  * `:workspace`: Read-only outside workspace; write permitted inside workspace.
  * `:read-only`: Purely exploratory and diagnostic without file mutations.
  * `danger-full-access`: Unrestricted filesystem and command execution.
* **Approval Policies (`ask_for_approval`):**
  * `always`: Prompts on every mutation or command execution.
  * `on-request`: Prompts only when commands match risk patterns.
  * `never`: Unattended / autonomous execution without interactive prompts.
* **Guardian Scoring (V1/V2):** Background LLM and heuristic safety classifier assessing destructive potential before presenting modal dialogs.

### E. Embedded Terminal & Background Process Management
* **PTY Drawer:** Embedded terminal supporting Windows terminal sequences, ANSI colors, and shell profiles.
* **Background Process Execution:** Long-running commands (e.g. dev servers, test runners) run asynchronously with live stdout/stderr tailing, input previews, and cancel/interrupt triggers.

### F. Model Catalog & Reasoning Controls
* **Frontier Model Catalog:**
  * **GPT-6-Astra:** Standard and Fast tiers ("2x speed, increased usage") with structured asynchronous user questions (`request_user_input_async`).
  * **GPT-5.6 Family:** `sol` (high reasoning), `terra` (balanced execution), `luna` (fast lightweight).
* **Reasoning Effort Meter:** Visual sliders for `Low`, `Medium`, `High`, and `Extra High`.

### G. Model Context Protocol (MCP) Management
* Dedicated UI panel under Settings to configure, enable, or disable local (`stdio`) and remote (`sse` / `http`) MCP servers.
* Per-account scoping and OAuth credential management for cloud MCP connectors.

---

## 4. Empirical Windows Friction Points & Upstream Defects

Field reports, GitHub issues, and local investigation highlight several recurring Windows-specific failure modes:

| Category | Upstream Symptom & Issue Signature | Underlying Root Cause |
|---|---|---|
| **Win32 Long Paths** | **OS Error 206 (`The filename or extension is too long`)** on launch (#40245). | Win32 path normalization inside `codex-rs` fails when merging glob denial rules with long workspace paths. Markdown composer also fails to normalize `&#x20;` and escapes underscores in file paths (`\_`), corrupting CLI commands (#41389). |
| **Terminal & Process Hangs** | **`exec_command` hangs indefinitely** on trivial PowerShell / cmd.exe calls (#39574). | Windows Console Host (`conhost.exe`) deadlock, pipe buffer stalls, and lack of timeout propagation in child process wrappers. |
| **Console Window Flashes** | **Command prompt windows flash** in the foreground during background tasks (#37153). | Spawning background CLI tools and MCP daemons without the Win32 `CREATE_NO_WINDOW` (`0x08000000`) process creation flag. |
| **Sandbox Isolation Failures** | **Windows Sandbox blocks Docker Desktop and WSL** (#41415). | Restricted-token sandboxing prevents access to Windows named pipes (`\\.\pipe\docker_engine`) and denies WSL service creation (`E_ACCESSDENIED`). |
| **MCP Process Proliferation** | **Subagents spawn duplicate MCP stacks**, leaking dozens of `node.exe` processes and CPU cycles (#42000). | Every spawned subagent eagerly instantiates its own isolated stdio MCP process tree instead of pooling or starting lazily, and completed subagents fail to reap child process trees. |
| **Database & WAL Contention** | **`logs_2.sqlite` swells to gigabytes**; `codex_app_server_client` drops events with `consumer queue is full` (#5983). | SQLite freelist bloat and slow WAL checkpoints under concurrent subagent writes freeze the UI and cause `MoAppHang 1002` crashes (#41551). |
| **Subagent Thread Leaks** | **False "agent thread limit reached"** errors with 1 Active / 12 Done subagents (#39694); thousands of hidden rows leak in `state_5.sqlite` (#40971). | Completed subagents fail to transition out of scheduler-active capacity in memory and leave `archived=0` entries in SQLite indefinitely. |
| **Approval Deadlocks & Fatigue** | **Deadlock: `AskForApproval=Never` vs. Project `prompt` rules** (#41068); typing accidentally dismisses approval dialogs (#19165). | Selecting Full Access (`never`) silently conflicts with project rules requiring confirmation, causing hard rejection without showing a dialog. |
| **Misleading Token Accounting** | **Lifetime tokens mistaken for context window size** (#38154). | Desktop UI displays lifetime cumulative tokens (`total_token_usage`) in the status bar instead of the last turn's context pressure (`last_token_usage`), making users believe their context window is exhausted. |

---

## 5. Architectural Opportunities for `fak` Harness Context

The architectural gaps in the Codex Desktop App provide clear integration opportunities for an agent kernel like `fak`:

```
┌────────────────────────────────────────────────────────┐
│             OpenAI Codex Desktop UI / TUI              │
│       (Composer, Diff Viewer, Subagents, Model)        │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │ Tool Call / Hook Boundary │
             ▼                           ▼
┌─────────────────────────┐ ┌──────────────────────────────┐
│  fak Gateway / Proxy    │ │  fak Immutable MCP Server    │
│  (/v1/responses proxy)  │ │  (fak_adjudicate, syscall)   │
└────────────┬────────────┘ └──────────────┬───────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
      ┌──────────────────────────────────────────────┐
      │               FAK AGENT KERNEL               │
      │  • Capability Floor (Default-Deny Policy)    │
      │  • Windowgate (No Console Flashes on Windows)│
      │  • Subagent Lease & Concurrency (DOS Kernel) │
      │  • MCP Multiplexing & Daemon Pooling         │
      │  • Deduplicating Compaction & Reset Budget   │
      └──────────────────────────────────────────────┘
```

### 1. Decoupled Capability Floor (Resolving Approval Fatigue)
* **Problem:** Codex desktop forces a binary choice: either suffer constant interactive approval prompts or enable `approval_policy = "never"` (`danger-full-access`), which exposes the system to catastrophic deletions or deadlocks against safety rules.
* **`fak` Solution:** Install `fak` as an immutable MCP server (`fak codex mcp install`). Codex can safely run with `approval_policy = "never"`:
  * Safe developer operations (`git status`, `go test`, file reads) are deterministically allowed without prompting.
  * Dangerous actions (`rm -rf`, `Remove-Item -Recurse -Force`, `git push`, writing to `.git/`) are structurally blocked with typed refusal tokens (`POLICY_BLOCK`, `SELF_MODIFY`) before execution.

### 2. Windowgate Process Virtualization (Zero Console Flashes)
* **Problem:** Codex on Windows flashes transient `conhost.exe` command prompts and suffers PTY pipe deadlocks during shell operations.
* **`fak` Solution:** Route background and MCP child execution through `internal/windowgate`:
  * Injects `CREATE_NO_WINDOW` (`0x08000000`) into Win32 `SysProcAttr` on all spawned tool processes.
  * Normalizes Win32 exit codes (mapping `0xFFFFFFFF` forced kills to clean abnormal termination receipts).

### 3. MCP Daemon Multiplexing & Process Pooling
* **Problem:** Every spawned subagent launches redundant copies of external stdio MCP processes (Node.js, Python), multiplying memory consumption and leaking orphan processes.
* **`fak` Solution:** Run `fak serve --stdio` as a unified, content-addressed multiplexing MCP bridge. Subagents connect to `fak` as a single lightweight client; `fak` arbitrates tool access lazily and pools external connections.

### 4. Disjoint File-Tree Arbitration for Multi-Agent Work (`dos_arbitrate`)
* **Problem:** Codex subagents collide when editing shared repositories, causing Git index locks, stale worktree state, and false thread-limit exhaustion.
* **`fak` Solution:** Integrate `dos_arbitrate` and detached worktrees (`fak worktree worker prepare|land|reap`):
  * Before subagents execute, their target file trees are checked for disjointness against active leases.
  * Changes are isolated in ephemeral worktrees, preventing trunk collisions and lock contention.

### 5. Honest Context Accounting & Output Deduplication
* **Problem:** The Codex desktop UI conflates lifetime session tokens with active context window occupancy, and repetitive tool outputs (e.g. build logs, git status) accelerate context exhaustion.
* **`fak` Solution:**
  * Accurate metric splitting: separate active turn occupancy (`last_token_usage`) from lifetime session usage (`total_token_usage`) using `fak vcache score`.
  * Content-addressed output admission (`fak_admit`): replace repeated multi-kilobyte spans with cryptographic hash references, dramatically reducing context blowup.

---

## 6. Implementation Checklist & Next Actions

1. **Maintain Dedicated Guide:** Update [`docs/integrations/openai-codex.md`](../integrations/openai-codex.md) to detail the Desktop App Windows UI/UX integration path alongside the CLI and IDE extension paths.
2. **Harmonize Harness Tables:** Update [`docs/supported/agent-harnesses.md`](../supported/agent-harnesses.md) and [`docs/integrations/harness-acceptance-checklist.md`](../integrations/harness-acceptance-checklist.md) to reflect modern Codex Responses wire and desktop app-server capabilities.
3. **Preserve Windowgate Invariants:** Ensure all Windows background process spawns in `cmd/fak` and `internal/windowgate` enforce `CREATE_NO_WINDOW`.
4. **Witness Verification:** Verify documentation freshness and link integrity across the harness documentation suite.
