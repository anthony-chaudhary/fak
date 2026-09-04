# Comparative Git Study: Leading Open-Source Coding Agent Harnesses
**Date:** 2026-09-03  
**Status:** Completed  
**Domain:** Agent Harness Architecture, Context Lifecycle, Tool Execution Protocols, Runtime Isolation, & Verification

---

## 1. Executive Summary & Comparative Matrix

We conducted rigorous, shallow-cloned git studies of four benchmark open-source agent harnesses representing distinct design paradigms across the AI engineering landscape:

1. **Aider (`paul-gauthier/aider`)**: The pragmatic, terminal-native pair programmer optimizing for precision editing, AST-ranked repository mapping, and fast compiler/linter reflection loops.
2. **OpenHands (`All-Hands-AI/OpenHands`)**: The enterprise autonomous software engineering platform optimizing for typed event DAGs, isolated sandboxes, invariant-safe context windowing, and distributed session leases.
3. **SWE-agent (`princeton-nlp/SWE-agent`)**: The benchmark-setting SWE engineering harness optimizing for Agent-Computer Interface (ACI) ergonomics, syntax-checked revertive edits, and persistent stateful bash runtimes.
4. **OpenCode (`anomalyco/opencode`)**: The modern, reactive, Effect-TS powered terminal harness optimizing for multi-standard skill discovery, doom-loop detection, two-tier compaction, and async permission gates.

### Comparative Dimensions Matrix

| Harness Dimension | Aider (`paul-gauthier/aider`) | OpenHands (`All-Hands-AI/OpenHands`) | SWE-agent (`princeton-nlp/SWE-agent`) | OpenCode (`anomalyco/opencode`) |
|---|---|---|---|---|
| **Primary Paradigm** | Interactive Pair Programmer & Reflector | Autonomous Headless & Web Agent Platform | Benchmark-Grade Autonomous SWE Harness | Extensible Terminal Coding Agent |
| **Architecture / Stack** | Python (prompt_toolkit, tree-sitter, networkx) | Python + TypeScript (FastAPI, React, Docker) | Python (SWE-ReX, LiteLLM, Jinja2, Docker) | TypeScript / Bun (Effect-TS, SolidJS, OpenTUI) |
| **Core Execution Loop** | Linear reflection loop with AST error re-queries | Async EventStream loop over parent-linked Event DAG | Step/Forward loop with bounded requery retries | Effect-TS service turn processor with stream state machine |
| **Context Map / Budgeting** | Tree-sitter + PageRank RepoMap with binary search | Invariant-checked View Manipulation Indices | Windowed file pager & observation history processors | Two-tier compaction (output clearance + semantic summary) |
| **Edit Mechanism** | Custom search/replace blocks (`<<<<<<< SEARCH`) | FileEditor tool + `apply_patch` unified diffs | Windowed edit with pre/post flake8 lint check | Multi-tool suite (`edit`, `write`, `apply_patch`) |
| **Execution Sandboxing** | Host PTY / subprocess with approval prompts | Persistent `tmux` socket with JSON PS1 metadata | Docker / Modal via `swe-rex` stateful bash | Local shell process with wildcard permission evaluator |
| **Error Handling / Self-Heal** | Syntax lint pass + `try_dotdotdots` fuzzy matching | Stuck detectors + LLM reset context scaling | Auto-reverting edit on lint failure + requery loop | Doom loop breaker (threshold = 3) + argument schema hints |
| **Subagent / Multi-Agent** | Architect/Editor dual-model split (`map_tokens=0`) | `DelegateExecutor` threads + git worktree isolation | ThreadPoolExecutor for batch benchmark instances | Primary/Subagent modes with step budget constraints |
| **Session Persistence** | Markdown chat log + guarded `/undo` git reset | FileLock event store + 45s TTL distributed lease | Trajectory JSON (`.traj`) + harvest-on-crash diff | SQLite database via Drizzle ORM + turn snapshots |

---

## 2. Deep Dive: Aider (`paul-gauthier/aider`)

- **Repository**: `https://github.com/paul-gauthier/aider.git`
- **Pinned SHA**: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c` (v0.86.3.dev)
- **License**: Apache-2.0

### Key Mechanisms & Source Anchors

1. **AST Dependency Graph & Personalized PageRank (`RepoMap`)**:
   - `aider/repomap.py:279-337@5dc9490b`: Uses Tree-sitter SCM queries to extract `name.definition.*` and `name.reference.*` across Python, JS/TS, Go, Rust, and C++.
   - `aider/repomap.py:470-532@5dc9490b`: Constructs a `networkx.MultiDiGraph` linking reference sites to definition files. Edge weights scale by $\sqrt{\text{refs}} \times \text{multiplier}$ (50× for active files, 10× for prompt-mentioned identifiers).
   - `aider/repomap.py:676-706@5dc9490b`: Employs a binary search over tag slices to fit the rendered repo map within a strict token budget (`max_map_tokens = 1024`).
2. **Resilient Edit Formats (`SEARCH/REPLACE`)**:
   - `aider/coders/editblock_coder.py:190-240@5dc9490b`: Implements `try_dotdotdots`, decomposing search blocks with ellipsis (`...`) into distinct contiguous segments to match elided model outputs.
   - `aider/coders/editblock_coder.py:85-124@5dc9490b`: On match failure, computes `difflib.SequenceMatcher` to generate targeted "Did you mean to match..." reflection hints.
3. **Architect / Editor Dual-Model Delegation**:
   - `aider/coders/architect_coder.py:11-48@5dc9490b`: High-reasoning architect models (o1/o3, Sonnet 3.7) plan changes; an `EditorCoder` is dynamically spawned to apply patches with `map_tokens = 0` (zero repo map overhead).
4. **Guarded Git Rollback (`/undo`)**:
   - `aider/commands.py:560-656@5dc9490b`: Confirms the target commit was authored by the current Aider session, ensures the working copy is clean, verifies it has not been pushed to `origin`, and performs a safe selective file checkout.

---

## 3. Deep Dive: OpenHands (`All-Hands-AI/OpenHands`)

- **Repository**: `https://github.com/All-Hands-AI/OpenHands.git` & `OpenHands/software-agent-sdk`
- **Pinned SHA**: `4524a919930d62535a5cdca143c8a54eaf0ede42` (OpenHands) / `07307cb8edfcd9b4675be2761df0646d075a9c36` (SDK)
- **License**: MIT License

### Key Mechanisms & Source Anchors

1. **Invariant-Checked View Manipulation Indices**:
   - `openhands-sdk/openhands/sdk/context/view/view.py:39@07307cb`: Computes valid context split points by intersecting four strict invariant properties:
     - `BatchAtomicityProperty`: Prevents splitting parallel tool execution turns.
     - `ToolCallMatchingProperty`: Enforces exact pairing between tool call IDs and observation IDs.
     - `ToolLoopAtomicityProperty` (`tool_loop_atomicity.py:14@07307cb`): Preserves or discards Anthropic thinking block sequences atomically to prevent checksum mismatch errors.
2. **Deterministic PS1 Metadata Protocol**:
   - `openhands-tools/openhands/tools/terminal/metadata.py:45@07307cb`: Injects structured JSON formatting into `$PS1`:
     `{"pid": "$!", "exit_code": "$?", "username": "\u", "hostname": "\h", "working_dir": "$(pwd)"}` bounded by escape tokens, capturing exit codes and working directory changes without regex heuristics on stdout.
3. **Path-Scoped Rules (`PathTrigger`) via Extended Content**:
   - `openhands-sdk/openhands/sdk/skills/trigger.py:40@07307cb`: Evaluates gitignore-style glob patterns against touched files. Injects micro-rules into `ObservationEvent.extended_content` on demand, keeping base system prompts small.
4. **Git Worktree Child Isolation**:
   - `openhands/src/services/child-conversation-launch.ts:23@4524a91` & `conversation_service.py:195@07307cb`: Subagents spawned with `isolation: "worktree"` receive isolated checkouts under `/tmp/conversation-worktrees/<id>` branched from trunk, preventing file lock contention.
5. **Distributed Session Leases**:
   - `openhands-agent-server/openhands/agent_server/conversation_lease.py:101@07307cb`: Coordinates server access via atomic `owner_lease.json` flock, TTL expiration (45s), and `os.kill(pid, 0)` liveness probing.

---

## 4. Deep Dive: SWE-agent (`princeton-nlp/SWE-agent`)

- **Repository**: `https://github.com/princeton-nlp/SWE-agent.git`
- **Pinned SHA**: `3ea751c087f32b16e039a2233dd6eefecef325d5` (v1.1.0)
- **License**: MIT License

### Key Mechanisms & Source Anchors

1. **Lint-Checked Auto-Reverting Edits**:
   - `tools/windowed_edit_linting/bin/edit:95-120@3ea751c`: Executes `flake8` before and after an edit. If *new* syntax errors are introduced, it executes `wf.undo_edit()` immediately and returns a diff view of the syntax fault to the agent, refusing to persist corrupted files.
2. **Requery Loop Without History Pollution**:
   - `sweagent/agent/agents.py:1088-1141@3ea751c`: Catches `FormatError`, `_BlockedActionError`, and `bash -n` syntax errors in an active retry loop (up to 3 requeries). Records attempts in `trajectory` for auditing, but discards intermediate failures from `history` once resolved to keep prompt tokens clean.
3. **Harvest-on-Crash**:
   - `sweagent/agent/agents.py:823-869@3ea751c` (`attempt_autosubmission_after_error`): On fatal runtime failures (token exhaustion, timeouts), extracts `git diff` and saves `/root/model.patch` before container teardown, salvaging partial progress.
4. **Bash Heredoc Multiline Wrapping**:
   - `sweagent/tools/tools.py:382-410@3ea751c`: Auto-wraps multiline model commands into bash heredocs (`<< 'end_of_cmd'`) to eliminate quote escaping and subshell parsing errors.

---

## 5. Deep Dive: OpenCode (`anomalyco/opencode`)

- **Repository**: `https://github.com/anomalyco/opencode.git`
- **Pinned SHA**: `03cb6324352b5e09477e56324aaaefb9e149b298` (v1.18.27)
- **License**: MIT License

### Key Mechanisms & Source Anchors

1. **Doom-Loop Circuit Breaker**:
   - `packages/opencode/src/session/processor.ts:356-378@03cb63243`: Tracks `DOOM_LOOP_THRESHOLD = 3`. If the agent executes the identical tool call tuple 3 consecutive times, execution halts and escalates to user confirmation via `permission.ask`.
2. **Two-Stage Context Compaction**:
   - `packages/opencode/src/session/compaction.ts:280-316@03cb63243`: Distinguishes between mechanical tool output clearance (`[Old tool result content cleared]`) once history exceeds `PRUNE_MINIMUM = 20,000` tokens, and semantic LLM summarization of older conversation segments, preserving recent active turns (`PRUNE_PROTECT = 40,000`).
3. **Pre-Stream Workspace Snapshotting**:
   - `packages/opencode/src/session/processor.ts:101-105@03cb63243`: Takes a filesystem snapshot *prior* to starting the LLM stream, ensuring modifications made by early-turn tool events are captured with accurate diffs.
4. **Universal Multi-Directory Skill Discovery**:
   - `packages/opencode/src/skill/index.ts:20-25@03cb63243`: Interoperably resolves skills across `.claude/skills/**/SKILL.md`, `.agents/skills/**/SKILL.md`, and `.opencode/skills/**/SKILL.md`.

---

## 6. Synthesis & Key Transferable Patterns for Agent Kernels (`fak`)

From studying these four harnesses, four foundational design invariants emerge for agent kernels:

1. **Context Slicing Must Be Invariant-Safe (OpenHands `ViewManipulationIndices`)**:
   - Truncating or summarizing message histories cannot be a blind string or array slice. Kernels must maintain atomic boundaries for tool batches and reasoning/thinking signatures.
2. **Edit Transactions Must Self-Verify and Revert (SWE-agent `undo_edit` & Aider `try_dotdotdots`)**:
   - Tool execution layers should gate file mutations behind syntax lint passes (e.g. `gofmt` / `go vet` in Go workspaces). Corrupting edits must auto-revert before disk commitment.
3. **Transient Requeries Must Not Pollute Prefix Caches (SWE-agent `trajectory` vs `history`)**:
   - Intermediary syntax retries and format corrections should be preserved in audit logs but purged from model prompt histories to avoid prompt pollution and preserve KV-cache stability.
4. **Deterministic Host Boundary via Structured Prompts & Circuit Breakers (OpenHands PS1 & OpenCode Doom Breaker)**:
   - Shell interactions should use structured exit status protocols (like JSON PS1 formatting) rather than regex matching, paired with hard doom-loop limits on repeating tool errors.
