# Cross-provider and cross-harness session interchange: portability, state migration, and tool mediation

**Date:** 2026-09-02  
**Issue:** [#10757](https://github.com/anthony-chaudhary/fak/issues/10757)  
**Status:** Normative research and architecture design note  
**Lane:** `session`  
**Centrality:** Core  

---

## 1. Executive Summary & Problem Framing

As agentic development matures from single-CLI interactions into continuous, heterogeneous workflows, operators encounter severe friction:
1. **Model provider lock-in & quota cliffs:** An agent running on Anthropic Claude 3.7 Sonnet reaches an hourly rate-limit (429) or weekly tier spend ceiling mid-task. The developer cannot seamlessly continue on OpenAI GPT-4o or Gemini 2.5 Pro without discarding context, re-explaining the goal, and paying full re-prefill penalties from scratch.
2. **Harness fragmentation:** Work initiated in a terminal CLI (e.g. Claude Code or OpenCode) cannot be inspected, continued, or debugged in an IDE (e.g. Cursor / VS Code) or browser console without manual copy-paste handoffs.
3. **Multi-agent heterogeneous delegation:** A supervisor agent running in one harness cannot safely delegate a sub-phase to a specialized external worker (e.g., a Codex or local in-kernel Qwen worker) and ingest the completed session state back into its primary execution thread.

### The Core Axiom in `fak`

In `fak`, **a session is a logical task and interaction history**, not a process handle, a socket, or an ephemeral prompt-cache allocation. 

As established in `internal/sessionimage`:
> `Portability.KVIncluded = false` and `ContentModelAgnostic = true`. The KV cache is a cache, rebuilt on resume; logical content survives a model change.

Furthermore, `docs/session-client-contract.md` establishes the separation between:
- **Logical Session** (`session_id`): Stable objective, addressed journal, transcript/effect lineage, policy, and budgets.
- **Execution Epoch** (`execution_epoch`): Replaceable tuple of runtime bindings: `(harness, provider, account, model, compute)`.
- **Client Attachment** (`attachment_id`): Ephemeral connection with a single-writer input lease and multi-reader presence.

Today, `fak` already implements `.faksession` archive dumping/rehydration (`internal/sessionimage`), cross-compute placement movements (`internal/gateway/session_move.go`), and trajectory projection (`internal/atif`). However, true cross-provider and cross-harness interchange has remained incomplete because existing systems conflate **episodic task state** with **vendor-specific ephemeral preambles and tool calling wire dialects**.

This research note defines the architecture, canonical data model, tool mediation matrix, and migration state machine required to achieve zero-loss cross-provider and cross-harness session sharing.

---

## 2. The Multi-Dimensional Divergence Matrix

To understand why naive transcript sharing fails, we examine how state, instructions, and calling conventions differ across the five primary developer environments:

| Dimension | Anthropic / Claude Code | OpenAI / Codex CLI | Google / Gemini CLI | OpenCode | fak Native / In-Kernel (Qwen) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **System Preamble** | ~10k–30k tokens proprietary instructions (`STDLIB`, style guides, hidden instructions) | Developer instructions, `AGENTS.md` parser, bash prompt | System instructions, tool schemas, safety filters | OpenCode system prompt, agent personas, tool schemas | In-kernel prompt templates / ChatML tokens |
| **Wire Dialect** | Content blocks: `tool_use` (id, name, input) + `tool_result` | Function calls: `tool_calls` array (id, type, function) + `role: "tool"` | `parts` array: `functionCall` + `functionResponse` | Content blocks / provider-mapped tools | ChatML / XML tool call tokens (`<tool_call>`) |
| **Tool Taxonomy** | `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep`, `TodoWrite` | `exec_command`, `file_search`, `patch_apply` | Custom chat tool structures | `read`, `edit`, `write`, `bash`, `glob`, `grep`, `todowrite` | vDSO / syscall boundary (`fs:read`, `fs:write`, `shell:exec`) |
| **On-Disk Session Store** | `~/.claude/projects/<hash>/<uuid>.jsonl` | `~/.codex/sessions/...` | `~/.gemini/tmp/**/chats/*.json` | `.opencode/` or `~/.local/share/opencode/` | `.faksession` tar bundle (`image.json`, `session.json`, `cas.json`) |
| **Context Window** | 200,000 tokens | 128,000 tokens | 1,000,000–2,000,000 tokens | Variable (follows routed model) | 32,768–131,072 tokens |
| **Prompt Cache Lineage** | Anthropic 5m/1h ephemeral cache | OpenAI automatic prompt cache | Gemini context caching | Provider-specific | Local KV-prefix reuse (`radixkv` / in-kernel) |
| **Stateful Sidecars** | Internal todo store, memory file (`MEMORY.md`) | In-memory thread state, lock files | In-memory chat tree | Session JSON / SQLite memory | `session.Table` drive state, recall CAS |

### Analysis of the Failure Modes

1. **Preamble Poisoning & Confusion:**  
   If a Claude Code transcript is fed directly to GPT-4o or OpenCode, the foreign system prompt creates severe model confusion. The destination model attempts to call Anthropic-specific tools or follow Anthropic markdown formatting constraints, resulting in hallucinated tool calls and syntax rejections. Furthermore, carrying 25k tokens of obsolete system instructions across turns incurs substantial financial waste.
2. **Tool Schema Rejection:**  
   When Anthropic generates a `tool_use` content block with `Read` and arguments `{"file_path": "foo.go", "offset": 1, "limit": 100}`, an OpenAI endpoint rejects it unless converted to a `tool_calls` array with function `file_search` or `exec_command`. If a tool call cannot be evaluated by the destination harness, the conversation loop deadlocks.
3. **Context Overflow & Cache Cliff:**  
   Transferring a 180k-token session from Gemini 2.0 to GPT-4o (128k limit) or a local Qwen 2.5/3.8 instance (32k limit) causes an immediate context length error unless compacted. Furthermore, moving between cloud providers destroys the provider-side prompt cache, causing a 100% cold write prefill spike on the first turn.

---

## 3. The Decoupled Session Model

To solve preamble poisoning and dialect lock-in, `fak` introduces the **Decoupled Session Model**:

```
+-------------------------------------------------------------------------+
|                          LOGICAL SESSION                                |
|                                                                         |
|  1. Canonical Metadata (ID, objective, cwd, git HEAD, branch, created)  |
|  2. Canonical Task State (active plan, todos, verified constraints)     |
|  3. Working Set & Verified Effects (touched files, diffs, test state)  |
|  4. Episodic Interaction History (canonical turn steps & tool results)  |
+-------------------------------------------------------------------------+
                                    |
                    Export / Transpile Boundary
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-------------------------------+       +-------------------------------+
|     HARNESS ADAPTER A         |       |     HARNESS ADAPTER B         |
|        (Claude Code)          |       |          (OpenCode)           |
|                               |       |                               |
|  - Native System Preamble     |       |  - Native System Preamble     |
|  - Content Block Dialect      |       |  - OpenCode Tool Definitions  |
|  - Tools: Read, Edit, Bash    |       |  - Tools: read, edit, bash    |
+-------------------------------+       +-------------------------------+
```

### State Segregation Rules

1. **Episodic Task Context (Must Move):**
   - User intent and prompts
   - Assistant rationales and high-level reasoning
   - Canonical tool invocations and verified results
   - File modification receipts and working-tree git status
   - Task plan and todo checklists
2. **Ephemeral Preamble (Must Be Stripped and Replaced):**
   - Harness-specific instructions, persona prompts, and system preambles
   - Internal harness function call wrappers
   - Provider-specific formatting instructions
   - On migration, the destination harness reconstructs its own native preamble.
3. **Runtime Local Bindings (Placement-Specific):**
   - OS process handles (PID), PTY file descriptors
   - Remote provider connection IDs and bearer tokens
   - Provider prompt-cache handles (KV cache is rebuilt on resume)

---

## 4. Canonical Session Interchange Schema (CSIS v1)

`fak` specifies `fak.csis.v1` as the lossless, provider-neutral, harness-neutral interchange schema.

```json
{
  "schema": "fak.csis.v1",
  "metadata": {
    "session_id": "sess_01J6X9R4...",
    "parent_id": "",
    "title": "Refactor gateway session routing",
    "objective": "Move execution epoch handling into session_move.go",
    "created_unix_ms": 1756828800000,
    "updated_unix_ms": 1756832400000,
    "working_directory": "/workspace/fak",
    "git": {
      "branch": "main",
      "head_commit": "e16eea27e...",
      "clean": false
    }
  },
  "task_state": {
    "active_plan": [
      {
        "id": "step-1",
        "content": "Add Harness field to SessionPlacement",
        "status": "completed",
        "priority": "high"
      },
      {
        "id": "step-2",
        "content": "Implement tool dialect transpiler",
        "status": "in_progress",
        "priority": "high"
      }
    ],
    "user_preferences": {
      "language": "Go",
      "test_framework": "native"
    },
    "verified_facts": [
      "dos.toml registers session and sessionimage lanes"
    ]
  },
  "working_set": {
    "files_read": ["internal/gateway/session_move.go", "dos.toml"],
    "files_modified": ["internal/gateway/session_move.go"],
    "active_diff_digest": "sha256:7f83b1..."
  },
  "history": [
    {
      "step_index": 0,
      "turn_id": "turn_001",
      "role": "user",
      "content": "Inspect gateway session movement for harness support"
    },
    {
      "step_index": 1,
      "turn_id": "turn_002",
      "role": "assistant",
      "reasoning": "I need to check internal/gateway/session_move.go to see if Harness is declared in SessionPlacement.",
      "tool_invocations": [
        {
          "call_id": "call_001",
          "canonical_tool": "fs:read",
          "args": {
            "path": "internal/gateway/session_move.go",
            "offset": 1,
            "limit": 50
          }
        }
      ]
    },
    {
      "step_index": 2,
      "turn_id": "turn_003",
      "role": "tool",
      "tool_results": [
        {
          "call_id": "call_001",
          "canonical_tool": "fs:read",
          "status": "success",
          "output": "package gateway\n...",
          "output_digest": "sha256:abcd...",
          "bytes": 1420
        }
      ]
    }
  ]
}
```

---

## 5. Tool Mediation & Transpilation Matrix

When importing a session history into a destination harness or re-prefilling to a new model provider, canonical tool calls must map cleanly into destination dialects:

| Canonical Tool | Claude Code | OpenCode | OpenAI Codex CLI | Gemini CLI | Local / ChatML |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fs:read`** | `Read(file_path, offset, limit)` | `read(filePath, offset, limit)` | `file_search(path)` or `exec_command("cat ...")` | `read_file(path)` | `<tool_call>{"name":"fs_read",...}</tool_call>` |
| **`fs:write`** | `Write(file_path, content)` | `write(filePath, content)` | `exec_command("tee ...")` | `write_file(path, content)` | `<tool_call>{"name":"fs_write",...}</tool_call>` |
| **`fs:edit`** | `Edit(file_path, old_string, new_string, replace_all)` | `edit(filePath, oldString, newString, replaceAll)` | `patch_apply(patch)` or `exec_command("sed ...")` | `apply_patch(...)` | `<tool_call>{"name":"fs_edit",...}</tool_call>` |
| **`shell:exec`** | `Bash(command, timeout)` | `bash(command, timeout, workdir)` | `exec_command(cmd)` | `terminal_run(command)` | `<tool_call>{"name":"shell_exec",...}</tool_call>` |
| **`search:glob`**| `Glob(pattern, path)` | `glob(pattern, path)` | `file_search(glob)` | `glob(pattern)` | `<tool_call>{"name":"glob",...}</tool_call>` |
| **`search:grep`**| `Grep(pattern, path, include)` | `grep(pattern, path, include)` | `exec_command("rg ...")` | `grep(pattern)` | `<tool_call>{"name":"grep",...}</tool_call>` |
| **`task:plan`**  | `TodoWrite(todos)` | `todowrite(todos)` | Markdown checklist in context | Plan parts in chat | `<tool_call>{"name":"plan_update",...}</tool_call>` |

### Two Modes of Tool History Rehydration

When resuming across harnesses, the adapter chooses one of two projection strategies based on destination capability:
1. **Full-Fidelity Synthetic Replay:**  
   Every past canonical turn is synthesized into the destination harness's wire schema (e.g. converting Anthropic `tool_use`/`tool_result` blocks into OpenAI `tool_calls` and `role: "tool"` messages). Used when the destination model has sufficient context window and full function calling support.
2. **Compacted Effect Summary (Context Distillation):**  
   If the destination model has a restricted context budget (e.g. moving from Gemini to a 32k local Qwen) or mismatched tool capabilities, intermediate tool calls are squashed into an authoritative **Prior Session Summary**:
   - Objective & user instructions
   - Files read & key facts discovered
   - Current git diff and modified files
   - Active todo/plan status
   This frees up to 80% of the token budget while preserving 100% of task continuity.

---

## 6. Execution Epoch Movement with Harness Placement

In `internal/gateway/session_move.go`, `SessionPlacement` must be extended to include `Harness`:

```go
type SessionPlacement struct {
    Harness              string   `json:"harness"`               // e.g. "claude-code", "opencode", "codex", "gemini", "fak-native"
    HarnessConfig        string   `json:"harness_config,omitempty"` // path or profile reference
    Provider             string   `json:"provider"`              // e.g. "anthropic", "openai", "gemini", "fak-in-kernel"
    AccountRef           string   `json:"account_ref"`          // destination-resolved credential ref
    Model                string   `json:"model"`                // e.g. "claude-3-7-sonnet", "gpt-4o", "qwen-3.8"
    Compute              string   `json:"compute"`              // e.g. "local-workstation", "gcp-l4", "da33"
    Capabilities         []string `json:"capabilities,omitempty"`
    ContextLimit         int64    `json:"context_limit,omitempty"`
    BudgetAvailable      int64    `json:"budget_available,omitempty"`
    ComputeAvailable     bool     `json:"compute_available"`
    CacheLineage         string   `json:"cache_lineage,omitempty"`
    SemanticDegradations []string `json:"semantic_degradations,omitempty"`
}
```

### The 5-Stage Transition State Machine

```
RUNNING (Epoch E1: Claude Code / Anthropic Sonnet)
  |
  +---> 1. SAFE_POINT_REQUESTED
  |        Wait for turn boundary (post-tool-result, before model dispatch).
  |
  +---> 2. CHECKPOINTED
  |        Emit Canonical Session Interchange Schema (CSIS) bundle.
  |        Strip harness preamble; compute transcript & effect SHA-256 digests.
  |
  +---> 3. DESTINATION_ADMITTED
  |        Destination adapter verifies:
  |        - Context limit >= working-set requirements (or schedules compaction).
  |        - Budget and destination credentials resolve.
  |        - Evaluates cache posture (calls internal/resume.Plan).
  |
  +---> 4. RESTORED
  |        Destination harness synthesizes native preamble.
  |        Materializes working directory and sets new execution epoch E2.
  |        Generates provider witness verifying model continuation.
  |
  +---> 5. CUTOVER_COMMITTED
           Atomic lease transition in session.Table.
           Source epoch marked closed/migrated; destination epoch is now active.
```

If an error occurs before `CUTOVER_COMMITTED`, the source epoch remains authoritative (`ROLLBACK`).

---

## 7. Operational Tooling & CLI Interfaces

To give operators and agents direct control over session sharing, `fak` will provide both offline tooling and online gateway routes:

### Offline Transpiler & Discovery Verbs

```bash
# Discover sessions across all harnesses on the local machine
fak session discover --all-harnesses
# Output: Claude Code (12), Codex (5), OpenCode (8), Gemini (2)

# Export a session from Claude Code into the Canonical Session Interchange Schema
fak session export --id sess_01J6X9R4 --format csis --out session.csis.json

# Transpile a Claude Code session directly into an OpenCode session
fak session transpile --from claude --to opencode --input ~/.claude/projects/.../sess.jsonl --out .opencode/sessions/sess.json

# Import an external session into fak's portable archive
fak session import --file session.csis.json --as-faksession ~/.fak/sessions/sess_01J6X9R4.faksession
```

### Online Gateway Attachment (`fak serve`)

When running `fak serve`, the gateway serves an OpenAI/MCP-compatible endpoint. Any client (OpenCode, Cursor, or terminal) can attach to an active session by passing:
`X-Fak-Session-ID: sess_01J6X9R4`
`X-Fak-Epoch-Cutover: true`

The gateway automatically handles protocol translation, tool mediation, and cache-residence planning.

---

## 8. Security, Privacy & Integrity Boundaries

1. **Credentials Never Travel:**  
   Account credentials, API keys, and bearer tokens must **never** enter the interchange format (`AccountRef` is a destination-resolved pointer only). Cross-host migration requires the destination machine to resolve its own credential under destination policy.
2. **Deterministic Integrity Index:**  
   Like `sessionimage`, every CSIS bundle carries SHA-256 digests over its task state, working set diffs, and turn history. Tampered or truncated exports fail admission closed.
3. **Structured Redaction:**  
   Following `internal/atif` conventions, session exports default to standard redaction (stripping environment secrets and sensitive bash outputs unless `--full-fidelity` is explicitly requested).

---

## 9. Decomposed Implementation Roadmap

This research decomposes into five sequentially dispatchable implementation leaves:

```
                          [#10757 Research & Architecture]
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
[Leaf 1: CSIS Schema]        [Leaf 2: Harness Placement]     [Leaf 3: Tool Transpiler]
(internal/csis data types)   (gateway.SessionPlacement)      (Claude <-> OpenCode <-> Codex)
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
                            [Leaf 4: CLI Export/Import]
                            (fak session transpile/export)
                                         |
                                         v
                            [Leaf 5: Admission & Gating]
                            (Context limit & cache re-entry)
```

1. **Leaf 1 — Canonical Session Interchange Schema (`internal/csis`):**  
   Define the Go data models, JSON serialization, and SHA-256 validation for `fak.csis.v1`.
2. **Leaf 2 — Harness Placement Axis (`internal/gateway/session_move.go`):**  
   Add `Harness` and `HarnessConfig` to `SessionPlacement`; update `FinalizeSessionMoveCheckpoint` and `AdmitSessionMoveCheckpoint`.
3. **Leaf 3 — Dialect & Tool Mediation Transpiler (`internal/csistranspile`):**  
   Implement bidirectional mapping between canonical tool definitions and Claude Code, OpenCode, Codex, and Gemini formats.
4. **Leaf 4 — Offline Session CLI (`cmd/fak/session_transpile.go`):**  
   Add `fak session export`, `fak session import`, and `fak session transpile` CLI commands.
5. **Leaf 5 — Context Adaptation & Cache Admission (`internal/gateway/session_move_admit.go`):**  
   Integrate `internal/resume.Plan` and `internal/ctxmmu` working-set compaction into the migration admission check.

---

## 10. Conclusion

By separating **episodic task state** from **ephemeral harness preambles** and mediating tool calling dialects through a canonical interchange format, `fak` turns agent sessions into universal, portable execution assets. Operators gain freedom of movement across frontier model providers and harnesses without sacrificing context, safety, or token economics.
