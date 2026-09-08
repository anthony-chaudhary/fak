---
name: ban-powershell-scripts-mandatory-go-programs
description: "User directive 2026-09-08 — Strict ban on new PowerShell (.ps1) or other scripts across fak and fak-private; all automation and tooling MUST be native Go programs designed for modular, integrated, long-term value."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: prompt-ban-powershell-scripts-2026-09-08
---

# Mandate: Strict Ban on PowerShell & Loose Scripts — Modular Go Programs Only

**Directive (2026-09-08):** "New BAN in this repo any .powershell or other types of scripts unless super super heavily justified reason (basically like 1 in whole repo). MUST be actual go lang program instead of random new powerhell scripts. think modular integrated long term value,. sub agents make notes for both repos"

## Core Policy
1. **Zero-Tolerance Ban on Scripts:**
   - Authoring or committing new PowerShell scripts (`.ps1`), POSIX/Bash shell scripts (`.sh`), batch files (`.bat`, `.cmd`), or ad-hoc glue scripts is **STRICTLY BANNED** across `fak` and companion `fak-private`.
   - The "~1 Justified Exception" Rule: An exception is permitted ONLY under extraordinary, super heavily justified circumstances (basically ~1 in the whole repo, such as initial bare-metal host bootstrapping before a Go toolchain is installed).
2. **Mandatory Native Go Programs:**
   - All automation, workflows, background orchestrators, test runners, session dispatchers, diagnostic probes, migration helpers, or deployment scripts **MUST be implemented as native Go programs**.
   - Implement them as subcommands within established Go CLI suites:
     - In `fak`: `cmd/fak` (registered CLI verbs backed by `internal/<pkg>`), `cmd/fak-dev`, or quarantined sub-modules under `tools/<name>/go.mod`.
     - In `fak-private`: `cmd/fak-sync`, `cmd/fak-flow`, `cmd/fak-strix`, `cmd/fak-installer`, `cmd/fak-boundary`, `cmd/fak-server`.
   - Or introduce a dedicated, clean binary under `cmd/<tool-name>`.
3. **Architectural Rationale: Modular, Integrated, Long-Term Value:**
   - **Static Typing & Compile Verification:** Go programs catch typos, argument mismatches, and syntax errors at build time via `go vet` and `go test`, eliminating runtime failures common in loose scripts.
   - **Cross-Platform Portability:** PowerShell scripts break on Linux/macOS; shell scripts fail under Windows PowerShell or Git Bash. Native Go programs compile portably across all platforms (`GOOS=windows`, `GOOS=linux`, `GOOS=darwin`).
   - **Modular Integration & Reuse:** Go tools import shared internal packages (`internal/*`, `pkg/*`), leveraging typed data models, logging, HTTP clients, receipts, and file locks instead of duplicating ad-hoc text parsing.
   - **Discoverability & Unified Ergonomics:** Unified CLI commands with `--help`, JSON output flags (`--json`), and typed receipts provide professional, discoverable ergonomics compared to dozens of scattered script files.
   - **Pre-Commit Enforcement:** The git pre-commit filter refuses un-grandfathered scripts (`FILE_ADMISSION`).

## Subagent & Worker Operational Rules
When any coordinator or subagent (`worker`, `general`, `tester`, `lander`, `issue-orchestrator`, etc.) is dispatched:
- **NEVER author new `.ps1` or `.sh` files** to execute steps, run test matrices, or spawn background jobs.
- **Extend Existing Go CLIs:** Add subcommands, flags, or helpers to existing Go tools rather than reaching for quick scripts.
- **Migrate on Touch:** Existing legacy scripts in `tools/` or `scripts/` should be actively ported to Go subcommands when modified; never cite existing scripts as justification for adding new ones.
