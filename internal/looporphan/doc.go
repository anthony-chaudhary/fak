// Package looporphan is the pure duplicate-loop-supervisor reaper core: it folds
// a process census of loop/drainer SUPERVISORS into a closed keep/reap plan that
// never strands live work.
//
// Tier: foundation (1) - see internal/architest. This package imports only the
// standard library; the impure census (procguard.CollectRelations) and the
// parent-liveness probe (looprecover.Probe) live in the shell (cmd/fak) and hand
// this core plain data. See AGENTS.md and internal/architest for the layering
// contract.
//
// # Why this leaf exists
//
// The recurring, hand-remedied pain is OS-level, not semantic: overlapping
// DETACHED loop/drainer engine processes (`fak loop drive`, `dos loop --json`, a
// drainer) that outlive their owning /goal session's compaction (on Windows
// TaskStop is not a kill), keep looping, and burn account seats. The manual fix
// documented in the resume-consolidate-duplicate-loops note is: enumerate the
// duplicate supervisors by command line, then kill all but the one still
// parenting live work.
//
// No committed detector consolidates them. The sibling internal/doomloop is a
// different axis entirely - it classifies ONE live worker's effort-vs-progress
// SEMANTICS, with the worker's liveness an injected bit; it holds no process
// census and structurally cannot see three duplicate detached engines. The
// adjacent OS reapers each target something else: procguard's orphan-sprawl set
// is only {dos_mcp.server} and its classes exclude a supervisor that still has a
// live child; the fleet janitor PROTECTS loop ancestors; looprecover works at the
// ledger level and deliberately leaves live orphans running. This leaf is the
// missing "keep the one parenting live work, reap the idle duplicates" core.
//
// # The census the core folds
//
// Each Supervisor is one candidate loop-engine process the shell already matched
// by command-line marker: its PID/PPID, a start-time fence (Start, the same
// reuse-safe fingerprint procguard.Proc.Start and looprecover.Probe compare), the
// raw Cmdline, a parsed Lane identity (the loop's --lane/goal, "" if unknown), the
// tri-state liveness of its owning session/parent (Parent: alive/dead/unknown,
// from looprecover.Probe), and how many live `fak c` workers its process subtree
// contains (LiveWorkers). The core reads only this data - it makes no syscalls and
// kills nothing.
//
// # The keep/reap rule (reversible-first, fail-closed)
//
// Supervisors are grouped by Lane (falling back to Cmdline). Within a group:
//
//   - The one(s) PARENTING LIVE WORK are never reaped. Exactly one live keeper
//     is KEEP; two-or-more live supervisors for the same lane is a COLLISION
//     (real duplication, but reaping either strands live work) - an operator
//     decision the core will not auto-resolve.
//   - Idle duplicates (no live worker) of a group that has a keeper are REAP.
//   - A lane with NO live work: an idle supervisor whose parent is gone is an
//     orphan - REAP; one whose parent is still alive is an attached idle loop
//     between workers - KEEP.
//
// Two fail-closed guards bound every REAP: a supervisor with no start-time fence
// (StartUnixMs <= 0) or no identity (empty group key) yields UNKNOWN, never REAP,
// so the core never asks the shell to kill a PID it cannot fence against reuse.
// The core NEVER emits a hard kill itself - REAP is a RECOMMENDATION; the shell
// (`fak loop reap`) reports by default and acts only behind an explicit --reap,
// then kills via procguard.KillPID with the janitor's protection layers.
package looporphan
