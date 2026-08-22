// Package codetools is the kernel-mediated coding toolset: real Read / Write / Edit /
// Bash / Grep / Glob engines the owned agent loop dispatches to through
// abi.RegisterEngine, the same seam internal/agent/readengine.go opened for `fak_read`
// (#795) — generalized from one read-only MCP tool to the full coding surface (#6658).
//
// WHY THIS EXISTS. The fak-owned loop (internal/agent RunArm, driven live by
// `fak serve --native`) had no coding work surface: its catalog is the airline-support
// demo and the only real filesystem engine is readengine's read-only `fak_read` miss
// path. So an operator asking the native harness to perform a coding task had nothing
// to dispatch — the loop could not Read, could not mutate, could not search, and could
// not run a process. Every one of those operations has to cross the SAME in-kernel
// syscall/adjudication boundary as every other tool call, or "fak owns dispatch" stops
// being true exactly where it matters most.
//
// THE CONTRACTS ARE CODE, NOT PROMPT TEXT. A system prompt that says "stay in the
// workspace" is an instruction to a model; the model is the untrusted component. Each
// invariant below is enforced by a function on the dispatch path and witnessed by a
// behavior test:
//
//   - Workspace-root confinement and CANONICAL-path checks run BEFORE policy matching
//     (confine.go: resolve). A policy that matched on the model's raw spelling would be
//     a policy over strings, not over files: "a/../../etc/passwd" and "/etc/passwd" name
//     one file and must reach the policy as one canonical path.
//   - Symlink/traversal escape is DENIED, not followed (confine.go: evalWithin). The
//     longest existing ancestor of the target is symlink-resolved and re-confined, so a
//     symlink planted inside the tree cannot be used as a door out of it. readengine.go
//     deliberately skipped this because it was read-only; a toolset that WRITES cannot.
//   - Operation, argument, and result schemas are TYPED (args.go). Decoding is strict —
//     an unknown field is a refusal, not a silently-ignored typo that changes what the
//     call does.
//   - Read/search/process output is BOUNDED and process execution carries a DEADLINE
//     (limits.go); every engine honors ctx cancellation on entry and mid-walk.
//   - Side effects are DEFAULT-DENY (policy.go). Read/Grep/Glob are admitted by the
//     default policy; Write/Edit/Bash are not admitted until an operator policy says so,
//     and a protected path (.git, .dos) is refused for mutation regardless.
//   - Request/tool identity rides the vDSO cache scope (CallMeta): a mutating tool may
//     never be tagged readOnlyHint, and a version-bearing Read stays cache-ineligible so
//     a retry can observe peer filesystem writes. A named principal keys cacheable reads.
//   - Edit semantics are DETERMINISTIC and create/overwrite is EXPLICIT: an edit whose
//     old_string matches more than once is refused rather than guessed at, and a Write
//     over an existing file requires overwrite=true.
//   - No shell is invoked for a non-Bash tool. Grep and Glob are Go walks over
//     regexp/filepath.Match, never a subprocess; only bash.go imports os/exec, and
//     TestOnlyBashEngineImportsExec pins that.
//
// WHAT THIS PACKAGE DOES NOT DO. It does not advertise itself to a planner and it does
// not edit the owned loop's catalog or policy: Catalog() and CallMeta() are the seams a
// caller binds. Wiring them into internal/agent's ToolCatalog + Configure policy is
// tracked separately — this leaf is the mechanism, deliberately importable by the loop
// (tier 1, internal/abi only) rather than tangled into it.
package codetools
