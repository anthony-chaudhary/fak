// Package shadowgit attributes every file write an agent makes to the exact step that
// made it, without touching the repo the agent is working in.
//
// # The mechanism
//
// It drives ordinary git against the agent's worktree through a SEPARATE git
// directory — `git --git-dir=<shadow> --work-tree=<repo>` — so the real repo's own
// .git (its index, HEAD, refs, hooks) is never read or written. The shadow repo is a
// private, throwaway ledger: at each agent step it stages the whole worktree and
// commits a snapshot, and the diff between consecutive snapshots is the precise set of
// files that changed during that step. The shadow index is the diff cache that makes
// the attribution O(changed files), not O(worktree).
//
// This is the witness layer under "which step wrote this byte?": a Turn says a tool
// was allowed to run; a shadow snapshot proves what it actually changed on disk. It is
// author-neutral in the same spirit as the dos commit-audit syscall — the diff is what
// git recorded, not what the agent claimed.
//
// # Non-invasive by construction
//
//   - All git invocations carry --git-dir=<shadow> --work-tree=<repo>; the real .git
//     is out of the command's addressable state entirely.
//   - The shadow's info/exclude always ignores the real .git/ and (when the shadow dir
//     lives inside the worktree) the shadow dir itself, so a snapshot never recurses
//     into either repo's metadata.
//   - Snapshots honor the worktree's .gitignore by default (no node_modules noise);
//     [Options.IncludeIgnored] captures ignored writes too when an audit needs them.
//
// # Output
//
// [ShadowGit.Snapshot] returns a [Snapshot] (step, shadow commit, and the []Change
// since the previous snapshot). [WriteChangelogLine] appends it to a state_changelog
// JSONL — the per-step write ledger a checkpoint (#2394) or a trajectory joins against
// by step to bind step -> bytes-changed.
//
// Borrowed from agent-lens@9eab2b0 (StateManager / shadow_git.py: commit_baseline,
// snapshot, check_for_writes).
package shadowgit
