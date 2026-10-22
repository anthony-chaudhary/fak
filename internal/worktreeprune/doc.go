// Package worktreeprune is the shared refusal for a repository-wide
// `git worktree prune`, and the targeted cleanup that replaces it.
//
// A broad prune deletes the .git/worktrees/<name> admin entry of every unlocked
// registration whose gitdir it cannot stat. A registration that git wrote on the
// other side of a Windows/WSL host (C:/... read from Linux, /mnt/... or /tmp/...
// read from Windows) can never be stat-ed, so one broad prune from the wrong side
// strands every such lane. GuardBroadPrune refuses that prune; CleanupOwn and
// RemoveOwnRegistration remove only the caller's own registration.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package worktreeprune
