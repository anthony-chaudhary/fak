// gitmaint.go is the SAFE object-DB consolidation half of gitgate: where the
// Adjudicate rung (gitgate.go) provably REFUSES the destructive git verbs
// (gc --prune=now, repack -adb, worktree prune), this file provides the guarded
// forward path the guard otherwise only defers — a lock-aware, posture-asserting
// "consolidate-never-prune" maintenance run over an always-hot, shared, multi-worktree
// object DB. It is the executor behind `fak git-maint` (cmd/fak/git_maint.go).
//
// WHY IT IS SAFE ON A HOT CLONE. Object-DB maintenance is orthogonal to working-tree
// hotness: it reads .git/objects + refs and never touches the tree, so it is safely
// automatable while 80–100 files are dirty across peer sessions. The ONLY risk is
// object DELETION, which this path forbids outright. Two tiers, gated differently:
//
//   - ALWAYS-SAFE (RunMaint runs it unconditionally, even mid-commit): `git
//     multi-pack-index write` and `git commit-graph write --reachable`. Both are
//     add-only and atomic — they build/refresh an index file and delete nothing, so a
//     concurrent commit cannot be harmed. Verified 2026-07-04 on this box: both ran +
//     `verify`-clean with the loose/pack object counts unchanged.
//
//   - SAFE-WITH-GRACE (gated on BOTH the lock preflight AND the posture assert):
//     `git maintenance run --task=loose-objects` (prune-packed drops only loose
//     objects that ALREADY have a packed duplicate) and `--task=incremental-repack`
//     (multi-pack-index expire removes only fully-covered redundant packs; on Windows
//     an open pack simply fails to unlink → fail-safe). These UNLINK redundant copies,
//     so they run only when no git/fak transaction is in flight and the shared
//     no-auto-gc posture still holds — and the locks are RE-CHECKED before every
//     mutating step (TOCTOU), so a peer that starts committing mid-run defers the rest.
//
// WHAT IT NEVER DOES (the forbidden set, mirroring gitgate.go's refusals): never
// `--prune=now` / bare `git prune` (they ignore gc.pruneexpire and drop unreachable
// loose immediately), never `git gc` / `--task=gc`, never a full `repack -a/-A/-adb`,
// never `git worktree prune`, never `git maintenance register/start` (those write the
// GLOBAL ~/.gitconfig), never edits .git/config. It only ever ADDS a midx/commit-graph
// and folds redundant loose/pack copies that git itself proves are already covered.
//
// Like treedoctor/safecommit, every git effect goes through an injected MaintRunner so
// the whole tier-gating + preflight decision tree is unit-testable with no git and no
// real repo — see gitmaint_test.go.
package gitgate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// MaintRunner runs a git command with working dir `dir`, returning combined
// stdout/stderr, the process exit code, and an error only when git could not be
// executed at all. It mirrors treedoctor.Runner / safecommit.Runner so the git-maint
// decision tree is testable with no git and no repo.
type MaintRunner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// MaintOptions configures a safe object-DB consolidation run.
type MaintOptions struct {
	// RepoRoot is the working dir every maintenance git verb runs from (the main
	// worktree). git resolves the shared object DB from here.
	RepoRoot string
	// GitCommonDir is the shared `.git` (from `git rev-parse --git-common-dir`) that
	// holds objects/, config, and the lock files the preflight probes. For the main
	// worktree this is <RepoRoot>/.git; a linked worktree still shares this one dir.
	GitCommonDir string
	// Apply performs the consolidation. When false the run is a DRY-RUN: it probes
	// locks + posture and the before-count and reports the plan, mutating nothing.
	Apply bool
}

// MaintReason is a structured, closed-vocabulary reason the safe-with-grace tier was
// held back — never free-text, so a loop (and the tests) can match on it.
type MaintReason string

const (
	// MaintReasonLocked: a live git/fak lock in the common dir means a concurrent
	// transaction is in flight, so a pack-unlinking step could race it. The grace tier
	// is DEFERRED (not an incident — a rerun when quiet consolidates); the always-safe
	// tier still ran.
	MaintReasonLocked MaintReason = "LOCKED"
	// MaintReasonPostureDrift: the shared safe-maintenance posture has drifted
	// (gc.auto != 0 or maintenance.auto is not an explicit false) — an unsupervised
	// auto-gc could prune-race. The grace tier REFUSES and the drift is surfaced as an
	// incident for an operator to repair the shared config.
	MaintReasonPostureDrift MaintReason = "POSTURE_DRIFT"
)

// alwaysSafeSteps are the add-only, atomic, safe-even-mid-commit git verbs — they
// build/refresh an index and delete nothing.
var alwaysSafeSteps = [][]string{
	{"multi-pack-index", "write"},
	{"commit-graph", "write", "--reachable"},
}

// graceSteps are the redundant-copy folds — each unlinks ONLY objects/packs git has
// proven are already covered, and only runs when unlocked + posture-safe.
var graceSteps = [][]string{
	{"maintenance", "run", "--task=loose-objects"},
	{"maintenance", "run", "--task=incremental-repack"},
}

// maintLockNames is the fixed set of common-dir lock files whose presence means a
// concurrent git/fak transaction is in flight; a pack-unlinking step must not run
// while any is held.
var maintLockNames = []string{
	"index.lock",
	"gc.pid",
	"packed-refs.lock",
	"fak-commit.lock",
	"maintenance.lock",
	filepath.Join("objects", "info", "commit-graph.lock"),
	filepath.Join("objects", "pack", "multi-pack-index.lock"),
}

// Posture is the read of the two shared-config knobs that keep automatic,
// unsupervised maintenance (which CAN prune) from ever firing on the hot clone.
type Posture struct {
	GCAuto          string `json:"gc_auto"`          // configured gc.auto ("" = unset → git default 6700, unsafe)
	MaintenanceAuto string `json:"maintenance_auto"` // configured maintenance.auto ("" = unset → git default true, unsafe)
	Safe            bool   `json:"safe"`             // gc.auto == 0 AND maintenance.auto is an explicit git-false
	Drift           string `json:"drift,omitempty"`  // human reason when !Safe
}

// CountObjects is the parsed `git count-objects -vH` snapshot for the before/after
// witness. Raw is the verbatim text printed to the operator; the parsed fields let a
// caller assert the loose backlog dropped with nothing pruned.
type CountObjects struct {
	Raw       string `json:"raw"`
	Count     int    `json:"count"`   // loose objects
	InPack    int    `json:"in_pack"` // objects in packs
	Packs     int    `json:"packs"`
	Available bool   `json:"available"`
}

// MaintStep is one executed (or planned, or skipped) git step and its outcome.
type MaintStep struct {
	Tier    string      `json:"tier"` // "always-safe" | "safe-with-grace"
	Args    []string    `json:"args"` // the git argv (after the "git" program word)
	Ran     bool        `json:"ran"`  // false when dry-run-planned or skipped
	Skipped MaintReason `json:"skipped,omitempty"`
	Code    int         `json:"code,omitempty"`
	Err     string      `json:"err,omitempty"`
}

// MaintResult is the full structured outcome of a git-maint run.
type MaintResult struct {
	Apply        bool         `json:"apply"`
	Posture      Posture      `json:"posture"`
	Locks        []string     `json:"locks,omitempty"`         // lock paths (common-dir-relative) live at preflight
	GraceRefused MaintReason  `json:"grace_refused,omitempty"` // "" when the grace tier ran
	Incident     bool         `json:"incident"`                // posture drift — surfaced as an incident
	Steps        []MaintStep  `json:"steps"`
	Before       CountObjects `json:"before"`
	After        CountObjects `json:"after"`
}

// LooseDelta reports how many loose objects the run folded away (before − after). A
// positive value with nothing pruned is the "consolidated, never deleted" witness.
func (r MaintResult) LooseDelta() int { return r.Before.Count - r.After.Count }

// RunMaint executes the safe object-DB consolidation. The ALWAYS-SAFE tier
// (multi-pack-index write, commit-graph write --reachable) runs unconditionally — it
// is add-only and atomic, safe even mid-commit. The SAFE-WITH-GRACE tier (git
// maintenance run --task=loose-objects / --task=incremental-repack, which may UNLINK a
// fully-covered redundant copy) runs only when BOTH the posture assert and the lock
// preflight pass, re-checking locks before every mutating step (TOCTOU). It never
// prunes unreachable objects, never full-repacks, never edits config. Idempotent: a
// rerun with nothing to consolidate is a no-op.
func RunMaint(ctx context.Context, run MaintRunner, opts MaintOptions) MaintResult {
	res := MaintResult{Apply: opts.Apply}
	res.Before = countObjects(ctx, run, opts.RepoRoot)
	res.Posture = readPosture(ctx, run, opts.RepoRoot)
	res.Locks = probeLocks(opts.GitCommonDir)

	// Always-safe tier: unconditional (safe under any lock/posture state).
	for _, args := range alwaysSafeSteps {
		res.Steps = append(res.Steps, runStep(ctx, run, opts, "always-safe", args))
	}

	// Safe-with-grace tier: gated on posture, then locks.
	switch {
	case !res.Posture.Safe:
		res.GraceRefused = MaintReasonPostureDrift
		res.Incident = true
		res.Steps = appendSkipped(res.Steps, MaintReasonPostureDrift)
	case len(res.Locks) > 0:
		res.GraceRefused = MaintReasonLocked
		res.Steps = appendSkipped(res.Steps, MaintReasonLocked)
	default:
		for _, args := range graceSteps {
			// TOCTOU: re-probe locks immediately before each mutating grace step, so a
			// peer that starts a commit mid-run defers the remaining steps.
			if live := probeLocks(opts.GitCommonDir); len(live) > 0 {
				res.GraceRefused = MaintReasonLocked
				res.Locks = live
				res.Steps = append(res.Steps, MaintStep{Tier: "safe-with-grace", Args: args, Skipped: MaintReasonLocked})
				continue
			}
			res.Steps = append(res.Steps, runStep(ctx, run, opts, "safe-with-grace", args))
		}
	}

	res.After = countObjects(ctx, run, opts.RepoRoot)
	return res
}

// appendSkipped records every grace step as skipped with the given reason (used when
// the whole tier is held back before any step runs).
func appendSkipped(steps []MaintStep, reason MaintReason) []MaintStep {
	for _, args := range graceSteps {
		steps = append(steps, MaintStep{Tier: "safe-with-grace", Args: args, Skipped: reason})
	}
	return steps
}

// runStep runs one git step (or, in a dry-run, records it as planned-only). A non-zero
// exit is captured, not fatal: an always-safe verb that cannot take its own lock
// mid-commit fails safe (it deletes nothing), and the operator sees the code.
func runStep(ctx context.Context, run MaintRunner, opts MaintOptions, tier string, args []string) MaintStep {
	st := MaintStep{Tier: tier, Args: args}
	if !opts.Apply {
		return st // dry-run: planned only, no mutation
	}
	_, code, err := run(ctx, opts.RepoRoot, args...)
	st.Ran = true
	st.Code = code
	if err != nil {
		st.Err = err.Error()
	}
	return st
}

// readPosture reads gc.auto and maintenance.auto from the effective git config and
// grades the safe posture. An UNSET key reads as "" and is treated as unsafe: the safe
// posture must be EXPLICITLY configured (git's own defaults — gc.auto 6700,
// maintenance.auto true — are both unsafe for a hot shared clone), so a repo that has
// not been posture-set refuses the grace tier rather than trusting a silent default.
func readPosture(ctx context.Context, run MaintRunner, dir string) Posture {
	p := Posture{
		GCAuto:          configGet(ctx, run, dir, "gc.auto"),
		MaintenanceAuto: configGet(ctx, run, dir, "maintenance.auto"),
	}
	var drift []string
	if strings.TrimSpace(p.GCAuto) != "0" {
		drift = append(drift, fmt.Sprintf("gc.auto=%s (want 0)", displayConfig(p.GCAuto)))
	}
	if !isGitFalse(p.MaintenanceAuto) {
		drift = append(drift, fmt.Sprintf("maintenance.auto=%s (want false)", displayConfig(p.MaintenanceAuto)))
	}
	p.Safe = len(drift) == 0
	p.Drift = strings.Join(drift, "; ")
	return p
}

// configGet returns the trimmed value of a git config key, or "" when the key is unset
// (git exits non-zero) or unreadable.
func configGet(ctx context.Context, run MaintRunner, dir, key string) string {
	out, code, err := run(ctx, dir, "config", "--get", key)
	if err != nil || code != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// displayConfig renders a config value for a drift message, spelling the unset case as
// <unset> so the incident is legible.
func displayConfig(v string) string {
	if strings.TrimSpace(v) == "" {
		return "<unset>"
	}
	return v
}

// probeLocks returns the sorted, de-duplicated set of live lock paths (common-dir
// relative) under gitDir: the fixed maintLockNames, plus any *.lock beneath refs/ and
// worktrees/ (ref-transaction locks and every linked worktree's own locks, incl. its
// fak-commit.lock). An empty result is the clean, safe-to-fold state.
func probeLocks(gitDir string) []string {
	if strings.TrimSpace(gitDir) == "" {
		return nil
	}
	seen := map[string]bool{}
	var found []string
	add := func(rel string) {
		rel = filepath.ToSlash(rel)
		if !seen[rel] {
			seen[rel] = true
			found = append(found, rel)
		}
	}
	for _, name := range maintLockNames {
		if fileExists(filepath.Join(gitDir, name)) {
			add(name)
		}
	}
	for _, sub := range []string{"refs", "worktrees"} {
		for _, rel := range walkLockFiles(gitDir, sub) {
			add(rel)
		}
	}
	sort.Strings(found)
	return found
}

// walkLockFiles returns every path ending in .lock beneath gitDir/sub, as a
// gitDir-relative path. A missing subtree yields nothing.
func walkLockFiles(gitDir, sub string) []string {
	root := filepath.Join(gitDir, sub)
	var out []string
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(d.Name(), ".lock") {
			if rel, rerr := filepath.Rel(gitDir, p); rerr == nil {
				out = append(out, rel)
			}
		}
		return nil
	})
	return out
}

// fileExists reports whether path names an existing file or directory.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// countObjects runs `git count-objects -vH` in dir and parses the loose/in-pack/pack
// counts for the before/after witness, retaining the raw text for display. An
// unreadable count (git error) yields Available=false with whatever text git emitted.
func countObjects(ctx context.Context, run MaintRunner, dir string) CountObjects {
	out, code, err := run(ctx, dir, "count-objects", "-vH")
	co := CountObjects{Raw: strings.TrimRight(out, "\r\n")}
	if err != nil || code != 0 {
		return co
	}
	co.Available = true
	for _, line := range strings.Split(out, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "count":
			co.Count = leadingInt(val)
		case "in-pack":
			co.InPack = leadingInt(val)
		case "packs":
			co.Packs = leadingInt(val)
		}
	}
	return co
}

// leadingInt parses the leading integer of a trimmed string (count-objects -vH values
// are bare integers; the -H suffixes only decorate the size lines we do not parse).
func leadingInt(s string) int {
	s = strings.TrimSpace(s)
	end := 0
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0
	}
	n, err := strconv.Atoi(s[:end])
	if err != nil {
		return 0
	}
	return n
}
