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
// object DELETION, which this path forbids except for the strictly supervised
// grace-prune tier below. Three tiers, gated progressively harder:
//
//   - ALWAYS-SAFE (RunMaint runs it unconditionally, even mid-commit): `git
//     multi-pack-index write` and `git commit-graph write --reachable`. Both are
//     add-only and atomic — they build/refresh an index file and delete nothing, so a
//     concurrent commit cannot be harmed. Verified 2026-07-04 on this box: both ran +
//     `verify`-clean with the loose/pack object counts unchanged.
//
//   - SAFE-WITH-GRACE (gated on BOTH the lock preflight AND the posture assert):
//     `git maintenance run --task=loose-objects`, followed by `git prune-packed`
//     (drops only loose objects that ALREADY have a packed duplicate), and
//     `--task=incremental-repack`
//     (multi-pack-index expire removes only fully-covered redundant packs; on Windows
//     an open pack simply fails to unlink → fail-safe). These UNLINK redundant copies,
//     so they run only when no git/fak transaction is in flight and the shared
//     no-auto-gc posture still holds — and the locks are RE-CHECKED before every
//     mutating step (TOCTOU), so a peer that starts committing mid-run defers the rest.
//
//   - GRACE-PRUNE (#5079, #4602 Phase 4 — OPT-IN, default OFF): `git prune
//     --expire=<≥2w>` only, the reclaim half the fold tiers structurally cannot
//     deliver (folding relocates loose objects into packs; it never removes the
//     ~86% that are UNREACHABLE). Gated strictly harder than the fold tier: the
//     posture assert AND a fresh transaction-lock re-probe AND a genuine QUIET
//     WINDOW — no live session/intent lease under refs/fak/locks/, the exact
//     namespace the fold tiers deliberately ignore. This is the one tier where a
//     live session lease IS load-bearing: a fold can run beside a live session,
//     a prune must not.
//
// WHAT IT NEVER DOES (the forbidden set, mirroring gitgate.go's refusals): never
// `--prune=now` / bare `git prune` (they ignore gc.pruneexpire and drop unreachable
// loose immediately — including an object a concurrent commit wrote milliseconds
// ago), never `git gc` / `--task=gc`, never a full `repack -a/-A/-adb`, never
// `git worktree prune`, never `git maintenance register/start` (those write the
// GLOBAL ~/.gitconfig), never edits .git/config — the operator-invoked fsmonitor
// repair (RepairFsmonitor, fsmonitor_repair.go / #5068) may unset core.fsmonitor,
// but it is never called from RunMaint or any auto-run path, so this invariant
// holds for every unattended tier. The ONE permitted prune form is
// the grace-prune tier's `git prune --expire=<≥2w>`: the ≥2-week expire floor
// (enforced in code — a sub-floor or `now` expire is REFUSED, never executed)
// means it can only drop objects that have been unreachable for weeks, which no
// in-flight commit could have just written, and the quiet-window gate means no
// session is even live to race. That is precisely why it is not the forbidden
// form: the forbidden prunes are dangerous because `now` drops just-written
// objects mid-transaction; a supervised ≥2w prune in a quiet window cannot.
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

	"github.com/anthony-chaudhary/fak/internal/walkfiles"
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
	// GracePrune opts in to the supervised grace-prune tier (#5079): a single
	// `git prune --expire=<≥2w>` reclaiming unreachable loose objects, run only
	// under the full preflight AND a quiet window (no live session lease). Default
	// false = the tier is OFF and recorded as skipped with MaintReasonPruneOff.
	GracePrune bool
	// PruneExpire optionally overrides the grace-prune expire window. Empty means
	// the defaultPruneExpire floor. Fail-closed: any value validPruneExpire cannot
	// prove is at/above the 2-week floor (`now`, `1.weeks.ago`, free-text dates…)
	// REFUSES the tier with MaintReasonPruneExpireUnsafe — the argv is never built.
	PruneExpire string
	// RequireBacklogHigh makes the safe-with-grace fold tier a HIGH-WATER trigger
	// (#5084): the tier runs only when the PRE-run count already collected by this
	// run shows LooseBacklogHigh, and is otherwise held back with
	// MaintReasonBacklogLow. It gates nothing else — the always-safe tier still runs
	// unconditionally, and the gate reads the count RunMaint collects anyway, so an
	// unattended host below the threshold pays no extra git invocation. Default false
	// keeps the operator-invoked `fak git-maint` verb folding on demand: an operator
	// who asks for a fold means it regardless of the backlog.
	RequireBacklogHigh bool
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
	// (gc.auto != 0, maintenance.auto is not an explicit false, or core.fsmonitor is
	// enabled without a watching builtin daemon) — an unsupervised auto-gc could
	// prune-race, or a "true but dead" fsmonitor makes every cold git op pay a
	// dead-IPC handshake and fall back to a full working-tree scan (#4603). The grace
	// tier REFUSES and the drift is surfaced as an incident for an operator to repair
	// the shared config.
	MaintReasonPostureDrift MaintReason = "POSTURE_DRIFT"
	// MaintReasonPruneOff: the grace-prune tier was not requested — it is opt-in and
	// default-OFF (MaintOptions.GracePrune=false) until soaked (#5079). Not an
	// incident; the fold tiers are unaffected.
	MaintReasonPruneOff MaintReason = "PRUNE_OFF"
	// MaintReasonSessionLive: a live session/intent lease is held under
	// refs/fak/locks/ — the box is not in a quiet window, so the grace-prune tier
	// refuses immediately (no added latency). This is the ONE tier where a lease is
	// load-bearing: the fold tiers deliberately ignore that namespace (#4602 GAP 2),
	// but a prune must not run while any session is live.
	MaintReasonSessionLive MaintReason = "SESSION_LIVE"
	// MaintReasonPruneExpireUnsafe: the requested prune expire is below the 2-week
	// floor (or unparseable — `now`, `all`, free-text). The tier REFUSES and the
	// sub-floor argv is never constructed, so `git prune --expire=now` cannot be
	// emitted through this path even by misconfiguration.
	MaintReasonPruneExpireUnsafe MaintReason = "PRUNE_EXPIRE_UNSAFE"
	// MaintReasonBacklogLow: the caller asked for a HIGH-WATER-triggered fold
	// (MaintOptions.RequireBacklogHigh, #5084) and the pre-run loose count is below
	// LooseBacklogThreshold, so there is no backlog worth folding. Not an incident and
	// not a deferral to retry — the clone is healthy; the always-safe tier still ran.
	MaintReasonBacklogLow MaintReason = "BACKLOG_LOW"
	// MaintReasonLooseBacklogHigh: the fold tier actually RAN against a high pre-run
	// backlog and the loose count did NOT come down. This is the one maintenance
	// outcome that is an incident rather than a refusal: folding is structurally
	// unable to clear this backlog (it relocates loose objects, it never removes
	// UNREACHABLE ones), so the clone keeps paying the cold-start object-store walk
	// until the supervised grace-prune tier (#5079) reclaims them. Recorded in
	// MaintResult.LooseBacklogIncident, never as a skip reason on a step.
	MaintReasonLooseBacklogHigh MaintReason = "LOOSE_BACKLOG_HIGH"
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
	// Git's loose-objects task prunes OLD packed duplicates before it packs the
	// remaining reachable loose objects. Without this final pass, every object it just
	// packed remains loose until tomorrow's tick: count-objects reports no reduction
	// and the one-shot maintenance promise takes two days. prune-packed deletes only a
	// byte-for-byte object already present in a pack and stays under the same posture +
	// per-step lock re-probe as every other redundant-copy fold.
	{"prune-packed"},
	{"maintenance", "run", "--task=incremental-repack"},
}

// Grace-prune tier constants (#5079): the tier name, and the expire floor. The floor
// is git approxidate `2.weeks.ago` — wide enough that no object a concurrent commit
// just wrote can be inside the window, and enforced in code by validPruneExpire (a
// sub-floor request refuses; it is never merely clamped or ignored).
const (
	gracePruneTier     = "grace-prune"
	defaultPruneExpire = "2.weeks.ago"
)

// validPruneExpire validates a requested grace-prune expire window against the
// 2-week floor, returning the canonical value to place after `--expire=`. Empty
// means the default floor. FAIL-CLOSED: only the closed forms `<n>.weeks.ago`
// (n ≥ 2), `<n>.months.ago` / `<n>.years.ago` (n ≥ 1) are provably at/above the
// floor; everything else — `now`, `all`, day/hour units, free-text dates — is
// refused rather than parsed generously, so the forbidden `--expire=now` argv is
// structurally unbuildable through this path.
func validPruneExpire(v string) (string, bool) {
	v = strings.ToLower(strings.TrimSpace(v))
	if v == "" {
		return defaultPruneExpire, true
	}
	for _, u := range []struct {
		suffix string
		floor  int
	}{
		{".weeks.ago", 2},
		{".months.ago", 1},
		{".years.ago", 1},
	} {
		num, ok := strings.CutSuffix(v, u.suffix)
		if !ok {
			continue
		}
		n, err := strconv.Atoi(num)
		if err != nil || n < u.floor {
			return "", false
		}
		return v, true
	}
	return "", false
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

// Posture is the read of the shared-config knobs that keep the hot clone in its
// intended safe state: the two that keep automatic, unsupervised maintenance (which
// CAN prune) from ever firing, plus core.fsmonitor, whose "true but dead daemon" drift
// (#4603) silently stalls every cold git op behind a dead-IPC handshake and a full
// working-tree scan.
type Posture struct {
	GCAuto          string `json:"gc_auto"`          // configured gc.auto ("" = unset → git default 6700, unsafe)
	MaintenanceAuto string `json:"maintenance_auto"` // configured maintenance.auto ("" = unset → git default true, unsafe)
	Fsmonitor       string `json:"fsmonitor"`        // configured core.fsmonitor ("" = unset → off, which is safe)
	// FsmonitorDaemon is the builtin-daemon health probe, populated only when
	// core.fsmonitor selects the builtin daemon (a git-true value): "watching" (healthy),
	// "not-watching" (config says true but no daemon is up — the #4603 stall), or
	// "unknown" (the status probe itself could not run). Empty when fsmonitor is off or a
	// hook-program path (neither has a builtin daemon to probe).
	FsmonitorDaemon string `json:"fsmonitor_daemon,omitempty"`
	// UntrackedCache is the configured core.untrackedCache ("" = unset → off, drift:
	// every cold `git status` full-scans the ~10k-file working tree without it, #5069).
	UntrackedCache string `json:"untracked_cache"`
	Safe           bool   `json:"safe"`            // gc.auto == 0 AND maintenance.auto explicit git-false AND fsmonitor off-or-watching AND untrackedCache on
	Drift          string `json:"drift,omitempty"` // human reason when !Safe
}

// fsmonitor builtin-daemon health classes, from `git fsmonitor--daemon status`.
const (
	fsmonitorWatching    = "watching"
	fsmonitorNotWatching = "not-watching"
	fsmonitorUnknown     = "unknown"
)

// CountObjects is the parsed `git count-objects -vH` snapshot for the before/after
// witness. Raw is the verbatim text printed to the operator; the parsed fields let a
// caller assert the loose backlog dropped and distinguish packed duplicates waiting for
// prune-packed from genuinely unpacked objects.
type CountObjects struct {
	Raw           string `json:"raw"`
	Count         int    `json:"count"`   // loose objects
	InPack        int    `json:"in_pack"` // objects in packs
	Packs         int    `json:"packs"`
	PrunePackable int    `json:"prune_packable"` // loose objects already duplicated in packs
	Available     bool   `json:"available"`
}

// MaintStep is one executed (or planned, or skipped) git step and its outcome.
type MaintStep struct {
	Tier    string      `json:"tier"` // "always-safe" | "safe-with-grace" | "grace-prune"
	Args    []string    `json:"args"` // the git argv (after the "git" program word)
	Ran     bool        `json:"ran"`  // false when dry-run-planned or skipped
	Skipped MaintReason `json:"skipped,omitempty"`
	Code    int         `json:"code,omitempty"`
	Err     string      `json:"err,omitempty"`
}

// MaintResult is the full structured outcome of a git-maint run.
type MaintResult struct {
	Apply        bool        `json:"apply"`
	Posture      Posture     `json:"posture"`
	Locks        []string    `json:"locks,omitempty"`         // lock paths (common-dir-relative) live at preflight
	GraceRefused MaintReason `json:"grace_refused,omitempty"` // "" when the grace tier ran
	// GracePruneRefused is "" only when the grace-prune tier actually ran (or was
	// dry-run-planned); otherwise the structured reason it was held back — PRUNE_OFF
	// (default), PRUNE_EXPIRE_UNSAFE, POSTURE_DRIFT, LOCKED, or SESSION_LIVE.
	GracePruneRefused MaintReason `json:"grace_prune_refused,omitempty"`
	// SessionLeases are the live session/intent lease files under refs/fak/locks/
	// seen by the grace-prune quiet-window probe (populated only when they refused
	// the tier). The fold tiers ignore this namespace; the prune tier must not.
	SessionLeases []string     `json:"session_leases,omitempty"`
	Incident      bool         `json:"incident"` // posture drift — surfaced as an incident
	Steps         []MaintStep  `json:"steps"`
	Before        CountObjects `json:"before"`
	After         CountObjects `json:"after"`
	// LooseBacklogHigh is set from the PRE-run count: true when the loose-object backlog
	// is at/above LooseBacklogThreshold. A read-only high-water witness — it mutates
	// nothing and gates no tier — so an operator/caller can SEE the invisible backlog
	// before any auto-trigger is wired. See #4602 Phase 0.
	LooseBacklogHigh bool `json:"loose_backlog_high"`
	// LooseBacklogIncident is MaintReasonLooseBacklogHigh when a fold RAN against a
	// high pre-run backlog and the loose count failed to come down — the non-reduction
	// signal that escalates to grace-prune (#5079). Empty when the backlog was not
	// high, when no fold step ran, or when the count did fall. Fail-closed: an
	// unavailable before/after count never raises the incident. See #5084.
	LooseBacklogIncident MaintReason `json:"loose_backlog_incident,omitempty"`
}

// LooseDelta reports how many loose objects the run folded away (before − after). A
// positive value with nothing pruned is the "consolidated, never deleted" witness.
func (r MaintResult) LooseDelta() int { return r.Before.Count - r.After.Count }

// LooseBacklogThreshold is the loose-object count at/above which the object DB is
// considered to carry a high backlog worth folding. It is a read-only witness bound
// only to the operator surface (#4602 Phase 0); by itself it gates no maintenance tier.
const LooseBacklogThreshold = 10_000

// LooseBacklogHigh reports whether a count-objects snapshot shows a loose-object
// backlog at/above LooseBacklogThreshold. Fail-closed: an unavailable count (git error)
// is never reported as high.
func LooseBacklogHigh(co CountObjects) bool {
	return co.Available && co.Count >= LooseBacklogThreshold
}

// RunMaint executes the safe object-DB consolidation. The ALWAYS-SAFE tier
// (multi-pack-index write, commit-graph write --reachable) runs unconditionally — it
// is add-only and atomic, safe even mid-commit. The SAFE-WITH-GRACE tier (git
// maintenance run --task=loose-objects / git prune-packed /
// --task=incremental-repack, which may UNLINK a fully-covered redundant copy) runs only
// when BOTH the posture assert and the lock
// preflight pass, re-checking locks before every mutating step (TOCTOU). The
// GRACE-PRUNE tier (opt-in, default-off) additionally requires the ≥2-week expire
// floor and a quiet window (no live session lease) before its single supervised
// `git prune --expire=<≥2w>`; with GracePrune unset RunMaint never prunes anything.
// It never full-repacks, never edits config. Idempotent: a rerun with nothing to
// consolidate is a no-op.
//
// An unattended host may additionally set MaintOptions.RequireBacklogHigh to make the
// fold tier a HIGH-WATER trigger (#5084): it then runs only when the pre-run count is
// at/above LooseBacklogThreshold, and is held back with MaintReasonBacklogLow
// otherwise. When such a gated fold RUNS and the loose count still does not come down,
// RunMaint raises the LOOSE_BACKLOG_HIGH incident (MaintResult.LooseBacklogIncident) —
// the measured proof that folding cannot clear this backlog and it needs the
// grace-prune tier (#5079).
func RunMaint(ctx context.Context, run MaintRunner, opts MaintOptions) MaintResult {
	res := MaintResult{Apply: opts.Apply}
	res.Before = countObjects(ctx, run, opts.RepoRoot)
	res.LooseBacklogHigh = LooseBacklogHigh(res.Before)
	res.Posture = readPosture(ctx, run, opts.RepoRoot)
	res.Locks = probeLocks(opts.GitCommonDir)

	// Always-safe tier: unconditional (safe under any lock/posture state).
	for _, args := range alwaysSafeSteps {
		res.Steps = append(res.Steps, runStep(ctx, run, opts, "always-safe", args))
	}

	// Safe-with-grace tier: gated on posture, then the optional high-water mark, then
	// locks. Posture stays FIRST so a drifted shared config is still surfaced as an
	// incident on a low-backlog box — the drift is a config-health signal an operator
	// must repair whether or not there is anything to fold today.
	switch {
	case !res.Posture.Safe:
		res.GraceRefused = MaintReasonPostureDrift
		res.Incident = true
		res.Steps = appendSkipped(res.Steps, MaintReasonPostureDrift)
	case opts.RequireBacklogHigh && !res.LooseBacklogHigh:
		res.GraceRefused = MaintReasonBacklogLow
		res.Steps = appendSkipped(res.Steps, MaintReasonBacklogLow)
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

	// Grace-prune tier (#5079, #4602 Phase 4): opt-in, default-off, gated strictly
	// harder than the fold tier — expire floor, posture, a fresh transaction-lock
	// re-probe, AND a quiet window (no live session lease). At most one prune per run.
	res.Steps = append(res.Steps, gracePruneStep(ctx, run, opts, &res))

	res.After = countObjects(ctx, run, opts.RepoRoot)
	if res.LooseBacklogIncident = looseBacklogIncident(res); res.LooseBacklogIncident != "" {
		res.Incident = true
	}
	return res
}

// looseBacklogIncident decides the #5084 non-reduction signal: a fold that actually RAN
// against a high pre-run backlog and left the loose count no lower than it found it.
// Every clause is a reason the observation would be unearned — a dry run mutated
// nothing, a held-back tier never got the chance, and an unavailable count proves
// nothing either way — so the incident only fires on a real, measured failure to reduce.
func looseBacklogIncident(res MaintResult) MaintReason {
	switch {
	case !res.Apply, !res.LooseBacklogHigh:
		return ""
	case !res.Before.Available || !res.After.Available:
		return ""
	case !ranTier(res.Steps, "safe-with-grace"):
		return ""
	case res.After.Count < res.Before.Count:
		return ""
	}
	return MaintReasonLooseBacklogHigh
}

// ranTier reports whether any step of the given tier actually executed.
func ranTier(steps []MaintStep, tier string) bool {
	for _, s := range steps {
		if s.Tier == tier && s.Ran {
			return true
		}
	}
	return false
}

// gracePruneStep decides and (when every gate passes) executes the single supervised
// grace-prune step, recording the structured refusal otherwise. Gate order: opt-in
// flag, expire floor, posture, transaction locks (a FRESH probe — the preflight
// snapshot may be stale by now), then the quiet window. A refused step never builds
// a sub-floor argv: the displayed args always carry a validated (or the default
// floor) expire, so `--expire=now` cannot appear in a step record, let alone hit git.
func gracePruneStep(ctx context.Context, run MaintRunner, opts MaintOptions, res *MaintResult) MaintStep {
	expire, expireOK := validPruneExpire(opts.PruneExpire)
	if !expireOK {
		expire = defaultPruneExpire // display-only: the refused value is never placed in an argv
	}
	args := []string{"prune", "--expire=" + expire}
	refuse := func(reason MaintReason) MaintStep {
		res.GracePruneRefused = reason
		return MaintStep{Tier: gracePruneTier, Args: args, Skipped: reason}
	}
	switch {
	case !opts.GracePrune:
		return refuse(MaintReasonPruneOff)
	case !expireOK:
		return refuse(MaintReasonPruneExpireUnsafe)
	case !res.Posture.Safe:
		return refuse(MaintReasonPostureDrift)
	}
	if live := probeLocks(opts.GitCommonDir); len(live) > 0 {
		res.Locks = live
		return refuse(MaintReasonLocked)
	}
	if leases := probeSessionLeases(opts.GitCommonDir); len(leases) > 0 {
		res.SessionLeases = leases
		return refuse(MaintReasonSessionLive)
	}
	return runStep(ctx, run, opts, gracePruneTier, args)
}

// probeSessionLeases returns every file (sorted, common-dir-relative) under fak's
// lease namespace refs/fak/locks/ — loose lease refs (session-*, intent-*) and their
// transient *.lock twins alike. For the fold tiers this namespace is deliberately
// invisible (#4602 GAP 2: a fold is object-DB-orthogonal to a live session); for the
// grace-prune tier it is THE load-bearing quiet-window signal — any file here means
// a session is (or may be) live, and the prune refuses with MaintReasonSessionLive.
// Conservative on purpose: a stale lease also blocks (reap it first), because a
// false "quiet" is the failure mode this tier exists to avoid.
func probeSessionLeases(gitDir string) []string {
	if strings.TrimSpace(gitDir) == "" {
		return nil
	}
	root := filepath.Join(gitDir, filepath.FromSlash(leaseLockPrefix))
	var out []string
	_ = walkfiles.Files(root, func(p string, d os.DirEntry) error {
		if rel, rerr := filepath.Rel(gitDir, p); rerr == nil {
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Strings(out)
	return out
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

// readPosture reads gc.auto, maintenance.auto, core.fsmonitor, and core.untrackedCache
// from the effective git config and grades the safe posture. An UNSET key reads as "" and is treated as unsafe: the safe
// posture must be EXPLICITLY configured (git's own defaults — gc.auto 6700,
// maintenance.auto true — are both unsafe for a hot shared clone), so a repo that has
// not been posture-set refuses the grace tier rather than trusting a silent default.
func readPosture(ctx context.Context, run MaintRunner, dir string) Posture {
	p := Posture{
		GCAuto:          configGet(ctx, run, dir, "gc.auto"),
		MaintenanceAuto: configGet(ctx, run, dir, "maintenance.auto"),
		Fsmonitor:       configGet(ctx, run, dir, "core.fsmonitor"),
		UntrackedCache:  configGet(ctx, run, dir, "core.untrackedCache"),
	}
	var drift []string
	if strings.TrimSpace(p.GCAuto) != "0" {
		drift = append(drift, fmt.Sprintf("gc.auto=%s (want 0)", displayConfig(p.GCAuto)))
	}
	if !isGitFalse(p.MaintenanceAuto) {
		drift = append(drift, fmt.Sprintf("maintenance.auto=%s (want false)", displayConfig(p.MaintenanceAuto)))
	}
	// core.fsmonitor: only a git-TRUE value selects the builtin daemon, which is the sole
	// form with a dead-IPC failure mode (#4603). OFF (false/unset) needs no daemon and is
	// safe; a hook-program PATH runs no builtin daemon, so there is nothing to probe and
	// no dead handshake to pay. When the builtin daemon IS selected, the safe state is that
	// it is AFFIRMATIVELY watching this tree — "true but dead" (not-watching) and an
	// unprobeable daemon (unknown) both fail the assert, because a cold git op would then
	// pay the dead-IPC handshake and fall back to a full working-tree scan.
	if isGitTrue(p.Fsmonitor) {
		p.FsmonitorDaemon = fsmonitorDaemonHealth(ctx, run, dir)
		if p.FsmonitorDaemon != fsmonitorWatching {
			drift = append(drift, fmt.Sprintf("core.fsmonitor=%s but builtin daemon is %s (want a watching daemon, or unset core.fsmonitor)",
				p.Fsmonitor, p.FsmonitorDaemon))
		}
	}
	// core.untrackedCache: the daemon-independent cold-status speedup (#5069, the
	// follow-up #4603 scoped out). Unset/off means every cold `git status` walks the
	// whole ~10k-file working tree; TRUE caches untracked-dir mtimes in the index so a
	// cold status re-scans only dirs that changed. Measured on this hot clone
	// (2026-07-17, git 2.51, no fsmonitor daemon): steady-state status ~700ms → ~420ms,
	// first cold run 2082ms → 425ms. Asserted HERE — the posture check, not a setup
	// verb — so the setting is MANAGED: drift surfaces as a POSTURE_DRIFT incident
	// instead of silently regressing status latency (operator repair:
	// `git config core.untrackedCache true`). feature.manyFiles was evaluated and
	// REJECTED for this tree: beyond the untracked cache it also flips index.version=4
	// and (git >= 2.40) index.skipHash=true — index FORMAT changes that non-git readers
	// of .git/index on a shared always-hot clone may not parse — while the measured
	// cold-status win comes from the untracked cache alone.
	if !isGitTrue(p.UntrackedCache) {
		drift = append(drift, fmt.Sprintf("core.untrackedCache=%s (want true — cold `git status` full-scans the tree without it, #5069)", displayConfig(p.UntrackedCache)))
	}
	p.Safe = len(drift) == 0
	p.Drift = strings.Join(drift, "; ")
	return p
}

// fsmonitorDaemonHealth probes `git fsmonitor--daemon status` and classifies the builtin
// daemon: fsmonitorWatching (healthy — the daemon is watching this tree), fsmonitorNotWatching
// (config says true but no daemon is up — the #4603 cold-op stall), or fsmonitorUnknown (the
// probe itself could not run). It classifies by the message TEXT rather than the exit code so
// it is robust across git versions (a not-watching status exits non-zero on some builds); the
// "not watching" case is checked first because its message contains "watching" as a substring.
func fsmonitorDaemonHealth(ctx context.Context, run MaintRunner, dir string) string {
	out, _, err := run(ctx, dir, "fsmonitor--daemon", "status")
	if err != nil {
		return fsmonitorUnknown // git could not be executed at all
	}
	low := strings.ToLower(out)
	switch {
	case strings.Contains(low, "not watching"):
		return fsmonitorNotWatching
	case strings.Contains(low, "watching"):
		return fsmonitorWatching
	default:
		return fsmonitorUnknown
	}
}

// isGitTrue reports whether a git config value is a boolean TRUE in the forms git's own
// config parser accepts. core.fsmonitor set to one of these selects the builtin fsmonitor
// daemon; any other non-false value is a hook-program path, and false/unset is off. Mirrors
// isGitFalse (gitgate.go).
func isGitTrue(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "true", "yes", "on", "1":
		return true
	}
	return false
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

// leaseLockPrefix is fak's own lease namespace under refs/. When something CAS-updates
// a refs/fak/locks/session-* (or intent-*) ref, git writes a transient <ref>.lock for
// the millisecond the update-ref holds — but that ref is a session LIVENESS heartbeat
// LEASE (TTL ~2400s), NOT an in-flight object/ref transaction the object-DB fold could
// race. Object folding touches objects/, never refs/, and on Windows an open pack fails
// safe to unlink (see the file header). On a box that always has ≥1 live session these
// lease locks are ALWAYS present, so counting them as transaction locks pinned the grace
// tier at MaintReasonLocked forever — the loose backlog was unbounded by construction
// (GAP 2, #4602). probeLocks EXCLUDES this namespace while still counting every genuine
// ref-transaction lock (refs/heads/**.lock, refs/tags/**.lock, packed-refs.lock, …).
const leaseLockPrefix = "refs/fak/locks/"

// isLeaseLock reports whether a common-dir-relative *.lock path lives in fak's lease
// namespace (refs/fak/locks/…) — the transient heartbeat locks probeLocks excludes from
// the transaction-lock set. The path is slash-normalized so the prefix test holds on
// Windows, where filepath.Rel returns backslash separators.
func isLeaseLock(rel string) bool {
	return strings.HasPrefix(filepath.ToSlash(rel), leaseLockPrefix)
}

// probeLocks returns the sorted, de-duplicated set of live TRANSACTION lock paths
// (common-dir relative) under gitDir: the fixed maintLockNames, plus any *.lock beneath
// refs/ and worktrees/ (ref-transaction locks and every linked worktree's own locks,
// incl. its fak-commit.lock) — but EXCLUDING fak's own lease-heartbeat namespace
// (refs/fak/locks/…, see leaseLockPrefix), which is object-DB-orthogonal and always
// present on a live box. An empty result is the clean, safe-to-fold state.
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
			if isLeaseLock(rel) {
				continue // fak session/intent lease heartbeat — not a transaction (GAP 2, #4602)
			}
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
	_ = walkfiles.Files(root, func(p string, d os.DirEntry) error {
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
		case "prune-packable":
			co.PrunePackable = leadingInt(val)
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
