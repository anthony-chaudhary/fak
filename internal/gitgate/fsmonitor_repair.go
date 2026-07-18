// fsmonitor_repair.go — the OPERATOR-INVOKED repair half of the fsmonitor-drift
// story (#5068, follow-up to #4603's detector). readPosture (gitmaint.go) DETECTS
// the "core.fsmonitor=true but the builtin daemon is dead" drift and raises a
// POSTURE_DRIFT incident; this file gives the operator the one-command repair the
// detector deliberately left to them.
//
// Two repair modes, per the #5068 decision for this multi-worktree, always-hot
// Windows/macOS clone:
//
//   - OFF (the DEFAULT): unset core.fsmonitor on the hot clone. The builtin daemon
//     has proven unable to stay up on this clone shape (git #75781, #26154), and a
//     "true but dead" key makes every cold git op pay a dead-IPC handshake before
//     falling back to a full working-tree scan — strictly worse than off. With the
//     key unset there is no daemon to keep alive and readPosture reads SAFE.
//   - START: start the builtin daemon (`git fsmonitor--daemon start`) for an
//     operator who wants to keep fsmonitor and re-arm it after a crash.
//
// INVARIANT SEAM. RunMaint's "never edits .git/config" invariant is untouched:
// RepairFsmonitor is NEVER called from RunMaint or any auto-run maintenance path.
// It runs only when an operator explicitly invokes it (`fak git-maint
// --repair-fsmonitor`), which is exactly the boundary #5068 scopes: the repair
// stays operator-invoked, off the always-safe/grace tiers.
//
// FAIL-CLOSED VERDICT. Cleared is decided by RE-READING the config and re-probing
// the daemon after the repair — never from the repair command's own exit code — so
// a start that "succeeded" but left no watching daemon still reads NOT cleared.
package gitgate

import (
	"context"
	"fmt"
)

// FsmonitorRepairMode selects how RepairFsmonitor clears the drift.
type FsmonitorRepairMode string

const (
	// FsmonitorRepairOff unsets core.fsmonitor on the hot clone — the decided
	// default for this multi-worktree clone, where the builtin daemon has proven
	// unable to stay up (#5068).
	FsmonitorRepairOff FsmonitorRepairMode = "off"
	// FsmonitorRepairStart starts the builtin daemon instead, for an operator who
	// wants to keep fsmonitor enabled.
	FsmonitorRepairStart FsmonitorRepairMode = "start"
)

// Fsmonitor repair actions — what the repair decided to do, closed-vocabulary so a
// caller/test can match structurally.
const (
	// FsmonitorActionNone: nothing to repair — the key is off/unset, a hook-program
	// path (no builtin daemon), or the builtin daemon is already watching.
	FsmonitorActionNone = "none"
	// FsmonitorActionUnsetKey: the drift was cleared by unsetting core.fsmonitor.
	FsmonitorActionUnsetKey = "unset-key"
	// FsmonitorActionStartDaemon: the drift was cleared by starting the builtin daemon.
	FsmonitorActionStartDaemon = "start-daemon"
)

// FsmonitorRepairOptions configures one operator-invoked fsmonitor repair.
type FsmonitorRepairOptions struct {
	// RepoRoot is the working dir the repair git verbs run from (the hot clone's
	// main worktree); `git config --unset` from here edits THIS clone's .git/config,
	// never the global one.
	RepoRoot string
	// Mode is the repair mode; empty means FsmonitorRepairOff (the #5068 default).
	Mode FsmonitorRepairMode
	// Apply performs the repair. When false the run is a DRY-RUN: it probes the
	// before-state and reports the planned action, mutating nothing.
	Apply bool
}

// FsmonitorRepairResult is the structured before/after witness of one repair.
type FsmonitorRepairResult struct {
	Mode  FsmonitorRepairMode `json:"mode"`
	Apply bool                `json:"apply"`
	// BeforeValue/BeforeDaemon are the pre-repair core.fsmonitor value and (when the
	// value selects the builtin daemon) its health class — the "before" half of the
	// #5068 witness.
	BeforeValue  string `json:"before_value"`
	BeforeDaemon string `json:"before_daemon,omitempty"`
	// Action is the closed-vocabulary repair decision (FsmonitorAction*).
	Action string `json:"action"`
	// Args is the git argv the repair ran (or planned, in a dry-run); empty when
	// there was nothing to repair.
	Args []string `json:"args,omitempty"`
	Ran  bool     `json:"ran"`
	Code int      `json:"code,omitempty"`
	Err  string   `json:"err,omitempty"`
	// AfterValue/AfterDaemon are RE-READ from git after the repair (in a dry-run
	// they mirror the before-state).
	AfterValue  string `json:"after_value"`
	AfterDaemon string `json:"after_daemon,omitempty"`
	// Cleared reports whether the after-state reads fsmonitor-SAFE to readPosture
	// (key off/unset, hook path, or a watching builtin daemon). Decided from the
	// re-probe, never from the repair command's exit code — fail-closed.
	Cleared bool `json:"cleared"`
}

// fsmonitorSafe reports whether a (config value, daemon health) pair passes the
// readPosture fsmonitor assert: only a git-TRUE value with a non-watching builtin
// daemon is drift; off/unset and hook-program paths need no daemon.
func fsmonitorSafe(value, daemon string) bool {
	return !isGitTrue(value) || daemon == fsmonitorWatching
}

// RepairFsmonitor is the operator-invoked repair for the #4603 fsmonitor drift: on a
// "core.fsmonitor=true but the builtin daemon is dead/unprobeable" clone it either
// unsets the key (FsmonitorRepairOff, the default) or starts the daemon
// (FsmonitorRepairStart), then re-reads the state and reports whether the drift
// cleared. It handles all three daemon health classes: watching (no-op),
// not-watching, and unknown (both repaired). Never called from RunMaint — the
// auto-run maintenance path still never edits .git/config.
func RepairFsmonitor(ctx context.Context, run MaintRunner, opts FsmonitorRepairOptions) FsmonitorRepairResult {
	mode := opts.Mode
	if mode == "" {
		mode = FsmonitorRepairOff
	}
	res := FsmonitorRepairResult{Mode: mode, Apply: opts.Apply, Action: FsmonitorActionNone}
	if mode != FsmonitorRepairOff && mode != FsmonitorRepairStart {
		// Fail-closed: an unknown mode repairs nothing and reads NOT cleared.
		res.Err = fmt.Sprintf("unknown fsmonitor repair mode %q (want off|start)", mode)
		return res
	}

	res.BeforeValue = configGet(ctx, run, opts.RepoRoot, "core.fsmonitor")
	if isGitTrue(res.BeforeValue) {
		res.BeforeDaemon = fsmonitorDaemonHealth(ctx, run, opts.RepoRoot)
	}
	res.AfterValue, res.AfterDaemon = res.BeforeValue, res.BeforeDaemon

	if fsmonitorSafe(res.BeforeValue, res.BeforeDaemon) {
		res.Cleared = true // off/unset, hook path, or already watching — nothing to repair
		return res
	}

	switch mode {
	case FsmonitorRepairStart:
		res.Action = FsmonitorActionStartDaemon
		res.Args = []string{"fsmonitor--daemon", "start"}
	default: // FsmonitorRepairOff
		res.Action = FsmonitorActionUnsetKey
		res.Args = []string{"config", "--unset", "core.fsmonitor"}
	}
	if !opts.Apply {
		return res // dry-run: planned only; .git/config and the daemon are untouched
	}

	_, code, err := run(ctx, opts.RepoRoot, res.Args...)
	res.Ran = true
	res.Code = code
	if err != nil {
		res.Err = err.Error()
	}

	// The verdict comes from the re-read state, never the repair's exit code.
	res.AfterValue = configGet(ctx, run, opts.RepoRoot, "core.fsmonitor")
	res.AfterDaemon = ""
	if isGitTrue(res.AfterValue) {
		res.AfterDaemon = fsmonitorDaemonHealth(ctx, run, opts.RepoRoot)
	}
	res.Cleared = fsmonitorSafe(res.AfterValue, res.AfterDaemon)
	return res
}
