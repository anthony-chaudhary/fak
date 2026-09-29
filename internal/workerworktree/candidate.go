package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/processstart"
)

// THE TOPOLOGY-CANDIDATE LEAK
//
// verifyTopologyCandidate materializes a verify-only checkout (a detached git
// worktree) and removes it in a deferred cleanup. A defer does not run when the
// process is killed, and a kill is the ordinary way a slow land ends: a supervisor
// that bounds the land with a deadline terminates the whole process tree
// (`taskkill /T /F` on Windows), and so does an agent tool timeout. Every such kill
// left a full checkout, usually with a half-written build cache, plus its git
// worktree registration; over a hundred accumulated beside one Windows fleet host's
// repositories, and every registration slowed each `git worktree list`.
//
// A killed process cannot clean up after itself, so the next one does. The
// candidate's directory name carries its creator's identity (pid and kernel
// process-start time), and every candidate creation first sweeps its parent for
// candidates whose creator is provably gone. A candidate is verification scratch by
// construction: its content is a pure function of a commit that already exists and
// a diff the worker worktree still holds, so collecting one never loses work.
// Liveness still fails toward keeping: an owner whose start time cannot be read is
// treated as live, and a legacy name without an identity is collected only after
// LegacyCandidateMaxAge, far past any bounded land.

const (
	topologyCandidatePrefix = ".fak-cand-validate-"

	// execWitnessCandidatePrefix is the sibling verify-checkout family the exec
	// witness rung creates (internal/witness scratchWorktree). A witness run killed
	// by a deadline leaks it the same way a killed land leaks a topology candidate,
	// so the one owner-aware sweep collects both.
	execWitnessCandidatePrefix = "fak-exec-witness-"

	// LegacyCandidateMaxAge is how long a candidate without an owner identity in
	// its name (created by a binary older than the owner-named candidates) must sit
	// untouched before a sweep collects it. Its directory mtime moves when the
	// verify starts writing build output, so the age measures time since the
	// verify began; a supervised land is killed long before this.
	LegacyCandidateMaxAge = 2 * time.Hour

	// candidateSweepInterval rate-limits the sweep a land runs before creating a
	// candidate, per parent directory and process.
	candidateSweepInterval = 10 * time.Minute

	// candidateSweepLandLimit bounds how many candidates one land collects, so a
	// large backlog cannot stall a land; the explicit gc verb has no limit.
	candidateSweepLandLimit = 2
)

// topologyCandidatePrefixes is every verify-checkout family the one owner-aware
// sweep collects. Both producers name their checkout through
// OwnerNamedScratchPattern, so the same pid/start identity parsing applies to each.
var topologyCandidatePrefixes = []string{topologyCandidatePrefix, execWitnessCandidatePrefix}

// candidateOwner is the creator identity encoded in a candidate name. Start is the
// process start time in Unix milliseconds, or 0 on a platform that exposes none.
type candidateOwner struct {
	PID   int
	Start int64
}

// isTopologyCandidateName reports whether name is a verify-checkout directory of
// any family the sweep collects.
func isTopologyCandidateName(name string) bool {
	for _, prefix := range topologyCandidatePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// OwnerNamedScratchPattern returns an os.MkdirTemp pattern that encodes this
// process's identity (pid + base36 start millis) under prefix, so a later sweep
// can prove the creator is gone. prefix must end with '-'.
func OwnerNamedScratchPattern(prefix string) string {
	pid := os.Getpid()
	start := int64(0)
	if t, ok := processstart.Start(pid); ok {
		start = t.UnixMilli()
	}
	return prefix + strconv.Itoa(pid) + "-" + strconv.FormatInt(start, 36) + "-*"
}

// topologyCandidatePattern is the os.MkdirTemp pattern for a land-verify candidate
// owned by this process: prefix, pid, base-36 start millis, then MkdirTemp's random
// suffix.
func topologyCandidatePattern() string {
	return OwnerNamedScratchPattern(topologyCandidatePrefix)
}

// parseTopologyCandidateOwner reads the owner identity from a candidate directory
// name. It returns false for a legacy name (a bare MkdirTemp suffix) and for
// anything malformed, which the sweep then judges by age alone.
func parseTopologyCandidateOwner(name string) (candidateOwner, bool) {
	rest := ""
	found := false
	for _, prefix := range topologyCandidatePrefixes {
		if r, ok := strings.CutPrefix(name, prefix); ok {
			rest, found = r, true
			break
		}
	}
	if !found {
		return candidateOwner{}, false
	}
	parts := strings.Split(rest, "-")
	if len(parts) != 3 || parts[2] == "" {
		return candidateOwner{}, false
	}
	pid, err := strconv.Atoi(parts[0])
	if err != nil || pid <= 0 {
		return candidateOwner{}, false
	}
	start, err := strconv.ParseInt(parts[1], 36, 64)
	if err != nil || start < 0 {
		return candidateOwner{}, false
	}
	return candidateOwner{PID: pid, Start: start}, true
}

// CandidateSweepItem is one topology candidate found in the sweep's parent
// directory, with the decision and the evidence behind it.
type CandidateSweepItem struct {
	Path     string `json:"path"`
	OwnerPID int    `json:"owner_pid,omitempty"`
	AgeSec   int64  `json:"age_sec"`
	Eligible bool   `json:"eligible"`
	Removed  bool   `json:"removed,omitempty"`
	Reason   string `json:"reason"`
}

// CandidateSweepOptions supplies the clock, the liveness probes, and the apply
// opt-in. Nil probes default to the real process table.
type CandidateSweepOptions struct {
	Now          time.Time
	LegacyMaxAge time.Duration
	Apply        bool
	// Limit caps removals under Apply; zero means no cap.
	Limit        int
	ProcessAlive ProcessLiveFn
	ProcessStart func(pid int) (time.Time, bool)
}

// CandidateSweepReport is the dry-run/apply result of SweepTopologyCandidates.
// Candidates lists every candidate found, kept ones included, so an operator can
// see why a candidate stays.
type CandidateSweepReport struct {
	Mode            string               `json:"mode"`
	Parent          string               `json:"parent"`
	LegacyMaxAgeSec int64                `json:"legacy_max_age_sec"`
	Candidates      []CandidateSweepItem `json:"candidates"`
	Failures        []GCFailure          `json:"failures"`
	WouldReap       int                  `json:"would_reap"`
	Reaped          int                  `json:"reaped"`
}

// TopologyCandidateParent is the directory verifyTopologyCandidate creates root's
// candidates in: beside root for a workspace whose modules escape it, otherwise
// the system temp directory.
func TopologyCandidateParent(root string) (string, error) {
	if !workspaceNeedsSiblingTopology(root) {
		return os.TempDir(), nil
	}
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	return filepath.Dir(rootAbs), nil
}

// SweepTopologyCandidates classifies every topology candidate in parent (the
// system temp directory when empty) and, under Apply, removes the ones whose owner
// is gone. Each directory is removed before any git registration is pruned, and
// the prune runs against the repository the candidate was registered in, so a
// directory that cannot be removed stays registered (#6510).
func SweepTopologyCandidates(root, parent string, git GitRunner, opts CandidateSweepOptions) CandidateSweepReport {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.LegacyMaxAge <= 0 {
		opts.LegacyMaxAge = LegacyCandidateMaxAge
	}
	if opts.ProcessAlive == nil {
		opts.ProcessAlive = processalive.Check
	}
	if opts.ProcessStart == nil {
		opts.ProcessStart = processstart.Start
	}
	if parent == "" {
		parent = os.TempDir()
	}
	report := CandidateSweepReport{
		Mode:            "dry-run",
		Parent:          parent,
		LegacyMaxAgeSec: int64(opts.LegacyMaxAge / time.Second),
		Candidates:      []CandidateSweepItem{},
		Failures:        []GCFailure{},
	}
	if opts.Apply {
		report.Mode = "apply"
	}
	entries, err := os.ReadDir(parent)
	if err != nil {
		report.Failures = append(report.Failures, GCFailure{Path: parent, Reason: "read_parent_failed: " + err.Error()})
		return report
	}
	pruneDirs := map[string]bool{}
	for _, entry := range entries {
		if !entry.IsDir() || !isTopologyCandidateName(entry.Name()) {
			continue
		}
		item := classifyTopologyCandidate(filepath.Join(parent, entry.Name()), opts)
		if item.Eligible {
			report.WouldReap++
			if opts.Apply && (opts.Limit <= 0 || report.Reaped < opts.Limit) {
				commonDir := candidateCommonGitDir(item.Path)
				if reason := removeTopologyCandidateDir(item.Path); reason != "" {
					report.Failures = append(report.Failures, GCFailure{Path: item.Path, Reason: reason})
				} else {
					item.Removed = true
					report.Reaped++
					pruneDirs[commonDir] = true
				}
			}
		}
		report.Candidates = append(report.Candidates, item)
	}
	if commonDir, unlocked := sweepLockedCandidateRegistrations(root, git, opts, &report); unlocked {
		pruneDirs[commonDir] = true
	}
	for commonDir := range pruneDirs {
		args := []string{"worktree", "prune"}
		if commonDir != "" {
			args = append([]string{"--git-dir=" + commonDir}, args...)
		}
		if rc, out := run(git, root, args); rc != 0 {
			report.Failures = append(report.Failures, GCFailure{Path: commonDir, Reason: "prune_failed: " + tail(out, 200)})
		}
	}
	return report
}

// candidateOwnerState reports whether a candidate's creator still runs:
// owner_process_live, owner_exited, or owner_pid_reused (the pid now names a
// process that started at a different time). An unreadable start time counts as
// live.
func candidateOwnerState(owner candidateOwner, opts CandidateSweepOptions) string {
	if !opts.ProcessAlive(owner.PID) {
		return "owner_exited"
	}
	if owner.Start != 0 {
		if started, known := opts.ProcessStart(owner.PID); known && started.UnixMilli() != owner.Start {
			return "owner_pid_reused"
		}
	}
	return "owner_process_live"
}

// classifyTopologyCandidate decides one candidate. An owner-named candidate is
// eligible exactly when its owner process is gone; a legacy name is eligible only
// once it has been untouched for LegacyMaxAge.
func classifyTopologyCandidate(path string, opts CandidateSweepOptions) CandidateSweepItem {
	item := CandidateSweepItem{Path: path, Reason: "kept"}
	info, statErr := os.Stat(path)
	age := time.Duration(0)
	if statErr == nil {
		if age = opts.Now.Sub(info.ModTime()); age < 0 {
			age = 0
		}
		item.AgeSec = int64(age / time.Second)
	}
	if owner, ok := parseTopologyCandidateOwner(filepath.Base(path)); ok {
		item.OwnerPID = owner.PID
		item.Reason = candidateOwnerState(owner, opts)
		item.Eligible = item.Reason != "owner_process_live"
		return item
	}
	switch {
	case statErr != nil:
		item.Reason = "stat_failed"
	case age < opts.LegacyMaxAge:
		item.Reason = "legacy_too_young"
	default:
		item.Eligible = true
		item.Reason = fmt.Sprintf("legacy_stale: no owner identity, untouched for %s", age.Round(time.Second))
	}
	return item
}

// sweepLockedCandidateRegistrations handles the one registration `git worktree
// prune` never clears: git holds an "initializing" lock while `worktree add` runs,
// so a land killed during the checkout leaves a locked entry that outlives its
// directory forever. For a root candidate registration whose directory is gone and
// whose lock is held by a dead owner (or, for a legacy name, has not moved for
// LegacyMaxAge), Apply removes the lock file, which is all `git worktree unlock`
// does, so the caller's prune can collect the entry. It reports root's common git
// directory and whether any lock was removed.
func sweepLockedCandidateRegistrations(root string, git GitRunner, opts CandidateSweepOptions, report *CandidateSweepReport) (string, bool) {
	rc, out := run(git, root, []string{"rev-parse", "--git-common-dir"})
	commonDir := strings.TrimSpace(out)
	if rc != 0 || commonDir == "" {
		return "", false
	}
	if !filepath.IsAbs(commonDir) {
		commonDir = filepath.Join(root, commonDir)
	}
	admins, err := os.ReadDir(filepath.Join(commonDir, "worktrees"))
	if err != nil {
		return commonDir, false
	}
	unlocked := false
	for _, admin := range admins {
		adminDir := filepath.Join(commonDir, "worktrees", admin.Name())
		lock, err := os.Stat(filepath.Join(adminDir, "locked"))
		if err != nil {
			continue
		}
		gitdir, err := os.ReadFile(filepath.Join(adminDir, "gitdir"))
		if err != nil {
			continue
		}
		wt := filepath.Dir(filepath.Clean(strings.TrimSpace(string(gitdir))))
		if !isTopologyCandidateName(filepath.Base(wt)) {
			continue
		}
		if _, err := os.Stat(wt); !os.IsNotExist(err) {
			continue
		}
		item := CandidateSweepItem{Path: wt, Reason: "kept"}
		age := opts.Now.Sub(lock.ModTime())
		if age < 0 {
			age = 0
		}
		item.AgeSec = int64(age / time.Second)
		if owner, ok := parseTopologyCandidateOwner(filepath.Base(wt)); ok {
			item.OwnerPID = owner.PID
			state := candidateOwnerState(owner, opts)
			item.Eligible = state != "owner_process_live"
			item.Reason = "locked_registration_" + state
		} else if age < opts.LegacyMaxAge {
			item.Reason = "locked_registration_too_young"
		} else {
			item.Eligible = true
			item.Reason = fmt.Sprintf("locked_registration_legacy_stale: directory gone, lock untouched for %s", age.Round(time.Second))
		}
		if item.Eligible {
			report.WouldReap++
			if opts.Apply {
				if err := os.Remove(filepath.Join(adminDir, "locked")); err != nil && !os.IsNotExist(err) {
					report.Failures = append(report.Failures, GCFailure{Path: wt, Reason: "unlock_failed: " + err.Error()})
				} else {
					item.Removed = true
					report.Reaped++
					unlocked = true
				}
			}
		}
		report.Candidates = append(report.Candidates, item)
	}
	return commonDir, unlocked
}

// candidateCommonGitDir returns the git common directory a candidate worktree is
// registered in, read from its .git file (gitdir: <common>/worktrees/<id>), or ""
// when the file is absent or does not have that shape.
func candidateCommonGitDir(path string) string {
	b, err := os.ReadFile(filepath.Join(path, ".git"))
	if err != nil {
		return ""
	}
	admin, found := strings.CutPrefix(strings.TrimSpace(string(b)), "gitdir:")
	if !found {
		return ""
	}
	admin = filepath.Clean(strings.TrimSpace(admin))
	if !strings.EqualFold(filepath.Base(filepath.Dir(admin)), "worktrees") {
		return ""
	}
	common := filepath.Dir(filepath.Dir(admin))
	if info, err := os.Stat(common); err != nil || !info.IsDir() {
		return ""
	}
	return common
}

// removeTopologyCandidateDir removes a candidate directory with the Windows
// read-only/lock retry and reports why it remains when it does.
func removeTopologyCandidateDir(path string) string {
	if err := safeRemoveAll(path); err != nil {
		return "directory_remove_failed: " + err.Error()
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return "directory_remains"
	}
	return ""
}

// cleanupTopologyCandidate is verifyTopologyCandidate's deferred cleanup. git's
// own removal is the fast path; the directory is then removed directly, retrying
// transient Windows locks, and the registration is pruned only once the directory
// is gone. Whatever still remains carries its owner's identity and is collected by
// a later sweep once this process exits.
func cleanupTopologyCandidate(root, candDir string, git GitRunner) {
	run(git, root, []string{"worktree", "remove", "--force", candDir})
	if removeTopologyCandidateDir(candDir) == "" {
		run(git, root, []string{"worktree", "prune"})
	}
}

var (
	candidateSweepMu   sync.Mutex
	candidateSweepLast = map[string]time.Time{}
)

// sweepTopologyCandidatesBeforeCreate runs the bounded, rate-limited sweep a land
// performs before creating its own candidate. A package variable so tests of the
// land flow can stub it out.
var sweepTopologyCandidatesBeforeCreate = func(root, parent string, git GitRunner) {
	key := parent
	if key == "" {
		key = os.TempDir()
	}
	now := time.Now()
	candidateSweepMu.Lock()
	last, seen := candidateSweepLast[key]
	if seen && now.Sub(last) < candidateSweepInterval {
		candidateSweepMu.Unlock()
		return
	}
	candidateSweepLast[key] = now
	candidateSweepMu.Unlock()
	SweepTopologyCandidates(root, parent, git, CandidateSweepOptions{Now: now, Apply: true, Limit: candidateSweepLandLimit})
}
