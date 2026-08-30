package selfinstall

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// buildDirPrefix is the fixed name prefix of every self-update build worktree. The build
// directory is "<buildDirPrefix><pid>" (BuildDirName), so the prefix both MARKS a directory
// as self-update's own throwaway origin checkout and ENCODES the pid of the process that
// created it — which is exactly what makes dead-owner reaping provably safe.
const buildDirPrefix = "fak-selfupdate-build-"

const (
	buildOwnerStampSchema = "fak-selfupdate-build-owner/1"
	buildOwnerStateDir    = ".fak-selfupdate-owners"

	// DefaultBuildGCMinAge is the grace floor for a leaked self-update checkout.
	// A false keep costs disk; a false reap can destroy an active build, so unknown
	// or younger worktrees stay registered.
	DefaultBuildGCMinAge = 30 * time.Minute
)

// BuildDirName returns the per-process build-worktree directory name for pid. cmdSelfUpdate
// joins this onto os.TempDir() to place the worktree; ReapStaleBuilds parses the pid back
// out of it. Single-sourcing the name here keeps the creator and the reaper from ever
// drifting apart (a drift would silently turn the reaper into a no-op).
func BuildDirName(pid int) string { return buildDirPrefix + strconv.Itoa(pid) }

// IsSelfUpdateBuildWorktree reports whether path has the exact basename reserved for a
// self-update source checkout. Generic worktree janitors use this marker to defer cleanup
// to GarbageCollectStaleBuilds, whose owner, age, process-census, cleanliness, and ancestry
// gates are stronger than a generic last-touch window.
func IsSelfUpdateBuildWorktree(path string) bool {
	_, ok := pidFromBuildDir(filepath.Base(filepath.Clean(strings.TrimSpace(path))))
	return ok
}

// pidFromBuildDir extracts the pid encoded in a build-worktree directory's basename, or
// (0,false) when base is not a "<buildDirPrefix><pid>" name with a positive numeric pid.
func pidFromBuildDir(base string) (int, bool) {
	rest, ok := strings.CutPrefix(base, buildDirPrefix)
	if !ok {
		return 0, false
	}
	pid, err := strconv.Atoi(rest)
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

// BuildOwnerStamp is the durable owner/lease record for one self-update checkout.
// It lives beside the worktree, not inside it, so a pristine origin checkout remains
// clean under `git status`.
type BuildOwnerStamp struct {
	Schema    string    `json:"schema"`
	PID       int       `json:"pid"`
	LeaseID   string    `json:"lease_id"`
	CreatedAt time.Time `json:"created_at"`
}

// BuildOwnerStampPath returns the sidecar bound one-to-one to buildDir.
func BuildOwnerStampPath(buildDir string) string {
	clean := filepath.Clean(strings.TrimSpace(buildDir))
	return filepath.Join(filepath.Dir(clean), buildOwnerStateDir, filepath.Base(clean)+".json")
}

func defaultBuildOwnerStamp() BuildOwnerStamp {
	leaseID := strings.TrimSpace(os.Getenv("FAK_LEASE_ID"))
	if leaseID == "" {
		leaseID = "self-update-single-flight"
	}
	return BuildOwnerStamp{
		Schema:    buildOwnerStampSchema,
		PID:       os.Getpid(),
		LeaseID:   leaseID,
		CreatedAt: time.Now().UTC(),
	}
}

func writeBuildOwnerStamp(buildDir string, stamp BuildOwnerStamp) error {
	stamp.Schema = buildOwnerStampSchema
	if stamp.PID <= 0 {
		return fmt.Errorf("invalid pid %d", stamp.PID)
	}
	stamp.LeaseID = strings.TrimSpace(stamp.LeaseID)
	if stamp.LeaseID == "" {
		return fmt.Errorf("empty lease id")
	}
	if stamp.CreatedAt.IsZero() {
		return fmt.Errorf("empty creation time")
	}
	stamp.CreatedAt = stamp.CreatedAt.UTC()

	path := BuildOwnerStampPath(buildDir)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(stamp, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Rename(tmpName, path)
}

func readBuildOwnerStamp(buildDir string) (BuildOwnerStamp, bool, error) {
	b, err := os.ReadFile(BuildOwnerStampPath(buildDir))
	if os.IsNotExist(err) {
		return BuildOwnerStamp{}, false, nil
	}
	if err != nil {
		return BuildOwnerStamp{}, false, err
	}
	var stamp BuildOwnerStamp
	if err := json.Unmarshal(b, &stamp); err != nil {
		return BuildOwnerStamp{}, true, err
	}
	if stamp.Schema != buildOwnerStampSchema || stamp.PID <= 0 ||
		strings.TrimSpace(stamp.LeaseID) == "" || stamp.CreatedAt.IsZero() {
		return BuildOwnerStamp{}, true, fmt.Errorf("invalid owner stamp")
	}
	stamp.LeaseID = strings.TrimSpace(stamp.LeaseID)
	stamp.CreatedAt = stamp.CreatedAt.UTC()
	return stamp, true, nil
}

func removeBuildOwnerStamp(buildDir string) {
	path := BuildOwnerStampPath(buildDir)
	_ = os.Remove(path)
	_ = os.Remove(filepath.Dir(path))
}

// BuildPathActiveFn reports whether any process command line still references path.
// An error means the process census is unavailable and must fail toward keeping.
type BuildPathActiveFn func(path string) (active bool, err error)

// BuildGCOptions configures the deletion-free plan and the explicit apply step.
// Apply defaults false; a nil PID liveness probe keeps every worktree. PathActive
// defaults to the procguard command-line census.
type BuildGCOptions struct {
	Now          time.Time
	MinAge       time.Duration
	Apply        bool
	SelfPID      int
	TempRoot     string
	BaseRef      string
	ProcessAlive func(pid int) bool
	PathActive   BuildPathActiveFn
}

// BuildGCWorktree is one self-update checkout proven safe enough for the dry-run plan.
type BuildGCWorktree struct {
	Path          string          `json:"path"`
	Owner         BuildOwnerStamp `json:"owner"`
	OwnerStamped  bool            `json:"owner_stamped"`
	AgeSec        int64           `json:"age_sec"`
	ProcessActive bool            `json:"process_active"`
	Eligible      bool            `json:"eligible"`
	Removed       bool            `json:"removed,omitempty"`
	Reason        string          `json:"reason"`
}

type BuildGCFailure struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// BuildGCReport is dry-run by default. WouldReap is the plan size; Reaped counts
// only paths whose directory disappeared and whose git admin record was pruned.
type BuildGCReport struct {
	Mode      string            `json:"mode"`
	MinAgeSec int64             `json:"min_age_sec"`
	Worktrees []BuildGCWorktree `json:"worktrees"`
	Failures  []BuildGCFailure  `json:"failures"`
	WouldReap int               `json:"would_reap"`
	Reaped    int               `json:"reaped"`
}

// GarbageCollectStaleBuilds plans, and only with Apply=true collects, self-update
// worktrees left by dead prior runs. Selection is deliberately conjunctive:
//
//   - exact direct-child path under the allowlisted temp root;
//   - exact fak-selfupdate-build-<pid> shape, with a matching owner stamp when present;
//   - old enough, owner PID dead, and no process command line referencing the path;
//   - when the directory exists, a clean tree whose HEAD is contained by BaseRef;
//   - when only the Git admin record remains, a valid owner stamp is mandatory.
//
// Apply re-runs every gate immediately before deletion. It removes the directory
// first and prunes the git admin record only after absence is verified, preserving
// #6510's rule that GC must never unregister an externally active/undeleted tree.
func GarbageCollectStaleBuilds(ctx context.Context, run Runner, repoRoot string, opts BuildGCOptions) BuildGCReport {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.MinAge <= 0 {
		opts.MinAge = DefaultBuildGCMinAge
	}
	if strings.TrimSpace(opts.TempRoot) == "" {
		opts.TempRoot = os.TempDir()
	}
	if strings.TrimSpace(opts.BaseRef) == "" {
		opts.BaseRef = "origin/main"
	}
	if opts.SelfPID <= 0 {
		opts.SelfPID = os.Getpid()
	}
	if opts.PathActive == nil {
		opts.PathActive = buildProcessCommandLineActive
	}
	report := BuildGCReport{
		Mode:      "dry-run",
		MinAgeSec: int64(opts.MinAge / time.Second),
		Worktrees: []BuildGCWorktree{},
		Failures:  []BuildGCFailure{},
	}
	if opts.Apply {
		report.Mode = "apply"
	}

	out, ok := run(ctx, repoRoot, "git", "worktree", "list", "--porcelain")
	if !ok {
		return report
	}
	for _, path := range worktreePaths(out) {
		row, eligible, _ := staleBuildEntry(ctx, run, repoRoot, path, opts)
		if !eligible {
			continue
		}
		report.Worktrees = append(report.Worktrees, row)
		report.WouldReap++
	}

	if !opts.Apply {
		return report
	}
	for i := range report.Worktrees {
		planned := &report.Worktrees[i]
		current, eligible, reason := staleBuildEntry(ctx, run, repoRoot, planned.Path, opts)
		if !eligible {
			report.Failures = append(report.Failures, BuildGCFailure{Path: planned.Path, Reason: reason})
			continue
		}
		if current.Owner.PID != planned.Owner.PID ||
			current.Owner.LeaseID != planned.Owner.LeaseID ||
			!current.Owner.CreatedAt.Equal(planned.Owner.CreatedAt) {
			report.Failures = append(report.Failures, BuildGCFailure{Path: planned.Path, Reason: "owner_stamp_changed"})
			continue
		}
		if err := os.RemoveAll(planned.Path); err != nil {
			report.Failures = append(report.Failures, BuildGCFailure{Path: planned.Path, Reason: "directory_remove_failed: " + err.Error()})
			continue
		}
		if _, err := os.Lstat(planned.Path); !os.IsNotExist(err) {
			report.Failures = append(report.Failures, BuildGCFailure{Path: planned.Path, Reason: "directory_remains"})
			continue
		}
		if out, ok := run(ctx, repoRoot, "git", "worktree", "prune"); !ok {
			report.Failures = append(report.Failures, BuildGCFailure{Path: planned.Path, Reason: "prune_failed: " + strings.TrimSpace(out)})
			continue
		}
		removeBuildOwnerStamp(planned.Path)
		planned.Removed = true
		report.Reaped++
	}
	return report
}

// ReapStaleBuilds is the explicit-apply compatibility seam used by older callers.
// The selection is no longer basename/PID-only: it delegates to the fully gated GC.
func ReapStaleBuilds(ctx context.Context, run Runner, repoRoot string, selfPID int, alive func(int) bool) []string {
	report := GarbageCollectStaleBuilds(ctx, run, repoRoot, BuildGCOptions{
		Apply:        true,
		SelfPID:      selfPID,
		ProcessAlive: alive,
	})
	reaped := make([]string, 0, report.Reaped)
	for _, row := range report.Worktrees {
		if row.Removed {
			reaped = append(reaped, row.Path)
		}
	}
	return reaped
}

func staleBuildEntry(ctx context.Context, run Runner, repoRoot, path string, opts BuildGCOptions) (BuildGCWorktree, bool, string) {
	clean := filepath.Clean(strings.TrimSpace(path))
	row := BuildGCWorktree{Path: clean, Reason: "kept"}
	if !directChildOf(clean, opts.TempRoot) {
		return row, false, "path_not_under_temp_root"
	}
	pid, ok := pidFromBuildDir(filepath.Base(clean))
	if !ok {
		return row, false, "path_shape_not_selfupdate"
	}
	if pid == opts.SelfPID {
		return row, false, "owner_is_current_process"
	}
	info, statErr := os.Lstat(clean)
	missing := os.IsNotExist(statErr)
	if statErr != nil && !missing {
		return row, false, "worktree_unreadable"
	}
	if !missing && !info.IsDir() {
		return row, false, "worktree_unreadable"
	}

	stamp, stamped, stampErr := readBuildOwnerStamp(clean)
	if stampErr != nil {
		return row, false, "owner_stamp_unreadable"
	}
	if stamped {
		if stamp.PID != pid {
			return row, false, "owner_stamp_pid_mismatch"
		}
	} else if missing {
		// A missing legacy directory has neither a sidecar nor filesystem
		// metadata from which to derive durable owner/age evidence.
		return row, false, "missing_worktree_owner_unknown"
	} else {
		// Legacy build worktrees predate owner sidecars. The PID encoded by the
		// creator is still ownership evidence, but every other gate remains required.
		stamp = BuildOwnerStamp{
			Schema:    buildOwnerStampSchema,
			PID:       pid,
			LeaseID:   "legacy-name-owner",
			CreatedAt: info.ModTime().UTC(),
		}
	}
	row.Owner = stamp
	row.OwnerStamped = stamped
	age := opts.Now.Sub(stamp.CreatedAt)
	if age < 0 {
		age = 0
	}
	row.AgeSec = int64(age / time.Second)
	if age < opts.MinAge {
		return row, false, "worktree_too_young"
	}
	if opts.ProcessAlive == nil {
		return row, false, "owner_liveness_unknown"
	}
	if opts.ProcessAlive(pid) {
		return row, false, "owner_process_live"
	}
	if opts.PathActive == nil {
		return row, false, "process_command_line_liveness_unknown"
	}
	active, activeErr := opts.PathActive(clean)
	if activeErr != nil {
		return row, false, "process_command_line_probe_failed"
	}
	row.ProcessActive = active
	if active {
		return row, false, "process_command_line_active"
	}
	if missing {
		row.Eligible = true
		row.Reason = fmt.Sprintf("gc: registered path absent, stamped owner pid %d dead, age %s past floor %s, and no active command line",
			pid, age.Round(time.Second), opts.MinAge.Round(time.Second))
		return row, true, ""
	}
	if cleanOut, ok := run(ctx, clean, "git", "status", "--porcelain=v1", "--untracked-files=all"); !ok {
		return row, false, "working_tree_unreadable"
	} else if strings.TrimSpace(cleanOut) != "" {
		return row, false, "working_tree_dirty"
	}
	headOut, ok := run(ctx, clean, "git", "rev-parse", "HEAD")
	head := strings.TrimSpace(headOut)
	if !ok || head == "" {
		return row, false, "head_unreadable"
	}
	if _, ok := run(ctx, repoRoot, "git", "merge-base", "--is-ancestor", head, opts.BaseRef); !ok {
		return row, false, "unpushed_commit"
	}
	row.Eligible = true
	row.Reason = fmt.Sprintf("gc: owner pid %d dead, age %s past floor %s, no active command line, clean, and HEAD contained by %s",
		pid, age.Round(time.Second), opts.MinAge.Round(time.Second), opts.BaseRef)
	return row, true, ""
}

func directChildOf(path, root string) bool {
	pathAbs, err1 := filepath.Abs(filepath.Clean(path))
	rootAbs, err2 := filepath.Abs(filepath.Clean(root))
	if err1 != nil || err2 != nil {
		return false
	}
	return strings.EqualFold(filepath.Dir(pathAbs), rootAbs)
}

func buildProcessCommandLineActive(path string) (bool, error) {
	procs, collectErr := procguard.CollectRelations()
	if collectErr != "" {
		return true, fmt.Errorf("process census: %s", collectErr)
	}
	clean := strings.ToLower(filepath.Clean(path))
	slash := strings.ReplaceAll(clean, `\`, `/`)
	self := os.Getpid()
	for _, proc := range procs {
		if proc.PID == self {
			continue
		}
		cmd := strings.ToLower(proc.Cmdline)
		if strings.Contains(cmd, clean) || strings.Contains(strings.ReplaceAll(cmd, `\`, `/`), slash) {
			return true, nil
		}
	}
	return false, nil
}

// worktreePaths returns the worktree paths from `git worktree list --porcelain` output (the
// value after each "worktree " header line).
func worktreePaths(porcelain string) []string {
	var paths []string
	for _, line := range strings.Split(porcelain, "\n") {
		line = strings.TrimRight(line, "\r")
		if p, ok := strings.CutPrefix(line, "worktree "); ok {
			paths = append(paths, strings.TrimSpace(p))
		}
	}
	return paths
}
