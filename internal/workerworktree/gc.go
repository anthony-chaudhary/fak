package workerworktree

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// THE OWNER-STAMPED GC (#3573)
//
// The later lease-only cold sweep (coldreap.go) closed most of the leaked-worktree
// problem, then gained a third "do not destroy unlanded work" gate. It still cannot
// identify the process that prepared one worktree or the exact/coarse lease identity
// that owned it. This file adds that missing identity without weakening the newer
// collection rules:
//
//   - every successful Prepare writes a PID + lease + creation-time owner stamp;
//   - GC is a pure dry-run plan unless Apply is explicitly true;
//   - a worktree is collectable only when it is under an allowlisted worker root,
//     old, owner-dead, lease-released, command-line inactive, clean, and carries no
//     detached commit absent from trunk;
//   - missing/unreadable stamps or liveness probes fail toward KEEPING;
//   - apply revalidates every gate, removes the directory first, then prunes the git
//     admin record only after absence is verified (#6510).
//
// Owner stamps live in a sidecar directory beside the managed worktrees rather than
// inside them. An in-tree owner file would itself appear in `git status --porcelain`
// and make every otherwise-clean leaked worktree look like irreplaceable unlanded work.

const (
	ownerStampSchema = "fak-worker-worktree-owner/1"
	ownerStateDir    = ".fak-worker-owners"
	ownerStampExt    = ".json"
)

// OwnerStamp is the durable ownership record written for one successful Prepare.
// CreatedAt is the GC age authority; directory mtimes can change during builds and are
// therefore not an ownership-age witness.
type OwnerStamp struct {
	Schema    string    `json:"schema"`
	PID       int       `json:"pid"`
	LeaseID   string    `json:"lease_id"`
	CreatedAt time.Time `json:"created_at"`
}

// ProcessLiveFn reports whether a stamped owner PID is still alive.
type ProcessLiveFn func(pid int) bool

// LeaseIDLiveFn reports whether the stamped lease (or a protecting equivalent such
// as a live lease on the same lane) is still live.
type LeaseIDLiveFn func(leaseID string) bool

// WorktreePathActiveFn reports whether any external process command line still
// references a worktree. A census error fails toward keeping.
type WorktreePathActiveFn func(path string) (active bool, err error)

// GCWorktree is one owner-stamped GC candidate. GarbageCollect reports candidates
// only: a path outside the allowlist, live-owner, live-lease, command-line-active,
// young, dirty, unpushed, or unreadable worktree is never listed. The fields retain
// the evidence that made the candidate safe to select.
type GCWorktree struct {
	Path          string     `json:"path"`
	Owner         OwnerStamp `json:"owner"`
	PoolState     string     `json:"pool_state,omitempty"`
	PoolUpdatedAt time.Time  `json:"pool_updated_at,omitempty"`
	AgeSec        int64      `json:"age_sec"`
	OwnerLive     bool       `json:"owner_live"`
	LeaseLive     bool       `json:"lease_live"`
	ProcessActive bool       `json:"process_active"`
	Unlanded      int        `json:"unlanded"`
	Unpushed      int        `json:"unpushed"`
	HeldByWork    bool       `json:"held_by_work,omitempty"`
	Eligible      bool       `json:"eligible"`
	Removed       bool       `json:"removed,omitempty"`
	Reason        string     `json:"reason"`
}

// GCOptions supplies the clock and the two independent liveness witnesses. Nil
// PID/lease liveness functions mean "unknown" and therefore keep every worktree.
// PathActive defaults to the procguard command-line census. AllowedRoots defaults
// to DefaultRoot(); explicit roots are principally for a reviewed custom worker root
// and hermetic tests.
type GCOptions struct {
	Now          time.Time
	MaxAge       time.Duration
	Apply        bool
	ProcessAlive ProcessLiveFn
	LeaseLive    LeaseIDLiveFn
	PathActive   WorktreePathActiveFn
	AllowedRoots []string
}

// GCReport is the dry-run/apply result returned by GarbageCollect.
type GCReport struct {
	Mode      string       `json:"mode"`
	MaxAgeSec int64        `json:"max_age_sec"`
	Worktrees []GCWorktree `json:"worktrees"`
	Failures  []GCFailure  `json:"failures"`
	WouldReap int          `json:"would_reap"`
	Reaped    int          `json:"reaped"`
}

type GCFailure struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// OwnerStampPath returns the sidecar path bound one-to-one to wtPath.
func OwnerStampPath(wtPath string) string {
	clean := filepath.Clean(strings.TrimSpace(wtPath))
	return filepath.Join(filepath.Dir(clean), ownerStateDir, filepath.Base(clean)+ownerStampExt)
}

// OwnerProcessLive reports whether the stamped worker process is still alive.
// The second result is false when the path is not a stamped worker worktree or
// liveness cannot be established, so generic janitors fail toward keeping it.
func OwnerProcessLive(wtPath string, processAlive ProcessLiveFn) (bool, bool) {
	if !IsWorkerWorktree(wtPath) || processAlive == nil {
		return false, false
	}
	stamp, err := readOwnerStamp(wtPath)
	if err != nil || stamp.PID <= 0 {
		return false, false
	}
	return processAlive(stamp.PID), true
}

// HandoffOwner replaces the transient preparing process with the stable spawned
// worker PID. Reapers read this sidecar, so the update must complete before the
// launcher exposes a successful spawn receipt.
func HandoffOwner(wtPath string, pid int) error {
	if pid <= 0 {
		return fmt.Errorf("worker owner pid must be positive")
	}
	stamp, err := readOwnerStamp(wtPath)
	if err != nil {
		return fmt.Errorf("read owner stamp: %w", err)
	}
	stamp.PID = pid
	stamp.CreatedAt = time.Now().UTC()
	if lease, lerr := ReadWorkerLease(wtPath); lerr == nil {
		lease.PID = pid
		lease.HeartbeatTS = time.Now().UTC()
		_ = WriteWorkerLease(wtPath, lease)
	}
	return writeOwnerStamp(wtPath, stamp)
}

func defaultOwnerStamp(lane string) OwnerStamp {
	leaseID := strings.TrimSpace(os.Getenv("FAK_LEASE_ID"))
	if leaseID == "" {
		// Prepare's long-standing API has lane but not the dispatcher's resolved lease
		// id. The coarse lane identity is conservative: any live issue lease on the
		// lane protects it in the CLI oracle, and a false keep is safer than a false
		// reap. CLI callers with an exact identity pass --lease-id.
		cleanLane := strings.TrimSpace(lane)
		if cleanLane != "" {
			leaseID = "resolve-" + cleanLane
		}
	}
	return OwnerStamp{PID: os.Getpid(), LeaseID: leaseID, CreatedAt: time.Now().UTC()}
}

func normalizeOwnerStamp(stamp OwnerStamp) OwnerStamp {
	stamp.Schema = ownerStampSchema
	stamp.LeaseID = strings.TrimSpace(stamp.LeaseID)
	if stamp.PID <= 0 {
		stamp.PID = os.Getpid()
	}
	if stamp.CreatedAt.IsZero() {
		stamp.CreatedAt = time.Now().UTC()
	} else {
		stamp.CreatedAt = stamp.CreatedAt.UTC()
	}
	return stamp
}

func writeOwnerStamp(wtPath string, stamp OwnerStamp) error {
	stamp = normalizeOwnerStamp(stamp)
	return atomicWriteJSON(OwnerStampPath(wtPath), stamp, 0o600)
}

func readOwnerStamp(wtPath string) (OwnerStamp, error) {
	b, err := os.ReadFile(OwnerStampPath(wtPath))
	if err != nil {
		return OwnerStamp{}, err
	}
	var stamp OwnerStamp
	if err := json.Unmarshal(b, &stamp); err != nil {
		return OwnerStamp{}, err
	}
	if stamp.Schema != ownerStampSchema || stamp.PID <= 0 || stamp.CreatedAt.IsZero() {
		return OwnerStamp{}, fmt.Errorf("invalid owner stamp")
	}
	stamp.LeaseID = strings.TrimSpace(stamp.LeaseID)
	stamp.CreatedAt = stamp.CreatedAt.UTC()
	return stamp, nil
}

func removeOwnerStamp(wtPath string) {
	path := OwnerStampPath(wtPath)
	_ = os.Remove(path)
	_ = os.Remove(filepath.Dir(path)) // succeeds only when this was the final stamp
}

// GCList enumerates worker worktrees and returns only the deletion-free,
// owner-stamped candidates satisfying every gate in GCOptions. Kept worktrees are
// deliberately omitted so dry-run is a literal list of what apply would attempt.
func GCList(root string, git GitRunner, opts GCOptions) []GCWorktree {
	opts = normalizeGCOptions(opts)
	_, paths := Count(root, git)
	out := make([]GCWorktree, 0, len(paths))
	for _, wtPath := range paths {
		if row, eligible, _ := gcEntry(root, wtPath, git, opts); eligible {
			out = append(out, row)
		}
	}
	return out
}

// GarbageCollect returns the dry-run plan by default. Apply re-runs every gate for
// each planned row, then removes the directory before pruning the git admin record.
// That ordering preserves #6510: a directory that cannot be removed remains registered.
func GarbageCollect(root string, git GitRunner, opts GCOptions) GCReport {
	opts = normalizeGCOptions(opts)
	report := GCReport{
		Mode:      "dry-run",
		MaxAgeSec: int64(opts.MaxAge / time.Second),
		Worktrees: GCList(root, git, opts),
		Failures:  []GCFailure{},
	}
	if opts.Apply {
		report.Mode = "apply"
	}
	for i := range report.Worktrees {
		row := &report.Worktrees[i]
		if !row.Eligible {
			continue
		}
		report.WouldReap++
		if opts.Apply {
			failure := ""
			lockErr := withPoolLaneLock(filepath.Dir(row.Path), LaneOf(row.Path), func() error {
				current, eligible, reason := gcEntry(root, row.Path, git, opts)
				if !eligible {
					failure = reason
					return nil
				}
				if current.Owner.PID != row.Owner.PID ||
					current.Owner.LeaseID != row.Owner.LeaseID ||
					!current.Owner.CreatedAt.Equal(row.Owner.CreatedAt) {
					failure = "owner_stamp_changed"
					return nil
				}
				if current.PoolState != row.PoolState ||
					!current.PoolUpdatedAt.Equal(row.PoolUpdatedAt) {
					failure = "pool_state_changed"
					return nil
				}
				if reason := gcRemoveDirectoryFirst(root, row.Path, git); reason != "" {
					failure = reason
					return nil
				}
				row.Removed = true
				report.Reaped++
				return nil
			})
			if lockErr != nil {
				failure = "pool_lane_lock_failed: " + lockErr.Error()
			}
			if failure != "" {
				report.Failures = append(report.Failures, GCFailure{Path: row.Path, Reason: failure})
			}
		}
	}
	return report
}

func normalizeGCOptions(opts GCOptions) GCOptions {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.MaxAge <= 0 {
		opts.MaxAge = DefaultColdAgeFloor
	}
	if opts.PathActive == nil {
		opts.PathActive = processCommandLineReferencesWorktree
	}
	if len(opts.AllowedRoots) == 0 {
		opts.AllowedRoots = []string{DefaultRoot()}
	}
	return opts
}

func gcEntry(root, wtPath string, git GitRunner, opts GCOptions) (GCWorktree, bool, string) {
	clean := filepath.Clean(strings.TrimSpace(wtPath))
	row := GCWorktree{
		Path:      clean,
		OwnerLive: true,
		LeaseLive: true,
		Unpushed:  -1,
		Reason:    "kept",
	}
	if !gcPathAllowed(clean, opts.AllowedRoots) {
		return row, false, "path_not_under_allowed_worker_root"
	}
	stamp, meta, ownershipReason := gcOwnerState(clean)
	if ownershipReason != "" {
		return row, false, ownershipReason
	}
	row.Owner = stamp
	ageBase := stamp.CreatedAt
	if meta != nil {
		row.PoolState = meta.State
		row.PoolUpdatedAt = meta.UpdatedAt
		ageBase = meta.UpdatedAt
	}
	age := opts.Now.Sub(ageBase)
	if age < 0 {
		age = 0
	}
	row.AgeSec = int64(age / time.Second)

	if opts.ProcessAlive == nil {
		return row, false, "owner_liveness_unknown"
	}
	row.OwnerLive = opts.ProcessAlive(stamp.PID)
	if row.OwnerLive {
		return row, false, "owner_process_live"
	}

	if opts.LeaseLive == nil || stamp.LeaseID == "" {
		return row, false, "lease_liveness_unknown"
	}
	row.LeaseLive = opts.LeaseLive(stamp.LeaseID)
	if row.LeaseLive {
		return row, false, "owner_lease_live"
	}

	if age < opts.MaxAge {
		return row, false, "worktree_too_young"
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

	row.Unlanded = UnlandedCount(clean, git)
	switch {
	case row.Unlanded < 0:
		row.HeldByWork = true
		return row, false, "working_tree_unreadable"
	case row.Unlanded > 0:
		row.HeldByWork = true
		return row, false, "working_tree_dirty"
	}

	row.Unpushed = unpushedCommitCount(root, clean, git)
	switch {
	case row.Unpushed < 0:
		row.HeldByWork = true
		return row, false, "commit_ancestry_unreadable"
	case row.Unpushed > 0:
		row.HeldByWork = true
		return row, false, "unpushed_commit"
	}

	row.Eligible = true
	row.Reason = fmt.Sprintf("gc: owner pid %d dead, lease %q released, age %s past max-age %s, no active command line, working tree clean, and HEAD contained by trunk",
		stamp.PID, stamp.LeaseID, age.Round(time.Second), opts.MaxAge.Round(time.Second))
	return row, true, ""
}

// gcOwnerState reads the durable pool record when present. Pool metadata embeds
// the owner that performed the idle/leased transition, so a process that crashes
// between reserving a member and refreshing the compatibility owner-stamp file
// still leaves a reclaimable leased owner instead of an ambiguous permanent leak.
// Non-pool worktrees retain the standalone owner-stamp authority.
func gcOwnerState(wtPath string) (OwnerStamp, *poolMemberMetadata, string) {
	meta, metaErr := readPoolMember(wtPath)
	if metaErr == nil {
		return meta.Owner, &meta, ""
	}
	if _, statErr := os.Stat(poolMemberPath(wtPath)); statErr == nil {
		return OwnerStamp{}, nil, "pool_metadata_unreadable"
	}
	stamp, err := readOwnerStamp(wtPath)
	if err != nil {
		return OwnerStamp{}, nil, "owner_stamp_unreadable"
	}
	return stamp, nil, ""
}

// unpushedCommitCount returns 0 when the worktree HEAD is already contained by the
// root checkout's HEAD, 1 when it is not, and -1 when ancestry cannot be proved.
// A clean detached worktree may still hold its only durable work as a local commit;
// status alone is therefore not a deletion witness.
func unpushedCommitCount(root, wtPath string, git GitRunner) int {
	rc, out := run(git, wtPath, []string{"rev-parse", "HEAD"})
	head := strings.TrimSpace(out)
	if rc != 0 || head == "" {
		return -1
	}
	rc, _ = run(git, root, []string{"merge-base", "--is-ancestor", head, "HEAD"})
	switch rc {
	case 0:
		return 0
	case 1:
		return 1
	default:
		return -1
	}
}

// gcPathAllowed binds deletion to a direct child of a reviewed worker root and
// the worker marker. A matching basename elsewhere is never enough.
func gcPathAllowed(path string, roots []string) bool {
	if !IsWorkerWorktree(path) {
		return false
	}
	pathAbs, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		return false
	}
	parent := filepath.Dir(pathAbs)
	for _, root := range roots {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		rootAbs, err := filepath.Abs(filepath.Clean(root))
		if err == nil && strings.EqualFold(parent, rootAbs) {
			return true
		}
	}
	return false
}

func processCommandLineReferencesWorktree(path string) (bool, error) {
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

// gcRemoveDirectoryFirst performs the destructive half only after gcEntry has
// been evaluated twice. The filesystem goes first; prune is allowed only after the
// directory is confirmed absent, so failure cannot strand an externally active tree
// by unregistering it underneath a live process (#6510).
func gcRemoveDirectoryFirst(root, wtPath string, git GitRunner) string {
	if err := os.RemoveAll(wtPath); err != nil {
		return "directory_remove_failed: " + err.Error()
	}
	if _, err := os.Stat(wtPath); !os.IsNotExist(err) {
		return "directory_remains"
	}
	if rc, out := run(git, root, []string{"worktree", "prune"}); rc != 0 {
		return "prune_failed: " + strings.TrimSpace(out)
	}
	removePoolMemberState(wtPath)
	removeOwnerStamp(wtPath)
	return ""
}
