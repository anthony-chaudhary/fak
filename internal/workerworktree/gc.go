package workerworktree

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
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
//   - a worktree is collectable only when it is old, its owner PID is confirmed dead,
//     its stamped lease is no longer live, and its working tree is clean;
//   - missing/unreadable stamps or liveness probes fail toward KEEPING;
//   - apply uses ForceReap, bypassing the warm pool because GC is collecting a leaked
//     member, and always follows worktree remove with worktree prune.
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

// GCWorktree is one owner-stamped GC candidate. GarbageCollect reports candidates
// only: a live-owner, live-lease, young, dirty, or unreadable worktree is never listed.
// The fields retain the evidence that made the candidate safe to select.
type GCWorktree struct {
	Path       string     `json:"path"`
	Owner      OwnerStamp `json:"owner"`
	AgeSec     int64      `json:"age_sec"`
	OwnerLive  bool       `json:"owner_live"`
	LeaseLive  bool       `json:"lease_live"`
	Unlanded   int        `json:"unlanded"`
	HeldByWork bool       `json:"held_by_work,omitempty"`
	Eligible   bool       `json:"eligible"`
	Removed    bool       `json:"removed,omitempty"`
	Reason     string     `json:"reason"`
}

// GCOptions supplies the clock and the two independent liveness witnesses. Nil
// liveness functions mean "unknown" and therefore keep every worktree.
type GCOptions struct {
	Now          time.Time
	MaxAge       time.Duration
	Apply        bool
	ProcessAlive ProcessLiveFn
	LeaseLive    LeaseIDLiveFn
}

// GCReport is the dry-run/apply result returned by GarbageCollect.
type GCReport struct {
	Mode      string       `json:"mode"`
	MaxAgeSec int64        `json:"max_age_sec"`
	Worktrees []GCWorktree `json:"worktrees"`
	WouldReap int          `json:"would_reap"`
	Reaped    int          `json:"reaped"`
}

// OwnerStampPath returns the sidecar path bound one-to-one to wtPath.
func OwnerStampPath(wtPath string) string {
	clean := filepath.Clean(strings.TrimSpace(wtPath))
	return filepath.Join(filepath.Dir(clean), ownerStateDir, filepath.Base(clean)+ownerStampExt)
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
	path := OwnerStampPath(wtPath)
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
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	// os.Rename does not replace an existing destination on Windows. Removing the
	// previous stamp first makes same-key Prepare retries and warm-pool leases refresh
	// correctly; a crash in this tiny gap fails GC toward keeping (missing stamp).
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Rename(tmpName, path)
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

func ownerAge(stamp OwnerStamp, now time.Time) time.Duration {
	if now.IsZero() {
		now = time.Now()
	}
	age := now.Sub(stamp.CreatedAt)
	if age < 0 {
		return 0
	}
	return age
}

// GCList enumerates worker worktrees and returns only the deletion-free
// owner-stamped GC candidates. It requires BOTH dead-owner and released-lease
// evidence. This conjunctive gate preserves coldreap.go's newer rule that a dead PID
// alone is not enough: dispatcher processes can exit while their worker's lane lease
// remains live. Kept worktrees are deliberately omitted so `gc --dry-run` is a literal
// list of what `gc --apply` would remove, and a live-owner worktree is never listed.
func GCList(root string, git GitRunner, now time.Time, maxAge time.Duration, processAlive ProcessLiveFn, leaseLive LeaseIDLiveFn) []GCWorktree {
	if maxAge <= 0 {
		maxAge = DefaultColdAgeFloor
	}
	_, paths := Count(root, git)
	out := make([]GCWorktree, 0, len(paths))
	for _, wtPath := range paths {
		row := GCWorktree{Path: wtPath, OwnerLive: true, LeaseLive: true, Reason: "kept: owner liveness unknown"}
		stamp, err := readOwnerStamp(wtPath)
		if err != nil {
			continue
		}
		row.Owner = stamp
		age := ownerAge(stamp, now)
		row.AgeSec = int64(age / time.Second)

		if processAlive == nil {
			continue
		}
		row.OwnerLive = processAlive(stamp.PID)
		if row.OwnerLive {
			continue
		}

		if leaseLive == nil || stamp.LeaseID == "" {
			continue
		}
		row.LeaseLive = leaseLive(stamp.LeaseID)
		if row.LeaseLive {
			continue
		}

		if age < maxAge {
			continue
		}

		row.Unlanded = UnlandedCount(wtPath, git)
		switch {
		case row.Unlanded < 0:
			continue
		case row.Unlanded > 0:
			continue
		default:
			row.Eligible = true
			row.Reason = fmt.Sprintf("gc: owner pid %d dead, lease %q released, age %s past max-age %s, working tree clean",
				stamp.PID, stamp.LeaseID, age.Round(time.Second), maxAge.Round(time.Second))
			out = append(out, row)
		}
	}
	return out
}

// GarbageCollect returns the dry-run plan by default. Apply removes only Eligible
// worktrees through ForceReap; a failed removal stays visible with Removed=false.
func GarbageCollect(root string, git GitRunner, opts GCOptions) GCReport {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.MaxAge <= 0 {
		opts.MaxAge = DefaultColdAgeFloor
	}
	report := GCReport{
		Mode:      "dry-run",
		MaxAgeSec: int64(opts.MaxAge / time.Second),
		Worktrees: GCList(root, git, opts.Now, opts.MaxAge, opts.ProcessAlive, opts.LeaseLive),
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
			res := ForceReap(root, row.Path, git)
			row.Removed = res.Removed
			if res.Removed {
				report.Reaped++
			}
		}
	}
	return report
}
