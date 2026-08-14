package workerworktree

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// THE PER-WORKER WORKTREE TAX (#3572, child of #3165)
// Prepare materializes a full detached worktree and Reap destroys it. At fleet
// scale, checkout and .git/worktrees administration become a per-dispatch tax.
//
// THE POOL. Keep a small bounded set of idle worktrees per lane. A new worker
// leases an idle member by atomically changing its durable state to leased, then
// running `reset --hard <base>` and `clean -fd`. Reap returns only a worktree that
// is proven clean and whose detached HEAD is already contained by trunk; overflow
// and unsafe returns fall back to the pre-pool forced removal.
//
// THE STATE. Every pool member has one JSON record under the configured worker
// root. The record exists in BOTH idle and leased states, embeds the owner stamp
// that owns the state transition, and is replaced atomically. A per-lane advisory
// lock serializes capacity checks and idle->leased->idle transitions across the
// separate prepare/reap processes used by dispatch.
const (
	// PoolCapEnv bounds the number of IDLE members kept per lane. Zero disables
	// the pool and preserves the pre-#3572 create/reap lifecycle.
	PoolCapEnv = "FLEET_WORKER_WORKTREE_POOL"

	// Small by default: enough to amortize short-worker churn without retaining an
	// unbounded checkout set. Operators can set PoolCapEnv=0 for the exact old path.
	defaultPoolCap = 2

	poolStateDir      = ".fak-wt-pool"
	poolMemberExt     = ".json"
	poolLegacyIdleExt = ".idle"
	poolMemberSchema  = "fak-worker-worktree-pool/2"
	poolStateIdle     = "idle"
	poolStateLeased   = "leased"
	poolLockPoll      = 10 * time.Millisecond
	poolLockWait      = 2 * time.Second
	poolMetadataPerm  = 0o600
	poolStateDirPerm  = 0o700
	poolLockFilePerm  = 0o600
)

type poolMemberMetadata struct {
	Schema    string     `json:"schema"`
	Lane      string     `json:"lane"`
	State     string     `json:"state"`
	Owner     OwnerStamp `json:"owner"`
	UpdatedAt time.Time  `json:"updated_at"`
}

type poolMemberRecord struct {
	Path string
	Meta poolMemberMetadata
}

// PoolCap reports the effective per-lane idle capacity. Unset, malformed, or
// negative values fall back to the small default; only an explicit zero disables.
func PoolCap() int {
	v := strings.TrimSpace(os.Getenv(PoolCapEnv))
	if v == "" {
		return defaultPoolCap
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return defaultPoolCap
	}
	return n
}

func resolveWorktreeRoot(wtRoot string) string {
	if strings.TrimSpace(wtRoot) == "" {
		return DefaultRoot()
	}
	return filepath.Clean(wtRoot)
}

func poolStatePath(wtRoot string) string {
	return filepath.Join(resolveWorktreeRoot(wtRoot), poolStateDir)
}

// poolMarker retains the old helper name because cleanup and tests use it. It now
// names the durable v2 member record rather than an idle-only marker.
func poolMarker(wtRoot, dirName string) string {
	return filepath.Join(poolStatePath(wtRoot), dirName+poolMemberExt)
}

func poolLegacyMarker(wtRoot, dirName string) string {
	return filepath.Join(poolStatePath(wtRoot), dirName+poolLegacyIdleExt)
}

func poolMemberPath(wtPath string) string {
	clean := filepath.Clean(wtPath)
	return poolMarker(filepath.Dir(clean), filepath.Base(clean))
}

func canonicalPoolLane(lane string) string {
	return LaneOf(DirName(lane, "pool-lane"))
}

func poolLaneLockPath(wtRoot, lane string) string {
	return filepath.Join(poolStatePath(wtRoot), "locks", safeKey(canonicalPoolLane(lane))+".lock")
}

// withPoolLaneLock serializes one lane's member selection, cap accounting, and
// state transition across goroutines and processes. A bounded wait preserves the
// package's fail-open posture: callers fall back to create/remove rather than wedge.
func withPoolLaneLock(wtRoot, lane string, fn func() error) error {
	lane = canonicalPoolLane(lane)
	if lane == "" {
		return fmt.Errorf("pool lane is empty")
	}
	lockPath := poolLaneLockPath(wtRoot, lane)
	if err := os.MkdirAll(filepath.Dir(lockPath), poolStateDirPerm); err != nil {
		return err
	}
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, poolLockFilePerm)
	if err != nil {
		return err
	}
	defer f.Close()

	deadline := time.Now().Add(poolLockWait)
	for {
		err = flock.TryLock(f)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("pool lane %q stayed locked for %s", lane, poolLockWait)
		}
		time.Sleep(poolLockPoll)
	}
	defer func() { _ = flock.Unlock(f) }()
	return fn()
}

// atomicWriteJSON writes a complete JSON record to a sibling temporary file,
// fsyncs it, and atomically replaces the destination. Go's Windows rename path
// uses MOVEFILE_REPLACE_EXISTING, so no remove-then-rename visibility gap is needed.
func atomicWriteJSON(path string, value any, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), poolStateDirPerm); err != nil {
		return err
	}
	b, err := json.MarshalIndent(value, "", "  ")
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
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func validPoolState(state string) bool {
	return state == poolStateIdle || state == poolStateLeased
}

func readPoolMember(wtPath string) (poolMemberMetadata, error) {
	path := poolMemberPath(wtPath)
	b, err := os.ReadFile(path)
	if err != nil {
		return poolMemberMetadata{}, err
	}
	var meta poolMemberMetadata
	if err := json.Unmarshal(b, &meta); err != nil {
		return poolMemberMetadata{}, err
	}
	wantLane := LaneOf(wtPath)
	if meta.Schema != poolMemberSchema ||
		meta.Lane == "" || meta.Lane != wantLane ||
		!validPoolState(meta.State) ||
		meta.Owner.PID <= 0 || meta.Owner.CreatedAt.IsZero() ||
		meta.UpdatedAt.IsZero() {
		return poolMemberMetadata{}, fmt.Errorf("invalid pool member metadata")
	}
	meta.Owner.CreatedAt = meta.Owner.CreatedAt.UTC()
	meta.UpdatedAt = meta.UpdatedAt.UTC()
	return meta, nil
}

// writePoolMemberLocked replaces one member record. The caller must hold the
// canonical lane lock when this is part of a lifecycle transition.
func writePoolMemberLocked(wtPath, lane, state string, owner OwnerStamp, updatedAt time.Time) error {
	lane = canonicalPoolLane(lane)
	if lane == "" || lane != LaneOf(wtPath) {
		return fmt.Errorf("pool lane %q does not match worktree %q", lane, wtPath)
	}
	if !validPoolState(state) {
		return fmt.Errorf("invalid pool state %q", state)
	}
	owner = normalizeOwnerStamp(owner)
	if updatedAt.IsZero() {
		updatedAt = time.Now()
	}
	meta := poolMemberMetadata{
		Schema:    poolMemberSchema,
		Lane:      lane,
		State:     state,
		Owner:     owner,
		UpdatedAt: updatedAt.UTC(),
	}
	return atomicWriteJSON(poolMemberPath(wtPath), meta, poolMetadataPerm)
}

func removePoolMemberState(wtPath string) {
	clean := filepath.Clean(wtPath)
	wtRoot := filepath.Dir(clean)
	dirName := filepath.Base(clean)
	_ = os.Remove(poolMarker(wtRoot, dirName))
	_ = os.Remove(poolLegacyMarker(wtRoot, dirName))
}

// migrateLegacyIdleLocked turns the v1 idle-only marker into a v2 state record.
// Missing owner evidence is kept as legacy state rather than guessed.
func migrateLegacyIdleLocked(wtRoot, wtPath string) (poolMemberMetadata, error) {
	owner, err := readOwnerStamp(wtPath)
	if err != nil {
		return poolMemberMetadata{}, err
	}
	updatedAt := owner.CreatedAt
	if fi, statErr := os.Stat(poolLegacyMarker(wtRoot, filepath.Base(wtPath))); statErr == nil {
		updatedAt = fi.ModTime()
	}
	if err := writePoolMemberLocked(wtPath, LaneOf(wtPath), poolStateIdle, owner, updatedAt); err != nil {
		return poolMemberMetadata{}, err
	}
	_ = os.Remove(poolLegacyMarker(wtRoot, filepath.Base(wtPath)))
	return readPoolMember(wtPath)
}

// poolMembersLocked enumerates valid durable member records for one lane. Vanished
// worktrees release their slots. Legacy idle markers are upgraded in place.
func poolMembersLocked(wtRoot, lane string) []poolMemberRecord {
	wtRoot = resolveWorktreeRoot(wtRoot)
	lane = canonicalPoolLane(lane)
	entries, err := os.ReadDir(poolStatePath(wtRoot))
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var out []poolMemberRecord
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		var dirName string
		switch {
		case strings.HasSuffix(name, poolMemberExt):
			dirName = strings.TrimSuffix(name, poolMemberExt)
		case strings.HasSuffix(name, poolLegacyIdleExt):
			dirName = strings.TrimSuffix(name, poolLegacyIdleExt)
		default:
			continue
		}
		wt := filepath.Join(wtRoot, dirName)
		key := strings.ToLower(filepath.Clean(wt))
		if seen[key] {
			continue
		}
		if _, err := os.Stat(wt); err != nil {
			removePoolMemberState(wt)
			continue
		}
		if lane != "" && LaneOf(wt) != lane {
			continue
		}
		meta, err := readPoolMember(wt)
		if err != nil && strings.HasSuffix(name, poolLegacyIdleExt) {
			meta, err = migrateLegacyIdleLocked(wtRoot, wt)
		}
		if err != nil {
			continue
		}
		seen[key] = true
		out = append(out, poolMemberRecord{Path: wt, Meta: meta})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

func poolIdleMembers(wtRoot, lane string) []string {
	var out []string
	_ = withPoolLaneLock(wtRoot, lane, func() error {
		for _, member := range poolMembersLocked(wtRoot, lane) {
			if member.Meta.State == poolStateIdle {
				out = append(out, member.Path)
			}
		}
		return nil
	})
	return out
}

// recordPoolLease refreshes the durable leased owner after Prepare has secured a
// member (or created a new one). Pool-disabled mode writes no pool state at all.
func recordPoolLease(wtPath, lane string, owner OwnerStamp) error {
	if PoolCap() == 0 {
		return nil
	}
	wtRoot := filepath.Dir(filepath.Clean(wtPath))
	return withPoolLaneLock(wtRoot, lane, func() error {
		return writePoolMemberLocked(wtPath, lane, poolStateLeased, owner, time.Now())
	})
}

// resetPooled makes a materialized checkout equivalent to a fresh detached add at
// base while preserving the intentionally ignored per-worktree Go caches.
func resetPooled(wt, base string, git GitRunner) bool {
	if rc, _ := run(git, wt, []string{"reset", "--hard", base}); rc != 0 {
		return false
	}
	rc, _ := run(git, wt, []string{"clean", "-fd", "-e", ".gocache", "-e", ".gotmp"})
	return rc == 0
}

func reserveAndResetLocked(root string, member poolMemberRecord, base string, git GitRunner, owner OwnerStamp) (Result, bool) {
	meta := member.Meta
	meta.State = poolStateLeased
	meta.Owner = normalizeOwnerStamp(owner)
	meta.UpdatedAt = time.Now().UTC()
	if err := atomicWriteJSON(poolMemberPath(member.Path), meta, poolMetadataPerm); err != nil {
		return Result{}, false
	}
	if resetPooled(member.Path, base, git) {
		return Result{
			OK: true, Path: member.Path, BaseSHA: base, Reused: true,
			Reason: "warm worktree pool hit (#3572)", pooled: true,
		}, true
	}
	// The reservation prevents a second lease while the broken member is removed.
	_ = ForceReap(root, member.Path, git)
	return Result{}, false
}

// leasePooled leases the first deterministic idle member of lane. The lane lock
// covers selection, state transition, and reset so return/GC cannot race the lease.
func leasePooled(root, lane, base, wtRoot string, git GitRunner, owner OwnerStamp) (Result, bool) {
	var result Result
	var leased bool
	err := withPoolLaneLock(wtRoot, lane, func() error {
		for _, member := range poolMembersLocked(wtRoot, lane) {
			if member.Meta.State != poolStateIdle {
				continue
			}
			if res, ok := reserveAndResetLocked(root, member, base, git, owner); ok {
				result, leased = res, true
				return nil
			}
		}
		return nil
	})
	return result, err == nil && leased
}

// leaseSpecificPooled handles the same-key case when that path is currently an
// idle pool member. It must be reset like any other new lease, not returned via the
// historical same-key retry path that intentionally preserves leased WIP.
func leaseSpecificPooled(root, wtPath, base string, git GitRunner, owner OwnerStamp) (Result, bool) {
	lane := LaneOf(wtPath)
	wtRoot := filepath.Dir(filepath.Clean(wtPath))
	var result Result
	var leased bool
	err := withPoolLaneLock(wtRoot, lane, func() error {
		meta, err := readPoolMember(wtPath)
		if err != nil || meta.State != poolStateIdle {
			return nil
		}
		result, leased = reserveAndResetLocked(root, poolMemberRecord{Path: wtPath, Meta: meta}, base, git, owner)
		return nil
	})
	return result, err == nil && leased
}

// returnPooled parks a proven-safe worktree as idle. It never resets a dirty,
// unreadable, or unpushed member: those cases return handled=false and the caller
// executes the old forced-removal path instead.
func returnPooled(root, wtPath string, capacity int, git GitRunner) (result Result, handled bool, fallback string) {
	lane := LaneOf(wtPath)
	if lane == "" {
		return Result{}, false, "unclassifiable_lane"
	}
	wtRoot := filepath.Dir(filepath.Clean(wtPath))
	err := withPoolLaneLock(wtRoot, lane, func() error {
		idle := 0
		for _, member := range poolMembersLocked(wtRoot, lane) {
			if member.Meta.State == poolStateIdle && !samePath(member.Path, wtPath) {
				idle++
			}
		}
		if idle >= capacity {
			fallback = "pool_capacity"
			return nil
		}

		switch unlanded := UnlandedCount(wtPath, git); {
		case unlanded < 0:
			fallback = "working_tree_unreadable"
			return nil
		case unlanded > 0:
			fallback = "working_tree_dirty"
			return nil
		}
		switch unpushed := unpushedCommitCount(root, wtPath, git); {
		case unpushed < 0:
			fallback = "commit_ancestry_unreadable"
			return nil
		case unpushed > 0:
			fallback = "unpushed_commit"
			return nil
		}

		owner, err := readOwnerStamp(wtPath)
		if err != nil {
			fallback = "owner_stamp_unreadable"
			return nil
		}
		// The safety probes above prove this reset discards no unlanded or detached
		// durable work. It merely normalizes the already-clean checkout before idle.
		if !resetPooled(wtPath, "HEAD", git) {
			fallback = "reset_failed"
			return nil
		}
		if err := writePoolMemberLocked(wtPath, lane, poolStateIdle, owner, time.Now()); err != nil {
			fallback = "pool_metadata_write_failed"
			return nil
		}
		result = Result{
			OK: true, Path: wtPath, Removed: false,
			Reason: "returned to warm worktree pool (#3572)",
		}
		handled = true
		return nil
	})
	if err != nil {
		return Result{}, false, "pool_lane_lock_failed"
	}
	return result, handled, fallback
}
