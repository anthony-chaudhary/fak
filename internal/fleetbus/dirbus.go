package fleetbus

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// Bus is the transport contract. It is an interface so the cross-host transports
// named in epic #5599 (the git-mediated leaseref path, the private comms bridge) are
// a swap rather than a rewrite. Exactly ONE implementation ships in the spine:
// DirBus.
//
// Every method is safe to call from several processes at once. That is the whole
// difficulty of the contract, and the reason ClaimApply is on it rather than left to
// each caller: exactly-once APPLICATION under at-least-once DELIVERY is a property
// only the transport can provide, and a transport that quietly omitted it would let
// every consumer re-derive it differently.
type Bus interface {
	// Announce publishes (or refreshes) this process's presence record.
	Announce(inst Instance) error
	// Instances returns the roster of instances still fresh at now. This is the
	// DENOMINATOR the ack fold needs; without it "outstanding" is uncomputable.
	Instances(now time.Time, ttl time.Duration) ([]Instance, error)
	// Publish appends a directive. Publishing is NOT proof of anything (see the
	// package doc) — it only makes the directive available to drainers.
	Publish(d Directive) error
	// Directives returns the live directive log, oldest first. Expired directives
	// are INCLUDED: a drainer must be able to ack one expired rather than pretend
	// it never saw it.
	Directives() ([]Directive, error)
	// ClaimApply atomically stakes instanceID's one-and-only attempt at directiveID.
	// It returns true exactly once per pair, ever; every later call returns false.
	ClaimApply(instanceID, directiveID string) (bool, error)
	// Ack appends one instance's outcome.
	Ack(a Ack) error
	// Acks returns every ack recorded for one directive.
	Acks(directiveID string) ([]Ack, error)
}

// --- directory-backed transport -------------------------------------------- //

// Layout under the bus root. The shape is chosen so that the only cross-process
// coordination needed is (a) an advisory lock around the two append-only ledgers and
// (b) one atomic O_EXCL create for the apply claim — no daemon, no lock server, no
// leader.
//
//	<root>/instances/<instance>.json          presence, replaced whole each announce
//	<root>/directives.jsonl                   the directive log (+ .1 sealed generation)
//	<root>/acks/<directive>.jsonl             one ack log per directive
//	<root>/applied/<instance>/<directive>     the apply claim marker
//
// Acks are partitioned per directive rather than kept in one log so that N instances
// answering N different directives never contend on one lock, and so folding one
// directive's report is a single small read instead of a scan of the fleet's history.
const (
	instancesDir  = "instances"
	directivesLog = "directives.jsonl"
	acksDir       = "acks"
	appliedDir    = "applied"
)

// DefaultLockWait bounds how long an append waits for the ledger lock before giving
// up. Appends are small and the critical section is a few syscalls, so real
// contention is brief; the ceiling exists so a crashed holder on Windows (whose
// LockFileEx region can outlive the process) degrades to a loud error instead of a
// wedged drain loop.
const DefaultLockWait = 5 * time.Second

// lockPoll is the gap between flock.TryLock attempts. flock is non-blocking by
// design, so a blocking acquire IS this poll — the same idiom idempotency.withLock
// and loopmgr.withLedgerLock use.
const lockPoll = 15 * time.Millisecond

// DirBus is the v1 transport: a shared DIRECTORY. On one machine that is a real
// cross-PROCESS control plane. It is an honest cross-HOST one only where the
// directory itself is shared (a UNC path, an SMB/NFS mount) — which is exactly what
// FLEET_STATE_DIR already exists to point at. That limitation is stated, not hidden:
// a genuine network transport is a separate child of epic #5599.
type DirBus struct {
	// Root is the bus directory.
	Root string
	// MaxLedgerBytes bounds each append-only log (0 = jsonlledger's default). The
	// active file rotates to ".1" before it would cross the bound, and readers read
	// both, so the bound costs history, never a live directive.
	MaxLedgerBytes int64
	// LockWait bounds ledger-lock acquisition (0 = DefaultLockWait).
	LockWait time.Duration
}

var _ Bus = (*DirBus)(nil)

// OpenDir prepares a bus rooted at dir, creating the layout if it is not there.
func OpenDir(dir string) (*DirBus, error) {
	if dir == "" {
		return nil, errors.New("fleetbus: bus root is required")
	}
	for _, sub := range []string{instancesDir, acksDir, appliedDir} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			return nil, fmt.Errorf("fleetbus: prepare %s: %w", dir, err)
		}
	}
	return &DirBus{Root: dir}, nil
}

func (b *DirBus) lockWait() time.Duration {
	if b.LockWait > 0 {
		return b.LockWait
	}
	return DefaultLockWait
}

// --- presence -------------------------------------------------------------- //

// Announce replaces this instance's presence record. The write is temp-then-rename
// so a concurrent reader never observes a half-written record — a torn presence file
// would drop an instance out of the roster and silently shrink the denominator.
//
// The temp name is UNIQUE PER WRITE, not "<id>.json.tmp". Two writers wearing one
// identity is a real state (a restart overlapping a shutdown, a copied config), and a
// shared temp path would let them interleave into the same file and then rename the
// torn result into place — defeating the exact atomicity this dance exists for. The
// temp does not end in ".json", so a strand left by a crash is invisible to the roster
// reader rather than a phantom instance.
//
// The rename is serialized on a per-record lock, which is a PLATFORM requirement and
// not belt-and-braces. POSIX rename(2) onto a busy destination is atomic and always
// succeeds; Windows MoveFileEx(MOVEFILE_REPLACE_EXISTING) fails ERROR_ACCESS_DENIED
// when two replaces of one destination overlap. Without the lock this call returns a
// hard error on Windows exactly when the fleet is at its most interesting — a serve
// restarting into its own still-live identity — and the instance drops out of the
// roster, shrinking the denominator a control point measures completeness against.
// The lock file lives beside the record and ends in ".lock", so the roster reader (which
// admits only ".json") never sees it.
func (b *DirBus) Announce(inst Instance) error {
	if r := inst.Validate(); r != nil {
		return r
	}
	data, err := json.Marshal(inst)
	if err != nil {
		return err
	}
	dir := filepath.Join(b.Root, instancesDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	final := filepath.Join(dir, inst.ID+".json")
	return b.withLedgerLock(final, func() error {
		tmp, err := os.CreateTemp(dir, inst.ID+".json.*.tmp")
		if err != nil {
			return err
		}
		name := tmp.Name()
		if _, err := tmp.Write(append(data, '\n')); err != nil {
			tmp.Close()
			_ = os.Remove(name)
			return err
		}
		if err := tmp.Close(); err != nil {
			_ = os.Remove(name)
			return err
		}
		if err := os.Rename(name, final); err != nil {
			_ = os.Remove(name)
			return err
		}
		return nil
	})
}

// Instances returns the fresh roster, sorted by id for a stable report order. A
// record that is unparsable, mis-schema'd, or stale is left out — but never deleted
// here: reaping is an operator act, and a reader that silently rewrote the roster it
// was asked to read would be the wrong kind of helpful.
func (b *DirBus) Instances(now time.Time, ttl time.Duration) ([]Instance, error) {
	entries, err := os.ReadDir(filepath.Join(b.Root, instancesDir))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []Instance
	for _, e := range entries {
		if e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			return nil, fmt.Errorf("instance record %s is a directory", e.Name())
		}
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(b.Root, instancesDir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read instance %s: %w", e.Name(), err)
		}
		var inst Instance
		if err := json.Unmarshal(data, &inst); err != nil {
			continue
		}
		if inst.Validate() != nil || !inst.Fresh(now, ttl) {
			continue
		}
		out = append(out, inst)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// --- directives ------------------------------------------------------------ //

// Publish appends a validated directive to the log.
func (b *DirBus) Publish(d Directive) error {
	if r := d.Validate(); r != nil {
		return r
	}
	line, err := json.Marshal(d)
	if err != nil {
		return err
	}
	path := filepath.Join(b.Root, directivesLog)
	return b.withLedgerLock(path, func() error {
		return jsonlledger.AppendBounded(path, line, b.MaxLedgerBytes)
	})
}

// Directives returns every well-formed directive in the log, oldest first, reading
// the sealed generation ahead of the active one so a rotation mid-flight cannot hide
// a directive that is still inside its TTL.
func (b *DirBus) Directives() ([]Directive, error) {
	path := filepath.Join(b.Root, directivesLog)
	var out []Directive
	for _, p := range []string{path + ".1", path} {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		out = append(out, jsonlledger.Parse(string(data), func(d Directive) bool {
			return d.Validate() == nil
		})...)
	}
	return out, nil
}

// --- the apply claim ------------------------------------------------------- //

// ClaimApply stakes instanceID's single attempt at directiveID by creating a marker
// with O_CREATE|O_EXCL — one filesystem operation that is atomic on both NTFS and
// POSIX, needs no lock, and is permanent.
//
// Why not internal/idempotency, which already does keyed cross-process dedup? Two
// reasons, both structural. Its lock is per-LEDGER and is held across the caller's
// whole apply, so every instance in the fleet would serialize behind one lock — the
// exact opposite of a fan-out. And its dedup is time-WINDOWED (DefaultWindow, 24h),
// so a directive could legitimately re-apply a day later; for a control plane, a
// stale `go` re-firing tomorrow is precisely the failure this package exists to
// prevent. The claim here is instead partitioned per instance (no cross-instance
// contention at all) and never expires.
//
// The claim is taken BEFORE the apply, which makes this at-most-once, not
// exactly-once: an instance that dies between claiming and acking leaves a directive
// it never applied and never answered for. That is deliberate. The failure surfaces
// at the control point as OUTSTANDING — visibly unanswered — whereas claiming after
// the apply would risk applying twice and reporting success. An honest gap beats a
// silent double-apply.
func (b *DirBus) ClaimApply(instanceID, directiveID string) (bool, error) {
	if !ValidToken(instanceID) || !ValidToken(directiveID) {
		return false, refuse(Malformed, "claim needs bus tokens (instance %q, directive %q)", instanceID, directiveID)
	}
	dir := filepath.Join(b.Root, appliedDir, instanceID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return false, err
	}
	f, err := os.OpenFile(filepath.Join(dir, directiveID), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return false, nil
		}
		return false, err
	}
	return true, f.Close()
}

// --- acks ------------------------------------------------------------------ //

// Ack appends one instance's outcome to the directive's ack log.
func (b *DirBus) Ack(a Ack) error {
	if r := a.Validate(); r != nil {
		return r
	}
	line, err := json.Marshal(a)
	if err != nil {
		return err
	}
	dir := filepath.Join(b.Root, acksDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, a.Directive+".jsonl")
	return b.withLedgerLock(path, func() error {
		return jsonlledger.AppendBounded(path, line, b.MaxLedgerBytes)
	})
}

// Acks returns every well-formed ack for one directive, in append order.
func (b *DirBus) Acks(directiveID string) ([]Ack, error) {
	if !ValidToken(directiveID) {
		return nil, refuse(Malformed, "ack lookup needs a bus token, got %q", directiveID)
	}
	path := filepath.Join(b.Root, acksDir, directiveID+".jsonl")
	var out []Ack
	for _, p := range []string{path + ".1", path} {
		data, err := os.ReadFile(p)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read ack ledger %s: %w", filepath.Base(p), err)
		}
		out = append(out, jsonlledger.Parse(string(data), func(a Ack) bool {
			return a.Validate() == nil
		})...)
	}
	return out, nil
}

// --- shared append lock ---------------------------------------------------- //

// withLedgerLock serializes a mutation of <path> across processes (and goroutines —
// flock conflicts between two open handles to one file, so it is not merely advisory
// between processes) on <path>.lock. It guards the rotate-then-append sequence, not
// just the write: two processes that both decided to rotate would otherwise race
// os.Rename and lose one generation. Announce reuses it for the same reason at a
// different path: its replace-in-place is also a rename nobody else may be doing.
//
// The wall clock here is real time.Now, deliberately: this is I/O back-pressure, not
// domain logic. Every clock that decides an OUTCOME on this bus (freshness, expiry,
// the fold) is injected by the caller.
func (b *DirBus) withLedgerLock(path string, fn func() error) error {
	lockPath := path + ".lock"
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return err
	}
	deadline := time.Now().Add(b.lockWait())
	for {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
		if err != nil {
			return err
		}
		err = flock.TryLock(f)
		if err == nil {
			runErr := fn()
			_ = flock.Unlock(f)
			_ = f.Close()
			return runErr
		}
		_ = f.Close()
		if !errors.Is(err, flock.ErrLockBusy) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("fleetbus: ledger %s stayed locked for %s", filepath.Base(path), b.lockWait())
		}
		time.Sleep(lockPoll)
	}
}
