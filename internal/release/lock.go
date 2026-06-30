// Package release holds the single, process-safe release lock that every code
// path which mutates the two single-writer release resources — the bare VERSION
// marker and the monotone vX.Y.Z tag sequence — must take before its critical
// section.
//
// # Why a shared primitive
//
// The fleet is a live multi-session cluster: several agent sessions and, since
// default-on auto-cut (#1355), an UNATTENDED cadence tick that fires every two
// hours. A release mutates VERSION and the tag sequence, and two concurrent
// cutters race them: both bump VERSION, both try to create the same tag, then
// collide on push. tools/release_lock.py already guards concurrent HUMAN
// /release sessions, but the scheduled auto-cut path and the human path are on
// different code paths and did not share a lock (#1391). This package is the one
// primitive both paths call so they serialize against EACH OTHER, not just
// within their own kind.
//
// # Policy (mirrors tools/release_lock.py exactly)
//
//   - The lock is a JSON file at a well-known path (.release.lock at the repo
//     root, overridable via FAK_RELEASE_LOCK_ROOT) created with O_EXCL so the
//     create is the atomic mutual-exclusion act.
//   - Staleness is TTL-based, NOT pid-based — on purpose. os.Kill(pid, 0) is a
//     liveness probe on POSIX but TERMINATES the target on Windows (this is a
//     Windows host), so we never signal. A crashed cutter's lock simply expires
//     after its TTL (default 30 min); the recorded pid/host are diagnostics only.
//   - Owner identity is a token stable across a session's separate process
//     invocations: FAK_RELEASE_OWNER, else CLAUDE_CODE_SESSION_ID, else a
//     host+user+pid fallback (which is unique but NOT stable across calls — real
//     sessions hit the env path; this only bites bare-shell manual use).
//   - A live lock held by ANOTHER owner is refused (ErrHeld). A stale lock is
//     taken over (StealStale); --force steals a live one. Release only frees a
//     lock the caller owns (unless forced).
//
// The read-modify-write of the lockfile is itself serialized with the
// internal/flock advisory lock on a sibling .release.lock.guard fd, so two
// processes that race the steal path cannot both unlink-then-create.
package release

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// LockName is the well-known lockfile basename, shared with tools/release_lock.py
// so the Go and Python paths contend on the SAME file.
const LockName = ".release.lock"

// DefaultTTL is how long a lock stays live before it is considered stale and
// stealable — a crashed cutter auto-recovers after this. Matches the Python
// helper's DEFAULT_TTL (30 min).
const DefaultTTL = 30 * time.Minute

// ErrHeld is returned by Acquire when a live lock is held by another owner and
// neither steal-stale nor force applies.
var ErrHeld = errors.New("release: lock held by another owner")

// ErrNotOwner is returned by Release when the caller does not own the live lock
// and force was not set.
var ErrNotOwner = errors.New("release: lock not owned by caller")

// Lock is the on-disk lock record. The JSON shape is compatible with
// tools/release_lock.py so a lock written by either side is readable by the
// other.
type Lock struct {
	Owner      string  `json:"owner"`
	PID        int     `json:"pid"`
	Host       string  `json:"host"`
	Branch     string  `json:"branch,omitempty"`
	GitUser    string  `json:"git_user,omitempty"`
	HeadSHA    string  `json:"head_sha,omitempty"`
	AcquiredAt float64 `json:"acquired_at"`
	TTL        float64 `json:"ttl"`
	ExpiresAt  float64 `json:"expires_at"`
	Note       string  `json:"note,omitempty"`
}

// State is a human/diagnostic view of the current lock.
type State struct {
	Held      bool    `json:"held"`
	Stale     bool    `json:"stale"`
	Reason    string  `json:"reason"`
	Lock      *Lock   `json:"lock"`
	Remaining float64 `json:"remaining_s,omitempty"`
	Stolen    *Lock   `json:"stole,omitempty"`
}

// Options control where the lock lives and identity/clock injection for tests.
type Options struct {
	// Root is the repo root the lockfile lives under. FAK_RELEASE_LOCK_ROOT, if
	// set, overrides this. Empty Root falls back to the current directory.
	Root string
	// Owner is the identity token. Empty resolves via DefaultOwner.
	Owner string
	// Now injects the clock (tests); zero uses time.Now.
	Now func() time.Time
}

func (o Options) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}

func (o Options) lockPath() string {
	if env := strings.TrimSpace(os.Getenv("FAK_RELEASE_LOCK_ROOT")); env != "" {
		return filepath.Join(env, LockName)
	}
	root := o.Root
	if root == "" {
		root = "."
	}
	return filepath.Join(root, LockName)
}

func (o Options) owner() string {
	if o.Owner != "" {
		return o.Owner
	}
	return DefaultOwner()
}

// DefaultOwner resolves the owner token the same way tools/release_lock.py does:
// FAK_RELEASE_OWNER, else CLAUDE_CODE_SESSION_ID, else host+user+pid (unique but
// not stable across calls — only bare-shell manual use lands here).
func DefaultOwner() string {
	if v := strings.TrimSpace(os.Getenv("FAK_RELEASE_OWNER")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID")); v != "" {
		return v
	}
	user := os.Getenv("USERNAME")
	if user == "" {
		user = os.Getenv("USER")
	}
	if user == "" {
		user = "user"
	}
	return fmt.Sprintf("%s@%s:%d", user, hostname(), os.Getpid())
}

func hostname() string {
	h, err := os.Hostname()
	if err != nil || h == "" {
		return "?"
	}
	return h
}

func epoch(t time.Time) float64 { return float64(t.UnixNano()) / 1e9 }

// readLock parses the on-disk lock, or nil if absent/corrupt (corrupt is treated
// as stealable by Acquire, matching the Python helper).
func readLock(path string) *Lock {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var l Lock
	if err := json.Unmarshal(data, &l); err != nil {
		return nil
	}
	return &l
}

// isStale reports whether the lock is past its expiry at time at. A lock with no
// usable expiry (and no reconstructable acquired+ttl) is treated as stale.
func isStale(l *Lock, at time.Time) (bool, string) {
	now := epoch(at)
	expires := l.ExpiresAt
	if expires == 0 {
		if l.AcquiredAt != 0 && l.TTL != 0 {
			expires = l.AcquiredAt + l.TTL
		} else {
			return true, "no expiry recorded"
		}
	}
	if now >= expires {
		return true, "expired"
	}
	return false, "live"
}

// guardLock serializes the read-modify-write of the lockfile across processes via
// the internal/flock advisory lock on a sibling .guard fd. The returned release
// func must be called to drop the guard.
func guardLock(lockPath string) (func(), error) {
	guardPath := lockPath + ".guard"
	f, err := os.OpenFile(guardPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	// Bounded poll: the guarded section is a single unlink+create, sub-ms; a few
	// retries cover a concurrent stealer.
	var locked bool
	for i := 0; i < 200; i++ {
		err = flock.TryLock(f)
		if err == nil {
			locked = true
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			f.Close()
			return nil, err
		}
		time.Sleep(2 * time.Millisecond)
	}
	if !locked {
		f.Close()
		return nil, fmt.Errorf("release: guard contended: %w", flock.ErrLockBusy)
	}
	return func() {
		_ = flock.Unlock(f)
		_ = f.Close()
	}, nil
}

// Acquire takes the release lock for opts.Owner with the given ttl. A live lock
// held by another owner is refused with ErrHeld unless stealStale (and the lock
// is stale) or force is set. On success the returned State carries the written
// Lock (and, if a takeover happened, the stolen lock in State.Stolen).
func Acquire(opts Options, ttl time.Duration, note string, stealStale, force bool) (*State, error) {
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	path := opts.lockPath()
	unguard, err := guardLock(path)
	if err != nil {
		return nil, err
	}
	defer unguard()

	at := opts.now()
	started := epoch(at)
	rec := &Lock{
		Owner:      opts.owner(),
		PID:        os.Getpid(),
		Host:       hostname(),
		Branch:     gitOut(opts.Root, "rev-parse", "--abbrev-ref", "HEAD"),
		GitUser:    gitOut(opts.Root, "config", "user.name"),
		HeadSHA:    gitOut(opts.Root, "rev-parse", "--short", "HEAD"),
		AcquiredAt: started,
		TTL:        ttl.Seconds(),
		ExpiresAt:  started + ttl.Seconds(),
		Note:       note,
	}

	var stolen *Lock
	existing := readLock(path)
	if existing != nil {
		stale, _ := isStale(existing, at)
		switch {
		case force, stealStale && stale:
			stolen = existing
			_ = os.Remove(path)
		default:
			return &State{Held: false, Reason: "held", Lock: existing}, ErrHeld
		}
	}

	if err := writeLockExcl(path, rec); err != nil {
		if errors.Is(err, os.ErrExist) {
			// Lost a race after our own takeover decision; re-read and refuse.
			return &State{Held: false, Reason: "held", Lock: readLock(path)}, ErrHeld
		}
		return nil, err
	}
	st := &State{Held: true, Reason: "held", Lock: rec, Remaining: ttl.Seconds(), Stolen: stolen}
	return st, nil
}

// writeLockExcl atomically creates the lockfile (O_EXCL) and writes the record.
func writeLockExcl(path string, rec *Lock) error {
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Release frees the lock iff it is owned by opts.Owner (or force). A missing lock
// is a no-op success. A live lock owned by another caller refuses with
// ErrNotOwner.
func Release(opts Options, force bool) (*State, error) {
	path := opts.lockPath()
	unguard, err := guardLock(path)
	if err != nil {
		return nil, err
	}
	defer unguard()

	l := readLock(path)
	if l == nil {
		return &State{Held: false, Reason: "no lock present"}, nil
	}
	if !force && l.Owner != opts.owner() {
		return &State{Held: true, Reason: "not owner", Lock: l}, ErrNotOwner
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	return &State{Held: false, Reason: "released", Lock: l}, nil
}

// Status reports the current lock without mutating it: held/stale/free plus the
// remaining seconds. It does not take the guard — a status read is advisory.
func Status(opts Options) *State {
	path := opts.lockPath()
	l := readLock(path)
	if l == nil {
		return &State{Held: false, Reason: "free", Lock: nil}
	}
	at := opts.now()
	stale, why := isStale(l, at)
	st := &State{Held: !stale, Stale: stale, Reason: why, Lock: l}
	if l.ExpiresAt != 0 {
		st.Remaining = l.ExpiresAt - epoch(at)
	}
	return st
}

// HeldByOther reports whether a LIVE lock is held by someone other than
// opts.Owner — the single predicate the scheduled auto-cut consults to decide
// whether to defer to the next tick when a human /release is mid-flight.
func HeldByOther(opts Options) (bool, *Lock) {
	st := Status(opts)
	if !st.Held || st.Lock == nil {
		return false, nil
	}
	if st.Lock.Owner == opts.owner() {
		return false, st.Lock
	}
	return true, st.Lock
}

// gitOut runs git in root and returns the trimmed stdout, or "" on any failure.
// Diagnostics only — a missing git never blocks a lock op.
func gitOut(root string, args ...string) string {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	if root != "" {
		cmd.Dir = root
	}
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
