// ticklock.go — the host-level tick lock that serializes overlapping resume-watchdog
// ticks (#3110). Every liveness/census check a tick makes (sessionLive's per-tick
// process-table snapshot, the host-wide MaxLiveResumes census) is a prepare-time read;
// the actual spawns land afterward. Two independent `fak` processes — a slow tick plus a
// cron/`--live`/manual tick — each hold their own stale snapshot, and nothing
// cross-process serializes them: both can admit the SAME still-starting session and
// briefly run two `claude --resume` on one transcript (wasted tokens, transcript
// contention). TryTickLock is the cheap host-level backstop: one `fak resume watchdog`
// tick holds the lockfile for its whole run, so a second concurrent tick observes it
// held and skips cleanly rather than racing the same plan.
//
// # Design
//
// A single well-known file (resume_watchdog.lock) in the registry dir, created with
// O_CREATE|O_EXCL so the create itself is the atomic mutual-exclusion act — no
// read-modify-write, no separate guard fd. Staleness is TTL-based on the lockfile's
// mtime, deliberately NOT pid-liveness based: internal/resume must not import procguard
// (known import cycles), and a real tick is seconds long, so a lock older than the TTL
// can only be a crashed holder's leftover, safe to reclaim. This mirrors
// internal/release's TTL-not-pid staleness policy for the same
// os.Kill(pid, 0)-terminates-on-Windows reason, just without that package's flock
// guard/JSON-record machinery — a resume tick only ever needs "am I the one tick
// running right now", not an owner identity or a renew heartbeat.
package resume

import (
	"github.com/anthony-chaudhary/fak/internal/exclusivefile"
	"os"
	"path/filepath"
	"time"
)

// TickLockName is the well-known lockfile basename inside the registry dir.
const TickLockName = "resume_watchdog.lock"

// TickLockTTL is how old an existing lockfile's mtime must be before it is treated as
// abandoned (a crashed tick) and reclaimed. Comfortably longer than any real tick
// (per-tick cap x launch spacing is on the order of tens of seconds), so a live tick's
// lock is never mistaken for stale.
const TickLockTTL = 2 * time.Minute

// TryTickLock attempts to acquire the host-level resume-watchdog tick lock in regDir.
//
// acquired=true means the caller now owns the lock and must call release() when the
// tick ends (typically via defer). acquired=false with err==nil means another tick
// currently holds a live lock — the caller should skip this tick cleanly, not treat it
// as an error. A non-nil err means the lock could not even be evaluated (e.g. regDir
// unwritable); the caller should fail OPEN rather than block the tick on a broken lock.
func TryTickLock(regDir string) (release func() error, acquired bool, err error) {
	path := filepath.Join(regDir, TickLockName)

	if release, acquired, cerr, held := attemptTickLock(path); !held {
		return release, acquired, cerr
	}

	// The lockfile already exists. Live (another tick holds it) or stale (a crashed
	// tick's leftover, past TickLockTTL)?
	fi, statErr := os.Stat(path)
	switch {
	case statErr == nil && time.Since(fi.ModTime()) < TickLockTTL:
		return noopRelease, false, nil // live: another tick holds it
	case statErr != nil && !os.IsNotExist(statErr):
		return noopRelease, false, statErr // real stat failure, not "it vanished"
	}
	// Either stale (mtime past TTL) or it vanished between our failed create and this
	// stat (the holder released concurrently) — either way, reclaim: remove + retry the
	// create once.
	_ = os.Remove(path)
	if release, acquired, cerr, held := attemptTickLock(path); !held {
		return release, acquired, cerr
	}
	// A concurrent tick won the retry-create race; back off correctly rather than
	// force a second acquire.
	return noopRelease, false, nil
}

// attemptTickLock runs the one O_EXCL create act and folds its outcome into
// TryTickLock's return shape — the step TryTickLock performs twice, once on the
// first try and once after reclaiming a stale lockfile. held=true means the create
// lost to an existing lockfile (os.IsExist); the caller decides whether that means
// "reclaim and retry" or "back off", which is the only thing the two sites differ on.
func attemptTickLock(path string) (release func() error, acquired bool, err error, held bool) {
	if cerr := createTickLock(path); cerr == nil {
		return tickLockReleaser(path), true, nil, false
	} else if !os.IsExist(cerr) {
		return noopRelease, false, cerr, false
	}
	return noopRelease, false, nil, true
}

// createTickLock does the one atomic O_EXCL create that IS the acquire act, then best-
// effort records "pid unix-ts" for a human/diagnostic read. A write/close failure after
// a successful O_EXCL create does not undo the acquire — the caller already owns the
// lock; only the create itself can fail with os.IsExist for TryTickLock to read as
// "held".
func createTickLock(path string) error {
	return exclusivefile.CreatePIDTime(path)
}

func tickLockReleaser(path string) func() error {
	return func() error {
		err := os.Remove(path)
		if err != nil && os.IsNotExist(err) {
			return nil
		}
		return err
	}
}

func noopRelease() error { return nil }
