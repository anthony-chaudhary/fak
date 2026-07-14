package metrics

// stale_lock_wedge.go — a GIT-FREE classifier that tells a self-healing commit
// lull apart from a frozen `.git/index.lock` wedge on the shared trunk (#4601).
//
// The 2026-07-13 KPI dispatch run droughted for ~28 min: a crashed/killed worker
// left `.git/index.lock` behind, 8 live workers plus 5 `git.exe` processes queued
// silently on it, and 0 trunk commits landed until the supervisor exited and
// reaped the holder. #3915 already teaches the `fak commit` PATH to auto-reap a
// stale index.lock — but a BLOCKED committer never reaches that path: it waits on
// the lock, so the whole fleet droughts with no committer surfacing the error.
//
// The missing piece is an ACTIVE watchdog in the supervisor/drought loop, and its
// hard constraint is that it must be git-free: during the wedge `git log` /
// `git status` themselves hang (they queue on the same lock, cf #4595), so the
// only trustworthy evidence is a `stat` of the lock file and a read of the
// `.git/logs/HEAD` tail — never a shelled-out git. This fold is that watchdog's
// DECISION core: it consumes two git-free samples plus the live-worker count and
// classifies the lane, deciding whether a lock is SAFE TO CLEAR. The I/O (stat,
// HEAD-log tail parse, worker inventory, and the eventual `rm`) belongs to the
// caller in whatever lane wires it; keeping the decision here as a pure fold is
// what lets the metrics layer own it without an upward import (an import of the
// gateway/engine would red architest with ARCH_LAYER_VIOLATION).
//
// The detection signal that worked by hand during the incident, reproduced here:
// index.lock present AND its mtime frozen across two samples AND `.git/logs/HEAD`
// last-commit epoch frozen across that window AND >= 1 worker active. A benign
// batch lull looks different — a live/churning lock (mtime advancing) with
// commits still landing (HEAD advancing) — and self-heals, so it must never be
// cleared. The verdict carries a DroughtKind so the drought monitor can finally
// distinguish a frozen-lock wedge from an ordinary lull, which today are
// indistinguishable without manual git-free forensics.

import (
	"fmt"
	"time"
)

// StaleLockWedgeSchema versions the verdict record so a consumer (the drought
// monitor, a tick-stream row, a JSON reader) can pin the shape.
const StaleLockWedgeSchema = "fak-stale-lock-wedge/1"

// DefaultStaleLockWedgeAge is how long index.lock must sit with a frozen mtime
// before the watchdog treats it as a wedge CANDIDATE. It is deliberately tighter
// than safecommit.DefaultStaleIndexLockAge (15 min, the commit-path reaper's
// age-alone threshold) because the watchdog does not rely on age alone: it also
// requires a frozen HEAD across two samples and an active fleet, so a shorter
// trigger recovers the fleet faster without the age-only false-positive risk.
// The issue proposed 3–5 min; 5 min is the conservative end of that range.
const DefaultStaleLockWedgeAge = 5 * time.Minute

// WedgeState is the closed vocabulary a lane-classification lands on.
type WedgeState string

const (
	// WedgeClear — no index.lock present. The commit lane is not lock-blocked.
	WedgeClear WedgeState = "CLEAR"
	// WedgeLive — index.lock present with evidence of a live writer: its mtime
	// advanced across the two samples, or it is younger than the staleness
	// threshold. A live git may hold it; never clear. This is the benign,
	// self-healing case (a normal in-flight commit or a churning batch lull).
	WedgeLive WedgeState = "LIVE_LOCK"
	// WedgeUnconfirmed — index.lock present and stale, but the full frozen-wedge
	// signature is not positively confirmed (no second sample yet, mtime/HEAD
	// freeze not proven across two samples, or no active worker). Resample before
	// acting; never clear on an unconfirmed signature.
	WedgeUnconfirmed WedgeState = "WEDGE_UNCONFIRMED"
	// WedgeHeldLive — the full frozen-wedge signature holds, but a RECORDED holder
	// pid is still alive, so a live process owns the lock even though it is not
	// touching the file right now. Do not clear; wait. (Maps to a LOCK-HELD-LIVE
	// event.)
	WedgeHeldLive WedgeState = "LOCK_HELD_LIVE"
	// WedgeFrozen — the full frozen-wedge signature holds and no live holder was
	// found: index.lock frozen across two samples, older than the threshold,
	// `.git/logs/HEAD` frozen across the window, and >= 1 worker active. Safe to
	// clear; the caller may `rm` the lock and emit a LOCK-RECOVERED event.
	WedgeFrozen WedgeState = "FROZEN_WEDGE"
)

// Drought kinds discriminate WHY the fleet is not landing commits, so the drought
// monitor can act (recover a wedge) vs wait (ride out a lull).
const (
	DroughtNone       = "none"
	DroughtBatchLull  = "batch-lull"
	DroughtFrozenLock = "frozen-lock-wedge"
)

// LockSample is one git-free observation of the shared trunk's commit state. All
// epochs are Unix seconds; a zero epoch means "not observed" (an absent lock, or
// an unreadable HEAD-log tail), which the classifier treats as missing evidence
// rather than a real value.
type LockSample struct {
	// SampleUnix is when this sample was taken (Unix seconds).
	SampleUnix int64
	// LockPresent is whether `.git/index.lock` existed at sample time.
	LockPresent bool
	// LockModUnix is the lock file's mtime (Unix seconds); 0 when absent/unknown.
	LockModUnix int64
	// HeadLogUnix is the epoch of the last `.git/logs/HEAD` reflog entry (Unix
	// seconds), i.e. the last-commit time; 0 when the tail could not be read.
	HeadLogUnix int64
}

// WedgeInput is the two-sample window plus the live-fleet context the classifier
// folds into a verdict. Prev is the earlier sample, Curr the later one.
type WedgeInput struct {
	Prev LockSample
	Curr LockSample
	// ActiveWorkers is the count of live fleet workers observed. >= 1 means the
	// fleet is actively trying to commit, which is what turns a frozen lock into a
	// fleet-wide wedge worth auto-recovering.
	ActiveWorkers int
	// HolderKnown reports whether HolderPID is a real recorded holder. git's
	// index.lock records no pid, so this is usually false; it exists so a caller
	// that CAN attribute a holder (e.g. from fak's own lock sidecar) can veto a
	// clear when that holder is provably alive.
	HolderKnown bool
	// HolderPID is the recorded holder pid (meaningful only when HolderKnown).
	HolderPID int
	// HolderAlive is whether HolderPID is still alive (meaningful only when
	// HolderKnown).
	HolderAlive bool
	// StaleAfter is how old the frozen lock must be before it is a wedge
	// candidate; <= 0 disables the staleness gate (nothing is ever a wedge).
	StaleAfter time.Duration
}

// WedgeVerdict is the classifier's structured decision.
type WedgeVerdict struct {
	Schema         string     `json:"schema"`
	State          WedgeState `json:"state"`
	SafeToClear    bool       `json:"safe_to_clear"`
	Drought        bool       `json:"drought"`
	DroughtKind    string     `json:"drought_kind"`
	LockAgeSeconds int64      `json:"lock_age_seconds"`
	LockAdvancing  bool       `json:"lock_advancing"`
	HeadAdvancing  bool       `json:"head_advancing"`
	ActiveWorkers  int        `json:"active_workers"`
	Event          string     `json:"event,omitempty"`
	Reason         string     `json:"reason"`
}

// Structured event tags a caller can emit verbatim to the tick-stream.
const (
	EventLockRecoveredCandidate = "LOCK-RECOVERED-CANDIDATE"
	EventLockHeldLive           = "LOCK-HELD-LIVE"
)

// ClassifyStaleLockWedge folds a two-sample git-free window into a lane verdict.
// It is pure: identical input always yields identical output, with no clock, no
// filesystem, and no git — which is exactly what lets it run safely while real
// git is wedged, and be exercised deterministically in tests.
//
// Safety is asymmetric on purpose. Waiting on a live lock costs at most a resample
// interval; clearing a lock a live git is mid-writing corrupts a peer's index. So
// SafeToClear is set only on POSITIVE evidence of a wedge — the lock mtime proven
// frozen across two present samples, the HEAD reflog proven frozen across the same
// window, the lock older than the threshold, >= 1 worker active, and no recorded
// holder proven alive. Any missing or ambiguous signal lands on a non-clearing
// state.
func ClassifyStaleLockWedge(in WedgeInput) WedgeVerdict {
	prev, curr := in.Prev, in.Curr

	lockAdvancing := curr.LockPresent && prev.LockPresent &&
		curr.LockModUnix > 0 && prev.LockModUnix > 0 &&
		curr.LockModUnix > prev.LockModUnix
	lockFrozen := curr.LockPresent && prev.LockPresent &&
		curr.LockModUnix > 0 && prev.LockModUnix > 0 &&
		curr.LockModUnix == prev.LockModUnix

	headKnown := curr.HeadLogUnix > 0 && prev.HeadLogUnix > 0
	headAdvancing := headKnown && curr.HeadLogUnix > prev.HeadLogUnix
	headFrozen := headKnown && curr.HeadLogUnix == prev.HeadLogUnix

	lockAge := int64(0)
	if curr.LockPresent && curr.LockModUnix > 0 && curr.SampleUnix >= curr.LockModUnix {
		lockAge = curr.SampleUnix - curr.LockModUnix
	}

	// Drought: the fleet is trying to commit but nothing landed across the window.
	drought := in.ActiveWorkers >= 1 && !headAdvancing

	v := WedgeVerdict{
		Schema:         StaleLockWedgeSchema,
		ActiveWorkers:  in.ActiveWorkers,
		LockAgeSeconds: lockAge,
		LockAdvancing:  lockAdvancing,
		HeadAdvancing:  headAdvancing,
		Drought:        drought,
		DroughtKind:    DroughtNone,
	}
	if drought {
		v.DroughtKind = DroughtBatchLull
	}

	// No lock: the lane is not index-lock-blocked. A concurrent drought here is an
	// ordinary lull, not a lock wedge.
	if !curr.LockPresent {
		v.State = WedgeClear
		v.Reason = "no index.lock present; commit lane is not lock-blocked"
		return v
	}

	// A churning lock (mtime advancing) means a live git is writing right now.
	if lockAdvancing {
		v.State = WedgeLive
		v.Reason = "index.lock mtime advancing across samples; a live git is writing — self-healing, do not clear"
		return v
	}

	// Below the staleness threshold: a merely-slow live index write, not a crash.
	stale := in.StaleAfter > 0 && lockAge >= int64(in.StaleAfter/time.Second)
	if !stale {
		v.State = WedgeLive
		v.Reason = "index.lock present but younger than the staleness threshold; may be a live write"
		return v
	}

	// Stale from here, but only a POSITIVELY confirmed signature may be cleared.
	// Require the mtime and the HEAD reflog both proven frozen across two present
	// samples, and an active fleet — otherwise resample.
	if !lockFrozen || !headFrozen || in.ActiveWorkers < 1 {
		v.State = WedgeUnconfirmed
		v.Reason = unconfirmedReason(lockFrozen, headFrozen, headKnown, in.ActiveWorkers, lockAge, in.StaleAfter)
		return v
	}

	// Full frozen-wedge signature holds: the drought is caused by the lock.
	v.DroughtKind = DroughtFrozenLock

	// A recorded holder proven alive vetoes the clear even under a frozen file.
	if in.HolderKnown && in.HolderAlive {
		v.State = WedgeHeldLive
		v.Event = EventLockHeldLive
		v.Reason = fmt.Sprintf("index.lock frozen %ds (>= %s) but recorded holder pid %d is alive; waiting, not clearing",
			lockAge, in.StaleAfter, in.HolderPID)
		return v
	}

	v.State = WedgeFrozen
	v.SafeToClear = true
	v.Event = EventLockRecoveredCandidate
	v.Reason = fmt.Sprintf("index.lock frozen %ds (>= %s), HEAD reflog frozen across window, %d worker(s) active, no live holder — safe to clear",
		lockAge, in.StaleAfter, in.ActiveWorkers)
	return v
}

func unconfirmedReason(lockFrozen, headFrozen, headKnown bool, workers int, lockAge int64, staleAfter time.Duration) string {
	switch {
	case !lockFrozen:
		return fmt.Sprintf("index.lock stale %ds (>= %s) but its frozen mtime is not confirmed across two present samples; resample before clearing",
			lockAge, staleAfter)
	case !headKnown:
		return "index.lock stale but the .git/logs/HEAD tail could not be read in both samples; resample before clearing"
	case !headFrozen:
		return "index.lock stale but HEAD reflog advanced (commits are still landing); a live git holds it — do not clear"
	default: // workers < 1
		return "index.lock stale and frozen but no fleet worker is active; no wedge to recover — leave it for the commit-path reaper"
	}
}

// StaleLockWedgeFragment renders a compact one-line summary of a verdict for the
// drought monitor / tick-stream (#4601 observability): the distinct signal that
// lets an operator tell a frozen-lock wedge from an ordinary batch lull at a
// glance. Pure and deterministic — the same verdict always renders the same
// bytes. Example: "lock=FROZEN_WEDGE age=1560s drought=frozen-lock-wedge workers=8 clear=yes".
func StaleLockWedgeFragment(v WedgeVerdict) string {
	clear := "no"
	if v.SafeToClear {
		clear = "yes"
	}
	kind := v.DroughtKind
	if kind == "" {
		kind = DroughtNone
	}
	return fmt.Sprintf("lock=%s age=%ds drought=%s workers=%d clear=%s",
		v.State, v.LockAgeSeconds, kind, v.ActiveWorkers, clear)
}
