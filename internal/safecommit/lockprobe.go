package safecommit

import (
	"errors"
	"io/fs"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Reason tokens stamped on a broken lock's LOCK_BROKEN event (issue #2339). The set
// is closed: a break is either a dead recorded holder, or a live PID whose process is
// provably not the one that took the lock (a reused PID number now owned by something
// unrelated). An empty Reason means the lock was NOT reapable.
const (
	ReapReasonHolderDead = "holder_dead" // recorded PID is no longer a running process
	// ReapReasonHolderForeign covers both proofs of PID reuse, because from the lane's
	// point of view they are the same fact: the PID is alive and it is not the holder.
	// Either the process started after the lock was written (#5892), or its image is not a
	// fak/git committer (#2339). The event carries which one decided it.
	ReapReasonHolderForeign = "holder_foreign"
)

// Reason tokens classifying WHY an attempted reap's os.Remove was refused by the OS.
// The set is closed so a log consumer can group repeated failures without parsing
// localized OS text (the Windows messages this class produces are localized; the
// numeric codes behind them are not).
const (
	ReapRemoveErrPermission = "permission_denied" // ERROR_ACCESS_DENIED / EPERM / EACCES
	ReapRemoveErrBusy       = "file_in_use"       // ERROR_SHARING_VIOLATION / ERROR_LOCK_VIOLATION / EBUSY
	ReapRemoveErrNotExist   = "not_exist"         // raced away between the probe and the remove
	ReapRemoveErrOther      = "other"             // any other failure; read RemoveErr for the text
)

// Win32 error codes returned when a still-open handle prevents a delete. Named here
// because they are the two most likely causes of the exact wedge this observability
// exists to explain, and because Go surfaces them as bare syscall.Errno values with no
// portable constant.
const (
	winErrSharingViolation = 32 // ERROR_SHARING_VIOLATION
	winErrLockViolation    = 33 // ERROR_LOCK_VIOLATION
)

// lockRemove is os.Remove, injectable so the reap-FAILURE path is unit-testable. A real
// remove refusal needs another process to hold an un-shareable handle on the lockfile,
// which a table-driven test cannot manufacture portably; the seam lets the failure
// classification and the surfaced error be proven without one.
var lockRemove = os.Remove

// classifyRemoveErr maps an os.Remove failure to a closed ReapRemoveErr* token. It
// checks the Windows sharing/lock-violation codes FIRST because those are the ones the
// wedge produces and they do not map onto fs.ErrPermission, then falls back to the
// portable fs sentinels.
func classifyRemoveErr(err error) string {
	if err == nil {
		return ""
	}
	if runtime.GOOS == "windows" {
		var errno syscall.Errno
		if errors.As(err, &errno) {
			switch uintptr(errno) {
			case winErrSharingViolation, winErrLockViolation:
				return ReapRemoveErrBusy
			}
		}
	}
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return ReapRemoveErrNotExist
	case errors.Is(err, fs.ErrPermission):
		return ReapRemoveErrPermission
	case errors.Is(err, syscall.EBUSY):
		return ReapRemoveErrBusy
	}
	return ReapRemoveErrOther
}

// LockProbe is the result of inspecting an advisory lockfile without mutating it.
// It reports the recorded holder PID and whether that holder is still a live process,
// so an operator tool can DIAGNOSE a wedged lock (a dead PID still owning the file)
// before deciding to reap it.
type LockProbe struct {
	Path       string    // the lockfile path probed
	Exists     bool      // the file is present and readable
	HolderPID  int       // the PID recorded in the file (0 if absent/unparseable)
	Alive      bool      // the recorded holder is a currently-running process
	Stale      bool      // Exists && HolderPID>0 && !Alive — a DEAD holder still owns the file
	Foreign    bool      // Exists && HolderPID>0 && Alive but the process is provably not the holder (PID reuse)
	Image      string    // the recorded holder's process image base name, when readable ("" otherwise)
	ModTime    time.Time // the lockfile's last-modified time (zero when absent/unreadable)
	AgeSeconds int64     // whole seconds since ModTime at probe time (0 when absent/unreadable)
	Reason     string    // ReapReason* naming WHY the lock is reapable, or "" when it is not

	// StartedAt is when the process now at HolderPID started, when that is readable (zero
	// otherwise). StartedAfterLock is the proof derived from it: the process started after
	// this lockfile was written, so whatever it is now, it cannot be the process that wrote
	// it. See the startTimeSkewGrace comment for why that is the stronger discriminator.
	StartedAt        time.Time
	StartedAfterLock bool
}

// Reapable reports whether the probed lock may be broken: a dead recorded holder, OR a
// live PID that is provably not the process that took the lock — one that started after
// the lock was written, or whose image is provably foreign (both are reused PIDs). A live
// committer, an absent file, and an unattributable/unidentifiable file are all NOT
// reapable — the fail-safe stance that never breaks a lock a live committer holds.
func (p LockProbe) Reapable() bool { return p.Stale || p.Foreign }

// processImageNameFn resolves a running PID's process image base name (lowercased,
// without a trailing ".exe"), reporting ok=false when the image cannot be read. It is
// a package var so the PID-reuse image guard is unit-testable without spawning a real
// foreign process. Default is the platform implementation in alive_image_*.go.
var processImageNameFn = processImageName

// processStartTimeFn resolves when the process at a live PID started, reporting ok=false
// when that cannot be read. Package var for the same reason processImageNameFn is one: the
// PID-reuse guard has to be provable in a unit test without arranging a real recycled PID.
// Default is the platform implementation in starttime_*.go.
var processStartTimeFn = processStartTime

// startTimeSkewGrace is how much LATER than the lockfile's own mtime the process at the
// recorded PID may claim to have started before the probe calls it a reused PID (#5892).
//
// The comparison works because gpulease writes the holder's PID into the lockfile at the
// instant it wins the flock and nothing touches the file again for the rest of the hold, so
// the mtime IS the moment the lock was recorded. A process cannot predate its own writes,
// so a holder that genuinely took this lock always started BEFORE the mtime; only a PID
// recycled onto a later process can start after it. That makes this a proof rather than the
// guess an image name can offer — and, unlike a start id persisted into the record, it
// works on lockfiles already sitting on disk instead of only on ones written from now on.
//
// The grace is pure fail-safe margin for the two clocks not being the same instrument: a
// FILETIME and an NTFS mtime share the system clock, but the procfs derivation goes through
// a whole-second btime plus USER_HZ ticks, so it can land up to about a second out. A real
// reuse wedge is minutes to hours old, so spending seconds here costs nothing and keeps the
// one error this guard may never make — breaking a live committer's lock — out of reach.
const startTimeSkewGrace = 5 * time.Second

// nowFn is time.Now, injectable so age computation is deterministic in tests.
var nowFn = time.Now

// committerImageTokens are the substrings that mark a live PID's image as a plausible
// fak committer, so the PID-reuse guard NEVER breaks a live holder it cannot rule out.
// The list is deliberately BROAD (a false "foreign" would break a live committer's
// lock — the one outcome issue #2339's acceptance forbids): the fak binary itself, a
// `go run`/`go test` harness, the interpreters a fak session launches under, and the
// git subprocess it drives. A reused PID owned by an image matching none of these is
// treated as foreign; an image we cannot read at all is treated as committer-like
// (not reaped), the safe direction.
var committerImageTokens = []string{
	"fak", "git", "go", "dlv", ".test",
	"claude", "node", "pwsh", "powershell", "cmd", "bash", "sh", "zsh",
}

// looksLikeCommitterImage reports whether an image base name plausibly belongs to a
// fak committer (or its launcher/interpreter). An empty/unreadable image is treated
// as committer-like so an unidentifiable live holder is never reaped.
func looksLikeCommitterImage(image string) bool {
	image = strings.ToLower(strings.TrimSpace(image))
	if image == "" {
		return true // cannot identify => do not treat as foreign, do not reap a live holder
	}
	for _, tok := range committerImageTokens {
		if strings.Contains(image, tok) {
			return true
		}
	}
	return false
}

// ProcessAlive reports whether a process with the given pid is currently running. It is
// the exported form of the same liveness check the commit-lock reap uses (Windows:
// OpenProcess + GetExitCodeProcess; unix: Kill(pid,0)). Exposed so sibling tooling — e.g.
// treedoctor — can reuse one audited implementation instead of copying it.
func ProcessAlive(pid int) bool { return processAlive(pid) }

// ProbeLock inspects the lockfile at path WITHOUT modifying it and reports whether it is
// reapable (a dead holder, or a live-but-foreign reused PID). It never deletes anything —
// callers decide. A missing/unreadable file or an unparseable PID yields a non-reapable
// probe, matching the reaper's fail-safe stance: only a provably-dead holder or a
// provably-foreign live image is reapable.
func ProbeLock(path string) LockProbe {
	p := LockProbe{Path: path}
	data, err := os.ReadFile(path)
	if err != nil {
		return p // absent/unreadable => Exists=false, not reapable
	}
	p.Exists = true
	if fi, err := os.Stat(path); err == nil {
		p.ModTime = fi.ModTime()
		if age := nowFn().Sub(p.ModTime); age > 0 {
			p.AgeSeconds = int64(age / time.Second)
		}
	}
	pid := parseHolderPID(data)
	if pid <= 0 {
		return p // no parseable holder => not attributable, not reapable
	}
	p.HolderPID = pid
	p.Alive = processAlive(pid)
	if !p.Alive {
		p.Stale = true
		p.Reason = ReapReasonHolderDead
		return p
	}
	// The holder PID is alive. It is only reapable if the running process is provably NOT
	// the committer that took this lock — a reused PID number now owned by something
	// unrelated. Two independent discriminators can establish that, and each is consulted
	// in full: they fail in different directions, so neither may short-circuit the other.
	//
	// First, identity by start time. A process that started after this lockfile was written
	// cannot be the process that wrote it, whatever it is now called — the proof the image
	// name cannot supply. On a fleet host this is the discriminator that matters, because
	// the image allowlist below IS the ambient process population there (pwsh, node, claude,
	// git), so a recycled PID reads as committer-like and the lane wedges (#5892).
	if started, ok := processStartTimeFn(pid); ok {
		p.StartedAt = started
		if !p.ModTime.IsZero() && started.After(p.ModTime.Add(startTimeSkewGrace)) {
			p.StartedAfterLock = true
			p.Foreign = true
			p.Reason = ReapReasonHolderForeign
		}
	}
	// Second, the original image heuristic (#2339). It still runs when the start time
	// proved nothing — an unreadable start time, a platform without one, or a lockfile
	// whose mtime we could not stat — so no lock that was reapable before this guard
	// existed stops being reapable now. The image is read either way, because a break
	// justified by start time still wants it on the LOCK_BROKEN event as evidence.
	//
	// An image we cannot read is treated as committer-like, so a live holder is never
	// broken on a failed image read.
	if image, ok := processImageNameFn(pid); ok {
		p.Image = image
		if !looksLikeCommitterImage(image) {
			p.Foreign = true
			p.Reason = ReapReasonHolderForeign
		}
	}
	return p
}

// ReapResult is the structured outcome of a reap attempt — the payload behind a
// LOCK_BROKEN event (issue #2339): what was broken, whose PID held it, how old the
// lock was, and the closed Reason that justified the break.
type ReapResult struct {
	Reaped     bool
	Path       string
	HolderPID  int
	AgeSeconds int64
	Reason     string // ReapReason* naming why the lock was judged breakable; "" when it was not reapable
	Image      string // the recorded holder PID's process image at probe time, else ""

	// StartedAfterLock records that a holder_foreign break was justified by process START
	// TIME rather than by the image name — the process at the recorded PID began after the
	// lockfile was written. It matters on the event: the image that accompanies such a
	// break is typically an ordinary committer-like name (pwsh, node), so reporting it
	// alone would read as if THAT name were the evidence, when the name is exactly what
	// could not decide the question.
	StartedAfterLock bool

	// Attempted records that the probe judged the lock REAPABLE and a remove was actually
	// issued. It splits the two very different states that Reaped=false used to collapse
	// into one: "we never tried, a live committer holds it" (Attempted=false — the
	// fail-safe, expected outcome) and "we tried and the OS refused" (Attempted=true — a
	// wedge that will not clear on its own).
	Attempted bool
	// RemoveErr is the os.Remove failure text when Attempted && !Reaped, else "".
	//
	// Discarding this error is the observability defect behind #5335: a .git/fak-commit.lock
	// holding a dead PID wedged the whole fleet's commit lane for 85 minutes while every
	// queued committer's acquireWithReap re-reaped it every 250ms for the whole of its 10s
	// bounded wait — so a reaper ran on every one of those attempts, its remove was failing
	// every time, and not one byte of evidence was recorded. `fak commit status --json`
	// correctly reported verdict=stale throughout (commitlane.finalize keys that verdict off
	// exactly this dead-holder lock), so DETECTION was never the gap; the missing errno was.
	RemoveErr string
	// RemoveErrClass is the closed ReapRemoveErr* token for RemoveErr, so repeated failures
	// can be grouped without parsing OS text.
	RemoveErrClass string
}

// Failed reports whether a reap was ATTEMPTED (the lock was judged breakable) and the
// removal was nonetheless refused. This is the state that must never be silent: the lock
// is provably not held by a live committer, yet it survives, so nothing downstream will
// clear it and the lane stays wedged until a human intervenes.
func (r ReapResult) Failed() bool { return r.Attempted && !r.Reaped }

// ReapStaleLockResult removes the lockfile at path IFF the probe says it is reapable
// (a dead holder, or a live-but-foreign reused PID), and returns the structured outcome
// so the caller can emit a LOCK_BROKEN event. It is fail-safe: a live committer, an
// absent file, or an unattributable file are all left untouched and yield Reaped=false.
//
// A remove that the OS REFUSES is still fail-safe — Acquire's bounded wait remains the
// backstop and no lock is corrupted — but it is no longer silent: the error is carried
// out on RemoveErr/RemoveErrClass with Attempted=true, so the caller can say WHY a
// provably-breakable lock survived instead of reporting an indistinguishable
// Reaped=false. Reap POLICY is unchanged: exactly the same locks are broken as before.
func ReapStaleLockResult(path string) ReapResult {
	p := ProbeLock(path)
	res := ReapResult{
		Path:             path,
		HolderPID:        p.HolderPID,
		AgeSeconds:       p.AgeSeconds,
		Image:            p.Image,
		StartedAfterLock: p.StartedAfterLock,
	}
	if !p.Reapable() {
		return res
	}
	// Reason is set here, before the remove, because it explains the DECISION to break the
	// lock — which the probe has already made — not the outcome of the syscall. A failed
	// remove needs it just as much as a successful one does.
	res.Reason = p.Reason
	res.Attempted = true
	if err := lockRemove(path); err != nil {
		res.RemoveErr = err.Error()
		res.RemoveErrClass = classifyRemoveErr(err)
		return res // remove refused => report not-reaped WITH the reason; Acquire's bounded wait is the backstop
	}
	res.Reaped = true
	return res
}

// ReapStaleLock removes the lockfile at path IFF it is reapable, and reports whether it
// removed anything. It is the boolean form of ReapStaleLockResult — the in-code
// equivalent of the manual `rm .git/fak-commit.lock` that unblocked a 56-minute commit
// wedge, PID-guarded so it is safe to run blind: a live holder, an absent file, or an
// unattributable file are all left untouched.
func ReapStaleLock(path string) (reaped bool) {
	return ReapStaleLockResult(path).Reaped
}

// parseHolderPID extracts the numeric holder PID from a lockfile body (first line only,
// matching gpulease's record format), or 0 when absent/unparseable.
func parseHolderPID(data []byte) int {
	s := string(data)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	pid, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || pid <= 0 {
		return 0
	}
	return pid
}
