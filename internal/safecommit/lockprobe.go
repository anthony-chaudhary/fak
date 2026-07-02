package safecommit

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// Reason tokens stamped on a broken lock's LOCK_BROKEN event (issue #2339). The set
// is closed: a break is either a dead recorded holder, or a live PID whose process
// image is provably not a committer (a reused PID number now owned by something
// unrelated). An empty Reason means the lock was NOT reapable.
const (
	ReapReasonHolderDead    = "holder_dead"    // recorded PID is no longer a running process
	ReapReasonHolderForeign = "holder_foreign" // PID is alive but its image is not a fak/git committer (PID reuse)
)

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
	Foreign    bool      // Exists && HolderPID>0 && Alive but the image is provably not a committer (PID reuse)
	Image      string    // the recorded holder's process image base name, when readable ("" otherwise)
	ModTime    time.Time // the lockfile's last-modified time (zero when absent/unreadable)
	AgeSeconds int64     // whole seconds since ModTime at probe time (0 when absent/unreadable)
	Reason     string    // ReapReason* naming WHY the lock is reapable, or "" when it is not
}

// Reapable reports whether the probed lock may be broken: a dead recorded holder, OR
// a live PID whose image is provably foreign (a reused PID). A live committer, an
// absent file, and an unattributable/unidentifiable file are all NOT reapable — the
// fail-safe stance that never breaks a lock a live committer holds.
func (p LockProbe) Reapable() bool { return p.Stale || p.Foreign }

// processImageNameFn resolves a running PID's process image base name (lowercased,
// without a trailing ".exe"), reporting ok=false when the image cannot be read. It is
// a package var so the PID-reuse image guard is unit-testable without spawning a real
// foreign process. Default is the platform implementation in alive_image_*.go.
var processImageNameFn = processImageName

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
	// The holder PID is alive. It is only reapable if the running process is provably
	// NOT a committer — a reused PID number now owned by something unrelated. An image
	// we cannot read is treated as committer-like, so a live holder is never broken on
	// a failed image read.
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
	Reason     string // ReapReason* when Reaped, else ""
	Image      string // the foreign image that justified a holder_foreign break, else ""
}

// ReapStaleLockResult removes the lockfile at path IFF the probe says it is reapable
// (a dead holder, or a live-but-foreign reused PID), and returns the structured outcome
// so the caller can emit a LOCK_BROKEN event. It is fail-safe: a live committer, an
// absent file, or an unattributable file are all left untouched and yield Reaped=false.
func ReapStaleLockResult(path string) ReapResult {
	p := ProbeLock(path)
	res := ReapResult{Path: path, HolderPID: p.HolderPID, AgeSeconds: p.AgeSeconds, Image: p.Image}
	if !p.Reapable() {
		return res
	}
	if err := os.Remove(path); err != nil {
		return res // remove failed => report not-reaped; Acquire's bounded wait is the backstop
	}
	res.Reaped = true
	res.Reason = p.Reason
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
