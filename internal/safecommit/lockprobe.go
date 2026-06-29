package safecommit

import (
	"os"
	"strconv"
	"strings"
)

// LockProbe is the result of inspecting an advisory lockfile without mutating it.
// It reports the recorded holder PID and whether that holder is still a live process,
// so an operator tool can DIAGNOSE a wedged lock (a dead PID still owning the file)
// before deciding to reap it.
type LockProbe struct {
	Path      string // the lockfile path probed
	Exists    bool   // the file is present and readable
	HolderPID int    // the PID recorded in the file (0 if absent/unparseable)
	Alive     bool   // the recorded holder is a currently-running process
	Stale     bool   // Exists && HolderPID>0 && !Alive — safe to reap
}

// ProcessAlive reports whether a process with the given pid is currently running. It is
// the exported form of the same liveness check the commit-lock reap uses (Windows:
// OpenProcess + GetExitCodeProcess; unix: Kill(pid,0)). Exposed so sibling tooling — e.g.
// treedoctor — can reuse one audited implementation instead of copying it.
func ProcessAlive(pid int) bool { return processAlive(pid) }

// ProbeLock inspects the lockfile at path WITHOUT modifying it and reports whether it is
// stale (a dead holder still owns it). It never deletes anything — callers decide. A
// missing/unreadable file or an unparseable PID yields a non-stale probe (Stale=false),
// matching reapStaleLock's fail-safe stance: only a provably-dead numeric holder is stale.
func ProbeLock(path string) LockProbe {
	p := LockProbe{Path: path}
	data, err := os.ReadFile(path)
	if err != nil {
		return p // absent/unreadable => Exists=false, not stale
	}
	p.Exists = true
	pid := parseHolderPID(data)
	if pid <= 0 {
		return p // no parseable holder => not attributable, not stale
	}
	p.HolderPID = pid
	p.Alive = processAlive(pid)
	p.Stale = !p.Alive
	return p
}

// ReapStaleLock removes the lockfile at path IFF its recorded holder PID is dead, and
// reports whether it removed anything. It is the exported, return-valued form of the
// internal reapStaleLock — the in-code equivalent of the manual `rm .git/fak-commit.lock`
// that unblocked a 56-minute commit wedge, PID-guarded so it is safe to run blind: a
// live holder, an absent file, or an unattributable file are all left untouched.
func ReapStaleLock(path string) (reaped bool) {
	p := ProbeLock(path)
	if !p.Stale {
		return false
	}
	if err := os.Remove(path); err != nil {
		return false
	}
	return true
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
