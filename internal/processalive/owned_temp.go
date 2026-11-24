package processalive

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ownedTempMarker separates an OwnedTempDir prefix from the owning pid, so a
// reaper can tell whose dir it is: <prefix>pid<PID>-<random>.
const ownedTempMarker = "pid"

// LegacyTempAge is how old a pid-less entry with an owned-temp prefix must be
// before ReapDeadOwnerTemp removes it. It exceeds any realistic run, so an
// entry this old was left by a process that died without cleanup.
const LegacyTempAge = 2 * time.Hour

// OwnedTempDir creates a directory under os.TempDir() whose name records this
// process as its owner, after reaping same-prefix siblings whose owner is dead.
// The caller still removes its own dir; a process killed before that leaks it
// only until the next process using the prefix starts.
func OwnedTempDir(prefix string) (string, error) {
	ReapDeadOwnerTemp(prefix)
	return os.MkdirTemp("", prefix+ownedTempMarker+strconv.Itoa(os.Getpid())+"-*")
}

// ReapDeadOwnerTemp removes entries directly under os.TempDir() named
// <prefix>pid<PID>-... whose PID is not running, and pid-less entries with the
// prefix older than LegacyTempAge. A live owner's dir is never touched; a reused
// pid only delays the reap. It returns how many entries it removed.
func ReapDeadOwnerTemp(prefix string) int {
	if prefix == "" {
		return 0
	}
	root := os.TempDir()
	entries, err := os.ReadDir(root)
	if err != nil {
		return 0
	}
	cutoff := time.Now().Add(-LegacyTempAge)
	self := os.Getpid()
	removed := 0
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		if pid, ok := OwnedTempOwner(name, prefix); ok {
			if pid == self || Check(pid) {
				continue
			}
		} else if info, err := e.Info(); err != nil || !info.ModTime().Before(cutoff) {
			continue
		}
		if os.RemoveAll(filepath.Join(root, name)) == nil {
			removed++
		}
	}
	return removed
}

// OwnedTempOwner parses the owner pid from a <prefix>pid<PID>-... name.
func OwnedTempOwner(name, prefix string) (int, bool) {
	rest, ok := strings.CutPrefix(name, prefix+ownedTempMarker)
	if !ok {
		return 0, false
	}
	digits, _, ok := strings.Cut(rest, "-")
	if !ok || digits == "" {
		return 0, false
	}
	pid, err := strconv.Atoi(digits)
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}
