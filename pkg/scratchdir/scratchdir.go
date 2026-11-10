// Package scratchdir creates, removes, and sweeps tool scratch directories so
// a tool's temp trees cannot accumulate on disk.
//
// os.RemoveAll (go1.26.0, Windows) was witnessed returning "unlinkat <dir>: is
// a directory" for every non-empty tree under %LOCALAPPDATA% while deleting
// nothing, although a bottom-up os.Remove walk of the same tree succeeds. Every
// fak-run build export and pruned slot leaked that way (2026-10-12: 532 export
// dirs, 21.7 GB; 192 manifest-less fak-flow slots), so RemoveAll falls back to
// that walk.
package scratchdir

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/procalive"
)

// HolderFile records the creating process inside a scratch dir so a sweep
// never removes a tree a live process still owns.
const HolderFile = ".fak-scratch-holder"

// unparseableHolderAge is how long a holder record that cannot be read keeps
// its dir; reading it as held forever would turn a torn write into a leak.
const unparseableHolderAge = 7 * 24 * time.Hour

var (
	removeAttempts = 4
	removeBackoff  = 50 * time.Millisecond
	sleep          = time.Sleep
	holderAlive    = liveHolder
)

// Create makes a fresh dir under parent (os.MkdirTemp semantics) and records
// this process as its holder.
func Create(parent, pattern string) (string, error) {
	if parent != "" {
		if err := os.MkdirAll(parent, 0o755); err != nil {
			return "", err
		}
	}
	dir, err := os.MkdirTemp(parent, pattern)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, HolderFile), holderRecord(os.Getpid()), 0o644); err != nil {
		_ = RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

func holderRecord(pid int) []byte {
	start := int64(0)
	if t, ok := procalive.StartTime(pid); ok {
		start = t.UnixNano()
	}
	return []byte(fmt.Sprintf("pid=%d start=%d\n", pid, start))
}

// WriteHolder records the calling process as a live holder of dir, writing the
// same record IsHeld and Sweep read. Use it to mark a directory the process
// already owns (no scratch dir is created), e.g. a verify root's in-flight
// materialize marker.
func WriteHolder(dir string) error {
	return os.WriteFile(filepath.Join(dir, HolderFile), holderRecord(os.Getpid()), 0o644)
}

func parseHolder(b []byte) (pid int, start int64, ok bool) {
	for _, field := range strings.Fields(string(b)) {
		k, v, _ := strings.Cut(field, "=")
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0, 0, false
		}
		switch k {
		case "pid":
			pid = int(n)
		case "start":
			start = n
		}
	}
	return pid, start, pid > 0
}

// liveHolder fails closed: a pid that is running reads as the holder unless
// its start time is known and differs (the pid was reused).
func liveHolder(pid int, start int64) bool {
	if !procalive.Alive(pid) {
		return false
	}
	if start == 0 {
		return true
	}
	t, ok := procalive.StartTime(pid)
	if !ok {
		return true
	}
	d := t.UnixNano() - start
	return d > -int64(time.Second) && d < int64(time.Second)
}

// RemoveAll removes path and everything under it; a missing path is success.
func RemoveAll(path string) error {
	err := os.RemoveAll(path)
	for attempt := 0; err != nil && attempt < removeAttempts; attempt++ {
		if attempt > 0 {
			sleep(removeBackoff << uint(attempt-1))
		}
		err = removeBottomUp(path)
	}
	return err
}

func removeBottomUp(root string) error {
	var dirs []string
	var first error
	keep := func(err error) {
		if first == nil && err != nil && !errors.Is(err, fs.ErrNotExist) {
			first = err
		}
	}
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			keep(err)
			if d != nil && d.IsDir() && p != root {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			dirs = append(dirs, p)
			return nil
		}
		keep(removeFile(p))
		return nil
	})
	// WalkDir is pre-order, so reversed every child precedes its parent.
	for i := len(dirs) - 1; i >= 0; i-- {
		keep(removeFile(dirs[i]))
	}
	if _, err := os.Lstat(root); errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if first == nil {
		first = fmt.Errorf("scratchdir: %s still present after removal", root)
	}
	return first
}

func removeFile(p string) error {
	err := os.Remove(p)
	if errors.Is(err, fs.ErrPermission) {
		_ = os.Chmod(p, 0o700)
		err = os.Remove(p)
	}
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return err
}

// Verdict classifies one swept dir.
type Verdict string

const (
	VerdictStale  Verdict = "stale"
	VerdictHeld   Verdict = "held"
	VerdictYoung  Verdict = "young"
	VerdictFailed Verdict = "remove_failed"
)

// Entry is one dir a sweep looked at.
type Entry struct {
	Path    string  `json:"path"`
	Verdict Verdict `json:"verdict"`
	AgeSec  int64   `json:"age_sec"`
	Bytes   int64   `json:"bytes,omitempty"`
	Removed bool    `json:"removed"`
	Error   string  `json:"error,omitempty"`
}

// Report is a sweep receipt. StaleBytes is only filled when Measure is set.
type Report struct {
	Parent       string  `json:"parent"`
	Prefix       string  `json:"prefix"`
	MinAgeSec    int64   `json:"min_age_sec"`
	Applied      bool    `json:"applied"`
	Entries      []Entry `json:"entries"`
	Stale        int     `json:"stale"`
	Held         int     `json:"held"`
	Young        int     `json:"young"`
	Removed      int     `json:"removed"`
	Failed       int     `json:"failed"`
	StaleBytes   int64   `json:"stale_bytes"`
	RemovedBytes int64   `json:"removed_bytes"`
}

// SweepOptions selects the direct children of Parent whose name starts with
// Prefix (empty matches all). A dir is stale when it is at least MinAge old
// and no live process holds it.
type SweepOptions struct {
	Parent  string
	Prefix  string
	MinAge  time.Duration
	Apply   bool
	Measure bool
	Now     time.Time
	// BeforeRemove runs on a stale dir just before removal, e.g. to stop a
	// daemon that keeps a handle open inside it.
	BeforeRemove func(path string)
}

// Sweep classifies and, with Apply, removes stale scratch dirs. A dir with a
// live holder is never removed, whatever its age.
func Sweep(o SweepOptions) Report {
	now := o.Now
	if now.IsZero() {
		now = time.Now()
	}
	rep := Report{Parent: o.Parent, Prefix: o.Prefix, MinAgeSec: int64(o.MinAge / time.Second), Applied: o.Apply}
	entries, err := os.ReadDir(o.Parent)
	if err != nil {
		return rep
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, de := range entries {
		if !de.IsDir() || !strings.HasPrefix(de.Name(), o.Prefix) {
			continue
		}
		path := filepath.Join(o.Parent, de.Name())
		info, err := de.Info()
		if err != nil {
			continue
		}
		age := now.Sub(info.ModTime())
		e := Entry{Path: path, AgeSec: int64(age / time.Second), Verdict: classify(path, age, o.MinAge)}
		switch e.Verdict {
		case VerdictHeld:
			rep.Held++
		case VerdictYoung:
			rep.Young++
		case VerdictStale:
			rep.Stale++
			if o.Measure {
				e.Bytes = TreeBytes(path)
				rep.StaleBytes += e.Bytes
			}
			if o.Apply {
				if o.BeforeRemove != nil {
					o.BeforeRemove(path)
				}
				if info, err := os.Lstat(path); err != nil || !info.IsDir() || IsHeld(path, now.Sub(info.ModTime())) {
					e.Verdict = VerdictHeld
					rep.Stale--
					rep.StaleBytes -= e.Bytes
					rep.Held++
					rep.Entries = append(rep.Entries, e)
					continue
				}
				if err := RemoveAll(path); err != nil {
					e.Verdict, e.Error = VerdictFailed, err.Error()
					rep.Failed++
				} else {
					e.Removed = true
					rep.Removed++
					rep.RemovedBytes += e.Bytes
				}
			}
		}
		rep.Entries = append(rep.Entries, e)
	}
	return rep
}

func classify(path string, age, minAge time.Duration) Verdict {
	if IsHeld(path, age) {
		return VerdictHeld
	}
	if age < minAge {
		return VerdictYoung
	}
	return VerdictStale
}

// IsHeld reports whether a scratch directory's holder prevents removal.
// Unreadable records and malformed records younger than seven days retain it.
// Age is the directory age, as used by Sweep.
func IsHeld(path string, age time.Duration) bool {
	b, err := os.ReadFile(filepath.Join(path, HolderFile))
	switch {
	case err == nil:
		pid, start, ok := parseHolder(b)
		if !ok {
			if age < unparseableHolderAge {
				return true
			}
		} else if holderAlive(pid, start) {
			return true
		}
	case !errors.Is(err, fs.ErrNotExist):
		return true
	}
	return false
}

// TreeBytes is the apparent size of the regular files under root.
func TreeBytes(root string) int64 {
	var n int64
	_ = filepath.WalkDir(root, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}
