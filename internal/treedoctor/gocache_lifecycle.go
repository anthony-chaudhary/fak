package treedoctor

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	DefaultGoCacheHighBytes  = int64(32 << 30)
	DefaultGoCacheLowBytes   = int64(24 << 30)
	DefaultGoCacheMinAge     = 7 * 24 * time.Hour
	DefaultGoCacheMinFree    = int64(64 << 30)
	DefaultGoCacheMaxEntries = 200000
	DefaultGoCacheDeadline   = 5 * time.Second
)

func GoCacheRootFromEnv(lookup func(string) string, userCacheDir func() (string, error)) string {
	if v := strings.TrimSpace(lookup("GOCACHE")); v != "" {
		if strings.EqualFold(v, "off") {
			return ""
		}
		return filepath.Clean(v)
	}
	r, e := userCacheDir()
	if e != nil || strings.TrimSpace(r) == "" {
		return ""
	}
	return filepath.Join(r, "go-build")
}

type GoCacheProgress struct {
	Entries int    `json:"entries"`
	Bytes   int64  `json:"bytes"`
	Path    string `json:"path,omitempty"`
}
type GoCacheOptions struct {
	Root                string
	Now                 time.Time
	HighBytes, LowBytes int64
	MinAge              time.Duration
	MinFreeBytes        int64
	FreeBytes           int64
	FreeBytesKnown      bool
	FreeBytesFunc       func(string) (int64, error)
	ActiveBuild         func() (bool, error)
	RemoveAll           func(string) error
	Context             context.Context
	Deadline            time.Duration
	MaxWalkEntries      int
	Progress            func(GoCacheProgress) error
}
type GoCacheEntry struct {
	Path    string    `json:"path"`
	Bytes   int64     `json:"bytes"`
	ModTime time.Time `json:"mod_time"`
}
type GoCacheReport struct {
	Root                  string   `json:"root,omitempty"`
	BytesBefore           int64    `json:"bytes_before,omitempty"`
	BytesAfter            int64    `json:"bytes_after,omitempty"`
	BytesAfterSemantics   string   `json:"bytes_after_semantics,omitempty"`
	FreeBytes             int64    `json:"free_bytes,omitempty"`
	TriggeredBy           []string `json:"triggered_by,omitempty"`
	Candidates            []string `json:"candidates,omitempty"`
	Reaped                []string `json:"reaped,omitempty"`
	ReclaimedBytes        int64    `json:"reclaimed_bytes,omitempty"`
	CandidateBytes        int64    `json:"candidate_bytes,omitempty"`
	CandidateBytesKnown   int      `json:"candidate_bytes_known,omitempty"`
	CandidateBytesUnknown int      `json:"candidate_bytes_unknown,omitempty"`
	ScanEntries           int      `json:"scan_entries,omitempty"`
	ScanComplete          bool     `json:"scan_complete"`
	IncompleteReason      string   `json:"incomplete_reason,omitempty"`
	TargetShortfallBytes  int64    `json:"target_shortfall_bytes,omitempty"`
	CleanupHints          []string `json:"cleanup_hints,omitempty"`
	Skipped               string   `json:"skipped,omitempty"`
	Err                   string   `json:"err,omitempty"`
}

func (r GoCacheReport) Summary() string {
	if r.Root == "" {
		return "Go build cache: disabled"
	}
	if r.Err != "" {
		return "Go build cache: error: " + r.Err
	}
	s := fmt.Sprintf("Go build cache: %d -> %d bytes (%s)", r.BytesBefore, r.BytesAfter, r.BytesAfterSemantics)
	if r.Skipped != "" {
		s += "; skipped: " + r.Skipped
	}
	if !r.ScanComplete {
		s += "; incomplete: " + r.IncompleteReason
	}
	return s
}

func SweepGoCache(o GoCacheOptions, apply bool) GoCacheReport {
	r := GoCacheReport{Root: filepath.Clean(o.Root), ScanComplete: true, CleanupHints: []string{"use tree-doctor GOTMP reaper for orphaned go-build WORK dirs", "use git-daily --prune-worktrees for stale worktrees"}}
	if strings.TrimSpace(o.Root) == "" {
		r.Root = ""
		r.Skipped = "disabled"
		return r
	}
	root, resolved, err := validateGoCacheRoot(o.Root)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.Root = root
	if o.Now.IsZero() {
		o.Now = time.Now()
	}
	if o.HighBytes <= 0 {
		o.HighBytes = DefaultGoCacheHighBytes
	}
	if o.LowBytes <= 0 {
		o.LowBytes = DefaultGoCacheLowBytes
	}
	if o.MinAge <= 0 {
		o.MinAge = DefaultGoCacheMinAge
	}
	if o.MinFreeBytes <= 0 {
		o.MinFreeBytes = DefaultGoCacheMinFree
	}
	if o.MaxWalkEntries <= 0 {
		o.MaxWalkEntries = DefaultGoCacheMaxEntries
	}
	if o.Deadline <= 0 {
		o.Deadline = DefaultGoCacheDeadline
	}
	if o.Context == nil {
		o.Context = context.Background()
	}
	if o.RemoveAll == nil {
		o.RemoveAll = os.RemoveAll
	}
	if !apply {
		o.RemoveAll = os.RemoveAll
		r.BytesAfterSemantics = "projected"
	} else {
		r.BytesAfterSemantics = "actual"
	}
	var unlock func()
	if apply {
		unlock, err = acquireGoCacheLock(resolved)
		if err != nil {
			r.Skipped = "lifecycle lock busy"
			r.Err = err.Error()
			return r
		}
		defer unlock()
		if o.ActiveBuild == nil {
			r.Skipped = "active-build state unavailable"
			return r
		}
		active, witnessErr := o.ActiveBuild()
		if witnessErr != nil {
			r.Err = "active-build witness: " + witnessErr.Error()
			r.Skipped = "build state unknown"
			return r
		}
		if active {
			r.Skipped = "active build"
			return r
		}
	}
	if o.FreeBytesKnown {
		r.FreeBytes = o.FreeBytes
	} else {
		fn := o.FreeBytesFunc
		if fn == nil {
			fn = GoCacheFreeBytes
		}
		if n, e := fn(resolved); e == nil {
			r.FreeBytes = n
			o.FreeBytesKnown = true
		}
	}
	ctx, cancel := context.WithTimeout(o.Context, o.Deadline)
	defer cancel()
	entries, bytes, seen, complete, reason, e := scanGoCache(ctx, resolved, o.MaxWalkEntries, o.Progress)
	r.BytesBefore = bytes
	r.BytesAfter = bytes
	r.ScanEntries = seen
	r.ScanComplete = complete
	r.IncompleteReason = reason
	if e != nil {
		r.Err = e.Error()
		return r
	}
	if !complete {
		r.Skipped = "incomplete census"
		return r
	}
	if bytes > o.HighBytes {
		r.TriggeredBy = append(r.TriggeredBy, "size")
	}
	if o.FreeBytesKnown && r.FreeBytes < o.MinFreeBytes {
		r.TriggeredBy = append(r.TriggeredBy, "pressure")
	}
	if len(r.TriggeredBy) == 0 {
		r.Skipped = "below high-water marks"
		return r
	}
	cutoff := o.Now.Add(-o.MinAge)
	sort.Slice(entries, func(i, j int) bool { return entries[i].ModTime.Before(entries[j].ModTime) })
	target := bytes - o.LowBytes
	if target < 0 {
		target = 0
	}
	if o.FreeBytesKnown && r.FreeBytes < o.MinFreeBytes {
		if pressureTarget := o.MinFreeBytes - r.FreeBytes; pressureTarget > target {
			target = pressureTarget
		}
	}
	var chosen []GoCacheEntry
	for _, e := range entries {
		if e.ModTime.After(cutoff) {
			continue
		}
		chosen = append(chosen, e)
		r.Candidates = append(r.Candidates, e.Path)
		r.CandidateBytes += e.Bytes
		r.CandidateBytesKnown++
		if r.CandidateBytes >= target {
			break
		}
	}
	if r.CandidateBytes < target {
		r.TargetShortfallBytes = target - r.CandidateBytes
	}
	if !apply {
		r.ReclaimedBytes = r.CandidateBytes
		r.BytesAfter = bytes - r.CandidateBytes
		return r
	}
	if len(chosen) == 0 {
		r.Skipped = "no stale candidates"
		return r
	}
	if o.ActiveBuild == nil {
		r.Skipped = "active-build state unavailable"
		return r
	}
	active, e := o.ActiveBuild()
	if e != nil {
		r.Err = "active-build witness: " + e.Error()
		r.Skipped = "build state unknown"
		return r
	}
	if active {
		r.Skipped = "active build"
		return r
	}
	for _, e := range chosen {
		if err := safeCandidate(resolved, e.Path); err != nil {
			r.Err = err.Error()
			return r
		}
		if err := o.RemoveAll(e.Path); err != nil {
			r.Err = err.Error()
			return r
		}
		r.Reaped = append(r.Reaped, e.Path)
		r.ReclaimedBytes += e.Bytes
	}
	r.BytesAfter = bytes - r.ReclaimedBytes
	return r
}

func validateGoCacheRoot(root string) (string, string, error) {
	a, e := filepath.Abs(root)
	if e != nil {
		return "", "", e
	}
	a = filepath.Clean(a)
	base := strings.ToLower(filepath.Base(a))
	if base != "go-build" {
		return "", "", fmt.Errorf("refusing non-go-build cache root %q", a)
	}
	for _, p := range strings.Split(strings.ToLower(filepath.ToSlash(a)), "/") {
		switch p {
		case "models", "model", "huggingface", "ollama", "lmstudio":
			return "", "", fmt.Errorf("refusing protected model-store path %q", a)
		}
	}
	resolved, e := filepath.EvalSymlinks(a)
	if e != nil {
		return "", "", fmt.Errorf("resolve GOCACHE: %w", e)
	}
	for _, p := range strings.Split(strings.ToLower(filepath.ToSlash(resolved)), "/") {
		switch p {
		case "models", "model", "huggingface", "ollama", "lmstudio":
			return "", "", fmt.Errorf("refusing protected resolved model-store path %q", resolved)
		}
	}
	return a, resolved, nil
}
func acquireGoCacheLock(root string) (func(), error) {
	p := filepath.Join(root, ".fak-gocache-lifecycle.lock")
	f, e := os.OpenFile(p, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return nil, e
	}
	f.Close()
	return func() { _ = os.Remove(p) }, nil
}
func scanGoCache(ctx context.Context, root string, max int, progress func(GoCacheProgress) error) ([]GoCacheEntry, int64, int, bool, string, error) {
	dir, err := os.Open(root)
	if err != nil {
		return nil, 0, 0, true, "", err
	}
	defer dir.Close()
	var out []GoCacheEntry
	var total int64
	seen := 0
	for {
		select {
		case <-ctx.Done():
			return out, total, seen, false, ctx.Err().Error(), nil
		default:
		}
		if seen >= max {
			return out, total, seen, false, "entry budget exhausted", nil
		}
		batch, readErr := dir.ReadDir(1)
		if len(batch) == 0 {
			if readErr == nil || errors.Is(readErr, fs.ErrClosed) {
				return out, total, seen, true, "", nil
			}
			if errors.Is(readErr, io.EOF) {
				return out, total, seen, true, "", nil
			}
			return nil, total, seen, false, "directory read failed", readErr
		}
		d := batch[0]
		if d.Name() == ".fak-gocache-lifecycle.lock" {
			continue
		}
		p := filepath.Join(root, d.Name())
		li, err := os.Lstat(p)
		if err != nil {
			return nil, total, seen, false, "entry metadata failed", err
		}
		seen++
		if li.Mode()&os.ModeSymlink != 0 {
			continue
		}
		var sz int64
		newest := li.ModTime()
		err = filepath.WalkDir(p, func(q string, de fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if q != p {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
				if seen >= max {
					return errEntryBudget
				}
				seen++
			}
			info, err := de.Info()
			if err != nil {
				return err
			}
			if info.ModTime().After(newest) {
				newest = info.ModTime()
			}
			if info.Mode().IsRegular() {
				sz += info.Size()
			}
			if progress != nil {
				if err := progress(GoCacheProgress{Entries: seen, Bytes: total + sz, Path: q}); err != nil {
					return fmt.Errorf("progress callback: %w", err)
				}
			}
			return nil
		})
		if errors.Is(err, errEntryBudget) {
			return out, total, seen, false, "entry budget exhausted", nil
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return out, total, seen, false, err.Error(), nil
		}
		if err != nil {
			return nil, total, seen, false, "entry scan failed", err
		}
		total += sz
		out = append(out, GoCacheEntry{Path: p, Bytes: sz, ModTime: newest})
	}
}

var errEntryBudget = errors.New("entry budget exhausted")

func safeCandidate(root, path string) error {
	li, e := os.Lstat(path)
	if e != nil {
		return e
	}
	if li.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("refusing symlink candidate %q", path)
	}
	resolved, e := filepath.EvalSymlinks(path)
	if e != nil {
		return e
	}
	rel, e := filepath.Rel(root, resolved)
	if e != nil || rel == "." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) || strings.Contains(rel, string(os.PathSeparator)) {
		return fmt.Errorf("candidate escapes GOCACHE %q", path)
	}
	return nil
}
