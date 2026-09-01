package treedoctor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
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
	if lookup != nil {
		if v := strings.TrimSpace(lookup("GOCACHE")); v != "" {
			if strings.EqualFold(v, "off") {
				return ""
			}
			return filepath.Clean(v)
		}
	}
	if userCacheDir == nil {
		return ""
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
	Remove              func(string) error
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
	r := GoCacheReport{Root: filepath.Clean(o.Root), ScanComplete: true, CleanupHints: []string{"GOTMP cleanup: fak git-daily --gotmp-dir <GOTMPDIR>", "GOTMP cleanup: set $FAK_GOTMPDIR, then run fak git-daily", "worktree cleanup dry-run: fak worktree worker reap --all-cold", "worktree cleanup apply: fak worktree worker reap --all-cold --apply"}}
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
	if o.Remove == nil {
		o.Remove = os.Remove
	}
	if !apply {
		o.Remove = os.Remove
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
		if err := o.Remove(e.Path); err != nil {
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
	// Custom GOCACHE basenames are intentionally refused: a path supplied through the
	// environment is not enough evidence that an arbitrary directory is Go-owned.
	if strings.ToLower(filepath.Base(a)) != "go-build" {
		return "", "", fmt.Errorf("refusing non-go-build cache root %q", a)
	}
	if containsProtectedModelStore(a) {
		return "", "", fmt.Errorf("refusing protected model-store path %q", a)
	}
	resolved, e := filepath.EvalSymlinks(a)
	if e != nil {
		return "", "", fmt.Errorf("resolve GOCACHE: %w", e)
	}
	if containsProtectedModelStore(resolved) {
		return "", "", fmt.Errorf("refusing protected resolved model-store path %q", resolved)
	}
	return a, resolved, nil
}

func containsProtectedModelStore(path string) bool {
	for _, component := range strings.Split(strings.ToLower(filepath.ToSlash(path)), "/") {
		switch component {
		case "models", "model", "fak-models", "huggingface", "ollama", ".ollama", "llama.cpp", "lmstudio":
			return true
		}
		if strings.HasPrefix(component, "qwen") {
			return true
		}
	}
	return false
}

type goCacheLockOwner struct {
	PID   int    `json:"pid"`
	Token string `json:"token"`
}

func acquireGoCacheLock(root string) (func(), error) {
	return acquireGoCacheLockWith(root, goCacheProcessAlive)
}

func acquireGoCacheLockWith(root string, processAlive func(int) (bool, error)) (func(), error) {
	path := filepath.Join(root, ".fak-gocache-lifecycle.lock")
	owner := goCacheLockOwner{PID: os.Getpid(), Token: strconv.FormatInt(time.Now().UnixNano(), 36) + "-" + strconv.Itoa(os.Getpid())}
	contents, err := json.Marshal(owner)
	if err != nil {
		return nil, err
	}
	create := func() error {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		if _, err = f.Write(contents); err != nil {
			_ = f.Close()
			_ = os.Remove(path)
			return err
		}
		return f.Close()
	}
	if err := create(); err != nil {
		if !errors.Is(err, fs.ErrExist) || processAlive == nil {
			return nil, err
		}
		staleContents, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil, err
		}
		var stale goCacheLockOwner
		if json.Unmarshal(staleContents, &stale) != nil || stale.PID <= 0 || stale.Token == "" {
			return nil, err
		}
		alive, aliveErr := processAlive(stale.PID)
		if aliveErr != nil || alive {
			return nil, err
		}
		current, readErr := os.ReadFile(path)
		if readErr != nil || string(current) != string(staleContents) {
			return nil, err
		}
		if removeErr := os.Remove(path); removeErr != nil {
			return nil, err
		}
		if retryErr := create(); retryErr != nil {
			return nil, retryErr
		}
	}
	return func() {
		current, err := os.ReadFile(path)
		if err == nil && string(current) == string(contents) {
			_ = os.Remove(path)
		}
	}, nil
}

func scanGoCache(ctx context.Context, root string, max int, progress func(GoCacheProgress) error) ([]GoCacheEntry, int64, int, bool, string, error) {
	var out []GoCacheEntry
	var total int64
	seen := 0
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if path == root {
			return nil
		}
		if entry.Name() == ".fak-gocache-lifecycle.lock" && filepath.Dir(path) == root {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("refusing symlink in GOCACHE %q", path)
		}
		if info.IsDir() {
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("refusing non-regular GOCACHE entry %q", path)
		}
		if seen >= max {
			return errEntryBudget
		}
		seen++
		total += info.Size()
		out = append(out, GoCacheEntry{Path: path, Bytes: info.Size(), ModTime: info.ModTime()})
		if progress != nil {
			if err := progress(GoCacheProgress{Entries: seen, Bytes: total, Path: path}); err != nil {
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
	return out, total, seen, true, "", nil
}

var errEntryBudget = errors.New("entry budget exhausted")

func safeCandidate(root, path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return fmt.Errorf("refusing non-regular cache candidate %q", path)
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
		return fmt.Errorf("candidate escapes GOCACHE %q", path)
	}
	return nil
}
