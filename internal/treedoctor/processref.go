package treedoctor

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// GoTmpProcess is the one-snapshot liveness evidence used for every candidate. Windows
// fills both fields from Win32_Process; Unix implementations provide the equivalent
// executable and command-line references exposed by the host.
type GoTmpProcess struct {
	PID            int    `json:"pid"`
	CommandLine    string `json:"command_line,omitempty"`
	ExecutablePath string `json:"executable_path,omitempty"`
}

func canonicalGoTmpRoot(opts GoTmpOptions) (string, error) {
	root := strings.TrimSpace(opts.Root)
	if root == "" {
		return "", nil
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve Go temp root: %w", err)
	}
	canonicalRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", fmt.Errorf("canonicalize Go temp root: %w", err)
	}
	canonicalRoot = filepath.Clean(canonicalRoot)
	if strings.TrimSpace(opts.RepoRoot) == "" {
		return canonicalRoot, nil
	}
	absRepo, err := filepath.Abs(opts.RepoRoot)
	if err != nil {
		return "", fmt.Errorf("resolve repository root: %w", err)
	}
	canonicalRepo, err := filepath.EvalSymlinks(absRepo)
	if err != nil {
		return "", fmt.Errorf("canonicalize repository root: %w", err)
	}
	canonicalScratch := filepath.Join(filepath.Clean(canonicalRepo), "_scratch")
	rel, err := filepath.Rel(canonicalScratch, canonicalRoot)
	if err != nil || rel == "." || relEscapesRoot(rel) {
		return "", fmt.Errorf("Go temp root %q must be a child of repository _scratch", canonicalRoot)
	}
	return canonicalRoot, nil
}

func relEscapesRoot(rel string) bool {
	return rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel)
}

func resolveGoTmpChild(root string, entry *GoTmpEntry) {
	if entry.Reason == GoTmpReasonReparsePoint || entry.Reason == GoTmpReasonScanFailed {
		return
	}
	canonical, err := filepath.EvalSymlinks(entry.Path)
	if err != nil {
		entry.ScanErr = err.Error()
		entry.Reason = GoTmpReasonScanFailed
		return
	}
	canonical = filepath.Clean(canonical)
	rel, err := filepath.Rel(root, canonical)
	if err != nil || relEscapesRoot(rel) || strings.Contains(rel, string(filepath.Separator)) {
		entry.Reason = GoTmpReasonOutsideRoot
		return
	}
	if rel == "." {
		entry.Reason = GoTmpReasonRoot
		return
	}
	entry.Path = canonical
}

func goTmpProcessSnapshot(opts GoTmpOptions) ([]GoTmpProcess, error) {
	if opts.Processes != nil {
		return opts.Processes()
	}
	return listGoTmpProcesses()
}

func goTmpReferencingPIDs(processes []GoTmpProcess, candidate string) []int {
	pids := make([]int, 0)
	for _, process := range processes {
		if goTmpFieldReferencesPath(process.CommandLine, candidate) || goTmpFieldReferencesPath(process.ExecutablePath, candidate) {
			pids = append(pids, process.PID)
		}
	}
	sort.Ints(pids)
	return pids
}

func goTmpFieldReferencesPath(field, candidate string) bool {
	if field == "" || candidate == "" {
		return false
	}
	field = filepath.ToSlash(field)
	candidate = strings.TrimSuffix(filepath.ToSlash(filepath.Clean(candidate)), "/")
	if runtime.GOOS == "windows" {
		field = strings.ToLower(field)
		candidate = strings.ToLower(candidate)
	}
	for start := 0; start <= len(field)-len(candidate); {
		i := strings.Index(field[start:], candidate)
		if i < 0 {
			return false
		}
		i += start
		beforeOK := i == 0 || goTmpPathBoundary(field[i-1])
		after := i + len(candidate)
		afterOK := after == len(field) || field[after] == '/' || goTmpPathBoundary(field[after])
		if beforeOK && afterOK {
			return true
		}
		start = i + 1
	}
	return false
}

func goTmpPathBoundary(b byte) bool {
	switch b {
	case 0, ' ', '\t', '\r', '\n', '"', '\'', '=', ':', ',', ';', '(', ')', '[', ']':
		return true
	default:
		return false
	}
}

func rebuildGoTmpReaped(rep *GoTmpReport) {
	rep.Reaped = nil
	rep.ReapedBytes = 0
	for _, entry := range rep.Entries {
		if entry.Verdict == GoTmpReap {
			rep.Reaped = append(rep.Reaped, entry.Path)
			rep.ReapedBytes += entry.Bytes
		}
	}
	sort.Strings(rep.Reaped)
}

type movedGoTmp struct {
	entryIndex int
	source     string
	dest       string
}

func applyGoTmpQuarantine(opts GoTmpOptions, rep GoTmpReport) GoTmpReport {
	eligible := make([]int, 0)
	for i := range rep.Entries {
		if rep.Entries[i].Verdict == GoTmpReap {
			eligible = append(eligible, i)
		}
	}
	rep.Reaped, rep.ReapedBytes = nil, 0
	if len(eligible) == 0 {
		return rep
	}

	quarantineParent := strings.TrimSpace(opts.QuarantineRoot)
	if quarantineParent == "" {
		quarantineParent = filepath.Join(os.TempDir(), "fak-maintenance-quarantine")
	}
	if err := os.MkdirAll(quarantineParent, 0o700); err != nil {
		markGoTmpApplyError(&rep, eligible, fmt.Errorf("create quarantine parent: %w", err))
		return rep
	}
	runDir, err := os.MkdirTemp(quarantineParent, "run-")
	if err != nil {
		markGoTmpApplyError(&rep, eligible, fmt.Errorf("create unique quarantine: %w", err))
		return rep
	}
	rep.Quarantine = runDir

	moved := make([]movedGoTmp, 0, len(eligible))
	for _, i := range eligible {
		entry := &rep.Entries[i]
		refreshed := scanGoTmpEntry(entry.Path, entry.Name, goTmpNow(opts), normalizedGoTmpWalkLimit(opts.MaxWalkEntries))
		resolveGoTmpChild(opts.Root, &refreshed)
		if refreshed.ScanErr != "" || refreshed.Truncated || refreshed.Reason != "" {
			entry.ScanErr = refreshed.ScanErr
			entry.Truncated = refreshed.Truncated
			entry.Reason = refreshed.Reason
			if entry.Reason == "" {
				entry.Reason = GoTmpReasonScanFailed
			}
			entry.Verdict = GoTmpKeepIndeterminate
			continue
		}
		entry.Bytes = refreshed.Bytes
		entry.Files = refreshed.Files
		entry.Directories = refreshed.Directories
		entry.NewestAgeSec = refreshed.NewestAgeSec
		if refreshed.NewestAgeSec < rep.MinAgeSec {
			entry.Verdict = GoTmpKeepLive
			entry.Reason = GoTmpReasonFresh
			continue
		}
		dest := filepath.Join(runDir, entry.Name)
		if err := os.Rename(entry.Path, dest); err != nil {
			entry.RemoveErr = fmt.Sprintf("move exact candidate to quarantine: %v", err)
			entry.Reason = GoTmpReasonQuarantined
			continue
		}
		entry.QuarantinePath = dest
		moved = append(moved, movedGoTmp{entryIndex: i, source: entry.Path, dest: dest})
	}
	if len(moved) == 0 {
		_ = os.Remove(runDir)
		return rep
	}

	postProcesses, postErr := goTmpProcessSnapshot(opts)
	rep.ProcessSnapshots++
	if postErr != nil {
		rep.ProcessErr = postErr.Error()
	}
	for _, movedEntry := range moved {
		entry := &rep.Entries[movedEntry.entryIndex]
		if postErr != nil {
			entry.Reason = GoTmpReasonPostMoveEnumerationFailed
			rollbackGoTmp(entry, movedEntry)
			continue
		}
		pids := append(goTmpReferencingPIDs(postProcesses, movedEntry.source), goTmpReferencingPIDs(postProcesses, movedEntry.dest)...)
		if len(pids) > 0 {
			sort.Ints(pids)
			entry.ReferencedBy = compactInts(pids)
			entry.Reason = GoTmpReasonPostMoveReferenced
			rollbackGoTmp(entry, movedEntry)
			continue
		}
		postScan := scanGoTmpEntry(movedEntry.dest, entry.Name, goTmpNow(opts), normalizedGoTmpWalkLimit(opts.MaxWalkEntries))
		resolveGoTmpChild(runDir, &postScan)
		if postScan.ScanErr != "" || postScan.Truncated || postScan.Reason != "" {
			entry.Reason = postScan.Reason
			if entry.Reason == "" {
				entry.Reason = GoTmpReasonScanFailed
			}
			entry.RemoveErr = postScan.ScanErr
			rollbackGoTmp(entry, movedEntry)
			continue
		}
		if postScan.NewestAgeSec < rep.MinAgeSec {
			entry.Reason = GoTmpReasonFresh
			entry.Verdict = GoTmpKeepLive
			rollbackGoTmp(entry, movedEntry)
			continue
		}
		files, dirs, err := deleteExactGoTmpTree(movedEntry.dest)
		if err != nil {
			entry.RemoveErr = fmt.Sprintf("delete exact quarantine tree: %v", err)
			entry.Reason = GoTmpReasonQuarantined
			continue
		}
		entry.Files = files
		entry.Directories = dirs
		entry.Removed = true
		entry.Reason = GoTmpReasonReclaimed
		entry.QuarantinePath = ""
		rep.Reaped = append(rep.Reaped, movedEntry.source)
		rep.ReapedBytes += entry.Bytes
	}
	sort.Strings(rep.Reaped)
	if err := os.Remove(runDir); err == nil {
		rep.Quarantine = ""
	}
	return rep
}

func goTmpNow(opts GoTmpOptions) time.Time {
	if !opts.Now.IsZero() {
		return opts.Now
	}
	return time.Now()
}

func normalizedGoTmpWalkLimit(limit int) int {
	if limit <= 0 {
		return DefaultGoTmpMaxWalkEntries
	}
	return limit
}

func rollbackGoTmp(entry *GoTmpEntry, moved movedGoTmp) {
	if err := os.Rename(moved.dest, moved.source); err != nil {
		entry.RemoveErr = fmt.Sprintf("restore referenced quarantine candidate: %v", err)
		return
	}
	entry.QuarantinePath = ""
	entry.Verdict = GoTmpKeepIndeterminate
}

func markGoTmpApplyError(rep *GoTmpReport, indexes []int, err error) {
	for _, i := range indexes {
		rep.Entries[i].RemoveErr = err.Error()
		rep.Entries[i].Reason = GoTmpReasonQuarantined
	}
}

func compactInts(values []int) []int {
	if len(values) < 2 {
		return values
	}
	out := values[:1]
	for _, value := range values[1:] {
		if value != out[len(out)-1] {
			out = append(out, value)
		}
	}
	return out
}

// deleteExactGoTmpTree removes enumerated exact paths bottom-up. It never expands a glob,
// follows a reparse point, or invokes recursive deletion.
func deleteExactGoTmpTree(root string) (files int, dirs int, retErr error) {
	directories := make([]string, 0)
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		reparse, err := goTmpIsReparse(path, info)
		if err != nil {
			return err
		}
		if reparse {
			return fmt.Errorf("reparse point appeared after quarantine: %s", path)
		}
		if entry.IsDir() {
			directories = append(directories, path)
			return nil
		}
		if err := os.Remove(path); err != nil {
			return err
		}
		files++
		return nil
	})
	if err != nil {
		return files, 0, err
	}
	sort.Slice(directories, func(i, j int) bool {
		return len(directories[i]) > len(directories[j])
	})
	for _, directory := range directories {
		if err := os.Remove(directory); err != nil {
			return files, dirs, err
		}
		dirs++
	}
	return files, dirs, nil
}
