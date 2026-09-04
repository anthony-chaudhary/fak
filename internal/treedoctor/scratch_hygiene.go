package treedoctor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultScratchGoFilesThreshold is the maximum number of untracked .go files
// permitted in _scratch before a scratch hygiene warning is triggered to prevent
// language server memory explosion.
const DefaultScratchGoFilesThreshold = 10000

// DefaultScratchQuotaFiles is the default maximum ceiling of files permitted in
// _scratch before proactive quota enforcement triggers a refusal.
const DefaultScratchQuotaFiles = 1000

// ErrCodeScratchQuotaExceeded is the refusal code returned when scratch file count exceeds quota.
const ErrCodeScratchQuotaExceeded = "SCRATCH_QUOTA_EXCEEDED"

// ErrScratchQuotaExceeded is the sentinel error matching SCRATCH_QUOTA_EXCEEDED refusals.
var ErrScratchQuotaExceeded = errors.New("SCRATCH_QUOTA_EXCEEDED")

// ScratchQuotaError is a structured refusal emitted when _scratch exceeds the file quota limit.
type ScratchQuotaError struct {
	Code  string `json:"code"`
	Count int    `json:"count"`
	Quota int    `json:"quota"`
}

func (e *ScratchQuotaError) Error() string {
	return fmt.Sprintf("%s: scratch file count (%d) exceeds quota limit (%d)", e.Code, e.Count, e.Quota)
}

func (e *ScratchQuotaError) Is(target error) bool {
	if target == ErrScratchQuotaExceeded {
		return true
	}
	if q, ok := target.(*ScratchQuotaError); ok {
		return e.Code == q.Code
	}
	return false
}

// ScratchQuotaReport provides structured result of a scratch quota evaluation.
type ScratchQuotaReport struct {
	Count    int    `json:"count"`
	Quota    int    `json:"quota"`
	Exceeded bool   `json:"exceeded"`
	Code     string `json:"code,omitempty"`
}

// ScratchHygieneReport classifies untracked .go files in _scratch.
type ScratchHygieneReport struct {
	ScratchUntrackedGoFiles int    `json:"scratch_untracked_go_files"`
	Threshold               int    `json:"threshold"`
	Exceeded                bool   `json:"exceeded"`
	Warning                 string `json:"warning,omitempty"`
}

// diagnoseScratchHygiene inspects repoRoot/_scratch for untracked .go files
// against DefaultScratchGoFilesThreshold (10,000).
func diagnoseScratchHygiene(repoRoot string) ScratchHygieneReport {
	return DiagnoseScratchHygieneThreshold(repoRoot, DefaultScratchGoFilesThreshold)
}

// DiagnoseScratchHygiene inspects repoRoot/_scratch for untracked .go files
// against DefaultScratchGoFilesThreshold.
func DiagnoseScratchHygiene(repoRoot string) ScratchHygieneReport {
	return DiagnoseScratchHygieneThreshold(repoRoot, DefaultScratchGoFilesThreshold)
}

// DiagnoseScratchHygieneThreshold inspects repoRoot/_scratch for untracked .go files
// against the provided threshold.
func DiagnoseScratchHygieneThreshold(repoRoot string, threshold int) ScratchHygieneReport {
	if threshold <= 0 {
		threshold = DefaultScratchGoFilesThreshold
	}
	if repoRoot == "" {
		return ScratchHygieneReport{Threshold: threshold}
	}
	scratchDir := filepath.Join(repoRoot, scratchNamespace)
	info, err := os.Stat(scratchDir)
	if err != nil || !info.IsDir() {
		return ScratchHygieneReport{Threshold: threshold}
	}

	count := 0
	_ = filepath.WalkDir(scratchDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(d.Name(), ".go") {
			count++
		}
		return nil
	})

	return BuildScratchHygieneReport(count, threshold)
}

// BuildScratchHygieneReport constructs a ScratchHygieneReport from an untracked
// .go file count and threshold.
func BuildScratchHygieneReport(count, threshold int) ScratchHygieneReport {
	if threshold <= 0 {
		threshold = DefaultScratchGoFilesThreshold
	}
	rep := ScratchHygieneReport{
		ScratchUntrackedGoFiles: count,
		Threshold:               threshold,
	}
	if count > threshold {
		rep.Exceeded = true
		if threshold == DefaultScratchGoFilesThreshold {
			rep.Warning = fmt.Sprintf("_scratch contains >10,000 untracked .go files (%d) without quarantine; isolate workspace scope or reap scratch to prevent LSP/gopls memory explosion", count)
		} else {
			rep.Warning = fmt.Sprintf("_scratch contains >%d untracked .go files (%d) without quarantine; isolate workspace scope or reap scratch to prevent LSP/gopls memory explosion", threshold, count)
		}
	}
	return rep
}

// CountScratchFiles counts the non-directory files contained under repoRoot/_scratch,
// ignoring .git directories.
func CountScratchFiles(repoRoot string) (int, error) {
	if repoRoot == "" {
		return 0, nil
	}
	scratchDir := filepath.Join(repoRoot, scratchNamespace)
	info, err := os.Stat(scratchDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("inspect scratch directory: %w", err)
	}
	if !info.IsDir() {
		return 0, nil
	}

	count := 0
	err = filepath.WalkDir(scratchDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		count++
		return nil
	})
	if err != nil {
		return 0, fmt.Errorf("walk scratch directory: %w", err)
	}
	return count, nil
}

// CheckScratchQuota evaluates the scratch file count against the quota ceiling.
// If quota <= 0, DefaultScratchQuotaFiles (1,000) is used.
func CheckScratchQuota(repoRoot string, quota int) (ScratchQuotaReport, error) {
	if quota <= 0 {
		quota = DefaultScratchQuotaFiles
	}
	count, err := CountScratchFiles(repoRoot)
	if err != nil {
		return ScratchQuotaReport{Quota: quota}, err
	}
	rep := ScratchQuotaReport{
		Count:    count,
		Quota:    quota,
		Exceeded: count > quota,
	}
	if rep.Exceeded {
		rep.Code = ErrCodeScratchQuotaExceeded
	}
	return rep, nil
}

// EnforceScratchQuota verifies that repoRoot/_scratch does not exceed quota files.
// If quota <= 0, DefaultScratchQuotaFiles (1,000) is used.
// When exceeded, a structured *ScratchQuotaError (SCRATCH_QUOTA_EXCEEDED) refusal is returned.
func EnforceScratchQuota(repoRoot string, quota int) error {
	if quota <= 0 {
		quota = DefaultScratchQuotaFiles
	}
	count, err := CountScratchFiles(repoRoot)
	if err != nil {
		return err
	}
	if count > quota {
		return &ScratchQuotaError{
			Code:  ErrCodeScratchQuotaExceeded,
			Count: count,
			Quota: quota,
		}
	}
	return nil
}

// ReapSessionScratch removes automated session scratch files and directories associated
// with sessionID under repoRoot/_scratch. If no scratch exists for sessionID, it returns nil.
func ReapSessionScratch(repoRoot, sessionID string) error {
	if err := validateSessionID(sessionID); err != nil {
		return err
	}
	if strings.TrimSpace(repoRoot) == "" {
		return fmt.Errorf("repository root is empty")
	}

	repo, err := filepath.Abs(repoRoot)
	if err != nil {
		return fmt.Errorf("resolve repository root: %w", err)
	}
	repo, err = filepath.EvalSymlinks(repo)
	if err != nil {
		return fmt.Errorf("resolve repository root: %w", err)
	}
	repoInfo, err := os.Stat(repo)
	if err != nil {
		return fmt.Errorf("inspect repository root: %w", err)
	}
	if !repoInfo.IsDir() {
		return fmt.Errorf("repository root %q is not a directory", repo)
	}

	scratchRoot := filepath.Join(repo, scratchNamespace)
	scratchInfo, err := os.Lstat(scratchRoot)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect scratch root: %w", err)
	}
	if !scratchInfo.IsDir() {
		return fmt.Errorf("scratch root %q is not a directory", scratchRoot)
	}
	resolvedScratch, err := filepath.EvalSymlinks(scratchRoot)
	if err != nil {
		return fmt.Errorf("resolve scratch root: %w", err)
	}
	if !sameFilesystemPath(resolvedScratch, scratchRoot) {
		return fmt.Errorf("scratch root resolves outside its declared path: %s", resolvedScratch)
	}

	cleanID := filepath.Clean(filepath.FromSlash(strings.TrimSpace(sessionID)))
	if first := strings.Split(cleanID, string(filepath.Separator))[0]; first == scratchNamespace {
		cleanID = strings.TrimPrefix(cleanID, scratchNamespace+string(filepath.Separator))
	}

	candidates := []string{
		cleanID,
	}
	if !strings.HasPrefix(cleanID, "sessions"+string(filepath.Separator)) && cleanID != "sessions" {
		candidates = append(candidates, filepath.Join("sessions", cleanID))
	}
	base := filepath.Base(cleanID)
	dir := filepath.Dir(cleanID)
	if !strings.HasPrefix(base, "session-") {
		sessionPrefixed := "session-" + base
		if dir != "." {
			sessionPrefixed = filepath.Join(dir, sessionPrefixed)
		}
		candidates = append(candidates, sessionPrefixed)
		if !strings.HasPrefix(sessionPrefixed, "sessions"+string(filepath.Separator)) {
			candidates = append(candidates, filepath.Join("sessions", sessionPrefixed))
		}
	}

	seen := make(map[string]struct{})
	for _, cand := range candidates {
		cand = filepath.Clean(cand)
		if _, ok := seen[cand]; ok {
			continue
		}
		seen[cand] = struct{}{}

		target := filepath.Join(scratchRoot, cand)
		info, err := os.Lstat(target)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect session scratch target %s: %w", target, err)
		}

		resolvedTarget, err := filepath.EvalSymlinks(target)
		if err != nil {
			return fmt.Errorf("resolve session scratch target %s: %w", target, err)
		}
		rel, err := filepath.Rel(resolvedScratch, resolvedTarget)
		if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("%w: session target %q escapes scratch root", ErrUnsafeScratchProducer, target)
		}

		if err := refuseReparse(target, info); err != nil {
			return err
		}

		if !info.IsDir() {
			if err := os.Remove(target); err != nil {
				return fmt.Errorf("remove session scratch file %s: %w", target, err)
			}
			continue
		}

		files, directories, err := enumerateScratchProducer(resolvedTarget)
		if err != nil {
			return fmt.Errorf("enumerate session scratch tree %s: %w", target, err)
		}
		if _, err := removeScratchProducerExact(files, directories); err != nil {
			return fmt.Errorf("remove session scratch entries from %s: %w", target, err)
		}
	}

	sessionsParent := filepath.Join(scratchRoot, "sessions")
	_ = os.Remove(sessionsParent)

	return nil
}

func validateSessionID(sessionID string) error {
	trimmed := strings.TrimSpace(sessionID)
	if trimmed == "" {
		return fmt.Errorf("session ID is empty")
	}
	if sessionID != trimmed {
		return fmt.Errorf("session ID %q contains surrounding whitespace", sessionID)
	}
	if sessionID == "." || sessionID == ".." || sessionID == scratchNamespace {
		return fmt.Errorf("session ID %q names a protected root", sessionID)
	}
	if filepath.IsAbs(sessionID) || filepath.VolumeName(sessionID) != "" {
		return fmt.Errorf("session ID %q must not be an absolute path", sessionID)
	}
	if strings.ContainsAny(sessionID, "*?[") {
		return fmt.Errorf("session ID %q contains glob syntax", sessionID)
	}
	clean := filepath.Clean(filepath.FromSlash(sessionID))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return fmt.Errorf("session ID %q escapes scratch root", sessionID)
	}
	return nil
}
