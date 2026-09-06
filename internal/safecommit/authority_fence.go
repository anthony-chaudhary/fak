package safecommit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Reason tokens for authority fence revalidation (#11849).
const (
	// ReasonAuthorityStale reports that the caller's authority generation has been
	// superseded by a newer generation in the shared authority store.
	ReasonAuthorityStale = "AUTHORITY_STALE"

	// ReasonAuthorityForeign reports that the active authority lease belongs to a
	// different owner, session, workspace, or does not cover the requested pathset.
	ReasonAuthorityForeign = "AUTHORITY_FOREIGN"

	// ReasonAuthorityUnavailable reports that required authority cannot be verified
	// because the authority record is missing, unreadable, or the validator failed.
	ReasonAuthorityUnavailable = "AUTHORITY_UNAVAILABLE"
)

// AuthorityOutcome is the typed decision for an authority check.
type AuthorityOutcome string

const (
	// OutcomeAdmitted indicates authority was successfully validated.
	OutcomeAdmitted AuthorityOutcome = "admitted"

	// OutcomeRefused indicates authority was rejected (stale, foreign, or unavailable).
	OutcomeRefused AuthorityOutcome = "refused"
)

// DefaultAuthorityFileName is the standard authority lease record filename.
const DefaultAuthorityFileName = "fak-authority.json"

// AuthorityRecord is the durable authority lease record stored in the workspace.
type AuthorityRecord struct {
	Workspace  string   `json:"workspace,omitempty"`
	Owner      string   `json:"owner,omitempty"`
	SessionID  string   `json:"session_id,omitempty"`
	Generation uint64   `json:"generation"`
	Paths      []string `json:"paths,omitempty"`
}

// AuthorityValidator is a function that revalidates authority for the given fence and paths.
type AuthorityValidator func(ctx context.Context, fence AuthorityFence, requestedPaths []string) (AuthorityReceipt, error)

// AuthorityChecker is an optional interface for custom authority validation.
type AuthorityChecker interface {
	ValidateAuthority(ctx context.Context, fence AuthorityFence, requestedPaths []string) (AuthorityReceipt, error)
}

// AuthorityFence binds the declared workspace, trusted owner/session, exact path set,
// and generation to an authority check before staging paths (#11849).
type AuthorityFence struct {
	Workspace  string             `json:"workspace,omitempty"`
	Owner      string             `json:"owner,omitempty"`
	SessionID  string             `json:"session_id,omitempty"`
	Generation uint64             `json:"generation,omitempty"`
	Paths      []string           `json:"paths,omitempty"`
	Path       string             `json:"path,omitempty"` // optional explicit authority file path
	Validator  AuthorityValidator `json:"-"`
	Checker    AuthorityChecker   `json:"-"`
}

// IsZero reports whether the fence is unconfigured (nil or zero values).
func (f *AuthorityFence) IsZero() bool {
	if f == nil {
		return true
	}
	return f.Workspace == "" &&
		f.Owner == "" &&
		f.SessionID == "" &&
		f.Generation == 0 &&
		len(f.Paths) == 0 &&
		f.Path == "" &&
		f.Validator == nil &&
		f.Checker == nil
}

// AuthorityReceipt is the typed witness of an authority fence evaluation.
type AuthorityReceipt struct {
	Outcome          AuthorityOutcome `json:"outcome"`
	Reason           string           `json:"reason,omitempty"`
	Detail           string           `json:"detail,omitempty"`
	Workspace        string           `json:"workspace,omitempty"`
	Owner            string           `json:"owner,omitempty"`
	SessionID        string           `json:"session_id,omitempty"`
	Generation       uint64           `json:"generation,omitempty"`
	Paths            []string         `json:"paths,omitempty"`
	ActiveOwner      string           `json:"active_owner,omitempty"`
	ActiveGeneration uint64           `json:"active_generation,omitempty"`
}

// Refused reports whether the authority check refused the commit.
func (r AuthorityReceipt) Refused() bool {
	return r.Outcome == OutcomeRefused
}

// AuthorityExitCode classifies an authority refusal reason into its process exit code.
func AuthorityExitCode(reason string) (code int, ok bool) {
	switch reason {
	case ReasonAuthorityStale, ReasonAuthorityForeign, ReasonAuthorityUnavailable:
		return ExitRefused, true
	}
	return 0, false
}

// WriteAuthorityRecord writes an authority record for workspace dir to disk.
func WriteAuthorityRecord(dir string, rec AuthorityRecord) error {
	gitDir := filepath.Join(dir, ".git")
	targetDir := dir
	if fi, err := os.Stat(gitDir); err == nil && fi.IsDir() {
		targetDir = gitDir
	}
	return WriteAuthorityRecordFile(filepath.Join(targetDir, DefaultAuthorityFileName), rec)
}

// WriteAuthorityRecordFile writes an authority record to an explicit file path.
func WriteAuthorityRecordFile(filePath string, rec AuthorityRecord) error {
	if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0o644)
}

// ReadAuthorityRecord reads the active authority record for workspace dir.
func ReadAuthorityRecord(dir string) (AuthorityRecord, error) {
	gitDir := filepath.Join(dir, ".git")
	if fi, err := os.Stat(gitDir); err == nil && fi.IsDir() {
		gitPath := filepath.Join(gitDir, DefaultAuthorityFileName)
		if _, err := os.Stat(gitPath); err == nil {
			return ReadAuthorityRecordFile(gitPath)
		}
	}
	return ReadAuthorityRecordFile(filepath.Join(dir, DefaultAuthorityFileName))
}

// ReadAuthorityRecordFile reads an authority record from an explicit file path.
func ReadAuthorityRecordFile(filePath string) (AuthorityRecord, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return AuthorityRecord{}, err
	}
	var rec AuthorityRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return AuthorityRecord{}, err
	}
	return rec, nil
}

func resolveAuthorityPath(ctx context.Context, run Runner, dir string, fence *AuthorityFence) string {
	if fence != nil && fence.Path != "" {
		return fence.Path
	}
	if run != nil && dir != "" {
		if gitDir := resolveGitDir(ctx, run, dir); gitDir != "" {
			p := filepath.Join(gitDir, DefaultAuthorityFileName)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	gitDir := filepath.Join(dir, ".git")
	if fi, err := os.Stat(gitDir); err == nil && fi.IsDir() {
		p := filepath.Join(gitDir, DefaultAuthorityFileName)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	rootPath := filepath.Join(dir, DefaultAuthorityFileName)
	if _, err := os.Stat(rootPath); err == nil {
		return rootPath
	}
	if run != nil && dir != "" {
		if gitDir := resolveGitDir(ctx, run, dir); gitDir != "" {
			return filepath.Join(gitDir, DefaultAuthorityFileName)
		}
	}
	if fi, err := os.Stat(gitDir); err == nil && fi.IsDir() {
		return filepath.Join(gitDir, DefaultAuthorityFileName)
	}
	return rootPath
}

func equalWorkspace(a, b string) bool {
	aClean := filepath.Clean(a)
	bClean := filepath.Clean(b)
	if aClean == bClean {
		return true
	}
	if runtime.GOOS == "windows" && strings.EqualFold(aClean, bClean) {
		return true
	}
	aAbs, errA := filepath.Abs(aClean)
	bAbs, errB := filepath.Abs(bClean)
	if errA == nil && errB == nil {
		if aAbs == bAbs {
			return true
		}
		if runtime.GOOS == "windows" && strings.EqualFold(aAbs, bAbs) {
			return true
		}
	}
	if aReal, err := filepath.EvalSymlinks(aClean); err == nil {
		if bReal, err := filepath.EvalSymlinks(bClean); err == nil {
			if aReal == bReal || (runtime.GOOS == "windows" && strings.EqualFold(aReal, bReal)) {
				return true
			}
		}
	}
	return false
}

func pathCovered(allowed []string, target string) bool {
	target = filepath.ToSlash(filepath.Clean(target))
	for _, a := range allowed {
		a = filepath.ToSlash(filepath.Clean(a))
		if a == target || a == "." {
			return true
		}
		if strings.HasSuffix(a, "/") && strings.HasPrefix(target, a) {
			return true
		}
		if strings.HasPrefix(target, a+"/") {
			return true
		}
		if runtime.GOOS == "windows" {
			if strings.EqualFold(a, target) {
				return true
			}
			if len(target) > len(a) && target[len(a)] == '/' && strings.EqualFold(target[:len(a)], a) {
				return true
			}
		}
	}
	return false
}

// checkAuthorityFence evaluates the authority fence immediately before path staging.
func checkAuthorityFence(ctx context.Context, run Runner, dir string, fence *AuthorityFence, customValidator AuthorityValidator, requestedPaths []string) (AuthorityReceipt, bool) {
	if (fence == nil || fence.IsZero()) && customValidator == nil {
		return AuthorityReceipt{Outcome: OutcomeAdmitted}, false
	}

	f := AuthorityFence{}
	if fence != nil {
		f = *fence
	}

	// 1. Custom validator configured on Options or fence
	if customValidator != nil {
		receipt, err := customValidator(ctx, f, requestedPaths)
		if err != nil {
			return AuthorityReceipt{
				Outcome:    OutcomeRefused,
				Reason:     ReasonAuthorityUnavailable,
				Detail:     fmt.Sprintf("authority validator failed: %v", err),
				Workspace:  f.Workspace,
				Owner:      f.Owner,
				SessionID:  f.SessionID,
				Generation: f.Generation,
				Paths:      requestedPaths,
			}, true
		}
		if receipt.Workspace == "" {
			receipt.Workspace = f.Workspace
		}
		if receipt.Owner == "" {
			receipt.Owner = f.Owner
		}
		if receipt.SessionID == "" {
			receipt.SessionID = f.SessionID
		}
		if receipt.Generation == 0 {
			receipt.Generation = f.Generation
		}
		if len(receipt.Paths) == 0 {
			receipt.Paths = requestedPaths
		}
		if receipt.Outcome == OutcomeRefused && receipt.Reason == "" {
			receipt.Reason = ReasonAuthorityForeign
		}
		return receipt, receipt.Outcome == OutcomeRefused
	}
	if f.Validator != nil {
		receipt, err := f.Validator(ctx, f, requestedPaths)
		if err != nil {
			return AuthorityReceipt{
				Outcome:    OutcomeRefused,
				Reason:     ReasonAuthorityUnavailable,
				Detail:     fmt.Sprintf("authority validator failed: %v", err),
				Workspace:  f.Workspace,
				Owner:      f.Owner,
				SessionID:  f.SessionID,
				Generation: f.Generation,
				Paths:      requestedPaths,
			}, true
		}
		if receipt.Workspace == "" {
			receipt.Workspace = f.Workspace
		}
		if receipt.Owner == "" {
			receipt.Owner = f.Owner
		}
		if receipt.SessionID == "" {
			receipt.SessionID = f.SessionID
		}
		if receipt.Generation == 0 {
			receipt.Generation = f.Generation
		}
		if len(receipt.Paths) == 0 {
			receipt.Paths = requestedPaths
		}
		if receipt.Outcome == OutcomeRefused && receipt.Reason == "" {
			receipt.Reason = ReasonAuthorityForeign
		}
		return receipt, receipt.Outcome == OutcomeRefused
	}
	if f.Checker != nil {
		receipt, err := f.Checker.ValidateAuthority(ctx, f, requestedPaths)
		if err != nil {
			return AuthorityReceipt{
				Outcome:    OutcomeRefused,
				Reason:     ReasonAuthorityUnavailable,
				Detail:     fmt.Sprintf("authority checker failed: %v", err),
				Workspace:  f.Workspace,
				Owner:      f.Owner,
				SessionID:  f.SessionID,
				Generation: f.Generation,
				Paths:      requestedPaths,
			}, true
		}
		if receipt.Workspace == "" {
			receipt.Workspace = f.Workspace
		}
		if receipt.Owner == "" {
			receipt.Owner = f.Owner
		}
		if receipt.SessionID == "" {
			receipt.SessionID = f.SessionID
		}
		if receipt.Generation == 0 {
			receipt.Generation = f.Generation
		}
		if len(receipt.Paths) == 0 {
			receipt.Paths = requestedPaths
		}
		if receipt.Outcome == OutcomeRefused && receipt.Reason == "" {
			receipt.Reason = ReasonAuthorityForeign
		}
		return receipt, receipt.Outcome == OutcomeRefused
	}

	// 2. Default on-disk authority lease record evaluation
	path := resolveAuthorityPath(ctx, run, dir, &f)
	record, err := ReadAuthorityRecordFile(path)
	if err != nil {
		return AuthorityReceipt{
			Outcome:    OutcomeRefused,
			Reason:     ReasonAuthorityUnavailable,
			Detail:     fmt.Sprintf("authority record unavailable at %s: %v", path, err),
			Workspace:  f.Workspace,
			Owner:      f.Owner,
			SessionID:  f.SessionID,
			Generation: f.Generation,
			Paths:      requestedPaths,
		}, true
	}

	receipt := AuthorityReceipt{
		Workspace:        f.Workspace,
		Owner:            f.Owner,
		SessionID:        f.SessionID,
		Generation:       f.Generation,
		Paths:            requestedPaths,
		ActiveOwner:      record.Owner,
		ActiveGeneration: record.Generation,
	}
	if receipt.Workspace == "" {
		receipt.Workspace = record.Workspace
	}
	if receipt.Workspace == "" {
		receipt.Workspace = dir
	}

	// Workspace check
	if f.Workspace != "" && record.Workspace != "" && !equalWorkspace(f.Workspace, record.Workspace) {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityForeign
		receipt.Detail = fmt.Sprintf("authority workspace %q does not match active workspace %q", f.Workspace, record.Workspace)
		return receipt, true
	}
	if f.Workspace != "" && dir != "" && !equalWorkspace(f.Workspace, dir) {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityForeign
		receipt.Detail = fmt.Sprintf("authority workspace %q does not match repository workspace %q", f.Workspace, dir)
		return receipt, true
	}
	if dir != "" && record.Workspace != "" && !equalWorkspace(dir, record.Workspace) {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityForeign
		receipt.Detail = fmt.Sprintf("repository workspace %q does not match active authority workspace %q", dir, record.Workspace)
		return receipt, true
	}

	// Generation check: stale generation
	if record.Generation > f.Generation {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityStale
		receipt.Detail = fmt.Sprintf("authority generation %d is stale; active generation is %d (held by %q)", f.Generation, record.Generation, record.Owner)
		return receipt, true
	}
	if f.Generation != 0 && record.Generation != f.Generation {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityStale
		receipt.Detail = fmt.Sprintf("authority generation %d does not match active generation %d", f.Generation, record.Generation)
		return receipt, true
	}

	// Owner check
	if f.Owner != "" && f.Owner != record.Owner {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityForeign
		receipt.Detail = fmt.Sprintf("authority owner %q is foreign; active owner is %q", f.Owner, record.Owner)
		return receipt, true
	}

	// Session check
	if f.SessionID != "" && f.SessionID != record.SessionID {
		receipt.Outcome = OutcomeRefused
		receipt.Reason = ReasonAuthorityForeign
		receipt.Detail = fmt.Sprintf("authority session %q is foreign; active session is %q", f.SessionID, record.SessionID)
		return receipt, true
	}

	// Paths check
	if len(f.Paths) > 0 {
		for _, p := range requestedPaths {
			if !pathCovered(f.Paths, p) {
				receipt.Outcome = OutcomeRefused
				receipt.Reason = ReasonAuthorityForeign
				receipt.Detail = fmt.Sprintf("requested path %q is outside fence authorized path set", p)
				return receipt, true
			}
		}
	}
	if len(record.Paths) > 0 {
		for _, p := range requestedPaths {
			if !pathCovered(record.Paths, p) {
				receipt.Outcome = OutcomeRefused
				receipt.Reason = ReasonAuthorityForeign
				receipt.Detail = fmt.Sprintf("requested path %q is outside active lease authorized path set", p)
				return receipt, true
			}
		}
	}

	receipt.Outcome = OutcomeAdmitted
	return receipt, false
}
