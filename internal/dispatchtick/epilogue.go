package dispatchtick

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// EpilogueSchema is the canonical schema for dispatch epilogues.
const EpilogueSchema = "fak-dispatch-epilogue/1"

// EpilogueStatus represents the lifecycle state of a commit epilogue.
type EpilogueStatus string

const (
	EpilogueStatusPending  EpilogueStatus = "pending"
	EpilogueStatusLanding  EpilogueStatus = "landing"
	EpilogueStatusLanded   EpilogueStatus = "landed"
	EpilogueStatusConflict EpilogueStatus = "conflict"
	EpilogueStatusFailed   EpilogueStatus = "failed"
)

// EpilogueRecord models one queued commit epilogue.
type EpilogueRecord struct {
	Schema      string         `json:"schema"`
	ID          string         `json:"id"`
	Issue       int            `json:"issue"`
	Lane        string         `json:"lane"`
	WorkerPID   int            `json:"worker_pid,omitempty"`
	BaseSHA     string         `json:"base_sha,omitempty"`
	Paths       []string       `json:"paths"`
	Message     string         `json:"message"`
	WorktreeDir string         `json:"worktree_dir,omitempty"`
	Patch       string         `json:"patch,omitempty"`
	Status      EpilogueStatus `json:"status"`
	SubmittedAt time.Time      `json:"submitted_at"`
	LandedSHA   string         `json:"landed_sha,omitempty"`
	LandedAt    *time.Time     `json:"landed_at,omitempty"`
	Error       string         `json:"error,omitempty"`
}

// EpilogueGitRunner is a function type for running git commands against a target directory.
type EpilogueGitRunner func(ctx context.Context, dir string, args ...string) (string, error)

// EpilogueDrainOptions parameterizes epilogue draining.
type EpilogueDrainOptions struct {
	Git     EpilogueGitRunner
	Limit   int
	DryRun  bool
	Push    bool
	Signoff bool
}

// EpilogueDrainResult summarizes the outcome of draining epilogues.
type EpilogueDrainResult struct {
	Landed     int              `json:"landed"`
	Conflicted int              `json:"conflicted"`
	Failed     int              `json:"failed"`
	Total      int              `json:"total"`
	Records    []EpilogueRecord `json:"records,omitempty"`
}

func epiloguesDir(runsDir string) string {
	if runsDir == "" {
		runsDir = RunsDirName
	}
	if filepath.Base(runsDir) == "epilogues" {
		return runsDir
	}
	return filepath.Join(runsDir, "epilogues")
}

// SubmitEpilogue writes an epilogue record atomically to disk under <runsDir>/epilogues/<id>.json.
func SubmitEpilogue(runsDir string, rec EpilogueRecord) (EpilogueRecord, error) {
	if rec.Schema == "" {
		rec.Schema = EpilogueSchema
	}
	if rec.ID == "" {
		rec.ID = fmt.Sprintf("epilogue-%d-%d", rec.Issue, time.Now().UTC().UnixNano())
	}
	if rec.Status == "" {
		rec.Status = EpilogueStatusPending
	}
	if rec.SubmittedAt.IsZero() {
		rec.SubmittedAt = time.Now().UTC()
	}

	dir := epiloguesDir(runsDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return rec, fmt.Errorf("create epilogues dir: %w", err)
	}

	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return rec, fmt.Errorf("marshal epilogue: %w", err)
	}
	data = append(data, '\n')

	targetPath := filepath.Join(dir, rec.ID+".json")
	tmpFile, err := os.CreateTemp(dir, rec.ID+"-*.tmp")
	if err != nil {
		return rec, fmt.Errorf("create temp epilogue: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		_ = os.Remove(tmpName)
	}()

	if _, err := tmpFile.Write(data); err != nil {
		_ = tmpFile.Close()
		return rec, fmt.Errorf("write temp epilogue: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return rec, fmt.Errorf("close temp epilogue: %w", err)
	}

	if err := os.Rename(tmpName, targetPath); err != nil {
		_ = os.Remove(targetPath)
		if err2 := os.Rename(tmpName, targetPath); err2 != nil {
			return rec, fmt.Errorf("atomic rename epilogue: %w", err2)
		}
	}
	return rec, nil
}

// ListEpilogues returns all epilogues from <runsDir>/epilogues, optionally filtered by status,
// sorted FIFO by SubmittedAt.
func ListEpilogues(runsDir string, statusFilter EpilogueStatus) ([]EpilogueRecord, error) {
	dir := epiloguesDir(runsDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read epilogues dir: %w", err)
	}

	var records []EpilogueRecord
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var rec EpilogueRecord
		if err := json.Unmarshal(data, &rec); err != nil {
			continue
		}
		if statusFilter != "" && rec.Status != statusFilter {
			continue
		}
		records = append(records, rec)
	}

	sort.Slice(records, func(i, j int) bool {
		if records[i].SubmittedAt.Equal(records[j].SubmittedAt) {
			return records[i].ID < records[j].ID
		}
		return records[i].SubmittedAt.Before(records[j].SubmittedAt)
	})

	return records, nil
}

// UpdateEpilogueStatus updates status, LandedSHA, LandedAt, and Error fields atomically.
func UpdateEpilogueStatus(runsDir, id string, status EpilogueStatus, landedSHA string, errDetail string) (EpilogueRecord, error) {
	dir := epiloguesDir(runsDir)
	targetPath := filepath.Join(dir, id+".json")

	data, err := os.ReadFile(targetPath)
	if err != nil {
		return EpilogueRecord{}, fmt.Errorf("read epilogue %s: %w", id, err)
	}
	var rec EpilogueRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return EpilogueRecord{}, fmt.Errorf("unmarshal epilogue %s: %w", id, err)
	}

	rec.Status = status
	if landedSHA != "" {
		rec.LandedSHA = landedSHA
	}
	if status == EpilogueStatusLanded && rec.LandedAt == nil {
		now := time.Now().UTC()
		rec.LandedAt = &now
	}
	rec.Error = errDetail

	outData, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return rec, fmt.Errorf("marshal epilogue %s: %w", id, err)
	}
	outData = append(outData, '\n')

	tmpFile, err := os.CreateTemp(dir, id+"-update-*.tmp")
	if err != nil {
		return rec, fmt.Errorf("create temp file for epilogue update: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		_ = os.Remove(tmpName)
	}()

	if _, err := tmpFile.Write(outData); err != nil {
		_ = tmpFile.Close()
		return rec, fmt.Errorf("write temp epilogue update: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return rec, fmt.Errorf("close temp epilogue update: %w", err)
	}

	if err := os.Rename(tmpName, targetPath); err != nil {
		_ = os.Remove(targetPath)
		if err2 := os.Rename(tmpName, targetPath); err2 != nil {
			return rec, fmt.Errorf("atomic rename epilogue update: %w", err2)
		}
	}

	return rec, nil
}

func defaultGitRunner(ctx context.Context, dir string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=fak-epilogue",
		"GIT_AUTHOR_EMAIL=fak@localhost",
		"GIT_COMMITTER_NAME=fak-epilogue",
		"GIT_COMMITTER_EMAIL=fak@localhost",
	)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// DrainEpilogues sequentially processes pending epilogues, applies diffs/patches, verifies path
// disjointness, commits landed changes, and pushes if requested.
func DrainEpilogues(root, runsDir string, opts EpilogueDrainOptions) (EpilogueDrainResult, error) {
	git := opts.Git
	if git == nil {
		git = defaultGitRunner
	}
	ctx := context.Background()

	var result EpilogueDrainResult
	pending, err := ListEpilogues(runsDir, EpilogueStatusPending)
	if err != nil {
		return result, fmt.Errorf("list pending epilogues: %w", err)
	}
	result.Total = len(pending)
	if len(pending) == 0 {
		return result, nil
	}

	if opts.Limit > 0 && len(pending) > opts.Limit {
		pending = pending[:opts.Limit]
	}

	if opts.DryRun {
		result.Records = pending
		return result, nil
	}

	conflictedPaths := make(map[string]bool)

	for _, rec := range pending {
		// Check path conflict against failed/conflicted paths in this drain run
		hasConflict := false
		for _, p := range rec.Paths {
			if conflictedPaths[p] {
				hasConflict = true
				break
			}
		}
		if hasConflict {
			for _, p := range rec.Paths {
				conflictedPaths[p] = true
			}
			updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusConflict, "", "path conflict: overlaps with prior conflicting epilogue")
			result.Conflicted++
			result.Records = append(result.Records, updated)
			continue
		}

		// Mark landing
		_, _ = UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusLanding, "", "")

		// Apply patch or diff
		applied := false
		if rec.Patch != "" {
			tmpPatch, err := os.CreateTemp("", "epilogue-patch-*.diff")
			if err != nil {
				updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("create temp patch: %v", err))
				result.Failed++
				result.Records = append(result.Records, updated)
				continue
			}
			patchPath := tmpPatch.Name()
			_, _ = tmpPatch.WriteString(rec.Patch)
			_ = tmpPatch.Close()

			out, err := git(ctx, root, "apply", "--whitespace=nowarn", patchPath)
			_ = os.Remove(patchPath)
			if err != nil {
				// git apply failed => mark conflict
				for _, p := range rec.Paths {
					conflictedPaths[p] = true
				}
				if len(rec.Paths) > 0 {
					_, _ = git(ctx, root, append([]string{"checkout", "--"}, rec.Paths...)...)
				}
				updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusConflict, "", fmt.Sprintf("git apply failed: %v: %s", err, strings.TrimSpace(out)))
				result.Conflicted++
				result.Records = append(result.Records, updated)
				continue
			}
			applied = true
		} else if rec.WorktreeDir != "" {
			copyErr := false
			for _, p := range rec.Paths {
				src := filepath.Join(rec.WorktreeDir, p)
				dst := filepath.Join(root, p)
				content, err := os.ReadFile(src)
				if err != nil {
					copyErr = true
					updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("read worktree path %s: %v", p, err))
					result.Failed++
					result.Records = append(result.Records, updated)
					break
				}
				if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
					copyErr = true
					updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("mkdir for %s: %v", dst, err))
					result.Failed++
					result.Records = append(result.Records, updated)
					break
				}
				if err := os.WriteFile(dst, content, 0o644); err != nil {
					copyErr = true
					updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("write path %s: %v", dst, err))
					result.Failed++
					result.Records = append(result.Records, updated)
					break
				}
			}
			if copyErr {
				continue
			}
			applied = true
		} else {
			// No patch and no worktree dir; paths in root might already be modified
			applied = true
		}

		if !applied {
			updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", "no patch or worktree changes available")
			result.Failed++
			result.Records = append(result.Records, updated)
			continue
		}

		// Stage paths
		if len(rec.Paths) > 0 {
			addArgs := append([]string{"add", "--"}, rec.Paths...)
			if out, err := git(ctx, root, addArgs...); err != nil {
				updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("git add failed: %v: %s", err, strings.TrimSpace(out)))
				result.Failed++
				result.Records = append(result.Records, updated)
				continue
			}
		}

		// Commit with author message and DCO sign-off
		commitMsg := strings.TrimSpace(rec.Message)
		if commitMsg == "" {
			commitMsg = fmt.Sprintf("resolve(#%d): drain epilogue %s", rec.Issue, rec.ID)
		}
		if !strings.Contains(commitMsg, "Signed-off-by:") {
			commitMsg += "\n\nSigned-off-by: fak <fak@localhost>"
		}

		commitArgs := []string{"commit", "-m", commitMsg}
		if out, err := git(ctx, root, commitArgs...); err != nil {
			for _, p := range rec.Paths {
				conflictedPaths[p] = true
			}
			updated, _ := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusFailed, "", fmt.Sprintf("git commit failed: %v: %s", err, strings.TrimSpace(out)))
			result.Failed++
			result.Records = append(result.Records, updated)
			continue
		}

		revOut, err := git(ctx, root, "rev-parse", "HEAD")
		landedSHA := strings.TrimSpace(revOut)
		if err != nil || landedSHA == "" {
			landedSHA = "unknown"
		}

		updated, err := UpdateEpilogueStatus(runsDir, rec.ID, EpilogueStatusLanded, landedSHA, "")
		if err != nil {
			result.Failed++
			continue
		}
		result.Landed++
		result.Records = append(result.Records, updated)
	}

	if opts.Push && result.Landed > 0 {
		if out, err := git(ctx, root, "push"); err != nil {
			return result, fmt.Errorf("git push failed: %v: %s", err, strings.TrimSpace(out))
		}
	}

	return result, nil
}
