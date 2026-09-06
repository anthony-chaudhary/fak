package codetools

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// patch.go — unified diff application with path confinement, fuzz tolerance,
// and atomic optimistic CAS replacement.

var hunkHeaderRE = regexp.MustCompile(`^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@`)

// PatchArgs describes arguments to the apply_patch tool.
type PatchArgs struct {
	Patch           string `json:"patch"`
	ExpectedVersion string `json:"expected_version,omitempty"`
	FuzzMargin      int    `json:"fuzz_margin,omitempty"`
	Fuzz            int    `json:"fuzz,omitempty"` // backward-compatible alias
}

// ApplyPatchArgs is an alias for PatchArgs.
type ApplyPatchArgs = PatchArgs

func (a PatchArgs) Validate() *Refusal {
	if strings.TrimSpace(a.Patch) == "" {
		return refuse(CodeMalformed, "apply_patch: missing required field: patch")
	}
	if a.FuzzMargin < 0 || a.FuzzMargin > 5 {
		return refuse(CodeMalformed, "apply_patch: fuzz_margin must be between 0 and 5")
	}
	if a.Fuzz < 0 || a.Fuzz > 5 {
		return refuse(CodeMalformed, "apply_patch: fuzz must be between 0 and 5")
	}
	return nil
}

func (a PatchArgs) effectiveFuzz(body []byte) int {
	if a.FuzzMargin > 0 {
		return a.FuzzMargin
	}
	if a.Fuzz > 0 {
		return a.Fuzz
	}
	if bytes.Contains(body, []byte(`"fuzz_margin"`)) || bytes.Contains(body, []byte(`"fuzz"`)) {
		if a.FuzzMargin == 0 && bytes.Contains(body, []byte(`"fuzz_margin"`)) {
			return 0
		}
		if a.Fuzz == 0 && bytes.Contains(body, []byte(`"fuzz"`)) {
			return 0
		}
	}
	return 2
}

func matchExpectedVersion(expected, actualVersion string, content []byte) bool {
	if expected == "" {
		return true
	}
	if expected == actualVersion {
		return true
	}
	if strings.TrimPrefix(actualVersion, "fv1:") == strings.TrimPrefix(expected, "fv1:") {
		return true
	}
	contentHash := sha256.Sum256(content)
	contentHex := hex.EncodeToString(contentHash[:])
	if strings.EqualFold(expected, contentHex) || strings.EqualFold(strings.TrimPrefix(expected, "sha256:"), contentHex) {
		return true
	}
	return false
}

type hunkLine struct {
	Type    byte   // ' ', '-', '+'
	Content string // line content without marker
}

type diffHunk struct {
	OldStart int
	OldCount int
	NewStart int
	NewCount int
	Lines    []hunkLine
}

type filePatch struct {
	OldPath string
	NewPath string
	Action  string // "create", "delete", "modify"
	Target  string
	Hunks   []diffHunk
}

// PatchFileSummary reports the outcome for one file touched by apply_patch.
type PatchFileSummary struct {
	Path    string `json:"path"`
	Action  string `json:"action"` // "created", "deleted", "modified"
	Bytes   int    `json:"bytes"`
	Version string `json:"version,omitempty"`
}

// PatchResult is the success envelope emitted by apply_patch.
type PatchResult struct {
	FilesModified []string           `json:"files_modified"`
	FilesCreated  []string           `json:"files_created"`
	FilesDeleted  []string           `json:"files_deleted"`
	HunksApplied  int                `json:"hunks_applied"`
	Path          string             `json:"path,omitempty"`
	Action        string             `json:"action,omitempty"`
	Bytes         int                `json:"bytes,omitempty"`
	Version       string             `json:"version,omitempty"`
	Files         []PatchFileSummary `json:"files,omitempty"`
}

type plannedPatchFile struct {
	target   mutationTarget
	action   string
	content  []byte
	perm     fs.FileMode
	observed fileObservation
}

func (t *Toolset) resolvePatchTarget(target string) (mutationTarget, *Refusal) {
	target = strings.TrimSpace(target)
	if target == "" {
		return mutationTarget{}, refuse(CodeMalformed, "empty patch target path")
	}
	// On Windows, paths starting with / or \ without volume are drive-root relative (e.g. \etc\passwd).
	// Qualify them with the workspace root's volume so confinement reliably detects the escape.
	if (strings.HasPrefix(target, "/") || strings.HasPrefix(target, "\\")) && filepath.VolumeName(target) == "" {
		target = filepath.VolumeName(t.root) + target
	}
	return t.resolveMutation(target)
}

func (t *Toolset) applyPatch(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	var a PatchArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if int64(len(a.Patch)) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Patch exceeds byte bound").JSON(), true
	}

	fuzz := a.effectiveFuzz(body)

	patches, r := parseUnifiedDiff(a.Patch)
	if r != nil {
		return r.JSON(), true
	}

	// Phase 1: Resolve and confine all targets.
	initialTargets := make([]mutationTarget, len(patches))
	for i, fp := range patches {
		target, ref := t.resolvePatchTarget(fp.Target)
		if ref != nil {
			return ref.JSON(), true
		}
		initialTargets[i] = target
	}

	// Phase 2: Compute planned changes in-memory (including hunk matching and CAS checks).
	planned := make([]plannedPatchFile, len(patches))
	for i, fp := range patches {
		target := initialTargets[i]
		info, statErr := os.Lstat(target.Abs)
		exists := statErr == nil
		if statErr != nil && !errors.Is(statErr, fs.ErrNotExist) {
			return refuse(CodeIO, statErr.Error()).JSON(), true
		}
		if exists && info.IsDir() {
			return refuse(CodeIsDir, "patch target is a directory: "+target.Rel).JSON(), true
		}

		switch fp.Action {
		case "create":
			if a.ExpectedVersion != "" {
				return staleVersion("create target has no prior version")
			}
			if exists {
				return refuse(CodeExists, "target file already exists: "+target.Rel).JSON(), true
			}
			newLines, err := applyHunks([]string{}, fp.Hunks, fuzz)
			if err != nil {
				return refuse(CodeEditConflict, err.Error()).JSON(), true
			}
			newContent := joinLines(newLines, "\n", len(newLines) > 0)
			if int64(len(newContent)) > t.limits.MaxWriteBytes {
				return refuse(CodeTooLarge, "Patch result exceeds byte bound").JSON(), true
			}
			planned[i] = plannedPatchFile{
				target:  target,
				action:  "create",
				content: []byte(newContent),
				perm:    0o644,
			}

		case "delete":
			if !exists {
				if a.ExpectedVersion != "" {
					return staleVersion("target file no longer exists")
				}
				return refuse(CodeNotFound, "target file does not exist: "+target.Rel).JSON(), true
			}
			obs, ref := observeFile(ctx, target.Abs, t.limits.MaxWriteBytes)
			if ref != nil {
				return ref.JSON(), true
			}
			if obs.Truncated {
				return refuse(CodeTooLarge, "Patch target exceeds byte bound").JSON(), true
			}
			if a.ExpectedVersion != "" && !matchExpectedVersion(a.ExpectedVersion, obs.Version, obs.Content) {
				return staleVersion("patch target changed since it was read")
			}
			origLines, _ := splitLines(string(obs.Content))
			if len(fp.Hunks) > 0 {
				newLines, err := applyHunks(origLines, fp.Hunks, fuzz)
				if err != nil {
					return refuse(CodeEditConflict, err.Error()).JSON(), true
				}
				if len(newLines) != 0 {
					return refuse(CodeEditConflict, "delete patch hunks did not remove all content").JSON(), true
				}
			}
			planned[i] = plannedPatchFile{
				target:   target,
				action:   "delete",
				observed: obs,
				perm:     info.Mode().Perm(),
			}

		case "modify":
			if !exists {
				if a.ExpectedVersion != "" {
					return staleVersion("target file no longer exists")
				}
				return refuse(CodeNotFound, "target file does not exist: "+target.Rel).JSON(), true
			}
			obs, ref := observeFile(ctx, target.Abs, t.limits.MaxWriteBytes)
			if ref != nil {
				return ref.JSON(), true
			}
			if obs.Truncated {
				return refuse(CodeTooLarge, "Patch target exceeds byte bound").JSON(), true
			}
			if a.ExpectedVersion != "" && !matchExpectedVersion(a.ExpectedVersion, obs.Version, obs.Content) {
				return staleVersion("patch target changed since it was read")
			}
			eol := detectLineEnding(obs.Content)
			origLines, hasTrailingNewline := splitLines(string(obs.Content))
			newLines, err := applyHunks(origLines, fp.Hunks, fuzz)
			if err != nil {
				return refuse(CodeEditConflict, err.Error()).JSON(), true
			}
			newContent := joinLines(newLines, eol, hasTrailingNewline)
			if int64(len(newContent)) > t.limits.MaxWriteBytes {
				return refuse(CodeTooLarge, "Patch result exceeds byte bound").JSON(), true
			}
			planned[i] = plannedPatchFile{
				target:   target,
				action:   "modify",
				content:  []byte(newContent),
				perm:     info.Mode().Perm(),
				observed: obs,
			}
		}
	}

	// Phase 3: Acquire mutation locks for all target keys in deterministic order.
	lockKeys := make([]string, len(planned))
	for i, p := range planned {
		lockKeys[i] = p.target.Key
	}

	return t.withMutationLocks(lockKeys, func() ([]byte, bool) {
		if r := canceled(ctx); r != nil {
			return r.JSON(), true
		}

		// Re-verify targets under lock to detect identity shifts or concurrent edits.
		freshTargets := make([]mutationTarget, len(planned))
		for i, p := range planned {
			fresh, ref := t.resolvePatchTarget(patches[i].Target)
			if ref != nil {
				return ref.JSON(), true
			}
			if fresh.Key != p.target.Key {
				return staleVersion("patch target identity changed before publication")
			}
			freshTargets[i] = fresh

			switch p.action {
			case "create":
				if _, err := os.Lstat(fresh.Abs); err == nil {
					return refuse(CodeExists, "patch create target already exists").JSON(), true
				}
			case "modify", "delete":
				curr, ref := observeFile(ctx, fresh.Abs, 0)
				if ref != nil {
					return ref.JSON(), true
				}
				if curr.Version != p.observed.Version {
					return staleVersion("patch target changed before publication")
				}
			}
		}

		// Phase 4: Atomic application with rollback.
		// Step 4a: Write temporary files (.fak-write-*) for all modified/created files.
		// If writing/syncing any temp file fails, abort before renaming any files.
		type preparedTemp struct {
			tmpPath string
			target  string
			action  string
			perm    fs.FileMode
		}
		prepared := make([]preparedTemp, len(planned))
		cleanupTemps := func() {
			for _, pt := range prepared {
				if pt.tmpPath != "" {
					_ = os.Remove(pt.tmpPath)
				}
			}
		}

		for i, p := range planned {
			fresh := freshTargets[i]
			if p.action == "create" || p.action == "modify" {
				dir := filepath.Dir(fresh.Abs)
				if err := os.MkdirAll(dir, 0o755); err != nil {
					cleanupTemps()
					return refuse(CodeIO, err.Error()).JSON(), true
				}
				f, err := os.CreateTemp(dir, ".fak-write-*")
				if err != nil {
					cleanupTemps()
					return refuse(CodeIO, err.Error()).JSON(), true
				}
				tmpName := f.Name()
				prepared[i] = preparedTemp{
					tmpPath: tmpName,
					target:  fresh.Abs,
					action:  p.action,
					perm:    p.perm,
				}
				if p.action == "create" {
					_ = f.Chmod(0o644)
				} else if p.perm != 0 {
					_ = f.Chmod(p.perm)
				}
				if _, err = f.Write(p.content); err == nil {
					err = f.Sync()
				}
				closeErr := f.Close()
				if err == nil {
					err = closeErr
				}
				if err != nil {
					cleanupTemps()
					return refuse(CodeIO, err.Error()).JSON(), true
				}
			} else {
				prepared[i] = preparedTemp{
					target: fresh.Abs,
					action: p.action,
				}
			}
		}

		// Step 4b: Apply renames / removals with rollback on failure.
		type rollbackAction struct {
			path    string
			action  string // "delete_created", "restore_file"
			content []byte
			perm    fs.FileMode
		}
		var rollbacks []rollbackAction
		var commitErr error

		for i, pt := range prepared {
			fresh := freshTargets[i]
			switch pt.action {
			case "create":
				if err := os.Rename(pt.tmpPath, fresh.Abs); err != nil {
					commitErr = err
					break
				}
				prepared[i].tmpPath = "" // safely renamed
				rollbacks = append(rollbacks, rollbackAction{
					path:   fresh.Abs,
					action: "delete_created",
				})

			case "modify":
				origContent := planned[i].observed.Content
				origPerm := planned[i].perm
				if err := os.Rename(pt.tmpPath, fresh.Abs); err != nil {
					commitErr = err
					break
				}
				prepared[i].tmpPath = "" // safely renamed
				rollbacks = append(rollbacks, rollbackAction{
					path:    fresh.Abs,
					action:  "restore_file",
					content: origContent,
					perm:    origPerm,
				})

			case "delete":
				origContent := planned[i].observed.Content
				origPerm := planned[i].perm
				if err := os.Remove(fresh.Abs); err != nil && !errors.Is(err, fs.ErrNotExist) {
					commitErr = err
					break
				}
				rollbacks = append(rollbacks, rollbackAction{
					path:    fresh.Abs,
					action:  "restore_file",
					content: origContent,
					perm:    origPerm,
				})
			}
			if commitErr != nil {
				break
			}
		}

		if commitErr != nil {
			// Roll back all already-applied changes in reverse order
			for j := len(rollbacks) - 1; j >= 0; j-- {
				rb := rollbacks[j]
				switch rb.action {
				case "delete_created":
					_ = os.Remove(rb.path)
				case "restore_file":
					_ = os.WriteFile(rb.path, rb.content, rb.perm)
				}
			}
			cleanupTemps()
			return refuse(CodeIO, commitErr.Error()).JSON(), true
		}

		// Step 4c: Record results and observed versions
		fileSummaries := make([]PatchFileSummary, len(planned))
		filesModified := make([]string, 0)
		filesCreated := make([]string, 0)
		filesDeleted := make([]string, 0)

		for i, p := range planned {
			fresh := freshTargets[i]
			switch p.action {
			case "create":
				after, ref := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
				version := ""
				if ref == nil {
					version = after.Version
				}
				filesCreated = append(filesCreated, fresh.Rel)
				fileSummaries[i] = PatchFileSummary{
					Path:    fresh.Rel,
					Action:  "created",
					Bytes:   len(p.content),
					Version: version,
				}

			case "modify":
				after, ref := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
				version := ""
				if ref == nil {
					version = after.Version
				}
				filesModified = append(filesModified, fresh.Rel)
				fileSummaries[i] = PatchFileSummary{
					Path:    fresh.Rel,
					Action:  "modified",
					Bytes:   len(p.content),
					Version: version,
				}

			case "delete":
				filesDeleted = append(filesDeleted, fresh.Rel)
				fileSummaries[i] = PatchFileSummary{
					Path:   fresh.Rel,
					Action: "deleted",
					Bytes:  0,
				}
			}
		}

		hunksApplied := 0
		for _, fp := range patches {
			hunksApplied += len(fp.Hunks)
		}

		res := PatchResult{
			FilesModified: filesModified,
			FilesCreated:  filesCreated,
			FilesDeleted:  filesDeleted,
			HunksApplied:  hunksApplied,
			Files:         fileSummaries,
		}
		if len(fileSummaries) == 1 {
			res.Path = fileSummaries[0].Path
			res.Action = fileSummaries[0].Action
			res.Bytes = fileSummaries[0].Bytes
			res.Version = fileSummaries[0].Version
		} else if len(fileSummaries) > 1 {
			res.Version = fileSummaries[len(fileSummaries)-1].Version
		}

		return okJSON(res), false
	})
}

func (t *Toolset) withMutationLocks(keys []string, fn func() ([]byte, bool)) ([]byte, bool) {
	if len(keys) == 0 {
		return fn()
	}
	unique := make([]string, 0, len(keys))
	seen := make(map[string]bool, len(keys))
	for _, k := range keys {
		if !seen[k] {
			seen[k] = true
			unique = append(unique, k)
		}
	}
	sort.Strings(unique)

	var acquire func(int) ([]byte, bool)
	acquire = func(i int) ([]byte, bool) {
		if i >= len(unique) {
			return fn()
		}
		return t.withMutationLock(unique[i], func() ([]byte, bool) {
			return acquire(i + 1)
		})
	}
	return acquire(0)
}

func parseUnifiedDiff(patch string) ([]filePatch, *Refusal) {
	patch = strings.ReplaceAll(patch, "\r\n", "\n")
	patch = strings.TrimRight(patch, "\n")
	if strings.TrimSpace(patch) == "" {
		return nil, refuse(CodeMalformed, "patch is empty")
	}
	rawLines := strings.Split(patch, "\n")

	var files []filePatch
	var curFile *filePatch
	var curHunk *diffHunk

	flushHunk := func() {
		if curHunk != nil && curFile != nil {
			curFile.Hunks = append(curFile.Hunks, *curHunk)
			curHunk = nil
		}
	}

	flushFile := func() {
		flushHunk()
		if curFile != nil {
			files = append(files, *curFile)
			curFile = nil
		}
	}

	for i := 0; i < len(rawLines); i++ {
		line := rawLines[i]

		// Check for file header: --- <oldPath> followed by +++ <newPath>
		if strings.HasPrefix(line, "--- ") && i+1 < len(rawLines) && strings.HasPrefix(rawLines[i+1], "+++ ") {
			flushFile()
			oldPath, _ := parseHeaderPath(line, "--- ")
			newPath, _ := parseHeaderPath(rawLines[i+1], "+++ ")
			i++ // skip +++ line

			var action string
			var target string
			if oldPath == "/dev/null" {
				action = "create"
				target = newPath
			} else if newPath == "/dev/null" {
				action = "delete"
				target = oldPath
			} else {
				action = "modify"
				target = newPath
			}
			if target == "/dev/null" || strings.TrimSpace(target) == "" {
				return nil, refuse(CodeMalformed, "patch has invalid target path")
			}

			curFile = &filePatch{
				OldPath: oldPath,
				NewPath: newPath,
				Action:  action,
				Target:  target,
			}
			continue
		}

		// Check for hunk header: @@ -l,s +l,s @@ [optional heading]
		if strings.HasPrefix(line, "@@") {
			m := hunkHeaderRE.FindStringSubmatch(line)
			if m == nil {
				return nil, refuse(CodeMalformed, "malformed hunk header: "+line)
			}
			if curFile == nil {
				return nil, refuse(CodeMalformed, "hunk header without preceding file header: "+line)
			}
			flushHunk()

			oldStart, _ := strconv.Atoi(m[1])
			oldCount := 1
			if m[2] != "" {
				oldCount, _ = strconv.Atoi(m[2])
			}
			newStart, _ := strconv.Atoi(m[3])
			newCount := 1
			if m[4] != "" {
				newCount, _ = strconv.Atoi(m[4])
			}

			curHunk = &diffHunk{
				OldStart: oldStart,
				OldCount: oldCount,
				NewStart: newStart,
				NewCount: newCount,
			}
			continue
		}

		// Inside a hunk
		if curHunk != nil {
			if strings.HasPrefix(line, "\\") {
				// e.g. \ No newline at end of file
				continue
			}
			if len(line) == 0 {
				curHunk.Lines = append(curHunk.Lines, hunkLine{Type: ' ', Content: ""})
				continue
			}
			marker := line[0]
			if marker == ' ' || marker == '-' || marker == '+' {
				curHunk.Lines = append(curHunk.Lines, hunkLine{Type: marker, Content: line[1:]})
				continue
			}
			flushHunk()
		}
	}

	flushFile()

	if len(files) == 0 {
		return nil, refuse(CodeMalformed, "patch contains no valid file hunks")
	}
	return files, nil
}

func parseHeaderPath(line, prefix string) (string, bool) {
	if !strings.HasPrefix(line, prefix) {
		return "", false
	}
	p := strings.TrimPrefix(line, prefix)
	if idx := strings.IndexByte(p, '\t'); idx >= 0 {
		p = p[:idx]
	}
	p = strings.TrimSpace(p)
	if len(p) >= 2 && p[0] == '"' && p[len(p)-1] == '"' {
		p = p[1 : len(p)-1]
	}
	if p == "/dev/null" || p == "dev/null" {
		return "/dev/null", true
	}
	if strings.HasPrefix(p, "a/") || strings.HasPrefix(p, "b/") {
		p = p[2:]
	}
	return filepath.ToSlash(filepath.Clean(p)), true
}

func applyHunks(originalLines []string, hunks []diffHunk, fuzz int) ([]string, error) {
	result := make([]string, 0, len(originalLines))
	fileIdx := 0

	for hIdx, hunk := range hunks {
		var oldLines []string
		for _, hl := range hunk.Lines {
			switch hl.Type {
			case ' ', '-':
				oldLines = append(oldLines, hl.Content)
			}
		}

		expectedIdx := hunk.OldStart - 1
		if expectedIdx < 0 {
			expectedIdx = 0
		}
		if expectedIdx < fileIdx {
			expectedIdx = fileIdx
		}

		if len(oldLines) == 0 {
			// Pure insertion
			matchIdx := expectedIdx
			if matchIdx < fileIdx {
				matchIdx = fileIdx
			}
			if matchIdx > len(originalLines) {
				matchIdx = len(originalLines)
			}
			result = append(result, originalLines[fileIdx:matchIdx]...)
			for _, hl := range hunk.Lines {
				if hl.Type == '+' {
					result = append(result, hl.Content)
				}
			}
			fileIdx = matchIdx
			continue
		}

		// Find match with offset drift up to fuzz lines.
		// Check exact match first (d=0), then +/-1, +/-2, ..., up to +/-fuzz.
		matchIdx := -1
		for d := 0; d <= fuzz; d++ {
			candPos := expectedIdx + d
			if candPos >= fileIdx && candPos+len(oldLines) <= len(originalLines) {
				if linesMatch(originalLines[candPos:candPos+len(oldLines)], oldLines) {
					matchIdx = candPos
					break
				}
			}
			if d > 0 {
				candNeg := expectedIdx - d
				if candNeg >= fileIdx && candNeg+len(oldLines) <= len(originalLines) {
					if linesMatch(originalLines[candNeg:candNeg+len(oldLines)], oldLines) {
						matchIdx = candNeg
						break
					}
				}
			}
		}

		if matchIdx < 0 {
			return nil, fmt.Errorf("hunk %d (line %d) failed to match context within fuzz %d", hIdx+1, hunk.OldStart, fuzz)
		}

		// Append unchanged lines before hunk match
		result = append(result, originalLines[fileIdx:matchIdx]...)

		// Apply hunk lines preserving context lines from original
		currFileIdx := matchIdx
		for _, hl := range hunk.Lines {
			switch hl.Type {
			case ' ':
				if currFileIdx < len(originalLines) {
					result = append(result, originalLines[currFileIdx])
					currFileIdx++
				}
			case '-':
				currFileIdx++
			case '+':
				result = append(result, hl.Content)
			}
		}

		fileIdx = matchIdx + len(oldLines)
	}

	result = append(result, originalLines[fileIdx:]...)
	return result, nil
}

func linesMatch(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] == b[i] {
			continue
		}
		// Trailing whitespace tolerance
		if strings.TrimRight(a[i], " \t\r") != strings.TrimRight(b[i], " \t\r") {
			return false
		}
	}
	return true
}

func detectLineEnding(b []byte) string {
	if bytes.Contains(b, []byte("\r\n")) {
		return "\r\n"
	}
	return "\n"
}

func splitLines(content string) ([]string, bool) {
	if len(content) == 0 {
		return []string{}, false
	}
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	content = strings.ReplaceAll(content, "\r\n", "\n")
	if hasTrailingNewline {
		content = strings.TrimSuffix(content, "\n")
	}
	lines := strings.Split(content, "\n")
	return lines, hasTrailingNewline
}

func joinLines(lines []string, eol string, hasTrailingNewline bool) string {
	if len(lines) == 0 {
		return ""
	}
	res := strings.Join(lines, eol)
	if hasTrailingNewline {
		res += eol
	}
	return res
}
