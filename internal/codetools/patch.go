package codetools

import (
	"bytes"
	"context"
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
	Path    string             `json:"path,omitempty"`
	Action  string             `json:"action,omitempty"`
	Bytes   int                `json:"bytes,omitempty"`
	Version string             `json:"version,omitempty"`
	Files   []PatchFileSummary `json:"files"`
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
	var a ApplyPatchArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if int64(len(a.Patch)) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Patch exceeds byte bound").JSON(), true
	}

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
			newLines, err := applyHunks([]string{}, fp.Hunks, a.Fuzz)
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
			if a.ExpectedVersion != "" && obs.Version != a.ExpectedVersion {
				return staleVersion("patch target changed since it was read")
			}
			origLines, _ := splitLines(string(obs.Content))
			if len(fp.Hunks) > 0 {
				newLines, err := applyHunks(origLines, fp.Hunks, a.Fuzz)
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
			if a.ExpectedVersion != "" && obs.Version != a.ExpectedVersion {
				return staleVersion("patch target changed since it was read")
			}
			eol := detectLineEnding(obs.Content)
			origLines, hasTrailingNewline := splitLines(string(obs.Content))
			newLines, err := applyHunks(origLines, fp.Hunks, a.Fuzz)
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

		// Phase 4: Commit changes atomically.
		fileSummaries := make([]PatchFileSummary, len(planned))
		for i, p := range planned {
			fresh := freshTargets[i]
			switch p.action {
			case "create":
				if err := atomicReplace(fresh.Abs, p.content, false, 0o644); err != nil {
					return refuse(CodeIO, err.Error()).JSON(), true
				}
				after, ref := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
				if ref != nil {
					return ref.JSON(), true
				}
				fileSummaries[i] = PatchFileSummary{
					Path:    fresh.Rel,
					Action:  "created",
					Bytes:   len(p.content),
					Version: after.Version,
				}

			case "modify":
				if err := atomicReplace(fresh.Abs, p.content, true, p.perm); err != nil {
					return refuse(CodeIO, err.Error()).JSON(), true
				}
				after, ref := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
				if ref != nil {
					return ref.JSON(), true
				}
				fileSummaries[i] = PatchFileSummary{
					Path:    fresh.Rel,
					Action:  "modified",
					Bytes:   len(p.content),
					Version: after.Version,
				}

			case "delete":
				if err := os.Remove(fresh.Abs); err != nil && !errors.Is(err, fs.ErrNotExist) {
					return refuse(CodeIO, err.Error()).JSON(), true
				}
				fileSummaries[i] = PatchFileSummary{
					Path:   fresh.Rel,
					Action: "deleted",
					Bytes:  0,
				}
			}
		}

		res := PatchResult{
			Files: fileSummaries,
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

		// Check for hunk header: @@ -l,s +l,s @@
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
		var newLines []string
		for _, hl := range hunk.Lines {
			switch hl.Type {
			case ' ':
				oldLines = append(oldLines, hl.Content)
				newLines = append(newLines, hl.Content)
			case '-':
				oldLines = append(oldLines, hl.Content)
			case '+':
				newLines = append(newLines, hl.Content)
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
			result = append(result, newLines...)
			fileIdx = matchIdx
			continue
		}

		// Find match with offset drift up to fuzz lines
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

		result = append(result, originalLines[fileIdx:matchIdx]...)
		result = append(result, newLines...)
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
		if a[i] != b[i] {
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
