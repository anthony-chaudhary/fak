// cdmap.go — the drive-stripped-workspace `cd` rung of the repo-guard
// PreToolUse hook (worker path-mapping; driver: the 2026-07-09 trajectory audit).
//
// A headless worker's shell already starts IN the workspace (the launcher sets
// cmd.Dir to the workspace root), so a leading `cd` into it is redundant. But a
// model reconstructing that absolute path — from the project slug (C--work-fak)
// or from a half-remembered C:/work/fak — sometimes drops the Windows drive
// letter and emits `cd /work/fak`. On the git-bash / MSYS host that produced it
// the checkout lives at /c/work/fak, so the drive-less /work/fak resolves
// NOWHERE: the cd fails with "No such file or directory" and, chained under
// `&&`, aborts the whole command — a pure wasted turn. It was one of the top
// recurring failure CLASSES in the 2026-07-09 trajectory audit (session
// d5b3ed57, 3×: `cd: /work/fak: No such file or directory`).
//
// Curation principle: fire ONLY when the target is the workspace root with its
// drive letter removed — a form that provably cannot resolve on the
// drive-lettered host that emitted it, and that always means the same mistake.
// Deliberately NOT flagged: a drive-ful path (C:/work/fak), the MSYS form
// (/c/work/fak, which normalize() folds back to the workspace), any in-tree or
// relative cd, and EVERY cd on a drive-less real-POSIX host (where a leading
// /work/... is a legitimate absolute path). Pure: no filesystem access,
// hermetically testable like the rest of the core.
package repoguard

import "strings"

// ReasonWorkspacePathUnmapped is the structured advisory token for a `cd` into
// the workspace root with its Windows drive letter dropped.
const ReasonWorkspacePathUnmapped = "WORKSPACE_PATH_UNMAPPED"

// ClassifyWorkspaceCd returns WORKSPACE_PATH_UNMAPPED advisories for a shell
// command whose `cd` targets the drive-stripped workspace root.
func ClassifyWorkspaceCd(command, workspaceRoot string) []Violation {
	return classifyWorkspaceCd(command, workspaceRoot)
}

// classifyWorkspaceCd flags a `cd` whose sole operand is the workspace root with
// the leading drive prefix removed (`cd /work/fak` for a C:/work/fak workspace).
// Pure string work.
func classifyWorkspaceCd(command, workspaceRoot string) []Violation {
	stripped, ok := driveStrippedRoot(workspaceRoot)
	if !ok {
		return nil // drive-less workspace (real-POSIX host): a leading /path cd is legit
	}
	var out []Violation
	for _, seg := range splitSegments(command) {
		verb, operands, _ := tokenizeSegment(seg)
		if verb != "cd" || len(operands) != 1 {
			continue // only a bare `cd <one-path>` is provably this mistake
		}
		if trimSlash(normalize(operands[0])) != stripped {
			continue
		}
		out = append(out, Violation{
			Reason:   ReasonWorkspacePathUnmapped,
			Op:       "cd " + operands[0],
			Target:   strings.TrimSpace(seg),
			Resolved: stripped,
			Why:      "the workspace path dropped its drive letter — " + stripped + " resolves nowhere on this drive-lettered host",
			Fix:      "the shell already starts in the workspace, so drop the cd; or use the host path: cd '" + normalize(workspaceRoot) + "'",
		})
	}
	return out
}

// driveStrippedRoot returns the normalized workspace root with a leading `X:`
// drive prefix removed (C:/work/fak -> /work/fak). ok is false when the root
// carries no drive letter (a real-POSIX workspace) or strips down to the bare
// filesystem root — in both cases a leading-slash cd is not the drive-drop
// mistake and must never be flagged.
func driveStrippedRoot(workspaceRoot string) (string, bool) {
	ws := trimSlash(normalize(workspaceRoot))
	if len(ws) >= 2 && ws[1] == ':' && isAlpha(ws[0]) {
		stripped := ws[2:] // "C:/work/fak" -> "/work/fak"
		if stripped == "" || stripped == "/" {
			return "", false // "C:" / "C:/" — too broad to match a real workspace cd
		}
		return stripped, true
	}
	return "", false
}

// trimSlash drops a single trailing slash so `/work/fak/` and `/work/fak`
// compare equal (but keeps a bare "/").
func trimSlash(p string) string {
	if len(p) > 1 {
		return strings.TrimRight(p, "/")
	}
	return p
}

func renderWorkspaceCdReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Target + " — fix: " + v.Fix
	}
	return ReasonWorkspacePathUnmapped + ": this cd targets the workspace with its drive letter dropped, so it resolves nowhere on this host — the command (and any && chain after it) fails as a wasted turn. " +
		strings.Join(parts, "; ") + ". " +
		"To silence this per reason set FAK_REPO_GUARD_SEVERITY=" + ReasonWorkspacePathUnmapped + "=record or =off; " +
		"the master switch FAK_REPO_GUARD=warn|off still overrides."
}
