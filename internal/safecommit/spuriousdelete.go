package safecommit

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"
)

// ReasonSpuriousStagedDeletion is the stale-index guard's refusal token. It is emitted when a
// requested path is staged as a DELETION (index-vs-HEAD shows `D`) while an UNTRACKED copy of
// the SAME path sits in the working tree — the unmistakable shape of a stale/desynced index on
// a shared multi-session clone (a peer `git reset`/`git rm` left the deletion staged after the
// file was recreated, or a `git add` ran against a tree mid-rewrite).
//
// Committing such a path by pathspec would DELETE a file HEAD still carries, only for the
// untracked copy to resurrect it on the next `git add` — a churn commit whose `git show --stat`
// reports a deletion the author never intended, and a broken intermediate state on the trunk.
// It is the whole-file sibling of ReasonStaleBaseDeletion (which guards a CONTENT block); this
// one guards a spurious WHOLE-PATH deletion that the working tree itself contradicts.
//
// The token is part of the same closed Reason vocabulary as the rest of safecommit (see the
// const block in safecommit.go).
const ReasonSpuriousStagedDeletion = "SPURIOUS_STAGED_DELETION"

// spuriousDeleteEnvVar gates the guard, mirroring the FAK_STALE_BASE_GUARD knob family.
const spuriousDeleteEnvVar = "FAK_SPURIOUS_DELETE_GUARD"

// spuriousDeleteGuardMode reads FAK_SPURIOUS_DELETE_GUARD. Default (unset / unrecognized) is
// block — the safe posture on a shared no-amend trunk. `warn` records the would-be refusal in
// Detail and proceeds; `off` skips the guard entirely. It reuses the staleBaseMode vocabulary
// so the two stale-index guards share one block|warn|off shape.
func spuriousDeleteGuardMode() staleBaseMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(spuriousDeleteEnvVar))) {
	case "off", "0", "false":
		return staleBaseOff
	case "warn", "advisory":
		return staleBaseWarn
	default:
		return staleBaseBlock
	}
}

// checkSpuriousStagedDeletion is the whole-path stale-index guard (step 4c). It asks a single
// `git status --porcelain -- <paths>` and refuses if any requested path appears BOTH as a
// staged deletion (`D` in the index column) AND as an untracked entry (`??`) for the same path.
// That co-occurrence means the working tree still holds the file the index is about to delete:
// the deletion is index drift, not the author's intent (a genuine delete leaves no untracked
// twin). Reads no blob and touches no disk beyond the one porcelain call, so it is fully
// testable through the injected Runner.
//
// Returns (detail, fired):
//   - fired == false, detail == "": no requested path has the spurious shape, OR the guard
//     could not run (fail-open: status unreadable). CommitWith proceeds exactly as before.
//   - fired == true, detail != "": at least one path is a spurious staged deletion. The caller
//     refuses (block) or records the detail and proceeds (warn). detail names the path(s) and
//     the `git restore --staged` remedy.
func checkSpuriousStagedDeletion(ctx context.Context, run Runner, dir string, paths []string) (detail string, fired bool) {
	statusArgs := append([]string{"status", "--porcelain", "--"}, paths...)
	out, code, err := run(ctx, dir, statusArgs...)
	if err != nil || code != 0 {
		return "", false // cannot read status — fail-open, exactly as the other read probes do
	}

	requested := map[string]bool{}
	for _, p := range paths {
		requested[p] = true
	}

	stagedDelete := map[string]bool{}
	untracked := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		path, staged, untrack := parsePorcelainLine(line)
		if path == "" || !requested[path] {
			continue
		}
		if staged {
			stagedDelete[path] = true
		}
		if untrack {
			untracked[path] = true
		}
	}

	var spurious []string
	for p := range stagedDelete {
		if untracked[p] {
			spurious = append(spurious, p)
		}
	}
	if len(spurious) == 0 {
		return "", false
	}
	sort.Strings(spurious)
	return fmt.Sprintf(
		"path(s) staged as a deletion while an untracked copy still sits in the working tree — "+
			"the index is stale (a peer reset/rm left the deletion staged after the file was recreated). "+
			"Committing would delete a file HEAD still carries, only to resurrect it on the next add. "+
			"Fix: git restore --staged %s (the untracked copy stays; re-stage your real edits). Paths: %s",
		strings.Join(spurious, " "), strings.Join(spurious, ", "),
	), true
}

// parsePorcelainLine reads one `git status --porcelain` line and reports the path plus whether
// the line is a staged deletion (index column 'D') and whether it is an untracked entry ("??").
// The porcelain v1 format is two status columns (index, work-tree) followed by a space then the
// path; an untracked entry is the literal "?? <path>". A renamed entry ("R  old -> new") is not
// the shape we guard and yields path == "" so it is ignored.
func parsePorcelainLine(line string) (path string, stagedDelete, untracked bool) {
	if len(line) < 4 {
		return "", false, false
	}
	xy := line[:2]
	rest := line[3:]
	if xy == "??" {
		return strings.TrimSpace(rest), false, true
	}
	if strings.Contains(rest, " -> ") { // a rename/copy line — not the spurious-delete shape
		return "", false, false
	}
	return strings.TrimSpace(rest), xy[0] == 'D', false
}
