package adjudicator

import (
	"strings"
)

// gitPushSidestepCommand is the sanctioned safe-push verb offered in place of a bare
// `git push`: a non-force, current-branch push the kernel admits because its command
// head is `fak`, not `git push`. It is pinned to the SAME string the git-push
// reversibility family's redirect hint advertises (familyRemedyCommands["git-push"][0])
// — TestSidestepRewriteMatchesRemedyTable binds the two so the machine-readable
// rewrite target and the human-facing advisory hint can never drift apart.
const gitPushSidestepCommand = "fak sync push"

const gitPullSidestepCommand = "fak sync apply --fetch"

// safeBareGitPush reports whether cmd is the SAFE push subset a sidestep may substitute:
// a single command (no sibling riding along a `&&`/`;`/`|`) whose head is `git push`
// with, at most, a bare remote name and NO branch, NO refspec, and NO flag. Every
// departure — a force/delete/tags/all/mirror/prune flag, an upstream `-u`, a
// `local:remote` refspec, a named target branch (`git push origin main`), or a second
// segment — differs from `fak sync push` (non-force, current-branch, fast-forward-only),
// so laundering it into that verb would silently change intent; those return false and
// stay on the preview-confirm hold.
//
// It parses the RAW command (dashes intact, unlike commandSegments' family-matcher
// normalization) so a flag is visible, strips the same leading env assignments and
// wrapper heads, and splits on commandSegmentRE. That split is deliberately NOT the
// quote-aware one commandSegments uses: a sequencing byte inside a quoted payload
// manufactures an extra segment here and pushes the call OUT of the safe subset. For a
// family matcher that would be a false positive; for a safe-subset admission it is the
// conservative direction — the call keeps its preview-confirm hold.
func safeBareGitPush(cmd string) bool {
	var nonEmpty []string
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		if strings.TrimSpace(raw) != "" {
			nonEmpty = append(nonEmpty, raw)
		}
	}
	if len(nonEmpty) != 1 { // a sibling command disqualifies the whole line
		return false
	}
	fields := strings.Fields(nonEmpty[0])
	for len(fields) > 0 &&
		(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
		fields = fields[1:]
	}
	if len(fields) < 2 || strings.ToLower(fields[0]) != "git" || strings.ToLower(fields[1]) != "push" {
		return false
	}
	positional := 0
	for _, tok := range fields[2:] {
		if strings.HasPrefix(tok, "-") {
			return false // any flag (force, -u, --tags, --all, --delete, --mirror, --prune, …)
		}
		if strings.Contains(tok, ":") {
			return false // an explicit refspec (local:remote)
		}
		positional++
		if positional > 1 {
			return false // a second positional names a specific branch, not the current one
		}
	}
	return true
}

// safeBareGitPull reports whether cmd is the SAFE pull subset a sidestep may substitute:
// a single command (no sibling riding along a `&&`/`;`/`|`) whose head is `git pull`
// with, at most, 2 positionals (remote and branch) and NO flags and NO refspec with `:`.
func safeBareGitPull(cmd string) bool {
	var nonEmpty []string
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		if strings.TrimSpace(raw) != "" {
			nonEmpty = append(nonEmpty, raw)
		}
	}
	if len(nonEmpty) != 1 { // a sibling command disqualifies the whole line
		return false
	}
	fields := strings.Fields(nonEmpty[0])
	for len(fields) > 0 &&
		(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
		fields = fields[1:]
	}
	if len(fields) < 2 || strings.ToLower(fields[0]) != "git" || strings.ToLower(fields[1]) != "pull" {
		return false
	}
	positional := 0
	for _, tok := range fields[2:] {
		if strings.HasPrefix(tok, "-") {
			return false // any flag starting with - (--rebase, --autostash, --force, -f, etc.)
		}
		if strings.Contains(tok, ":") {
			return false // refspec with :
		}
		positional++
		if positional > 2 {
			return false // at most 2 positionals after pull (e.g. git pull origin main)
		}
	}
	return true
}
