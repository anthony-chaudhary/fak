package wipattr

import (
	"regexp"
	"strings"
)

// The dispatch guards that refuse on orphan WIP report the offending paths inside a
// prose summary rather than a structured field, so recovering the per-path cost signal
// Rank needs means parsing that sentence. The two producers (both in
// tools/issue_resolve_dispatch.py) share one clause:
//
//	issue #4776 names dirty local path(s) already modified in this checkout:
//	  internal/gateway/http.go, cmd/fak/guard.go (+3 more) — refusing
//	  DIRTY_PATH_COLLISION so a new worker cannot overwrite peer WIP; ...
//
//	issue #2477 has recent same-issue uncommitted WIP (…) naming dirty local
//	  path(s): cmd/fak/version_modules.go — refusing SAME_ISSUE_WIP so a second
//	  resolver cannot stack onto unfinished work; ...
//
// Both are orphan-WIP refusals and both count toward a path's blocking cost, so one
// parser covers them. Kept pure and separate from the ledger reader (cmd/fak owns
// loopmgr.LoadPrefix) so the sentence shape is testable without a ledger on disk.
//
// The parse is deliberately CONSERVATIVE: an unrecognised summary yields no paths
// rather than a guess. Under-counting a path's blocks makes the sweep miss a lever;
// mis-attributing them would recommend landing the wrong file, which is worse.

// blockedPathsRE captures everything after the "dirty local path(s)" clause's colon.
// The middle `[^:]*` spans the first producer's " already modified in this checkout"
// without crossing into the path list (that phrase holds no colon, and the class
// cannot cross one, so the greedy match stops at the clause's own colon).
var blockedPathsRE = regexp.MustCompile(`dirty local path\(s\)[^:]*:\s*(.*)$`)

// tailMarkers end the path list and begin the "— refusing <REASON> so …" rationale.
// The em-dash is what both producers emit today; the ASCII fallbacks are here because
// truncating on the WRONG marker fails silently rather than loudly — a summary whose
// dash was normalised to "--" would glue the final path to the following prose
// ("x.go refusing DIRTY_PATH_COLLISION"), the whitespace filter would drop that field
// as prose, and a single-path refusal would parse to ZERO paths. Under-counting the
// dominant single-path cluster to nothing is exactly the failure this fold exists to
// surface, so the tail is cut on any of them.
var tailMarkers = []string{"—", " -- ", " refusing "}

// truncationNoteRE strips the "(+3 more)" the producers append when more than 8 paths
// are dirty. The elided paths are unrecoverable from the summary — see
// ParseBlockedPaths's doc comment on the resulting under-count.
var truncationNoteRE = regexp.MustCompile(`\s*\(\+\d+\s+more\)\s*$`)

// ParseBlockedPaths extracts the repo-relative paths a dispatch refusal summary names
// as dirty. It returns nil for any summary without the clause (a different reason, a
// reworded producer, an empty string) — never a partial guess.
//
// Known under-count: the producers list at most 8 paths and summarise the rest as
// "(+N more)", so a refusal naming 20 dirty paths contributes to only 8. The
// resulting Blocks figure is therefore a LOWER BOUND on what a path has refused.
// That bias is the safe direction — it can only under-rank a lever, never invent one —
// and it does not affect the dominant clusters, where the contract named one path.
func ParseBlockedPaths(summary string) []string {
	m := blockedPathsRE.FindStringSubmatch(summary)
	if m == nil {
		return nil
	}
	list := m[1]
	for _, marker := range tailMarkers {
		if i := strings.Index(list, marker); i >= 0 {
			list = list[:i]
		}
	}
	list = truncationNoteRE.ReplaceAllString(strings.TrimSpace(list), "")

	var out []string
	for _, field := range strings.Split(list, ",") {
		if p := strings.TrimSpace(field); p != "" && !strings.ContainsAny(p, " \t") {
			// A field containing whitespace is prose that leaked past the clause, not a
			// path — drop it rather than counting a sentence fragment as a file.
			out = append(out, strings.TrimPrefix(p, "./"))
		}
	}
	return out
}

// CountBlocks folds refusal summaries into the per-path admission counts Rank consumes.
// A path named by the same summary twice counts once — the cost unit is "admissions
// this path refused", not "mentions".
func CountBlocks(summaries []string) map[string]int {
	counts := make(map[string]int)
	for _, s := range summaries {
		seen := make(map[string]bool)
		for _, p := range ParseBlockedPaths(s) {
			if !seen[p] {
				seen[p] = true
				counts[p]++
			}
		}
	}
	return counts
}
