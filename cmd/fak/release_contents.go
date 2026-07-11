package main

// fak release status folds a "what is shipping" digest into its rolling
// section so an operator deciding whether to cut sees the SHAPE of the pending
// release — which lanes, how many commits, which issues would close, the
// feature/fix mix — not just a bare commits-since-tag count. The digest reuses
// the same (fak <leaf>) ship-stamp fold that `fak release prplan` uses for the
// promotion range, aimed here at the rolling range (last_tag..HEAD). It is
// deterministic over git history: every stamped commit is already a line item
// in the lane that owns it, so there is nothing to keep in sync.

import (
	"fmt"
	"sort"
	"strings"
)

const releaseContentsSchema = "fak.release.contents.v1"

// releaseStatusContents folds last_tag..HEAD into the release-contents digest.
// With no rolling tag the range is undefined (folding all of history is noise,
// not a release preview), so it returns an empty, well-formed digest.
func releaseStatusContents(root, lastTag string) map[string]any {
	if strings.TrimSpace(lastTag) == "" {
		out := foldReleaseContents(nil)
		out["range"] = nil
		return out
	}
	rng := lastTag + "..HEAD"
	raw := releaseStatusGitOutput(root, "log", "--no-merges", "--name-only",
		"--format=%x1e%H%x1f%s%x1f%b%x1f", rng)
	out := foldReleaseContents(parsePRPlanLog(raw))
	out["range"] = rng
	return out
}

// foldReleaseContents groups commits into lane units (biggest-first) and rolls
// up the cross-cutting facts an operator wants before cutting: the conventional
// -commit type histogram, the issues that would close on release (subject-bound
// #N — closure-grade, never body mentions), and the legibility debt (commits
// with no (fak <leaf>) ship-stamp). It is pure so it unit-tests without git.
func foldReleaseContents(commits []prPlanCommit) map[string]any {
	units, unstamped := foldPRPlanUnits(commits)
	types := map[string]int{}
	var resolves []string
	seen := map[string]bool{}
	for _, c := range commits {
		if c.Type != "" {
			types[c.Type]++
		}
		for _, ref := range c.Resolves {
			if !seen[ref] {
				seen[ref] = true
				resolves = append(resolves, ref)
			}
		}
	}
	sort.Strings(resolves)
	lanes := make([]map[string]any, 0, len(units))
	for _, unit := range units {
		lanes = append(lanes, map[string]any{
			"leaf":    unit.Leaf,
			"commits": len(unit.Commits),
			"title":   unit.Title,
		})
	}
	if resolves == nil {
		resolves = []string{}
	}
	return map[string]any{
		"schema":          releaseContentsSchema,
		"commit_count":    len(commits),
		"lane_count":      len(units),
		"unstamped_count": len(unstamped),
		"types":           types,
		"resolves":        resolves,
		"resolves_count":  len(resolves),
		"lanes":           lanes,
	}
}

// releaseStatusContentsLanes reads the lane rows tolerating both the in-memory
// shape (rendered straight from buildReleaseStatus) and the JSON round-trip
// shape (a test that unmarshals the envelope first).
func releaseStatusContentsLanes(contents map[string]any) []map[string]any {
	switch x := contents["lanes"].(type) {
	case []map[string]any:
		return x
	case []any:
		out := make([]map[string]any, 0, len(x))
		for _, item := range x {
			out = append(out, releaseStatusMap(item))
		}
		return out
	}
	return nil
}

// releaseStatusRenderContents renders the digest as one compact human line, or
// "" when the range is empty (the commits-since-tag line already says zero).
func releaseStatusRenderContents(contents map[string]any) string {
	n := releaseStatusInt(contents["commit_count"])
	if n == 0 {
		return ""
	}
	detail := fmt.Sprintf("%d commit(s) across %d lane(s)", n, releaseStatusInt(contents["lane_count"]))
	var top []string
	for i, lane := range releaseStatusContentsLanes(contents) {
		if i >= 3 {
			break
		}
		top = append(top, fmt.Sprintf("%s %d", releaseStatusString(lane["leaf"]), releaseStatusInt(lane["commits"])))
	}
	if len(top) > 0 {
		detail += "; top: " + strings.Join(top, ", ")
	}
	if resolves := releaseStatusStringSlice(contents["resolves"]); len(resolves) > 0 {
		shown, extra := resolves, 0
		if len(shown) > 4 {
			extra = len(shown) - 4
			shown = shown[:4]
		}
		detail += "; closes " + strings.Join(shown, ", ")
		if extra > 0 {
			detail += fmt.Sprintf(" (+%d more)", extra)
		}
	}
	if unstamped := releaseStatusInt(contents["unstamped_count"]); unstamped > 0 {
		detail += fmt.Sprintf("; %d unstamped", unstamped)
	}
	return "  contents: " + detail
}
