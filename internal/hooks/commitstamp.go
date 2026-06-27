package hooks

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// commitstamp.go — the PRE-commit ship-stamp lint. The commit-msg gate (gate_commitmsg.go)
// answers "is this subject witness-gradeable?"; this answers the two questions that only bite
// you AFTER the commit lands, when the shared trunk has already moved and you can no longer
// amend (a sibling can push your local commit before you fix it):
//
//  1. Can `dos verify` BIND this commit to a unit of work? — i.e. does the subject carry a
//     `(fak <leaf>)` ship-stamp at all (or is it an exempt merge/revert/release subject)?
//  2. Does the stamped <leaf> actually MATCH the lane the committed paths fall in? — a
//     `(fak gatway)` typo or a `(fak gateway)` stamp on an internal/policy edit binds the
//     commit to a unit nobody queries by name, the silent half of the trust gap.
//
// It is the Go, pre-commit, path-aware complement of tools/commit_stamp_doctor.py (which scans
// history AFTER the fact). The lane taxonomy is read from dos.toml — the SAME source of truth
// the DOS referee binds to — so the check tracks new leaves automatically instead of hard-coding
// a list someone has to remember to update. Exposed through `fak commit --preview`.

// Stamp-shape regexes, ported verbatim from tools/commit_stamp_doctor.py (the two bindable
// per-leaf ship shapes the oracle recognizes plus the release anchor).
var (
	trailerLeafRE = regexp.MustCompile(`(?i)\((?:refs\s+)?fak[ :]+([A-Za-z0-9][\w.\-]*)\)\s*$`)
	directLeafRE  = regexp.MustCompile(`(?i)^fak/([A-Za-z0-9][\w.\-]*):`)
	releaseRE     = regexp.MustCompile(`^v\d+\.\d+\.\d+:`)
	// bookkeepingRE — a subject that names work as narrative (a merge, a bulk snapshot, a
	// docs rollup) and was never meant as a ship attribution, so it owes no stamp.
	bookkeepingRE = regexp.MustCompile(`(?i)^(?:Merge\b|Revert\b|[^:]*\bsnapshot:|docs/(?:_plans|_soaks|fanout|dispatch|dispatch-loop):)`)
	// laneTokenRE — a bare lane identifier inside a dos.toml `[lanes]` array.
	laneTokenRE = regexp.MustCompile(`"([A-Za-z0-9][\w.\-]*)"`)
)

// CommitLintReport is the structured verdict over a proposed commit (subject + the paths it
// will touch). OK is true when nothing BLOCKING was found; Issues are blocking, Notes advisory.
type CommitLintReport struct {
	Subject        string   `json:"subject"`
	Gradeable      bool     `json:"gradeable"`                 // CommitMsgVerdict ok (verb-led, conventional type)
	GradeWhy       string   `json:"grade_why,omitempty"`       // why not, if !Gradeable
	StampKind      string   `json:"stamp_kind"`                // "trailer" | "direct" | "release" | "exempt" | "none"
	Leaf           string   `json:"leaf,omitempty"`            // the stamped <leaf>, "" if unstamped
	LeafRecognized bool     `json:"leaf_recognized"`           // <leaf> is a declared dos.toml lane (or a real cmd/<leaf> demo)
	PathLanes      []string `json:"path_lanes,omitempty"`      // the lanes the paths fall in (deduped, sorted)
	LeafMatches    bool     `json:"leaf_matches"`              // <leaf> is an acceptable stamp for those lanes
	SuggestTrailer string   `json:"suggest_trailer,omitempty"` // the trailer the paths imply, e.g. "(fak gateway)"
	Issues         []string `json:"issues,omitempty"`          // BLOCKING defects, each with a fix
	Notes          []string `json:"notes,omitempty"`           // advisory observations
	OK             bool     `json:"ok"`                        // len(Issues)==0
}

// LintCommitMessage runs the pre-commit ship-stamp lint over a proposed commit message and the
// set of repo-relative paths it will commit. root is the repo root (for reading dos.toml); ""
// or an unreadable dos.toml degrades gracefully — the leaf-recognition check is then SKIPPED
// rather than failing (parity with commit_stamp_doctor.py's empty-set behaviour).
func LintCommitMessage(message string, paths []string, root string) CommitLintReport {
	r := CommitLintReport{Subject: firstSubjectLine(message)}
	r.StampKind, r.Leaf = stampOf(r.Subject)
	// A release (vX.Y.Z:) / merge / revert / bookkeeping subject is intentionally NOT
	// `type(scope): <verb>` and is exempt from both the verb-led requirement and the per-leaf
	// stamp requirement.
	exempt := r.StampKind == "release" || bookkeepingRE.MatchString(r.Subject)

	// 1. Witness-gradeability (the existing commit-msg gate, reused not duplicated).
	ok, why := CommitMsgVerdict(message)
	if exempt {
		r.Gradeable = true
	} else {
		r.Gradeable, r.GradeWhy = ok, why
		if !ok {
			r.Issues = append(r.Issues, "subject is not witness-gradeable: "+why)
		} else if h := abstainHazard(r.Subject); h != "" {
			// Recognized-verb subjects the DOS referee has been observed to ABSTAIN on despite
			// leading with a verb (cross-session finding: `gate X on Y` earns no diff-witness).
			r.Notes = append(r.Notes, h)
		}
	}

	// 2. Ship-stamp bindability.
	tax := readLaneTaxonomy(root)
	r.PathLanes, _ = lanesForPaths(paths, tax)
	r.SuggestTrailer = suggestTrailer(r.PathLanes)

	switch r.StampKind {
	case "none":
		// An exempt subject (merge/revert/release/bookkeeping) owes no per-leaf stamp.
		if exempt {
			r.StampKind = "exempt"
			r.LeafMatches = true
		} else {
			fix := "append a `(fak <leaf>)` trailer so `dos verify` can bind this commit to a unit of work"
			if r.SuggestTrailer != "" {
				fix = "append the trailer `" + r.SuggestTrailer + "` so `dos verify` can bind this commit to its leaf"
			}
			r.Issues = append(r.Issues, "no ship-stamp: "+fix)
		}
	case "release":
		// A vX.Y.Z release bundles many ships; recognized, not a per-leaf stamp.
		r.LeafRecognized = true
		r.LeafMatches = true
	default: // trailer | direct
		recog, note, issue := classifyLeaf(r.Leaf, root, tax)
		r.LeafRecognized = recog
		if note != "" {
			r.Notes = append(r.Notes, note)
		}
		if issue != "" {
			r.Issues = append(r.Issues, issue)
		}
		r.LeafMatches = leafMatchesPaths(r.Leaf, paths, r.PathLanes)
		if len(r.PathLanes) > 0 && !r.LeafMatches {
			fix := "the stamped leaf `" + r.Leaf + "` is not the lane these paths live in"
			if r.SuggestTrailer != "" {
				fix += " — these paths imply `" + r.SuggestTrailer + "`"
			}
			r.Issues = append(r.Issues, "stamp/path lane mismatch: "+fix)
		}
	}

	if len(paths) > 0 && len(r.PathLanes) == 0 {
		r.Notes = append(r.Notes, "no lane could be inferred for these paths (root-level files?); the stamp/path match was not checked")
	}

	r.OK = len(r.Issues) == 0
	return r
}

// StampOf is the exported wrapper over stampOf: it returns the ship-stamp
// ("trailer"|"direct"|"release"|"none") and the lowercased <leaf> token (empty
// unless trailer/direct) for a commit SUBJECT line. It exists so callers outside
// the hooks package (e.g. internal/cadencereport's WORK-DONE ships count) decide
// ship-ness with the SAME grammar the pre-commit lint binds to, instead of a
// second copy of the regexes drifting out of sync. A merge/bookkeeping/body-only
// `(fak x)` subject returns kind=="none"; only kind in {trailer,direct} is a
// per-leaf ship attribution.
func StampOf(subject string) (kind, leaf string) {
	return stampOf(subject)
}

// stampOf returns the ship-stamp ("trailer"|"direct"|"release"|"none") and the lowercased
// <leaf> token (empty unless trailer/direct) for a subject line.
func stampOf(subject string) (kind, leaf string) {
	if m := trailerLeafRE.FindStringSubmatch(subject); m != nil {
		return "trailer", strings.ToLower(m[1])
	}
	if m := directLeafRE.FindStringSubmatch(subject); m != nil {
		return "direct", strings.ToLower(m[1])
	}
	if releaseRE.MatchString(subject) {
		return "release", ""
	}
	return "none", ""
}

// laneTaxonomy is the slice of dos.toml the stamp lint needs: every declared lane name, and the
// path-prefix -> lane map from [lanes.trees]. loaded is false when dos.toml was unreadable, in
// which case the recognition check is skipped (never failed).
type laneTaxonomy struct {
	declared map[string]bool   // lane name (lowercased) -> declared
	prefixes map[string]string // path prefix ("internal/gateway/") -> lane ("gateway")
	exact    map[string]string // exact path ("version") -> lane ("release"), for non-glob tree entries
	loaded   bool
}

// readLaneTaxonomy parses just the `[lanes]` arrays and `[lanes.trees]` table out of dos.toml.
// It is a deliberately tiny, scoped reader (the repo has zero external deps, so no TOML library):
// lane names and tree globs are simple quoted tokens, so a line scan suffices.
func readLaneTaxonomy(root string) laneTaxonomy {
	tax := laneTaxonomy{declared: map[string]bool{}, prefixes: map[string]string{}, exact: map[string]string{}}
	if strings.TrimSpace(root) == "" {
		return tax
	}
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return tax
	}
	tax.loaded = true
	section := ""
	for _, raw := range strings.Split(string(b), "\n") {
		line := raw
		if i := strings.IndexByte(line, '#'); i >= 0 { // lane names carry no '#'
			line = line[:i]
		}
		t := strings.TrimSpace(line)
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "[") {
			section = strings.Trim(t, "[]")
			continue
		}
		switch section {
		case "lanes":
			for _, m := range laneTokenRE.FindAllStringSubmatch(t, -1) {
				tax.declared[strings.ToLower(m[1])] = true
			}
		case "lanes.trees":
			eq := strings.IndexByte(t, '=')
			if eq < 0 {
				continue
			}
			key := strings.ToLower(strings.TrimSpace(t[:eq]))
			if key == "" {
				continue
			}
			tax.declared[key] = true
			for _, glob := range quotedTokens(t[eq+1:]) {
				if strings.HasSuffix(glob, "**") { // a subtree glob -> prefix match
					p := strings.TrimSuffix(strings.TrimSuffix(glob, "**"), "/")
					if p != "" && !strings.Contains(p, "*") {
						tax.prefixes[strings.ToLower(p)+"/"] = key
					}
				} else if !strings.Contains(glob, "*") { // a bare file entry (e.g. "VERSION")
					tax.exact[strings.ToLower(glob)] = key
				}
			}
		}
	}
	return tax
}

// quotedTokens returns every `"..."`-delimited token in s (tree globs contain slashes, so the
// strict laneTokenRE does not match them; this captures the raw glob string).
func quotedTokens(s string) []string {
	var out []string
	for {
		i := strings.IndexByte(s, '"')
		if i < 0 {
			return out
		}
		s = s[i+1:]
		j := strings.IndexByte(s, '"')
		if j < 0 {
			return out
		}
		out = append(out, s[:j])
		s = s[j+1:]
	}
}

// laneForPath maps one repo-relative path to its lane: the longest matching [lanes.trees] prefix
// (authoritative), else the directory convention (internal/<X> -> X, cmd/** -> cmd, a top-level
// lane dir -> itself). Returns "" when no lane can be inferred (e.g. a root-level file).
func laneForPath(path string, tax laneTaxonomy) string {
	p := normPath(path)
	lp := strings.ToLower(p)
	if lane, ok := tax.exact[lp]; ok {
		return lane
	}
	best, bestLane := "", ""
	for prefix, lane := range tax.prefixes {
		if strings.HasPrefix(lp, prefix) && len(prefix) > len(best) {
			best, bestLane = prefix, lane
		}
	}
	if bestLane != "" {
		return bestLane
	}
	seg := strings.Split(p, "/")
	if len(seg) >= 2 {
		switch seg[0] {
		case "internal":
			return strings.ToLower(seg[1])
		case "cmd":
			return "cmd"
		case "docs", "tools", "examples", "visuals", "experiments":
			return seg[0]
		}
	}
	return ""
}

// lanesForPaths returns the deduped, sorted set of lanes the paths fall in, plus a count of paths
// for which no lane could be inferred.
func lanesForPaths(paths []string, tax laneTaxonomy) (lanes []string, unknown int) {
	seen := map[string]bool{}
	for _, p := range paths {
		l := laneForPath(p, tax)
		if l == "" {
			unknown++
			continue
		}
		if !seen[l] {
			seen[l] = true
			lanes = append(lanes, l)
		}
	}
	sort.Strings(lanes)
	return lanes, unknown
}

// leafMatchesPaths reports whether the stamped leaf is ONE of the lanes the commit touches (its
// primary lane), accepting a cmd/<dir> demo's directory name too (#518). A leaf+shim commit
// legitimately spans `<leaf>` and `cmd`, so the test is membership in the touched set — NOT that
// every path matches — which would wrongly reject the standard pure-logic-plus-shim commit. With
// no path carrying a lane (root-level files only) the match is vacuously true.
func leafMatchesPaths(leaf string, paths, pathLanes []string) bool {
	if len(pathLanes) == 0 {
		return true
	}
	acc := map[string]bool{}
	for _, l := range pathLanes {
		acc[l] = true
	}
	for _, p := range paths {
		seg := strings.Split(normPath(p), "/")
		if len(seg) >= 2 && seg[0] == "cmd" && seg[1] != "" && strings.ToLower(seg[1]) != "fak" {
			acc[strings.ToLower(seg[1])] = true
		}
	}
	return acc[strings.ToLower(leaf)]
}

// classifyLeaf grades a stamped leaf against the lane taxonomy AND the tree on disk:
//   - a declared dos.toml lane            -> recognized, no issue;
//   - a real internal/<leaf> or cmd/<leaf> package that simply has no declared lane yet
//     -> recognized, ADVISORY note (the arbiter cannot protect an undeclared leaf — a true gap
//     worth declaring, surfaced deterministically instead of relying on memory);
//   - neither (and dos.toml was readable) -> NOT recognized, BLOCKING issue (a likely typo that
//     would bind `dos verify` to a phantom unit), with a nearest-lane hint;
//   - dos.toml unreadable                 -> recognized (skip; never fail on a missing taxonomy).
func classifyLeaf(leaf, root string, tax laneTaxonomy) (recognized bool, note, issue string) {
	ll := strings.ToLower(leaf)
	if tax.declared[ll] {
		return true, "", ""
	}
	if leafIsRealDir(ll, root) {
		return true, "leaf `" + leaf + "` names a real package but no dos.toml lane declares it — the arbiter cannot detect a same-tree collision on it; consider declaring lane `" + leaf + "`", ""
	}
	if !tax.loaded {
		return true, "", ""
	}
	msg := "off-lane stamp `(fak " + leaf + ")`: binds to a unit no lane declares — likely a typo or a non-lane label"
	if near := nearestLane(ll, tax); near != "" {
		msg += " (did you mean `(fak " + near + ")`?)"
	}
	return false, "", msg
}

// leafIsRealDir reports whether leaf is a real internal/<leaf> or cmd/<leaf> directory.
func leafIsRealDir(leaf, root string) bool {
	if strings.TrimSpace(root) == "" || leaf == "" {
		return false
	}
	for _, base := range []string{"internal", "cmd"} {
		if fi, err := os.Stat(filepath.Join(root, base, leaf)); err == nil && fi.IsDir() {
			return true
		}
	}
	return false
}

// nearestLane returns the declared lane closest to leaf (edit distance <= 2), for a "did you
// mean" hint on a likely typo. "" if nothing is close.
func nearestLane(leaf string, tax laneTaxonomy) string {
	leaf = strings.ToLower(leaf)
	best, bestD := "", 3
	for lane := range tax.declared {
		d := levenshtein(leaf, lane)
		if d < bestD || (d == bestD && lane < best) {
			best, bestD = lane, d
		}
	}
	if bestD <= 2 {
		return best
	}
	return ""
}

// suggestTrailer renders the trailer the paths imply: a single lane -> "(fak <lane>)"; several ->
// the first with a note that the commit spans lanes. "" when no lane is known.
func suggestTrailer(pathLanes []string) string {
	switch len(pathLanes) {
	case 0:
		return ""
	case 1:
		return "(fak " + pathLanes[0] + ")"
	default:
		return "(fak " + pathLanes[0] + ")  [paths span lanes: " + strings.Join(pathLanes, ", ") + " — stamp the primary]"
	}
}

// abstainHazard returns a non-empty advisory when a verb-led subject still tends to earn an
// ABSTAIN from the DOS commit-audit referee. Cross-session finding: `gate <X> on <Y>` reads as
// a noun phrase to the witness grammar despite "gate" being a recognized verb — phrase the work
// as add/implement/fix/test to earn `diff-witnessed`.
func abstainHazard(subject string) string {
	m := subjectRE.FindStringSubmatch(subject)
	if m == nil {
		return ""
	}
	rest := strings.ToLower(strings.TrimSpace(m[4]))
	if strings.HasPrefix(rest, "gate ") && strings.Contains(rest, " on ") {
		return "subject is verb-led but `gate X on Y` has been observed to ABSTAIN at the DOS referee (no diff-witness); if you need `diff-witnessed`, phrase as add/implement/fix/test"
	}
	return ""
}

func normPath(path string) string {
	p := strings.ReplaceAll(path, "\\", "/")
	return strings.TrimPrefix(p, "./")
}

// levenshtein is the standard edit distance (small inputs: lane names), for the typo hint.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		cur := make([]int, lb+1)
		cur[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min3(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
