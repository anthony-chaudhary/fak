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
	Score          int      `json:"score"`                     // 0-100 commit-readiness score; notes lower it, issues crater it
	Grade          string   `json:"grade"`                     // A-F grade derived from Score
	IssueRefs      []int    `json:"issue_refs,omitempty"`      // every #N referenced in the message (#312)
	IssueResolving bool     `json:"issue_resolving"`           // a #N is in a resolving position the closure audit binds (#312)
	Generation     string   `json:"generation,omitempty"`      // optional body sidecar: Generation: gen/now|gen/next|gen/second-next|gen/future
	Issues         []string `json:"issues,omitempty"`          // BLOCKING defects, each with a fix
	Notes          []string `json:"notes,omitempty"`           // advisory observations
	OK             bool     `json:"ok"`                        // len(Issues)==0
}

// LintCommitMessage runs the pre-commit ship-stamp lint over a proposed commit message and the
// set of repo-relative paths it will commit. root is the repo root (for reading dos.toml); ""
// or an unreadable dos.toml degrades gracefully — the leaf-recognition check is then SKIPPED
// rather than failing (parity with commit_stamp_doctor.py's empty-set behaviour).
//
// The missing issue-link is ADVISORY here (a Note); use LintCommitMessageWithOptions with
// requireIssue=true to make it a BLOCKING issue (the dispatch-worker contract, #312).
func LintCommitMessage(message string, paths []string, root string) CommitLintReport {
	return LintCommitMessageWithOptions(message, paths, root, false)
}

// LintCommitMessageWithOptions is LintCommitMessage plus the #312 author-time issue-link gate:
// when requireIssue is true, a message that carries no bindable resolving `#N` (the form the
// closure auditor counts) becomes a BLOCKING issue instead of an advisory note. Opt-in because
// not every commit closes an issue — a hard block belongs on a worker spawned to resolve `#N`,
// not on every mid-feature human commit.
func LintCommitMessageWithOptions(message string, paths []string, root string, requireIssue bool) CommitLintReport {
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
		// A `fix(...)` that touches source but no test may pass commit-audit while the symptom
		// is still live (#1326); nudge for a red-then-green witness. Advisory only — never blocks.
		if len(paths) > 0 {
			if w := fixWantsSymptomWitness(r.Subject, paths); w != "" {
				r.Notes = append(r.Notes, w)
			}
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

	// 3. Issue-link bindability (#312). A close with no resolving commit reads as CLAIMED_CLOSED;
	// detect whether this message carries a `#N` the closure auditor will bind to. An exempt
	// (merge/revert/release/bookkeeping) subject owes no issue link.
	il := lintIssueLink(message)
	r.IssueRefs = il.refs
	r.IssueResolving = il.resolving
	if !il.resolving && !exempt {
		fix := "name the issue this resolves — put `#N` in the subject, or `Closes #N` in the body — so `issue_closure_audit` binds the close (lifts closure_rate; baseline 0.196)"
		if len(il.refs) > 0 {
			// It references issues but only as a MENTION (a body `#N` with no closing verb).
			fix = "this commit mentions an issue but in a non-resolving position; to bind the close, put `#N` in the SUBJECT or write `Closes #N` in the body"
		}
		if requireIssue {
			r.Issues = append(r.Issues, "no bindable issue link: "+fix)
		} else {
			r.Notes = append(r.Notes, "no bindable issue link: "+fix)
		}
	}

	if gen, malformed := lintGenerationSidecar(message); gen != "" {
		r.Generation = gen
	} else if malformed && !exempt {
		r.Notes = append(r.Notes, "generation sidecar is not recognized: use `Generation: gen/now`, `gen/next`, `gen/second-next`, or `gen/future` in the commit body")
	}

	r.OK = len(r.Issues) == 0
	r.Score = commitLintScore(r)
	r.Grade = commitLintGrade(r.Score)
	return r
}

func lintGenerationSidecar(message string) (label string, malformed bool) {
	for _, raw := range strings.Split(message, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok || !strings.EqualFold(strings.TrimSpace(key), "Generation") {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "gen/now":
			return "gen/now", false
		case "gen/next":
			return "gen/next", false
		case "gen/second-next":
			return "gen/second-next", false
		case "gen/future":
			return "gen/future", false
		case "now":
			return "gen/now", false
		case "next":
			return "gen/next", false
		case "second-next":
			return "gen/second-next", false
		case "future":
			return "gen/future", false
		default:
			return "", true
		}
	}
	return "", false
}

func commitLintScore(r CommitLintReport) int {
	score := 100 - 35*len(r.Issues) - 6*len(r.Notes)
	if !r.Gradeable {
		score -= 10
	}
	if r.StampKind == "none" {
		score -= 10
	}
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return score
}

func commitLintGrade(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
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
	if !strings.Contains(p, "/") && allowedRootMD[p] {
		return "docs"
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

// fixWantsSymptomWitness returns a non-empty advisory when a `fix(...)` commit touches Go
// SOURCE but ships NO Go test — the canonical shape of a fix that may pass `dos commit-audit`
// (diff-witnessed) while the symptom is still live (#1326). A green commit-audit grades whether
// the diff did the KIND of thing claimed, NOT whether the bug is gone; the missing artifact is a
// test that FAILS on the parent and PASSES here (red-then-green). This is the advisory, path-only
// counterpart of the git-backed `symptom:` witness rung (internal/witness): it never blocks (a
// Note, not an Issue), so it nudges without refusing — a fix whose nature is genuinely
// untestable (a doc/config-only fix, or one whose witness lands in a sibling lane) still commits.
//
// It fires only when the type is exactly `fix`, at least one Go source file is touched, and no
// `_test.go` is among the paths; a fix touching only docs/config, or one already carrying a
// test, earns nothing.
func fixWantsSymptomWitness(subject string, paths []string) string {
	m := subjectRE.FindStringSubmatch(subject)
	if m == nil || m[1] != "fix" {
		return ""
	}
	sawSource, sawTest := false, false
	for _, p := range paths {
		switch {
		case isGoTestPath(p):
			sawTest = true
		case isGoSourcePath(p):
			sawSource = true
		}
	}
	if sawTest || !sawSource {
		return ""
	}
	return "fix(...) touches Go source but ships no test: a fix should carry a symptom witness — " +
		"a test that FAILS on the parent commit and PASSES here (red-then-green), not just a green " +
		"`dos commit-audit` (which grades the diff's KIND, not that the bug is gone). See #1326."
}

// isGoTestPath reports whether a repo path is a Go test file (basename `*_test.go`). It matches
// the language's own definition of a gating test and deliberately mirrors witness.isGatingTestPath
// so the two surfaces agree on what counts as a symptom witness.
func isGoTestPath(p string) bool {
	return strings.HasSuffix(baseName(normPath(p)), "_test.go")
}

// isGoSourcePath reports whether a repo path is non-test Go SOURCE — a `.go` file that is not a
// test and not under a testdata/ tree (testdata is fixtures, not the bug surface). This is the
// "did the fix touch real code?" half of the symptom-witness heuristic.
func isGoSourcePath(p string) bool {
	q := normPath(p)
	if !strings.HasSuffix(q, ".go") || strings.HasSuffix(baseName(q), "_test.go") {
		return false
	}
	for _, seg := range strings.Split(q, "/") {
		if seg == "testdata" {
			return false
		}
	}
	return true
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
