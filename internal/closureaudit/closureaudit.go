// Package closureaudit is a pure, stdlib-only port of the grader half of
// tools/issue_closure_audit.py (#1406): it binds commits to issue numbers from
// commit text (ClassifyRefs / RefsFromCommits), then grades each issue into
// exactly one witness bucket (Grade / Build) using the per-SHA `dos
// commit-audit` verdicts the caller supplies.
//
// It touches no process, network, filesystem, or clock — all gh/git/dos I/O is
// the caller's job (cmd/fak/dispatch_closure_audit.go) — so the load-bearing
// classification and witness-gated bucketing are tested hermetically against
// canned facts, exactly the pure-fold pattern the sibling internal/commitissuelink
// and internal/closebatch packages follow.
//
// The buckets, witness rungs, closure_rate math, and schema mirror the Python
// grader so a consumer reading fleet-issue-closure-audit/1 sees the same shape.
// The Python script stays as a compatibility shim until fixture parity is pinned.
package closureaudit

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Schema is the payload schema tag, identical to the Python auditor's SCHEMA so
// a consumer that keys off the schema field cannot tell the surfaces apart.
const Schema = "fleet-issue-closure-audit/1"

// Kind classifies how one commit references one issue.
type Kind string

const (
	// Resolving is a GitHub-closing reference: a close/fix/resolve verb form, the
	// repo house noun form (issue #N / issues #N, #M), or any #N in the subject.
	Resolving Kind = "resolving"
	// Mention is a bare body reference (e.g. "see #118") that never resolves.
	Mention Kind = "mention"
)

// Buckets — one per graded issue, mirroring issue_closure_audit.py exactly.
const (
	TrueResolved     = "TRUE_RESOLVED"
	DataResolved     = "DATA_RESOLVED"
	ClaimedClosed    = "CLAIMED_CLOSED"
	ClosedNotPlanned = "CLOSED_NOT_PLANNED"
	OpenWitnessed    = "OPEN_WITNESSED"
	Open             = "OPEN"
)

// Witness rungs mirror the Python constants: a diff-witness is the gold standard,
// a data-witness is honest-but-weaker, and an OK verdict is the gate on both.
const (
	witnessOK   = "diff-witnessed"
	witnessData = "data-witnessed"
	verdictOK   = "OK"
)

// These mirror internal/hooks/commit_issuelink.go and the Python auditor so the
// author-time commit gate and this audit agree on what closes an issue.
var (
	resolveRE      = regexp.MustCompile(`(?i)\b(?:close|fixe?|resolve)[sd]?\s+#(\d+)\b`)
	issueNounRE    = regexp.MustCompile(`(?i)\bissues?\b[\s:]*((?:#\d+[\s,]*(?:and\s+)?)+)`)
	dependencyTail = regexp.MustCompile(`(?i)^\s*(?:builds?\s+on|depends?\s+on|blocks?|blocked\s+by)\b`)
)

func isWordByte(b byte) bool {
	return b == '_' ||
		(b >= '0' && b <= '9') ||
		(b >= 'a' && b <= 'z') ||
		(b >= 'A' && b <= 'Z')
}

// findRefs is the RE2-safe equivalent of the Python auditor's
// `(?<![\w-])#(\d+)\b`: RE2 has no lookbehind, so we scan for a '#' that is not
// glued to a preceding word char or '-' (so "xoxb-...#118" tokens are excluded),
// read the digit run, and require it to end on a word boundary (a trailing
// letter/underscore voids the match, matching Python's trailing `\b`).
func findRefs(s string) []int {
	var out []int
	for i := 0; i < len(s); i++ {
		if s[i] != '#' {
			continue
		}
		if i > 0 {
			p := s[i-1]
			if isWordByte(p) || p == '-' {
				continue
			}
		}
		j := i + 1
		for j < len(s) && s[j] >= '0' && s[j] <= '9' {
			j++
		}
		if j == i+1 {
			continue // no digits after '#'
		}
		if j < len(s) && isWordByte(s[j]) {
			continue // e.g. "#123abc" — no word boundary after the digits
		}
		if n, err := strconv.Atoi(s[i+1 : j]); err == nil {
			out = append(out, n)
		}
	}
	return out
}

// ClassifyRefs classifies every #N in one commit as Resolving or Mention, faithful
// to issue_closure_audit.py.classify_refs: a closing VERB form, the repo house
// noun form (excluding a "builds on / depends on / blocks" dependency tail), or
// any #N in the subject is Resolving; a resolving classification always wins over
// a mention for the same issue.
func ClassifyRefs(subject, body string) map[int]Kind {
	text := subject + "\n" + body
	resolve := map[int]bool{}
	for _, m := range resolveRE.FindAllStringSubmatch(text, -1) {
		if n, err := strconv.Atoi(m[1]); err == nil {
			resolve[n] = true
		}
	}
	for _, loc := range issueNounRE.FindAllStringSubmatchIndex(text, -1) {
		group1 := text[loc[2]:loc[3]]
		if dependencyTail.MatchString(text[loc[3]:]) {
			continue
		}
		for _, n := range findRefs(group1) {
			resolve[n] = true
		}
	}
	for _, n := range findRefs(subject) {
		resolve[n] = true
	}
	out := map[int]Kind{}
	for _, n := range findRefs(text) {
		if resolve[n] {
			out[n] = Resolving
		} else if _, ok := out[n]; !ok {
			out[n] = Mention
		}
	}
	for n := range resolve {
		out[n] = Resolving
	}
	return out
}

// Commit is one pre-read git commit fact (the caller reads these via `git log`).
type Commit struct {
	SHA     string
	Subject string
	Body    string
}

// CommitRef binds one commit to one issue with a resolving/mention kind.
type CommitRef struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject"`
	Kind    Kind   `json:"kind"`
}

// RefsFromCommits folds pre-read commits into an issue-number -> commit-ref map,
// deterministic in commit order (so a per-issue evidence list is stable).
func RefsFromCommits(commits []Commit) map[int][]CommitRef {
	refs := map[int][]CommitRef{}
	for _, c := range commits {
		for n, kind := range ClassifyRefs(c.Subject, c.Body) {
			refs[n] = append(refs[n], CommitRef{SHA: c.SHA, Subject: strings.TrimSpace(c.Subject), Kind: kind})
		}
	}
	// ClassifyRefs returns a Go map (unordered) per commit, so a single commit
	// referencing several issues can append them to their lists in any order.
	// Sort each issue's own list by SHA to make the fold fully deterministic.
	for n := range refs {
		sort.SliceStable(refs[n], func(i, j int) bool { return refs[n][i].SHA < refs[n][j].SHA })
	}
	return refs
}

// Audit is one per-SHA `dos commit-audit` verdict the caller re-read.
type Audit struct {
	Verdict   string `json:"verdict"`
	Witness   string `json:"witness"`
	ClaimKind string `json:"claim_kind,omitempty"`
}

// Issue is a pre-read gh issue fact.
type Issue struct {
	Number      int
	Title       string
	State       string
	StateReason string
}

// Graded is one issue folded into exactly one witness bucket, with its evidence.
type Graded struct {
	Number               int      `json:"number"`
	Title                string   `json:"title"`
	State                string   `json:"state"`
	StateReason          string   `json:"state_reason"`
	Bucket               string   `json:"bucket"`
	ResolvingCommits     []string `json:"resolving_commits"`
	WitnessedCommits     []string `json:"witnessed_commits"`
	DataWitnessedCommits []string `json:"data_witnessed_commits"`
	Mentions             []string `json:"mentions"`
}

func commitIsWitnessed(a Audit) bool { return a.Verdict == verdictOK && a.Witness == witnessOK }
func commitIsData(a Audit) bool      { return a.Verdict == verdictOK && a.Witness == witnessData }

// Grade folds one issue into exactly one bucket using the witness rung, faithful
// to issue_closure_audit.py.grade_issue.
func Grade(issue Issue, refs []CommitRef, audits map[string]Audit) Graded {
	var resolving, witnessed, dataWit, mentions []CommitRef
	for _, r := range refs {
		switch r.Kind {
		case Resolving:
			resolving = append(resolving, r)
			a := audits[r.SHA]
			if commitIsWitnessed(a) {
				witnessed = append(witnessed, r)
			} else if commitIsData(a) {
				dataWit = append(dataWit, r)
			}
		case Mention:
			mentions = append(mentions, r)
		}
	}
	closed := strings.EqualFold(strings.TrimSpace(issue.State), "CLOSED")
	reason := strings.ToUpper(strings.TrimSpace(issue.StateReason))

	var bucket string
	switch {
	case closed && reason == "NOT_PLANNED":
		bucket = ClosedNotPlanned
	case closed && len(witnessed) > 0:
		bucket = TrueResolved
	case closed && len(dataWit) > 0:
		bucket = DataResolved
	case closed:
		bucket = ClaimedClosed
	case len(witnessed) > 0:
		bucket = OpenWitnessed
	default:
		bucket = Open
	}
	return Graded{
		Number:               issue.Number,
		Title:                truncate(issue.Title, 80),
		State:                issue.State,
		StateReason:          issue.StateReason,
		Bucket:               bucket,
		ResolvingCommits:     shortSHAs(resolving),
		WitnessedCommits:     shortSHAs(witnessed),
		DataWitnessedCommits: shortSHAs(dataWit),
		Mentions:             shortSHAs(mentions),
	}
}

// Totals is the audited-population summary.
type Totals struct {
	IssuesAudited int `json:"issues_audited"`
	ClosedAudited int `json:"closed_audited"`
}

// Report is the fleet-issue-closure-audit/1 payload (grader half; the caller
// supplies coverage/caching around it).
type Report struct {
	Schema          string         `json:"schema"`
	OK              bool           `json:"ok"`
	Verdict         string         `json:"verdict"`
	Finding         string         `json:"finding"`
	Reason          string         `json:"reason"`
	NextAction      string         `json:"next_action"`
	Workspace       string         `json:"workspace,omitempty"`
	ClosureRate     *float64       `json:"closure_rate"`
	HonestCloseRate *float64       `json:"honest_close_rate"`
	Counts          map[string]int `json:"counts"`
	Totals          Totals         `json:"totals"`
	Issues          []Graded       `json:"issues"`
	// Coverage records whether the audit window saw the whole backlog or only a
	// slice of it. The pure Build does not populate it (it has no window facts);
	// the I/O shell computes it via ComputeCoverage and attaches it, so a
	// narrowed audit can never present as complete coverage. nil = not computed.
	Coverage *Coverage `json:"coverage,omitempty"`
}

// Coverage verdict + warning tokens. A truncated window makes closure_rate a
// number over a *slice* of the backlog, not the backlog — so it is surfaced
// loudly rather than letting issues_audited read as comprehensive.
const (
	CoverageComplete   = "COVERAGE_COMPLETE"
	CoverageIncomplete = "COVERAGE_INCOMPLETE"
	// AuditWindowTruncated is the warning emitted when the issue fetch or the
	// git-log scan was capped below the backlog: closures whose issue or
	// resolving commit fell outside the window are unseen and could be
	// under-reported as witnessed. It is the closure-audit analogue of scoring a
	// KPI while unmeasured — "didn't look" must not read as "looks clean".
	AuditWindowTruncated = "AUDIT_WINDOW_TRUNCATED"
)

// CoverageCaps is the machine-actionable re-run recommendation that would clear
// a truncation, plus the exact command to run.
type CoverageCaps struct {
	IssueLimit int    `json:"issue_limit"`
	MaxCommits int    `json:"max_commits"`
	Command    string `json:"command"`
}

// Coverage reports whether either load surface (gh issue fetch, git-log scan)
// hit its cap, mirroring issue_closure_audit.py.compute_coverage.
type Coverage struct {
	Complete         bool         `json:"complete"`
	Verdict          string       `json:"verdict"` // COVERAGE_COMPLETE | COVERAGE_INCOMPLETE
	Warning          string       `json:"warning,omitempty"`
	IssuesTruncated  bool         `json:"issues_truncated"`
	CommitsTruncated bool         `json:"commits_truncated"`
	IssuesFetched    int          `json:"issues_fetched"`
	IssueLimit       int          `json:"issue_limit"`
	CommitsScanned   int          `json:"commits_scanned"`
	CommitsWindow    int          `json:"commits_window"`
	CommitsTotal     *int         `json:"commits_total"` // nil when git could not answer
	Notes            []string     `json:"notes,omitempty"`
	Recommended      CoverageCaps `json:"recommended"`
}

// ComputeCoverage detects whether the audit saw the whole backlog or only a
// slice. `gh issue list` returns newest-first, so a fetch returning exactly the
// limit almost certainly dropped older issues — disproportionately the closed
// ones this auditor grades. Likewise a git-log window narrower than history can
// leave a closed issue's resolving commit unbindable, mis-grading it CLAIMED.
// Pure: totalCommits is nil when git could not answer, in which case a full
// window (commitsScanned >= maxCommits) is treated conservatively as truncated.
func ComputeCoverage(issuesFetched, issueLimit, commitsScanned, maxCommits int, totalCommits *int) Coverage {
	issuesTruncated := issuesFetched >= issueLimit
	commitsTruncated := (totalCommits != nil && *totalCommits > commitsScanned) ||
		(totalCommits == nil && commitsScanned >= maxCommits)

	var notes []string
	if issuesTruncated {
		notes = append(notes, fmt.Sprintf(
			"gh fetch returned %d issue(s) = the --issue-limit cap; older issues "+
				"(disproportionately the closed ones) may be unseen — raise --issue-limit",
			issuesFetched))
	}
	if commitsTruncated {
		scanned := commitsScanned
		total := "?"
		if totalCommits != nil {
			total = strconv.Itoa(*totalCommits)
			if *totalCommits < scanned {
				scanned = *totalCommits
			}
		}
		notes = append(notes, fmt.Sprintf(
			"git-log window scanned %d of %s commit(s); a resolving commit older "+
				"than the window can't bind — raise --max-commits", scanned, total))
	}

	complete := !(issuesTruncated || commitsTruncated)
	verdict, warning := CoverageComplete, ""
	if !complete {
		verdict, warning = CoverageIncomplete, AuditWindowTruncated
	}
	return Coverage{
		Complete:         complete,
		Verdict:          verdict,
		Warning:          warning,
		IssuesTruncated:  issuesTruncated,
		CommitsTruncated: commitsTruncated,
		IssuesFetched:    issuesFetched,
		IssueLimit:       issueLimit,
		CommitsScanned:   commitsScanned,
		CommitsWindow:    maxCommits,
		CommitsTotal:     totalCommits,
		Notes:            notes,
		Recommended:      recommendCaps(issuesTruncated, commitsTruncated, issueLimit, maxCommits, totalCommits),
	}
}

// recommendCaps returns caps that would clear the truncation: a truncated issue
// fetch doubles the issue cap (gh gives no total, so headroom is the honest
// move); a truncated commit window jumps above known history (+1000 for growth)
// or doubles when the total is unknown.
func recommendCaps(issuesTruncated, commitsTruncated bool, issueLimit, maxCommits int, totalCommits *int) CoverageCaps {
	recIssueLimit := issueLimit
	if issuesTruncated {
		recIssueLimit = issueLimit * 2
	}
	recMaxCommits := maxCommits
	if commitsTruncated {
		if totalCommits != nil {
			recMaxCommits = *totalCommits + 1000
		} else {
			recMaxCommits = maxCommits * 2
		}
	}
	return CoverageCaps{
		IssueLimit: recIssueLimit,
		MaxCommits: recMaxCommits,
		Command: fmt.Sprintf("fak dispatch closure-audit --issue-limit %d --max-commits %d",
			recIssueLimit, recMaxCommits),
	}
}

// Build grades every issue and folds the standard closure-audit payload, faithful
// to issue_closure_audit.py.build_payload (minus the gh/git coverage-truncation
// block, which the I/O shell owns; auditError carries any read-back failure).
func Build(workspace string, issues []Issue, refs map[int][]CommitRef, audits map[string]Audit, auditError string) Report {
	graded := make([]Graded, 0, len(issues))
	counts := map[string]int{
		TrueResolved: 0, DataResolved: 0, ClaimedClosed: 0,
		ClosedNotPlanned: 0, OpenWitnessed: 0, Open: 0,
	}
	for _, issue := range issues {
		g := Grade(issue, refs[issue.Number], audits)
		graded = append(graded, g)
		counts[g.Bucket]++
	}
	sortGraded(graded)

	trueResolved := counts[TrueResolved]
	dataResolved := counts[DataResolved]
	claimed := counts[ClaimedClosed]
	openWit := counts[OpenWitnessed]

	var closureRate *float64
	if denom := trueResolved + claimed; denom > 0 {
		r := round4(float64(trueResolved) / float64(denom))
		closureRate = &r
	}
	var honest *float64
	if denom := trueResolved + dataResolved + claimed; denom > 0 {
		r := round4(float64(trueResolved+dataResolved) / float64(denom))
		honest = &r
	}

	ok, verdict, finding, reason, next := gradeVerdict(auditError, claimed, openWit, closureRate)

	return Report{
		Schema:          Schema,
		OK:              ok,
		Verdict:         verdict,
		Finding:         finding,
		Reason:          reason,
		NextAction:      next,
		Workspace:       workspace,
		ClosureRate:     closureRate,
		HonestCloseRate: honest,
		Counts:          counts,
		Totals: Totals{
			IssuesAudited: len(graded),
			ClosedAudited: trueResolved + dataResolved + claimed + counts[ClosedNotPlanned],
		},
		Issues: graded,
	}
}

func gradeVerdict(auditError string, claimed, openWit int, closureRate *float64) (ok bool, verdict, finding, reason, next string) {
	switch {
	case auditError != "":
		return false, "AUDIT_ERROR", "tooling_error", auditError,
			"fix the gh/git/dos read-back error, then re-run the closure audit"
	case claimed > 0:
		reason = strconv.Itoa(claimed) + " issue(s) are CLOSED with no diff-witnessed resolving commit (closure_rate=" + rateStr(closureRate) + ")"
		next = "investigate the CLAIMED_CLOSED issues: each was closed without a DOS-witnessed commit — reopen, bind a (fak <leaf>) commit, or confirm it was a non-code close"
		if openWit > 0 {
			reason += "; " + strconv.Itoa(openWit) + " OPEN_WITNESSED issue(s) are closable now (fix shipped + witnessed)"
			next += "; and close the OPEN_WITNESSED issues whose fix already shipped"
		}
		return false, "ACTION", "claimed_closed", reason, next
	case openWit > 0:
		return true, "OK", "shipped_but_open",
			strconv.Itoa(openWit) + " issue(s) have a diff-witnessed resolving commit but are still OPEN; closure_rate=" + rateStr(closureRate),
			"close the OPEN_WITNESSED issues — the fix already shipped and is witnessed"
	default:
		return true, "OK", "closures_witnessed",
			"all closed issues are witnessed or non-code closes; closure_rate=" + rateStr(closureRate),
			"no closure-honesty action needed; re-run after the next supervisor tick"
	}
}

// WitnessedOpenNumbers is the OPEN_WITNESSED issue set — exactly the tickets the
// close arm should drive to CLOSED (the fix shipped and is witnessed).
func WitnessedOpenNumbers(rep Report) []int {
	var out []int
	for _, g := range rep.Issues {
		if g.Bucket == OpenWitnessed {
			out = append(out, g.Number)
		}
	}
	return out
}

func sortGraded(graded []Graded) {
	order := map[string]int{
		ClaimedClosed: 0, OpenWitnessed: 1, TrueResolved: 2,
		DataResolved: 3, Open: 4, ClosedNotPlanned: 5,
	}
	rank := func(b string) int {
		if v, ok := order[b]; ok {
			return v
		}
		return 9
	}
	sort.SliceStable(graded, func(i, j int) bool {
		if ri, rj := rank(graded[i].Bucket), rank(graded[j].Bucket); ri != rj {
			return ri < rj
		}
		return graded[i].Number > graded[j].Number
	})
}

func shortSHAs(refs []CommitRef) []string {
	out := make([]string, 0, len(refs))
	for _, r := range refs {
		out = append(out, shortSHA(r.SHA))
	}
	return out
}

func shortSHA(sha string) string {
	if len(sha) > 7 {
		return sha[:7]
	}
	return sha
}

// truncate mirrors the Python auditor's title[:80] (issue_closure_audit.py),
// which slices by Unicode code point — so it cuts to the first n RUNES, not n
// bytes. A byte slice (s[:n]) both diverges from that parity target on any
// multibyte title and, worse, can split a multibyte rune (e.g. the em-dash "—",
// 3 bytes, common in fak issue titles) into invalid UTF-8 in the emitted JSON.
// truncate mirrors the Python auditor's title[:80] (issue_closure_audit.py),
// which slices by Unicode code point — so it cuts to the first n RUNES, not n
// bytes. A byte slice (s[:n]) both diverges from that parity target on any
// multibyte title and, worse, can split a multibyte rune (e.g. the em-dash "—",
// 3 bytes, common in fak issue titles) into invalid UTF-8 in the emitted JSON.
func truncate(s string, n int) string {
	if n < 0 {
		n = 0
	}
	runes := 0
	for bytePos := range s {
		if runes == n {
			return s[:bytePos]
		}
		runes++
	}
	return s
}

func round4(x float64) float64 { return math.Round(x*10000) / 10000 }

func rateStr(r *float64) string {
	if r == nil {
		return "None"
	}
	return strconv.FormatFloat(*r, 'g', -1, 64)
}
