package closureaudit

// bindprobe.go — the per-issue BINDING PROBE (#5883). `Grade` answers "which bucket
// is this issue in?" over a whole audit sweep. This answers the question a stuck
// dispatch loop actually asks: "for THIS issue, does a binding-satisfying commit
// exist on trunk — and if not, which commits came CLOSE and exactly how did each
// one fail?"
//
// Why it exists. A dispatcher that re-selects an issue until it closes has no
// nameable state for "the work already landed but nothing will ever close it". It
// just re-dispatches. Each new worker arrives, finds the feature built, has nothing
// to add, and exits CLAIM_NO_COMMIT with reason `unknown` — an invisible loop that
// burns a seat per turn forever. Measured on this repo's own `.dispatch-runs/`
// ledger: 47 OPEN issues already carry a resolving + diff-witnessed commit on
// trunk, and 135 dispatch runs were spent on them AFTER that commit landed.
//
// The failure is NOT the `(fak <leaf>)` ship-stamp. The trailer is matched against
// TOUCHED PATHS (internal/hooks/commitstamp.go), never against an issue, and no
// closure surface reads it. The binding is exactly the one `Grade` uses:
//
//	resolving cite (ClassifyRefs)  AND  dos commit-audit OK / diff-witnessed
//	AND  a resolution-binding claim kind (#2998)
//
// plus one rung `Grade` cannot see, because it is a property of the CONSUMER and
// not of the issue: the fleet's OPEN_WITNESSED guard
// (tools/issue_resolve_dispatch.py:open_witnessed_dispositions) scans only the most
// recent `scan_limit` commits (600 by default). This repo lands ~229 commits on its
// busiest day, so that window can be under three days deep. A binding commit older
// than the window is PERMANENTLY invisible to the guard: the issue is provably
// shipped, provably witnessed, and will still be re-dispatched forever. That state
// is BindOutOfWindow — the one this file exists to name.
//
// Read-only and pure: no process, network, filesystem, or clock. The caller reads
// `git log`, runs `dos commit-audit`, and hands the facts in — the same pure-fold
// contract the rest of this package keeps.

import (
	"fmt"
	"sort"
	"strings"
)

// Binding verdicts for one issue.
const (
	// BindBound — a binding-satisfying commit exists AND sits inside the
	// consumer's scan window, so the close arm can still see it.
	BindBound = "BOUND"
	// BindOutOfWindow — a binding-satisfying commit exists but is deeper than
	// ScanLimit. Nothing is missing from the work; the WINDOW is the defect.
	// Re-dispatching this issue can never produce a close.
	BindOutOfWindow = "BOUND_OUT_OF_SCAN_WINDOW"
	// BindUnbound — no commit satisfies the binding. NearMisses says how close
	// the nearest attempts got, or is empty when nothing cites the issue at all.
	BindUnbound = "UNBOUND"
)

// Near-miss kinds — how one commit that TOUCHED this issue failed to bind.
const (
	// MissMentionOnly — the commit names `#N` only as a body mention. The closure
	// grammar counts a bare `#N` only in the SUBJECT; in the body it needs a
	// close/fix/resolve verb or the `issue(s) #N` noun form.
	MissMentionOnly = "mention_only"
	// MissUnwitnessed — a resolving cite whose diff does not witness its own
	// claim (verdict != OK, or witness != diff-witnessed).
	MissUnwitnessed = "unwitnessed"
	// MissNonBindingClaim — resolving and diff-witnessed, but the claim KIND is a
	// doc/triage rung and the issue itself is not docs-shaped (#2998).
	MissNonBindingClaim = "nonbinding_claim_kind"
	// MissOutOfScanWindow — fully binding, but too deep for the consumer's scan
	// window to reach. The only near-miss that is not the commit's fault.
	MissOutOfScanWindow = "out_of_scan_window"
)

// DefaultScanLimit mirrors open_witnessed_dispositions' `scan_limit` default in
// tools/issue_resolve_dispatch.py — the number of most-recent commits the fleet's
// pre-dispatch OPEN_WITNESSED guard reads. Pass 0 to Probe to disable the window
// rung entirely (a consumer that scans all of history is never window-blind).
const DefaultScanLimit = 600

// resolvingClaimKinds mirrors issue_closure_audit._RESOLVING_CLAIM_KINDS: the claim
// kinds that can resolve a non-docs issue. `dos commit-audit` emits the bare `test`,
// not `test_cover`; both are accepted so the set survives a dos-side rename.
var resolvingClaimKinds = map[string]bool{"code_effect": true, "test": true, "test_cover": true}

// IssueIsDocsRung reports whether an issue TITLE is docs-shaped, so a doc-kind claim
// may resolve it. Mirror of issue_closure_audit._issue_is_docs_rung and
// issue_resolve_witnessed.issue_is_docs_rung so all three layers agree.
func IssueIsDocsRung(title string) bool {
	t := strings.ToLower(strings.TrimLeft(title, " \t"))
	for _, p := range []string{"docs(", "docs:", "doc(", "doc:"} {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	return false
}

// ClaimKindBinds reports whether an audit's claim KIND can bind resolution of an
// issue with this title (#2998) — the Go port of
// issue_closure_audit.commit_binds_resolution, which this package had left on the
// Python side. An UNKNOWN kind fails OPEN: the gate only demotes a KNOWN
// doc/triage claim, it never slanders a witnessed commit whose kind we cannot read.
func ClaimKindBinds(a Audit, title string) bool {
	kind := strings.TrimSpace(a.ClaimKind)
	if kind == "" {
		return true
	}
	if resolvingClaimKinds[kind] {
		return true
	}
	return IssueIsDocsRung(title)
}

// BindCandidate is one pre-read commit plus its DEPTH from the tip of trunk (0 =
// HEAD, 1 = HEAD~1, …). Depth is what makes the window rung checkable; a caller
// that does not know or care about depth leaves it 0 and passes scanLimit 0.
type BindCandidate struct {
	Commit
	Depth int
}

// NearMiss is one commit that referenced the issue but did not bind, named with the
// rung it failed and a human-readable detail that says what to do about it.
type NearMiss struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject"`
	Kind    string `json:"kind"`
	Detail  string `json:"detail"`
	Depth   int    `json:"depth"`
}

// BindProbe is the per-issue verdict: does a binding-satisfying commit exist, and
// if not (or if it is unreachable), exactly why.
type BindProbe struct {
	Issue        int        `json:"issue"`
	Title        string     `json:"title,omitempty"`
	Verdict      string     `json:"verdict"`
	BindingSHA   string     `json:"binding_sha,omitempty"`
	BindingDepth int        `json:"binding_depth,omitempty"`
	ScanLimit    int        `json:"scan_limit"`
	NearMisses   []NearMiss `json:"near_misses,omitempty"`
	// Next is the one checkable step that changes the state — never a restatement
	// of the verdict. A stuck loop needs an action, not a diagnosis.
	Next string `json:"next"`
}

// Probe grades one issue's binding over the candidate commits the caller read.
//
// scanLimit is the consumer's window (use DefaultScanLimit for the fleet
// dispatcher, 0 to disable the window rung). A commit at Depth >= scanLimit is
// still a real binding — it is reported as BindOutOfWindow, never as UNBOUND,
// because the work HAS shipped and saying otherwise would send a worker to rebuild
// it. Deterministic: the shallowest binding commit wins, ties break on SHA, and
// near-misses are sorted by depth then SHA.
func Probe(issue Issue, cands []BindCandidate, audits map[string]Audit, scanLimit int) BindProbe {
	p := BindProbe{Issue: issue.Number, Title: issue.Title, ScanLimit: scanLimit}
	bindingDepth := -1
	var bindingSHA string
	var deepBindings []BindCandidate

	for _, c := range cands {
		kind, ok := ClassifyRefs(c.Subject, c.Body)[issue.Number]
		if !ok {
			continue // this commit does not touch the issue at all
		}
		subject := strings.TrimSpace(c.Subject)
		if kind == Mention {
			p.NearMisses = append(p.NearMisses, NearMiss{
				SHA: c.SHA, Subject: subject, Kind: MissMentionOnly, Depth: c.Depth,
				Detail: fmt.Sprintf("names #%d only as a body mention; a bare #%d binds only in the SUBJECT line "+
					"(in the body it needs `closes/fixes/resolves #%d` or the `issue #%d` noun form)",
					issue.Number, issue.Number, issue.Number, issue.Number),
			})
			continue
		}
		a := audits[c.SHA]
		if !commitIsWitnessed(a) {
			p.NearMisses = append(p.NearMisses, NearMiss{
				SHA: c.SHA, Subject: subject, Kind: MissUnwitnessed, Depth: c.Depth,
				Detail: fmt.Sprintf("resolving cite, but `dos commit-audit` graded verdict=%s witness=%s; "+
					"the close arm keeps only verdict=%s witness=%s",
					orNone(a.Verdict), orNone(a.Witness), verdictOK, witnessOK),
			})
			continue
		}
		if !ClaimKindBinds(a, issue.Title) {
			p.NearMisses = append(p.NearMisses, NearMiss{
				SHA: c.SHA, Subject: subject, Kind: MissNonBindingClaim, Depth: c.Depth,
				Detail: fmt.Sprintf("resolving and diff-witnessed, but claim_kind=%q is a doc/triage rung and "+
					"the issue title is not docs-shaped, so it cannot resolve this issue (#2998)", a.ClaimKind),
			})
			continue
		}
		// A real binding. Keep the shallowest; remember the rest for the window rung.
		deepBindings = append(deepBindings, c)
		if bindingDepth < 0 || c.Depth < bindingDepth || (c.Depth == bindingDepth && c.SHA < bindingSHA) {
			bindingDepth, bindingSHA = c.Depth, c.SHA
		}
	}

	sort.SliceStable(p.NearMisses, func(i, j int) bool {
		if p.NearMisses[i].Depth != p.NearMisses[j].Depth {
			return p.NearMisses[i].Depth < p.NearMisses[j].Depth
		}
		return p.NearMisses[i].SHA < p.NearMisses[j].SHA
	})

	switch {
	case bindingSHA == "":
		p.Verdict = BindUnbound
		p.Next = unboundNext(issue, p.NearMisses)
	case scanLimit > 0 && bindingDepth >= scanLimit:
		p.Verdict = BindOutOfWindow
		p.BindingSHA, p.BindingDepth = bindingSHA, bindingDepth
		for _, c := range deepBindings {
			p.NearMisses = append(p.NearMisses, NearMiss{
				SHA: c.SHA, Subject: strings.TrimSpace(c.Subject), Kind: MissOutOfScanWindow, Depth: c.Depth,
				Detail: fmt.Sprintf("binds, but sits %d commits back — past the consumer's %d-commit scan window, "+
					"so the OPEN_WITNESSED guard can never see it", c.Depth, scanLimit),
			})
		}
		p.Next = fmt.Sprintf("the work for #%d SHIPPED in %s and is witnessed; do not re-dispatch. "+
			"Close it against that sha by hand, or widen the consumer's scan window past %d commits",
			issue.Number, shortSHA(bindingSHA), bindingDepth)
	default:
		p.Verdict = BindBound
		p.BindingSHA, p.BindingDepth = bindingSHA, bindingDepth
		p.Next = fmt.Sprintf("#%d is closable now against %s; run the close arm rather than dispatching a worker",
			issue.Number, shortSHA(bindingSHA))
	}
	return p
}

// unboundNext names the cheapest repair for an unbound issue, chosen from the
// nearest miss rather than emitted as boilerplate.
func unboundNext(issue Issue, misses []NearMiss) string {
	best := ""
	for _, m := range misses {
		// Prefer the rung closest to binding: claim-kind > unwitnessed > mention.
		switch m.Kind {
		case MissNonBindingClaim:
			return fmt.Sprintf("%s is witnessed but claims a doc/triage kind; land a code-or-test commit "+
				"whose subject cites #%d, or re-title the issue as a docs rung", shortSHA(m.SHA), issue.Number)
		case MissUnwitnessed:
			if best == "" {
				best = fmt.Sprintf("%s cites #%d but its diff does not witness its own claim; "+
					"land the source/test change the subject promises", shortSHA(m.SHA), issue.Number)
			}
		case MissMentionOnly:
			if best == "" {
				best = fmt.Sprintf("%s already carries the work but mentions #%d only in the body; "+
					"a follow-on commit citing #%d in its SUBJECT binds it", shortSHA(m.SHA), issue.Number, issue.Number)
			}
		}
	}
	if best != "" {
		return best
	}
	return fmt.Sprintf("no commit on trunk references #%d; this issue has no landed work to bind — "+
		"dispatching a worker is the correct next step", issue.Number)
}

// Shipped reports whether the probe found landed, witnessed work — true for BOTH
// BindBound and BindOutOfWindow. This is the predicate a dispatcher should consult
// before re-selecting an issue: an out-of-window binding is still shipped work, and
// re-dispatching it is the loop this package exists to name.
func (p BindProbe) Shipped() bool {
	return p.Verdict == BindBound || p.Verdict == BindOutOfWindow
}

func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "<none>"
	}
	return s
}
