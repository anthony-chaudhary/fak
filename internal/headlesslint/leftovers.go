package headlesslint

// leftovers.go — the RUN-LEVEL dual of Scan.
//
// Scan is a per-line sensor: it types each operator-directed note in a text and,
// for DeferredWork, suppresses a line that already cites a ticket. What it cannot
// see is the RUN: a final summary that narrates "there are two more things worth
// doing" while the run filed zero gh issues is a doctrine breach even though each
// line, read alone, is just prose. AGENTS.md makes the rule explicit —
//
//	End of run: file the leftovers, don't narrate them.
//
// — and #3670 asks for the enforcement that was missing: detect a run whose final
// summary narrates deferred / out-of-scope follow-ups AND cross-check whether the
// run filed (or resolved) any gh issue during its lifetime; refuse when leftovers
// are narrated but zero issues were filed, with an operator-overridable escape for
// "genuinely nothing left".
//
// ScanLeftovers is that fold: (final summary text, issues-filed count, override) in,
// a typed LeftoversReport out. Pure and stdlib-only — any layer (the Stop-hook guard
// fold, `fak headless-lint --leftovers`, a loop gate) can bind an agent's final turn
// to the doctrine through the same taxonomy.

import (
	"regexp"
	"strings"
)

// Doctrine is the AGENTS.md spine-first rule this fold enforces, quoted verbatim so
// code and doctrine stay coupled — TestLeftoversDoctrineBindsAgentsMd asserts
// AGENTS.md still carries this exact line, so if the doctrine text moves the fold's
// binding reds instead of silently drifting.
const Doctrine = "End of run: file the leftovers, don't narrate them."

// LeftoversSchema is the versioned envelope tag for a LeftoversReport.
const LeftoversSchema = "fak-leftovers-fold/1"

// Leftovers verdicts — the closed top-level judgment of ScanLeftovers.
const (
	// LeftoversClean: no unfiled leftovers. Either nothing was narrated, every
	// narrated follow-up already cited a filed ticket, at least one gh issue was
	// filed during the run, or the operator escape was set.
	LeftoversClean = "clean"
	// LeftoversUnfiled: the final summary narrates deferred / out-of-scope work but
	// the run filed zero gh issues and no escape was given — the breach this refuses.
	LeftoversUnfiled = "leftovers_unfiled"
)

// LeftoversHit is one line of the final summary that narrates a leftover.
type LeftoversHit struct {
	Line    int    `json:"line"`
	Match   string `json:"match"`
	Excerpt string `json:"excerpt"`
}

// LeftoversReport is the fold over one final summary plus the run's issue-filing
// count. Verdict is LeftoversUnfiled iff leftovers were narrated, zero issues were
// filed, and no operator escape was set; otherwise LeftoversClean.
type LeftoversReport struct {
	Schema      string         `json:"schema"`
	Verdict     string         `json:"verdict"`
	Doctrine    string         `json:"doctrine"`
	Narrated    int            `json:"narrated"`
	IssuesFiled int            `json:"issues_filed"`
	Overridden  bool           `json:"overridden"`
	Hits        []LeftoversHit `json:"hits,omitempty"`
	Resolve     string         `json:"resolve,omitempty"`
}

// Refused reports whether this run narrated leftovers it did not file — the arm a
// Stop-hook / guard gate blocks (or nudges) on. A clean report is never refused.
func (r LeftoversReport) Refused() bool { return r.Verdict == LeftoversUnfiled }

// leftoverRes is the ordered detection table for deferred / out-of-scope narration:
// the "there are two more things worth doing", "out of scope", "follow-up", "left to
// do", "TODO" prose an agent lists at the end instead of filing. A line that already
// cites a ticket (hasTicketRef) is scoping, not narration, and is skipped before this
// runs — so only a BARE punt counts.
var leftoverRes = []*regexp.Regexp{
	re(`\b(a couple|a few|two|three|four|several|some|another|more) (more )?things?\b`),
	re(`\bthings? (worth|left|still|to do|remaining|we|you)\b`),
	re(`\bworth (doing|adding|fixing|handling|filing)\b`),
	re(`\bout[ -]of[ -]scope\b`),
	re(`\bfollow[ -]?ups?\b`),
	re(`\bleft ?overs?\b`),
	re(`\bleft to do\b`),
	re(`\bstill (to do|left|remaining|need|needs|outstanding)\b`),
	re(`\b(remaining|outstanding) (work|item|items|task|tasks|follow|piece|pieces)\b`),
	re(`\bnext steps?\b`),
	re(`\btodos?\b`),
	re(`\bfix ?me\b`),
	re(`\b(defer|deferring|deferred|punt|punted|punting)\b`),
	re(`\bnot (yet )?(done|addressed|handled|implemented|covered|wired)\b`),
	re(`\bcould also\b`),
	re(`\bwe (can|could|should|might) (also|still|later|revisit|follow)\b`),
	re(`\b(can|could|will) be (done|added|addressed|handled|fixed|implemented) later\b`),
}

// ScanLeftovers folds a final summary and the run's issue-filing count into a
// LeftoversReport. The two arms of #3670's done-condition:
//   - narrated leftovers + issuesFiled==0 + !override -> LeftoversUnfiled (refused).
//   - the same summary once the follow-ups were filed (issuesFiled>0), or a line
//     that itself cites a ticket, or an operator escape -> LeftoversClean.
//
// issuesFiled is the run-lifetime count of gh issues the run filed (or resolved) —
// the cross-check the doctrine turns on. override is the operator escape for
// "genuinely nothing left": it forces clean even when leftovers were narrated.
func ScanLeftovers(summary string, issuesFiled int, override bool) LeftoversReport {
	rep := LeftoversReport{
		Schema:      LeftoversSchema,
		Verdict:     LeftoversClean,
		Doctrine:    Doctrine,
		IssuesFiled: issuesFiled,
		Overridden:  override,
	}
	for i, raw := range splitLines(summary) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		if hasTicketRef(low) {
			// A leftover that cites a filed ticket ("out of scope, tracked in #4001")
			// is scoping, not bare narration — the honest end-of-run shape.
			continue
		}
		if m := firstMatch(leftoverRes, low, line); m != "" {
			rep.Hits = append(rep.Hits, LeftoversHit{
				Line:    i + 1,
				Match:   clip(m, 120),
				Excerpt: clip(line, 200),
			})
		}
	}
	rep.Narrated = len(rep.Hits)
	if rep.Narrated > 0 && issuesFiled <= 0 && !override {
		rep.Verdict = LeftoversUnfiled
		rep.Resolve = "file each narrated leftover as an open gh issue (dedupe → done-condition → leak-check → label), then report the issue numbers; pass the operator escape only if there is genuinely nothing left"
	}
	return rep
}
