package headlesslint

// blocked.go — the STOP-SHAPE dual of leftovers.go and closing.go.
//
// leftovers.go asks whether a run FILED the follow-ups it narrated. closing.go asks
// whether it CLOSES in a scannable shape. This asks a third run-level question about
// the same final summary: does it STOP? AGENTS.md makes the rule explicit —
//
//	There is no blocked state: name the next action (spawn / queue / route around / wait / hand off), never a terminal block.
//
// A worker facing an obstacle always has a disposition available, and internal/loopunblock
// already fixes the closed vocabulary of them: enter, clear_then_enter, bypass, wait,
// escalate, stand_down. That is why there is no blocked state to sit in — a block is not
// an outcome, it is a prompt to route through one of those actions. So a final summary
// that says only "blocked", "cannot proceed", "stuck on", "waiting on a blocker" names no
// disposition at all: it is a stop with nothing to execute, which is the unforced failure
// the kernel already refuses elsewhere.
//
// Fail-closed BY GRAMMAR, not by evidence count. There is no third "cannot say" arm and
// no operator override arm: a block phrase is refused unless the SAME summary also names
// a reachable next action (a next-step phrase, a stated intent, or a cited ticket), so
// the fold decides on the text alone — the grammar of the stop is the evidence, and
// nothing has to be counted, witnessed, or believed. An override here would also
// reintroduce the very thing the doctrine removes: a control plane whose job is to
// "unblock" a terminal block.

import (
	"regexp"
	"strings"
)

// NextActionDoctrine is the AGENTS.md rule this fold enforces, quoted verbatim so code
// and doctrine stay coupled — TestNextActionDoctrineBindsAgentsMd asserts AGENTS.md still
// carries this exact line, so a reworded rule reds the fold's binding instead of
// silently drifting. It is the stop-shape sibling of leftovers.go's Doctrine and
// closing.go's ClosingDoctrine.
const NextActionDoctrine = "There is no blocked state: name the next action (spawn / queue / route around / wait / hand off), never a terminal block."

// NextActionSchema is the versioned envelope tag for a NextActionReport.
const NextActionSchema = "fak-next-action-fold/1"

// NextAction verdicts — the closed top-level judgment of ScanNextAction.
const (
	// NextActionClean: the summary does not stop in terminal-block grammar at all, OR it
	// does and it ALSO names a reachable next action. A block phrase that arrives with a
	// disposition attached is a described next step ("blocked by the lease; routing around
	// to item 4"), not a terminal stop — the obstacle is stated and the move off it is too.
	NextActionClean = "clean"
	// NextActionTerminalBlock: the summary narrates a block ("blocked", "cannot proceed",
	// "stuck on", "waiting on a blocker", "no path forward") and names no next action
	// anywhere in it — a stop with no disposition. The breach this refuses.
	NextActionTerminalBlock = "terminal_block"
)

// BlockedHit is one line of the final summary that uses terminal-block grammar.
type BlockedHit struct {
	Line    int    `json:"line"`
	Match   string `json:"match"`
	Excerpt string `json:"excerpt"`
}

// NextActionReport is the fold over one final summary. Verdict is NextActionTerminalBlock
// iff the summary uses terminal-block grammar and names no next action; otherwise
// NextActionClean.
//
// Next carries the next-action phrase that discharged the block (or "" when none was
// found), so a clean-but-blocked report still shows WHY it was clean: the reader can see
// the exact phrase the fold took as the named next action instead of trusting the verdict.
type NextActionReport struct {
	Schema   string       `json:"schema"`
	Verdict  string       `json:"verdict"`
	Doctrine string       `json:"doctrine"`
	Blocked  int          `json:"blocked"`
	Next     string       `json:"next_action,omitempty"`
	Hits     []BlockedHit `json:"hits,omitempty"`
	Resolve  string       `json:"resolve,omitempty"`
}

// Refused reports whether this summary stopped in terminal-block grammar with no named
// next action — the arm a Stop-hook / guard gate blocks (or nudges) on. A clean report is
// never refused, and there is no undecided arm to disambiguate: the text decides.
func (r NextActionReport) Refused() bool { return r.Verdict == NextActionTerminalBlock }

// terminalBlockRes is the ordered detection table for TERMINAL BLOCK narration: the
// "blocked", "cannot proceed", "stuck on" grammar an agent reaches for when it has run out
// of dispositions and is describing a stop instead of a next step. More specific shapes
// come first so the reported match is the informative one ("blocked by" rather than the
// bare "blocked"), matching the ordering convention of the sibling folds.
var terminalBlockRes = []*regexp.Regexp{
	re(`\bblocked by\b`),
	re(`\bexternally blocked\b`),
	re(`\bwaiting on a blocker\b`),
	re(`\bno path forward\b`),
	re(`\bstuck on\b`),
	re(`\b(cannot|can'?t|can not|unable to) proceed\b`),
	re(`\bblocked\b`),
}

// nextActionRes is the ordered detection table for a NAMED next action — the five
// dispositions (spawn / queue / route around / wait / hand off) plus the stated-intent and
// next-step phrasings a summary uses to report one. Inflections are allowed ("routing
// around", "queued", "handed off") so an honest description of a move reads as the move.
// The loosest marker ("will …") is last: it is a genuine stated intent, but a tighter
// phrase should win when the summary carries both.
var nextActionRes = []*regexp.Regexp{
	re(`\bnext action\b`),
	re(`\bnext steps?\b`),
	re(`\bnext\s*:`),
	re(`\bhand(ing|ed|s)?\s+off\b`),
	re(`\brout\w+\s+around\b`),
	re(`\bspawn\w*\b`),
	re(`\bqueue\w*\b`),
	re(`\bretry\w*\b`),
	re(`\bwill\s`),
}

// nextAction returns the first next-action phrase anywhere in the summary, or "" when it
// names none — the next action does not have to sit on the blocked line; a summary may
// report the obstacle up front and the disposition at the end. A cited ticket counts: a
// next action that is tracked somewhere is a REACHABLE next action, which is exactly the
// difference between a deferral and a stop. Ticket detection is deliberately the bare
// `#\d+` citation rather than headlesslint's wider hasTicketRef — this fold must not be
// satisfied by a summary that merely mentions the word "filed".
//
// Matching is per line (the same shape the sibling folds use) so the reported phrase keeps
// the summary's original casing and no match index is ever taken against a whole
// lower-cased summary.
func nextAction(summary string) string {
	for _, raw := range splitLines(summary) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		if m := firstMatch(nextActionRes, low, line); m != "" {
			return m
		}
		if m := ticketRefRE.FindString(low); m != "" {
			return m
		}
	}
	return ""
}

// ScanNextAction folds a final summary into a NextActionReport. It records every line that
// uses terminal-block grammar and then asks the one question that decides the verdict: did
// the same summary also name a next action? No next action alongside a block phrase is
// NextActionTerminalBlock (refused); anything else is NextActionClean — a summary with no
// block grammar at all, or one whose block phrase carries its disposition.
//
// Pure and stdlib-only, like every fold in this package: text in, typed report out. There
// is no override parameter on purpose — see the file doc on why the fold has no operator
// escape.
func ScanNextAction(summary string) NextActionReport {
	rep := NextActionReport{
		Schema:   NextActionSchema,
		Verdict:  NextActionClean,
		Doctrine: NextActionDoctrine,
	}
	for i, raw := range splitLines(summary) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		if m := firstMatch(terminalBlockRes, low, line); m != "" {
			rep.Hits = append(rep.Hits, BlockedHit{
				Line:    i + 1,
				Match:   clip(m, 120),
				Excerpt: clip(line, 200),
			})
		}
	}
	rep.Blocked = len(rep.Hits)
	if rep.Blocked == 0 {
		return rep
	}
	// A block phrase with a named next action somewhere in the same summary is a DESCRIBED
	// next step, not a terminal stop — record which phrase discharged it so the clean
	// verdict is legible.
	if rep.Next = nextAction(summary); rep.Next != "" {
		return rep
	}
	rep.Verdict = NextActionTerminalBlock
	rep.Resolve = "replace the terminal block with the next action: spawn the next step, queue it, route around the obstacle, wait on a condition that self-resolves, or hand the earned decision to its decision owner — and cite the ticket it is tracked in"
	return rep
}
