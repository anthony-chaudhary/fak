// Package choicetriage decenters the human from a surfaced "choice".
//
// A fleet that narrates its work tends to manufacture decisions: it stops,
// prints two or three options, and waits for a person to pick one. Most of
// those choices are fake. Look closely and almost every surfaced choice is
// actually one of four things:
//
//   - one obvious option — there is really only one sane move, so an agent
//     should just take it (best-effort) instead of asking;
//   - an evaluation — the answer is not obvious, but it is knowable; it wants
//     a FRESH CONTEXT WINDOW at the most capable model tier to work it end to
//     end, not a human tap on the shoulder;
//   - a scope too large for one step — it is real work, but it does not fit in
//     this context, so it decomposes into a GH ticket the fleet picks up;
//   - the small irreducible remainder — a genuine policy / auth / release /
//     priority / trust decision that only a person holds the authority to make.
//
// This package is the closed vocabulary for that fold. It is the decision-side
// dual of the "no babysitting" doctrine (internal/waiting): waiting bounds how
// long a genuine human page may sit; choicetriage decides whether the page was
// genuine at all. The default is FRESH_CONTEXT, never HUMAN_RESIDUAL — a human
// decision must be EARNED by a real policy/auth signal, not assumed because the
// producer phrased its status as a question.
//
// Pure and stdlib-only: state in, disposition out, no I/O and no clock. It
// imports nothing internal so any layer — the operator brief, a dispatch loop,
// a stop hook — can fold a surfaced choice through the same taxonomy.
package choicetriage

import "strings"

// Disposition is the closed way a surfaced choice resolves. The four members
// are exhaustive by construction: Triage always returns exactly one of them.
type Disposition string

const (
	// TakeObvious: the choice is fake — there is one obvious option. An agent
	// takes it now, best-effort, and works through any issue it hits. No page.
	TakeObvious Disposition = "TAKE_OBVIOUS"

	// FreshContext: the answer is knowable but not obvious. Hand it to a fresh
	// context window at the most capable model tier and let it drive to a
	// verdict end-to-end. This is the DEFAULT for anything not proven to be a
	// human decision — the whole point of the fold is that "unclear" routes to
	// a clean evaluation, not to a person.
	FreshContext Disposition = "FRESH_CONTEXT"

	// FileTicket: real work, but too large to finish in this context. It
	// decomposes into a GH ticket (new or existing) the fleet picks up and
	// drives end-to-end. Scope, not a human, is what is missing.
	FileTicket Disposition = "FILE_TICKET"

	// HumanResidual: the irreducible remainder — a genuine policy / auth /
	// release / priority / trust decision a person holds the authority to make.
	// This is the ONLY disposition that legitimately waits on a human, and it
	// must be earned by a real signal (see humanResidualHints). It shrinks
	// toward zero as the other three absorb what was only ever pretending to
	// need a person.
	HumanResidual Disposition = "HUMAN_RESIDUAL"
)

// Valid reports whether d is one of the four closed members.
func (d Disposition) Valid() bool {
	switch d {
	case TakeObvious, FreshContext, FileTicket, HumanResidual:
		return true
	default:
		return false
	}
}

// NeedsHuman is the one bit a caller reads to decide "does this actually wait
// on a person?". True for exactly HumanResidual — every other disposition is
// something the fleet does itself.
func (d Disposition) NeedsHuman() bool { return d == HumanResidual }

// humanResidualHints are token substrings (upper-cased match) that mark a
// surfaced choice as a GENUINE human decision. Declared data, deliberately
// conservative and deliberately small: it lists only tokens that unambiguously
// name authority a person holds — approve/release a build, grant an auth or
// permission, set priority, make a policy/legal/budget/trust call. A choice
// that matches NONE of these is, by construction, not a human decision — it is
// obvious, an evaluation, or a ticket. Mirrors the closed-list style of
// internal/waiting.BlockedReasonHints; extend only as real authority-bearing
// reasons appear, never to sweep ordinary work back onto a human.
var humanResidualHints = []string{
	"POLICY", "APPROV", "AUTH", "LOGIN", "PERMISSION", "CREDENTIAL",
	"RELEASE", "PUBLISH", "PRIORITY", "TRUST", "LEGAL", "BUDGET", "SPEND",
	"CONSENT", "SIGN-OFF", "SIGNOFF",
}

// obviousActionHints are token substrings (upper-cased match) that mark a
// surfaced choice's Action as something an agent can just DO: a concrete,
// runnable next step (regenerate a report, run a command, re-emit an
// artifact). When the producer already handed us the fix, the "choice" was
// never a choice — it was a to-do. A leading backtick is treated the same way
// (the action opens with a command literal).
var obviousActionHints = []string{
	"GENERATE", "REGENERATE", "RUN ", "RERUN", "RE-RUN", "PASS IT WITH",
	"FAK ", "GO TEST", "GH ", "REBUILD", "RE-EMIT", "REEMIT", "EMIT ",
}

// scopeHints are token substrings (upper-cased match) that mark a surfaced
// choice as too large for one context: roadmap/epic/frontier-scale work, or an
// open-ended "investigate everything" with no single command behind it. These
// route to FileTicket rather than to a person. Matched against the DESCRIPTIVE
// text only (question/detail/action), never the source-pane name — "program"
// as a pane identity is not the same as "program-scale" work.
var scopeHints = []string{
	"ROADMAP", "EPIC", "FRONTIER", "DECOMPOSE", "BACKLOG", "MULTIPLE",
	"SEVERAL", "MIGRATE", "SWEEP", "REFACTOR",
}

// Signal is everything known about one surfaced choice, drawn from fields a
// producer already carries (the operator brief's Choice/Item, a notify, a stop
// hook). Zero values are meaningful: an empty Signal triages to FreshContext,
// the safe default — evaluate it, do not page.
type Signal struct {
	// Severity is the producer's own class for the item, when it has one:
	// "page" | "decision" | "action" | "watch" | "info". "decision" is a
	// strong (but not sole) human-residual signal; it still must clear the
	// authority test below.
	Severity string

	// Source names the producing pane/subsystem (e.g. "cadence", "release").
	Source string

	// Question is the surfaced prompt text ("let agents continue with X?").
	Question string

	// Detail is the "why" behind the choice.
	Detail string

	// Action is the concrete next move the producer already identified, if
	// any. A runnable Action is the single strongest TakeObvious signal.
	Action string

	// OptionCount is how many options the producer surfaced. 0 or 1 means the
	// choice is fake on its face — there is nothing to choose between.
	OptionCount int

	// ScopeLarge lets a caller that already knows the work is oversized force
	// FileTicket without relying on token hints.
	ScopeLarge bool
}

// Verdict is the fold's output: the disposition, a one-line reason, and the
// concrete next move phrased for whoever consumes it (the command to run, "open
// a fresh context window at the top model tier", "file a GH ticket", or the
// human decision to make).
type Verdict struct {
	Disposition Disposition `json:"disposition"`
	Reason      string      `json:"reason"`
	Resolve     string      `json:"resolve"`
	NeedsHuman  bool        `json:"needs_human"`
}

// Triage folds one surfaced choice into its disposition. The order of tests is
// the doctrine, most-specific first:
//
//  1. a genuine authority decision -> HumanResidual (earned, never assumed);
//  2. a runnable action already in hand, or a fake ≤1-option choice -> TakeObvious;
//  3. oversized scope -> FileTicket;
//  4. everything else -> FreshContext (the default: evaluate, do not page).
func Triage(s Signal) Verdict {
	// The authority haystack spans every field — the source pane ("release",
	// "gateway") is itself an authority signal. The scope haystack is the
	// descriptive text only, so a pane named "program" is not misread as
	// program-scale work.
	hay := strings.ToUpper(strings.Join([]string{s.Severity, s.Source, s.Question, s.Detail, s.Action}, " "))
	descHay := strings.ToUpper(strings.Join([]string{s.Question, s.Detail, s.Action}, " "))

	// 1. Genuine human authority. A "decision" severity only counts when it
	// also names authority — a bare "decision" with a runnable fix is still
	// obvious. This is what keeps the residual small.
	if containsAny(hay, humanResidualHints) {
		return Verdict{
			Disposition: HumanResidual,
			Reason:      "names authority only a person holds (policy/auth/release/priority)",
			Resolve:     firstNonEmpty(s.Action, "make the "+strings.ToLower(firstNonEmpty(s.Source, "policy"))+" decision"),
			NeedsHuman:  true,
		}
	}

	// 2. Already-actionable, or a fake choice with nothing to choose.
	if hasObviousAction(s.Action) {
		return Verdict{
			Disposition: TakeObvious,
			Reason:      "the next step is a concrete runnable action — take it, best-effort",
			Resolve:     s.Action,
		}
	}
	if s.OptionCount == 1 {
		return Verdict{
			Disposition: TakeObvious,
			Reason:      "one real option surfaced — there is nothing to decide",
			Resolve:     firstNonEmpty(s.Action, "take the single option"),
		}
	}

	// 3. Too large for one context.
	if s.ScopeLarge || containsAny(descHay, scopeHints) {
		return Verdict{
			Disposition: FileTicket,
			Reason:      "scope exceeds one context — decompose into a GH ticket",
			Resolve:     "file (or claim) a GH ticket and drive it end-to-end",
		}
	}

	// 4. Default: knowable, not obvious -> a clean evaluation, not a person.
	return Verdict{
		Disposition: FreshContext,
		Reason:      "knowable but not obvious — evaluate in a fresh context, do not page",
		Resolve:     "open a fresh context window at the most capable model tier and work it end-to-end",
	}
}

// hasObviousAction reports whether an Action string is a concrete runnable
// step: it opens with a command literal (leading backtick) or contains a
// runnable-action hint token.
func hasObviousAction(action string) bool {
	a := strings.TrimSpace(action)
	if a == "" {
		return false
	}
	if strings.HasPrefix(a, "`") {
		return true
	}
	return containsAny(strings.ToUpper(a), obviousActionHints)
}

// containsAny reports whether hay (already upper-cased) contains any hint.
func containsAny(hay string, hints []string) bool {
	for _, h := range hints {
		if strings.Contains(hay, h) {
			return true
		}
	}
	return false
}

// firstNonEmpty returns the first non-blank string, or "".
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
