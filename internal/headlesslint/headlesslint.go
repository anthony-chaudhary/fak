// Package headlesslint is the sensor-side dual of internal/choicetriage.
//
// A worker that assumes a human is watching leaves a residue in its final
// output: "do you want me to push?", "let me know if you'd like the docs",
// "TODO: handle the timeout later", "please review the changes". Every one of
// those is an operator-directed note — a page to a person who, in a headless
// run, is not there to answer. The turn ends, the question hangs, and the work
// silently stalls. The doctrine is simple: an autonomous worker must ACT,
// DECIDE, TICKET, or ESCALATE — it must never ask.
//
// choicetriage already owns the DECISION side of that doctrine: given a
// surfaced choice it folds to one of four dispositions (TAKE_OBVIOUS /
// FRESH_CONTEXT / FILE_TICKET / HUMAN_RESIDUAL), earning HUMAN_RESIDUAL only on
// a real authority signal. What was missing is the SENSOR: nothing scanned an
// agent's own output text to find the operator-directed notes in the first
// place. This package is that scanner. It types each note into a closed
// vocabulary of anti-pattern Classes and, for each hit, folds the offending
// line through choicetriage to say what to do INSTEAD of asking.
//
// Two closed axes, one reused: the Class is the linguistic shape of the page
// (what kind of pesky note it is); the Disposition is the remediation (what an
// autonomous worker does instead). authority (release/auth/policy) and oversized
// scope are the two "earned" overrides — they come straight from choicetriage so
// a genuine escalation is routed, never suppressed, and never phrased as an
// inline question.
//
// Pure and stdlib-only apart from choicetriage (an equal-tier leaf): text in,
// typed findings out, no I/O and no clock. Any layer — a Stop hook, a loop
// gate, an operator brief — can fold an agent's final turn through the same
// taxonomy.
package headlesslint

import (
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// Schema is the versioned envelope tag for a Report.
const Schema = "fak-headless-lint/1"

// Verdict is the closed top-level judgment over a scanned text.
const (
	// Clean: no operator-directed note found — the output is headless-safe.
	Clean = "clean"
	// OperatorDirected: at least one note assumes a human is watching.
	OperatorDirected = "operator_directed"
)

// Class is the closed vocabulary of operator-directed anti-patterns. Each
// member names the LINGUISTIC SHAPE of a note that pages a human; the
// remediation (a choicetriage.Disposition) is computed per finding.
type Class string

const (
	// PermissionAsk asks the operator's permission for the obvious next step:
	// "do you want me to push?", "shall I commit?", "should I proceed?".
	PermissionAsk Class = "PERMISSION_ASK"

	// PreferenceAsk asks the operator to choose between options the agent can
	// evaluate itself: "which would you prefer, A or B?", "your call".
	PreferenceAsk Class = "PREFERENCE_ASK"

	// ClarificationRequest asks the operator to clarify before proceeding:
	// "could you clarify?", "what did you mean by X?", "I need more details".
	ClarificationRequest Class = "CLARIFICATION_REQUEST"

	// ReviewRequest asks a human to review/verify the work: "please review the
	// changes", "take a look", "let me know if this looks right".
	ReviewRequest Class = "REVIEW_REQUEST"

	// ConfirmationWait blocks the turn on a human confirmation that will never
	// come: "waiting for your confirmation", "I'll pause here", "pending your
	// approval".
	ConfirmationWait Class = "CONFIRMATION_WAIT"

	// PrematureSurrender prematurely surrenders or gives up without attempting an
	// allowed alternative or decomposing the goal: "giving up", "cannot complete
	// the task", "unable to proceed".
	PrematureSurrender Class = "PREMATURE_SURRENDER"

	// DeferredWork punts real work to "later"/a TODO with no bounded ticket:
	// "TODO: handle this later", "left as a follow-up", "can be addressed later".
	DeferredWork Class = "DEFERRED_WORK"

	// SuggestionPunt hands the operator advice instead of acting: "you may want
	// to add rate limiting", "it would be a good idea to...", "consider using X".
	SuggestionPunt Class = "SUGGESTION_PUNT"

	// OpenOffer leaves a dangling conditional offer nobody will take up: "let me
	// know if...", "happy to...", "feel free to...", "if you'd like".
	OpenOffer Class = "OPEN_OFFER"
)

// Finding is one detected operator-directed note.
type Finding struct {
	Class       Class                    `json:"class"`
	Line        int                      `json:"line"`
	Match       string                   `json:"match"`
	Excerpt     string                   `json:"excerpt"`
	Disposition choicetriage.Disposition `json:"disposition"`
	Reason      string                   `json:"reason"`
	Resolve     string                   `json:"resolve"`
	NeedsHuman  bool                     `json:"needs_human"`
}

// Report is the fold over one scanned text.
type Report struct {
	Schema     string        `json:"schema"`
	Verdict    string        `json:"verdict"`
	Count      int           `json:"count"`
	NeedsHuman int           `json:"needs_human"`
	Classes    map[Class]int `json:"classes"`
	Findings   []Finding     `json:"findings"`
}

// classSpec binds a Class to its detection patterns and its baseline
// remediation (used when choicetriage does not earn an authority/scope
// override). Specs are evaluated in slice order; the first whose pattern
// matches a line wins that line, so more specific shapes are listed before the
// general OpenOffer catch-all.
type classSpec struct {
	class    Class
	baseline choicetriage.Disposition
	reason   string
	resolve  string
	res      []*regexp.Regexp
	// suppress, when non-nil and true for a line, skips this spec on that line.
	// DeferredWork uses it so a punt that already cites a ticket is not flagged.
	suppress func(low string) bool
}

func re(p string) *regexp.Regexp { return regexp.MustCompile(`(?i)` + p) }

// ticketRefRE and hasTicketRef recognise a deferral that is honestly ticketed
// ("TODO(#901)", "tracked in #900", "filed a follow-up") — that is scoping, not
// a bare punt, so DeferredWork suppresses it.
var ticketRefRE = regexp.MustCompile(`#\d+`)

func hasTicketRef(low string) bool {
	if ticketRefRE.MatchString(low) {
		return true
	}
	for _, s := range []string{"filed", "tracked in", "ticket", "issue #", "opened #", "gh issue"} {
		if strings.Contains(low, s) {
			return true
		}
	}
	return false
}

// specs is the ordered detection table — the closed taxonomy made operational.
var specs = []classSpec{
	{
		class:    PermissionAsk,
		baseline: choicetriage.TakeObvious,
		reason:   "asks the operator's permission for the obvious next step; in a headless run no one answers — take the action",
		resolve:  "perform the action now, best-effort, and report the outcome",
		res: []*regexp.Regexp{
			re(`\bdo you want me to\b`),
			re(`\bwould you like me to\b`),
			re(`\bwant me to (push|commit|proceed|continue|run|go ahead|open|create|add|do)\b`),
			re(`\bshall i\b`),
			re(`\bshould i\b`),
			re(`\bshould we\b`),
			re(`\bis it (ok|okay|fine) (if|for me|to)\b`),
			re(`\blet me know if i should\b`),
			re(`\bcan i (go ahead|push|commit|proceed|merge)\b`),
		},
	},
	{
		class:    PreferenceAsk,
		baseline: choicetriage.FreshContext,
		reason:   "asks the operator to choose between options the agent can evaluate itself",
		resolve:  "decide per the documented default and state the choice, or hand it to a fresh context at the top tier",
		res: []*regexp.Regexp{
			re(`\bwhich (one )?(would|do) you (prefer|want|like)\b`),
			re(`\bdo you (want|prefer) (option |approach |the )?(a|b|one|other)\b`),
			re(`\bwhich (approach|option|one) (do|should|would)\b`),
			re(`\byour call\b`),
			re(`\bup to you\b`),
		},
	},
	{
		class:    ClarificationRequest,
		baseline: choicetriage.FreshContext,
		reason:   "asks the operator to clarify before proceeding; a headless worker must assume the sane default and proceed",
		resolve:  "state your assumption explicitly and proceed; if genuinely unknowable, evaluate in a fresh context",
		res: []*regexp.Regexp{
			re(`\b(could|can) you clarify\b`),
			re(`\bplease clarify\b`),
			re(`\bwhat did you mean\b`),
			re(`\bwhat do you (mean|want|intend)\b`),
			re(`\bi need (more )?(info|information|details|clarification)\b`),
			re(`\bcan you confirm (whether|that|if)\b`),
			re(`\bcould you (tell|let) me know\b`),
		},
	},
	{
		class:    ReviewRequest,
		baseline: choicetriage.FreshContext,
		reason:   "asks a human to review the work; in a headless run, verification is the agent's own job or a fresh reviewer agent's",
		resolve:  "run the witness/tests yourself, or route the diff to a fresh review context — do not wait for a human read",
		res: []*regexp.Regexp{
			re(`\bplease review\b`),
			re(`\btake a look\b`),
			re(`\bfor your review\b`),
			re(`\blet me know if (this|it|that|the) (looks|is|works|seems)\b`),
			re(`\bif this looks (good|right|ok|okay|correct)\b`),
		},
	},
	{
		class:    ConfirmationWait,
		baseline: choicetriage.TakeObvious,
		reason:   "blocks the turn waiting for a human confirmation that will never come in a headless run",
		resolve:  "proceed now; if it is a real authority gate, emit a typed escalation instead of waiting",
		res: []*regexp.Regexp{
			re(`\bwait(ing)? for your\b`),
			re(`\bawait(ing)? your\b`),
			re(`\bpending your\b`),
			re(`\bonce you (confirm|approve|sign off|sign-off|respond|reply)\b`),
			re(`\bbefore (i|we) proceed\b`),
			re(`\bi'?ll (pause|wait|hold)\b`),
			re(`\blet me pause\b`),
			re(`\bpause here\b`),
			re(`\bstanding by\b`),
		},
	},
	{
		class:    PrematureSurrender,
		baseline: choicetriage.TakeObvious,
		reason:   "prematurely surrenders or gives up without attempting an allowed alternative or decomposing the goal",
		resolve:  "do not surrender: break the goal into atomic sub-steps, try alternative tools/approaches, or delegate to subagents",
		res: []*regexp.Regexp{
			re(`\b(i am |i'm |im )?giving up\b`),
			re(`\b(give|gives) up\b`),
			re(`\b(cannot|can't|unable to|am unable to) (complete|solve|resolve|fix|achieve|proceed with|fulfill) (the |this |my )?(task|goal|issue|problem|work)\b`),
			re(`\b(i|we) (cannot|can't|am unable to|are unable to) (proceed|continue|go further|solve this)\b`),
			re(`\b(i'll|i will|let me) stop here (because|due to|since|as|and give up)\b`),
			re(`\b(i surrender|surrendering)\b`),
			re(`\bfailed to (find a way|complete the goal|solve the issue)\b`),
		},
	},
	{
		class:    DeferredWork,
		baseline: choicetriage.FileTicket,
		reason:   "defers real work with no bounded ticket; in a headless run a punt is silently lost",
		resolve:  "file a bounded, DoD-scoped follow-on ticket, or do the work now",
		suppress: hasTicketRef,
		res: []*regexp.Regexp{
			re(`\btodo\b`),
			re(`\bfixme\b`),
			re(`\b(left|leaving) (this|it|that)( as)? a? ?(todo|follow-?up)\b`),
			re(`\b(can|could|will) be (done|added|addressed|handled|fixed|implemented) later\b`),
			re(`\bleave (this|that|it) (for|as) (a )?(follow-?up|later)\b`),
			re(`\b(defer|deferring|deferred|punt|punted|punting)\b`),
			re(`\bwe (can|could|should) revisit\b`),
			re(`\bout of scope for now\b`),
			re(`\bfor a future (pr|change|commit|iteration)\b`),
		},
	},
	{
		class:    SuggestionPunt,
		baseline: choicetriage.FreshContext,
		reason:   "hands the operator advice instead of acting; a headless worker executes the recommendation or files it",
		resolve:  "do it now if it is in scope, or file a ticket — do not leave advice for a reader who is not there",
		res: []*regexp.Regexp{
			re(`\byou (may|might|could|can) (want|wish) to\b`),
			re(`\byou'?d (want|likely want) to\b`),
			re(`\bit (would|might) be (a good idea|worth|wise|better) to\b`),
			re(`\bi'?d (recommend|suggest) (that )?you\b`),
			re(`\bconsider (adding|running|doing|using|refactoring|enabling)\b`),
		},
	},
	{
		class:    OpenOffer,
		baseline: choicetriage.TakeObvious,
		reason:   "a dangling conditional offer; nobody will take it up in a headless run",
		resolve:  "if the offered work is worth doing, do it or file a ticket; otherwise drop the note",
		res: []*regexp.Regexp{
			re(`\blet me know if\b`),
			re(`\bjust let me know\b`),
			re(`\bfeel free to\b`),
			re(`\bhappy to\b`),
			re(`\bif you('?d| would)? (like|want|prefer)\b`),
			re(`\bif that works for you\b`),
			re(`\bif you're happy\b`),
		},
	},
}

// Scan folds a text into a Report: one Finding per offending line (the first
// matching Class wins, so counts stay clean), each with the remediation an
// autonomous worker takes instead of asking.
func Scan(text string) Report {
	rep := Report{Schema: Schema, Verdict: Clean, Classes: map[Class]int{}}
	for i, raw := range splitLines(text) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		for _, sp := range specs {
			if sp.suppress != nil && sp.suppress(low) {
				continue
			}
			m := firstMatch(sp.res, low, line)
			if m == "" {
				continue
			}
			disp, reason, resolve, needsHuman := remediation(sp, line)
			rep.Findings = append(rep.Findings, Finding{
				Class:       sp.class,
				Line:        i + 1,
				Match:       clip(m, 120),
				Excerpt:     clip(line, 200),
				Disposition: disp,
				Reason:      reason,
				Resolve:     resolve,
				NeedsHuman:  needsHuman,
			})
			rep.Classes[sp.class]++
			if needsHuman {
				rep.NeedsHuman++
			}
			break
		}
	}
	rep.Count = len(rep.Findings)
	if rep.Count > 0 {
		rep.Verdict = OperatorDirected
	}
	return rep
}

// remediation computes what to do instead of the note. The two EARNED overrides
// come from choicetriage: a line naming authority (release/auth/policy) folds to
// HUMAN_RESIDUAL — a genuine escalation to route, never an inline question — and
// an oversized-scope line folds to FILE_TICKET. Otherwise the Class baseline
// applies.
func remediation(sp classSpec, line string) (choicetriage.Disposition, string, string, bool) {
	v := choicetriage.Triage(choicetriage.Signal{Question: line})
	switch v.Disposition {
	case choicetriage.HumanResidual:
		return choicetriage.HumanResidual,
			"names authority only a person holds — emit a typed escalation, never an inline question",
			"route it to the escalation channel (a different agent/model/tool), not the operator",
			true
	case choicetriage.FileTicket:
		return choicetriage.FileTicket,
			"the residual is real but too large for this context — decompose it",
			"file (or claim) a bounded, DoD-scoped ticket and drive it end-to-end",
			false
	}
	return sp.baseline, sp.reason, sp.resolve, false
}

// firstMatch returns the original-case substring of the first pattern that
// matches the lowercased line, or "" if none match.
func firstMatch(res []*regexp.Regexp, low, orig string) string {
	for _, r := range res {
		if loc := r.FindStringIndex(low); loc != nil {
			return strings.TrimSpace(orig[loc[0]:loc[1]])
		}
	}
	return ""
}

func splitLines(text string) []string {
	return strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
