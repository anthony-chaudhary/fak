// Package sessionobs scores the OBSERVABILITY of our own coding-session data for
// RSI loops. It is the value-side complement to tools/session_audit.py.
//
// Tier: foundation (1) -- see internal/architest. It is a pure analysis primitive:
// it imports only the standard library, has no I/O, and never touches the request
// path. The impure shell (a `fak sessions` command) reads transcripts + git, builds
// the Records and the Pipeline facts, and calls Score; this package owns only the
// deterministic scoring.
//
// THE PROBLEM IT MEASURES. The fleet runs thousands of Claude Code coding sessions a
// day, each leaving a transcript JSONL on the host that produced it.
// tools/session_audit.py already folds those transcripts into exact COST observability
// -- tokens, tool mix, cache reuse, dollars. That answers "what did a session SPEND".
// It cannot answer "what did a session ACHIEVE": there is no link from a session to
// its OUTCOME (a shipped + witnessed commit vs a STOP/interrupt, or a CLAIMED_CLOSED
// that no witness confirms). Without that outcome link the corpus is unlearnable -- an
// RSI loop cannot contrast the behavior of value-producing sessions against wasteful
// ones, because it cannot tell them apart. Guard refusals stay behavior signals, not
// cohort-defining outcome evidence.
//
// THE LADDER. Session data becomes RSI-useful one rung at a time, and this scorecard
// grades how far up the ladder the pipeline has climbed:
//
//	capture   -- the data exists and is retained (transcripts on disk).
//	structure -- each session is an analysis-shaped record, not an opaque blob.
//	link      -- each record is tied to its OUTCOME (the missing rung today).
//	aggregate -- the records carry behavior signals a loop can contrast across sessions.
//	learn     -- a registered RSI loop READS the committed corpus and changes behavior.
//
// The headline integer is sessionobs_debt: the count of HARD rungs not yet built. It
// follows the repo's scorecard doctrine (see .claude/skills/scorecard) -- deterministic
// (no clock, no RNG: two callers with the same inputs score identically), and you retire
// debt by BUILDING the real rung (ingest outcomes, commit a scrubbed corpus, wire a
// loop), never by weakening a check.
//
// THE COMMITTABLE-CORPUS DISCIPLINE. A Record carries only structured signal -- counts,
// durations, an outcome class, behavior flags -- and NEVER raw prompt or result prose.
// That is deliberate: the scrubbed corpus is what a fleet commits and folds across hosts
// without leaking private session content. The prose stays on the host; only the signal
// travels.
package sessionobs

import (
	"fmt"
	"io"
	"sort"
)

// Outcome classifies what a coding session PRODUCED -- the value-vs-waste axis that
// cost accounting cannot see. The buckets mirror the fleet's closure-honesty ledger
// (TRUE_RESOLVED / CLAIMED_CLOSED / ...) so a session outcome and an issue closure
// speak the same vocabulary.
type Outcome uint8

const (
	// OutcomeUnknown is a record not yet linked to any evidence -- the unobservable
	// default. A corpus full of Unknown is the failure this scorecard exists to catch.
	OutcomeUnknown Outcome = iota
	// OutcomeShipped: the session landed at least one commit whose claim a witness
	// later confirmed (the TRUE_RESOLVED analog). This is the "value" exemplar.
	OutcomeShipped
	// OutcomeClaimed: the session landed a commit but no witness confirms its claim
	// (the CLAIMED_CLOSED analog) -- value asserted, not proven.
	OutcomeClaimed
	// OutcomeStopped: the session ended at a STOP or interrupt with no commit landed
	// -- the "waste" exemplar a loop must be able to see. Guard refusals are stored
	// as Signals so the value-vs-waste contrast can rank them without circularity.
	OutcomeStopped
	// OutcomeNoOp: a read-only / exploratory session that mutated nothing by design
	// (a question answered, a tree explored). Not waste, but not a value ship either;
	// kept distinct so it never dilutes the value-vs-waste contrast.
	OutcomeNoOp
)

// String renders the outcome as its lowercase wire token; an out-of-range value
// renders "unknown" rather than panicking.
func (o Outcome) String() string {
	switch o {
	case OutcomeShipped:
		return "shipped"
	case OutcomeClaimed:
		return "claimed"
	case OutcomeStopped:
		return "stopped"
	case OutcomeNoOp:
		return "noop"
	default:
		return "unknown"
	}
}

// linked reports whether the outcome was derived from real evidence (anything but
// Unknown). The link rate over a corpus is the headline observability KPI.
func (o Outcome) linked() bool { return o != OutcomeUnknown }

// value reports whether the outcome is on the value side of the value-vs-waste axis
// (a real or claimed ship). NoOp and Stopped are not value; Unknown is unknown.
func (o Outcome) value() bool { return o == OutcomeShipped || o == OutcomeClaimed }

// ClassifyOutcome is the deterministic value-vs-waste classifier the impure shell
// calls once per session with primitive evidence booleans. Keeping the policy here
// (not in the disk shell) makes it unit-testable and keeps every caller consistent:
//
//   - a commit landed AND a witness confirmed its claim -> Shipped (proven value).
//   - a commit landed but no witness yet                -> Claimed (asserted value).
//   - no commit, and the session hit a STOP/interrupt   -> Stopped (waste).
//   - no commit, no stop, and nothing was mutated       -> NoOp (exploratory).
//   - none of the above                                 -> Unknown (unlinked).
func ClassifyOutcome(committed, witnessed, stopped, mutated bool) Outcome {
	switch {
	case committed && witnessed:
		return OutcomeShipped
	case committed:
		return OutcomeClaimed
	case stopped:
		return OutcomeStopped
	case !mutated:
		return OutcomeNoOp
	default:
		return OutcomeUnknown
	}
}

// Signals are the per-session BEHAVIOR features an RSI loop contrasts between value
// and waste sessions. Every field is a count derived from the transcript or the
// linked outcome -- structured, never prose, so the record stays committable.
type Signals struct {
	GuardRefusals int `json:"guard_refusals,omitempty"` // turns the kernel DENIED a proposed tool call (not result quarantines)
	ToolErrors    int `json:"tool_errors,omitempty"`    // is_error tool_results
	Interrupts    int `json:"interrupts,omitempty"`     // interrupted assistant turns
	Commits       int `json:"commits,omitempty"`        // commits the session landed (linked from git)
	StopEvents    int `json:"stop_events,omitempty"`    // STOP-hook fires / goal-block events
	GoalEvents    int `json:"goal_events,omitempty"`    // /goal directives the session ran under
}

// any reports whether the session carried ANY behavior signal -- the aggregate-rung
// test (a record with an outcome but zero features still teaches a loop very little).
func (s Signals) any() bool {
	return s.GuardRefusals > 0 || s.ToolErrors > 0 || s.Interrupts > 0 ||
		s.Commits > 0 || s.StopEvents > 0 || s.GoalEvents > 0
}

// Record is one session's scrubbed observability record -- the unit of a corpus that
// is safe to COMMIT and AGGREGATE across hosts. It carries only structured signal and
// NEVER raw prompt/result prose.
type Record struct {
	SessionID      string  `json:"session_id"`           // transcript uuid (an opaque id, not content)
	Namespace      string  `json:"namespace,omitempty"`  // project namespace (e.g. C--work-fak)
	Account        string  `json:"account,omitempty"`    // fleet account/home that produced it
	StartUnix      int64   `json:"start_unix,omitempty"` // first timestamped record
	EndUnix        int64   `json:"end_unix,omitempty"`   // last timestamped record
	AssistantTurns int     `json:"assistant_turns"`      // billed assistant turns (de-duplicated)
	ToolCalls      int     `json:"tool_calls"`           // tool_use blocks
	ReadOnlyCalls  int     `json:"read_only_calls"`      // tool_use blocks against read-only tools
	OutputTokens   int64   `json:"output_tokens"`        // the work actually generated
	Outcome        Outcome `json:"outcome"`              // the value-vs-waste class
	Signals        Signals `json:"signals,omitempty"`    // behavior features for the loop
}

// structured reports whether the record is analysis-shaped (has at least the turn
// structure a folder needs) rather than an empty husk.
func (r Record) structured() bool { return r.AssistantTurns > 0 }

// Pipeline carries the pipeline-state facts the score needs that are NOT derivable
// from the corpus rows -- whether the machinery to CAPTURE durably, COMMIT, and
// CONSUME the corpus exists. The impure shell discovers these from the tree/host; a
// test supplies fixtures. Passing them in is what keeps Score a pure function.
type Pipeline struct {
	// CorpusCommitted: a scrubbed session corpus is durable in-tree (not just on one
	// host's transient disk), the precondition for cross-host aggregation.
	CorpusCommitted bool `json:"corpus_committed"`
	// LoopConsumes: a registered RSI loop READS the corpus to change behavior -- the
	// difference between observability that informs and observability that just sits.
	LoopConsumes bool `json:"loop_consumes"`
	// Registered: the scorecard is wired into the control-pane ratchet so its debt
	// folds into the portfolio total and a regression reds the gate.
	Registered bool `json:"registered"`
}

// KPI is one graded rung. Score is 0..100; Debt counts the HARD failures it
// contributes (0 or 1 here -- each KPI is one rung). Detail is a human one-liner.
type KPI struct {
	Name   string `json:"kpi"`
	Group  string `json:"group"`
	Hard   bool   `json:"hard"`
	Score  int    `json:"score"`
	Debt   int    `json:"debt"`
	Detail string `json:"detail"`
}

// Report is the control-pane envelope every fak scorecard emits, specialized to the
// session-observability surface. The control pane reads corpus.sessionobs_debt and
// corpus.grade; the keys are stable.
type Report struct {
	Schema     string `json:"schema"`
	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"`
	Finding    string `json:"finding"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
	Corpus     Corpus `json:"corpus"`
	KPIs       []KPI  `json:"kpis"`
}

// Corpus is the headline summary block.
type Corpus struct {
	Sessions       int     `json:"sessions"`
	LinkedFrac     float64 `json:"linked_frac"`     // share of records with a known outcome
	ValueFrac      float64 `json:"value_frac"`      // share of records on the value side
	WasteFrac      float64 `json:"waste_frac"`      // share of records that are Stopped (waste)
	Score          int     `json:"score"`           // 0..100 weighted composite
	Grade          string  `json:"grade"`           // A..F
	SessionObsDebt int     `json:"sessionobs_debt"` // the headline integer: count of HARD rungs missing
}

const schema = "fak.sessionobs.v1"

// Score is the whole scorecard: a pure, deterministic function from a corpus and the
// pipeline facts to the control-pane Report. Same inputs -> identical output, always.
func Score(corpus []Record, pipe Pipeline) Report {
	n := len(corpus)
	var linked, value, waste, structured, withSignals int
	for _, r := range corpus {
		if r.Outcome.linked() {
			linked++
		}
		if r.Outcome.value() {
			value++
		}
		if r.Outcome == OutcomeStopped {
			waste++
		}
		if r.structured() {
			structured++
		}
		if r.Signals.any() {
			withSignals++
		}
	}
	linkedFrac := frac(linked, n)
	valueFrac := frac(value, n)
	wasteFrac := frac(waste, n)

	var kpis []KPI

	// --- capture rung -------------------------------------------------------
	// A corpus with no rows can score nothing else honestly; the row count IS the
	// gate (the empty-journal-honesty law, borrowed from guard-rsi-score).
	kpis = append(kpis, hardKPI("corpus_nonempty", "capture", n > 0,
		fmt.Sprintf("%d session records ingested", n),
		"no session records -- run the ingester over the host's transcripts"))

	// --- structure rung -----------------------------------------------------
	// Each record must be analysis-shaped (carry turn structure), not an empty husk.
	structuredFrac := frac(structured, n)
	kpis = append(kpis, ratioKPI("records_structured", "structure", structuredFrac, 0.95, true,
		fmt.Sprintf("%d/%d records carry turn structure (%.0f%%)", structured, n, 100*structuredFrac)))

	// --- link rung (the missing one) ----------------------------------------
	// The headline: can a session be tied to its outcome at all?
	kpis = append(kpis, ratioKPI("outcome_link_rate", "link", linkedFrac, 0.80, true,
		fmt.Sprintf("%d/%d records linked to an outcome (%.0f%%) -- the value-vs-waste rung", linked, n, 100*linkedFrac)))
	// And does the corpus actually contain BOTH value and waste, so a loop has a
	// contrast to learn from? An all-one-class corpus teaches nothing.
	separable := value > 0 && waste > 0
	kpis = append(kpis, hardKPI("value_waste_separable", "link", separable,
		fmt.Sprintf("value=%d waste=%d -- both classes present", value, waste),
		"corpus lacks a value/waste contrast -- a loop cannot learn what produces value"))

	// --- aggregate rung -----------------------------------------------------
	// Records must carry behavior features, not just an outcome label, or the loop
	// has nothing to correlate the outcome WITH. SOFT: useful records, not a gate.
	signalFrac := frac(withSignals, n)
	kpis = append(kpis, ratioKPI("behavior_signal_present", "aggregate", signalFrac, 0.50, false,
		fmt.Sprintf("%d/%d records carry behavior signals (%.0f%%)", withSignals, n, 100*signalFrac)))

	// --- learn / RSI rung ---------------------------------------------------
	kpis = append(kpis, hardKPI("corpus_committed", "learn", pipe.CorpusCommitted,
		"a scrubbed session corpus is durable in-tree",
		"corpus lives only on one host's disk -- commit a scrubbed corpus so the fleet can fold it"))
	kpis = append(kpis, hardKPI("loop_consumes", "learn", pipe.LoopConsumes,
		"a registered RSI loop reads the corpus",
		"nothing consumes the corpus -- wire a loop that reads it and changes behavior"))
	kpis = append(kpis, softKPI("registered_in_control_pane", "learn", pipe.Registered,
		"scorecard folds into the control-pane ratchet",
		"scorecard is not in the control pane -- its debt does not gate regressions yet"))

	// --- fold ---------------------------------------------------------------
	debt := 0
	for _, k := range kpis {
		debt += k.Debt
	}
	score := compositeScore(kpis)
	grade := gradeLetter(score)

	rep := Report{
		Schema: schema,
		OK:     debt == 0,
		Corpus: Corpus{
			Sessions:       n,
			LinkedFrac:     round3(linkedFrac),
			ValueFrac:      round3(valueFrac),
			WasteFrac:      round3(wasteFrac),
			Score:          score,
			Grade:          grade,
			SessionObsDebt: debt,
		},
		KPIs: kpis,
	}
	rep.Verdict, rep.Finding, rep.Reason, rep.NextAction = verdict(debt, kpis)
	return rep
}

// verdict picks the one-line finding + the worst-first next action from the KPI set.
func verdict(debt int, kpis []KPI) (verdict, finding, reason, next string) {
	if debt == 0 {
		return "OK",
			"session-data pipeline is observable for RSI: linked, separable, and consumed",
			"every HARD rung of the observability ladder is built",
			"keep the loop closing on fresh sessions; weigh the SOFT signals"
	}
	worst := worstHard(kpis)
	return "ACTION",
		fmt.Sprintf("session-data observability incomplete: %d HARD rung(s) missing", debt),
		fmt.Sprintf("worst-first rung: %s -- %s", worst.Name, worst.Detail),
		worst.fix()
}

// fix is the next-action hint for a failed KPI -- the Detail of a failing KPI already
// carries the "fix by" message for the hard/soft helpers (we stored it as Detail when
// failing), so surface it directly.
func (k KPI) fix() string { return k.Detail }

// worstHard returns the first failing HARD KPI in ladder order (the slice is already
// in rung order, so the first failure is the lowest unbuilt rung -- the worst-first pick).
func worstHard(kpis []KPI) KPI {
	for _, k := range kpis {
		if k.Hard && k.Debt > 0 {
			return k
		}
	}
	// No HARD failure (only SOFT lowered the score). Return the first failing SOFT.
	for _, k := range kpis {
		if k.Debt > 0 || k.Score < 100 {
			return k
		}
	}
	return kpis[0]
}

// hardKPI builds a pass/fail HARD rung. On failure it scores 0, contributes 1 debt,
// and its Detail carries the fix hint (so worstHard surfaces the recovery directly).
func hardKPI(name, group string, ok bool, passDetail, failFix string) KPI {
	if ok {
		return KPI{Name: name, Group: group, Hard: true, Score: 100, Debt: 0, Detail: passDetail}
	}
	return KPI{Name: name, Group: group, Hard: true, Score: 0, Debt: 1, Detail: failFix}
}

// softKPI builds a pass/fail SOFT rung: it lowers the score but is NEVER debt.
func softKPI(name, group string, ok bool, passDetail, failHint string) KPI {
	if ok {
		return KPI{Name: name, Group: group, Hard: false, Score: 100, Debt: 0, Detail: passDetail}
	}
	return KPI{Name: name, Group: group, Hard: false, Score: 0, Debt: 0, Detail: failHint}
}

// ratioKPI grades a fraction against a threshold. The Score is the rounded percentage
// (so a near-miss reads honestly); a HARD ratio below threshold is 1 debt.
func ratioKPI(name, group string, got, threshold float64, hard bool, detail string) KPI {
	k := KPI{Name: name, Group: group, Hard: hard, Score: int(round(100 * got)), Detail: detail}
	if got < threshold && hard {
		k.Debt = 1
	}
	return k
}

// compositeScore is the weighted mean of the KPI scores: HARD rungs weigh 2, SOFT 1.
// It is monotone -- building a rung never lowers the score -- so it can headline a
// "are we getting better" trend independent of the debt count.
func compositeScore(kpis []KPI) int {
	var num, den float64
	for _, k := range kpis {
		w := 1.0
		if k.Hard {
			w = 2.0
		}
		num += w * float64(k.Score)
		den += w
	}
	if den == 0 {
		return 0
	}
	return int(round(num / den))
}

func gradeLetter(score int) string {
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

func frac(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func round(x float64) float64 {
	if x < 0 {
		return -round(-x)
	}
	return float64(int64(x + 0.5))
}

func round3(x float64) float64 { return round(1000*x) / 1000 }

// Render writes the human work-list -- the headline, then the KPIs worst-first
// (failures before passes, HARD before SOFT). It is the terminal view a skill reads.
func Render(w io.Writer, rep Report) {
	c := rep.Corpus
	fmt.Fprintf(w, "session-observability scorecard (RSI): %s grade %s, sessionobs_debt=%d\n",
		rep.Verdict, c.Grade, c.SessionObsDebt)
	fmt.Fprintf(w, "  sessions=%d  linked=%.0f%%  value=%.0f%%  waste=%.0f%%  score=%d\n",
		c.Sessions, 100*c.LinkedFrac, 100*c.ValueFrac, 100*c.WasteFrac, c.Score)
	fmt.Fprintf(w, "  finding: %s\n", rep.Finding)
	fmt.Fprintf(w, "  next: %s\n", rep.NextAction)

	ranked := append([]KPI(nil), rep.KPIs...)
	sort.SliceStable(ranked, func(i, j int) bool {
		fi, fj := ranked[i].Debt > 0 || ranked[i].Score < 100, ranked[j].Debt > 0 || ranked[j].Score < 100
		if fi != fj {
			return fi // failing/partial first
		}
		if ranked[i].Hard != ranked[j].Hard {
			return ranked[i].Hard // HARD before SOFT among equals
		}
		return false
	})
	for _, k := range ranked {
		mark := "ok "
		if k.Debt > 0 {
			mark = "DEBT"
		} else if k.Score < 100 {
			mark = "soft"
		}
		tag := "SOFT"
		if k.Hard {
			tag = "HARD"
		}
		fmt.Fprintf(w, "  [%s] %-26s %-9s %3d  %s\n", mark, k.Name, "("+tag+")", k.Score, k.Detail)
	}
}
