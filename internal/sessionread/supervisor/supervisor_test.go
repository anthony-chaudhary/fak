package supervisor

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/directory"
	"github.com/anthony-chaudhary/fak/internal/sessionread/query"
	"github.com/anthony-chaudhary/fak/internal/sessionread/transcriptfeed"
)

// rowWithUUID builds a minimal C4 directory row carrying the trace<->UUID address pair.
func rowWithUUID(trace, uuid string) directory.DirectoryRow {
	return directory.DirectoryRow{TraceID: trace, UUID: uuid, Evidence: sessionread.EvidenceObserved}
}

// --- fixtures -------------------------------------------------------------------------

// fixtureChecker is a deterministic CHECK seam: it reports a session stuck iff its trace is
// in the stuck set, grounding the verdict in the (fixture) evidence rather than any
// self-report. A real Checker binds to the dos_* truth verbs; this stand-in makes the
// supervisor's decision loop testable in isolation.
type fixtureChecker struct{ stuck map[string]bool }

func (c fixtureChecker) Check(v SessionView) Verdict {
	if c.stuck[v.Trace] {
		return Verdict{
			Stuck:    true,
			Reason:   CheckStalledNoProgress,
			Evidence: sessionread.EvidenceObserved,
			Detail:   "no forward progress witnessed",
		}
	}
	return Verdict{
		Progressing: true,
		Reason:      CheckLiveProgressing,
		Evidence:    sessionread.EvidenceObserved,
		Detail:      "witnessed forward motion",
	}
}

// call is one recorded Controller verb invocation.
type call struct {
	Verb       Verb
	Target     string
	Capability Capability
	Payload    string
	Reason     string
}

// fixtureController is a recording, capability-checked CONTROL seam. It records EVERY verb
// call (so a test can assert exactly which interventions fired) and enforces req.Capability
// against the set it holds — a verb whose capability it lacks refuses (Applied=false), the
// capability-checked property. A real Controller binds these verbs to the #2753 control route.
type fixtureController struct {
	caps  map[Capability]bool
	calls []call
}

func newController(caps ...Capability) *fixtureController {
	held := map[Capability]bool{}
	for _, c := range caps {
		held[c] = true
	}
	return &fixtureController{caps: held}
}

func witnessKindFor(v Verb) string {
	switch v {
	case VerbHold:
		return "boundary-stop"
	case VerbApprove:
		return "same-turn-wake"
	case VerbSteer:
		return "splice"
	case VerbRedirect:
		return "directive"
	default:
		return ""
	}
}

func (c *fixtureController) record(v Verb, req ControlRequest) (Witness, error) {
	c.calls = append(c.calls, call{
		Verb: v, Target: req.Target, Capability: req.Capability,
		Payload: req.Payload, Reason: req.Reason,
	})
	if !c.caps[req.Capability] {
		return Witness{
			Verb: v, Target: req.Target, Capability: req.Capability,
			Applied: false, Detail: "capability not held",
		}, fmt.Errorf("capability %q not held", req.Capability)
	}
	return Witness{
		Verb: v, Target: req.Target, Capability: req.Capability,
		Applied: true, Kind: witnessKindFor(v), Detail: "fixture applied",
	}, nil
}

func (c *fixtureController) Hold(req ControlRequest) (Witness, error) { return c.record(VerbHold, req) }
func (c *fixtureController) Steer(req ControlRequest) (Witness, error) {
	return c.record(VerbSteer, req)
}
func (c *fixtureController) Redirect(req ControlRequest) (Witness, error) {
	return c.record(VerbRedirect, req)
}
func (c *fixtureController) Approve(req ControlRequest) (Witness, error) {
	return c.record(VerbApprove, req)
}

func (c *fixtureController) callsFor(trace string) []call {
	var out []call
	for _, cc := range c.calls {
		if cc.Target == trace {
			out = append(out, cc)
		}
	}
	return out
}

// turn builds an assistant turn at an index with the given text (a decision turn).
func asst(idx int, text string) query.Turn {
	return query.Turn{Index: idx, Role: "assistant", Text: text}
}

// userTurn builds a user turn.
func userTurn(idx int, text string) query.Turn {
	return query.Turn{Index: idx, Role: "user", Text: text}
}

// confirmTail returns turns whose TAIL is an unresolved awaiting-confirm assistant decision.
func confirmTail() []query.Turn {
	return []query.Turn{
		userTurn(0, "please run the migration"),
		asst(1, "planning the migration"),
		asst(2, "await-confirm: apply the destructive migration?"),
	}
}

// progressingTurns returns turns with no confirm gate (a plain working session).
func progressingTurns() []query.Turn {
	return []query.Turn{
		userTurn(0, "add the feature"),
		asst(1, "implemented and tested the feature"),
	}
}

// --- Test 1: read-only fleet drive, control only the stuck session --------------------

func TestDrivesFleetReadOnlyThenControlsOnlyStuck(t *testing.T) {
	fleet := []SessionView{
		{Trace: "s-stuck", Turns: progressingTurns(), Score: 0.10},   // stuck + low-score => candidate
		{Trace: "s-healthy", Turns: progressingTurns(), Score: 0.90}, // progressing + high-score => protected
	}
	chk := fixtureChecker{stuck: map[string]bool{"s-stuck": true}}
	ctl := newController(CapOperatorSend, CapOperatorControl)
	gate := RegimeGate{Threshold: 0.50}

	ivs := Supervise(fleet, chk, ctl, gate)

	// Exactly one intervention, on the stuck session, through a typed WITNESSED verb.
	if len(ivs) != 1 {
		t.Fatalf("want exactly 1 intervention, got %d: %+v", len(ivs), ivs)
	}
	iv := ivs[0]
	if iv.Target != "s-stuck" {
		t.Fatalf("intervention targeted %q, want s-stuck", iv.Target)
	}
	if iv.Verb != VerbHold {
		t.Fatalf("verb = %q, want %q (conservative default for a plain stall)", iv.Verb, VerbHold)
	}
	if !iv.Witness.Applied {
		t.Fatalf("intervention not witnessed-applied: %+v", iv.Witness)
	}
	if iv.Witness.Verb != VerbHold || iv.Witness.Kind == "" {
		t.Fatalf("witness is not a real control receipt: %+v", iv.Witness)
	}
	if iv.Evidence != sessionread.EvidenceObserved {
		t.Fatalf("intervention evidence = %q, want OBSERVED", iv.Evidence)
	}

	// The Controller was called EXACTLY once, and never for the healthy session.
	if len(ctl.calls) != 1 {
		t.Fatalf("controller called %d times, want 1: %+v", len(ctl.calls), ctl.calls)
	}
	if got := ctl.callsFor("s-healthy"); len(got) != 0 {
		t.Fatalf("healthy session received %d control calls, want 0: %+v", len(got), got)
	}
	if ctl.calls[0].Verb != VerbHold || ctl.calls[0].Target != "s-stuck" {
		t.Fatalf("recorded call = %+v, want Hold on s-stuck", ctl.calls[0])
	}
}

// --- Test 2: regime gate protects a healthy/high-score fleet --------------------------

func TestRegimeGateHonoredNoActionOnHealthy(t *testing.T) {
	// A fleet of only high-score sessions — INCLUDING one the Checker flags stuck (the naive
	// heuristic). The regime gate must protect it because its score is high.
	fleet := []SessionView{
		{Trace: "h1", Turns: progressingTurns(), Score: 0.80},
		{Trace: "h2-flagged-stuck", Turns: confirmTail(), Score: 0.95}, // stuck-flagged but HIGH score
		{Trace: "h3", Turns: progressingTurns(), Score: 0.75},
	}
	chk := fixtureChecker{stuck: map[string]bool{"h2-flagged-stuck": true}}
	ctl := newController(CapOperatorSend, CapOperatorControl)
	gate := RegimeGate{Threshold: 0.50}

	ivs := Supervise(fleet, chk, ctl, gate)

	if len(ivs) != 0 {
		t.Fatalf("want 0 interventions on a healthy fleet, got %d: %+v", len(ivs), ivs)
	}
	if len(ctl.calls) != 0 {
		t.Fatalf("controller must not be called on a healthy fleet, got %d calls: %+v", len(ctl.calls), ctl.calls)
	}
}

// --- Test 3: fleet query returns exactly the confirm-gate-stuck subset ----------------

func TestFleetQueryConfirmGate(t *testing.T) {
	fleet := []SessionView{
		{Trace: "a-gated", Turns: confirmTail()},
		{Trace: "b-working", Turns: progressingTurns()},
		{Trace: "c-gated", Row: rowWithUUID("c-gated", "uuid-c"), Turns: confirmTail()},
		// A session that HAD a confirm decision but resolved it (a later turn follows) — NOT stuck.
		{Trace: "d-resolved", Turns: []query.Turn{
			userTurn(0, "go"),
			asst(1, "await-confirm: proceed?"),
			userTurn(2, "yes"),
			asst(3, "proceeding"),
		}},
	}

	hits := FleetQuery{}.ConfirmGateStuck(fleet)

	gotTraces := map[string]bool{}
	for _, h := range hits {
		gotTraces[h.Trace] = true
		if h.Evidence != sessionread.EvidenceObserved {
			t.Fatalf("hit %q evidence = %q, want OBSERVED", h.Trace, h.Evidence)
		}
	}
	want := map[string]bool{"a-gated": true, "c-gated": true}
	if len(gotTraces) != len(want) {
		t.Fatalf("confirm-gate hits = %v, want %v", gotTraces, want)
	}
	for tr := range want {
		if !gotTraces[tr] {
			t.Fatalf("missing expected confirm-gate hit %q; got %v", tr, gotTraces)
		}
	}
	// The C4 directory address rides through the hit.
	for _, h := range hits {
		if h.Trace == "c-gated" && h.UUID != "uuid-c" {
			t.Fatalf("c-gated hit lost its directory UUID: %+v", h)
		}
	}
}

// --- Test 4: control is never a read side effect --------------------------------------

func TestControlIsNeverAReadSideEffect(t *testing.T) {
	feed := transcriptfeed.NewFeed(0)
	for _, ev := range transcriptfeed.EventsFromRecords(nil, "p1") {
		feed.Append(ev)
	}
	fleet := []SessionView{
		{Trace: "x", Turns: confirmTail(), Feed: feed, Principal: "p1"},
		{Trace: "y", Turns: progressingTurns()},
	}
	ctl := newController(CapOperatorSend, CapOperatorControl)

	// Exercise the READ surfaces directly: the fleet query, a raw query.Answer over the
	// turns, and a feed drain. NONE of these is handed the Controller.
	_ = FleetQuery{}.ConfirmGateStuck(fleet)
	for _, v := range fleet {
		if _, err := query.Answer(query.Query{Kind: query.KindToolFailures}, v.Turns, sessionread.DisclosureRedacted); err != nil {
			t.Fatalf("read query errored: %v", err)
		}
		if v.Feed != nil {
			_, _ = v.Feed.Drain(v.Principal, 0)
		}
	}

	if len(ctl.calls) != 0 {
		t.Fatalf("a pure read pass called the controller %d times, want 0: %+v", len(ctl.calls), ctl.calls)
	}
}

// --- Test 5: every intervention goes through a typed, witnessed, capability-checked verb ---

func TestInterventionGoesThroughTypedVerb(t *testing.T) {
	// A stuck, low-score session halted at a confirm gate => the supervisor fires Approve.
	fleet := []SessionView{
		{Trace: "gated-stuck", Turns: confirmTail(), Score: 0.05},
	}
	chk := fixtureChecker{stuck: map[string]bool{"gated-stuck": true}}
	ctl := newController(CapOperatorSend, CapOperatorControl)
	gate := RegimeGate{Threshold: 0.50}

	ivs := Supervise(fleet, chk, ctl, gate)

	if len(ivs) != 1 {
		t.Fatalf("want 1 intervention, got %d", len(ivs))
	}
	iv := ivs[0]

	// It is a REAL typed verb (one of the closed #2753-modeled set), not a bare boolean.
	switch iv.Verb {
	case VerbHold, VerbSteer, VerbRedirect, VerbApprove:
	default:
		t.Fatalf("intervention verb %q is not a typed control verb", iv.Verb)
	}
	if iv.Verb != VerbApprove {
		t.Fatalf("confirm-gate stall should fire Approve, got %q", iv.Verb)
	}

	// It is CAPABILITY-CHECKED: dispatched with the verb's required capability, which the
	// Controller confirmed it held.
	if iv.Capability != VerbApprove.RequiredCapability() {
		t.Fatalf("dispatched capability %q != verb's required %q", iv.Capability, VerbApprove.RequiredCapability())
	}
	if iv.Capability != CapOperatorControl {
		t.Fatalf("Approve capability = %q, want operator-control", iv.Capability)
	}

	// It is WITNESSED: a receipt with the verb, target, capability, and applied proof.
	if !iv.Witness.Applied || iv.Witness.Verb != VerbApprove || iv.Witness.Target != "gated-stuck" || iv.Witness.Kind == "" {
		t.Fatalf("intervention is not a witnessed control receipt: %+v", iv.Witness)
	}
	if iv.Err != "" {
		t.Fatalf("clean apply should carry no error, got %q", iv.Err)
	}

	// The Controller actually recorded the typed Approve call with the checked capability.
	got := ctl.callsFor("gated-stuck")
	if len(got) != 1 || got[0].Verb != VerbApprove || got[0].Capability != CapOperatorControl {
		t.Fatalf("recorded control call = %+v, want one Approve with operator-control", got)
	}
}
