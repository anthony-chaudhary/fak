package sessionctl

// vocab.go — the control-op VOCABULARY SPINE of the out-of-band operator control
// epic (#2754, epic #2753): the one first-class, closed type every operator control
// op belongs to, and the uniform contract for what "applied" means per op.
//
// Why this exists. Operator control shipped fragmented: `fak session` drives the
// drive-state ops (pause/resume/stop→cancel/throttle/budget/pace/priority), `fak
// signal` adds job-control plus the freeform `steer`, and #2755 added `redirect`
// here. Each op independently re-invents "who may send it", "when does it take",
// "how is it proven consumed", and "how does it refuse when illegal". redirect.go
// (#2755) and internal/session/ctlrefuse.go (#2766) both defer their ENVELOPE to
// "the vocabulary spine (#2754)"; this file IS that spine. It does not re-implement
// any op — it NAMES each shipped op with its four fixed properties, grounded in the
// real tokens the ops already refuse with, so the grammar is defined once.
//
// The four fixed properties (per op):
//
//  1. Capability — the send-right required to submit the op.
//  2. Boundary   — when the op takes effect relative to the loop.
//  3. Witness    — the SHAPE of the loop-side proof it was CONSUMED (not merely
//     enqueued): the enqueue/table write is never the witness; the running arm
//     observing it at a boundary is.
//  4. RefusalReasons — the closed refusal token(s) the op surfaces when it is
//     illegal for the current run-state.
//
// The witness-of-applied is realized loop-side by the shipped #2766 table
// internal/agent/loop_control_witness_test.go; this spine declares which witness
// SHAPE each op carries so the two agree. The add-constraint op (#2756) is not yet
// shipped and so is deliberately NOT registered here — a new op registers its row
// (and its tokens) when its child lands, exactly as the #2766 table requires.

import (
	"slices"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// ControlOp is the closed set of out-of-band operator control ops at HEAD. It is a
// string enum so the wire/CLI verb and the vocabulary key are the same token. A new
// op MUST add its constant here and its spec to vocabulary — the completeness test
// names any op missing either.
type ControlOp string

const (
	// OpSteer delivers freeform operator INPUT, spliced into the running session's
	// next turn as a user message. The injection-shaped escape hatch every other op
	// exists to make structured.
	OpSteer ControlOp = "steer"
	// OpRedirect changes WHAT the session pursues — it lands a first-class objective
	// directive (#2755), not another operator turn.
	OpRedirect ControlOp = "redirect"
	// OpPause holds the running arm at its next boundary (resumable).
	OpPause ControlOp = "pause"
	// OpResume wakes a parked (paused) arm and completes its held turn.
	OpResume ControlOp = "resume"
	// OpCancel drains the session to a safe stop at its next boundary (the drive-state
	// verb `fak session stop` maps here — enqueue Draining, finalize at the boundary).
	OpCancel ControlOp = "cancel"
	// OpThrottle lowers the per-turn output cap into the current turn's sampling.
	OpThrottle ControlOp = "throttle"
	// OpBudget re-allots the session's turn/token/context budget; an exhausted
	// allotment stops the arm at the boundary with its closed exhaustion reason.
	OpBudget ControlOp = "budget"
	// OpPriority re-ranks the session for the scheduler's next pick.
	OpPriority ControlOp = "priority"
)

// Capability is the closed set of send-rights an op requires. The spine NAMES the
// requirement per op; it does NOT itself enforce, and today only steer's named right is
// actually wired (its a2achan send floor fails a capless send closed). The drive-state
// control route and redirect do not yet gate on a capability — their rows record the
// right their transport SHOULD check when the gate is wired.
type Capability string

const (
	// CapOperatorSend is the adjudicated a2achan send-right (a2achan.CapA2ASend). Steer
	// enforces it today: it enqueues onto the taint-aware operator bus and fails closed
	// without it (a capless steer refuses DEFAULT_DENY; a tainted/over-scoped one refuses
	// TRUST_VIOLATION). Redirect is ASSIGNED the same send-right requirement but does not
	// yet ride the bus — EnqueueRedirect appends to its own per-session mailbox and today
	// refuses only on shape/state; its transport will enforce the right when wired.
	CapOperatorSend Capability = "operator-send"
	// CapOperatorControl is the send-right the drive-state control route SHOULD gate for
	// POST /v1/fak/session/{id}/{verb} (pause/resume/stop/throttle/budget/priority). It is
	// NOT yet enforced: that route today applies only run-state/CAS legality (the closed
	// CONTROL_* refusals), with no capability floor. The spine records the requirement so
	// the future gate has a name to adopt.
	CapOperatorControl Capability = "operator-control"
)

// Boundary is the closed vocabulary for WHEN an op takes effect relative to the loop
// — the three-value grammar #2754 fixes. It is a deliberate RE-PROJECTION of the #2766
// witness table's boundary column {next turn, same turn, next pick}, not a 1:1 mirror:
// NextTurn absorbs #2766 "next turn" (steer/redirect/pause/throttle/budget); Quiesce
// splits cancel out of "next turn" to name its drain; Immediate merges #2766 "same turn"
// (resume) and "next pick" (priority). Each op's Summary states the precise per-op
// behavior the rollup elides, so no timing detail is lost — only unindexed by Boundary.
type Boundary string

const (
	// BoundaryNextTurn — taken at the loop's next turn boundary (steer, redirect,
	// pause, throttle, budget). The op waits for a clean boundary; it never lands
	// mid-decode.
	BoundaryNextTurn Boundary = "next-turn"
	// BoundaryQuiesce — the session drains to a safe stop at the boundary (cancel):
	// Draining is enqueued and finalized to Stopped at the next boundary, never
	// mid-turn.
	BoundaryQuiesce Boundary = "quiesce"
	// BoundaryImmediate — consumed without waiting for a fresh next-turn gate: resume
	// wakes the HELD turn in place; priority is read by the scheduler at its next pick,
	// out of band of the turn loop.
	BoundaryImmediate Boundary = "immediate"
)

// WitnessKind is the closed set of loop-side proof SHAPES — the "witness-of-applied"
// contract generalized from steer's splice assertion across every op. Each kind names
// how internal/agent/loop_control_witness_test.go (#2766) proves the op was CONSUMED.
type WitnessKind string

const (
	// WitnessSplice — the op's payload is spliced into the turn's user INPUT and the
	// mailbox drains (steer). This is the original assertion the contract generalizes.
	WitnessSplice WitnessKind = "splice"
	// WitnessDirective — the op is carried into the turn as a standing SYSTEM directive
	// and the mailbox drains; the live objective reflects it (redirect).
	WitnessDirective WitnessKind = "directive"
	// WitnessBoundaryStop — the running arm HALTS at its boundary with the op's closed
	// stop-reason recorded on ArmMetrics.StoppedBySession (pause holds; cancel drains;
	// budget exhausts). The record — not the table write — is the proof.
	WitnessBoundaryStop WitnessKind = "boundary-stop"
	// WitnessSameTurnWake — a parked arm WAKES and completes its held turn (resume): a
	// final answer from exactly the resumed turn, no terminal stop.
	WitnessSameTurnWake WitnessKind = "same-turn-wake"
	// WitnessSamplingCap — the write reaches THIS turn's SampleParams: the effective
	// per-turn output cap the turn was sampled under reflects it (throttle).
	WitnessSamplingCap WitnessKind = "sampling-cap"
	// WitnessSchedulerRead — the scheduler's rank read (Table.Snapshot yield order)
	// reflects the write; the turn loop itself never reads priority (priority).
	WitnessSchedulerRead WitnessKind = "scheduler-read"
)

// OpSpec is one registered control op with its four fixed properties. It is pure data
// — a declarative row a consumer (a control route, an audit, a CLI help surface) can
// adopt as the op's contract. The spine DECLARES the contract; no production caller
// consults it yet (wiring each consumer is per-op follow-on). Its fidelity is pinned by
// this package's completeness test and by the loop-side #2766 witness table
// (internal/agent/loop_control_witness_test.go), which proves the behavior each row
// describes and cross-binds its op set + per-op refusal tokens to this spine.
type OpSpec struct {
	// Op is the closed op token (also the wire/CLI verb).
	Op ControlOp `json:"op"`
	// Capability is the send-right required to submit the op.
	Capability Capability `json:"capability"`
	// Boundary is when the op takes effect relative to the loop.
	Boundary Boundary `json:"boundary"`
	// Witness is the SHAPE of the loop-side proof the op was consumed.
	Witness WitnessKind `json:"witness"`
	// RefusalReasons is the closed refusal token(s) the op surfaces for an
	// illegal-for-state submission. Non-empty for every op.
	RefusalReasons []string `json:"refusal_reasons"`
	// Summary is the one-line human/audit description of the op's precise behavior.
	Summary string `json:"summary"`
}

// steerRefusals / redirectRefusals ground the injection ops' refusal tokens in their
// authoritative sources rather than restating string literals: steer refuses with the
// abi send-floor codes, redirect with its own closed reasons (redirect.go).
func steerRefusals() []string {
	return []string{abi.ReasonName(abi.ReasonDefaultDeny), abi.ReasonName(abi.ReasonTrustViolation)}
}

func redirectRefusals() []string {
	return []string{string(RedirectMalformed), string(RedirectNoRedirectableState)}
}

// vocabulary is the closed registry, in stable op order. Every drive-state op is
// grounded in session.ControlRefusalTokens (its terminal-session / stale-rev tokens);
// cancel additionally carries the stale-rev token for the --if-rev CAS race.
var vocabulary = []OpSpec{
	{
		Op:             OpSteer,
		Capability:     CapOperatorSend,
		Boundary:       BoundaryNextTurn,
		Witness:        WitnessSplice,
		RefusalReasons: steerRefusals(),
		Summary:        "freeform operator input spliced into the next turn's user message; a2achan mailbox drained",
	},
	{
		Op:             OpRedirect,
		Capability:     CapOperatorSend,
		Boundary:       BoundaryNextTurn,
		Witness:        WitnessDirective,
		RefusalReasons: redirectRefusals(),
		Summary:        "first-class objective change carried as a standing system directive; mailbox drained; live objective updated",
	},
	{
		Op:             OpPause,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryNextTurn,
		Witness:        WitnessBoundaryStop,
		RefusalReasons: []string{session.ReasonControlSessionTerminal},
		Summary:        "arm holds at the next boundary: 0 turns run, StoppedBySession=PAUSED (resumable)",
	},
	{
		Op:             OpResume,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryImmediate,
		Witness:        WitnessSameTurnWake,
		RefusalReasons: []string{session.ReasonControlSessionTerminal},
		Summary:        "parked arm wakes and completes its held turn in place",
	},
	{
		Op:             OpCancel,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryQuiesce,
		Witness:        WitnessBoundaryStop,
		RefusalReasons: []string{session.ReasonControlSessionTerminal, session.ReasonControlRevStale},
		Summary:        "enqueued Draining finalized to Stopped at the next boundary; arm stops with StoppedBySession=DRAINING (session.ReasonDrained)",
	},
	{
		Op:             OpThrottle,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryNextTurn,
		Witness:        WitnessSamplingCap,
		RefusalReasons: []string{session.ReasonControlSessionTerminal},
		Summary:        "per-turn output cap lowered into this turn's sampling",
	},
	{
		Op:             OpBudget,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryNextTurn,
		Witness:        WitnessBoundaryStop,
		RefusalReasons: []string{session.ReasonControlSessionTerminal},
		Summary:        "re-allotment applied; the WITNESSED case is exhaustion — a spent allotment stops the arm at the boundary with its closed exhaustion reason (a grant is consumed by the arm running on, outside the boundary-stop witness)",
	},
	{
		Op:             OpPriority,
		Capability:     CapOperatorControl,
		Boundary:       BoundaryImmediate,
		Witness:        WitnessSchedulerRead,
		RefusalReasons: []string{session.ReasonControlSessionTerminal},
		Summary:        "scheduler-consumed: the next pick's rank order reflects the write",
	},
}

// specByOp is the lookup index, built once from vocabulary.
var specByOp = func() map[ControlOp]OpSpec {
	m := make(map[ControlOp]OpSpec, len(vocabulary))
	for _, s := range vocabulary {
		m[s.Op] = s
	}
	return m
}()

// Vocabulary returns the closed control-op registry in stable order — the declarative
// contract for the operator control ops. The result is a deep copy (the nested
// RefusalReasons slices are cloned), so a caller cannot mutate the registry through it.
func Vocabulary() []OpSpec {
	out := make([]OpSpec, len(vocabulary))
	copy(out, vocabulary)
	for i := range out {
		out[i].RefusalReasons = slices.Clone(vocabulary[i].RefusalReasons)
	}
	return out
}

// Ops returns the closed op set in stable order.
func Ops() []ControlOp {
	out := make([]ControlOp, len(vocabulary))
	for i, s := range vocabulary {
		out[i] = s.Op
	}
	return out
}

// Spec returns the registered spec for op and whether it is a known control op. The
// returned spec is a deep copy (RefusalReasons cloned) so a caller cannot mutate the
// registry through the shared slice header.
func Spec(op ControlOp) (OpSpec, bool) {
	s, ok := specByOp[op]
	s.RefusalReasons = slices.Clone(s.RefusalReasons)
	return s, ok
}
