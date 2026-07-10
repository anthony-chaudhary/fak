// Package supervisor is the CAPSTONE of the session read/query/observe plane (epic
// #4176, child C7 / #4198): the reference implementation of the external-supervisor
// pattern that BRIDGES the read plane (this epic) to the control plane (#2753). The
// two shipped unbridged — the read seams (query C2, directory C4, transcriptfeed C5)
// let an external process OBSERVE a session as evidence, and the control ops (#2753)
// let an operator DRIVE one, but nothing composed "observe → decide → drive" into a
// closed loop. This package is that loop, written once as a pattern a host adopts.
//
// # The read → decide → control loop
//
// Supervise drives a fleet of SessionViews through three rungs, in order, for each
// session:
//
//  1. READ — compose the landed read plane, read-only and evidence-qualified. It runs
//     the C2 query grammar (query.Answer) over the session's projected turns to derive
//     tool-failure counts and a confirm-gate signal, addresses the session through its
//     C4 directory row, and drains its C5 transcript feed (if attached) — all without a
//     side effect (a read can never advance the loop; every answer carries
//     sessionread.EvidenceObserved). READING IS NOT CONTROLLING: the READ rung never
//     touches the Controller.
//
//  2. CHECK — ask an injected Checker for an evidence-backed verdict (stuck / progressing),
//     NEVER the session's own self-report. The Checker is an interface; the real
//     implementation binds it to the dos_* truth verbs (dos_verify / dos_commit_audit /
//     dos_recall / dos_status), whose verdicts have exactly this stuck/verified/progress
//     shape. The fixture Checker in the test supplies deterministic verdicts.
//
//  3. CONTROL — for a session that is a genuine intervention CANDIDATE, dispatch EXACTLY
//     ONE typed control verb through the injected Controller and record the witness it
//     returns. The Controller is an interface with typed verbs modeled on the #2753
//     control ops (Hold≈pause, Steer≈steer, Redirect≈redirect, Approve≈resume/confirm-
//     clear); each is capability-checked and returns a witnessed record. The real
//     implementation binds it to the #2753 control route; the fixture Controller in the
//     test records every verb call.
//
// # The regime gate (#2533) — the load-bearing safety invariant
//
// A supervisor that intervenes on a HEALTHY, high-scoring session HARMS it (#2533:
// interventions are corrective, and correcting a session that is doing well degrades it).
// So the gate is a HARD AND: a session is an intervention candidate ONLY when it is BOTH
//
//   - demonstrably STUCK (from the CHECK verdict grounded in READ evidence), AND
//   - LOW-score (its injected regime score is at or below the gate threshold).
//
// A high-score session gets ZERO control actions even when a naive heuristic — or the
// Checker itself — flags it stuck; a low-score-but-progressing session likewise gets
// zero. Neither half alone fires. This is structural, not advisory: a non-candidate is
// `continue`d before any Controller method is reachable, so the Controller is never even
// called for a protected session.
//
// # The interface seams (why CHECK and CONTROL are injected, not imported)
//
// The real control plane (#2753) lives in internal/session / internal/gateway, and the
// real regime score (#2533) in the scoring packages — all volatile siblings this
// hermetic read leaf MUST NOT import (its only edges are stdlib + the internal/sessionread*
// read packages, so it builds and tests in isolation in the shared multi-writer tree).
// CHECK and CONTROL are therefore modeled as INTERFACES with fixture implementations in
// the test, mirroring how the sibling children (C1–C5) landed a pure kernel + witness and
// left the live wiring to the host. Binding the Checker to dos_* and the Controller to the
// #2753 control route — and sourcing the regime score from #2533 — is deliberate follow-on,
// named here and not done here.
package supervisor

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/directory"
	"github.com/anthony-chaudhary/fak/internal/sessionread/query"
	"github.com/anthony-chaudhary/fak/internal/sessionread/transcriptfeed"
)

// defaultGranted is the read disclosure the reference supervisor reads at: REDACTED, the
// minimum that lets the confirm-gate probe (decisions-about, a redacted-text query) run
// while still refusing raw-span disclosure. The supervisor never needs full span bytes to
// decide, so it never requests them.
const defaultGranted = sessionread.DisclosureRedacted

// ConfirmGateMarker is the sentinel a producer stamps into an assistant turn's text when
// the session has HALTED awaiting an operator confirm/approval. It is the confirm-gate
// signal modeled in the turns: the supervisor detects a stuck confirm gate by running the
// C2 decisions-about grammar for this marker and checking the match is the session's TAIL
// turn (an unresolved gate), never by reading a self-reported "I am blocked" flag. A real
// producer would set it from the loop's actual awaiting-approval state.
const ConfirmGateMarker = "await-confirm"

// RegimeScore is a session's injected health/regime score (#2533). Higher is healthier;
// the gate protects a high score. It is INJECTED (the fixture, and in production the #2533
// scorer, supplies it) — the supervisor never computes it.
type RegimeScore float64

// RegimeGate is the #2533 admission gate. A session is LOW-score (intervention-eligible)
// iff its score is at or below Threshold; strictly above Threshold is HIGH-score and
// protected. The gate is only HALF of the candidate rule — the other half is a stuck
// CHECK verdict — and both must hold.
type RegimeGate struct {
	// Threshold is the score at or below which a session is low-score. A session scoring
	// strictly above it is high-score and receives zero control actions.
	Threshold RegimeScore
}

// LowScore reports whether s is at or below the gate threshold (intervention-eligible).
func (g RegimeGate) LowScore(s RegimeScore) bool { return s <= g.Threshold }

// HighScore reports whether s is strictly above the gate threshold (protected).
func (g RegimeGate) HighScore(s RegimeScore) bool { return s > g.Threshold }

// SessionView is the per-session input the supervisor READs. It is pure data the host
// projects from the read plane before the loop runs: the session's address (Trace, or the
// canonical TraceID/UUID off a C4 directory Row), the read-only C2 projection of its turns,
// its injected regime score, and an optional C5 transcript Feed to observe. The supervisor
// never mutates it.
type SessionView struct {
	// Trace is the session's trace/principal id — the address a control verb targets and
	// the principal a scoped read is bound to. When Row.TraceID is set it takes precedence.
	Trace string
	// Principal is the isolation principal a feed drain is scoped to; defaults to the trace.
	Principal string
	// Row is the optional C4 directory row addressing this session (Source / Lifecycle /
	// RunState / UUID). When present it supplies the canonical TraceID and UUID.
	Row directory.DirectoryRow
	// Turns is the read-only C2 projection of the session's transcript the supervisor reads
	// with query.Answer. It is never streamed whole — the query engine bounds every answer.
	Turns []query.Turn
	// Score is the injected #2533 regime/health score. The gate protects a high score.
	Score RegimeScore
	// Feed is the optional C5 transcript-event feed the supervisor drains (read-only) to
	// observe recent events. nil skips the observe rung. Draining it never advances the loop.
	Feed *transcriptfeed.Feed
	// Since is the cursor the feed drain resumes from (0 = everything retained).
	Since uint64
}

// trace resolves the session's canonical address: the directory row's TraceID when set,
// else the bare Trace.
func (v SessionView) trace() string {
	if t := strings.TrimSpace(v.Row.TraceID); t != "" {
		return t
	}
	return strings.TrimSpace(v.Trace)
}

// uuid resolves the session's transcript UUID from its directory row, if the identity join
// knows it.
func (v SessionView) uuid() string { return strings.TrimSpace(v.Row.UUID) }

// principal resolves the read-scope principal: an explicit Principal, else the trace.
func (v SessionView) principal() string {
	if p := strings.TrimSpace(v.Principal); p != "" {
		return p
	}
	return v.trace()
}

// The closed CHECK-verdict reason vocabulary. These are the supervisor's own reason tokens
// for the reference Checker contract; a dos_*-bound Checker supplies its verbs' own reasons
// (dos_status STALLED, dos_verify NOT_SHIPPED, ...) into the same Verdict.Reason field.
const (
	// CheckLiveProgressing — the session shows witnessed forward motion; not a candidate.
	CheckLiveProgressing = "LIVE_AND_PROGRESSING"
	// CheckStalledConfirmGate — the session has halted at an unresolved confirm/approval gate.
	CheckStalledConfirmGate = "STALLED_ON_CONFIRM_GATE"
	// CheckStalledNoProgress — the session is stuck with no forward progress (a stall).
	CheckStalledNoProgress = "STALLED_NO_PROGRESS"
)

// Verdict is a CHECK answer: the evidence-backed shape the dos_* truth verbs return
// (dos_status liveness/progress, dos_verify shipped, dos_commit_audit witness). It is
// EVIDENCE, never the session's self-report. A stuck verdict is a NECESSARY (not
// sufficient) condition for intervention — the regime gate is the other half.
type Verdict struct {
	// Stuck is the evidence-backed judgment that the session is not making forward progress
	// — the analog of dos_status liveness=STALLED. A candidate must be Stuck.
	Stuck bool
	// Progressing mirrors dos_status's ledger-VERIFIED progress rung: witnessed forward motion.
	Progressing bool
	// Reason is a closed token explaining the verdict (a Check* constant, or a dos_* reason).
	Reason string
	// Evidence qualifies the verdict OBSERVED vs WITNESSED, reusing the read plane's grammar:
	// a live check is OBSERVED, a ledger/artifact-backed one WITNESSED.
	Evidence sessionread.Evidence
	// Detail is a bounded, byte-free explanation for humans/audit.
	Detail string
}

// Checker is the CHECK seam: it turns a read-only SessionView into an evidence-backed
// Verdict. It is an INTERFACE so this hermetic leaf never imports the truth plane — the real
// implementation binds Check to dos_verify / dos_commit_audit / dos_recall / dos_status
// (whose verdicts have exactly this stuck/verified/progress shape); the fixture Checker in
// the test supplies deterministic verdicts. A Checker MUST decide from evidence (the view's
// READ projection), never from a self-reported status the session wrote about itself.
type Checker interface {
	// Check returns the evidence-backed verdict for one session.
	Check(SessionView) Verdict
}

// Capability is the send-right a control verb requires, modeled on the #2753 send-rights
// (sessionctl.CapOperatorSend / CapOperatorControl). A verb is dispatched with its required
// capability stamped on the request, and the Controller enforces it — the capability-checked
// property. It is NOT a bare boolean the caller can bypass.
type Capability string

const (
	// CapOperatorSend gates the INPUT-injection verbs (Steer, Redirect): an a2achan send.
	CapOperatorSend Capability = "operator-send"
	// CapOperatorControl gates the DRIVE-STATE verbs (Hold, Approve): a control-route write.
	CapOperatorControl Capability = "operator-control"
)

// Verb is the closed set of typed control verbs the reference supervisor may fire, each
// modeled on a #2753 control op. It is a string enum so the wire verb and the token agree.
type Verb string

const (
	// VerbHold holds the running arm at its next boundary — models #2753 OpPause. The
	// conservative default corrective for a stalled session.
	VerbHold Verb = "hold"
	// VerbSteer splices operator input into the next turn — models #2753 OpSteer.
	VerbSteer Verb = "steer"
	// VerbRedirect lands a first-class objective change — models #2753 OpRedirect.
	VerbRedirect Verb = "redirect"
	// VerbApprove clears an awaiting confirm/approval gate — models #2753 OpResume applied
	// to a session halted at a confirm boundary. The corrective for a confirm-gate stall.
	VerbApprove Verb = "approve"
)

// RequiredCapability is the send-right a verb needs: the input-injection verbs need the
// operator-send right, the drive-state verbs the operator-control right. The supervisor
// stamps this onto every ControlRequest so the Controller can enforce it.
func (v Verb) RequiredCapability() Capability {
	switch v {
	case VerbSteer, VerbRedirect:
		return CapOperatorSend
	default: // VerbHold, VerbApprove
		return CapOperatorControl
	}
}

// ControlRequest is the typed input to a Controller verb: the target session, the capability
// the verb requires (for the Controller's capability check), an optional payload (steer input
// / redirect objective; empty for Hold/Approve), the CHECK reason that justified it, and the
// READ evidence qualifier. It carries no span bytes.
type ControlRequest struct {
	// Target is the session trace the verb addresses.
	Target string
	// Capability is the send-right the verb requires — the Controller enforces it.
	Capability Capability
	// Payload is the verb's argument (steer input / redirect objective); "" for Hold/Approve.
	Payload string
	// Reason is the closed CHECK reason justifying the intervention.
	Reason string
	// Evidence is the read-plane evidence qualifier for the justification (OBSERVED).
	Evidence sessionread.Evidence
}

// Witness is the record a Controller verb returns proving it was CONSUMED — the outbound
// "witness-of-applied" the #2753 control plane defines (sessionctl.WitnessKind). It is the
// non-forgeable receipt an Intervention carries; it is NOT a bare boolean. Applied is the
// load-bearing bit: a verb the control plane confirmed took has Applied=true.
type Witness struct {
	// Verb is the typed verb that fired.
	Verb Verb
	// Target is the session it addressed.
	Target string
	// Capability is the send-right it was checked against.
	Capability Capability
	// Applied reports whether the control plane confirmed the verb took (the witness-of-applied).
	Applied bool
	// Kind is the SHAPE of the loop-side proof (models sessionctl.WitnessKind), e.g.
	// "boundary-stop" for Hold or "same-turn-wake" for Approve.
	Kind string
	// Detail is a bounded, byte-free explanation for humans/audit.
	Detail string
}

// Controller is the CONTROL seam: typed verbs modeled on the #2753 control ops, each
// capability-checked and returning a Witness. It is an INTERFACE so this leaf never imports
// the control plane — the real implementation binds these verbs to the #2753 control route;
// the fixture Controller in the test records every call so a test can assert exactly which
// interventions fired. A verb MUST enforce req.Capability (refuse when it is not held) and
// return a witnessed record, never a bare success.
type Controller interface {
	// Hold holds the running arm at its next boundary (models #2753 OpPause).
	Hold(ControlRequest) (Witness, error)
	// Steer splices operator input into the next turn (models #2753 OpSteer).
	Steer(ControlRequest) (Witness, error)
	// Redirect lands a first-class objective change (models #2753 OpRedirect).
	Redirect(ControlRequest) (Witness, error)
	// Approve clears an awaiting confirm/approval gate (models #2753 OpResume at a gate).
	Approve(ControlRequest) (Witness, error)
}

// ReadEvidence is the read-only evidence the READ rung gathered for one session — the
// justification an Intervention carries alongside the CHECK verdict. Every field is derived
// from the composed read plane (C2 query grammar, C4 directory, C5 feed) and is OBSERVED.
type ReadEvidence struct {
	// Trace is the session's resolved address.
	Trace string
	// UUID is the session's transcript UUID from its directory row, if known.
	UUID string
	// ConfirmGate reports the session's TAIL turn is an unresolved awaiting-confirm decision
	// (derived from the C2 decisions-about grammar) — a stuck confirm gate.
	ConfirmGate bool
	// ConfirmGateIndex is the awaiting-confirm tail turn's index when ConfirmGate is set.
	ConfirmGateIndex int
	// ToolFailures is the count of tool-terminal turns that failed (C2 tool-failures grammar).
	ToolFailures int
	// ObservedEvents is the number of transcript events drained from the C5 feed (0 if none).
	ObservedEvents int
	// Evidence is always sessionread.EvidenceObserved — a live reading, not an attested artifact.
	Evidence sessionread.Evidence
}

// Intervention is one recorded control action the supervisor took: the typed verb, its
// target, the Controller's witness, and the READ + CHECK evidence that justified it (plus the
// regime score at decision time). It is the audit record proving the intervention went
// through a #2753 control verb — NOT a read side effect. Err is set (non-empty) when the
// Controller verb refused (e.g. a capability it did not hold); Witness.Applied is then false.
type Intervention struct {
	// Target is the session the verb addressed.
	Target string
	// Verb is the typed control verb that fired.
	Verb Verb
	// Capability is the send-right the verb was dispatched with (and the Controller checked).
	Capability Capability
	// Witness is the Controller's witnessed receipt.
	Witness Witness
	// Verdict is the CHECK evidence that justified the intervention.
	Verdict Verdict
	// Read is the READ evidence that justified the intervention.
	Read ReadEvidence
	// Score is the regime score at decision time (at or below the gate threshold).
	Score RegimeScore
	// Evidence is the read-plane evidence qualifier for the whole decision (OBSERVED).
	Evidence sessionread.Evidence
	// Err is the Controller refusal text when the verb did not apply; "" on a clean apply.
	Err string
}

// Supervise drives fleet through the read → decide → control loop and returns the
// interventions it took, in fleet order. For EACH session it:
//
//   - READs the session read-only (query.Answer over its turns, its C4 directory row, its C5
//     feed) — never touching the Controller;
//   - CHECKs it via chk for an evidence-backed verdict;
//   - DECIDEs: a session is a candidate ONLY when it is BOTH stuck (the verdict) AND low-score
//     (gate.LowScore) — the #2533 AND-invariant. A non-candidate is skipped before any
//     Controller method is reachable, so a healthy/high-score session receives ZERO control
//     actions;
//   - CONTROLs a candidate with EXACTLY ONE typed verb through ctl and records the witness.
//
// The returned slice contains one Intervention per fired verb. A pure-read caller (no
// candidates) gets an empty slice and ctl is never called.
func Supervise(fleet []SessionView, chk Checker, ctl Controller, gate RegimeGate) []Intervention {
	var out []Intervention
	for _, view := range fleet {
		// READ — read-only compose of the landed read plane. This rung never calls ctl.
		rd := readSession(view, defaultGranted)

		// CHECK — evidence-backed verdict, never the session's self-report.
		verdict := chk.Check(view)

		// DECIDE — the regime gate (#2533): intervene ONLY on stuck AND low-score. Either
		// half missing (progressing, or high-score) protects the session absolutely.
		if !(verdict.Stuck && gate.LowScore(view.Score)) {
			continue
		}

		// CONTROL — exactly one typed verb, capability-stamped, witnessed.
		verb := chooseVerb(rd)
		req := ControlRequest{
			Target:     view.trace(),
			Capability: verb.RequiredCapability(),
			Payload:    payloadFor(verb),
			Reason:     verdict.Reason,
			Evidence:   sessionread.EvidenceObserved,
		}
		w, err := dispatch(ctl, verb, req)
		out = append(out, Intervention{
			Target:     req.Target,
			Verb:       verb,
			Capability: req.Capability,
			Witness:    w,
			Verdict:    verdict,
			Read:       rd,
			Score:      view.Score,
			Evidence:   sessionread.EvidenceObserved,
			Err:        errText(err),
		})
	}
	return out
}

// readSession is the READ rung: a read-only compose of the landed read plane for one
// session. It runs the C2 tool-failures + confirm-gate queries over the turns, records the
// session's C4 address, and drains the C5 feed if attached. It calls no Controller and
// mutates nothing (draining a Feed is a read); every field it fills is OBSERVED.
func readSession(v SessionView, granted sessionread.Disclosure) ReadEvidence {
	rd := ReadEvidence{Trace: v.trace(), UUID: v.uuid(), Evidence: sessionread.EvidenceObserved}

	// C2: tool-failures — a metadata query, read-only.
	if res, err := query.Answer(query.Query{Kind: query.KindToolFailures}, v.Turns, granted); err == nil {
		rd.ToolFailures = len(res.Items)
	}

	// C2: confirm-gate — decisions-about the confirm marker, then a TAIL check.
	if idx, ok := confirmGateTail(v.Turns, granted); ok {
		rd.ConfirmGate = true
		rd.ConfirmGateIndex = idx
	}

	// C5: drain the observe feed read-only, if present. Draining never advances the loop.
	if v.Feed != nil {
		evs, _ := v.Feed.Drain(v.principal(), v.Since)
		rd.ObservedEvents = len(evs)
	}
	return rd
}

// confirmGateTail reports whether the session is STUCK on a confirm gate: its TAIL turn is
// an unresolved awaiting-confirm assistant decision. It runs the C2 decisions-about grammar
// for ConfirmGateMarker (read-only) and returns the tail index when the matching decision is
// the last turn (nothing followed it to resolve the gate). A confirm decision with later
// turns after it is NOT stuck. Pure and read-only.
func confirmGateTail(turns []query.Turn, granted sessionread.Disclosure) (int, bool) {
	if len(turns) == 0 {
		return 0, false
	}
	res, err := query.Answer(query.Query{Kind: query.KindDecisionsAbout, Term: ConfirmGateMarker}, turns, granted)
	if err != nil || len(res.Items) == 0 {
		return 0, false
	}
	lastIdx := turns[len(turns)-1].Index
	for _, it := range res.Items {
		if it.Index == lastIdx {
			return lastIdx, true
		}
	}
	return 0, false
}

// chooseVerb picks the single typed verb for a candidate: Approve to clear a stuck confirm
// gate, else Hold to halt a stalled arm at its boundary (the conservative default). Steer
// and Redirect are part of the typed vocabulary for a host with a richer policy; the
// reference supervisor uses the two least-invasive drive-state verbs.
func chooseVerb(rd ReadEvidence) Verb {
	if rd.ConfirmGate {
		return VerbApprove
	}
	return VerbHold
}

// payloadFor is the verb's argument: the input-injection verbs carry a bounded directive;
// the drive-state verbs (Hold, Approve) carry none.
func payloadFor(v Verb) string {
	switch v {
	case VerbSteer:
		return "supervisor: unblock and continue toward the pending objective"
	case VerbRedirect:
		return "supervisor: re-focus on the pending objective"
	default:
		return ""
	}
}

// dispatch routes a verb to its typed Controller method. The switch is the ONLY place a verb
// becomes a Controller call — a read path can never reach it.
func dispatch(ctl Controller, v Verb, req ControlRequest) (Witness, error) {
	switch v {
	case VerbHold:
		return ctl.Hold(req)
	case VerbSteer:
		return ctl.Steer(req)
	case VerbRedirect:
		return ctl.Redirect(req)
	case VerbApprove:
		return ctl.Approve(req)
	default:
		return Witness{}, fmt.Errorf("supervisor: unknown control verb %q", v)
	}
}

// errText renders an error as its message, or "" for nil.
func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// FleetHit is one session the fleet query matched: its address (Trace/UUID from the C4
// directory row), the matching turn's index, and the OBSERVED evidence qualifier.
type FleetHit struct {
	// Trace is the session's resolved address.
	Trace string
	// UUID is the session's transcript UUID, if the directory row knows it.
	UUID string
	// Index is the awaiting-confirm tail turn's index.
	Index int
	// Evidence is always sessionread.EvidenceObserved.
	Evidence sessionread.Evidence
}

// FleetQuery answers cross-session questions by folding the C2 query grammar across a fleet
// and addressing each match through its C4 directory row. It is READ-ONLY: it never calls a
// Controller and mutates nothing — reading the fleet is not driving it.
type FleetQuery struct {
	// Granted is the disclosure the fold reads at; empty defaults to REDACTED (enough for the
	// decisions-about confirm probe, never raw span bytes).
	Granted sessionread.Disclosure
}

// ConfirmGateStuck returns the subset of the fleet whose sessions are STUCK on a confirm/
// approval gate — an unresolved awaiting-confirm TAIL turn — folding the C2 decisions-about
// grammar across every session and addressing each hit through its C4 directory row. The
// result is in fleet order and carries exactly the confirm-gate-stuck sessions. Read-only.
func (fq FleetQuery) ConfirmGateStuck(fleet []SessionView) []FleetHit {
	granted := fq.Granted
	if granted == "" {
		granted = defaultGranted
	}
	var out []FleetHit
	for _, v := range fleet {
		if idx, ok := confirmGateTail(v.Turns, granted); ok {
			out = append(out, FleetHit{
				Trace:    v.trace(),
				UUID:     v.uuid(),
				Index:    idx,
				Evidence: sessionread.EvidenceObserved,
			})
		}
	}
	return out
}
