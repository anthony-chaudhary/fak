// Package sessionsteer is the steering + admission half of the zero-knob automatic-context
// doctrine (epic #2198, spine #3512). It is a PURE decision core: given a content-free
// snapshot of a long session's context-value advice and its pending work, it emits one
// typed directive the guard hooks turn into
//
//   - a RULE injected at SessionStart (soft persistence + managed-context posture), and
//   - a PERSIST decision the Stop hook consults (hard block; ships in shadow until a soak).
//
// It sits BELOW the relay rotation driver (#1860 tracks G/H) that R7 (#2205) is blocked on,
// and deliberately does not rotate anything — it only DIRECTS the harness.
//
// Tier: foundation (1) — see internal/architest. This package imports only the standard
// library; the guard shell (cmd/fak) supplies the session snapshot and consumes the directive.
//
// No I/O, no clock, no randomness: every field of the directive is a deterministic function
// of the input, so the whole core is exercised by a golden table. The advice vocabulary is a
// string mirror of internal/gateway.StepClass (any|bounded|checkpoint|rebuild|unknown) so the
// core carries no dependency on the gateway package (and cannot form an import cycle with it);
// callers map gateway.StepClass -> Advice by its string value.
package sessionsteer

import (
	"strconv"
	"strings"
)

// Advice is the string mirror of gateway.StepClass — the closed step-advice vocabulary the
// context-value reporter emits. Kept as a local type so this core stays dependency-free.
type Advice string

const (
	AdviceAny        Advice = "any"        // wide headroom: a large multi-file step fits
	AdviceBounded    Advice = "bounded"    // moderate pressure: single-concern step
	AdviceCheckpoint Advice = "checkpoint" // window nearly spent: land durable state now
	AdviceRebuild    Advice = "rebuild"    // a context event just fired: re-anchor first
	AdviceUnknown    Advice = "unknown"    // no evidence to decide on
)

// NormalizeAdvice maps an arbitrary step-advice string (e.g. a gateway.StepClass value) onto
// the closed Advice vocabulary, defaulting unrecognized input to AdviceUnknown (fail-closed:
// an unknown advice never masquerades as headroom).
func NormalizeAdvice(s string) Advice {
	switch Advice(strings.ToLower(strings.TrimSpace(s))) {
	case AdviceAny:
		return AdviceAny
	case AdviceBounded:
		return AdviceBounded
	case AdviceCheckpoint:
		return AdviceCheckpoint
	case AdviceRebuild:
		return AdviceRebuild
	default:
		return AdviceUnknown
	}
}

// StepAdviceAffordance renders the action carried by a stable step_advice token.
// Unknown and absent inputs use the any framing so a presentation-only field stays
// useful without changing the machine token or inventing a constraint.
func StepAdviceAffordance(raw string, headroomTokens int64) string {
	a := NormalizeAdvice(raw)
	switch a {
	case AdviceBounded:
		return "Wrap the current sub-task, land its durable state, then continue with one bounded step."
	case AdviceCheckpoint:
		return "Checkpoint durable state now, then proceed from that checkpoint."
	case AdviceRebuild:
		return "Rebuild context from the checkpoint, then take the next step."
	default:
		if headroomTokens > 0 {
			return "Keep going with full headroom (" + humanHeadroom(headroomTokens) + " tokens available)."
		}
		return "Keep going with full headroom."
	}
}

func humanHeadroom(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	return strconv.FormatFloat(float64(n)/1000, 'f', 1, 64) + "k"
}

// AdmitClass is the admission decision — whether a session is admitted onto the managed
// long-horizon posture (persistence rule + managed-context steering) or stays legacy.
type AdmitClass string

const (
	AdmitManaged AdmitClass = "MANAGED"
	AdmitLegacy  AdmitClass = "LEGACY"
)

// AdmitReason is the closed, never-silent reason vocabulary for the admission decision. Every
// admission — including LEGACY — carries one, so a session is never silently left unmanaged.
type AdmitReason string

const (
	AdmitReasonHeadlessGoal   AdmitReason = "MANAGED_HEADLESS_GOAL"   // unattended worker with a standing goal
	AdmitReasonHeadless       AdmitReason = "MANAGED_HEADLESS"        // unattended worker (fleet) — persistence matters most
	AdmitReasonAttendedGoal   AdmitReason = "MANAGED_ATTENDED_GOAL"   // attended, but the operator set a standing goal
	AdmitReasonNoDurableStore AdmitReason = "LEGACY_NO_DURABLE_STORE" // cannot checkpoint/relay — never silent
	AdmitReasonAttendedNoGoal AdmitReason = "LEGACY_ATTENDED_NO_GOAL" // a human is driving; no heavy posture imposed
)

// PersistDecision is the Stop-hook decision: block a stop to keep the agent moving, or allow it.
type PersistDecision string

const (
	PersistBlockStop PersistDecision = "BLOCK_STOP"
	PersistAllowStop PersistDecision = "ALLOW_STOP"
)

// PersistReason is the closed reason vocabulary for the persist decision.
type PersistReason string

const (
	PersistReasonGoalUnmet   PersistReason = "PERSIST_GOAL_UNMET"       // a standing goal's done-condition does not hold
	PersistReasonHandoff     PersistReason = "PERSIST_HANDOFF_REQUIRED" // the task-handoff artifact is not yet written
	PersistReasonWorkRemains PersistReason = "PERSIST_WORK_REMAINS"     // open tasks / checkable work remain
	PersistReasonGoalMet     PersistReason = "STOP_CLEAN_GOAL_MET"      // the goal's done-condition holds
	PersistReasonNoWork      PersistReason = "STOP_CLEAN_NO_WORK"       // nothing checkable remains
	// PersistReasonFloorDenied is emitted when the work state would BLOCK_STOP but the capability
	// floor denies the agent's durable-persist path: the block would demand an impossible action
	// (e.g. a git-commit checkpoint the write-floor refuses) and wedge the session forever.
	// Persistence must be operator-mediated instead; the session is allowed to stop, never spun.
	PersistReasonFloorDenied PersistReason = "STOP_PERSIST_FLOOR_DENIED"
)

// SteerInput is the content-free session snapshot the core decides on. Every field is a plain
// fact about the session — none carries session content, so the directive is safe to log.
type SteerInput struct {
	Advice          Advice // step-advice class from the context-value reporter
	Phase           string // lifecycle phase (fresh|building|cruising|crowding|post_event); carried, not branched
	Headless        bool   // unattended/fleet child (a `-p` worker), vs an attended TUI
	GoalActive      bool   // a standing goal / done-condition is set for the session
	GoalMet         bool   // that goal's done-condition currently holds
	HandoffRequired bool   // the task-handoff gate is armed and its artifact is not yet written
	PendingWork     bool   // open tasks / uncommitted checkable work remain
	DurableStore    bool   // a durable store exists to checkpoint / relay into
	// PersistFloorDenied reports that the capability floor denies the agent's own durable-persist
	// path (e.g. a git-commit checkpoint refused as a write-shaped .git/ self-modify) AND no
	// within-floor sink is available. Zero value (false) preserves the plain work-state ladder; when
	// true, a would-be BLOCK_STOP is downgraded so the persist-hook cannot deadlock against the
	// write-floor. The Stop-hook wiring sets this from the live floor when it comes out of shadow.
	PersistFloorDenied bool
}

// SteerDirective is the typed decision the guard hooks consume.
type SteerDirective struct {
	Admit            AdmitClass      // MANAGED | LEGACY
	AdmitReason      AdmitReason     // never-silent admission reason
	Persist          PersistDecision // BLOCK_STOP | ALLOW_STOP (Stop hook honors only when Managed)
	PersistReason    PersistReason   // reason for the persist decision
	ContextDirective string          // step-advice steering text to inject (may be empty)
}

// Managed reports whether the directive admitted the session onto the managed posture.
func (d SteerDirective) Managed() bool { return d.Admit == AdmitManaged }

// Steer is the whole decision. It is a pure function: same input, same directive.
func Steer(in SteerInput) SteerDirective {
	admit, admitReason := admission(in)
	persist, persistReason := persistDecision(in)
	return SteerDirective{
		Admit:            admit,
		AdmitReason:      admitReason,
		Persist:          persist,
		PersistReason:    persistReason,
		ContextDirective: ContextDirective(in.Advice),
	}
}

// admission decides MANAGED vs LEGACY and the never-silent reason. A session that cannot be
// managed (no durable store to checkpoint into) is admitted LEGACY with a structured reason —
// never silently. A headless worker (where persistence matters most) or any goal-bearing
// session is admitted MANAGED; an attended session with no standing goal stays LEGACY (a human
// is driving — no heavy posture is imposed).
func admission(in SteerInput) (AdmitClass, AdmitReason) {
	if !in.DurableStore {
		return AdmitLegacy, AdmitReasonNoDurableStore
	}
	switch {
	case in.Headless && in.GoalActive:
		return AdmitManaged, AdmitReasonHeadlessGoal
	case in.Headless:
		return AdmitManaged, AdmitReasonHeadless
	case in.GoalActive:
		return AdmitManaged, AdmitReasonAttendedGoal
	default:
		return AdmitLegacy, AdmitReasonAttendedNoGoal
	}
}

// persistDecision reports whether checkable work remains, as a priority ladder. It is computed
// independent of admission — the core always reports the truth about the work state; the Stop
// hook wiring decides to ENFORCE a BLOCK_STOP only for a MANAGED session (and, in this spine,
// only in shadow until a fleet soak clears the flip).
func persistDecision(in SteerInput) (PersistDecision, PersistReason) {
	dec, reason := persistLadder(in)
	// Floor reconciliation. A BLOCK_STOP tells the Stop hook to keep the session from yielding until
	// it persists (checkpoint/commit). If the capability floor denies the agent's own durable-persist
	// path, that block demands an impossible action and wedges the session forever — the persist-hook
	// vs write-floor deadlock. (Found by dogfooding: this exact deadlock blocked committing the spine
	// itself — the Stop hook demanded a commit the write-floor refused the agent.) When persistence is
	// floor-denied, downgrade to ALLOW_STOP with a distinct reason so persistence is operator-mediated
	// rather than spun. Two levels that must compose are reconciled here, not left to collide.
	if dec == PersistBlockStop && in.PersistFloorDenied {
		return PersistAllowStop, PersistReasonFloorDenied
	}
	return dec, reason
}

// persistLadder is the priority ladder over the raw work state, independent of the capability
// floor. persistDecision layers the floor reconciliation on top.
func persistLadder(in SteerInput) (PersistDecision, PersistReason) {
	switch {
	case in.GoalActive && !in.GoalMet:
		return PersistBlockStop, PersistReasonGoalUnmet
	case in.HandoffRequired:
		return PersistBlockStop, PersistReasonHandoff
	case in.PendingWork:
		return PersistBlockStop, PersistReasonWorkRemains
	case in.GoalActive && in.GoalMet:
		return PersistAllowStop, PersistReasonGoalMet
	default:
		return PersistAllowStop, PersistReasonNoWork
	}
}

// ContextDirective maps a step-advice class onto the steering text injected to the agent. The
// thresholds that PRODUCE the advice live in the context-value reporter; this only renders the
// action. any/unknown produce no directive (no evidence to steer on).
func ContextDirective(a Advice) string {
	switch a {
	case AdviceCheckpoint:
		return "Context checkpoint: the active window is nearly spent. Land in-flight state now — commit what compiles, write the plan/ledger/handoff down — before the next context event rewrites the window."
	case AdviceRebuild:
		return "Context rebuild: a context event just fired. Re-anchor from durable state (plan, task ledger, index, git log) before starting any wide step; treat the just-rewritten window as incomplete."
	case AdviceBounded:
		return "Context bounded: the window is filling. Take a single-concern step and keep new residency (large reads, long tool output) deliberate."
	default:
		return ""
	}
}

// SessionStartRule renders the persistence + managed-context RULE injected as SessionStart
// additionalContext for a MANAGED session. For a LEGACY admission it returns "" — the base
// affordance hint still ships, but no heavy long-horizon posture is imposed on a human-driven
// session. This is the soft, always-on half of the persistence spine (the hard Stop-hook block
// is staged separately).
func SessionStartRule(d SteerDirective) string {
	if !d.Managed() {
		return ""
	}
	return "Keep working while checkable work remains; managed context is ON. Finish each unit, " +
		"commit its durable state, and pick up the next. End the turn when the task's done-condition " +
		"holds or a genuine blocker remains. The gateway preserves the cached prefix while shedding old " +
		"turns from the active window, extending the session beyond a raw context window. When directed " +
		"to CHECKPOINT, land durable state in a commit, plan, ledger, or handoff. When directed to REBUILD, " +
		"re-anchor from that durable state after the context event. Commits, plans, ledgers, handoffs, and " +
		"memory survive a window rewrite; treat the active window as working state. Close each operator-facing " +
		"turn in a shape the operator can scan: lead with the verdict, carry the body as scannable " +
		"bullets, and make the last line a bullet naming the next checkable step. Session-state tools: " +
		"mcp__fak__fak_context_value (window left + step advice), mcp__fak__fak_context_spans / " +
		"mcp__fak__fak_context_restore (recover a compacted originating task)."
}

// IndependentToolHint is the shadow-first launch-time nudge for independent tool work.
// It is advisory only: callers opt in explicitly and no Stop/admission decision depends on it.
func IndependentToolHint(shadow bool) string {
	mode := "advisory"
	if shadow {
		mode = "shadow-advisory"
	}
	return "TOOL_WIDTH_HINT (" + mode + "): when two or more tool calls are independent, prefer issuing them in one assistant turn; keep dependent calls sequential. This is a latency/width optimization, never permission to skip verification or batch conflicting writes."
}
