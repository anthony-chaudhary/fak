package dispatchtick

import "github.com/anthony-chaudhary/fak/internal/modelroute"

// ---------------------------------------------------------------------------
// THE META-ORCHESTRATOR CAPABILITY ENVELOPE — bound a cheap model's AUTHORITY
// mechanically, so cost savings from a T2 orchestrator can never buy a
// destructive action that only a prompt was asked to withhold (#3044, C7).
// ---------------------------------------------------------------------------
//
// This is the ACTION-GATE node of the model-tier working path:
//
//	model-tier route (C5)  ->  capability envelope (THIS)  ->  action gate
//
// C5 (RouteAccountForTier) decides WHICH model runs a unit of work. C7 answers a
// different question: given a meta-orchestrator ALREADY running at some tier —
// reading fleet status, ranking issues, suggesting next actions — what may it do
// with its own authority, and what must it escalate?
//
// THE ASYMMETRY (load-bearing). A cheap orchestrator is safe ONLY if its limit is
// mechanical, not a system-prompt instruction. The confusion risk #3044 names is
// exactly this: "do not rely on system prompts to prevent a cheap model from
// asking for destructive tools." So the envelope is a pure fold with a closed
// vocabulary, judged the same for every caller:
//
//   - READ-ONLY (read status, rank, summarize, suggest) — any tier, always. A
//     suggestion is NOT execution; a T2 output is useful but is never itself an
//     execution witness for the action it proposes.
//   - GUARDED MUTATION (propose a scoped, reversible change) — T1 and above.
//   - HIGH-RISK (mutate git, kill a process, launch a worker, edit a label, claim
//     done) — T0 only, OR an explicit execution witness. A sub-floor tier that
//     asks for one of these is not silently allowed and not silently dropped: it
//     gets a TYPED escalation, so the request is visible and routable, never lost.
//
// WITNESS SUBSTITUTES FOR TIER. The #3044 assumption is that a cheap model can
// drive routine monitoring "if their authority is read-only and their outputs are
// witnessed before action." So an explicit, out-of-band execution witness
// (a human sign-off, a T0 corroboration) authorizes an otherwise over-authority
// action — but the orchestrator's OWN suggestion never counts as that witness.
//
// THE NUMBERING TRAP is inherited from modelroute: T0 is the MOST capable/most
// trusted but the LOWEST number, so authority order runs opposite the labels.
// Every tier comparison routes through modelroute.WorkTier.MeetsRequirement — the
// same guard C3/C5 use — so the inversion can never leak into a raw `<`.
//
// PURITY: no I/O, no launch, no git. A dead-safe fold the dispatch/status surface
// can wire in; nothing routes through it yet (the gen/next posture — gated until
// promotion evidence lands).

// OrchestratorTier is the capability/authority tier of the model DRIVING a
// meta-orchestrator role, read on the same T0/T1/T2 ladder modelroute defines but
// as an AUTHORITY level, not a work-difficulty floor. T0 is the most capable/most
// trusted (frontier); T2 is the cheapest/least trusted.
type OrchestratorTier = modelroute.WorkTier

// ActionRisk classifies a meta-orchestrator action by the damage it can do if the
// orchestrator is wrong. The order is monotone: a higher risk demands a more
// capable/more trusted tier (a LOWER OrchestratorTier number).
type ActionRisk int

const (
	// RiskReadOnly has no side effects — reading status, ranking, summarizing,
	// suggesting. Safe for any tier because a wrong answer mutates nothing.
	RiskReadOnly ActionRisk = iota
	// RiskGuardedMutation is a scoped, reversible change PROPOSED into an existing
	// gate (not executed past it). A T1 orchestrator may propose these.
	RiskGuardedMutation
	// RiskHighRisk is irreversible or authority-bearing: git mutation, killing a
	// process, launching a worker, editing a label, or claiming done. T0 only, or
	// an explicit execution witness.
	RiskHighRisk
)

// String renders the canonical risk label for a status surface.
func (r ActionRisk) String() string {
	switch r {
	case RiskReadOnly:
		return "read-only"
	case RiskGuardedMutation:
		return "guarded-mutation"
	case RiskHighRisk:
		return "high-risk"
	default:
		return "risk?"
	}
}

// MetaAction is a closed-vocabulary meta-orchestrator action verb. An unknown verb
// is REFUSED (fail closed), never silently allowed — a cheap model must not be
// able to invent a verb the envelope has never classified.
type MetaAction string

const (
	// Read-only verbs — the entire authority of a T2 orchestrator.
	ActionReadStatus  MetaAction = "read-status"
	ActionRank        MetaAction = "rank"
	ActionSummarize   MetaAction = "summarize"
	ActionSuggestNext MetaAction = "suggest-next"

	// Guarded-mutation verbs — a T1 orchestrator may propose these into a gate.
	ActionProposeMutation MetaAction = "propose-guarded-mutation"
	ActionComment         MetaAction = "comment"

	// High-risk verbs — the exact set #3044 names as requiring escalation.
	ActionGitMutate    MetaAction = "git-mutate"
	ActionKillProcess  MetaAction = "kill-process"
	ActionLaunchWorker MetaAction = "launch-worker"
	ActionEditLabel    MetaAction = "edit-label"
	ActionClaimDone    MetaAction = "claim-done"
)

// actionRisk is the closed classification table. A verb absent from this map is
// unknown and fails closed in Authorize.
var actionRisk = map[MetaAction]ActionRisk{
	ActionReadStatus:  RiskReadOnly,
	ActionRank:        RiskReadOnly,
	ActionSummarize:   RiskReadOnly,
	ActionSuggestNext: RiskReadOnly,

	ActionProposeMutation: RiskGuardedMutation,
	ActionComment:         RiskGuardedMutation,

	ActionGitMutate:    RiskHighRisk,
	ActionKillProcess:  RiskHighRisk,
	ActionLaunchWorker: RiskHighRisk,
	ActionEditLabel:    RiskHighRisk,
	ActionClaimDone:    RiskHighRisk,
}

// metaActionOrder is the canonical, deterministic verb ordering the readout uses,
// so an envelope render never depends on Go map iteration order.
var metaActionOrder = []MetaAction{
	ActionReadStatus, ActionRank, ActionSummarize, ActionSuggestNext,
	ActionProposeMutation, ActionComment,
	ActionGitMutate, ActionKillProcess, ActionLaunchWorker, ActionEditLabel, ActionClaimDone,
}

// requiredTierForRisk maps a risk class to the LEAST-authoritative orchestrator
// tier that may perform it directly (without a witness). Read-only -> T2 (any),
// guarded -> T1, high-risk -> T0. Because a lower WorkTier number is more capable,
// "tier T is authorized" is tier.MeetsRequirement(required).
func requiredTierForRisk(r ActionRisk) OrchestratorTier {
	switch r {
	case RiskReadOnly:
		return modelroute.TierT2
	case RiskGuardedMutation:
		return modelroute.TierT1
	default: // RiskHighRisk (and any out-of-range value) stays at the frontier floor
		return modelroute.TierT0
	}
}

// ActionOutcome is the typed verdict of the action gate. It is deliberately a
// small closed set so the dispatch surface renders it without free text.
type ActionOutcome string

const (
	// OutcomeAllow: the tier (or an explicit witness) is authorized for the action.
	OutcomeAllow ActionOutcome = "allow"
	// OutcomeEscalate: the action is legitimate but exceeds this tier's authority —
	// it must route to a higher tier or attach an execution witness. This is the
	// typed escalation, never a silent drop.
	OutcomeEscalate ActionOutcome = "escalate-required"
	// OutcomeRefuse: the verb is unknown to the closed vocabulary — fail closed.
	OutcomeRefuse ActionOutcome = "refuse"
)

// Closed-vocabulary reason strings — a status surface renders these verbatim.
const (
	MetaReasonReadOnlyAllowed      = "meta-read-only-within-envelope"
	MetaReasonTierAuthorized       = "meta-tier-authorized-for-risk"
	MetaReasonEscalationRequired   = "meta-escalation-required-over-authority"
	MetaReasonWitnessAuthorized    = "meta-witnessed-escalation-authorized"
	MetaReasonUnknownActionRefused = "meta-unknown-action-refused-fail-closed"
)

// MetaDecision is the action gate's verdict for ONE requested action by an
// orchestrator at a given tier, carrying every field the dispatch surface needs to
// explain the decision without free text.
type MetaDecision struct {
	Action       MetaAction      `json:"action"`
	Tier         OrchestratorTier `json:"tier"`
	Risk         ActionRisk      `json:"risk"`
	RequiredTier OrchestratorTier `json:"required_tier"`
	Outcome      ActionOutcome   `json:"outcome"`
	Witnessed    bool            `json:"witnessed"`
	Reason       string          `json:"reason"`
}

// Allowed reports whether the action may proceed (a small helper so callers read
// intent, not a string compare).
func (d MetaDecision) Allowed() bool { return d.Outcome == OutcomeAllow }

// Authorize is the action gate: it decides whether an orchestrator at `tier` may
// perform `action`, given whether an explicit out-of-band execution `witnessed`
// corroboration is attached. It is a pure fold — the SAME verdict for every caller,
// so a cheap model cannot talk past it.
//
// The rules, in order:
//  1. Unknown verb -> REFUSE (fail closed). A cheap model cannot invent authority.
//  2. Read-only -> ALLOW for any tier.
//  3. The tier meets the risk's required floor -> ALLOW (T1 proposes guarded
//     mutations; T0 performs high-risk actions).
//  4. Below the floor but an explicit witness is attached -> ALLOW (witnessed
//     escalation — the #3044 "witnessed before action" path).
//  5. Below the floor, no witness -> ESCALATE (typed, routable, never silent).
func Authorize(tier OrchestratorTier, action MetaAction, witnessed bool) MetaDecision {
	d := MetaDecision{Action: action, Tier: tier, Witnessed: witnessed}

	risk, known := actionRisk[action]
	if !known {
		// Fail closed: an unclassified verb is treated as the most dangerous and
		// refused outright, so a new/typo'd verb can never slip through as allowed.
		d.Risk = RiskHighRisk
		d.RequiredTier = modelroute.TierT0
		d.Outcome = OutcomeRefuse
		d.Reason = MetaReasonUnknownActionRefused
		return d
	}

	d.Risk = risk
	required := requiredTierForRisk(risk)
	d.RequiredTier = required

	if risk == RiskReadOnly {
		d.Outcome = OutcomeAllow
		d.Reason = MetaReasonReadOnlyAllowed
		return d
	}

	if tier.MeetsRequirement(required) {
		d.Outcome = OutcomeAllow
		d.Reason = MetaReasonTierAuthorized
		return d
	}

	if witnessed {
		d.Outcome = OutcomeAllow
		d.Reason = MetaReasonWitnessAuthorized
		return d
	}

	d.Outcome = OutcomeEscalate
	d.Reason = MetaReasonEscalationRequired
	return d
}

// OrchestratorEnvelope is the operator readout of a tier's authority: the highest
// risk class it may perform directly, the verbs it is allowed, and the verbs it
// must escalate. It is derived from Authorize (with no witness), so the readout can
// never disagree with the gate.
type OrchestratorEnvelope struct {
	Tier               OrchestratorTier `json:"tier"`
	MaxRisk            ActionRisk       `json:"max_risk"`
	Allowed            []MetaAction     `json:"allowed"`
	EscalationRequired []MetaAction     `json:"escalation_required"`
}

// Envelope folds the closed action vocabulary into the readout for a tier, judging
// each verb through Authorize (no witness) so the envelope and the gate share one
// source of truth. The verbs are visited in the canonical order, so the readout is
// deterministic.
func Envelope(tier OrchestratorTier) OrchestratorEnvelope {
	env := OrchestratorEnvelope{Tier: tier}
	for _, a := range metaActionOrder {
		d := Authorize(tier, a, false)
		if d.Allowed() {
			env.Allowed = append(env.Allowed, a)
			if d.Risk > env.MaxRisk {
				env.MaxRisk = d.Risk
			}
		} else {
			env.EscalationRequired = append(env.EscalationRequired, a)
		}
	}
	return env
}
