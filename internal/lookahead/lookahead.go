// Package lookahead is the witness-gated Lesson core (#5204, child of #5202): the pure
// distillation of a fork-rollout's outcome into a Lesson whose assertive authority is bounded
// by the witness rung its evidence actually earned. A lesson may never claim more than its
// evidence proves — a W2 (structured-activity) rollout may only flag a RISK, and a self-report
// or judge verdict (W0/W1) rollout may assert nothing at all. This is "the kernel does not
// believe the agents" applied to the lookahead loop's own distilled foresight: opinion and
// self-report never masquerade as a witnessed fact.
//
// The leaf is pure (stdlib + trajctl for the rung/evidence vocabulary): no I/O, no model call.
// The model seam is injected (DistillFunc) exactly like sessionreset.modelDistill, so a nil
// seam, a too-short transcript, or a seam error is a graceful DECLINE that never poisons a seed,
// and the witness gate is a deterministic, golden-tabled function of (evidence rung, proposed
// kind).
package lookahead

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// RolloutEvidence is the witnessed outcome of one fork-rollout: which forked session produced
// it, the base trunk SHA it forked from, how many turns it ran, the MAX witness rung across its
// collected witnesses, and the witness refs themselves (build/test exits, shadowgit snapshot
// SHAs, curve rows). Rung is the authority the gate reads — it is the strongest rung the rollout
// actually earned, not a claim.
type RolloutEvidence struct {
	ForkSessionID string                `json:"fork_session_id"`
	BaseSHA       string                `json:"base_sha"`
	Turns         int                   `json:"turns"`
	Rung          trajctl.WitnessRung   `json:"rung"`
	Witnesses     []trajctl.EvidenceRef `json:"witnesses,omitempty"`
}

// LessonKind is the assertive force a lesson carries. A FACT is a witnessed claim the consumer
// may act on as ground truth; a RISK is a flag that MAY hold and wants corroboration. The gate
// decides which force the evidence's rung permits.
type LessonKind string

const (
	// KindFact is a witnessed claim — only a W3 (deterministic) rollout may assert it.
	KindFact LessonKind = "FACT"
	// KindRisk is a corroboration-wanting flag — the strongest a W2 rollout may assert.
	KindRisk LessonKind = "RISK"
)

// Lesson is the distilled, authority-bounded takeaway of a rollout. Rung mirrors the evidence
// rung so a consumer can render it visibly (Render) and never read a RISK as a FACT. ExpiresSHA
// pins the staleness horizon: the lesson decays once trunk moves past it (Stale).
type Lesson struct {
	Claim      string              `json:"claim"`
	Kind       LessonKind          `json:"kind"`
	Rung       trajctl.WitnessRung `json:"rung"`
	Evidence   RolloutEvidence     `json:"evidence"`
	ExpiresSHA string              `json:"expires_sha"`
}

// Proposal is what a distillation seam proposes from a rollout transcript: a claim and the kind
// it wants to assert. The gate then decides whether the evidence's rung permits that kind — the
// seam proposes, the witness gate disposes.
type Proposal struct {
	Claim string
	Kind  LessonKind
}

// DistillFunc is the injected model seam: it reads a rollout transcript and proposes a lesson,
// returning ok=false to decline (an empty/uncertain transcript, a model error). Mirrors
// sessionreset.modelDistill's SummarizeFunc — kept an interface seam so the leaf stays pure and
// testable, and a nil DistillFunc is a valid "no seam" that declines rather than panics.
type DistillFunc func(transcript string) (Proposal, bool)

// ReasonLessonOverclaims is the closed refusal reason (declared in dos.toml) for a lesson that
// asserts more authority than its witness rung earned: a FACT below W3, or ANY lesson distilled
// from a W1/W0 rollout. It is the gate's single verdict token, so an appeal or audit binds to
// one reason rather than free text.
const ReasonLessonOverclaims = "LESSON_OVERCLAIMS"

// Decline reasons — a graceful fall-through that yields no lesson AND no refusal, so the caller
// simply proceeds without a distilled seed (never a poisoned one). Mirrors the modelDistill
// decline vocabulary.
const (
	DeclineNoSeam         = "no_model_seam"
	DeclineTranscriptTiny = "transcript_too_short"
	DeclineModelError     = "model_error"
)

// minLessonTranscriptChars is the short-transcript decline floor: a rollout this brief is not
// worth a model call to distill (mirrors sessionreset.modelDistill's minChars gate).
const minLessonTranscriptChars = 200

// DistillOutcome is the honest result of a distillation attempt. Exactly one of OK / Declined /
// Refused is true. Declined is a graceful no-op (no seam / too short / model error); Refused is
// the witness gate rejecting an overclaiming lesson (Reason == ReasonLessonOverclaims). Reason
// names the decline or refusal so the audit surface can show WHY no lesson was produced.
type DistillOutcome struct {
	OK       bool   `json:"ok"`
	Declined bool   `json:"declined,omitempty"`
	Refused  bool   `json:"refused,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// Invariant: lookahead lesson distillation is fail-closed and deterministic.
// Guard: witness gating permits FACT only under W3 evidence; W2 is restricted to RISK, and W0/W1 are refused with LESSON_OVERCLAIMS.
//
// DistillLesson distills a Lesson from a fork-rollout's evidence and transcript through an
// injected seam, then applies the witness gate. Decline discipline first (never poisons a seed):
// a nil seam, a transcript below minLessonTranscriptChars, or a seam that returns ok=false is a
// graceful DECLINE. On a proposal the gate binds the lesson's authority to the evidence rung:
//
//   - W3 (deterministic) evidence may assert FACT or RISK;
//   - W2 (structured-activity) evidence may assert only RISK — a proposed FACT is REFUSED
//     LESSON_OVERCLAIMS;
//   - W1 (judge) / W0 (self-report) / any unknown rung may assert NOTHING — any lesson is
//     REFUSED LESSON_OVERCLAIMS.
//
// An unspecified/blank proposed kind at a permitted rung defaults to RISK (the safe floor).
// expiresSHA pins the returned lesson's staleness horizon.
func DistillLesson(ev RolloutEvidence, transcript, expiresSHA string, distill DistillFunc) (Lesson, DistillOutcome) {
	if distill == nil {
		return Lesson{}, DistillOutcome{Declined: true, Reason: DeclineNoSeam}
	}
	if len(strings.TrimSpace(transcript)) < minLessonTranscriptChars {
		return Lesson{}, DistillOutcome{Declined: true, Reason: DeclineTranscriptTiny}
	}
	prop, ok := distill(transcript)
	if !ok {
		return Lesson{}, DistillOutcome{Declined: true, Reason: DeclineModelError}
	}
	kind := LessonKind(strings.ToUpper(strings.TrimSpace(string(prop.Kind))))
	switch ev.Rung {
	case trajctl.W3:
		if kind != KindFact {
			kind = KindRisk
		}
	case trajctl.W2:
		if kind == KindFact {
			return Lesson{}, DistillOutcome{Refused: true, Reason: ReasonLessonOverclaims}
		}
		kind = KindRisk
	default:
		// W1, W0, or an unrecognized rung: judgement and self-report never masquerade as foresight.
		return Lesson{}, DistillOutcome{Refused: true, Reason: ReasonLessonOverclaims}
	}
	return Lesson{
		Claim:      strings.TrimSpace(prop.Claim),
		Kind:       kind,
		Rung:       ev.Rung,
		Evidence:   ev,
		ExpiresSHA: strings.TrimSpace(expiresSHA),
	}, DistillOutcome{OK: true}
}

// Render renders the lesson with its rung visible so a consumer never reads a RISK flag as a
// witnessed FACT: "Witnessed (W3): <claim>" vs "Risk flag (W2): <claim>".
func (l Lesson) Render() string {
	if l.Kind == KindFact {
		return "Witnessed (" + string(l.Rung) + "): " + l.Claim
	}
	return "Risk flag (" + string(l.Rung) + "): " + l.Claim
}

// Stale reports whether the lesson has expired at inject time. A lesson pins to ExpiresSHA and
// decays once trunk moves PAST it; the git ancestry check that decides "trunk moved past
// ExpiresSHA" is the caller's concern (this leaf stays pure), so the caller passes the boolean.
// An empty ExpiresSHA is an un-pinned lesson that never goes stale.
func (l Lesson) Stale(trunkMovedPastExpires bool) bool {
	if strings.TrimSpace(l.ExpiresSHA) == "" {
		return false
	}
	return trunkMovedPastExpires
}
