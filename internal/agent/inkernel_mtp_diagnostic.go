package agent

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Bound request-owned diagnostic retention without affecting execution or policy.
const vulkanMTPDiagnosticRoundLimit = 256

// VulkanMTPRoundDiagnostic is request-owned observation, not a performance
// qualification. InclusiveElapsed includes DraftElapsed and target accounting;
// these overlapping wall-time spans MUST NOT be summed. An absent Target means
// the generic verifier supplied no rich receipt, not that target work cost zero.
// Only successfully verified rounds are retained by this first collection seam.
// RoundClosed means timing/emission observation closed, not request success.
type VulkanMTPRoundDiagnostic struct {
	Round                 int
	OutputStart           int
	ProposedTokens        int
	AcceptedTokens        int
	AcceptedEmittedTokens int
	BonusEmitted          bool
	EmittedTokens         int
	DraftElapsed          time.Duration
	InclusiveElapsed      time.Duration
	RoundClosed           bool
	Target                *model.TargetVerificationReceipt
}

// VulkanMTPFallbackDiagnostic records the transition before the first serial
// remainder iteration. This first seam records acceptance-floor transitions
// only; other downgrade reasons remain in the existing request/target receipts.
// Acceptance.LifetimeRate is the engine's cumulative statistic, NOT request-local.
type VulkanMTPFallbackDiagnostic struct {
	Phase         string
	AfterRounds   int
	EmittedTokens int
	DecodeElapsed time.Duration
	Acceptance    model.SpeculativeAcceptanceStats
}

func newVulkanMTPRoundDiagnostic(round, outputStart, proposed, accepted int, draftElapsed time.Duration, target *model.TargetVerificationReceipt) *VulkanMTPRoundDiagnostic {
	d := &VulkanMTPRoundDiagnostic{
		Round: round, OutputStart: outputStart, ProposedTokens: proposed,
		AcceptedTokens: accepted, DraftElapsed: draftElapsed,
	}
	if target != nil {
		// VerifyGreedyDeviceDraft returns this receipt after Commit. Preserve its
		// actual verifier path, nested downgrade, repair/replay counts and every
		// measured flag rather than inventing missing category measurements.
		copy := *target
		if target.Shape != nil {
			shape := *target.Shape
			shape.TokensPerSeq = append([]int(nil), target.Shape.TokensPerSeq...)
			copy.Shape = &shape
		}
		d.Target = &copy
	}
	return d
}

// The optional clock belongs to one request context, never a shared planner or
// global recorder. It affects diagnostics only, not engine timing or policy.
type vulkanMTPDiagnosticClockKey struct{}

func vulkanMTPDiagnosticClock(ctx context.Context) func() time.Time {
	if now, ok := ctx.Value(vulkanMTPDiagnosticClockKey{}).(func() time.Time); ok && now != nil {
		return now
	}
	return time.Now
}

// VulkanMTPClosedRoundTotals covers every successfully verified round that
// closed, including partial emission stops and rounds beyond retention limits.
// It excludes proposal/verification failures, prefill and serial fallback work.
// InclusiveElapsed includes DraftElapsed: never add these overlapping spans.
// Nil totals mean no closed-round measurement, not measured zero cost.
type VulkanMTPClosedRoundTotals struct {
	Scope                 string
	Rounds                int
	ProposedTokens        int
	AcceptedTokens        int
	AcceptedEmittedTokens int
	BonusEmittedTokens    int
	EmittedTokens         int
	DraftElapsed          time.Duration
	InclusiveElapsed      time.Duration
}
