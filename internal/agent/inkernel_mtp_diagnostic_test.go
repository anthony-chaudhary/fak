package agent

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// This backend exercises the actual planner, verifier, Commit and stop paths.
// Like vulkanMTPPlannerBackend it is a software witness, not device evidence.
type mtpDiagnosticBackend struct {
	*vulkanMTPPlannerBackend
	now *time.Time
}

func (b *mtpDiagnosticBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	if req.NeedAllLogits {
		*b.now = b.now.Add(10 * time.Millisecond)
	}
	return b.vulkanMTPPlannerBackend.Qwen35SequencePrefill(req)
}

func (b *mtpDiagnosticBackend) Read(t compute.Tensor) []float32 {
	out := b.vulkanMTPPlannerBackend.Read(t)
	// Keep scalar Step decisions consistent with the controlled panel rows.
	if len(t.Shape) == 1 && t.Shape[0] == b.vocab {
		out = make([]float32, b.vocab)
		out[7] = 1
	}
	return out
}

func mtpDiagnosticPlanner(t *testing.T, depth int, draft []int) (*InKernelPlanner, []int, context.Context, *int) {
	t.Helper()
	now := time.Unix(100, 0)
	b := &mtpDiagnosticBackend{vulkanMTPPlannerBackend: newVulkanMTPPlannerBackend(t), now: &now}
	p, prompt := vulkanMTPPlanner(t, b)
	if err := p.EnableVulkanMTP(depth); err != nil {
		t.Fatal(err)
	}
	calls := new(int)
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(_ context.Context, _ []int, maxDraft int) ([]int, error) {
			*calls++
			now = now.Add(3 * time.Millisecond)
			return append([]int(nil), draft[:min(maxDraft, len(draft))]...), nil
		}), func() {}, nil
	}
	ctx := context.WithValue(context.Background(), vulkanMTPDiagnosticClockKey{}, func() time.Time { return now })
	return p, prompt, ctx, calls
}

func TestVulkanMTPDiagnosticPlannerInclusiveAndPostCommitStop(t *testing.T) {
	p, prompt, ctx, calls := mtpDiagnosticPlanner(t, 4, []int{7, 7, 7, 8})
	res, err := p.generateReusedRecovering(ctx, prompt, 8, 0, 0, 0, nil, 0, 0, nil, func(int) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if res.vulkanMTP == nil || len(res.vulkanMTP.RoundDiagnostics) != 1 || *calls != 1 {
		t.Fatalf("missing actual planner round: %+v calls=%d", res.vulkanMTP, *calls)
	}
	totals := res.vulkanMTP.ClosedRoundTotals
	if totals == nil || totals.Scope != "closed-successfully-verified-rounds-only" || totals.Rounds != 1 || totals.ProposedTokens != 4 || totals.AcceptedTokens != 3 || totals.AcceptedEmittedTokens != 1 || totals.BonusEmittedTokens != 0 || totals.EmittedTokens != 1 || totals.DraftElapsed != 3*time.Millisecond || totals.InclusiveElapsed != 13*time.Millisecond {
		t.Fatalf("partial-stop aggregate mismatch: %+v", totals)
	}
	d := res.vulkanMTP.RoundDiagnostics[0]
	if !d.RoundClosed || d.InclusiveElapsed != 13*time.Millisecond || d.DraftElapsed != 3*time.Millisecond || d.ProposedTokens != 4 || d.AcceptedTokens != 3 || d.AcceptedEmittedTokens != 1 || d.EmittedTokens != 1 || d.BonusEmitted {
		t.Fatalf("planner timing/emitted-prefix mismatch: %+v", d)
	}
	// The controlled backend lacks prefix replay: Commit replays all three
	// accepted tokens before publication stops after one. Pre-Commit receipts
	// would not have these committed counters.
	if d.Target == nil || d.Target.AcceptedTokens != 3 || d.Target.FullTargetReplaySteps != 3 || !d.Target.Accounting.Synchronization.Measured || d.Target.Accounting.Drafting.Measured || d.Target.EndToEndMeasured() {
		t.Fatalf("post-Commit accounting missing or promoted: %+v", d.Target)
	}
}

func TestVulkanMTPDiagnosticPlannerAcceptanceFloor(t *testing.T) {
	for _, tc := range []struct {
		name         string
		draft        []int
		wantAccepted int
		wantFallback bool
	}{
		{"eight of thirty-two", []int{7, 8, 8, 8}, 8, true},
		{"exact sixteen of thirty-two", []int{7, 7, 8, 8}, 16, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, prompt, ctx, calls := mtpDiagnosticPlanner(t, 4, tc.draft)
			res, err := p.generateReusedRecovering(ctx, prompt, 40, 0, 0, 0, nil, 0, 0, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			r := res.vulkanMTP
			if r == nil {
				t.Fatal("missing receipt")
			}
			f := r.AcceptanceFallback
			if tc.wantFallback {
				if f == nil || *calls != 8 || f.Phase != "decode" || f.AfterRounds != 8 || f.EmittedTokens != 16 || f.DecodeElapsed != 104*time.Millisecond || f.Acceptance.WindowProposed != 32 || f.Acceptance.WindowAccepted != tc.wantAccepted || f.Acceptance.RollingRate != .25 || f.Acceptance.MinAcceptanceRate != .5 || !f.Acceptance.InFallback {
					t.Fatalf("wrong real planner fallback onset: %+v calls=%d", f, *calls)
				}
			} else if f != nil || *calls <= 8 || r.DowngradeReason != "" {
				t.Fatalf("exact floor incorrectly fell back: %+v calls=%d", r, *calls)
			}
		})
	}
}

func TestVulkanMTPDiagnosticPlannerBoundedRequestOwnership(t *testing.T) {
	p, prompt, ctx, _ := mtpDiagnosticPlanner(t, 1, []int{7})
	wantRounds := vulkanMTPDiagnosticRoundLimit + 1
	wantTokens := wantRounds * 2
	first, err := p.generateReusedRecovering(ctx, prompt, wantTokens, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	a := first.vulkanMTP
	if a == nil {
		t.Fatal("missing receipt")
	}
	retainedRounds := len(a.RoundDiagnostics)
	if a.ProposalRounds != wantRounds || retainedRounds != min(a.ProposalRounds, vulkanMTPDiagnosticRoundLimit) || a.RoundDiagnosticsDropped != a.ProposalRounds-retainedRounds {
		t.Fatalf("retention changed execution or lost drop count: %+v", a)
	}
	totals := a.ClosedRoundTotals
	if totals == nil || totals.Rounds != a.ProposalRounds || totals.ProposedTokens != a.ProposalRounds || totals.AcceptedTokens != a.ProposalRounds || totals.AcceptedEmittedTokens != a.ProposalRounds || totals.BonusEmittedTokens != a.ProposalRounds || totals.EmittedTokens != wantTokens || totals.DraftElapsed != time.Duration(a.ProposalRounds)*3*time.Millisecond || totals.InclusiveElapsed != time.Duration(a.ProposalRounds)*13*time.Millisecond {
		t.Fatalf("aggregate truncated with round retention: %+v", totals)
	}
	second, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := second.vulkanMTP
	if b == nil || len(b.RoundDiagnostics) != 1 || b.RoundDiagnosticsDropped != 0 || b.RoundDiagnostics[0] == a.RoundDiagnostics[0] || b.RoundDiagnostics[0].Target == a.RoundDiagnostics[0].Target {
		t.Fatalf("request diagnostics alias or retain previous count: %+v", b)
	}
	if b.ClosedRoundTotals == nil || b.ClosedRoundTotals == a.ClosedRoundTotals || b.ClosedRoundTotals.Rounds != 1 {
		t.Fatal("aggregate leaked across requests")
	}
	b.RoundDiagnostics[0].Target.DowngradeReason = "mutation probe"
	if a.RoundDiagnostics[0].Target.DowngradeReason != "" {
		t.Fatal("cross-request target alias")
	}
}

func TestVulkanMTPDiagnosticReceiptShapeIsIndependent(t *testing.T) {
	r := model.TargetVerificationReceipt{Shape: &model.SpeculativeShape{TokensPerSeq: []int{4}}}
	d := newVulkanMTPRoundDiagnostic(1, 0, 4, 3, time.Millisecond, &r)
	r.Shape.TokensPerSeq[0] = 9
	if d.Target.Shape.TokensPerSeq[0] != 4 {
		t.Fatal("shape slice aliases source receipt")
	}
}
