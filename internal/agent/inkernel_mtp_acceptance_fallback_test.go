package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelPlannerMTPSpeculativeDecode is the software witness for issue
// #12184: the native Qwen3.8 MTP speculative decode loop in the in-kernel
// planner (1) drives a K=4 draft/verify/rollback loop without diverging from
// unassisted greedy output, and (2) falls back to unassisted serial decode once
// the engine's rolling 32-token acceptance rate drops below the 50% floor,
// without stalling the session. Physical tok/s is out of scope here and remains
// [HW-WITNESSED].
func TestInKernelPlannerMTPSpeculativeDecode(t *testing.T) {
	const maxNew = 40
	prompt := []int{3, 5, 11, 13}

	// Unassisted greedy reference: the target decision is token 7 at every step.
	referenceBackend := newVulkanMTPPlannerBackend(t)
	referenceModel := model.NewSynthetic(speculativeHybridConfig())
	referenceModel.Quantize()
	reference := NewInKernelPlanner(referenceModel, nil, "mtp-reference", false, referenceBackend, false)
	var want []int
	if _, err := reference.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("reference generate: %v", err)
	}
	if len(want) != maxNew {
		t.Fatalf("reference generated %d tokens, want %d", len(want), maxNew)
	}

	// Speculative planner with a K=4 draft factory that always proposes token 8,
	// which the target never accepts. Every round therefore contributes four
	// rejected draft tokens, so the trailing 32-token window fills at 0%.
	backend := newVulkanMTPPlannerBackend(t)
	p, _ := vulkanMTPPlanner(t, backend)
	if err := p.EnableVulkanMTP(4); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	factoryCalls, proposalCalls, closeCalls := 0, 0, 0
	p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
		factoryCalls++
		if depth != 4 {
			t.Fatalf("draft depth = %d, want 4", depth)
		}
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			proposalCalls++
			return []int{8, 8, 8, 8}, nil
		}), func() { closeCalls++ }, nil
	}

	var got []int
	res, err := p.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		got = append(got, token)
		return false
	})
	if err != nil {
		t.Fatalf("MTP generate: %v", err)
	}
	if res.gen != maxNew {
		t.Fatalf("MTP generated %d tokens, want %d (fallback must not stall the session)", res.gen, maxNew)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MTP output diverged from unassisted greedy:\n got: %v\nwant: %v", got, want)
	}
	if factoryCalls != 1 || closeCalls != 1 {
		t.Fatalf("request-bound factory/close = %d/%d, want 1/1", factoryCalls, closeCalls)
	}

	// The rolling monitor must have observed enough rejected draft tokens to fill
	// the 32-token window and tripped the serial fallback.
	eng := p.SpeculativeEngine()
	if eng == nil {
		t.Fatal("planner lost its speculative engine")
	}
	stats := eng.AcceptanceStats()
	if !stats.InFallback || !stats.TripwireTripped {
		t.Fatalf("acceptance monitor did not trip: %+v", stats)
	}
	if stats.WindowTokens != 32 {
		t.Fatalf("rolling window tokens = %d, want full 32", stats.WindowTokens)
	}
	if stats.RollingRate >= stats.MinAcceptanceRate {
		t.Fatalf("rolling rate %v is not below the %v floor", stats.RollingRate, stats.MinAcceptanceRate)
	}
	if stats.WindowProposed == 0 || stats.WindowAccepted != 0 {
		t.Fatalf("window proposed/accepted = %d/%d, want >0/0 for always-rejected drafts", stats.WindowProposed, stats.WindowAccepted)
	}

	// Drafting must stop once the fallback trips: with four rejected drafts per
	// full window (32/4 == 8 rounds) the remaining ~8 tokens decode serially, so
	// proposal calls stay well below one per generated token.
	if proposalCalls >= maxNew {
		t.Fatalf("proposal calls = %d, want < %d (fallback must stop drafting, not retry per token)", proposalCalls, maxNew)
	}

	// The request boundary reports the fallback as an explicit downgrade reason.
	if res.vulkanMTP == nil {
		t.Fatal("selected request omitted Vulkan MTP execution receipt")
	}
	if res.vulkanMTP.DowngradeReason != vulkanMTPDowngradeAcceptanceFloor {
		t.Fatalf("downgrade reason = %q, want %q", res.vulkanMTP.DowngradeReason, vulkanMTPDowngradeAcceptanceFloor)
	}
}

// TestInKernelPlannerMTPAcceptanceMonitorResetsPerRequest proves the rolling
// acceptance monitor is request-scoped: after one request trips the serial
// fallback, the next request on the same planner starts speculative again and
// still matches unassisted greedy output.
func TestInKernelPlannerMTPAcceptanceMonitorResetsPerRequest(t *testing.T) {
	const maxNew = 40
	prompt := []int{3, 5, 11, 13}

	referenceBackend := newVulkanMTPPlannerBackend(t)
	referenceModel := model.NewSynthetic(speculativeHybridConfig())
	referenceModel.Quantize()
	reference := NewInKernelPlanner(referenceModel, nil, "mtp-reset-reference", false, referenceBackend, false)
	var want []int
	if _, err := reference.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("reference generate: %v", err)
	}

	backend := newVulkanMTPPlannerBackend(t)
	p, _ := vulkanMTPPlanner(t, backend)
	if err := p.EnableVulkanMTP(4); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	proposalCalls := 0
	p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			proposalCalls++
			return []int{8, 8, 8, 8}, nil
		}), func() {}, nil
	}

	for request := 0; request < 2; request++ {
		var got []int
		if _, err := p.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			got = append(got, token)
			return false
		}); err != nil {
			t.Fatalf("MTP request %d: %v", request+1, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("MTP request %d output diverged from greedy:\n got: %v\nwant: %v", request+1, got, want)
		}
	}
	// Both requests must have drafted; if request 2 inherited the fallback it
	// would decode entirely serially and the second request would add no calls.
	// One full window plus the pre-window partial fill yields more than 8 calls
	// per request; two requests must exceed the single-request count.
	if proposalCalls <= 8 {
		t.Fatalf("proposal calls = %d, want > 8 across two requests (request 2 must draft again)", proposalCalls)
	}
}
