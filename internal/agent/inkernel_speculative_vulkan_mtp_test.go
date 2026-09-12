package agent

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// vulkanMTPPlannerBackend is a software witness for the resident sequence
// contract. It records the one target-panel operation without pretending to be
// physical Vulkan evidence.
type vulkanMTPPlannerBackend struct {
	*prefixReuseQwenBackend
	vocab             int
	sequenceCalls     int
	verificationCalls int
	verificationErr   error
}

func newVulkanMTPPlannerBackend(t *testing.T) *vulkanMTPPlannerBackend {
	t.Helper()
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	return &vulkanMTPPlannerBackend{
		prefixReuseQwenBackend: &prefixReuseQwenBackend{Backend: base},
		vocab:                  97,
	}
}

func (*vulkanMTPPlannerBackend) Name() string { return "vulkan" }

func (*vulkanMTPPlannerBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (*vulkanMTPPlannerBackend) Qwen35SequenceAllLogitsPath() string {
	return compute.Qwen35SequenceAllLogitsPath
}

func (*vulkanMTPPlannerBackend) Qwen35SequenceRawHiddenPath() string {
	return compute.Qwen35SequenceRawHiddenPath
}

func (b *vulkanMTPPlannerBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	b.sequenceCalls++
	if req.NeedAllLogits {
		b.verificationCalls++
		if b.verificationErr != nil {
			return compute.Qwen35SequencePrefillResult{}, b.verificationErr
		}
	}
	width := req.NumKVHeads * req.HeadDim
	for token := range req.TokenIDs {
		compactLayer := 0
		for _, layer := range req.Layers {
			if layer.Linear {
				continue
			}
			zero := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
			req.KV.AppendKV(compactLayer, zero, zero, zero, req.StartPos+token)
			compactLayer++
		}
	}

	logits := make([]float32, len(req.TokenIDs)*b.vocab)
	for row := range req.TokenIDs {
		// Token 7 is the deterministic target decision at every boundary.
		logits[row*b.vocab+7] = 1
	}
	result := compute.Qwen35SequencePrefillResult{
		LastHidden: compute.NewF32(b.Backend, []int{req.Hidden}, make([]float32, req.Hidden)),
		Tokens:     len(req.TokenIDs),
	}
	if req.NeedLogits {
		result.Logits = compute.NewF32(b.Backend, []int{b.vocab}, logits[len(logits)-b.vocab:])
	}
	if req.NeedAllLogits {
		result.LogitsRows = compute.NewF32(b.Backend, []int{len(req.TokenIDs), b.vocab}, logits)
	}
	if req.CaptureRawHidden {
		result.RawHiddenRows = compute.NewF32(b.Backend, []int{len(req.TokenIDs), req.Hidden}, make([]float32, len(req.TokenIDs)*req.Hidden))
	}
	return result, nil
}

type vulkanMTPCloser struct{ calls int }

func (c *vulkanMTPCloser) Close() error {
	c.calls++
	return nil
}

func vulkanMTPPlanner(t *testing.T, backend compute.Backend) (*InKernelPlanner, []int) {
	t.Helper()
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	cfg := speculativeHybridConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, nil, "qwen38-vulkan-mtp", false, backend, false)
	if err := p.EnableVulkanMTP(1); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	return p, []int{3, 5, 11, 13}
}

func TestInKernelPlannerVulkanMTPRequestLifecycle(t *testing.T) {
	for _, tc := range []struct {
		name        string
		proposalErr error
		factoryErr  error
	}{
		{name: "success"},
		{name: "proposal failure downgrades", proposalErr: errors.New("injected draft failure")},
		{name: "constructor refusal downgrades", factoryErr: errors.New("injected constructor refusal")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := newVulkanMTPPlannerBackend(t)
			p, prompt := vulkanMTPPlanner(t, backend)
			closer := &vulkanMTPCloser{}
			factoryCalls, proposalCalls := 0, 0
			p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
				factoryCalls++
				if depth != 1 || target.Backend != backend {
					t.Fatalf("factory target/depth = %T/%d, want request target/%d", target.Backend, depth, 1)
				}
				if _, err := target.VerifyTokenLineage(nil); err != nil {
					t.Fatalf("factory ran after target mutation: %v", err)
				}
				if tc.factoryErr != nil {
					return nil, nil, tc.factoryErr
				}
				gen := model.NewMTPProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
					proposalCalls++
					if _, err := target.VerifyTokenLineage(prompt); err != nil {
						t.Fatalf("proposal ran before prompt prefill: %v", err)
					}
					if tc.proposalErr != nil {
						return nil, tc.proposalErr
					}
					return []int{7}, nil
				})
				return gen, func() { _ = closer.Close() }, nil
			}

			res, err := p.generateReusedRecovering(context.Background(), prompt, 1, 0, 0, 0, nil, 0, 0, nil, nil)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			wantProposal, wantClose := 1, 1
			if tc.factoryErr != nil {
				wantProposal, wantClose = 0, 0
			}
			if factoryCalls != 1 || proposalCalls != wantProposal || closer.calls != wantClose {
				t.Fatalf("factory/propose/close = %d/%d/%d, want 1/%d/%d", factoryCalls, proposalCalls, closer.calls, wantProposal, wantClose)
			}
			if tc.proposalErr == nil && tc.factoryErr == nil && backend.verificationCalls != 1 {
				t.Fatalf("resident target verification calls = %d, want 1", backend.verificationCalls)
			}
			if (tc.proposalErr != nil || tc.factoryErr != nil) && backend.verificationCalls != 0 {
				t.Fatalf("downgraded draft reached target verifier %d times", backend.verificationCalls)
			}
			if res.vulkanMTP == nil {
				t.Fatal("selected request omitted Vulkan MTP execution receipt")
			}
			if tc.proposalErr == nil && tc.factoryErr == nil {
				if !res.vulkanMTP.Used || res.vulkanMTP.ProposalRounds != 1 || res.vulkanMTP.ProposedTokens != 1 || res.vulkanMTP.TargetOperations != 1 || res.vulkanMTP.DowngradeReason != "" {
					t.Fatalf("successful execution receipt = %+v", res.vulkanMTP)
				}
			} else if tc.proposalErr != nil && (res.vulkanMTP.Used || res.vulkanMTP.DowngradeReason != vulkanMTPDowngradeProposalError) {
				t.Fatalf("proposal failure receipt = %+v", res.vulkanMTP)
			} else if tc.factoryErr != nil && (res.vulkanMTP.Used || res.vulkanMTP.EffectiveDepth != 0 || res.vulkanMTP.DowngradeReason != vulkanMTPDowngradeConstructorRefused) {
				t.Fatalf("constructor refusal receipt = %+v", res.vulkanMTP)
			}
		})
	}
}

func TestInKernelPlannerVulkanMTPGreedyParity(t *testing.T) {
	const maxNew = 6
	prompt := []int{3, 5, 11, 13}
	ordinaryBackend := newVulkanMTPPlannerBackend(t)
	ordinaryModel := model.NewSynthetic(speculativeHybridConfig())
	ordinaryModel.Quantize()
	ordinary := NewInKernelPlanner(ordinaryModel, nil, "ordinary", false, ordinaryBackend, false)
	var want []int
	if _, err := ordinary.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("ordinary generate: %v", err)
	}

	backend := newVulkanMTPPlannerBackend(t)
	p, _ := vulkanMTPPlanner(t, backend)
	factoryCalls, proposalCalls, closeCalls := 0, 0, 0
	var targets []*model.Session
	p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
		factoryCalls++
		targets = append(targets, target)
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			proposalCalls++
			return []int{7}, nil
		}), func() { closeCalls++ }, nil
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
			t.Fatalf("MTP request %d output = %v, ordinary greedy = %v", request+1, got, want)
		}
	}
	if factoryCalls != 2 || closeCalls != 2 || len(targets) != 2 || targets[0] == targets[1] {
		t.Fatalf("request-bound factory/close/distinct targets = %d/%d/%t, want 2/2/true", factoryCalls, closeCalls, len(targets) == 2 && targets[0] != targets[1])
	}
	if proposalCalls < 4 || backend.verificationCalls != proposalCalls {
		t.Fatalf("multi-round propose/resident verify = %d/%d, want matching counts >= 4", proposalCalls, backend.verificationCalls)
	}
}

func TestInKernelPlannerVulkanMTPRestoredUncapturedPrefixReprefillsCold(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	backend := newVulkanMTPPlannerBackend(t)
	cfg := speculativeHybridConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, nil, "qwen38-vulkan-mtp-restore", false, backend, false)
	prompt := []int{3, 5, 11, 13}

	// Prime a reusable target snapshot before MTP capture is enabled.
	if _, err := p.generateReusedRecovering(context.Background(), prompt, 1, 0, 0, 0, nil, 0, 0, nil, nil); err != nil {
		t.Fatalf("prime uncaptured prefix: %v", err)
	}
	if err := p.EnableVulkanMTP(1); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}

	var targets []*model.Session
	closeCalls := 0
	p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
		targets = append(targets, target)
		switch len(targets) {
		case 1:
			if _, err := target.VerifyTokenLineage(prompt); err != nil {
				t.Fatalf("first constructor did not receive restored prompt state: %v", err)
			}
			return nil, nil, errors.New("restored prefix has no captured raw-hidden rows")
		case 2:
			if _, err := target.VerifyTokenLineage(nil); err != nil {
				t.Fatalf("retry constructor did not receive a fresh target: %v", err)
			}
			return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
				return []int{7}, nil
			}), func() { closeCalls++ }, nil
		default:
			t.Fatalf("factory called %d times, want restored attempt plus one fresh retry", len(targets))
			return nil, nil, errors.New("unexpected factory call")
		}
	}

	res, err := p.generateReusedRecovering(context.Background(), prompt, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("MTP generate after restored-prefix refusal: %v", err)
	}
	if len(targets) != 2 || targets[0] == targets[1] || closeCalls != 1 {
		t.Fatalf("restore retry targets/distinct/close = %d/%t/%d, want 2/true/1", len(targets), len(targets) == 2 && targets[0] != targets[1], closeCalls)
	}
	if res.matched != 0 || res.sourceTier != radixkv.SnapshotTierMiss {
		t.Fatalf("restored-prefix retry retained reused state: matched=%d source_tier=%v", res.matched, res.sourceTier)
	}
	if res.vulkanMTP == nil || !res.vulkanMTP.Used || res.vulkanMTP.DowngradeReason != "" {
		t.Fatalf("fresh re-prefill MTP receipt = %+v", res.vulkanMTP)
	}
}

func TestInKernelPlannerVulkanMTPResidentVerifierDowngradesOnce(t *testing.T) {
	const maxNew = 6
	prompt := []int{3, 5, 11, 13}
	ordinaryBackend := newVulkanMTPPlannerBackend(t)
	ordinaryModel := model.NewSynthetic(speculativeHybridConfig())
	ordinaryModel.Quantize()
	ordinary := NewInKernelPlanner(ordinaryModel, nil, "ordinary", false, ordinaryBackend, false)
	var want []int
	if _, err := ordinary.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("ordinary generate: %v", err)
	}

	backend := newVulkanMTPPlannerBackend(t)
	backend.verificationErr = model.ErrTargetVerificationDowngrade
	p, _ := vulkanMTPPlanner(t, backend)
	proposalCalls, closeCalls := 0, 0
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			proposalCalls++
			return []int{7}, nil
		}), func() { closeCalls++ }, nil
	}
	var got []int
	res, err := p.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		got = append(got, token)
		return false
	})
	if err != nil {
		t.Fatalf("MTP verifier downgrade: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("downgraded output = %v, ordinary greedy = %v", got, want)
	}
	if proposalCalls != 1 || backend.verificationCalls != 1 || closeCalls != 1 {
		t.Fatalf("proposal/resident verifier/close = %d/%d/%d, want 1/1/1", proposalCalls, backend.verificationCalls, closeCalls)
	}
	if res.vulkanMTP == nil || !res.vulkanMTP.Used || res.vulkanMTP.ProposalRounds != 1 || res.vulkanMTP.DowngradeReason != vulkanMTPDowngradeTargetVerifier {
		t.Fatalf("resident verifier downgrade receipt = %+v", res.vulkanMTP)
	}
}

func TestInKernelPlannerVulkanMTPExecutionReceiptAccountsResidentTransaction(t *testing.T) {
	backend := newVulkanMTPPlannerBackend(t)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	m := model.NewSynthetic(speculativeHybridConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, nil, "qwen38-vulkan-mtp-receipt", false, backend, false)
	if err := p.EnableVulkanMTP(2); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	closeCalls := 0
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		// The target chooses 7 at every row. The first token is accepted and 8 is
		// rejected, giving the planner a controlled partial-commit receipt.
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			// Guarantee a real monotonic tick so the receipt's positive elapsed-time
			// invariant holds portably, including coarse Windows timers (#12746):
			// production measures a true time.Since(started), so the synthetic request
			// must outlast the host clock's observable resolution.
			time.Sleep(time.Millisecond)
			return []int{7, 8}, nil
		}), func() { closeCalls++ }, nil
	}
	res, err := p.generateReusedRecovering(context.Background(), []int{3, 5, 11, 13}, 2, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", closeCalls)
	}
	r := res.vulkanMTP
	if r == nil {
		t.Fatal("missing Vulkan MTP execution receipt")
	}
	if r.Engine != vulkanMTPExecutionEngine || r.Backend != "vulkan" || r.RequestedDepth != 2 || r.EffectiveDepth != 2 || !r.Used || r.DowngradeReason != "" {
		t.Fatalf("receipt identity/admission = %+v", r)
	}
	if r.ProposalRounds != 1 || r.ProposedTokens != 2 || r.AcceptedTokens != 1 || r.RejectedTokens != 1 || r.RollbackTokens != 1 {
		t.Fatalf("receipt proposal accounting = %+v", r)
	}
	// This controlled backend intentionally omits prefix replay, so the one
	// accepted token takes the exact full-target replay path.
	if r.TargetOperations != 1 || r.TargetDecodeSteps != 0 || r.FullTargetReplaySteps != 1 || r.RecurrentRepairTokens != 0 {
		t.Fatalf("receipt resident target accounting = %+v", r)
	}
	if r.Elapsed <= 0 {
		t.Fatalf("receipt elapsed = %s, want positive request-local duration", r.Elapsed)
	}
}

func TestInKernelPlannerVulkanMTPCancellation(t *testing.T) {
	backend := newVulkanMTPPlannerBackend(t)
	p, prompt := vulkanMTPPlanner(t, backend)
	ctx, cancel := context.WithCancel(context.Background())
	closer := &vulkanMTPCloser{}
	p.vulkanMTPDraftFactory = func(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			cancel()
			return nil, context.Canceled
		}), func() { _ = closer.Close() }, nil
	}

	res, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("generate error = %v, want context.Canceled", err)
	}
	if closer.calls != 1 {
		t.Fatalf("close calls = %d, want 1", closer.calls)
	}
	if backend.verificationCalls != 0 {
		t.Fatalf("cancelled proposal reached target verifier %d times", backend.verificationCalls)
	}
	if res.vulkanMTP == nil || !res.vulkanMTP.Cancelled || res.vulkanMTP.Used {
		t.Fatalf("cancellation execution receipt = %+v", res.vulkanMTP)
	}
}

func TestInKernelPlannerVulkanMTPCompleteStreamPreservesErrorReceipt(t *testing.T) {
	backend := newVulkanMTPPlannerBackend(t)
	backend.vocab = 300
	cfg := speculativeHybridConfig()
	cfg.VocabSize = backend.vocab
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "qwen38-vulkan-mtp-stream", false, backend, false)
	if err := p.EnableVulkanMTP(1); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			cancel()
			return nil, context.Canceled
		}), func() {}, nil
	}
	comp, err := p.CompleteStream(ctx, func(string) error {
		t.Fatal("cancelled request emitted a stream fragment")
		return nil
	}, []Message{{Role: RoleUser, Content: "hi"}}, nil, WithMaxTokens(2))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CompleteStream error = %v, want context.Canceled", err)
	}
	if comp == nil || comp.VulkanMTP == nil || !comp.VulkanMTP.Cancelled || comp.VulkanMTP.DowngradeReason != vulkanMTPDowngradeCancelled {
		t.Fatalf("CompleteStream cancellation receipt = %+v", comp)
	}
}

// TestInKernelPlannerVulkanMTPRealArtifact is an opt-in physical correctness
// witness. The default package run skips it because loading the production
// Q4_K artifact is an explicitly scheduled hardware operation.
func TestInKernelPlannerVulkanMTPRealArtifact(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	path := strings.TrimSpace(os.Getenv("FAK_VULKAN_MTP_PLANNER_GGUF"))
	if path == "" {
		t.Skip("set FAK_VULKAN_MTP_PLANNER_GGUF to an admitted retained-head Q4_K artifact")
	}
	backend, ok := compute.Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_VULKAN_REQUIRE_DEVICE=1 but Vulkan backend is unavailable")
		}
		t.Skip("Vulkan backend unavailable")
	}
	if expected := strings.TrimSpace(os.Getenv("FAK_VULKAN_EXPECT_DEVICE")); expected != "" && !strings.Contains(strings.ToLower(backend.Tier()), strings.ToLower(expected)) {
		t.Fatalf("Vulkan device %q does not contain expected %q", backend.Tier(), expected)
	}

	oldRetain := model.RetainMTP
	model.RetainMTP = true
	defer func() { model.RetainMTP = oldRetain }()
	m, err := ggufload.LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatalf("load retained Q4_K artifact: %v", err)
	}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close weights: %v", err)
		}
	}()
	if !m.Cfg.IsQwen35Hybrid() || !m.Cfg.HasMTPHead() {
		t.Fatalf("artifact is not an admitted Qwen3.8 hybrid with retained MTP head: %+v", m.Cfg)
	}

	prompt := []int{1, 2, 3, 4}
	const maxNew = 16
	ordinary := NewInKernelPlanner(m, nil, "qwen38-vulkan-target", true, backend, false)
	var want []int
	if _, err := ordinary.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("ordinary resident target decode: %v", err)
	}

	mtp := NewInKernelPlanner(m, nil, "qwen38-vulkan-mtp", true, backend, false)
	if err := mtp.EnableVulkanMTP(model.Qwen35MTPMaxDraftDepth); err != nil {
		t.Fatalf("EnableVulkanMTP: %v", err)
	}
	window, available, err := compute.BeginBackendExecutionObservation(backend)
	if err != nil || !available {
		t.Fatalf("begin physical Vulkan observation: available=%t err=%v", available, err)
	}
	var got []int
	res, err := mtp.generateReusedRecovering(context.Background(), prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		got = append(got, token)
		return false
	})
	if err != nil {
		t.Fatalf("resident Vulkan MTP decode: %v", err)
	}
	observation, err := window.End()
	if err != nil {
		t.Fatalf("end physical Vulkan observation: %v", err)
	}
	if !strings.EqualFold(observation.Identity.Backend, "vulkan") || observation.Identity.Device == "" || observation.Counters.ComputeDispatches == 0 || observation.Counters.DispatchSubmits == 0 {
		t.Fatalf("physical Vulkan execution observation = %+v", observation)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MTP tokens = %v, ordinary target tokens = %v", got, want)
	}
	if res.vulkanMTP == nil || !res.vulkanMTP.Used || res.vulkanMTP.ProposedTokens == 0 || res.vulkanMTP.TargetOperations == 0 || res.vulkanMTP.DowngradeReason != "" || res.vulkanMTP.FullTargetReplaySteps != 0 {
		t.Fatalf("physical planner execution receipt = %+v", res.vulkanMTP)
	}
	t.Logf("vulkan_mtp device=%q dispatches=%d submits=%d generated=%d rounds=%d proposed=%d accepted=%d rejected=%d target_ops=%d elapsed=%s",
		observation.Identity.Device, observation.Counters.ComputeDispatches, observation.Counters.DispatchSubmits,
		len(got), res.vulkanMTP.ProposalRounds, res.vulkanMTP.ProposedTokens,
		res.vulkanMTP.AcceptedTokens, res.vulkanMTP.RejectedTokens, res.vulkanMTP.TargetOperations,
		res.vulkanMTP.Elapsed)
}
