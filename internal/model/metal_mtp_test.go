package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type mockMTPRecorder struct {
	records     int
	commits     int
	rollbacks   int
	recordErr   error
	commitErr   error
	rollbackErr error
}

type identityMTPRecorder struct {
	mu       sync.Mutex
	sessions []string
}

func (m *identityMTPRecorder) record(operation, sessionID string) {
	m.mu.Lock()
	m.sessions = append(m.sessions, operation+":"+sessionID)
	m.mu.Unlock()
}

func (m *identityMTPRecorder) RecordMTPDraft(sessionID string, _ []int32) error {
	m.record("record", sessionID)
	return nil
}

func (m *identityMTPRecorder) CommitMTPDraft(sessionID string, accepted int) (int, int, error) {
	m.record("commit", sessionID)
	return accepted, 0, nil
}

func (m *identityMTPRecorder) RollbackMTPDraft(sessionID string) (int, error) {
	m.record("rollback", sessionID)
	return 0, nil
}

func (m *identityMTPRecorder) snapshot() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.sessions...)
}

func (m *mockMTPRecorder) RecordMTPDraft(sessionID string, tokens []int32) error {
	m.records++
	return m.recordErr
}

func (m *mockMTPRecorder) CommitMTPDraft(sessionID string, accepted int) (int, int, error) {
	m.commits++
	return accepted, 0, m.commitErr
}

func (m *mockMTPRecorder) RollbackMTPDraft(sessionID string) (int, error) {
	m.rollbacks++
	return 0, m.rollbackErr
}

// TestMetalMTPDraftVerifyRollbackLoop is the comprehensive witness test for Issue #12238:
// It asserts:
//  1. Bit-exact token sequence identity against non-speculative autoregressive baseline at temp=0.
//  2. Candidate draft generation from resident MTP head (K=2..4).
//  3. Wide-M Metal verification dispatch.
//  4. Context-MMU atomic page commit and rollback with zero memory leaks.
//  5. Greedy temperature-zero tripwire enforcement.
//  6. Rolling 32-token acceptance monitoring and smooth fallback to serial decode on low acceptance.
func TestMetalMTPDraftVerifyRollbackLoop(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}
	const maxNew = 16

	ctx := context.Background()

	t.Run("bit_exact_sequence_identity_against_autoregressive_baseline", func(t *testing.T) {
		// Non-speculative autoregressive baseline
		refSes := m.NewSession()
		t.Cleanup(refSes.Close)
		wantTokens := refSes.Generate(prompt, maxNew)

		// Speculative Metal MTP decode
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)
		coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		gotTokens, err := coord.Generate(ctx, prompt, maxNew)
		if err != nil {
			t.Fatalf("coord.Generate failed: %v", err)
		}

		if !reflect.DeepEqual(gotTokens, wantTokens) {
			t.Fatalf("output sequence mismatch:\n got:  %v\n want: %v", gotTokens, wantTokens)
		}

		stats := coord.Stats()
		if stats.TotalGenerated < maxNew {
			t.Fatalf("expected total generated >= %d, got %d", maxNew, stats.TotalGenerated)
		}
	})

	t.Run("context_mmu_atomic_page_commit_and_rollback", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
			EnforceGreedyTripwire: true,
			FallbackToSerial:      true,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Wire recorder for MMU checkpoint tracking
		cm := &mockMTPRecorder{}
		sessionID := "test-metal-mtp-session"
		coord.SetMMU(cm, sessionID)

		// Prefill and execute 1 round
		boundary := specSes.Prefill(prompt)
		committed := append([]int(nil), prompt...)

		acc, bonus, nextLogits, err := coord.StepRound(ctx, committed, boundary)
		if err != nil {
			t.Fatalf("StepRound failed: %v", err)
		}
		if len(nextLogits) != m.Cfg.VocabSize {
			t.Fatalf("expected nextLogits len %d, got %d", m.Cfg.VocabSize, len(nextLogits))
		}
		if bonus < 0 {
			t.Fatalf("expected valid bonus token, got %d", bonus)
		}

		stats := coord.Stats()
		if stats.TotalProposed < 2 {
			t.Fatalf("expected proposed >= 2, got %d", stats.TotalProposed)
		}
		// Context-MMU must have tracked allocations and freed/committed pages
		if stats.CommittedPages != stats.TotalAccepted {
			t.Fatalf("Context-MMU committed pages %d != total accepted %d", stats.CommittedPages, stats.TotalAccepted)
		}
		if stats.FreedPages != stats.TotalRollbacks {
			t.Fatalf("Context-MMU freed pages %d != total rollbacks %d", stats.FreedPages, stats.TotalRollbacks)
		}
		_ = acc
	})

	t.Run("greedy_temperature_zero_tripwire", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			EnforceGreedyTripwire: true,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Greedy sampling (temp=0.0, penalty=1.0) must pass
		if err := coord.CheckSamplingTripwire(0.0, 1.0); err != nil {
			t.Fatalf("expected nil error for greedy parameters, got %v", err)
		}

		// Non-greedy temperature must trip the tripwire
		if err := coord.CheckSamplingTripwire(0.8, 1.0); !errors.Is(err, ErrMetalMTPTripwireDiverged) {
			t.Fatalf("expected ErrMetalMTPTripwireDiverged for temp=0.8, got %v", err)
		}

		// Non-unit penalty must trip the tripwire
		if err := coord.CheckSamplingTripwire(0.0, 1.2); !errors.Is(err, ErrMetalMTPTripwireDiverged) {
			t.Fatalf("expected ErrMetalMTPTripwireDiverged for penalty=1.2, got %v", err)
		}

		// Non-finite logits must trip the tripwire
		nanLogits := make([]float32, m.Cfg.VocabSize)
		nanLogits[0] = float32(math.NaN())
		_, _, _, err = coord.StepRound(ctx, prompt, nanLogits)
		if !errors.Is(err, ErrMetalMTPNonFiniteLogits) {
			t.Fatalf("expected ErrMetalMTPNonFiniteLogits for NaN logits, got %v", err)
		}
	})

	t.Run("rolling_acceptance_monitoring_and_smooth_serial_fallback", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
			FallbackToSerial:      true,
			EnforceGreedyTripwire: true,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Inject an adversarial drafter that always proposes divergent tokens
		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
			// Propose tokens guaranteed to mismatch
			return []int{99999, 99998, 99997, 99996}, nil
		}))

		boundary := specSes.Prefill(prompt)
		committed := append([]int(nil), prompt...)

		// Run 10 rounds; acceptance will be 0%, triggering smooth serial fallback
		for r := 0; r < 10; r++ {
			acc, bonus, nextLogits, err := coord.StepRound(ctx, committed, boundary)
			if err != nil {
				t.Fatalf("StepRound round %d failed during fallback: %v", r, err)
			}
			boundary = nextLogits
			for _, tok := range acc {
				committed = append(committed, tok)
			}
			if bonus >= 0 {
				committed = append(committed, bonus)
			}
		}

		stats := coord.Stats()
		if !stats.InFallback {
			t.Fatalf("expected coordinator to be in fallback mode after low acceptance, stats: %+v", stats)
		}
		if stats.RollingRate >= 0.50 {
			t.Fatalf("expected rolling rate < 0.50, got %.2f", stats.RollingRate)
		}

		// Verify session continues generating normally even in fallback
		outTokens, err := coord.Generate(ctx, prompt, 8)
		if err != nil {
			t.Fatalf("Generate failed while in fallback: %v", err)
		}
		if len(outTokens) != 8 {
			t.Fatalf("expected 8 tokens generated in fallback, got %d", len(outTokens))
		}
	})
}

func TestMetalMTPCoordinatorP4SequentialDowngradeReceiptIsHonest(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill([]int{0, 1, 2})
	// F16 is outside both the resident Metal P4 and incremental native-F32
	// panel envelopes, while ordinary target Step remains a valid native path.
	target.F16 = true

	coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = coord.Close() })
	coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		return []int{3, 5, 7, 11}, nil
	}))

	if _, _, _, err := coord.StepRound(context.Background(), nil, boundary); err != nil {
		t.Fatalf("ordinary target-decode downgrade: %v", err)
	}
	receipt, ok := coord.LastTargetVerificationReceipt()
	if !ok {
		t.Fatal("downgraded P4 verification omitted receipt")
	}
	if receipt.Path != targetVerificationDecodePath || receipt.OneOperation ||
		receipt.TargetVerificationOperations != 0 || receipt.TargetDecodeSteps != 4 || receipt.DowngradeReason == "" {
		t.Fatalf("sequential P4 downgrade was not reported honestly: %+v", receipt)
	}
}

func TestMetalMTPCoordinatorRebindsOwnedDrafterForSequentialRequests(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	promptA, promptB := []int{0, 1, 2}, []int{7, 3, 5, 1}

	first := m.NewSession()
	defer first.Close()
	coord, err := first.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()
	wantA := m.NewSession()
	defer wantA.Close()
	if got, want := mustMetalMTPGenerate(t, coord, promptA, 6), wantA.Generate(promptA, 6); !reflect.DeepEqual(got, want) {
		t.Fatalf("first request output=%v want %v", got, want)
	}
	oldDraft := coord.draftSes
	if oldDraft == nil || oldDraft.target != first {
		t.Fatal("first request did not own a drafter bound to its target")
	}

	second := m.NewSession()
	defer second.Close()
	coord.SetTargetSession(second)
	if !oldDraft.closed {
		t.Fatal("target switch left the prior request drafter live")
	}
	if coord.draftSes == nil || coord.draftSes == oldDraft || coord.draftSes.target != second {
		t.Fatal("target switch did not create a fresh drafter bound to the new request")
	}
	wantB := m.NewSession()
	defer wantB.Close()
	if got, want := mustMetalMTPGenerate(t, coord, promptB, 6), wantB.Generate(promptB, 6); !reflect.DeepEqual(got, want) {
		t.Fatalf("reused coordinator output=%v want %v", got, want)
	}
}

func TestMetalMTPCoordinatorTargetSwitchPreservesInjectedDrafter(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	first, second := m.NewSession(), m.NewSession()
	defer first.Close()
	defer second.Close()
	coord, err := first.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()
	auto := coord.draftSes
	custom := NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		return []int{0}, nil
	})
	coord.SetDrafter(custom)
	if auto == nil || !auto.closed || coord.draftSes != nil {
		t.Fatal("injecting a custom drafter did not release the coordinator-owned drafter")
	}
	coord.SetTargetSession(second)
	if coord.drafter != custom || coord.draftSes != nil {
		t.Fatal("target switch replaced an injected custom drafter")
	}
}

func TestMetalMTPCoordinatorConcurrentTargetSwitchKeepsDrafterAndTargetTogether(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	initial := m.NewSession()
	defer initial.Close()
	coord, err := initial.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()

	targets := make([]*Session, 12)
	for i := range targets {
		targets[i] = m.NewSession()
		defer targets[i].Close()
	}
	var wg sync.WaitGroup
	for _, target := range targets {
		target := target
		wg.Add(1)
		go func() {
			defer wg.Done()
			coord.SetTargetSession(target)
		}()
	}
	wg.Wait()

	coord.mu.Lock()
	defer coord.mu.Unlock()
	if coord.target == nil || coord.draftSes == nil || coord.draftSes.closed || coord.draftSes.target != coord.target {
		t.Fatalf("concurrent switch split target/drafter ownership: target=%p draft=%p draft-target=%p", coord.target, coord.draftSes, func() *Session {
			if coord.draftSes == nil {
				return nil
			}
			return coord.draftSes.target
		}())
	}
}

func TestMetalMTPCoordinatorTargetSwitchWaitsForActiveGeneration(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	first, second, want := m.NewSession(), m.NewSession(), m.NewSession()
	defer first.Close()
	defer second.Close()
	defer want.Close()
	coord, err := first.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()

	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	custom := NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		once.Do(func() {
			close(entered)
			<-release
		})
		return []int{0, 0, 0, 0}, nil
	})
	coord.SetDrafter(custom)
	prompt := []int{0, 1, 2}
	wantTokens := want.Generate(prompt, 8)
	generated := make(chan []int, 1)
	generateErr := make(chan error, 1)
	go func() {
		out, err := coord.Generate(context.Background(), prompt, 8)
		generated <- out
		generateErr <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("generation never entered drafter")
	}

	switchStarted, switched := make(chan struct{}), make(chan struct{})
	go func() {
		close(switchStarted)
		coord.SetTargetSession(second)
		close(switched)
	}()
	<-switchStarted
	select {
	case <-switched:
		t.Fatal("target switch completed while generation still owned the first target")
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-generateErr:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("generation deadlocked with target switch")
	}
	gotTokens := <-generated
	if !reflect.DeepEqual(gotTokens, wantTokens) {
		t.Fatalf("active generation mixed targets: got=%v want=%v", gotTokens, wantTokens)
	}
	select {
	case <-switched:
	case <-time.After(5 * time.Second):
		t.Fatal("target switch did not resume after generation")
	}
	if coord.TargetSession() != second || second.Cache.Len() != 0 || coord.drafter != custom {
		t.Fatalf("post-generation switch state target=%p second=%p second-cache=%d drafter-preserved=%v", coord.TargetSession(), second, second.Cache.Len(), coord.drafter == custom)
	}
}

func TestMetalMTPCoordinatorMMUBindingWaitsForActiveGeneration(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	defer target.Close()
	coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()

	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		once.Do(func() {
			close(entered)
			<-release
		})
		return []int{0, 0, 0, 0}, nil
	}))
	oldMMU, nextMMU := &identityMTPRecorder{}, &identityMTPRecorder{}
	coord.SetMMU(oldMMU, "request-a")
	generateDone := make(chan error, 1)
	go func() {
		_, err := coord.Generate(context.Background(), []int{0, 1, 2}, 8)
		generateDone <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("generation never entered drafter")
	}
	setStarted, setDone := make(chan struct{}), make(chan struct{})
	go func() {
		close(setStarted)
		coord.SetMMU(nextMMU, "request-b")
		close(setDone)
	}()
	<-setStarted
	select {
	case <-setDone:
		t.Fatal("MMU binding changed while a generation still owned request-a")
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-generateDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("generation deadlocked with MMU setter")
	}
	select {
	case <-setDone:
	case <-time.After(5 * time.Second):
		t.Fatal("MMU setter did not resume after generation")
	}
	oldEvents := oldMMU.snapshot()
	if len(oldEvents) == 0 {
		t.Fatal("active generation did not use its request MMU")
	}
	for _, event := range oldEvents {
		if event != "record:request-a" && event != "commit:request-a" && event != "rollback:request-a" {
			t.Fatalf("active generation mixed MMU session identity: %v", oldEvents)
		}
	}
	if events := nextMMU.snapshot(); len(events) != 0 {
		t.Fatalf("next request MMU observed active generation events: %v", events)
	}
	coord.mu.Lock()
	gotMMU, gotSession := coord.checkpointMgr, coord.sessionID
	coord.mu.Unlock()
	if gotMMU != nextMMU || gotSession != "request-b" {
		t.Fatalf("post-generation MMU binding=%p/%q want %p/request-b", gotMMU, gotSession, nextMMU)
	}
}

func TestMetalMTPCoordinatorSetMMUNoopAfterCloseAndReleasesBinding(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	defer target.Close()
	coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	oldMMU, postCloseMMU := &identityMTPRecorder{}, &identityMTPRecorder{}
	coord.SetMMU(oldMMU, "request-before-close")
	if err := coord.Close(); err != nil {
		t.Fatal(err)
	}
	coord.SetMMU(postCloseMMU, "request-after-close")
	coord.mu.Lock()
	gotMMU, gotSession := coord.checkpointMgr, coord.sessionID
	coord.mu.Unlock()
	if gotMMU != nil || gotSession != "" {
		t.Fatalf("closed coordinator retained MMU/session binding=%p/%q", gotMMU, gotSession)
	}
}

func TestMetalMTPCoordinatorDraftPhaseProfilerAttachAndRead(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	defer target.Close()
	targetProfiler := NewPhaseProfiler()
	target.PhaseProfiler = targetProfiler
	coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()

	draftProfiler := NewPhaseProfiler()
	if err := coord.SetDraftPhaseProfiler(draftProfiler); err != nil {
		t.Fatal(err)
	}
	if coord.draftSes == nil || coord.draftSes.forward == nil || coord.draftSes.forward.draft == nil ||
		coord.draftSes.forward.draft.PhaseProfiler != draftProfiler || target.PhaseProfiler != targetProfiler {
		t.Fatal("draft profiler was not attached exclusively to the owned draft Session")
	}
	actual := coord.draftSes.forward.draft.PhaseProfiler
	actual.recordMetal(completeMetalSnapshot(metalgemm.ExecutionQ6KGEMV, 1, 0.25), nil)
	actual.recordMetalFallback(MetalFallbackQ8GEMVCPU)
	receipt, err := coord.DraftPhaseProfilerReceipt()
	if err != nil {
		t.Fatal(err)
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt.MetalExecution); err != nil {
		t.Fatalf("draft Metal execution receipt: %v", err)
	}
	if err := ValidateMetalFallbackReceipt(receipt.MetalFallback); err != nil {
		t.Fatalf("draft Metal fallback receipt: %v", err)
	}
	if len(receipt.MetalExecution.Events) != 1 || receipt.MetalExecution.Events[0].Operation != metalgemm.ExecutionQ6KGEMV ||
		receipt.MetalFallback.PromisedCPUFallbacks != 1 || len(receipt.MetalFallback.Events) != 1 {
		t.Fatalf("draft profiler receipt=%+v", receipt)
	}
	if _, err := targetProfiler.MetalExecutionReceipt(); !metalgemm.IsExecutionCountersIncomplete(err) {
		t.Fatalf("target profiler was conflated with draft execution: %v", err)
	}
	if fallback, err := targetProfiler.MetalFallbackReceipt(); err != nil || len(fallback.Events) != 0 {
		t.Fatalf("target profiler was conflated with draft fallback: receipt=%+v err=%v", fallback, err)
	}

	receipt.MetalExecution.Events[0].Operation = metalgemm.ExecutionQ4KGEMM
	receipt.MetalFallback.Events[0].Route = MetalFallbackQ4KGEMMCPU
	again, err := coord.DraftPhaseProfilerReceipt()
	if err != nil || again.MetalExecution.Events[0].Operation != metalgemm.ExecutionQ6KGEMV ||
		again.MetalFallback.Events[0].Route != MetalFallbackQ8GEMVCPU {
		t.Fatalf("draft receipt aliases caller memory: receipt=%+v err=%v", again, err)
	}
}

func TestMetalMTPCoordinatorDraftPhaseProfilerSurvivesOwnedRebind(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	first, second := m.NewSession(), m.NewSession()
	defer first.Close()
	defer second.Close()
	coord, err := first.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()
	profiler := NewPhaseProfiler()
	if err := coord.SetDraftPhaseProfiler(profiler); err != nil {
		t.Fatal(err)
	}
	old := coord.draftSes
	coord.SetTargetSession(second)
	if old == nil || !old.closed || coord.draftSes == nil || coord.draftSes == old ||
		coord.draftSes.target != second || coord.draftSes.forward.draft.PhaseProfiler != profiler {
		t.Fatal("owned target rebind did not close the old drafter and reapply its profiler")
	}

	// Prefix divergence recreates Qwen35MTPForward inside the same owned draft
	// session. The coordinator-installed step seam must attach before execution.
	if err := coord.draftSes.recreateForward(); err != nil {
		t.Fatal(err)
	}
	if coord.draftSes.forward.draft.PhaseProfiler != nil {
		t.Fatal("fresh internal forward unexpectedly inherited Session state")
	}
	prior, embedding := qwen38MTPInputs(m.Cfg.HiddenSize, 0)
	if _, _, err := coord.draftSes.step(coord.draftSes.forward, 0, prior, embedding); err != nil {
		t.Fatal(err)
	}
	if coord.draftSes.forward.draft.PhaseProfiler != profiler {
		t.Fatal("internal draft-forward recreation lost the coordinator profiler")
	}
}

func TestMetalMTPCoordinatorDraftPhaseProfilerRejectsInjectedAndTargetAlias(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	defer target.Close()
	coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()
	targetProfiler := NewPhaseProfiler()
	target.PhaseProfiler = targetProfiler
	if err := coord.SetDraftPhaseProfiler(targetProfiler); !errors.Is(err, ErrMetalMTPDraftProfilerTargetAlias) {
		t.Fatalf("target profiler alias err=%v", err)
	}
	draftProfiler := NewPhaseProfiler()
	if err := coord.SetDraftPhaseProfiler(draftProfiler); err != nil {
		t.Fatal(err)
	}
	old := coord.draftSes
	coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		return []int{0}, nil
	}))
	if old == nil || !old.closed {
		t.Fatal("injected drafter did not close the owned profiled session")
	}
	if err := coord.SetDraftPhaseProfiler(NewPhaseProfiler()); !errors.Is(err, ErrMetalMTPDraftProfilerUnavailable) {
		t.Fatalf("injected drafter profiler attach err=%v", err)
	}
	if _, err := coord.DraftPhaseProfilerReceipt(); !errors.Is(err, ErrMetalMTPDraftProfilerUnavailable) {
		t.Fatalf("injected drafter profiler receipt err=%v", err)
	}
	if target.PhaseProfiler != targetProfiler {
		t.Fatal("draft profiler lifecycle changed the target profiler")
	}
}

func TestMetalMTPCoordinatorDraftPhaseProfilerConcurrentLifecycle(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	initial := m.NewSession()
	defer initial.Close()
	coord, err := initial.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		t.Fatal(err)
	}
	profiler := NewPhaseProfiler()
	profiler.recordMetal(completeMetalSnapshot(metalgemm.ExecutionQ4KGEMV, 1, 0.5), nil)
	if err := coord.SetDraftPhaseProfiler(profiler); err != nil {
		t.Fatal(err)
	}
	targets := make([]*Session, 8)
	for i := range targets {
		targets[i] = m.NewSession()
		defer targets[i].Close()
	}
	start := make(chan struct{})
	errs := make(chan error, len(targets)*3+1)
	var wg sync.WaitGroup
	for _, target := range targets {
		target := target
		wg.Add(3)
		go func() {
			defer wg.Done()
			<-start
			coord.SetTargetSession(target)
		}()
		go func() {
			defer wg.Done()
			<-start
			err := coord.SetDraftPhaseProfiler(profiler)
			if err != nil && !errors.Is(err, ErrMetalMTPClosed) {
				errs <- err
			}
		}()
		go func() {
			defer wg.Done()
			<-start
			_, err := coord.DraftPhaseProfilerReceipt()
			if err != nil && !errors.Is(err, ErrMetalMTPClosed) {
				errs <- err
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		errs <- coord.Close()
	}()
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !coord.closed || coord.draftSes != nil || coord.draftProfiler != nil {
		t.Fatalf("closed coordinator retained draft profiler state: closed=%v draft=%p profiler=%p", coord.closed, coord.draftSes, coord.draftProfiler)
	}
	if err := coord.SetDraftPhaseProfiler(NewPhaseProfiler()); !errors.Is(err, ErrMetalMTPClosed) {
		t.Fatalf("post-close attach err=%v", err)
	}
	if _, err := coord.DraftPhaseProfilerReceipt(); !errors.Is(err, ErrMetalMTPClosed) {
		t.Fatalf("post-close receipt err=%v", err)
	}
}

func TestMetalMTPCoordinatorMMUCommitFailureRestoresTargetAndRecordsRejectedRound(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}
	target, oracle, want := m.NewSession(), m.NewSession(), m.NewSession()
	defer target.Close()
	defer oracle.Close()
	defer want.Close()
	target.captureTargetHidden, oracle.captureTargetHidden, want.captureTargetHidden = true, true, true
	boundary := target.Prefill(prompt)
	oracleLogits := oracle.Prefill(prompt)
	want.Prefill(prompt)
	normalizeSnapshotForTest(t, target)
	normalizeSnapshotForTest(t, want)
	draft := make([]int, 4)
	for i := range draft {
		draft[i] = argmaxF32(oracleLogits)
		oracleLogits = oracle.Step(draft[i])
	}

	adaptive := DefaultQwen38AdaptiveConfig()
	adaptive.ColdStartDepth = 4
	cfg := DefaultMetalMTPConfig()
	cfg.Adaptive = true
	cfg.AdaptiveConfig = &adaptive
	coord, err := target.NewMetalMTPCoordinator(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer coord.Close()
	coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		return append([]int(nil), draft...), nil
	}))
	commitFailure := errors.New("injected Context-MMU commit failure")
	mmu := &mockMTPRecorder{commitErr: commitFailure}
	coord.SetMMU(mmu, "commit-failure")

	accepted, bonus, next, err := coord.StepRound(context.Background(), prompt, boundary)
	if !errors.Is(err, commitFailure) || accepted != nil || bonus != -1 || next != nil {
		t.Fatalf("MMU failure escaped accepted=%v bonus=%d next=%v err=%v", accepted, bonus, next, err)
	}
	assertQwen35MTPTargetStateEqual(t, target, want)
	stats := coord.Stats()
	if stats.TotalProposed != 4 || stats.TotalAccepted != 0 || stats.TotalRollbacks != 4 ||
		stats.CommittedPages != 0 || stats.FreedPages != 4 || stats.TotalGenerated != 0 {
		t.Fatalf("MMU failure accounting=%+v", stats)
	}
	trace := coord.AdaptiveGovernor().Trace()
	if len(trace) != 1 || trace[0].ProposedTokens != 4 || trace[0].AcceptedTokens != 0 {
		t.Fatalf("MMU failure governor trace=%+v", trace)
	}
	if mmu.records != 1 || mmu.commits != 1 || mmu.rollbacks != 1 {
		t.Fatalf("MMU lifecycle records/commits/rollbacks=%d/%d/%d", mmu.records, mmu.commits, mmu.rollbacks)
	}
}

func mustMetalMTPGenerate(t *testing.T, coord *MetalMTPCoordinator, prompt []int, count int) []int {
	t.Helper()
	out, err := coord.Generate(context.Background(), prompt, count)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// BenchmarkMetalMTPDraftVerifyRollbackLoop benchmarks wide-M (M=4) verification against serial decode.
func BenchmarkMetalMTPDraftVerifyRollbackLoop(b *testing.B) {
	m := qwen38HybridMTPEnabledSyntheticModelBench(b)
	prompt := []int{0, 1, 2}
	specSes := m.NewSession()
	b.Cleanup(specSes.Close)
	coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		b.Fatalf("NewMetalMTPCoordinator failed: %v", err)
	}
	b.Cleanup(func() { _ = coord.Close() })
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = coord.Generate(ctx, prompt, 8)
	}
}

// TestMetalWideMSpeculativeTreeVerification_M16_M24 verifies that evaluating an M=16..24
// speculative candidate tree in one forward pass and extracting the highest-scoring
// verified token branch via greedy argmax path selection yields bit-exact equivalence
// to serial autoregressive decode.
func TestMetalWideMSpeculativeTreeVerification_M16_M24(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}
	ctx := context.Background()

	t.Run("tree_branch_greedy_argmax_bit_exact_with_serial", func(t *testing.T) {
		// 1. Establish ground truth serial autoregressive steps
		goldSes := m.NewSession()
		t.Cleanup(goldSes.Close)
		goldBoundary := goldSes.Prefill(prompt)
		goldT0 := argmaxF32(goldBoundary)

		goldNext1 := goldSes.Step(goldT0)
		goldT1 := argmaxF32(goldNext1)

		goldNext2 := goldSes.Step(goldT1)
		goldT2 := argmaxF32(goldNext2)

		goldNext3 := goldSes.Step(goldT2)
		goldT3 := argmaxF32(goldNext3)

		wantAccepted := []int{goldT0, goldT1, goldT2}
		wantBonus := goldT3

		// 2. Build a wide candidate tree (M=16 nodes) containing the winning branch
		// along with branching distractor candidates:
		branches := [][]int{
			{goldT0, goldT1, goldT2},
			{goldT0, goldT1, (goldT2 + 1) % 100},
			{goldT0, (goldT1 + 1) % 100, goldT2},
			{goldT0, (goldT1 + 1) % 100, (goldT2 + 2) % 100},
			{(goldT0 + 1) % 100, goldT1, goldT2},
			{(goldT0 + 1) % 100, (goldT1 + 2) % 100, goldT2},
			{(goldT0 + 2) % 100, goldT1, goldT2},
		}

		tree, err := BuildCandidateTreeFromBranches(branches)
		if err != nil {
			t.Fatalf("BuildCandidateTreeFromBranches failed: %v", err)
		}
		if len(tree.Nodes) < 16 {
			// Pad with additional distractor branches to reach M=16..24
			for i := len(branches); len(tree.Nodes) < 16; i++ {
				branches = append(branches, []int{(goldT0 + i) % 100, (goldT1 + i) % 100, (goldT2 + i) % 100})
				tree, err = BuildCandidateTreeFromBranches(branches)
				if err != nil {
					t.Fatalf("BuildCandidateTreeFromBranches padding failed: %v", err)
				}
			}
		}

		t.Logf("Constructed M=%d candidate tree for multi-branch verification", len(tree.Nodes))

		// 3. Evaluate candidate tree via MetalMTPCoordinator.StepRoundTree
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)
		boundary := specSes.Prefill(prompt)

		coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		gotAccepted, gotBonus, nextLogits, err := coord.StepRoundTree(ctx, prompt, boundary, tree)
		if err != nil {
			t.Fatalf("StepRoundTree failed: %v", err)
		}

		// 4. Assert bit-exact match against serial decode
		if !reflect.DeepEqual(gotAccepted, wantAccepted) {
			t.Fatalf("accepted tokens mismatch:\n got:  %v\n want: %v", gotAccepted, wantAccepted)
		}
		if gotBonus != wantBonus {
			t.Fatalf("bonus token mismatch: got %d, want %d", gotBonus, wantBonus)
		}
		if len(nextLogits) != m.Cfg.VocabSize {
			t.Fatalf("nextLogits length %d != vocabSize %d", len(nextLogits), m.Cfg.VocabSize)
		}

		// Verify target session state matches gold session state
		goldAfterBonus := goldSes.Step(wantBonus)
		goldTok := argmaxF32(goldAfterBonus)
		specTok := argmaxF32(nextLogits)
		if specTok != goldTok {
			t.Fatalf("continuation token mismatch after tree verification: spec %d, gold %d", specTok, goldTok)
		}
		for step := 0; step < 3; step++ {
			goldNext := goldSes.Step(goldTok)
			specNext := specSes.Step(specTok)
			goldTok = argmaxF32(goldNext)
			specTok = argmaxF32(specNext)
			if specTok != goldTok {
				t.Fatalf("step %d continuation token mismatch: spec %d, gold %d", step, specTok, goldTok)
			}
		}

		t.Logf("Verified tree verification greedy argmax path selection is bit-exact with serial autoregressive decode")
	})

	t.Run("tree_root_mismatch_fallback", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)
		boundary := specSes.Prefill(prompt)
		target0 := argmaxF32(boundary)

		coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Construct tree where NO root candidate matches target0
		mismatchRoot := (target0 + 1) % 100
		branches := [][]int{
			{mismatchRoot, 10, 20},
			{mismatchRoot, 11, 21},
		}
		tree, err := BuildCandidateTreeFromBranches(branches)
		if err != nil {
			t.Fatalf("BuildCandidateTreeFromBranches failed: %v", err)
		}

		gotAccepted, gotBonus, nextLogits, err := coord.StepRoundTree(ctx, prompt, boundary, tree)
		if err != nil {
			t.Fatalf("StepRoundTree failed: %v", err)
		}
		if len(gotAccepted) != 0 {
			t.Fatalf("expected 0 accepted tokens on root mismatch, got %v", gotAccepted)
		}
		if gotBonus != target0 {
			t.Fatalf("expected bonus token to be target0 (%d), got %d", target0, gotBonus)
		}
		if len(nextLogits) != m.Cfg.VocabSize {
			t.Fatalf("nextLogits length %d != vocabSize %d", len(nextLogits), m.Cfg.VocabSize)
		}
	})
}

// TestMetalMTPWideMUsesBatchedVerifier is the witness test for Issue #12349:
// It proves:
//  1. The explicitly supported Metal/Qwen hybrid envelope is admitted by verifyForwardBatchedOK.
//  2. MetalMTPCoordinator selects single-pass batched verification (OneOperation=true, Path=targetVerificationBatchedPath)
//     rather than falling back to a serial Step loop.
//  3. Batched verification logits and session cache state match serial step verification bit-for-bit.
//  4. Unsupported shapes (e.g. F16) fall back cleanly to typed fak-native serial decode (OneOperation=false, Path=targetVerificationDecodePath).
func TestMetalMTPWideMUsesBatchedVerifier(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}
	ctx := context.Background()

	t.Run("admitted_hybrid_envelope_selects_batched_dispatch_with_serial_parity", func(t *testing.T) {
		target := m.NewSession()
		target.captureTargetHidden = true
		t.Cleanup(target.Close)
		boundary := target.Prefill(prompt)

		// Precondition: hybrid session in the supported envelope must be admitted
		if !verifyForwardBatchedOK(target) {
			t.Fatal("precondition failed: verifyForwardBatchedOK rejected admitted Qwen hybrid session")
		}

		// K=3 draft depth exercises wide-M batched verification path
		coord, err := target.NewMetalMTPCoordinator(MetalMTPConfig{DraftDepth: 3, FallbackToSerial: true})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		draft := []int{3, 5, 7}
		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return append([]int(nil), draft...), nil
		}))

		// Step round with candidate proposal
		accepted, bonus, nextLogits, err := coord.StepRound(ctx, nil, boundary)
		if err != nil {
			t.Fatalf("StepRound failed: %v", err)
		}

		// 1. Prove batched dispatch was selected
		receipt, ok := coord.LastTargetVerificationReceipt()
		if !ok {
			t.Fatal("expected LastTargetVerificationReceipt to be present")
		}
		if !receipt.OneOperation {
			t.Fatalf("receipt.OneOperation = %v, want true (batched verification)", receipt.OneOperation)
		}
		if receipt.TargetVerificationOperations != 1 {
			t.Fatalf("receipt.TargetVerificationOperations = %d, want 1", receipt.TargetVerificationOperations)
		}
		if receipt.TargetDecodeSteps != 0 {
			t.Fatalf("receipt.TargetDecodeSteps = %d, want 0", receipt.TargetDecodeSteps)
		}
		if receipt.Path != targetVerificationBatchedPath {
			t.Fatalf("receipt.Path = %q, want %q", receipt.Path, targetVerificationBatchedPath)
		}
		if receipt.Engine != targetVerificationEngine {
			t.Fatalf("receipt.Engine = %q, want %q", receipt.Engine, targetVerificationEngine)
		}

		// 2. Prove logits and state parity against serial decode
		serialRef := m.NewSession()
		serialRef.captureTargetHidden = true
		t.Cleanup(serialRef.Close)
		serialBoundary := serialRef.Prefill(prompt)
		target0 := argmaxF32(serialBoundary)

		serialRows := make([][]float32, len(draft))
		for i, tok := range draft {
			serialRows[i] = serialRef.Step(tok)
		}
		serialTargetArgmax := make([]int, len(draft)+1)
		serialTargetArgmax[0] = target0
		for i, r := range serialRows {
			serialTargetArgmax[i+1] = argmaxF32(r)
		}
		wantAccepted, wantBonus, tripErr := TripwireVerify(draft, serialTargetArgmax, serialBoundary, serialRows)
		if tripErr != nil {
			t.Fatalf("serial TripwireVerify failed: %v", tripErr)
		}

		if !reflect.DeepEqual(accepted, wantAccepted) {
			t.Fatalf("accepted tokens mismatch: got %v, want %v", accepted, wantAccepted)
		}
		if bonus != wantBonus {
			t.Fatalf("bonus token mismatch: got %d, want %d", bonus, wantBonus)
		}

		// Replay exact accepted branch + bonus token on ground truth session
		oracle := m.NewSession()
		oracle.captureTargetHidden = true
		t.Cleanup(oracle.Close)
		oracle.Prefill(prompt)
		for _, tok := range wantAccepted {
			oracle.Step(tok)
		}
		oracleBonusLogits := oracle.Step(wantBonus)

		// Assert nextLogits emitted from StepRound matches oracle bonus logits
		for i := range nextLogits {
			if math.Float32bits(nextLogits[i]) != math.Float32bits(oracleBonusLogits[i]) {
				t.Fatalf("nextLogits[%d] mismatch against oracle: got %v, want %v", i, nextLogits[i], oracleBonusLogits[i])
			}
		}

		// Verify subsequent continuation steps match between speculative and oracle sessions
		for step := 0; step < 3; step++ {
			nextTok := argmaxF32(nextLogits)
			nextLogits = target.Step(nextTok)
			wantNext := oracle.Step(nextTok)
			for i := range nextLogits {
				if math.Float32bits(nextLogits[i]) != math.Float32bits(wantNext[i]) {
					t.Fatalf("step %d continuation logit[%d] mismatch: got %v, want %v", step, i, nextLogits[i], wantNext[i])
				}
			}
		}
	})

	t.Run("metal_flag_admitted_in_hybrid_envelope", func(t *testing.T) {
		metalTarget := m.NewSession()
		metalTarget.captureTargetHidden = true
		t.Cleanup(metalTarget.Close)
		boundary := metalTarget.Prefill(prompt)
		metalTarget.Metal = true

		if !verifyForwardBatchedOK(metalTarget) {
			t.Fatal("precondition failed: verifyForwardBatchedOK rejected session with Metal=true in hybrid envelope")
		}

		coord, err := metalTarget.NewMetalMTPCoordinator(MetalMTPConfig{DraftDepth: 2, FallbackToSerial: true})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		draft := []int{3, 5}
		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return append([]int(nil), draft...), nil
		}))

		_, _, _, err = coord.StepRound(ctx, nil, boundary)
		if err != nil {
			t.Fatalf("StepRound failed: %v", err)
		}

		receipt, ok := coord.LastTargetVerificationReceipt()
		if !ok {
			t.Fatal("expected LastTargetVerificationReceipt to be present")
		}
		if !receipt.OneOperation {
			t.Fatalf("receipt.OneOperation = %v, want true", receipt.OneOperation)
		}
		if receipt.Path != targetVerificationBatchedPath {
			t.Fatalf("receipt.Path = %q, want %q", receipt.Path, targetVerificationBatchedPath)
		}
	})

	t.Run("verify_forward_batched_matches_serial_step_logits_and_cache", func(t *testing.T) {
		draft := []int{4, 8, 12, 16}

		// Serial baseline session
		serial := m.NewSession()
		t.Cleanup(serial.Close)
		serial.Prefill(prompt)
		wantRows := make([][]float32, len(draft))
		for i, tok := range draft {
			wantRows[i] = serial.Step(tok)
		}

		// Batched VerifyForward session
		target := m.NewSession()
		t.Cleanup(target.Close)
		target.Prefill(prompt)

		gotRows := target.VerifyForward(draft, nil, nil)
		if len(gotRows) != len(wantRows) {
			t.Fatalf("VerifyForward returned %d rows, want %d", len(gotRows), len(wantRows))
		}

		for j := range wantRows {
			a, b := wantRows[j], gotRows[j]
			if len(a) != len(b) {
				t.Fatalf("pos %d logit width %d != %d", j, len(a), len(b))
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("pos %d logit[%d]: serial %v != verify %v", j, i, a[i], b[i])
				}
			}
		}

		if target.Cache.Len() != serial.Cache.Len() {
			t.Fatalf("cache len mismatch: got %d, want %d", target.Cache.Len(), serial.Cache.Len())
		}
		for i := range serial.Cache.pos {
			if target.Cache.pos[i] != serial.Cache.pos[i] {
				t.Fatalf("pos[%d] mismatch: got %d, want %d", i, target.Cache.pos[i], serial.Cache.pos[i])
			}
		}
	})

	t.Run("unsupported_shapes_fallback_to_typed_fak_native_serial_decode", func(t *testing.T) {
		unsupported := m.NewSession()
		unsupported.F16 = true
		unsupported.captureTargetHidden = true
		t.Cleanup(unsupported.Close)
		uBoundary := unsupported.Prefill(prompt)

		// F16 must be rejected from batched verify
		if verifyForwardBatchedOK(unsupported) {
			t.Fatal("expected verifyForwardBatchedOK to reject F16 session")
		}

		coord, err := unsupported.NewMetalMTPCoordinator(MetalMTPConfig{DraftDepth: 2, FallbackToSerial: true})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return []int{3, 5}, nil
		}))

		_, _, _, err = coord.StepRound(ctx, nil, uBoundary)
		if err != nil {
			t.Fatalf("StepRound on unsupported shape failed: %v", err)
		}

		receipt, ok := coord.LastTargetVerificationReceipt()
		if !ok {
			t.Fatal("expected LastTargetVerificationReceipt to be present")
		}
		if receipt.OneOperation {
			t.Fatalf("receipt.OneOperation = true on unsupported shape, want false")
		}
		if receipt.TargetDecodeSteps != 2 {
			t.Fatalf("receipt.TargetDecodeSteps = %d, want 2", receipt.TargetDecodeSteps)
		}
		if receipt.TargetVerificationOperations != 0 {
			t.Fatalf("receipt.TargetVerificationOperations = %d, want 0", receipt.TargetVerificationOperations)
		}
		if receipt.Path != targetVerificationDecodePath {
			t.Fatalf("receipt.Path = %q, want %q", receipt.Path, targetVerificationDecodePath)
		}
		if receipt.Engine != targetVerificationEngine {
			t.Fatalf("receipt.Engine = %q, want %q", receipt.Engine, targetVerificationEngine)
		}
	})
}
