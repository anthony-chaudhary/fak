//go:build darwin && arm64 && cgo

package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestMetalMTPCoordinatorDispatchesProductionP4Panel(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPCoordinatorP4Fixture(t)
	ctx := context.Background()

	for acceptedCount := 0; acceptedCount <= qwen35MetalMTPVerifyPanelTokens; acceptedCount++ {
		acceptedCount := acceptedCount
		t.Run(string(rune('0'+acceptedCount)), func(t *testing.T) {
			target, boundary := newPreparedMetalMTPPanelSession(t, m)
			oracle, oracleBoundary := newPreparedMetalMTPPanelSession(t, m)
			want, _ := newPreparedMetalMTPPanelSession(t, m)
			defer target.Close()
			defer oracle.Close()
			defer want.Close()

			greedy := make([]int, qwen35MetalMTPVerifyPanelTokens+1)
			logits := oracleBoundary
			for i := range greedy {
				greedy[i] = argmaxF32(logits)
				logits = oracle.Step(greedy[i])
			}
			draft := append([]int(nil), greedy[:qwen35MetalMTPVerifyPanelTokens]...)
			if acceptedCount < len(draft) {
				draft[acceptedCount] = (draft[acceptedCount] + 1) % m.Cfg.VocabSize
			}

			coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
			if err != nil {
				t.Fatal(err)
			}
			defer coord.Close()
			coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
				return append([]int(nil), draft...), nil
			}))
			mmu := &mockMTPRecorder{}
			coord.SetMMU(mmu, "metal-p4-production")

			accepted, bonus, next, err := coord.StepRound(ctx, nil, boundary)
			if err != nil {
				t.Fatalf("StepRound P4: %v", err)
			}
			if !slices.Equal(accepted, greedy[:acceptedCount]) || bonus != greedy[acceptedCount] {
				t.Fatalf("emitted accepted=%v bonus=%d, want accepted=%v bonus=%d", accepted, bonus, greedy[:acceptedCount], greedy[acceptedCount])
			}

			var wantNext []float32
			for _, token := range greedy[:acceptedCount+1] {
				wantNext = want.Step(token)
			}
			assertCosineAtLeast(t, "P4 StepRound next logits", wantNext, next, Qwen35GDNParityCosineMin)
			if argmaxF32(next) != argmaxF32(wantNext) {
				t.Fatalf("next argmax=%d want %d", argmaxF32(next), argmaxF32(wantNext))
			}
			if acceptedCount < qwen35MetalMTPVerifyPanelTokens {
				compareExactQwen38SessionState(t, acceptedCount, want, target)
				if !metalMTPHiddenStateEqual(target, want) {
					t.Fatalf("r=%d target-hidden state differs from serial", acceptedCount)
				}
			} else {
				assertRetainedMetalMTPStateEquivalent(t, want, target)
				panelBase := target.Cache.Len() - qwen35MetalMTPVerifyPanelTokens - 1
				if len(target.targetHidden) != panelBase+qwen35MetalMTPVerifyPanelTokens ||
					!slices.Equal(target.targetHiddenTokens[panelBase:], draft) {
					t.Fatalf("full acceptance did not retain P4 target-hidden panel: hidden=%d tokens=%v", len(target.targetHidden), target.targetHiddenTokens)
				}
			}

			receipt, ok := coord.LastTargetVerificationReceipt()
			if !ok {
				t.Fatal("StepRound omitted target verification receipt")
			}
			if receipt.Path != Qwen35MetalMTPVerifyPanelPath || !receipt.OneOperation ||
				receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 {
				t.Fatalf("StepRound did not select one-operation Metal P4 panel: %+v", receipt)
			}
			if receipt.AcceptedTokens != acceptedCount || receipt.RejectedTokens != len(draft)-acceptedCount {
				t.Fatalf("receipt disposition accepted/rejected=%d/%d, want %d/%d", receipt.AcceptedTokens, receipt.RejectedTokens, acceptedCount, len(draft)-acceptedCount)
			}
			panel := receipt.Panel
			wantLinear := linearQwen35Layers(m.Cfg)
			if panel == nil || !panel.Committed || !panel.CompletedWait || panel.Q6KDownProjectionOperations != m.Cfg.NumLayers || panel.Q6KHeadOperations != 1 ||
				panel.CheckpointLayers != wantLinear || len(panel.GDNCheckpointLayers) != wantLinear ||
				len(panel.GDNCheckpointLineageSHA256) != wantLinear || len(panel.GDNCheckpointBindingSHA256) != 64 ||
				len(panel.StateSHA256) != 64 || len(panel.TransactionSHA256) != 64 {
				t.Fatalf("production P4 panel operation/state receipt=%+v", panel)
			}
			for i, lineageIdentity := range panel.GDNCheckpointLineageSHA256 {
				if len(lineageIdentity) != 64 {
					t.Fatalf("r=%d GDN checkpoint layer=%d lineage=%q", acceptedCount, panel.GDNCheckpointLayers[i], lineageIdentity)
				}
			}
			if want := qwen35MetalMTPTransactionDigest(panel.StateSHA256, panel.GDNCheckpointBindingSHA256); panel.TransactionSHA256 != want {
				t.Fatalf("r=%d transaction digest=%q want %q", acceptedCount, panel.TransactionSHA256, want)
			}
			receiptAgain, ok := coord.LastTargetVerificationReceipt()
			if !ok || !reflect.DeepEqual(receiptAgain, receipt) {
				t.Fatalf("r=%d published receipt changed without a transaction: first=%+v again=%+v", acceptedCount, receipt, receiptAgain)
			}
			if !receipt.Accounting.Setup.Measured || !receipt.Accounting.TargetVerification.Measured ||
				!receipt.Accounting.Rollback.Measured || !receipt.Accounting.Synchronization.Measured {
				t.Fatalf("receipt omitted transaction accounting: %+v", receipt.Accounting)
			}
			stats := coord.Stats()
			if stats.TotalProposed != len(draft) || stats.TotalAccepted != acceptedCount ||
				stats.TotalRollbacks != len(draft)-acceptedCount || stats.TotalGenerated != acceptedCount+1 ||
				stats.CommittedPages != acceptedCount || stats.FreedPages != len(draft)-acceptedCount {
				t.Fatalf("coordinator accounting for r=%d: %+v", acceptedCount, stats)
			}
			if mmu.records != 1 || mmu.commits != 1 || mmu.rollbacks != 0 {
				t.Fatalf("MMU accounting records/commits/rollbacks=%d/%d/%d", mmu.records, mmu.commits, mmu.rollbacks)
			}
		})
	}
}

func TestMetalMTPPanelTransactionDigestSeparatesCheckpointLineageFromExternalGDNOracle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPCoordinatorP4Fixture(t)
	left, _ := newPreparedMetalMTPPanelSession(t, m)
	right, _ := newPreparedMetalMTPPanelSession(t, m)
	defer left.Close()
	defer right.Close()
	draft := []int{3, 5, 7, 11}
	leftResult, leftAccepted, leftErr := left.Qwen35MetalMTPVerifyPanel(draft)
	rightResult, rightAccepted, rightErr := right.Qwen35MetalMTPVerifyPanel(draft)
	if leftErr != nil || rightErr != nil || !leftAccepted || !rightAccepted {
		t.Fatalf("deterministic panels accepted=%v/%v errors=%v/%v", leftAccepted, rightAccepted, leftErr, rightErr)
	}
	defer leftResult.Checkpoint.Close()
	defer rightResult.Checkpoint.Close()
	lr, rr := leftResult.Receipt, rightResult.Receipt
	if lr.StateSHA256 != rr.StateSHA256 || !slices.Equal(lr.GDNCheckpointLayers, rr.GDNCheckpointLayers) {
		t.Fatalf("identical transactions produced different value/layer state: left=%+v right=%+v", lr, rr)
	}
	if lr.GDNCheckpointBindingSHA256 == rr.GDNCheckpointBindingSHA256 || lr.TransactionSHA256 == rr.TransactionSHA256 ||
		slices.Equal(lr.GDNCheckpointLineageSHA256, rr.GDNCheckpointLineageSHA256) {
		t.Fatalf("distinct native checkpoint owners shared an owner-bound transaction identity: left=%+v right=%+v", lr, rr)
	}
	for _, receipt := range []Qwen35MetalMTPVerifyPanelReceipt{lr, rr} {
		if len(receipt.GDNCheckpointLineageSHA256) != len(receipt.GDNCheckpointLayers) ||
			receipt.TransactionSHA256 != qwen35MetalMTPTransactionDigest(receipt.StateSHA256, receipt.GDNCheckpointBindingSHA256) {
			t.Fatalf("checkpoint lineage was not transitively bound into transaction: %+v", receipt)
		}
		for _, identity := range receipt.GDNCheckpointLineageSHA256 {
			if len(identity) != 64 {
				t.Fatalf("checkpoint lineage identity=%q in receipt %+v", identity, receipt)
			}
		}
	}
	if lr.HostStateReadbacks != 0 || rr.HostStateReadbacks != 0 || lr.IntermediateReadbacks != 0 || rr.IntermediateReadbacks != 0 {
		t.Fatalf("production receipt added GDN/intermediate readback: left=%+v right=%+v", lr, rr)
	}
	leftOwner, leftOK := leftResult.Checkpoint.(*metalQwen35MTPCheckpoint)
	rightOwner, rightOK := rightResult.Checkpoint.(*metalQwen35MTPCheckpoint)
	if !leftOK || !rightOK || len(leftOwner.checkpoints) != len(lr.GDNCheckpointLineageSHA256) || len(rightOwner.checkpoints) != len(rr.GDNCheckpointLineageSHA256) {
		t.Fatalf("checkpoint owners do not cover receipt layers: left=%T/%d right=%T/%d", leftResult.Checkpoint, len(leftOwner.checkpoints), rightResult.Checkpoint, len(rightOwner.checkpoints))
	}
	leftNative := append([]*metalgemm.GDNGraphCheckpoint(nil), leftOwner.checkpoints...)
	rightNative := append([]*metalgemm.GDNGraphCheckpoint(nil), rightOwner.checkpoints...)
	for i, checkpoint := range leftNative {
		identity, err := checkpoint.StateIdentity()
		if err != nil || identity != lr.GDNCheckpointLineageSHA256[i] {
			t.Fatalf("left layer=%d checkpoint lineage=%q receipt=%q err=%v", lr.GDNCheckpointLayers[i], identity, lr.GDNCheckpointLineageSHA256[i], err)
		}
		identityAgain, err := checkpoint.StateIdentity()
		if err != nil || identityAgain != identity {
			t.Fatalf("left layer=%d checkpoint identity changed: first=%q again=%q err=%v", lr.GDNCheckpointLayers[i], identity, identityAgain, err)
		}
	}
	if got := leftOwner.Qwen35MetalMTPPanelReceipt(); !reflect.DeepEqual(got, lr) {
		t.Fatalf("checkpoint receipt provider changed checkpoint lineage binding: got=%+v want=%+v", got, lr)
	}
	// Release the device-private checkpoint lease before the test-only oracle
	// snapshots. Production formed both receipt hashes without these readbacks.
	leftResult.Checkpoint.Close()
	rightResult.Checkpoint.Close()
	for _, checkpoint := range append(leftNative, rightNative...) {
		if identity, err := checkpoint.StateIdentity(); err == nil || identity != "" {
			t.Fatalf("closed checkpoint exposed stale identity=%q err=%v", identity, err)
		}
	}
	leftBackend := left.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
	rightBackend := right.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
	for _, layer := range lr.GDNCheckpointLayers {
		leftConv, leftRecurrent, err := leftBackend.SnapshotQwen35GDNAuxState(left.qwen35HAL.sequenceLayers[layer])
		if err != nil {
			t.Fatal(err)
		}
		rightConv, rightRecurrent, err := rightBackend.SnapshotQwen35GDNAuxState(right.qwen35HAL.sequenceLayers[layer])
		if err != nil {
			t.Fatal(err)
		}
		assertFloat32BitsEqual(t, fmt.Sprintf("external GDN conv layer %d", layer), leftConv, rightConv)
		assertFloat32BitsEqual(t, fmt.Sprintf("external GDN recurrent layer %d", layer), leftRecurrent, rightRecurrent)
	}
}

func assertRetainedMetalMTPStateEquivalent(t *testing.T, want, got *Session) {
	t.Helper()
	if want.Cache.Len() != got.Cache.Len() || !slices.Equal(want.Cache.pos, got.Cache.pos) ||
		!slices.Equal(want.Cache.lineage.ids, got.Cache.lineage.ids) || want.Cache.lineage.fault != got.Cache.lineage.fault {
		t.Fatal("retained P4 cache position or token lineage differs from serial")
	}
	for layer := 0; layer < want.M.Cfg.NumLayers; layer++ {
		if want.M.Cfg.isLinearAttnLayer(layer) {
			wantSnapshotter := want.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
			gotSnapshotter := got.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
			wantConv, wantRecurrent, err := wantSnapshotter.SnapshotQwen35GDNAuxState(want.qwen35HAL.sequenceLayers[layer])
			if err != nil {
				t.Fatal(err)
			}
			gotConv, gotRecurrent, err := gotSnapshotter.SnapshotQwen35GDNAuxState(got.qwen35HAL.sequenceLayers[layer])
			if err != nil {
				t.Fatal(err)
			}
			assertCosineAtLeast(t, fmt.Sprintf("retained P4 layer %d conv", layer), wantConv, gotConv, Qwen35GDNParityCosineMin)
			assertCosineAtLeast(t, fmt.Sprintf("retained P4 layer %d recurrent", layer), wantRecurrent, gotRecurrent, Qwen35GDNParityCosineMin)
			continue
		}
		for _, state := range []struct {
			name      string
			want, got []float32
		}{
			{name: "K", want: want.Cache.K[layer], got: got.Cache.K[layer]},
			{name: "Kraw", want: want.Cache.Kraw[layer], got: got.Cache.Kraw[layer]},
			{name: "V", want: want.Cache.V[layer], got: got.Cache.V[layer]},
		} {
			assertCosineAtLeast(t, fmt.Sprintf("retained P4 layer %d %s", layer, state.name), state.want, state.got, Qwen35GDNParityCosineMin)
		}
	}
}

func TestMetalMTPCoordinatorP4FailureAndTripwireRestoreWithoutEscape(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPCoordinatorP4Fixture(t)
	draft := []int{29, 31, 37, 41}

	t.Run("post-submit failure", func(t *testing.T) {
		target, boundary := newPreparedMetalMTPPanelSession(t, m)
		defer target.Close()
		before := exactQwen38SessionDigest(t, target)
		beforeHidden := cloneTargetHidden(target.targetHidden)
		beforeTokens := append([]int(nil), target.targetHiddenTokens...)
		backend := target.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
		backend.injectMTPPanelPostSubmitFailure = true

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
		mmu := &mockMTPRecorder{}
		coord.SetMMU(mmu, "metal-p4-failure")

		accepted, bonus, next, err := coord.StepRound(context.Background(), nil, boundary)
		if err == nil || accepted != nil || bonus != -1 || next != nil {
			t.Fatalf("failed panel escaped accepted=%v bonus=%d next=%v err=%v", accepted, bonus, next, err)
		}
		assertMetalMTPCoordinatorRestored(t, target, before, beforeHidden, beforeTokens)
		assertFailedMetalMTPReceipt(t, coord)
		if stats := coord.Stats(); stats.TotalGenerated != 0 || stats.TotalProposed != 4 ||
			stats.TotalAccepted != 0 || stats.TotalRollbacks != 4 || stats.FreedPages != 4 {
			t.Fatalf("failed panel generated speculative output: %+v", stats)
		}
		trace := coord.AdaptiveGovernor().Trace()
		if len(trace) != 1 || trace[0].ProposedTokens != 4 || trace[0].AcceptedTokens != 0 {
			t.Fatalf("failed panel governor observation=%+v", trace)
		}
		if mmu.records != 1 || mmu.commits != 0 || mmu.rollbacks != 1 {
			t.Fatalf("failure MMU accounting records/commits/rollbacks=%d/%d/%d", mmu.records, mmu.commits, mmu.rollbacks)
		}
	})

	t.Run("Context-MMU commit failure", func(t *testing.T) {
		target, boundary := newPreparedMetalMTPPanelSession(t, m)
		oracle, oracleBoundary := newPreparedMetalMTPPanelSession(t, m)
		defer target.Close()
		defer oracle.Close()
		before := exactQwen38SessionDigest(t, target)
		beforeHidden := cloneTargetHidden(target.targetHidden)
		beforeTokens := append([]int(nil), target.targetHiddenTokens...)
		greedy := make([]int, qwen35MetalMTPVerifyPanelTokens)
		logits := oracleBoundary
		for i := range greedy {
			greedy[i] = argmaxF32(logits)
			logits = oracle.Step(greedy[i])
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
			return append([]int(nil), greedy...), nil
		}))
		commitFailure := errors.New("injected production Context-MMU commit failure")
		mmu := &mockMTPRecorder{commitErr: commitFailure}
		coord.SetMMU(mmu, "metal-p4-commit-failure")

		accepted, bonus, next, err := coord.StepRound(context.Background(), nil, boundary)
		if !errors.Is(err, commitFailure) || accepted != nil || bonus != -1 || next != nil {
			t.Fatalf("MMU failure escaped accepted=%v bonus=%d next=%v err=%v", accepted, bonus, next, err)
		}
		assertMetalMTPCoordinatorRestored(t, target, before, beforeHidden, beforeTokens)
		receipt, ok := coord.LastTargetVerificationReceipt()
		if !ok || receipt.Panel == nil || len(receipt.Panel.TransactionSHA256) != 64 {
			t.Fatalf("MMU failure omitted live panel transaction receipt: %+v", receipt)
		}
		stats := coord.Stats()
		if stats.TotalProposed != 4 || stats.TotalAccepted != 0 || stats.TotalRollbacks != 4 || stats.TotalGenerated != 0 {
			t.Fatalf("MMU failure stats=%+v", stats)
		}
		trace := coord.AdaptiveGovernor().Trace()
		if len(trace) != 1 || trace[0].ProposedTokens != 4 || trace[0].AcceptedTokens != 0 {
			t.Fatalf("MMU failure governor observation=%+v", trace)
		}
		if mmu.records != 1 || mmu.commits != 1 || mmu.rollbacks != 1 {
			t.Fatalf("MMU failure lifecycle records/commits/rollbacks=%d/%d/%d", mmu.records, mmu.commits, mmu.rollbacks)
		}
	})

	t.Run("non-finite tripwire", func(t *testing.T) {
		realPanel := qwen35MTPMetalP4Verify
		qwen35MTPMetalP4Verify = func(s *Session, ids []int) ([][]float32, Qwen35MetalMTPCheckpoint, TargetVerificationReceipt, bool, error) {
			rows, checkpoint, receipt, accepted, err := realPanel(s, ids)
			if accepted && err == nil {
				rows[0][0] = float32(math.NaN())
			}
			return rows, checkpoint, receipt, accepted, err
		}
		t.Cleanup(func() { qwen35MTPMetalP4Verify = realPanel })

		target, boundary := newPreparedMetalMTPPanelSession(t, m)
		defer target.Close()
		before := exactQwen38SessionDigest(t, target)
		beforeHidden := cloneTargetHidden(target.targetHidden)
		beforeTokens := append([]int(nil), target.targetHiddenTokens...)
		coord, err := target.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
		if err != nil {
			t.Fatal(err)
		}
		defer coord.Close()
		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return append([]int(nil), draft...), nil
		}))

		accepted, bonus, next, err := coord.StepRound(context.Background(), nil, boundary)
		if !errors.Is(err, ErrMetalMTPNonFiniteLogits) || accepted != nil || bonus != -1 || next != nil {
			t.Fatalf("tripwire escaped accepted=%v bonus=%d next=%v err=%v", accepted, bonus, next, err)
		}
		assertMetalMTPCoordinatorRestored(t, target, before, beforeHidden, beforeTokens)
		receipt, ok := coord.LastTargetVerificationReceipt()
		if !ok || receipt.Path != Qwen35MetalMTPVerifyPanelPath || receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 {
			t.Fatalf("tripwire receipt=%+v present=%v", receipt, ok)
		}
		if stats := coord.Stats(); !stats.TripwireTripped || stats.TotalGenerated != 0 {
			t.Fatalf("tripwire stats=%+v", stats)
		}
	})
}

// newMetalMTPCoordinatorP4Fixture matches the real Q4_K_M target inventory:
// dense down projections and the output head stay resident Q6_K while the
// remaining majority projections use Q4_K and the reordered minority uses Q8.
func newMetalMTPCoordinatorP4Fixture(t *testing.T) *Model {
	t.Helper()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajorityExceptDown(t, m, cfg)
	fillDownProjQ6KResident(t, m, cfg)
	m.kqw["lm_head.weight"] = randomQ6KTensor(cfg.VocabSize, cfg.HiddenSize, 3804)
	return m
}

func assertMetalMTPCoordinatorRestored(t *testing.T, target *Session, before [32]byte, beforeHidden [][]float32, beforeTokens []int) {
	t.Helper()
	if after := exactQwen38SessionDigest(t, target); after != before ||
		!reflect.DeepEqual(target.targetHidden, beforeHidden) || !reflect.DeepEqual(target.targetHiddenTokens, beforeTokens) {
		t.Fatal("failed P4 round did not restore GDN, KV, lineage, and target-hidden state")
	}
}

func assertFailedMetalMTPReceipt(t *testing.T, coord *MetalMTPCoordinator) {
	t.Helper()
	receipt, ok := coord.LastTargetVerificationReceipt()
	if !ok || receipt.Path != Qwen35MetalMTPVerifyPanelPath || receipt.OneOperation ||
		receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 ||
		receipt.AcceptedTokens != 0 || receipt.RejectedTokens != qwen35MetalMTPVerifyPanelTokens {
		t.Fatalf("failed panel receipt=%+v present=%v", receipt, ok)
	}
}
