package model

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// mtpRawHiddenPartialBackend keeps the transaction/state behavior of the
// existing injected device backend and independently supplies the exact raw
// residual rows produced by its serial oracle.
type mtpRawHiddenPartialBackend struct {
	*partialCommitTestBackend
	nonfiniteRawRow int
}

func (*mtpRawHiddenPartialBackend) Qwen35SequenceRawHiddenPath() string {
	return compute.Qwen35SequenceRawHiddenPath
}

func (b *mtpRawHiddenPartialBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	result, err := b.partialCommitTestBackend.Qwen35SequencePrefill(req)
	if err != nil || !req.CaptureRawHidden {
		return result, err
	}
	rows := make([]float32, 0, len(req.TokenIDs)*req.Hidden)
	for i := range req.TokenIDs {
		row, err := b.reference.TargetHiddenAt(req.StartPos + i)
		if err != nil {
			return compute.Qwen35SequencePrefillResult{}, fmt.Errorf("test raw hidden row %d: %w", i, err)
		}
		rows = append(rows, row...)
	}
	if b.nonfiniteRawRow >= 0 && b.nonfiniteRawRow < len(req.TokenIDs) {
		rows[b.nonfiniteRawRow*req.Hidden] = float32(math.NaN())
	}
	result.RawHiddenRows = compute.NewF32(b, []int{len(req.TokenIDs), req.Hidden}, rows)
	return result, nil
}

func newMTPRawHiddenPartialSession(t *testing.T, m *Model) (*Session, *mtpRawHiddenPartialBackend) {
	t.Helper()
	base := &partialCommitTestBackend{devicePanelTestBackend: newDevicePanelTestBackend(m)}
	backend := &mtpRawHiddenPartialBackend{partialCommitTestBackend: base, nonfiniteRawRow: -1}
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	base.target = target
	base.devicePanelTestBackend.target = target
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	return target, backend
}

func TestQwen35VulkanMTPRawHiddenMalformedPanelIsAtomic(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	target, backend := newMTPRawHiddenPartialSession(t, m)
	backend.nonfiniteRawRow = 1

	defer func() {
		if recovered := recover(); recovered == nil {
			t.Fatal("non-finite second raw-hidden row did not fail the device session")
		}
		target.targetHiddenMu.RLock()
		defer target.targetHiddenMu.RUnlock()
		if len(target.targetHidden) != 0 || len(target.targetHiddenTokens) != 0 {
			t.Fatalf("malformed panel exposed a valid prefix: hidden=%d tokens=%d", len(target.targetHidden), len(target.targetHiddenTokens))
		}
	}()
	target.Prefill([]int{3, 7, 11})
}

func assertTargetHiddenHistoryMatches(t *testing.T, got, want *Session, tokens []int) {
	t.Helper()
	for pos := range tokens {
		gotRow, err := got.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("device target hidden at %d: %v", pos, err)
		}
		wantRow, err := want.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("oracle target hidden at %d: %v", pos, err)
		}
		assertFloat32BitsEqual(t, fmt.Sprintf("target hidden row %d", pos), gotRow, wantRow)
	}
	if _, err := got.TargetHiddenAt(len(tokens)); err == nil {
		t.Fatalf("uncommitted target hidden row %d remained visible", len(tokens))
	}
	if _, err := got.VerifyTokenLineage(tokens); err != nil {
		t.Fatalf("committed token lineage: %v", err)
	}
}

func TestQwen35VulkanMTPRawHiddenCommitHistory(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5}
	draft := []int{17, 19, 23, 2}

	for _, accepted := range []int{len(draft), 2, 0} {
		t.Run(fmt.Sprintf("accepted_%d", accepted), func(t *testing.T) {
			target, backend := newMTPRawHiddenPartialSession(t, m)
			boundary := target.Prefill(prefix)
			if len(backend.requests) != 1 || !backend.requests[0].CaptureRawHidden {
				t.Fatalf("prompt raw-hidden request count/flag=%d/%t, want 1/true", len(backend.requests), len(backend.requests) == 1 && backend.requests[0].CaptureRawHidden)
			}

			tx, err := beginQwen35MTPTargetTransaction(target, boundary)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := tx.Verify(draft); err != nil {
				t.Fatal(err)
			}
			if len(backend.requests) != 2 || !backend.requests[1].CaptureRawHidden {
				t.Fatalf("verification raw-hidden request count/flag=%d/%t, want 2/true", len(backend.requests), len(backend.requests) == 2 && backend.requests[1].CaptureRawHidden)
			}
			if _, err := tx.Commit(accepted); err != nil {
				t.Fatal(err)
			}

			committed := append(append([]int(nil), prefix...), draft[:accepted]...)
			oracle := m.NewSession()
			oracle.captureTargetHidden = true
			oracle.Prefill(committed)
			t.Cleanup(oracle.Close)
			assertTargetHiddenHistoryMatches(t, target, oracle, committed)

			continuation := 29
			gotLogits := target.Step(continuation)
			wantLogits := oracle.Step(continuation)
			assertFloat32BitsEqual(t, "continuation logits", gotLogits, wantLogits)
			committed = append(committed, continuation)
			assertTargetHiddenHistoryMatches(t, target, oracle, committed)
		})
	}
}

func TestQwen35MTPVulkanTargetConstructorFailureIsAtomic(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	base := &partialCommitTestBackend{devicePanelTestBackend: newDevicePanelTestBackend(m)}
	backend := &mtpRawHiddenPartialBackend{partialCommitTestBackend: base, nonfiniteRawRow: -1}
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	base.target = target
	base.devicePanelTestBackend.target = target

	if target.captureTargetHidden {
		t.Fatal("fresh target unexpectedly started with hidden capture enabled")
	}
	if draft, err := NewQwen35MTPDraftSession(target, Qwen35MTPMaxDraftDepth+1); err == nil || draft != nil {
		t.Fatalf("invalid-depth constructor draft/error=%v/%v, want typed refusal", draft, err)
	}
	if target.captureTargetHidden || target.Cache.Len() != 0 || target.halKV.Len() != 0 {
		t.Fatalf("failed constructor mutated target: capture=%t host_kv=%d device_kv=%d", target.captureTargetHidden, target.Cache.Len(), target.halKV.Len())
	}
}

func TestQwen35MTPVulkanTargetWarmHistoryAdmissionIsAtomic(t *testing.T) {
	m := NewSyntheticQwen38MTP()
	prefix := []int{3, 7, 11, 5}

	missing := m.NewSession()
	defer missing.Close()
	missing.Prefill(prefix)
	before, err := missing.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer before.Close()
	draft, err := NewQwen35MTPDraftSession(missing, 2)
	if err == nil || draft != nil {
		t.Fatalf("warm target without raw history draft/error=%v/%v, want typed refusal", draft, err)
	}
	assertQwen35MTPUnsupported(t, err)
	if missing.captureTargetHidden {
		t.Fatal("failed warm-target constructor enabled capture after incomplete history")
	}
	after, err := missing.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer after.Close()
	if before.Backend != nil || after.Backend != nil || before.halKV != nil || after.halKV != nil || before.qwen35 != nil || after.qwen35 != nil {
		t.Fatal("CPU prefix snapshot unexpectedly carried backend state")
	}
	if before.Tokens != after.Tokens || !reflect.DeepEqual(before.Cache, after.Cache) ||
		before.captureTargetHidden != after.captureTargetHidden ||
		!reflect.DeepEqual(before.targetHidden, after.targetHidden) ||
		!reflect.DeepEqual(before.targetHiddenTokens, after.targetHiddenTokens) {
		t.Fatal("failed warm-target constructor mutated the CPU prefix snapshot")
	}
	if _, err := missing.VerifyTokenLineage(prefix); err != nil {
		t.Fatalf("failed constructor changed token lineage: %v", err)
	}

	complete := m.NewSession()
	defer complete.Close()
	complete.captureTargetHidden = true
	complete.Prefill(prefix)
	draft, err = NewQwen35MTPDraftSession(complete, 2)
	if err != nil {
		t.Fatalf("warm target with complete raw history: %v", err)
	}
	draft.Close()
	for pos := range prefix {
		if _, err := complete.TargetHiddenAt(pos); err != nil {
			t.Fatalf("complete warm history row %d: %v", pos, err)
		}
	}
}
