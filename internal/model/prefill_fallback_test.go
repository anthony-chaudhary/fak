package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func prefillFallbackCounts() map[string]uint64 {
	out := map[string]uint64{}
	for _, c := range PrefillFallbackCounts() {
		out[c.Reason] = c.Count
	}
	return out
}

// prefillFallbackDelta returns how many times reason was counted since before.
func prefillFallbackDelta(before map[string]uint64, reason string) uint64 {
	return prefillFallbackCounts()[reason] - before[reason]
}

// TestPrefillReplayFallbackCountsEachDeclineReason drives every whole-sequence
// prefill decline path and asserts each one increments its own labelled counter,
// so a silent drop to scalar token replay is visible per reason.
func TestPrefillReplayFallbackCountsEachDeclineReason(t *testing.T) {
	t.Run("EmbeddingCap", func(t *testing.T) {
		before := prefillFallbackCounts()
		cfg := qwen35HybridTestCfg()
		cfg.HiddenSize = 256
		m := NewSynthetic(cfg)
		q2k, err := NewQ2KEmbedding(makeTestQ2KPayload(cfg.VocabSize, cfg.HiddenSize), cfg.VocabSize, cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		m.Q2KEmbedding = q2k
		m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
		delete(m.manifest, "model.embed_tokens.weight")
		base := newSequencePrefillBackend(m)
		be := &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: 1024}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, advertised, err := s.tryQwen35SequencePrefill([]int{3, 7}, false); err != nil || advertised {
			t.Fatalf("advertised=%t err=%v, want a clean decline", advertised, err)
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineEmbeddingCap); got != 1 {
			t.Fatalf("embedding-cap delta=%d, want 1", got)
		}
	})

	t.Run("EmbeddingRowsUnsupported", func(t *testing.T) {
		before := prefillFallbackCounts()
		cfg := qwen35HybridTestCfg()
		cfg.HiddenSize = 256
		m := NewSynthetic(cfg)
		q2k, err := NewQ2KEmbedding(makeTestQ2KPayload(cfg.VocabSize, cfg.HiddenSize), cfg.VocabSize, cfg.HiddenSize)
		if err != nil {
			t.Fatal(err)
		}
		m.Q2KEmbedding = q2k
		m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
		delete(m.manifest, "model.embed_tokens.weight")
		be := &sequencePrefillWithoutEmbeddingRows{recordingQwen35Backend: newRecordingQwen35Backend(m)}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, advertised, err := s.tryQwen35SequencePrefill([]int{0, cfg.VocabSize - 1}, false); err != nil || advertised {
			t.Fatalf("advertised=%t err=%v, want a clean decline", advertised, err)
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineEmbeddingRowsUnsupported); got != 1 {
			t.Fatalf("embedding-rows-unsupported delta=%d, want 1", got)
		}
	})

	t.Run("OutputHeadCap", func(t *testing.T) {
		before := prefillFallbackCounts()
		m := NewSynthetic(qwen35HybridTestCfg())
		m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
		base := newSequencePrefillBackend(m)
		be := &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: 1024}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, advertised, err := s.tryQwen35SequencePrefill([]int{3, 7}, false); err != nil || advertised {
			t.Fatalf("advertised=%t err=%v, want a clean decline", advertised, err)
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineOutputHeadCap); got != 1 {
			t.Fatalf("output-head-cap delta=%d, want 1", got)
		}
	})

	t.Run("SuccessfulNativeRouteIsNotCounted", func(t *testing.T) {
		before := prefillFallbackCounts()
		m := NewSynthetic(qwen35HybridTestCfg())
		be := newSequencePrefillBackend(m)
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, used, err := s.tryQwen35SequencePrefill([]int{3, 7}, false); err != nil || !used {
			t.Fatalf("used=%t err=%v, want the native route", used, err)
		}
		for reason, count := range prefillFallbackCounts() {
			if count != before[reason] {
				t.Fatalf("native success changed fallback counter %q: %d -> %d", reason, before[reason], count)
			}
		}
	})
}

// TestPrefillReplayFallbackRefusesOnDeviceOnlySession asserts the Halo appliance
// profile refuses the scalar token-replay route, and that an admitted bypass
// keeps it while still counting it.
func TestPrefillReplayFallbackRefusesOnDeviceOnlySession(t *testing.T) {
	declined := Qwen35SequencePrefillRouteStatus{
		RequestedPath:  compute.Qwen35SequencePrefillPath,
		EffectivePath:  Qwen35SequencePrefillFallbackPath,
		DeclineReason:  Qwen35SequencePrefillDeclineEmbeddingCap,
		FallbackActive: true,
	}

	t.Run("DeviceOnlyRefuses", func(t *testing.T) {
		before := prefillFallbackCounts()
		s := &Session{M: &Model{}}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := deviceOnlyRefusal(t, func() { s.notePrefillFallback(declined) })
		if !errors.Is(err, ErrPrefillReplayFallback) {
			t.Fatalf("err=%v, want ErrPrefillReplayFallback", err)
		}
		var typed *BackendForwardOperationError
		if !errors.As(err, &typed) {
			t.Fatalf("err=%#v, want BackendForwardOperationError", err)
		}
		if !s.HostFallbackObserved() || !s.BackendSessionClosed() {
			t.Fatal("refusal did not record and close the session")
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineEmbeddingCap); got != 1 {
			t.Fatalf("refused decline delta=%d, want 1", got)
		}
	})

	t.Run("AdmittedBypassKeepsFallback", func(t *testing.T) {
		before := prefillFallbackCounts()
		s := &Session{M: &Model{}}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		s.SetHostPathBypass("device-only-session", func() {})
		s.notePrefillFallback(declined)
		if s.HostFallbackObserved() || s.BackendSessionClosed() {
			t.Fatal("admitted bypass refused the fallback")
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineEmbeddingCap); got != 1 {
			t.Fatalf("bypassed decline delta=%d, want 1", got)
		}
	})

	t.Run("PortableSessionKeepsFallback", func(t *testing.T) {
		before := prefillFallbackCounts()
		s := &Session{M: &Model{}}
		s.notePrefillFallback(declined)
		if s.HostFallbackObserved() || s.BackendSessionClosed() {
			t.Fatal("portable session refused the fallback")
		}
		if got := prefillFallbackDelta(before, Qwen35SequencePrefillDeclineEmbeddingCap); got != 1 {
			t.Fatalf("portable decline delta=%d, want 1", got)
		}
	})
}

// TestPrefillFallbackReasonLabelIsBounded pins the label grammar: an unbounded or
// malformed reason folds into the single "other" series.
func TestPrefillFallbackReasonLabelIsBounded(t *testing.T) {
	before := prefillFallbackCounts()
	CountPrefillFallback("Not A Label!")
	CountPrefillFallback("")
	if got := prefillFallbackDelta(before, PrefillFallbackOtherReason); got != 2 {
		t.Fatalf("other delta=%d, want 2", got)
	}
}
