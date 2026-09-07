package model

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type sequencePrefillBackend struct {
	*recordingQwen35Backend
	path string
	err  error

	calls    int
	retires  int
	requests []compute.Qwen35SequencePrefillRequest
	badKV    bool
	badCount bool
	badState bool

	orderClock        int
	readOrder         int
	retireOrder       int
	retired           bool
	failReadIfRetired bool
	lastHiddenData    []float32
}

func newSequencePrefillBackend(m *Model) *sequencePrefillBackend {
	return &sequencePrefillBackend{recordingQwen35Backend: newRecordingQwen35Backend(m), path: compute.Qwen35SequencePrefillPath}
}

func (b *sequencePrefillBackend) Qwen35SequencePrefillPath() string { return b.path }

func (b *sequencePrefillBackend) RetireRequestResources() {
	b.retires++
	b.retired = true
	b.orderClock++
	b.retireOrder = b.orderClock
}

func (b *sequencePrefillBackend) Read(t compute.Tensor) []float32 {
	b.orderClock++
	b.readOrder = b.orderClock
	if b.failReadIfRetired && b.retired {
		panic("read attempted after request resources retired")
	}
	return b.recordingQwen35Backend.Read(t)
}

func (b *sequencePrefillBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	b.calls++
	b.requests = append(b.requests, req)
	if b.err != nil {
		return compute.Qwen35SequencePrefillResult{}, b.err
	}
	if b.badState && len(req.States) > 0 {
		req.States[0].Conv = compute.Tensor{}
	}
	if !b.badKV {
		for token := range req.TokenIDs {
			pos := req.StartPos + token
			for layer := range req.Layers {
				width := req.NumKVHeads * req.HeadDim
				z := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
				req.KV.AppendKV(layer, z, z, z, pos)
			}
		}
	}
	hiddenData := make([]float32, req.Hidden)
	if len(b.lastHiddenData) == req.Hidden {
		copy(hiddenData, b.lastHiddenData)
	}
	hidden := compute.NewF32(b.Backend, []int{req.Hidden}, hiddenData)
	var logits compute.Tensor
	if req.NeedLogits {
		logits = compute.NewF32(b.Backend, []int{b.model.Cfg.VocabSize}, make([]float32, b.model.Cfg.VocabSize))
	}
	tokens := len(req.TokenIDs)
	if b.badCount {
		tokens--
	}
	return compute.Qwen35SequencePrefillResult{LastHidden: hidden, Logits: logits, Tokens: tokens}, nil
}

type vulkanPrefillBackend struct {
	*sequencePrefillBackend
}

func (b *vulkanPrefillBackend) Name() string          { return "vulkan" }
func (b *vulkanPrefillBackend) Qwen35GDNPath() string { return Qwen35GDNVulkanPath }

type sequenceMarkerOnlyBackend struct{ *recordingQwen35Backend }

func (*sequenceMarkerOnlyBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func TestQwen35SequencePrefillDispatchCarriesResidentStateAndKV(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newSequencePrefillBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	conv := make([]compute.Buffer, len(s.qwen35HAL.layers))
	recurrent := make([]compute.Buffer, len(s.qwen35HAL.layers))
	for i, state := range s.qwen35HAL.layers {
		conv[i], recurrent[i] = state.conv.Buf(), state.recurrent.Buf()
	}
	got := s.Prefill([]int{3, 7, 11})
	if len(got) != m.Cfg.VocabSize || be.calls != 1 || s.halKV.Len() != 3 {
		t.Fatalf("dispatch logits=%d calls=%d kv=%d", len(got), be.calls, s.halKV.Len())
	}
	req := be.requests[0]
	if req.KV != s.halKV || len(req.Layers) != m.Cfg.NumLayers || len(req.States) != m.Cfg.NumLayers {
		t.Fatalf("request kv/layers/states mismatch: kv_same=%v layers=%d states=%d", req.KV == s.halKV, len(req.Layers), len(req.States))
	}
	for i, state := range req.States {
		if state.Conv.Buf() != conv[i] || state.Recurrent.Buf() != recurrent[i] {
			t.Fatalf("layer %d state handle changed", i)
		}
	}
	for i, theta := range req.RoPEThetaForLayer {
		if want := m.Cfg.ropeThetaForLayer(i); theta != want {
			t.Fatalf("layer %d theta=%g want=%g", i, theta, want)
		}
	}
	if be.gdnCalls != 0 {
		t.Fatalf("sequence success replayed scalar GDN %d times", be.gdnCalls)
	}
	if be.retires != 1 {
		t.Fatalf("request retirement calls=%d, want 1 after successful sequence prefill", be.retires)
	}
}

func TestQwen35SequencePrefillNoCapabilityFallsBack(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Prefill([]int{3, 7}); len(got) != m.Cfg.VocabSize || be.gdnCalls == 0 {
		t.Fatalf("fallback logits=%d scalar_calls=%d", len(got), be.gdnCalls)
	}
}

func TestQwen35SequencePrefillAdvertisedFailuresCloseWithoutScalarReplay(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	injected := errors.New("sequence injected failure")
	cases := map[string]compute.Backend{
		"marker-only": &sequenceMarkerOnlyBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)},
		"wrong-path":  func() compute.Backend { b := newSequencePrefillBackend(m); b.path = "wrong"; return b }(),
		"execution":   func() compute.Backend { b := newSequencePrefillBackend(m); b.err = injected; return b }(),
		"token-count": func() compute.Backend { b := newSequencePrefillBackend(m); b.badCount = true; return b }(),
		"kv-length":   func() compute.Backend { b := newSequencePrefillBackend(m); b.badKV = true; return b }(),
	}
	for name, backend := range cases {
		t.Run(name, func(t *testing.T) {
			s, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				r := recover()
				if r == nil || !s.halClosed || s.halFailure == nil {
					t.Fatalf("panic=%v closed=%v failure=%v", r, s.halClosed, s.halFailure)
				}
				var recording *recordingQwen35Backend
				switch b := backend.(type) {
				case *sequencePrefillBackend:
					recording = b.recordingQwen35Backend
				case *sequenceMarkerOnlyBackend:
					recording = b.recordingQwen35Backend
				}
				if recording != nil && recording.gdnCalls != 0 {
					t.Fatalf("failed sequence replayed scalar GDN %d times", recording.gdnCalls)
				}
			}()
			s.Prefill([]int{3, 7})
		})
	}
}

func TestQwen35SequencePrefillNoLogitsUsesSameContract(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newSequencePrefillBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	s.PrefillNoLogits([]int{3, 7})
	if be.calls != 1 || be.requests[0].NeedLogits || s.halKV.Len() != 2 {
		t.Fatalf("calls=%d need_logits=%v kv=%d", be.calls, be.requests[0].NeedLogits, s.halKV.Len())
	}
	if be.retires != 1 {
		t.Fatalf("request retirement calls=%d, want 1 after successful no-logits sequence prefill", be.retires)
	}
}

func TestPrefillHAL_ReadbackBeforeRetirement(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newSequencePrefillBackend(m)
	be.failReadIfRetired = true
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	// Calling prefillHAL directly with wantLogits=true
	got := s.prefillHAL([]int{3, 7, 11}, true)
	if len(got) != m.Cfg.VocabSize {
		t.Fatalf("got logits len=%d, want %d", len(got), m.Cfg.VocabSize)
	}
	if be.calls != 1 {
		t.Fatalf("expected 1 sequence prefill call, got %d", be.calls)
	}
	if be.retires != 1 {
		t.Fatalf("expected 1 retirement, got %d", be.retires)
	}
	if be.readOrder == 0 || be.retireOrder == 0 {
		t.Fatalf("expected both read and retire to occur, readOrder=%d retireOrder=%d", be.readOrder, be.retireOrder)
	}
	if be.readOrder >= be.retireOrder {
		t.Fatalf("readback must occur before retirement: readOrder=%d, retireOrder=%d", be.readOrder, be.retireOrder)
	}
}

func TestPrefillHAL_NoLogitsRetiresWithoutRead(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newSequencePrefillBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	// Calling prefillHAL directly with wantLogits=false
	got := s.prefillHAL([]int{3, 7, 11}, false)
	if got != nil {
		t.Fatalf("expected nil logits when wantLogits=false, got len=%d", len(got))
	}
	if be.calls != 1 {
		t.Fatalf("expected 1 sequence prefill call, got %d", be.calls)
	}
	if be.readOrder != 0 {
		t.Fatalf("expected 0 reads when wantLogits=false, got readOrder=%d", be.readOrder)
	}
	if be.retires != 1 {
		t.Fatalf("expected 1 retirement, got %d", be.retires)
	}
}

func TestQwen35SequencePrefill_PathAttributionAndStepSync(t *testing.T) {
	t.Run("vulkan-path-attribution-on-failure", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		inner := newSequencePrefillBackend(m)
		inner.err = errors.New("injected vulkan sequence failure")
		vulkanBE := &vulkanPrefillBackend{sequencePrefillBackend: inner}

		s, err := m.NewBackendSessionChecked(vulkanBE)
		if err != nil {
			t.Fatalf("NewBackendSessionChecked: %v", err)
		}
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic from failBackendForward, got nil")
			}
			var berr *BackendForwardOperationError
			if !errors.As(r.(error), &berr) {
				t.Fatalf("expected *BackendForwardOperationError, got %T (%v)", r, r)
			}
			if berr.Backend != "vulkan" {
				t.Errorf("Backend = %q, want %q", berr.Backend, "vulkan")
			}
			if berr.Path != Qwen35GDNVulkanPath {
				t.Errorf("Path = %q, want %q", berr.Path, Qwen35GDNVulkanPath)
			}
			if berr.Stage != "sequence prefill" {
				t.Errorf("Stage = %q, want %q", berr.Stage, "sequence prefill")
			}
			if berr.Layer != -1 {
				t.Errorf("Layer = %d, want -1", berr.Layer)
			}
			if !s.halClosed {
				t.Error("expected session halClosed=true")
			}
			if s.halFailure == nil {
				t.Error("expected session halFailure != nil")
			}
		}()
		s.prefillHAL([]int{3, 7, 11}, true)
	})

	t.Run("step-sync-and-target-hidden-capture", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		be := newSequencePrefillBackend(m)
		testHidden := make([]float32, m.Cfg.HiddenSize)
		for i := range testHidden {
			testHidden[i] = float32(i + 1)
		}
		be.lastHiddenData = testHidden

		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatalf("NewBackendSessionChecked: %v", err)
		}
		defer s.Close()

		s.captureTargetHidden = true

		if s.halStep != 0 {
			t.Fatalf("initial halStep = %d, want 0", s.halStep)
		}
		if s.halLogitsWarm {
			t.Fatalf("initial halLogitsWarm = true, want false")
		}

		ids := []int{3, 7, 11}
		logits := s.prefillHAL(ids, true)
		if len(logits) != m.Cfg.VocabSize {
			t.Fatalf("logits len = %d, want %d", len(logits), m.Cfg.VocabSize)
		}
		if s.halStep != len(ids) {
			t.Fatalf("halStep = %d, want %d", s.halStep, len(ids))
		}
		if !s.halLogitsWarm {
			t.Fatal("halLogitsWarm = false, want true after prefillHAL with wantLogits=true")
		}

		// Verify TargetHiddenAt captures result.LastHidden at pos = halKV.Len() - 1
		lastPos := s.halKV.Len() - 1
		gotHidden, err := s.TargetHiddenAt(lastPos)
		if err != nil {
			t.Fatalf("TargetHiddenAt(%d): %v", lastPos, err)
		}
		if len(gotHidden) != len(testHidden) {
			t.Fatalf("TargetHiddenAt len = %d, want %d", len(gotHidden), len(testHidden))
		}
		for i := range testHidden {
			if gotHidden[i] != testHidden[i] {
				t.Fatalf("TargetHiddenAt[%d] = %g, want %g", i, gotHidden[i], testHidden[i])
			}
		}

		// Verify defensive copy: modifying returned slice does not mutate session cache
		gotHidden[0] = 9999.0
		againHidden, err := s.TargetHiddenAt(lastPos)
		if err != nil {
			t.Fatalf("TargetHiddenAt(%d) again: %v", lastPos, err)
		}
		if againHidden[0] != testHidden[0] {
			t.Fatalf("TargetHiddenAt returned mutable reference, again[0]=%g want %g", againHidden[0], testHidden[0])
		}

		// Verify bounds checks on TargetHiddenAt
		wantNegErr := fmt.Sprintf("target hidden pos -1 out of bounds (len=%d)", s.halKV.Len())
		if _, err := s.TargetHiddenAt(-1); err == nil || err.Error() != wantNegErr {
			t.Fatalf("TargetHiddenAt(-1) err = %v, want %q", err, wantNegErr)
		}
		wantOobErr := fmt.Sprintf("target hidden pos %d out of bounds (len=%d)", s.halKV.Len(), s.halKV.Len())
		if _, err := s.TargetHiddenAt(s.halKV.Len()); err == nil || err.Error() != wantOobErr {
			t.Fatalf("TargetHiddenAt(%d) err = %v, want %q", s.halKV.Len(), err, wantOobErr)
		}

		// Test second prefill with wantLogits=false
		moreIDs := []int{13, 17}
		noLogits := s.prefillHAL(moreIDs, false)
		if noLogits != nil {
			t.Fatalf("expected nil logits with wantLogits=false, got len=%d", len(noLogits))
		}
		if s.halStep != len(ids)+len(moreIDs) {
			t.Fatalf("halStep = %d, want %d", s.halStep, len(ids)+len(moreIDs))
		}
		if !s.halLogitsWarm {
			t.Fatal("halLogitsWarm should remain true")
		}
	})
}
