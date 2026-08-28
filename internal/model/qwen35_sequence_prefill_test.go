package model

import (
	"errors"
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
}

func newSequencePrefillBackend(m *Model) *sequencePrefillBackend {
	return &sequencePrefillBackend{recordingQwen35Backend: newRecordingQwen35Backend(m), path: compute.Qwen35SequencePrefillPath}
}

func (b *sequencePrefillBackend) Qwen35SequencePrefillPath() string { return b.path }

func (b *sequencePrefillBackend) RetireRequestResources() { b.retires++ }

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
	hidden := compute.NewF32(b.Backend, []int{req.Hidden}, make([]float32, req.Hidden))
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
