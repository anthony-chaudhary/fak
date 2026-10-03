//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func missingQ6GraphFixture(t *testing.T) (*Model, []int) {
	t.Helper()
	if !metalgemm.Available() {
		if os.Getenv("FAK_METAL_REQUIRE_DEVICE") == "1" {
			t.Fatal("physical Metal device required")
		}
		t.Skip("no Metal device available")
	}
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	name := layerName(0, "mlp.down_proj.weight")
	delete(m.q4kw, name)
	m.kqw = map[string]*kQuantTensor{name: randomQ6KTensor(cfg.HiddenSize, cfg.IntermediateSize, 13499)}
	// Cached nil is the existing unavailable-residency outcome. Keep the host
	// Q6 payload intact so declining graph admission cannot erase host fallback.
	metalQ4KMu.Lock()
	prior, existed := metalQ6KW[m]
	metalQ6KW[m] = map[string]*metalgemm.Q6KWeight{name: nil}
	metalQ4KMu.Unlock()
	t.Cleanup(func() {
		m.releaseMetalQ8Residency()
		releaseMetalQ4KResidency(m)
		metalQ4KMu.Lock()
		if existed {
			metalQ6KW[m] = prior
		} else {
			delete(metalQ6KW, m)
		}
		metalQ4KMu.Unlock()
	})
	ids := make([]int, 32)
	for i := range ids {
		ids[i] = (7 + i*19) % cfg.VocabSize
	}
	return m, ids
}

func requireMissingQ6DeclineUnchanged(t *testing.T, s *Session, before *KVCache, buffers int) {
	t.Helper()
	assertKVCacheQuantClose(t, "unavailable Q6 pre-admission", before, s.Cache)
	assertLinearAttnCacheQuantClose(t, "unavailable Q6 pre-admission", before.linear, s.Cache.linear)
	if s.qwen35HAL != nil {
		t.Fatal("unavailable Q6 attached auxiliary sequence owner")
	}
	if got := s.Qwen35MetalForwardSequenceReceipt(); got != (Qwen35MetalForwardSequenceReceipt{}) {
		t.Fatalf("decline published graph execution: %+v", got)
	}
	if got := metalgemm.GDNLiveBufferCount(); got != buffers {
		t.Fatalf("decline changed physical GDN buffers: %d -> %d", buffers, got)
	}
}

// fak-test:runtime slow est=15s
func TestMetalQwen35MissingQ6HandleAdmissionAndHostFallback(t *testing.T) {
	m, ids := missingQ6GraphFixture(t)
	reference := m.NewSession()
	reference.Q4K, reference.MetalQ4K = true, true
	defer reference.Close()
	var want []float32
	if err := recoverError(func() { want = reference.headResident(reference.prefillQwen35HybridQ4KHidden(ids)) }); err != nil {
		t.Fatalf("same-model host reference panic: %v", err)
	}
	candidate := m.NewSession()
	candidate.Q4K, candidate.MetalQ4K = true, true
	defer candidate.Close()
	before, buffers := candidate.Cache.Clone(), metalgemm.GDNLiveBufferCount()
	err := candidate.EnableQwen35MetalGDNPreprojectedSequence()
	var unsupported *UnsupportedGDNPreprojectedSequenceError
	if !errors.As(err, &unsupported) {
		t.Errorf("missing resident Q6 handle admitted or wrong refusal: %T %v", err, err)
	}
	requireMissingQ6DeclineUnchanged(t, candidate, before, buffers)
	var got []float32
	if err := recoverError(func() { got = candidate.Prefill(ids) }); err != nil {
		t.Fatalf("production fallback panicked: %v", err)
	}
	if len(got) != m.Cfg.VocabSize || len(got) == 0 {
		t.Fatalf("fallback logits=%d want %d", len(got), m.Cfg.VocabSize)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("fallback nonfinite logit[%d]=%g", i, v)
		}
	}
	if candidate.Cache.Len() != len(ids) || candidate.q4kHybridPrefillChunks == 0 {
		t.Fatalf("host fallback not invoked: cache=%d chunks=%d", candidate.Cache.Len(), candidate.q4kHybridPrefillChunks)
	}
	assertQuantLogitsClose(t, "same-model unavailable Q6 fallback", want, got)
	assertKVCacheQuantClose(t, "same-model unavailable Q6 fallback", reference.Cache, candidate.Cache)
	assertLinearAttnCacheQuantClose(t, "same-model unavailable Q6 fallback", reference.Cache.linear, candidate.Cache.linear)
}

// fak-test:runtime slow est=15s
func TestMetalQwen35MissingQ6DirectGraphDeclinesBeforeOpen(t *testing.T) {
	m, ids := missingQ6GraphFixture(t)
	name := layerName(0, "mlp.down_proj.weight")
	metalQ4KMu.Lock()
	delete(metalQ6KW[m], name)
	metalQ4KMu.Unlock()
	handle := m.metalQ6KWeight(name, m.kqw[name])
	if handle == nil {
		t.Fatal("valid-owner prerequisite: Q6 graph handle unavailable")
	}
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	defer s.Close()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatalf("valid-owner prerequisite: %v", err)
	}
	if s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		t.Fatal("valid owner not accepted before handle loss")
	}
	owner := s.qwen35HAL
	layers := append([]Qwen35GDNAuxState(nil), owner.sequenceLayers...)
	before, buffers := s.Cache.Clone(), metalgemm.GDNLiveBufferCount()
	priorReceipt := s.Qwen35MetalForwardSequenceReceipt()
	backend := owner.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	metalQ4KMu.Lock()
	metalQ6KW[m][name] = nil
	metalQ4KMu.Unlock()
	defer func() {
		metalQ4KMu.Lock()
		metalQ6KW[m][name] = handle
		metalQ4KMu.Unlock()
	}()
	var logits []float32
	var receipt Qwen35MetalForwardSequenceReceipt
	var accepted bool
	var err error
	if panicErr := recoverError(func() { logits, receipt, accepted, err = backend.Qwen35MetalForwardSequence(s, ids) }); panicErr != nil {
		t.Fatalf("direct pre-admission panicked: %v", panicErr)
	}
	if err != nil || accepted || len(logits) != 0 {
		t.Errorf("direct missing-handle result accepted=%v logits=%d err=%T %v", accepted, len(logits), err, err)
	}
	if receipt != (Qwen35MetalForwardSequenceReceipt{}) {
		t.Errorf("direct decline opened graph: %+v", receipt)
	}
	assertKVCacheQuantClose(t, "owned missing-Q6 decline", before, s.Cache)
	assertLinearAttnCacheQuantClose(t, "owned missing-Q6 decline", before.linear, s.Cache.linear)
	if s.qwen35HAL != owner || !reflect.DeepEqual(owner.sequenceLayers, layers) || !owner.sequenceAccepted || owner.sequenceFailure != nil {
		t.Fatal("direct decline changed accepted owner/state")
	}
	if got := s.Qwen35MetalForwardSequenceReceipt(); got != priorReceipt {
		t.Fatalf("direct decline changed session receipt: %+v", got)
	}
	if got := metalgemm.GDNLiveBufferCount(); got != buffers {
		t.Fatalf("direct decline changed GDN buffer count: %d -> %d", buffers, got)
	}
}
