package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41MHCConstantFixture(t *testing.T) *Model {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	// Two full layers distinguish cache keys even when checkpoint bytes are
	// shared. Attention and FFN constants already differ in this fixture.
	for name, meta := range m.manifest {
		if leaf, ok := strings.CutPrefix(name, "model.layers.0."); ok {
			m.manifest[layerName(1, leaf)] = meta
		}
	}
	m.Cfg.NumLayers = 2
	m.Cfg.DeepSeekV41.CompressRatios = []int{0, 0}
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	m.Cfg.DeepSeekV41.EngramLayerIDs = nil
	t.Cleanup(func() { _ = m.CloseWeights() })
	return m
}

func v41MHCConstantRecorder() *v41MHCTestBackend {
	b := newV41MHCTestBackend(true)
	b.paramFrees = map[compute.Buffer]int{}
	return b
}

func v41MHCConstantFrees(t *testing.T, b *v41MHCTestBackend, first, last, want int) {
	t.Helper()
	for _, tensor := range b.paramUploads[first:last] {
		if got := b.paramFrees[tensor.Buf()]; got != want {
			t.Fatalf("constant free count=%d want %d", got, want)
		}
	}
}

// This records only mHC split-constant upload bytes over CPU storage. It does
// not measure all H2D traffic, physical submits, latency, or KV residency.
// The changed recorder accepts both the parent's activation class and the
// candidate's weight class, so identical test bytes witness repeated uploads.
// fak-test:runtime medium est=5s lane=default
func TestV41DecodeResidentH2DMHCConstants(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("mHC constant residency witness requires cpu-ref")
	}
	t.Run("warm-decode-and-session-lifetime", func(t *testing.T) {
		m, oracle := v41MHCConstantFixture(t), v41MHCConstantFixture(t)
		b := v41MHCConstantRecorder()
		first := v41DenseTestSession(t, m, b)
		// Same dense seams, host mHC: only constant residency differs. The
		// pure host session reduces grouped output in another order.
		host := v41DenseTestSession(t, oracle, newV41DenseTestBackend())
		if got, want := first.Prefill([]int{1, 2}), host.Prefill([]int{1, 2}); !reflect.DeepEqual(got, want) {
			t.Fatal("constant residency changed prefill numerics")
		}
		count := 2 * m.Cfg.NumLayers
		if len(b.paramUploads) != count {
			t.Fatalf("cold parameter uploads=%d want %d", len(b.paramUploads), count)
		}
		for _, tensor := range b.paramUploads {
			if tensor.Numel() != compute.V41MHCParamWidth {
				t.Fatal("constant upload width changed")
			}
		}
		for _, token := range []int{3, 4, 5} {
			before := len(b.paramUploads)
			if got, want := first.Step(token), host.Step(token); !reflect.DeepEqual(got, want) {
				t.Fatal("constant residency changed decode numerics")
			}
			bytes := (len(b.paramUploads) - before) * compute.V41MHCParamWidth * 4
			t.Logf("software mHC split-constant upload: context=%d bytes=%d", len(first.v41State().history), bytes)
			if bytes != 0 {
				t.Fatalf("warm decode repeated %d bytes of immutable mHC split uploads", bytes)
			}
		}
		if !first.v41IncrementalEligible() || b.calls["rms-split"] != 4*count {
			t.Fatal("warm witness did not execute every layer's attention and FFN split")
		}
		v41MHCConstantFrees(t, b, 0, count, 0)
		second := v41DenseTestSession(t, m, b)
		second.Prefill([]int{1, 2})
		if len(b.paramUploads) != count {
			t.Fatal("second session re-uploaded model/backend constants")
		}
		first.Close()
		v41MHCConstantFrees(t, b, 0, count, 0)
		second.Step(3)
		if len(b.paramUploads) != count {
			t.Fatal("source session close evicted peer constants")
		}
		second.Close()
		v41MHCConstantFrees(t, b, 0, count, 0)
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, 0, count, 1)
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, 0, count, 1)
	})
	t.Run("backend-and-loaded-model-isolation", func(t *testing.T) {
		m, nextModel := v41MHCConstantFixture(t), v41MHCConstantFixture(t)
		v41WriteTensorF32(t, nextModel, layerName(0, "mhc.ffn_scale"), []float32{0.5, 0.5, 0.5})
		b, otherBackend := v41MHCConstantRecorder(), v41MHCConstantRecorder()
		a, peer, generation := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, otherBackend), v41DenseTestSession(t, nextModel, b)
		a.Prefill([]int{1})
		peer.Prefill([]int{1})
		generation.Prefill([]int{1})
		count := 2 * m.Cfg.NumLayers
		if len(b.paramUploads) != 2*count || len(otherBackend.paramUploads) != count {
			t.Fatal("cache confused loaded model or exact backend identity")
		}
		seen := map[compute.Buffer]bool{}
		for _, tensor := range append(append([]compute.Tensor(nil), b.paramUploads...), otherBackend.paramUploads...) {
			if seen[tensor.Buf()] {
				t.Fatal("different constant owners unexpectedly share a handle")
			}
			seen[tensor.Buf()] = true
		}
		a.Close()
		peer.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, 0, count, 1)
		v41MHCConstantFrees(t, b, count, 2*count, 0)
		v41MHCConstantFrees(t, otherBackend, 0, count, 1)
		generation.Step(2)
		if len(b.paramUploads) != 2*count {
			t.Fatal("closing another model evicted this generation")
		}
		generation.Close()
		if err := nextModel.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, count, 2*count, 1)
	})
	t.Run("warm-cache-still-refuses-malformed-coefficient", func(t *testing.T) {
		m := v41MHCConstantFixture(t)
		b := v41MHCConstantRecorder()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1})
		count, splits := len(b.paramUploads), b.calls["rms-split"]
		if count != 2*m.Cfg.NumLayers {
			t.Fatal("malformed-coefficient witness did not warm the constant cache")
		}
		base := append([]float32(nil), m.tensor(layerName(0, "mhc.base"))...)
		base[len(base)-1] = float32(math.NaN())
		v41WriteTensorF32(t, m, layerName(0, "mhc.base"), base)
		r := &v41MHCResident{d: s.v41MHCDeviceSeam()}
		// A nil stream pins admission before projection or stream access, even
		// when immutable device parameters already exist for this exact key.
		_, err := r.mixes(0, nil, "mhc.mixes.weight", "mhc.scale", "mhc.base", 1, 1e-6, 1e-6)
		var malformed *V41ForwardError
		if !errors.As(err, &malformed) || malformed.Stage != v41StageMHC || malformed.Layer != 0 || malformed.Err.Error() != "model: non-finite V41 mHC coefficient at 26" {
			t.Fatalf("warm malformed coefficient lost its original stage/index: %v", err)
		}
		if len(b.paramUploads) != count || b.calls["rms-split"] != splits || s.BackendSessionClosed() {
			t.Fatal("payload admission dispatched, uploaded, or retired the backend")
		}
		v41MHCConstantFrees(t, b, 0, count, 0)
	})
	t.Run("failed-staging-publishes-no-constant", func(t *testing.T) {
		m := v41MHCConstantFixture(t)
		b := v41MHCConstantRecorder()
		primary := &compute.BackendError{Backend: "mhc-constants", Class: compute.VulkanClassAllocationFailed, Err: compute.ErrVulkanAllocationFailed}
		b.paramUploadFailure = primary
		s := v41DenseTestSession(t, m, b)
		err := recoverError(func() { s.Prefill([]int{1}) })
		var selected *V41MHCOperationError
		var closed *BackendForwardOperationError
		if !errors.Is(err, primary) || !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Stage != "params upload" || !s.BackendSessionClosed() {
			t.Fatalf("constant upload failure lost typed retirement: %v", err)
		}
		if len(b.paramUploads) != 0 || b.calls["rms-split"] != 0 {
			t.Fatal("failed constant upload dispatched or entered the resident cache")
		}
		b.paramUploadFailure = nil
		retry := v41DenseTestSession(t, m, b)
		retry.Prefill([]int{1})
		if len(b.paramUploads) != 2*m.Cfg.NumLayers {
			t.Fatal("failed cache stage poisoned a later session")
		}
		retry.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, 0, len(b.paramUploads), 1)
	})
	t.Run("selected-split-failure-preserves-borrowed-owner", func(t *testing.T) {
		m := v41MHCConstantFixture(t)
		b := v41MHCConstantRecorder()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		count := len(b.paramUploads)
		primary := &compute.BackendError{Backend: "mhc-constants", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		b.failOp, b.failCall, b.cause = "rms-split", b.calls["rms-split"]+1, primary
		err := recoverError(func() { s.Step(3) })
		var selected *V41MHCOperationError
		if !errors.Is(err, primary) || !errors.As(err, &selected) || !s.BackendSessionClosed() || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatalf("selected split failure replayed or advanced state: %v", err)
		}
		if len(b.paramUploads) != count || b.calls["rms-split"] != b.failCall {
			t.Fatal("failed split repeated upload or dispatch")
		}
		v41MHCConstantFrees(t, b, 0, count, 0)
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41MHCConstantFrees(t, b, 0, count, 1)
	})
}
