package model

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The CPU recorder observes uploads and ownership, never physical traffic.
type v41RouterBiasRecorder struct {
	*v41DenseTestBackend
	biasUploads  []compute.Tensor
	roles        map[compute.Buffer]string
	frees        map[compute.Buffer]int
	calls        int
	fault        error
	uploadFault  error
	cleanupFault bool
}

func newV41RouterBiasRecorder() *v41RouterBiasRecorder {
	return &v41RouterBiasRecorder{v41DenseTestBackend: newV41DenseTestBackend(), roles: map[compute.Buffer]string{}, frees: map[compute.Buffer]int{}}
}
func (b *v41RouterBiasRecorder) V41RouterTopKAvailable() bool { return true }
func (b *v41RouterBiasRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	bias := site == "V4.1 router correction bias" || (strings.HasPrefix(site, "hal-weight ") && strings.HasSuffix(site, ".ffn.gate.e_score_correction_bias"))
	logits := site == "V4.1 router logits"
	if !bias && !logits {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 {
		panic("router operand is not F32")
	}
	if bias && b.uploadFault != nil {
		panic(b.uploadFault)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	if bias {
		b.biasUploads = append(b.biasUploads, out)
		b.roles[out.Buf()] = "bias"
	} else {
		b.roles[out.Buf()] = "logits"
	}
	return out
}
func (b *v41RouterBiasRecorder) Free(x compute.Tensor) {
	if role := b.roles[x.Buf()]; role != "" {
		b.frees[x.Buf()]++
		b.v41DenseTestBackend.Free(x)
		if role == "logits" && b.cleanupFault {
			panic(errors.New("secondary router logit cleanup"))
		}
		return
	}
	b.v41DenseTestBackend.Free(x)
}
func (b *v41RouterBiasRecorder) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.calls++
	if b.fault != nil {
		return compute.V41RouterPicks{}, b.fault
	}
	values := b.Backend.Read(logits)
	var correction []float32
	if hasBias {
		correction = b.Backend.Read(bias)
	}
	cfg := v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: routeScale}
	var out compute.V41RouterPicks
	for row := 0; row < rows; row++ {
		picks, err := v41RouteForGeometry(values[row*experts:(row+1)*experts], correction, cfg, full)
		if err != nil {
			return out, err
		}
		for _, pick := range picks {
			out.Experts = append(out.Experts, int32(pick.expert))
			out.Weights = append(out.Weights, pick.weight)
		}
	}
	return out, nil
}
func v41RouterBiasModel(t *testing.T) *Model {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	t.Cleanup(func() { _ = m.CloseWeights() })
	return m
}
func v41RouterBiasFreeCount(t *testing.T, b *v41RouterBiasRecorder, start, end, want int) {
	t.Helper()
	for _, tensor := range b.biasUploads[start:end] {
		if got := b.frees[tensor.Buf()]; got != want {
			t.Fatalf("bias free count=%d want %d", got, want)
		}
	}
}

// Existing APIs only: parent behavior must fail on repeated canonical uploads,
// not on new symbols or synthetic cache helpers.
// fak-test:runtime medium est=5s lane=default
func TestV41DecodeResidentH2DRouterBias(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("router residency witness requires cpu-ref")
	}
	t.Run("canonical-prefill-decode-and-lifetime", func(t *testing.T) {
		m, oracle := v41RouterBiasModel(t), v41RouterBiasModel(t)
		b := newV41RouterBiasRecorder()
		s := v41DenseTestSession(t, m, b)
		host := oracle.NewSession()
		defer host.Close()
		if got, want := s.Prefill([]int{1}), host.Prefill([]int{1}); !reflect.DeepEqual(got, want) {
			t.Fatal("resident router bias changed prefill logits")
		}
		if len(b.biasUploads) != m.Cfg.NumLayers {
			t.Fatal("cold bias count differs from layer count")
		}
		for _, token := range []int{2, 3, 4} {
			before := len(b.biasUploads)
			if got, want := s.Step(token), host.Step(token); !reflect.DeepEqual(got, want) {
				t.Fatal("resident router bias changed decode logits")
			}
			bytes := (len(b.biasUploads) - before) * V41RouterExperts * 4
			t.Logf("software router-bias upload: context=%d bytes=%d", len(s.v41State().history), bytes)
			if bytes != 0 {
				t.Fatalf("warm decode repeated %d bytes of canonical bias", bytes)
			}
		}
		if b.calls != 4*m.Cfg.NumLayers {
			t.Fatal("warm witness bypassed router dispatch")
		}
		peer := v41DenseTestSession(t, m, b)
		peer.Prefill([]int{1, 2})
		if len(b.biasUploads) != m.Cfg.NumLayers {
			t.Fatal("peer session repeated resident bias upload")
		}
		s.Close()
		v41RouterBiasFreeCount(t, b, 0, len(b.biasUploads), 0)
		peer.Step(3)
		peer.Close()
		v41RouterBiasFreeCount(t, b, 0, len(b.biasUploads), 0)
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 0, len(b.biasUploads), 1)
	})
	t.Run("noncanonical-mutable-and-nil-remain-transient", func(t *testing.T) {
		m := v41RouterBiasModel(t)
		b := newV41RouterBiasRecorder()
		s := v41DenseTestSession(t, m, b)
		cfg := v41DefaultRouterConfig()
		logits := make([]float32, cfg.Experts)
		canonical := m.tensor(layerName(0, "ffn.gate.e_score_correction_bias"))
		copyBias := append([]float32(nil), canonical...)
		call := s.v41RouterTopKFunc()
		check := func(bias []float32) {
			t.Helper()
			want, err := v41RouteForGeometry(logits, bias, cfg, true)
			if err != nil {
				t.Fatal(err)
			}
			got, err := call(0, logits, bias, cfg, true)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("router reused stale or different caller bias")
			}
		}
		check(canonical)
		check(copyBias) // equal values do not prove model ownership
		copyBias[383] = 100
		check(copyBias) // same slice address, changed values must be observed
		check(nil)
		if len(b.biasUploads) != 4 {
			t.Fatalf("canonical/copy/mutated/nil upload count=%d want4", len(b.biasUploads))
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 0)
		v41RouterBiasFreeCount(t, b, 1, 4, 1)
		check(canonical)
		if len(b.biasUploads) != 4 {
			t.Fatal("transient caller poisoned canonical cache")
		}
		// Missing/malformed canonical metadata must not introduce a view panic
		// or freeze an arbitrary input into the model's immutable cache.
		name := layerName(0, "ffn.gate.e_score_correction_bias")
		original := m.manifest[name]
		invalid := original
		invalid.Offset = len(m.raw) + 1
		m.manifest[name] = invalid
		_, err := call(0, logits, copyBias, cfg, true)
		m.manifest[name] = original
		if err != nil {
			t.Fatal(err)
		}
		if len(b.biasUploads) != 5 || b.frees[b.biasUploads[4].Buf()] != 1 {
			t.Fatal("malformed metadata admitted resident bias")
		}
		bare := &Session{M: m, Backend: b}
		if _, err := bare.v41RouterTopKFunc()(0, logits, canonical, cfg, true); err != nil {
			t.Fatal(err)
		}
		if len(b.biasUploads) != 6 || b.frees[b.biasUploads[5].Buf()] != 1 {
			t.Fatal("unheld model lifetime admitted resident bias")
		}
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 1)
	})
	t.Run("loaded-model-and-backend-isolation", func(t *testing.T) {
		m, reload := v41RouterBiasModel(t), v41RouterBiasModel(t)
		bias := make([]float32, V41RouterExperts)
		bias[383] = 100
		v41WriteTensorF32(t, reload, layerName(0, "ffn.gate.e_score_correction_bias"), bias)
		b, other := newV41RouterBiasRecorder(), newV41RouterBiasRecorder()
		a, peer, next := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, other), v41DenseTestSession(t, reload, b)
		a.Prefill([]int{1})
		peer.Prefill([]int{1})
		next.Prefill([]int{1})
		if len(b.biasUploads) != 2 || len(other.biasUploads) != 1 || b.biasUploads[0].Buf() == b.biasUploads[1].Buf() || b.biasUploads[0].Buf() == other.biasUploads[0].Buf() {
			t.Fatal("bias cache crossed loaded-model or backend identity")
		}
		a.Close()
		peer.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 1)
		v41RouterBiasFreeCount(t, b, 1, 2, 0)
		next.Step(2)
		next.Close()
		if err := reload.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 1, 2, 1)
		v41RouterBiasFreeCount(t, other, 0, 1, 1)
	})
	t.Run("primary-failure-survives-transient-cleanup", func(t *testing.T) {
		m := v41RouterBiasModel(t)
		b := newV41RouterBiasRecorder()
		s := v41DenseTestSession(t, m, b)
		cfg := v41DefaultRouterConfig()
		bias := m.tensor(layerName(0, "ffn.gate.e_score_correction_bias"))
		primary := &compute.BackendError{Backend: "router-residency", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
		b.fault, b.cleanupFault = primary, true
		var err error
		panicked := recoverError(func() {
			_, err = s.v41RouterTopKFunc()(0, make([]float32, cfg.Experts), bias, cfg, true)
		})
		if panicked != nil {
			t.Fatalf("selected router cleanup escaped as a panic: %v", panicked)
		}
		var closed *BackendForwardOperationError
		if !errors.Is(err, primary) || !errors.As(err, &closed) || closed.Path != "v41-router-topk" || !s.BackendSessionClosed() || b.calls != 1 {
			t.Fatalf("primary route error lost: %v", err)
		}
		if len(b.biasUploads) != 1 {
			t.Fatal("selected fault unexpectedly retried upload")
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 0)
		for buf, role := range b.roles {
			if role == "logits" && b.frees[buf] != 1 {
				t.Fatal("primary failure leaked or double-freed logits")
			}
		}
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 1)
	})
	t.Run("failed-resident-upload-is-not-published", func(t *testing.T) {
		m := v41RouterBiasModel(t)
		b := newV41RouterBiasRecorder()
		s := v41DenseTestSession(t, m, b)
		cfg := v41DefaultRouterConfig()
		bias := m.tensor(layerName(0, "ffn.gate.e_score_correction_bias"))
		primary := &compute.BackendError{Backend: "router-residency", Class: compute.VulkanClassAllocationFailed, Err: compute.ErrVulkanAllocationFailed}
		b.uploadFault = primary
		_, err := s.v41RouterTopKFunc()(0, make([]float32, cfg.Experts), bias, cfg, true)
		if !errors.Is(err, primary) || !s.BackendSessionClosed() || b.calls != 0 || len(b.biasUploads) != 0 {
			t.Fatalf("failed resident upload was published or replayed: %v", err)
		}
		b.uploadFault = nil
		next := v41DenseTestSession(t, m, b)
		if _, err := next.v41RouterTopKFunc()(0, make([]float32, cfg.Experts), bias, cfg, true); err != nil {
			t.Fatal(err)
		}
		if len(b.biasUploads) != 1 {
			t.Fatal("failed stage poisoned later session")
		}
		next.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		v41RouterBiasFreeCount(t, b, 0, 1, 1)
	})
}
