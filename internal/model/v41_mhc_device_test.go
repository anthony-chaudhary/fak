package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41MHCTestBackend executes an independent scalar transcription of the mHC
// stream arithmetic over cpu-ref storage. Its DeviceMemory promise tests
// routing, ownership and failure typing only, never physical qualification;
// numerical device parity is TestVulkanV41MHC in internal/compute.
type v41MHCTestBackend struct {
	*v41DenseTestBackend
	supported   bool
	calls       map[string]int
	outputs     map[compute.Buffer]string
	readScratch []float32
	failOp      string
	failCall    int
	cause       error
	// Optional residency instrumentation. Existing arithmetic remains unchanged.
	paramUploads       []compute.Tensor
	paramFrees         map[compute.Buffer]int
	paramUploadFailure error
	// Optional Engram gate and compressor pool transcriptions, off by default
	// (v41_engram_pool_device_test.go). streamUploads counts the Engram host
	// round trip's panel republications.
	engram, pool           bool
	engramFlag             bool
	engramCause, poolCause error
	streamUploads          int
}

func newV41MHCTestBackend(supported bool) *v41MHCTestBackend {
	return &v41MHCTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), supported: supported, calls: map[string]int{}, outputs: map[compute.Buffer]string{}}
}

func (b *v41MHCTestBackend) SupportsV41MHC() bool { return b.supported }

func (b *v41MHCTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 mHC streams" {
		b.streamUploads++
	}
	if !strings.HasPrefix(site, "V4.1 mHC ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	params := site == "V4.1 mHC split params"
	if dt != compute.F32 || (class != compute.MemoryActivation && !(params && class == compute.MemoryWeights)) {
		panic("mHC upload has an invalid F32 memory class")
	}
	if params && b.paramUploadFailure != nil {
		panic(b.paramUploadFailure)
	}
	// Device uploads own storage; cpu-ref's identity Upload would hide aliasing.
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	if params && b.paramFrees != nil {
		b.paramUploads = append(b.paramUploads, out)
		b.paramFrees[out.Buf()] = 0
	}
	return out
}

func (b *v41MHCTestBackend) operand(t compute.Tensor, shape ...int) []float32 {
	// Projection outputs come from the embedded cpu-ref MatMul.
	if (t.Backend() != b && t.Backend() != b.Backend) || t.Dtype != compute.F32 || !reflect.DeepEqual(t.Shape, shape) {
		panic(&compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassInvalidGeometry, Err: compute.ErrVulkanInvalidGeometry, Message: "invalid mHC operand"})
	}
	return b.Backend.Read(t)
}

func (b *v41MHCTestBackend) publish(op string, shape []int, values []float32) (compute.Tensor, error) {
	b.calls[op]++
	if b.failOp == op && b.calls[op] == b.failCall {
		return compute.Tensor{}, b.cause
	}
	out := compute.NewF32(b, shape, values)
	b.outputs[out.Buf()] = op
	return out, nil
}

func (b *v41MHCTestBackend) V41MHCInitialStreams(x compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	in := b.operand(x, rows, hidden)
	out := make([]float32, rows*4*hidden)
	for r := 0; r < rows; r++ {
		for h := 0; h < 4; h++ {
			for d := 0; d < hidden; d++ {
				out[(r*4+h)*hidden+d] = v41RoundBF16(in[r*hidden+d])
			}
		}
	}
	return b.publish("initial", []int{rows, 4, hidden}, out)
}

func (b *v41MHCTestBackend) V41MHCSplit(mixes, scale, base compute.Tensor, rows, iters int, eps float32) (compute.Tensor, error) {
	m, sc, ba := b.operand(mixes, rows, 24), b.operand(scale, 3), b.operand(base, 24)
	out := make([]float32, 0, rows*24)
	for r := 0; r < rows; r++ {
		// The split's own arithmetic is witnessed on silicon; here the host
		// reference stands in so routing parity is exact.
		mix, err := v41MHCSplit(m[r*24:(r+1)*24], sc, ba, 4, iters, eps)
		if err != nil {
			panic(err)
		}
		out = append(append(append(out, mix.pre...), mix.post...), mix.comb...)
	}
	return b.publish("split", []int{rows, 24}, out)
}

func (b *v41MHCTestBackend) V41MHCRMSSplit(raw, streams, params compute.Tensor, rows, hidden, iters int, eps, rmsEps float32) (compute.Tensor, error) {
	m, s, p := b.operand(raw, rows, 24), b.operand(streams, rows, 4, hidden), b.operand(params, 27)
	width := 4 * hidden
	out := make([]float32, 0, rows*24)
	for r := 0; r < rows; r++ {
		var ss float32
		for _, v := range s[r*width : (r+1)*width] {
			ss += float32(v * v)
		}
		rsqrt := float32(1 / math.Sqrt(float64(ss/float32(width)+rmsEps)))
		scaled := make([]float32, 24)
		for i := range scaled {
			scaled[i] = float32(m[r*24+i] * rsqrt)
		}
		mix, err := v41MHCSplit(scaled, p[:3], p[3:], 4, iters, eps)
		if err != nil {
			panic(err)
		}
		out = append(append(append(out, mix.pre...), mix.post...), mix.comb...)
	}
	return b.publish("rms-split", []int{rows, 24}, out)
}

func (b *v41MHCTestBackend) V41MHCPre(streams, coeffs compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	s, k := b.operand(streams, rows, 4, hidden), b.operand(coeffs, rows, 24)
	out := make([]float32, rows*hidden)
	for r := 0; r < rows; r++ {
		for d := 0; d < hidden; d++ {
			var acc float32
			for h := 0; h < 4; h++ {
				acc += float32(k[r*24+h] * s[(r*4+h)*hidden+d])
			}
			out[r*hidden+d] = v41RoundBF16(acc)
		}
	}
	return b.publish("pre", []int{rows, hidden}, out)
}

func (b *v41MHCTestBackend) V41MHCPost(x, streams, coeffs compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	in, s, k := b.operand(x, rows, hidden), b.operand(streams, rows, 4, hidden), b.operand(coeffs, rows, 24)
	out := make([]float32, rows*4*hidden)
	for r := 0; r < rows; r++ {
		for dst := 0; dst < 4; dst++ {
			for d := 0; d < hidden; d++ {
				var acc float32
				for src := 0; src < 4; src++ {
					acc += float32(k[r*24+8+src*4+dst] * s[(r*4+src)*hidden+d])
				}
				acc += float32(k[r*24+4+dst] * v41RoundBF16(in[r*hidden+d]))
				out[(r*4+dst)*hidden+d] = v41RoundBF16(acc)
			}
		}
	}
	return b.publish("post", []int{rows, 4, hidden}, out)
}

// Read reuses one host buffer across every mHC readback, so a seam that kept
// the borrowed slice instead of owning it would corrupt an earlier result.
func (b *v41MHCTestBackend) Read(x compute.Tensor) []float32 {
	if b.outputs[x.Buf()] == "" {
		return b.v41DenseTestBackend.Read(x)
	}
	values := b.Backend.Read(x)
	if cap(b.readScratch) < len(values) {
		b.readScratch = make([]float32, len(values))
	}
	b.readScratch = b.readScratch[:len(values)]
	copy(b.readScratch, values)
	return b.readScratch
}

func (b *v41MHCTestBackend) Free(x compute.Tensor) {
	if _, recorded := b.paramFrees[x.Buf()]; recorded {
		b.paramFrees[x.Buf()]++
	}
	if b.outputs[x.Buf()] != "" {
		delete(b.outputs, x.Buf())
		for i := range b.readScratch {
			b.readScratch[i] = float32(math.NaN())
		}
	}
	b.v41DenseTestBackend.Free(x)
}

// fak-test:runtime medium est=10s lane=default
func TestV41MHCDeviceSeamRoutesEveryFullOp(t *testing.T) {
	for _, tc := range []struct {
		name string
		role bool
	}{{"plain", false}, {"compressed-role", true}} {
		t.Run(tc.name, func(t *testing.T) {
			build := func(t *testing.T) *Model {
				if tc.role {
					return v41IncrementalExpertFixture(t, true, false)
				}
				return v41FullStepPatchedNorms(t)
			}
			m := build(t)
			if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
				t.Fatalf("fixture is not the full geometry: %v", err)
			}
			device := newV41MHCTestBackend(true)
			host := newV41MHCTestBackend(false)
			ds, hs := v41DenseTestSession(t, m, device), v41DenseTestSession(t, build(t), host)
			if ds.v41State().mhcDevice == nil || hs.v41State().mhcDevice != nil {
				t.Fatal("mHC seam selection does not follow backend capability")
			}
			L := m.Cfg.NumLayers
			gotPrefill, wantPrefill := ds.Prefill([]int{1, 2}), hs.Prefill([]int{1, 2})
			if !reflect.DeepEqual(gotPrefill, wantPrefill) {
				t.Fatal("device mHC prefill logits differ from the host mHC reference")
			}
			v41GroupedParity(t, gotPrefill, lastLogits(m.Forward([]int{1, 2})), 1e-4)
			// One resident panel for both prompt rows: per layer one attention and
			// one FFN projection split, two post-mixes, and attention, FFN and
			// hidden pre-collapses, each covering every row in one dispatch.
			want := map[string]int{"initial": 1, "rms-split": 2 * L, "pre": 3 * L, "post": 2 * L}
			if !reflect.DeepEqual(device.calls, want) || len(host.calls) != 0 {
				t.Fatalf("prefill mHC dispatches=%v host=%v want=%v", device.calls, host.calls, want)
			}
			if !ds.v41IncrementalEligible() {
				t.Fatal("fixture did not seed incremental state")
			}
			gotStep, wantStep := ds.Step(3), hs.Step(3)
			if !reflect.DeepEqual(gotStep, wantStep) {
				t.Fatal("device mHC step logits differ from the host mHC reference")
			}
			// One incremental row: the head's final collapse adds one pre.
			want = map[string]int{"initial": 2, "rms-split": 4 * L, "pre": 5*L + 1, "post": 4 * L}
			if !reflect.DeepEqual(device.calls, want) || len(device.outputs) != 0 {
				t.Fatalf("step mHC dispatches=%v live outputs=%d want=%v", device.calls, len(device.outputs), want)
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41MHCDeviceFailureClosesSessionWithoutHostReplay(t *testing.T) {
	for _, op := range []string{"initial", "rms-split", "pre", "post"} {
		t.Run(op, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41MHCTestBackend(true)
			s := v41DenseTestSession(t, m, b)
			s.Prefill([]int{1, 2})
			b.failOp, b.failCall = op, b.calls[op]+1
			b.cause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
			before := append([]int(nil), s.v41State().history...)
			func() {
				defer func() {
					r := recover()
					err, ok := r.(error)
					var selected *V41MHCOperationError
					var closed *BackendForwardOperationError
					if !ok || !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, compute.ErrVulkanExecutionFailed) || errors.Is(err, ErrV41ForwardStage) {
						t.Fatalf("step failure = %v, want a selected mHC operation error and no stage sentinel", r)
					}
					if !strings.HasPrefix(closed.Path, "v41-mhc-") || closed.Forward != ForwardPathKind("deepseek41") {
						t.Fatalf("closed failure path=%q forward=%q", closed.Path, closed.Forward)
					}
				}()
				s.Step(3)
			}()
			if !s.BackendSessionClosed() || !reflect.DeepEqual(s.v41Forward.history, before) {
				t.Fatalf("failed step left session open=%t or advanced history %v -> %v", !s.BackendSessionClosed(), before, s.v41Forward.history)
			}
			// No host replay: the full-history fallback would have re-dispatched.
			if b.calls[op] != b.failCall {
				t.Fatalf("%s dispatched %d times after the failure at call %d", op, b.calls[op], b.failCall)
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41MHCDeviceOnlyRefusesWithoutDeviceDispatch(t *testing.T) {
	t.Run("missing capability", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41MHCTestBackend(false)
		s := v41DenseTestSession(t, m, b)
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		// Earlier device-only seams (the embedding gather) may refuse first in a
		// whole forward, so exercise the bound mHC seam directly.
		d := s.v41State().mhcDevice
		if d == nil || d.refusal == nil {
			t.Fatal("device-only session without the capability did not bind a refusing mHC seam")
		}
		_, err := v41FullInitialStreamsOn(d, make([]float32, m.Cfg.HiddenSize))
		var selected *V41MHCOperationError
		if !errors.As(err, &selected) || !errors.Is(err, errV41MHCDeviceUnavailable) || errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("device-only mHC op = %v, want the typed mHC refusal", err)
		}
		if !s.HostFallbackObserved() || !s.BackendSessionClosed() || len(b.calls) != 0 {
			t.Fatalf("refusal observed=%t closed=%t calls=%v", s.HostFallbackObserved(), s.BackendSessionClosed(), b.calls)
		}
	})
	t.Run("reduced geometry", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		if full, err := v41ForwardGeometry(m.Cfg); err != nil || full {
			t.Fatalf("fixture is not the reduced geometry: %v", err)
		}
		b := newV41MHCTestBackend(true)
		s := v41DenseTestSession(t, m, b)
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		_, err := m.forwardV41([]int{1}, s.v41State())
		if !errors.Is(err, errV41MHCDeviceUnavailable) || !s.BackendSessionClosed() || len(b.calls) != 0 {
			t.Fatalf("reduced device-only forward = %v closed=%t calls=%v", err, s.BackendSessionClosed(), b.calls)
		}
	})
	t.Run("portable reduced keeps host", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41MHCTestBackend(true)
		s := v41DenseTestSession(t, m, b)
		if _, err := m.forwardV41([]int{1}, s.v41State()); err != nil || len(b.calls) != 0 || s.BackendSessionClosed() {
			t.Fatalf("portable reduced forward err=%v calls=%v", err, b.calls)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyInventoryDropsDeviceMHCOps(t *testing.T) {
	for _, op := range v41DeviceOnlyHostOps {
		switch op.name {
		case "mhc-initial-streams", "mhc-sinkhorn-split", "mhc-pre-collapse", "mhc-post-mix":
			t.Fatalf("%s has a device seam (v41MHCDevice) but is still inventoried as host", op.name)
		}
	}
}

// fak-test:runtime fast est=2s lane=default
func TestV41ForwardStateBindsExpertSumAndRouterSeams(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	device := v41DenseTestSession(t, m, newV41MHCTestBackend(true)).v41State()
	if device.routedSum == nil || device.sharedAdd == nil {
		t.Fatal("device-memory session did not bind the routed-sum and shared-add seams")
	}
	if device.routerTopK != nil {
		t.Fatal("router top-k bound on a backend without the V4.1 router primitive")
	}
	host := v41FullStepPatchedNorms(t).NewSession()
	defer host.Close()
	if st := host.v41State(); st.routedSum != nil || st.sharedAdd != nil || st.routerTopK != nil {
		t.Fatal("host session bound a device expert seam")
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41MHCResidentPanelCarriesEngramLayers(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		m, _ := v41IncrementalEngramFixture(t, compressed)
		oracle, _ := v41IncrementalEngramFixture(t, compressed)
		if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full || len(m.Cfg.DeepSeekV41.EngramLayerIDs) == 0 {
			t.Fatalf("fixture is not a full-geometry Engram model: %v", err)
		}
		b := newV41MHCTestBackend(true)
		device := v41DenseTestSession(t, m, b)
		host := oracle.NewSession()
		defer host.Close()
		prefix, next := v41IncrementalPrefix(m.Cfg, 4)
		assertV41RowsClose(t, "resident Engram prefill", device.Prefill(prefix), host.Prefill(prefix), 1e-6)
		assertV41RowsClose(t, "resident Engram step", device.Step(next), host.Step(next), 1e-6)
		assertV41RowsClose(t, "resident Engram suffix", device.Prefill([]int{2, 5}), host.Prefill([]int{2, 5}), 1e-6)
		if b.calls["rms-split"] == 0 || b.calls["post"] == 0 || len(b.outputs) != 0 {
			t.Fatalf("compressed=%t resident panel did not carry the forward: calls=%v live=%d", compressed, b.calls, len(b.outputs))
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyInventoryDropsResidentStreamOps(t *testing.T) {
	for _, op := range v41DeviceOnlyHostOps {
		if op.name == "residual-stream-carry" || op.name == "mhc-projection-rms-scale" {
			t.Fatalf("%s runs on the resident device panel (v41MHCResident) but is still inventoried as host", op.name)
		}
	}
}
