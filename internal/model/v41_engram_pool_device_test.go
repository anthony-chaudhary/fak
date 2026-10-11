package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The cpu-ref recorder transcribes the Engram gate/add and the compressor gated
// pool from the reference equations. Its DeviceMemory promise tests routing,
// ownership and failure typing only; numerical device parity is
// TestVulkanV41EngramGate / TestVulkanV41CompressorPool in internal/compute.

func (b *v41MHCTestBackend) SupportsV41EngramGate() bool { return b.engram }

func (b *v41MHCTestBackend) V41EngramGate(streams, projected, norms compute.Tensor, rows, hidden int, eps float32) (compute.Tensor, compute.Tensor, error) {
	H := hidden
	s := b.operand(streams, rows, 4, H)
	projShape := []int{rows, 5 * H}
	if rows == 1 {
		projShape = []int{5 * H}
	}
	p, n := b.operand(projected, projShape...), b.operand(norms, 8*H)
	if b.engramCause != nil {
		b.calls["engram"]++
		return compute.Tensor{}, compute.Tensor{}, b.engramCause
	}
	next := make([]float32, rows*4*H)
	status := make([]float32, rows*4)
	for r := 0; r < rows; r++ {
		value := make([]float32, H)
		for i := range value {
			value[i] = v41BF16(p[r*5*H+4*H+i])
		}
		for h := 0; h < 4; h++ {
			row := s[(r*4+h)*H : (r*4+h+1)*H]
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key := v41BF16(p[r*5*H+h*H+i])
				if !finite32(key) || !finite32(value[i]) {
					status[r*4+h] = 1
				}
				h2 += row[i] * row[i]
				k2 += key * key
				dotSum += row[i] * (n[h*H+i] * n[4*H+h*H+i]) * key
			}
			dot := dotSum * rsqrtf(h2/float32(H)+eps) * rsqrtf(k2/float32(H)+eps) * rsqrtf(float32(H))
			gate := float32(1) / (1 + expf(-copysignf(sqrtf(maxf(absf(dot), 1e-6)), dot)))
			for i := 0; i < H; i++ {
				next[(r*4+h)*H+i] = v41BF16(float32(row[i] + float32(gate*value[i])))
			}
		}
	}
	if b.engramFlag {
		status[len(status)-1] = 1
	}
	nt, err := b.publish("engram", []int{rows, 4, H}, next)
	if err != nil {
		return compute.Tensor{}, compute.Tensor{}, err
	}
	st := compute.NewF32(b, []int{rows, 4}, status)
	b.outputs[st.Buf()] = "engram-status"
	return nt, st, nil
}

func (b *v41MHCTestBackend) SupportsV41CompressorPool() bool { return b.pool }

func (b *v41MHCTestBackend) V41CompressorPool(kv, scores compute.Tensor, ratio, width int) (compute.Tensor, error) {
	k, sc := b.operand(kv, ratio, width), b.operand(scores, ratio, width)
	if b.poolCause != nil {
		b.calls["pool"]++
		return compute.Tensor{}, b.poolCause
	}
	out := make([]float32, width)
	for d := 0; d < width; d++ {
		m := sc[d]
		for t := 1; t < ratio; t++ {
			m = max(m, sc[t*width+d])
		}
		var denom float32
		for t := 0; t < ratio; t++ {
			denom += float32(math.Exp(float64(sc[t*width+d] - m)))
		}
		var acc float32
		for t := 0; t < ratio; t++ {
			w := float32(math.Exp(float64(sc[t*width+d]-m))) / denom
			acc += k[t*width+d] * w
		}
		out[d] = acc
	}
	return b.publish("pool", []int{width}, out)
}

func v41EngramPoolBackend(engram, pool bool) *v41MHCTestBackend {
	b := newV41MHCTestBackend(true)
	b.engram, b.pool = engram, pool
	return b
}

// fak-test:runtime medium est=10s lane=default
func TestV41EngramGateKeepsTheResidentPanelOnDevice(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		m, _ := v41IncrementalEngramFixture(t, compressed)
		oracle, _ := v41IncrementalEngramFixture(t, compressed)
		control, _ := v41IncrementalEngramFixture(t, compressed)
		b, hostTrip := v41EngramPoolBackend(true, true), v41EngramPoolBackend(false, false)
		device, roundTrip := v41DenseTestSession(t, m, b), v41DenseTestSession(t, control, hostTrip)
		prefix, next := v41IncrementalPrefix(m.Cfg, 4)
		for _, s := range []*Session{device, roundTrip} {
			host := oracle.NewSession()
			assertV41RowsClose(t, "Engram prefill", s.Prefill(prefix), host.Prefill(prefix), 1e-6)
			assertV41RowsClose(t, "Engram step", s.Step(next), host.Step(next), 1e-6)
			assertV41RowsClose(t, "Engram suffix", s.Prefill([]int{2, 5}), host.Prefill([]int{2, 5}), 1e-6)
			host.Close()
		}
		// Every Engram injection the gate-less control round-trips through the
		// host is one gate dispatch here, and the panel is never republished.
		if hostTrip.calls["engram"] != 0 || hostTrip.streamUploads < 2*len(m.Cfg.DeepSeekV41.EngramLayerIDs) {
			t.Fatalf("control without the gate: calls=%d republications=%d, want the host round trip", hostTrip.calls["engram"], hostTrip.streamUploads)
		}
		if b.calls["engram"] != hostTrip.streamUploads || b.streamUploads != 0 || len(b.outputs) != 0 {
			t.Fatalf("compressed=%t gate calls=%d (want %d) panel republications=%d live=%d",
				compressed, b.calls["engram"], hostTrip.streamUploads, b.streamUploads, len(b.outputs))
		}
		if (b.calls["pool"] > 0) != compressed || hostTrip.calls["pool"] != 0 {
			t.Fatalf("compressed=%t pool dispatches=%d control=%d", compressed, b.calls["pool"], hostTrip.calls["pool"])
		}
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41EngramGateFailureClosesSessionWithoutHostReplay(t *testing.T) {
	for _, tc := range []struct {
		name  string
		arm   func(*v41MHCTestBackend)
		cause error
	}{
		{"backend fault", func(b *v41MHCTestBackend) {
			b.engramCause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
		}, compute.ErrVulkanExecutionFailed},
		{"non-finite BF16 projection", func(b *v41MHCTestBackend) { b.engramFlag = true }, errV41ProjectionResult},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := v41IncrementalEngramFixture(t, false)
			b := v41EngramPoolBackend(true, false)
			s := v41DenseTestSession(t, m, b)
			prefix, next := v41IncrementalPrefix(m.Cfg, 4)
			s.Prefill(prefix)
			tc.arm(b)
			calls, before := b.calls["engram"], append([]int(nil), s.v41State().history...)
			func() {
				defer func() {
					err, _ := recover().(error)
					var projection *V41ProjectionOperationError
					var closed *BackendForwardOperationError
					if !errors.As(err, &projection) || projection.Leaf != "engram_kv.weight" || !errors.As(err, &closed) ||
						closed.Path != "v41-engram-gate" || !errors.Is(err, tc.cause) {
						t.Fatalf("step failure = %v, want the selected Engram gate failure", err)
					}
				}()
				s.Step(next)
			}()
			if !s.BackendSessionClosed() || !slices.Equal(s.v41Forward.history, before) {
				t.Fatalf("failed step left the session open=%t or advanced history", !s.BackendSessionClosed())
			}
			if b.calls["engram"] != calls+1 || b.streamUploads != 0 {
				t.Fatalf("gate dispatched %d times after the failure, panel republications=%d; want no host replay", b.calls["engram"]-calls, b.streamUploads)
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41CompressorPoolFailureClosesSessionWithoutHostReplay(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, true)
	b := v41EngramPoolBackend(true, true)
	b.poolCause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	s := v41DenseTestSession(t, m, b)
	_, err := m.forwardV41([]int{1, 2}, s.v41State())
	var projection *V41ProjectionOperationError
	var closed *BackendForwardOperationError
	if !errors.As(err, &projection) || projection.Leaf != "attn.compressor.wgate.weight" || !errors.As(err, &closed) ||
		closed.Path != "v41-compressor-pool" || !errors.Is(err, compute.ErrVulkanExecutionFailed) {
		t.Fatalf("prefill = %v, want the selected compressor pool failure", err)
	}
	if !s.BackendSessionClosed() || b.calls["pool"] != 1 {
		t.Fatalf("closed=%t pool dispatches=%d, want one selected dispatch and a closed session", s.BackendSessionClosed(), b.calls["pool"])
	}
}

// fak-test:runtime fast est=2s lane=default
func TestV41DeviceOnlyBindsEngramGateAndCompressorPool(t *testing.T) {
	for _, op := range v41DeviceOnlyHostOps {
		if op.name == "engram-stream-injection" || op.name == "compressor-gated-pool" {
			t.Fatalf("%s runs on its device seam but is still statically inventoried as host", op.name)
		}
	}
	names := func(s *Session) []string { return v41HostOpNames(s.v41DeviceOnlyUnboundOps()) }
	m, _ := v41IncrementalEngramFixture(t, true)
	bound := v41DenseTestSession(t, m, v41EngramPoolBackend(true, true))
	unbound := v41DenseTestSession(t, m, v41EngramPoolBackend(false, false))
	for _, op := range []string{"engram-stream-injection", "compressor-gated-pool"} {
		if slices.Contains(names(bound), op) || !slices.Contains(names(unbound), op) {
			t.Fatalf("%s: bound session unbound=%v, unbound session unbound=%v", op, names(bound), names(unbound))
		}
	}
	plain, _ := v41IncrementalEngramFixture(t, false)
	plain.Cfg.DeepSeekV41.EngramLayerIDs = nil
	if got := names(v41DenseTestSession(t, plain, v41EngramPoolBackend(false, false))); slices.Contains(got, "engram-stream-injection") || slices.Contains(got, "compressor-gated-pool") {
		t.Fatalf("model without Engram layers or compressed KV names unused ops: %v", got)
	}

	s := v41DenseTestSession(t, m, v41EngramPoolBackend(false, false))
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	ops := s.v41State().gatherOps
	for name, call := range map[string]func() error{
		"engram-stream-injection": func() error {
			_, err := ops.engramGateFunc()(0, nil, nil, 1, v41EngramGeometry{}, 1e-6)
			return err
		},
		"compressor-gated-pool": func() error {
			_, err := ops.compressorPoolFunc()(0, nil, nil, 2, 1)
			return err
		},
	} {
		err := call()
		var closed *BackendForwardOperationError
		if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Stage != name+": backend lacks the device operation" {
			t.Fatalf("%s under device-only without the op = %v, want the typed missing-op refusal", name, err)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CompressorPoolRoutesCompletedGroups(t *testing.T) {
	const ratio, width = 3, 2
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	var seen [][]float32
	pool.bindGatedPool(7, func(layer int, kv, scores []float32, r, w int) ([]float32, error) {
		if layer != 7 || r != ratio || w != width {
			t.Fatalf("pool bound layer=%d ratio=%d width=%d", layer, r, w)
		}
		seen = append(seen, append(append([]float32(nil), kv...), scores...))
		return []float32{float32(len(seen)), -1}, nil
	})
	for pos := 0; pos < 2*ratio; pos++ {
		out, emitted, err := pool.Push(pos, []float32{float32(pos), 10}, []float32{0.5, float32(-pos)})
		if err != nil || emitted != (pos%ratio == ratio-1) {
			t.Fatalf("pos %d emitted=%t err=%v", pos, emitted, err)
		}
		if emitted && (out[0] != float32(len(seen)) || out[1] != -1) {
			t.Fatalf("pos %d pooled %v, want the seam result", pos, out)
		}
	}
	want := [][]float32{{0, 10, 1, 10, 2, 10, 0.5, 0, 0.5, -1, 0.5, -2}, {3, 10, 4, 10, 5, 10, 0.5, -3, 0.5, -4, 0.5, -5}}
	if len(seen) != 2 || !slices.Equal(seen[0], want[0]) || !slices.Equal(seen[1], want[1]) {
		t.Fatalf("groups handed to the seam = %v, want token-ordered kv then scores %v", seen, want)
	}
	one, _ := NewV41CompressorPool(1, width)
	one.bindGatedPool(0, func(int, []float32, []float32, int, int) ([]float32, error) {
		t.Fatal("ratio-one pool has no gate to route")
		return nil, nil
	})
	if out, emitted, err := one.Push(0, []float32{1, 2}, nil); err != nil || !emitted || !slices.Equal(out, []float32{1, 2}) {
		t.Fatalf("ratio-one push = %v %t %v", out, emitted, err)
	}
	fault := errors.New("pool fault")
	failing, _ := NewV41CompressorPool(2, 1)
	failing.bindGatedPool(0, func(int, []float32, []float32, int, int) ([]float32, error) { return nil, fault })
	failing.Push(0, []float32{1}, []float32{0})
	if _, emitted, err := failing.Push(1, []float32{2}, []float32{0}); !errors.Is(err, fault) || emitted {
		t.Fatalf("failing seam push emitted=%t err=%v, want the seam error", emitted, err)
	}
}
