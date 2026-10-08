package model

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// [SW-VERIFIED]: the recorder executes real packed Q2_K MatMul through cpu-ref.
// Only public Backend.Read calls are charged as device-to-host transfers.
type v41ClampedReadBackend struct {
	*v41HalSeamBackend
	reads, readBytes             int
	readAttempts, failReadAt     int
	matmulAttempts, failMatmulAt int
	live                         map[compute.Buffer]bool
	failRead                     bool
	session                      *Session
	roles                        map[compute.Buffer]string
}

func newV41ClampedReadBackend() *v41ClampedReadBackend {
	return &v41ClampedReadBackend{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}, live: map[compute.Buffer]bool{}, roles: map[compute.Buffer]string{}}
}

func (b *v41ClampedReadBackend) expertWeightRole(w compute.Tensor) string {
	if b.session == nil {
		return ""
	}
	for name, weight := range b.session.halW {
		if !strings.HasPrefix(name, "kquant-raw:") || !strings.Contains(name, ".ffn.experts.") || weight.Buf() != w.Buf() {
			continue
		}
		for _, leaf := range []string{"w1.weight", "w3.weight", "w2.weight"} {
			if strings.HasSuffix(name, "."+leaf) {
				return leaf
			}
		}
	}
	return ""
}

func (b *v41ClampedReadBackend) recordActivation(g, u, out compute.Tensor) {
	if b.roles[g.Buf()] == "w1.weight" && b.roles[u.Buf()] == "w3.weight" {
		b.roles[out.Buf()] = "activation"
	}
}
func (b *v41ClampedReadBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	out := b.Backend.Upload(x, dt)
	if dt == compute.F32 && len(out.Shape) == 1 {
		b.live[out.Buf()] = true
	}
	return out
}
func (b *v41ClampedReadBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	role := b.expertWeightRole(w)
	if role == "w1.weight" || role == "w3.weight" {
		b.matmulAttempts++
		if b.matmulAttempts == b.failMatmulAt {
			panic(&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Site: "MatMul", Err: compute.ErrVulkanExecutionFailed})
		}
	}
	out := b.v41HalSeamBackend.MatMul(w, x)
	b.live[out.Buf()] = true
	if role != "" {
		b.roles[out.Buf()] = role
	}
	return out
}
func (b *v41ClampedReadBackend) Read(x compute.Tensor) []float32 {
	role := b.roles[x.Buf()]
	if role == "w1.weight" || role == "w3.weight" || role == "activation" {
		b.readAttempts++
		if b.failRead || b.readAttempts == b.failReadAt {
			panic(&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Site: "Read", Err: compute.ErrVulkanExecutionFailed})
		}
	}
	out := b.Backend.Read(x)
	if role != "" {
		b.reads++
		b.readBytes += 4 * len(out)
	}
	return out
}
func (b *v41ClampedReadBackend) Free(x compute.Tensor) {
	delete(b.live, x.Buf())
	delete(b.roles, x.Buf())
	b.Backend.Free(x)
}
func (b *v41ClampedReadBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	out := b.v41HalSeamBackend.SwiGLU(g, u)
	b.live[out.Buf()] = true
	b.recordActivation(g, u, out)
	return out
}

type v41ClampedLimitBackend struct {
	*v41ClampedReadBackend
	limited        int
	limits         []float32
	failActivation bool
}

func (b *v41ClampedLimitBackend) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	b.limited++
	b.limits = append(b.limits, limit)
	if b.failActivation {
		panic(&compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Site: "SwiGLUWithLimit", Err: compute.ErrVulkanExecutionFailed})
	}
	gate, up := b.Backend.Read(g), b.Backend.Read(u)
	values := make([]float32, len(gate))
	for i := range values {
		// Independent closed form: gate has no lower bound.
		clampedGate := min(gate[i], limit)
		clampedUp := max(-limit, min(up[i], limit))
		values[i] = clampedGate / (1 + float32(math.Exp(float64(-clampedGate)))) * clampedUp
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.live[out.Buf()] = true
	b.recordActivation(g, u, out)
	return out
}

func v41ClampedActivationPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, key := range []string{"expert_activation_device_calls", "expert_activation_host_calls", "expert_activation_readback_bytes", "expert_activation_nanos"} {
		value, ok := phases[phase][key]
		if !ok {
			t.Errorf("default %s attribution missing %s", phase, key)
			continue
		}
		var n float64
		if err := json.Unmarshal(value, &n); err != nil {
			t.Fatal(err)
		}
		out[key] = n
	}
	return out
}

// fak-test:runtime slow est=24s lane=default
func TestV41ClampedDeviceSwiGLUSessionStepAndSuffix(t *testing.T) {
	t.Parallel()
	for _, capable := range []bool{true, false} {
		for _, suffix := range []bool{false, true} {
			name := "step"
			if suffix {
				name = "suffix"
			}
			if !capable {
				name += "/host-activation"
			}
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				m := v41IncrementalExpertFixture(t, false, false)
				m.Cfg.SwigluLimit = 0.01
				b := &v41ClampedLimitBackend{v41ClampedReadBackend: newV41ClampedReadBackend()}
				var backend compute.Backend = b
				if !capable {
					backend = b.v41ClampedReadBackend
				}
				s, err := m.NewBackendSessionChecked(backend)
				if err != nil {
					t.Fatal(err)
				}
				b.session = s
				t.Cleanup(s.Close)
				if len(s.Prefill([]int{1, 2, 3})) == 0 || !s.v41IncrementalEligible() {
					t.Fatal("prefill did not seed incremental session")
				}
				reads, bytes, limited := b.reads, b.readBytes, b.limited
				phase := "decode"
				ids := []int{4}
				if suffix {
					phase, ids = "prefill", []int{4, 5}
				}
				before := v41ClampedActivationPhase(t, m, phase)
				var got []float32
				if suffix {
					got = s.Prefill(ids)
				} else {
					got = s.Step(4)
				}
				picks := len(ids) * m.Cfg.NumLayers * m.Cfg.NumExpertsPerTok
				wantLimited, wantReads := picks, 2*picks
				activationBytes := 4 * picks * m.Cfg.MoEIntermediateSize
				if !capable {
					wantLimited, wantReads = 0, 3*picks
					activationBytes *= 2
				}
				if b.limited-limited != wantLimited {
					t.Errorf("actual session limited activations=%d want %d", b.limited-limited, wantLimited)
				}
				if b.reads-reads != wantReads || b.readBytes-bytes != activationBytes+4*picks*m.Cfg.HiddenSize {
					t.Errorf("activation plus down readbacks=%d/%d bytes, want %d/%d", b.reads-reads, b.readBytes-bytes, wantReads, activationBytes+4*picks*m.Cfg.HiddenSize)
				}
				oracle := v41IncrementalExpertFixture(t, false, false)
				oracle.Cfg.SwigluLimit = m.Cfg.SwigluLimit
				want := oracle.Forward(append([]int{1, 2, 3}, ids...))
				assertV41LogitsClose(t, got, lastLogits(want), "limited activation actual session vs host oracle")
				var maximumDiff float64
				for i, value := range got {
					maximumDiff = max(maximumDiff, math.Abs(float64(value-lastLogits(want)[i])))
				}
				t.Logf("activation_phase=%s capable=%t logits_width=%d maximum_diff=%g", phase, capable, len(got), maximumDiff)
				after := v41ClampedActivationPhase(t, m, phase)
				receipt, err := json.Marshal(after)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("activation_phase=%s tokens=%d attribution=%s", phase, len(ids), receipt)
				if after["expert_activation_device_calls"]-before["expert_activation_device_calls"] != float64(wantLimited) || after["expert_activation_host_calls"]-before["expert_activation_host_calls"] != float64(picks-wantLimited) {
					t.Errorf("selected stage counters before=%v after=%v", before, after)
				}
				if after["expert_activation_readback_bytes"]-before["expert_activation_readback_bytes"] != float64(activationBytes) {
					t.Error("activation attribution included projections/down rather than single activation row")
				}
				if after["expert_activation_nanos"] <= before["expert_activation_nanos"] {
					t.Error("selected activation stage elapsed absent")
				}
			})
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41ClampedDeviceSwiGLUWidth2048AndLegacyFallback(t *testing.T) {
	t.Parallel()
	const H, I = 256, 2048
	gateName, upName := layerName(0, "ffn.experts.0.w1.weight"), layerName(0, "ffn.experts.0.w3.weight")
	m := &Model{Cfg: Config{HiddenSize: H, MoEIntermediateSize: I, SwigluLimit: 2}, kqw: map[string]*kQuantTensor{
		gateName: q2kFixtureTensor(I, H, 0x13358), upName: q2kFixtureTensor(I, H, 0x13359),
	}}
	w1, ok := m.residentF32Mat(gateName)
	if !ok {
		t.Fatal("missing gate oracle")
	}
	w3, ok := m.residentF32Mat(upName)
	if !ok {
		t.Fatal("missing up oracle")
	}
	var largest float32
	for _, weights := range [][]float32{w1, w3} {
		for row := 0; row < I; row++ {
			var sum float32
			for _, v := range weights[row*H : (row+1)*H] {
				sum += v
			}
			largest = max(largest, abs32(sum))
		}
	}
	if largest == 0 {
		t.Fatal("vacuous quantized fixture")
	}
	for _, sign := range []float32{1, -1} {
		x := make([]float32, H)
		for i := range x {
			x[i] = sign * 5 / largest
		}
		want := make([]float32, I)
		var positiveGate, negativeGate, positiveUp, negativeUp bool
		for row := range want {
			var gate, up float32
			for col, v := range x {
				gate += w1[row*H+col] * v
				up += w3[row*H+col] * v
			}
			positiveGate = positiveGate || gate > 2
			negativeGate = negativeGate || gate < -2
			positiveUp = positiveUp || up > 2
			negativeUp = negativeUp || up < -2
			gate = min(gate, 2)
			up = max(float32(-2), min(up, 2))
			want[row] = gate / (1 + float32(math.Exp(float64(-gate)))) * up
		}
		if (sign > 0 && (!positiveGate || !positiveUp)) || (sign < 0 && (!negativeGate || !negativeUp)) {
			t.Fatal("clamp saturation vacuous")
		}
		for _, capable := range []bool{true, false} {
			recorder := newV41ClampedReadBackend()
			limited := &v41ClampedLimitBackend{v41ClampedReadBackend: recorder}
			var be compute.Backend = recorder
			if capable {
				be = limited
			}
			s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
			recorder.session = s
			got, outcome, err := s.v41ExpertGateUpFunc()(0, "ffn.experts.0", x)
			if err != nil || outcome != v41GateUpHandled {
				t.Fatalf("callback outcome=%v error=%v", outcome, err)
			}
			assertV41LogitsClose(t, got, want, "2048-wide clamp vs independent oracle")
			wantReads := 2
			if capable {
				wantReads = 1
				if limited.limited != 1 || len(limited.limits) != 1 || limited.limits[0] != 2 {
					t.Error("configured finite limit did not select optional backend")
				}
			}
			if recorder.reads != wantReads || recorder.readBytes != 4*I*wantReads {
				t.Errorf("readbacks=%d/%d bytes want=%d/%d", recorder.reads, recorder.readBytes, wantReads, 4*I*wantReads)
			}
			if len(recorder.live) != 0 {
				t.Errorf("activation leaked %d transient buffers", len(recorder.live))
			}
			for _, name := range []string{gateName, upName} {
				if m.has(name) || s.halW["kquant-raw:"+name].Dtype != compute.Q2_K {
					t.Error("compressed gate/up expanded or not staged")
				}
			}
			s.Close()
		}
	}
	// A configured nonpositive limit retains the historical device SwiGLU path.
	for _, limit := range []float64{0, -2, math.NaN(), math.Inf(1)} {
		m.Cfg.SwigluLimit = limit
		b := &v41ClampedLimitBackend{v41ClampedReadBackend: newV41ClampedReadBackend()}
		s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
		b.session = s
		_, outcome, err := s.v41ExpertGateUpFunc()(0, "ffn.experts.0", make([]float32, H))
		wantSwiGLU, wantReads := 1, 1
		if math.IsInf(limit, 1) {
			wantSwiGLU, wantReads = 0, 2
		}
		if err != nil || outcome != v41GateUpHandled || b.limited != 0 || b.swiglu != wantSwiGLU || b.reads != wantReads {
			t.Errorf("legacy limit %g changed route: limited=%d legacy=%d reads=%d error=%v", limit, b.limited, b.swiglu, b.reads, err)
		}
		s.Close()
	}
}

// fak-test:runtime medium est=14s lane=default
func TestV41ClampedDeviceSwiGLUFailureFreesAndRetries(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"activation", "read", "second-matmul", "partial-host-read"} {
		t.Run(site, func(t *testing.T) {
			t.Parallel()
			m := v41IncrementalExpertFixture(t, false, false)
			m.Cfg.SwigluLimit = 0.01
			b := &v41ClampedLimitBackend{v41ClampedReadBackend: newV41ClampedReadBackend()}
			var selected compute.Backend = b
			if site == "partial-host-read" {
				selected = b.v41ClampedReadBackend
			}
			s, err := m.NewBackendSessionChecked(selected)
			if err != nil {
				t.Fatal(err)
			}
			b.session = s
			t.Cleanup(s.Close)
			s.Prefill([]int{1, 2, 3})
			b.live = map[compute.Buffer]bool{}
			before := captureV41ForwardSnapshot(s.v41Forward)
			calls := b.limited
			phaseBefore := v41ClampedActivationPhase(t, m, "decode")
			b.failActivation = site == "activation"
			b.failRead = site == "read"
			if site == "partial-host-read" {
				b.failReadAt = b.readAttempts + 2
			}
			if site == "second-matmul" {
				b.failMatmulAt = b.matmulAttempts + 2
			}
			err = panicAsError(func() { s.Step(4) })
			var backend *compute.BackendError
			if !errors.Is(err, compute.ErrVulkanExecutionFailed) || !errors.As(err, &backend) {
				t.Errorf("selected failure lost typed backend cause: %v", err)
			}
			wantCalls := 1
			if site == "second-matmul" || site == "partial-host-read" {
				wantCalls = 0
			}
			if b.limited-calls != wantCalls {
				t.Errorf("failure retried selected activation %d times", b.limited-calls)
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("failed activation changed retained state")
			}
			if len(b.live) != 0 {
				t.Errorf("failed activation leaked %d transient buffers", len(b.live))
			}
			phaseAfter := v41ClampedActivationPhase(t, m, "decode")
			if site == "partial-host-read" {
				if phaseAfter["expert_activation_readback_bytes"]-phaseBefore["expert_activation_readback_bytes"] != float64(4*m.Cfg.MoEIntermediateSize) {
					t.Error("successful first host read bytes lost when second read failed")
				}
				if phaseAfter["expert_activation_host_calls"] != phaseBefore["expert_activation_host_calls"] || phaseAfter["expert_activation_device_calls"] != phaseBefore["expert_activation_device_calls"] {
					t.Error("partial activation failure counted as successful activation")
				}
				if phaseAfter["expert_activation_nanos"] <= phaseBefore["expert_activation_nanos"] {
					t.Error("failed selected host activation elapsed absent")
				}
			}
			b.failActivation, b.failRead = false, false
			b.failReadAt, b.failMatmulAt = 0, 0
			got := s.Step(4)
			oracle := v41IncrementalExpertFixture(t, false, false)
			oracle.Cfg.SwigluLimit = m.Cfg.SwigluLimit
			assertV41LogitsClose(t, got, lastLogits(oracle.Forward([]int{1, 2, 3, 4})), "retry after selected activation failure")
		})
	}
}
