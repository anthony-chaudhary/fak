package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The storage and arithmetic are cpu-ref-backed. These tests witness the real
// Session wiring and lifetime, not execution of a physical Vulkan command.
type v41AttentionBF16Recorder struct {
	*v41DenseTestBackend
	live                                                  map[compute.Buffer]bool
	sites                                                 []uint32
	inputs                                                [][]float32
	borrowed                                              []float32
	unavailable                                           bool
	calls, uploads, reads, frees, closeCalls, cleanupHits int
	faultAt                                               int
	foreignReads, foreignFrees                            int
	armed                                                 bool
	fault                                                 error
	panicFault                                            error
	cleanup, closeFault                                   error
	mode                                                  string
	foreign                                               compute.Tensor
}

func newV41AttentionBF16Recorder() *v41AttentionBF16Recorder {
	return &v41AttentionBF16Recorder{v41DenseTestBackend: newV41DenseTestBackend(), live: map[compute.Buffer]bool{}}
}
func (b *v41AttentionBF16Recorder) SupportsV41AttentionBF16(n int) bool {
	return !b.unavailable && n > 0
}
func (b *v41AttentionBF16Recorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 attention BF16 input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("attention publication upload contract")
	}
	b.uploads++
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.live[out.Buf()] = true
	return out
}
func (b *v41AttentionBF16Recorder) V41AttentionBF16(x compute.Tensor, site uint32) (compute.Tensor, error) {
	b.calls++
	b.sites = append(b.sites, site)
	if x.Backend() != b || len(x.Shape) != 1 || (site != 3 && site != 4) {
		panic("attention publication owner/site")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	b.inputs = append(b.inputs, append([]float32(nil), values...))
	hit := b.calls == b.faultAt
	if hit {
		b.armed = true
	}
	if hit && b.panicFault != nil {
		panic(b.panicFault)
	}
	if hit && b.fault != nil && b.mode != "partial" {
		return compute.Tensor{}, b.fault
	}
	if b.mode == "alias" {
		return x, nil
	}
	if b.mode == "foreign" {
		return b.foreign, nil
	}
	for i, v := range values {
		bits := math.Float32bits(v)
		rounded := v41BF16BoundaryOracle(bits)
		if bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000 {
			return compute.Tensor{}, &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	if b.mode == "malformed" {
		values[0] = math.Float32frombits(0x3f800001)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.live[out.Buf()] = true
	if hit {
		return out, b.fault
	}
	return out, nil
}
func (b *v41AttentionBF16Recorder) Read(x compute.Tensor) []float32 {
	if b.foreign.Buf() != nil && x.Buf() == b.foreign.Buf() {
		b.foreignReads++
	}
	if !b.live[x.Buf()] {
		return b.v41DenseTestBackend.Read(x)
	}
	b.reads++
	b.borrowed = append(b.borrowed[:0], b.Backend.Read(x)...)
	return b.borrowed
}
func (b *v41AttentionBF16Recorder) Free(x compute.Tensor) {
	if b.foreign.Buf() != nil && x.Buf() == b.foreign.Buf() {
		b.foreignFrees++
	}
	if !b.live[x.Buf()] {
		b.v41DenseTestBackend.Free(x)
		return
	}
	b.frees++
	delete(b.live, x.Buf())
	b.Backend.Free(x)
	for i := range b.borrowed {
		b.borrowed[i] = 99
	}
	if b.cleanup != nil && (b.armed || b.faultAt == 0) {
		b.cleanupHits++
		panic(b.cleanup)
	}
}
func (b *v41AttentionBF16Recorder) Recycle() {
	b.closeCalls++
	if b.closeFault != nil {
		panic(b.closeFault)
	}
}
func v41AttentionBF16RequireCPU(t *testing.T) {
	t.Helper()
	b := compute.Default()
	if b == nil || b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("attention publication software tests require cpu-ref")
	}
}
func v41AttentionBF16Parity(t *testing.T, a, b []float32) {
	t.Helper()
	for _, r := range [][]float32{a, b} {
		for _, v := range r {
			if !finite32(v) {
				t.Fatal("nonfinite parity control")
			}
		}
	}
	v41GroupedParity(t, a, b, 1e-4)
}
func v41AttentionBF16Session(t *testing.T, m *Model, b *v41AttentionBF16Recorder) *Session {
	t.Helper()
	s := v41DenseTestSession(t, m, b)
	t.Cleanup(func() { b.cleanup = nil; b.closeFault = nil })
	return s
}

// fak-test:runtime medium est=5s lane=default
func TestV41AttentionBF16SessionDispatch(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	for _, role := range []bool{false, true} {
		for _, tokenMajor := range []bool{false, true} {
			t.Run(fmt.Sprintf("role=%t/tokenMajor=%t", role, tokenMajor), func(t *testing.T) {
				previous := v41ForceTokenMajor
				v41ForceTokenMajor = tokenMajor
				defer func() { v41ForceTokenMajor = previous }()
				m := v41FullStepPatchedNorms(t)
				if role {
					m = v41IncrementalExpertFixture(t, true, false)
				}
				b := newV41AttentionBF16Recorder()
				s := v41AttentionBF16Session(t, m, b)
				v41AttentionBF16Parity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})))
				if s.v41State().attentionBF16Func() == nil || b.calls != 2*m.Cfg.NumLayers {
					t.Fatalf("prefill missing output boundaries or republished BF16 inputs: calls=%d", b.calls)
				}
				v41AttentionBF16Parity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})))
				snap, err := s.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				defer snap.Close()
				// A genuinely fresh target has no synthetic v41State/token authority yet.
				target := v41AttentionBF16Session(t, m, b)
				if target.v41Forward != nil {
					t.Fatal("restore destination is not fresh")
				}
				if err := snap.Restore(target); err != nil {
					t.Fatal(err)
				}
				s.Close()
				v41AttentionBF16Parity(t, target.Prefill([]int{4}), lastLogits(m.Forward([]int{1, 2, 3, 4})))
				if b.calls != 4*m.Cfg.NumLayers || target.v41State().callbackOwner != target || target.BackendSessionClosed() {
					t.Fatal("restored suffix lost boundary owner")
				}
				if b.uploads != b.calls || b.reads != b.calls || b.frees != 2*b.calls || len(b.live) != 0 {
					t.Fatal("publication leaked or changed transfer count")
				}
				for i, site := range b.sites {
					if site != compute.V41BF16CheckedAttentionNormOutput {
						t.Fatalf("site order %d=%d", i, site)
					}
				}
			})
		}
	}
	// The existing mHC recorder reaches the distinct resident collapse branch.
	// Its injected publication callback tests that consumer's actual state/scratch
	// routing; the backend interface and captured owner are covered above.
	for _, role := range []bool{false, true} {
		t.Run(fmt.Sprintf("resident-consumer/role=%t", role), func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			if role {
				m = v41IncrementalExpertFixture(t, true, false)
			}
			b := newV41MHCTestBackend(true)
			s := v41DenseTestSession(t, m, b)
			st := s.v41State()
			if st.attentionNormFunc() == nil || st.mhcDevice == nil {
				t.Fatal("resident fixture not bound")
			}
			calls := 0
			st.setAttentionBF16(func(layer int, site string, values []float32) ([]float32, error) {
				want := "attn_norm.weight output"
				if site != want || layer != 0 || len(values) != m.Cfg.HiddenSize {
					t.Fatal("resident publication site/order")
				}
				calls++
				out := make([]float32, len(values))
				for i, v := range values {
					out[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(v)))
					if !finite32(v) || !finite32(out[i]) {
						t.Fatal("invalid resident control")
					}
				}
				return out, nil
			})
			v41AttentionBF16Parity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})))
			v41AttentionBF16Parity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})))
			if calls != 3*m.Cfg.NumLayers || b.calls["rms-split"] == 0 || len(b.outputs) != 0 {
				t.Fatal("resident attention publication path missing or leaked", calls, b.calls)
			}
		})
	}
	t.Run("unavailable-old-route", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41AttentionBF16Recorder()
		b.unavailable = true
		s := v41AttentionBF16Session(t, m, b)
		v41AttentionBF16Parity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})))
		if b.calls != 0 || s.v41State().attentionBF16Func() != nil || s.BackendSessionClosed() {
			t.Fatal("absent boundary capability changed old route")
		}
	})
}

// fak-test:runtime medium est=1s lane=default
func TestV41AttentionBF16ExactBoundary(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	m := v41FullStepPatchedNorms(t)
	b := newV41AttentionBF16Recorder()
	s := v41AttentionBF16Session(t, m, b)
	publish := s.v41AttentionBF16Func()
	if publish == nil {
		t.Fatal("missing ordinary publication")
	}
	values := make([]float32, m.Cfg.HiddenSize)
	for i := range values {
		values[i] = 1
	}
	words := []uint32{0x80000000, 0x00018000, 0x3f808000, 0x3f818000, 0xbf818000, 0x7f7f7fff}
	for i, u := range words {
		values[i] = math.Float32frombits(u)
	}
	original := append([]float32(nil), values...)
	for _, site := range []string{"attn_norm.weight input", "attn_norm.weight output"} {
		got, err := publish(0, site, values)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Float32bits(v) != v41BF16BoundaryOracle(math.Float32bits(original[i])) || math.Float32bits(values[i]) != math.Float32bits(original[i]) {
				t.Fatal("input changed, boundary rounded incorrectly or result borrowed freed storage")
			}
		}
	}
	if math.Float32bits(b.inputs[0][2]) != words[2] {
		t.Fatal("host rounded the input before selected publication")
	}
	normalized := append([]float32(nil), values...)
	normalized[2] = math.Float32frombits(0x3f818001)
	normCalls := 0
	got, err := m.v41AttentionInputNormWithPublications(0, values, float32(m.Cfg.RMSNormEps), func(_ int, in []float32) ([]float32, error) {
		normCalls++
		for i, v := range in {
			if math.Float32bits(v) != v41BF16BoundaryOracle(math.Float32bits(values[i])) {
				t.Fatal("norm consumed unrounded input")
			}
		}
		return normalized, nil
	}, publish)
	if err != nil || normCalls != 1 || math.Float32bits(b.inputs[len(b.inputs)-1][2]) != 0x3f818001 || math.Float32bits(got[2]) != v41BF16BoundaryOracle(0x3f818001) {
		t.Fatal("normalization output was rounded on host or stage order changed", err)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41AttentionBF16SelectedFailure(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	for _, suffix := range []bool{false, true} {
		for _, stage := range []int{1, 2} {
			for _, panicMode := range []bool{false, true} {
				t.Run(fmt.Sprintf("suffix=%t/site=%d/panic=%t", suffix, stage+2, panicMode), func(t *testing.T) {
					m := v41FullStepPatchedNorms(t)
					b := newV41AttentionBF16Recorder()
					s := v41AttentionBF16Session(t, m, b)
					s.Prefill([]int{1, 2})
					before := captureV41ForwardSnapshot(s.v41Forward)
					reads := b.reads
					status := &compute.V41BF16CheckedArithmeticError{Site: uint32(stage + 2), Index: 1}
					var primary error = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage, Recovered: status}
					b.faultAt = b.calls + 1
					b.mode = "partial"
					b.fault = primary
					if panicMode {
						primary = fmt.Errorf("wrapped attention cause: %w", &BackendForwardOperationError{Path: "incoming", Layer: 0, Stage: "status", Cause: status})
						b.fault = nil
						b.panicFault = primary
					}
					b.cleanup = errors.New("secondary attention Free")
					b.closeFault = errors.New("secondary attention Close")
					err := recoverError(func() {
						if stage == 1 {
							// Real session collapses are widened BF16 here, so
							// exercise the still-required input publication with
							// a non-BF16 row through the same graph helper.
							input := make([]float32, m.Cfg.HiddenSize)
							input[0] = math.Float32frombits(0x3f800001)
							out, err := m.v41AttentionInputNormWithPublications(0, input, float32(m.Cfg.RMSNormEps), func(int, []float32) ([]float32, error) {
								t.Fatal("failed input publication reached normalization")
								return nil, nil
							}, s.v41State().attentionBF16Func())
							if out != nil {
								t.Fatal("failed input publication returned a row")
							}
							if err != nil {
								panic(err)
							}
							return
						}
						if suffix {
							s.Prefill([]int{3})
						} else {
							s.Step(3)
						}
					})
					if !errors.Is(err, primary) || !s.BackendSessionClosed() || b.calls != b.faultAt || len(b.live) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
						t.Fatal("selected failure lost cause, leaked, replayed or committed continuation", err)
					}
					wantCleanup := 2
					if panicMode {
						wantCleanup = 1
					}
					if b.cleanupHits != wantCleanup || b.closeCalls != 1 {
						t.Fatal("secondary cleanup/Close controls did not execute", b.cleanupHits, b.closeCalls)
					}
					if b.reads != reads {
						t.Fatal("failed publication was read")
					}
					if panicMode && (err != primary || s.halFailure != primary) {
						t.Fatal("wrapped primary identity replaced")
					}
					if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.calls != b.faultAt {
						t.Fatal("retired publication replayed", retry)
					}
				})
			}
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41AttentionBF16PayloadOwnership(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	for _, mode := range []string{"unknown-site", "nan", "bf16-overflow", "alias", "foreign", "malformed", "cleanup"} {
		t.Run(mode, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41AttentionBF16Recorder()
			s := v41AttentionBF16Session(t, m, b)
			row := make([]float32, m.Cfg.HiddenSize)
			for i := range row {
				row[i] = 1
			}
			site := "attn_norm.weight input"
			switch mode {
			case "unknown-site":
				site = "model.norm.weight input"
			case "nan":
				row[0] = math.Float32frombits(0x7fffffff)
			case "bf16-overflow":
				row[0] = math.Float32frombits(0x7f7f8000)
			case "cleanup":
				b.cleanup = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: errors.New("cleanup-only fault")}
			default:
				b.mode = mode
			}
			if mode == "foreign" {
				b.foreign = compute.NewF32(compute.Default(), []int{len(row)}, append([]float32(nil), row...))
			}
			out, err := s.v41AttentionBF16Func()(0, site, row)
			if err == nil || out != nil || !errors.Is(err, ErrV41BF16BoundaryOperation) || !s.BackendSessionClosed() || len(b.live) != 0 {
				t.Fatal("malformed selected publication escaped", err)
			}
			if mode == "alias" && b.frees != 1 {
				t.Fatal("aliased input freed twice")
			}
			if mode == "foreign" && (b.frees != 1 || b.foreignReads != 0 || b.foreignFrees != 0 || compute.Default().Read(b.foreign)[0] != 1) {
				t.Fatal("foreign tensor was consumed or freed")
			}
		})
	}
}

// The input identity path copies finite widened BF16 without a publication;
// normalization and its output boundary still run, in order. This witnesses
// model routing and owned host bytes, not physical submit counts or a speedup.
// fak-test:runtime medium est=1s lane=default
func TestV41AttentionBF16InputIdentity(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	for _, alreadyBF16 := range []bool{true, false} {
		t.Run(fmt.Sprintf("already-bf16=%t", alreadyBF16), func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41AttentionBF16Recorder()
			s := v41AttentionBF16Session(t, m, b)
			input := make([]float32, m.Cfg.HiddenSize)
			words := []uint32{0x80000000, 0x00010000, 0x3f810000, 0xbf810000, 0x7f7f0000}
			for i := range input {
				input[i] = math.Float32frombits(words[i%len(words)])
			}
			if !alreadyBF16 {
				input[2] = math.Float32frombits(0x3f818001)
			}
			before := append([]float32(nil), input...)
			normalized := make([]float32, len(input))
			for i := range normalized {
				normalized[i] = math.Float32frombits(0x3f818001)
			}
			wantInputs := 0
			if !alreadyBF16 {
				wantInputs = 1
			}
			normCalls := 0
			normalize := func(layer int, row []float32) ([]float32, error) {
				normCalls++
				if layer != 0 || len(row) != len(input) || b.calls != wantInputs {
					t.Fatal("input publication count/order changed")
				}
				for i, value := range row {
					if math.Float32bits(value) != v41BF16BoundaryOracle(math.Float32bits(before[i])) {
						t.Fatal("normalization did not receive exact widened BF16")
					}
				}
				// The callback owns its input copy even when publication is the
				// identity; mutation must not escape to the mHC collapse row.
				row[0] = 7
				return normalized, nil
			}
			var got []float32
			var err error
			hostRounds := v41BF16PublishHostRounds(t, func() {
				got, err = m.v41AttentionInputNormWithPublications(0, input, float32(m.Cfg.RMSNormEps), normalize, s.v41AttentionBF16Func())
			})
			if hostRounds != 0 {
				t.Fatal("identity optimization replayed BF16 arithmetic on the host", hostRounds)
			}
			wantCalls := 2
			if alreadyBF16 {
				wantCalls = 1
			}
			if err != nil || normCalls != 1 || b.calls != wantCalls || b.uploads != wantCalls || b.reads != wantCalls || b.frees != 2*wantCalls || len(b.live) != 0 || s.BackendSessionClosed() {
				t.Fatalf("publication identity routing: calls=%d uploads=%d reads=%d frees=%d norm=%d err=%v", b.calls, b.uploads, b.reads, b.frees, normCalls, err)
			}
			if len(got) != len(input) || b.sites[len(b.sites)-1] != compute.V41BF16CheckedAttentionNormOutput {
				t.Fatal("normalization output publication was skipped")
			}
			for i := range input {
				if math.Float32bits(input[i]) != math.Float32bits(before[i]) || math.Float32bits(got[i]) != v41BF16BoundaryOracle(0x3f818001) {
					t.Fatal("input ownership or output BF16 bits changed")
				}
			}
		})
	}
	for _, word := range []uint32{0x7fc00000, 0x7f800000, 0x7f7f8000} {
		t.Run(fmt.Sprintf("invalid=%08x", word), func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41AttentionBF16Recorder()
			s := v41AttentionBF16Session(t, m, b)
			input := make([]float32, m.Cfg.HiddenSize)
			input[0] = math.Float32frombits(word)
			normCalls := 0
			out, err := m.v41AttentionInputNormWithPublications(0, input, float32(m.Cfg.RMSNormEps), func(int, []float32) ([]float32, error) {
				normCalls++
				return nil, nil
			}, s.v41AttentionBF16Func())
			var boundary *V41BF16BoundaryOperationError
			if out != nil || normCalls != 0 || !errors.As(err, &boundary) || boundary.Site != "attn_norm.weight input" || !s.BackendSessionClosed() || len(b.live) != 0 || math.Float32bits(input[0]) != word {
				t.Fatal("invalid input bypassed its selected boundary or changed ownership", err)
			}
		})
	}
}
