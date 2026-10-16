package model

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder keeps the existing independent mHC transcription and cpu-ref
// row reductions, but gives all transient tensors the selected owner's identity.
// Its dispatch counts are software evidence, not hardware throughput.
type v41NormPanelRecorder struct {
	*v41MHCTestBackend
	weight                    func() compute.Tensor
	shapes                    [][]int
	normOutputs               map[compute.Buffer]bool
	transient                 map[compute.Buffer]bool
	borrowed                  []float32
	mode                      string
	capsHook, publicationHook func()
	fault                     error
	reads, publications       int
	publicationBeforeNormFree bool
}

func newV41NormPanelRecorder() *v41NormPanelRecorder {
	return &v41NormPanelRecorder{v41MHCTestBackend: newV41MHCTestBackend(true), normOutputs: map[compute.Buffer]bool{}, transient: map[compute.Buffer]bool{}}
}
func (b *v41NormPanelRecorder) owned(shape []int, values []float32, transient bool) compute.Tensor {
	x := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	if transient {
		b.transient[x.Buf()] = true
	}
	return x
}
func (b *v41NormPanelRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("norm panel fixture requires F32")
	}
	return b.owned(x.Shape, b.Backend.Read(x), strings.HasPrefix(site, "V4.1 RMSNorm activation ") || site == "V4.1 attention BF16 input")
}
func (b *v41NormPanelRecorder) RMSNorm(x, w compute.Tensor, eps float32) compute.Tensor {
	attention := b.weight != nil && w.Buf() == b.weight().Buf()
	if attention {
		b.shapes = append(b.shapes, append([]int(nil), x.Shape...))
		switch b.mode {
		case "dispatch":
			panic(b.fault)
		case "dispatch nil":
			panic(nil)
		case "dispatch Goexit":
			runtime.Goexit()
		}
	}
	raw := b.Backend.RMSNorm(x, w, eps)
	values := append([]float32(nil), b.Backend.Read(raw)...)
	b.Backend.Free(raw)
	if attention && len(x.Shape) == 2 && len(values) > x.Shape[1] {
		switch b.mode {
		case "publication-first":
			values[0] = math.MaxFloat32
			values[x.Shape[1]] = float32(math.NaN())
		case "norm-first":
			values[0] = float32(math.NaN())
			values[x.Shape[1]] = math.MaxFloat32
		}
	}
	y := b.owned(x.Shape, values, true)
	if attention {
		b.normOutputs[y.Buf()] = true
		if b.mode == "shape" {
			y.Shape = []int{len(values)}
		}
	}
	return y
}
func (b *v41NormPanelRecorder) Read(x compute.Tensor) []float32 {
	if b.normOutputs[x.Buf()] {
		b.reads++
		if b.mode == "read" {
			panic(b.fault)
		}
		b.borrowed = append(b.borrowed[:0], b.Backend.Read(x)...)
		return b.borrowed
	}
	return b.v41MHCTestBackend.Read(x)
}
func (b *v41NormPanelRecorder) Free(x compute.Tensor) {
	norm := b.normOutputs[x.Buf()]
	delete(b.normOutputs, x.Buf())
	delete(b.transient, x.Buf())
	b.v41MHCTestBackend.Free(x)
	if norm {
		for i := range b.borrowed {
			b.borrowed[i] = float32(math.NaN())
		}
		switch b.mode {
		case "free":
			panic(b.fault)
		case "free nil":
			panic(nil)
		case "free Goexit":
			runtime.Goexit()
		}
	}
}
func (b *v41NormPanelRecorder) Caps() compute.Caps {
	if b.capsHook != nil {
		b.capsHook()
	}
	return b.v41MHCTestBackend.Caps()
}
func (b *v41NormPanelRecorder) SupportsV41AttentionBF16(n int) bool {
	if b.publicationHook != nil {
		b.publicationHook()
	}
	return n > 0
}
func (b *v41NormPanelRecorder) V41AttentionBF16(x compute.Tensor, site uint32) (compute.Tensor, error) {
	b.publications++
	b.publicationBeforeNormFree = b.publicationBeforeNormFree || len(b.normOutputs) != 0
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i, v := range values {
		bits := v41BF16BoundaryOracle(math.Float32bits(v))
		if bits&0x7f800000 == 0x7f800000 {
			return compute.Tensor{}, &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(bits)
	}
	return b.owned(x.Shape, values, true), nil
}
func (b *v41NormPanelRecorder) operand(x compute.Tensor) compute.Tensor {
	return compute.NewF32(b.v41MHCTestBackend, x.Shape, b.Backend.Read(x))
}
func (b *v41NormPanelRecorder) result(x compute.Tensor, err error) (compute.Tensor, error) {
	if err != nil {
		return compute.Tensor{}, err
	}
	out := b.owned(x.Shape, b.Backend.Read(x), false)
	b.v41MHCTestBackend.Free(x)
	return out, nil
}
func (b *v41NormPanelRecorder) V41MHCInitialStreams(x compute.Tensor, r, h int) (compute.Tensor, error) {
	return b.result(b.v41MHCTestBackend.V41MHCInitialStreams(b.operand(x), r, h))
}
func (b *v41NormPanelRecorder) V41MHCSplit(m, s, a compute.Tensor, r, i int, e float32) (compute.Tensor, error) {
	return b.result(b.v41MHCTestBackend.V41MHCSplit(b.operand(m), b.operand(s), b.operand(a), r, i, e))
}
func (b *v41NormPanelRecorder) V41MHCRMSSplit(m, s, p compute.Tensor, r, h, i int, e, re float32) (compute.Tensor, error) {
	return b.result(b.v41MHCTestBackend.V41MHCRMSSplit(b.operand(m), b.operand(s), b.operand(p), r, h, i, e, re))
}
func (b *v41NormPanelRecorder) V41MHCPre(s, k compute.Tensor, r, h int) (compute.Tensor, error) {
	return b.result(b.v41MHCTestBackend.V41MHCPre(b.operand(s), b.operand(k), r, h))
}
func (b *v41NormPanelRecorder) V41MHCPost(x, s, k compute.Tensor, r, h int) (compute.Tensor, error) {
	return b.result(b.v41MHCTestBackend.V41MHCPost(b.operand(x), b.operand(s), b.operand(k), r, h))
}

// Refs #12640. Bounded cold panels preserve scalar values and logical row-first
// numerical faults. Physical dispatch/transport timing is intentionally batched.
// fak-test:runtime medium est=5s lane=default
func TestV41AttentionNormColdPanel(t *testing.T) {
	if ref := compute.Default(); ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("requires cpu-ref")
	}
	fixture := func(t *testing.T) (*Model, *Session, *v41NormPanelRecorder) {
		t.Helper()
		m := v41FullStepPatchedNorms(t)
		b := newV41NormPanelRecorder()
		s := v41DenseTestSession(t, m, b)
		b.weight = func() compute.Tensor { return s.halW[layerName(0, "attn_norm.weight")] }
		st := s.v41State()
		if st.attentionNormBinding == nil || st.attentionNormBinding.native == nil {
			t.Fatal("canonical binding missing")
		}
		return m, s, b
	}
	panel := func(m *Model, s *Session, n int) ([][]float32, bool, error) {
		st := s.v41State()
		rows := make([][]float32, n)
		for r := range rows {
			rows[r] = make([]float32, m.Cfg.HiddenSize)
			for i := range rows[r] {
				rows[r][i] = float32((r+i)%11-5) / 8
			}
		}
		return st.attentionNormBinding.coldPanel(m, st, &v41ProjScratch{attentionNormBinding: st.attentionNormBinding}, &v41MHCResident{d: st.mhcDevice, rows: n}, 0, rows, float32(m.Cfg.RMSNormEps))
	}
	t.Run("captured publication provenance", func(t *testing.T) {
		_, s, original := fixture(t)
		other := newV41NormPanelRecorder()
		original.capsHook = func() { original.capsHook = nil; s.Backend = other }
		other.publicationHook = func() { s.Backend = original }
		binding := s.v41BindAttentionNorm()
		if s.Backend != original || binding == nil || binding.normalize == nil || binding.publish == nil || binding.native != nil {
			t.Fatal("admission mutation mislabeled a foreign captured publication")
		}
	})
	t.Run("bounded rows", func(t *testing.T) {
		m, s, b := fixture(t)
		const rows = 65
		got, handled, err := panel(m, s, rows)
		if err != nil || !handled || len(got) != rows || !reflect.DeepEqual(b.shapes, [][]int{{64, m.Cfg.HiddenSize}, {1, m.Cfg.HiddenSize}}) || b.reads != 2 || b.publications != rows || len(b.transient) != 0 || b.publicationBeforeNormFree {
			t.Fatalf("panel shapes=%v reads=%d publications=%d live=%d handled=%t err=%v", b.shapes, b.reads, b.publications, len(b.transient), handled, err)
		}
		st := s.v41State()
		for r := range got {
			row := make([]float32, m.Cfg.HiddenSize)
			for i := range row {
				row[i] = float32((r+i)%11-5) / 8
			}
			want, err := m.v41AttentionInputNormWithPublications(0, row, float32(m.Cfg.RMSNormEps), st.attentionNormFunc(), st.attentionBF16Func())
			if err != nil || !v41CompressorNormEqualBits(got[r], want) {
				t.Fatalf("row %d differs: %v", r, err)
			}
		}
	})
	for _, mode := range []string{"canonical", "same norm", "wrapped norm", "same publication", "wrapped publication"} {
		t.Run(mode, func(t *testing.T) {
			m, s, b := fixture(t)
			st := s.v41State()
			old := st.attentionNormBinding
			calls := 0
			switch mode {
			case "same norm":
				st.setAttentionNorm(st.attentionNormFunc())
			case "wrapped norm":
				f := st.attentionNormFunc()
				st.setAttentionNorm(func(l int, x []float32) ([]float32, error) { calls++; return f(l, x) })
			case "same publication":
				st.setAttentionBF16(st.attentionBF16Func())
			case "wrapped publication":
				f := st.attentionBF16Func()
				st.setAttentionBF16(func(l int, site string, x []float32) ([]float32, error) { calls++; return f(l, site, x) })
			}
			if mode != "canonical" && (st.attentionNormBinding == old || st.attentionNormBinding.native != nil) {
				t.Fatal("replacement retained authority")
			}
			got := s.Prefill([]int{1, 2})
			v41GroupedParity(t, got, lastLogits(m.Forward([]int{1, 2})), 1e-4)
			want := [][]int{{m.Cfg.HiddenSize}, {m.Cfg.HiddenSize}}
			if mode == "canonical" {
				want = [][]int{{2, m.Cfg.HiddenSize}}
			}
			if !reflect.DeepEqual(b.shapes, want) || len(b.transient) != 0 || b.publicationBeforeNormFree || (strings.HasPrefix(mode, "wrapped") && calls != 2) {
				t.Fatalf("shapes=%v calls=%d live=%d", b.shapes, calls, len(b.transient))
			}
		})
	}
	for _, mode := range []string{"publication-first", "norm-first", "dispatch", "read", "shape", "free", "wrapped norm NaN"} {
		t.Run(mode, func(t *testing.T) {
			m, s, b := fixture(t)
			before := captureV41ForwardSnapshot(s.v41State())
			fault := &compute.BackendError{Backend: "norm-panel-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			b.mode, b.fault = mode, fault
			if mode == "wrapped norm NaN" {
				st := s.v41State()
				ordinary, calls := st.attentionNormFunc(), 0
				st.setAttentionNorm(func(layer int, input []float32) ([]float32, error) {
					out, err := ordinary(layer, input)
					calls++
					if err == nil && calls == 2 {
						out[0] = float32(math.NaN())
					}
					return out, err
				})
			}
			var err error
			if mode == "publication-first" || mode == "norm-first" {
				_, handled, e := panel(m, s, 2)
				err = e
				if !handled {
					t.Fatal("not selected")
				}
			} else {
				err = recoverError(func() { s.Prefill([]int{1, 2}) })
			}
			var closed *BackendForwardOperationError
			if !errors.As(err, &closed) || !s.BackendSessionClosed() || len(b.transient) != 0 || b.publicationBeforeNormFree || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41State()), before) {
				t.Fatalf("failure=%v live=%d state changed=%t", err, len(b.transient), !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41State()), before))
			}
			if mode == "publication-first" {
				var arithmetic *compute.V41BF16CheckedArithmeticError
				if !errors.As(err, &arithmetic) || closed.Path != "v41-attention-bf16" || b.publications != 1 {
					t.Fatalf("wrong first error: %v", err)
				}
			} else if mode == "norm-first" {
				if closed.Path != "v41-attention-input-norm" || b.publications != 0 {
					t.Fatalf("wrong first error: %v", err)
				}
			} else if mode == "wrapped norm NaN" {
				if closed.Path != "v41-attention-bf16" || !reflect.DeepEqual(b.shapes, [][]int{{m.Cfg.HiddenSize}, {m.Cfg.HiddenSize}}) {
					t.Fatalf("wrapper bypassed strict scalar path: %v shapes=%v", err, b.shapes)
				}
			} else if mode != "shape" && !errors.Is(err, fault) {
				t.Fatalf("lost transport/cleanup identity: %v", err)
			}
			calls := len(b.shapes)
			retry := recoverError(func() { s.Prefill([]int{1, 2}) })
			if retry != s.halFailure || len(b.shapes) != calls {
				t.Fatal("retired session replayed")
			}
		})
	}
	for _, setting := range []string{"panicnil=0", "panicnil=1"} {
		t.Run(setting, func(t *testing.T) {
			t.Setenv("GODEBUG", setting)
			for _, mode := range []string{"dispatch nil", "free nil", "dispatch Goexit", "free Goexit"} {
				t.Run(mode, func(t *testing.T) {
					_, s, b := fixture(t)
					before := captureV41ForwardSnapshot(s.v41State())
					b.mode = mode
					done := make(chan struct{})
					returned := false
					var cause error
					go func() {
						defer close(done)
						cause = recoverError(func() { s.Prefill([]int{1, 2}); returned = true })
					}()
					<-done
					if returned || !s.BackendSessionClosed() || len(b.transient) != 0 || b.publicationBeforeNormFree || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41State()), before) {
						t.Fatalf("incomplete panel returned=%t closed=%t live=%d cause=%v", returned, s.BackendSessionClosed(), len(b.transient), cause)
					}
					if strings.HasSuffix(mode, "nil") {
						var closed *BackendForwardOperationError
						if !errors.As(s.halFailure, &closed) {
							t.Fatalf("missing selected retirement: %v", cause)
						}
					}
				})
			}
		})
	}

}
