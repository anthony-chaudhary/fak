package model

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder executes the actual CPU reference contractions through the
// production Session. Counts are structural evidence, never device timing.
type v41ColdCompressorRecorder struct {
	*v41CompressorTestBackend
	s          *Session
	owner      sync.Mutex
	pairs      int
	reads      [][]compute.Tensor
	compressor []v41DenseTestOp
	order      []string
	outputs    map[compute.Buffer]int
	fault      string
	cause      error
	cleanup    bool
	released   map[compute.Buffer]bool
}

func newV41ColdCompressorRecorder() *v41ColdCompressorRecorder {
	return &v41ColdCompressorRecorder{v41CompressorTestBackend: newV41CompressorTestBackend(), outputs: map[compute.Buffer]int{}, released: map[compute.Buffer]bool{}, cause: &compute.BackendError{Backend: "cold-compressor-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}}
}

func (b *v41ColdCompressorRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

func (b *v41ColdCompressorRecorder) interrupt(site string) {
	if b.fault == "goexit-"+site {
		runtime.Goexit()
	}
	if b.fault == "panicnil-"+site {
		panic(nil)
	}
}
func (b *v41ColdCompressorRecorder) SupportsV41DeferredStatus(int) bool { return false }
func (b *v41ColdCompressorRecorder) V41RoutedSwiGLUDeferred(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unexpected checked SwiGLU")
}
func (b *v41ColdCompressorRecorder) V41BF16PublishDeferred(compute.Tensor, uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unexpected checked publication")
}
func (b *v41ColdCompressorRecorder) which(w compute.Tensor) (int, string) {
	if b.s != nil {
		for key, weight := range b.s.halW {
			if weight.Buf() == w.Buf() {
				for i, leaf := range v41ColdCompressorLeaves {
					if strings.HasSuffix(key, leaf) {
						return i, key
					}
				}
			}
		}
	}
	return -1, ""
}
func (b *v41ColdCompressorRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	y := b.v41CompressorTestBackend.MatMul(w, x)
	if leaf, name := b.which(w); leaf >= 0 {
		b.compressor = append(b.compressor, b.ops[len(b.ops)-1])
		b.order = append(b.order, name)
	}
	return y
}
func (b *v41ColdCompressorRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	leaf, name := b.which(w)
	if leaf == 1 {
		b.interrupt("dispatch")
	}
	if leaf == 1 && b.fault == "later-recording" {
		panic(b.cause)
	}
	y := b.v41CompressorTestBackend.BatchedMatMul(w, x, rows)
	if leaf >= 0 {
		b.compressor = append(b.compressor, b.ops[len(b.ops)-1])
		b.order = append(b.order, name)
		b.outputs[y.Buf()] = leaf
		if leaf == 0 {
			b.pairs++
		}
		if b.fault == "ownership" && leaf == 1 {
			b.s.cacheGeometryEpoch++
		}
		if b.fault == "malformed-tensor" && leaf == 1 {
			y.Shape = []int{1, rows * w.Shape[0]}
		}
	}
	return y
}
func (b *v41ColdCompressorRecorder) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) != 0 {
		panic("unexpected status record")
	}
	b.reads = append(b.reads, append([]compute.Tensor(nil), rows...))
	b.interrupt("read")
	if b.fault == "transport" {
		return nil, nil, b.cause
	}
	values := make([][]float32, len(rows))
	for i, row := range rows {
		values[i] = append([]float32(nil), b.v41CompressorTestBackend.Read(row)...)
		which := b.outputs[row.Buf()]
		if b.fault == "mixed-numeric" && which == 0 {
			values[i][len(values[i])/2] = float32(math.NaN())
		}
		if (b.fault == "mixed-numeric" || b.fault == "later-cleanup") && which == 1 {
			values[i][0] = float32(math.NaN())
		}
		if b.fault == "later-recording" && which == 0 {
			values[i][0] = float32(math.NaN())
		}
		if b.fault == "malformed-values" && which == 1 {
			values[i] = values[i][:len(values[i])-1]
		}
	}
	return nil, values, nil
}
func (b *v41ColdCompressorRecorder) Free(x compute.Tensor) {
	leaf, isOutput := b.outputs[x.Buf()]
	b.v41CompressorTestBackend.Free(x)
	if isOutput {
		b.released[x.Buf()] = true
		if leaf == 1 && !b.cleanup && (strings.HasSuffix(b.fault, "-free")) {
			b.cleanup = true
			b.interrupt("free")
		}
		if b.fault == "later-cleanup" && leaf == 1 && !b.cleanup {
			b.cleanup = true
			panic(b.cause)
		}
	}
}

func (b *v41ColdCompressorRecorder) assertNoTransients(t *testing.T) {
	t.Helper()
	// The inherited recorder also tracks one-dimensional immutable weights.
	// Session retirement leaves their model owner alive; exclude only handles
	// currently registered to that exact model/backend, including after failure.
	live := make(map[compute.Buffer]bool, len(b.live))
	for buffer := range b.live {
		live[buffer] = true
	}
	if !b.s.M.VisitImmutableDeviceWeights(b, func(weight compute.Tensor) bool {
		delete(live, weight.Buf())
		return true
	}) {
		t.Fatal("could not establish immutable model-weight ownership")
	}
	if len(live) != 0 {
		t.Fatalf("compressor path leaked %d transient buffers (%d total live)", len(live), len(b.live))
	}
}

func v41ColdCompressorSession(t *testing.T, m *Model, scalar bool) (*Session, *v41ColdCompressorRecorder) {
	t.Helper()
	b := newV41ColdCompressorRecorder()
	s := v41DenseTestSession(t, m, b)
	b.s = s
	if scalar {
		st := s.v41State()
		ordinary := st.denseProjection()
		st.setDenseProjection(func(l int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			return ordinary(l, leaf, panel, out, in, rows)
		})
	}
	return s, b
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=8s lane=default
func TestV41ColdCompressorPanelSession(t *testing.T) {
	t.Run("prefill-step", func(t *testing.T) {
		m, reference := v41CompressorTestFixture(t), v41CompressorTestFixture(t)
		s, b := v41ColdCompressorSession(t, m, false)
		serial, scalar := v41ColdCompressorSession(t, reference, true)
		norms, scalarNorms := 0, 0
		for _, item := range []struct {
			session *Session
			count   *int
		}{{s, &norms}, {serial, &scalarNorms}} {
			st := item.session.v41State()
			normalize, count := st.compressorNorm, item.count
			st.compressorNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
				*count++
				if normalize == nil {
					pool, err := NewV41CompressorPool(1, len(input))
					if err != nil {
						return nil, err
					}
					output, _, err := pool.PushNormalized(0, input, nil, gain, eps)
					return output, err
				}
				return normalize(l, input, gain, eps)
			}
		}
		before := v41CompressorTestPhase(t, m, "prefill")
		got, want := s.Prefill([]int{1, 2, 3, 4, 5}), serial.Prefill([]int{1, 2, 3, 4, 5})
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(serial.v41Forward)) || !reflect.DeepEqual(s.v41Forward.indexKeyMemo, serial.v41Forward.indexKeyMemo) {
			t.Fatal("paired cold Session changed logits or retained source/index/window/partial state")
		}
		if b.pairs != 2 || len(b.reads) != 2 || len(b.compressor) != 6 || len(scalar.compressor) != 10 || scalar.pairs != 0 {
			t.Fatalf("pair/scalar dispatch mismatch: pairs=%d reads=%d calls=%d scalar=%d", b.pairs, len(b.reads), len(b.compressor), len(scalar.compressor))
		}
		if norms != 2 || scalarNorms != 2 {
			t.Fatalf("pool/norm boundary count=%d scalar=%d", norms, scalarNorms)
		}
		for i, op := range b.compressor {
			if !strings.Contains(b.order[i], "layers.0.") {
				t.Fatalf("reader dispatched private compressor: %s", b.order[i])
			}
			wantRows := 2
			if i >= 4 {
				wantRows = 1
			}
			if op.rows != wantRows || op.batch != (wantRows == 2) {
				t.Fatal("complete pairs or odd tail changed dispatch shape")
			}
			if i < 4 && i%2 == 1 && op.input != b.compressor[i-1].input {
				t.Fatal("paired leaves did not share their activation upload")
			}
		}
		for _, read := range b.reads {
			if len(read) != 2 {
				t.Fatal("successful pair did not use one two-output read")
			}
		}
		delta := v41DenseTestDelta(v41CompressorTestPhase(t, m, "prefill"), before)
		if delta["compressor_projection_device_calls"] != 6 || delta["compressor_projection_device_rows"] != 10 ||
			delta["compressor_projection_activation_upload_bytes"] != float64(6*m.Cfg.HiddenSize*4) || delta["compressor_projection_readback_bytes"] != float64(10*m.Cfg.HeadDim*4) || delta["compressor_projection_host_calls"] != 0 {
			t.Fatalf("compressor pair accounting changed: %v", delta)
		}
		got, want = s.Step(6), serial.Step(6)
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(serial.v41Forward)) || !reflect.DeepEqual(s.v41Forward.indexKeyMemo, serial.v41Forward.indexKeyMemo) {
			t.Fatal("Prefill-to-Step continuation changed")
		}
		if b.pairs != 2 || len(b.reads) != 2 || len(b.compressor) != 10 {
			t.Fatal("Step no longer uses its historical complete-group scalar projections")
		}
		b.assertNoTransients(t)
	})
	for _, dtype := range []compute.Dtype{compute.Q8_0, compute.Q4_K} {
		t.Run("packed-fallback/"+dtype.String(), func(t *testing.T) {
			m := v41CompressorTestFixture(t)
			const hidden = 256
			m.Cfg.HiddenSize = hidden
			m.q8w, m.q4kw = map[string]*q8Tensor{}, map[string]*q4kTensor{}
			for _, leaf := range v41ColdCompressorLeaves {
				name := layerName(0, leaf)
				delete(m.manifest, name)
				if dtype == compute.Q8_0 {
					weight := make([]float32, m.Cfg.HeadDim*hidden)
					for i := range weight {
						weight[i] = float32(i%31-15) * .003
					}
					m.q8w[name] = quantizeQ8(weight, m.Cfg.HeadDim, hidden)
				} else {
					m.q4kw[name] = quantizeQ4KFromRaw(v41DenseTestQ4Raw(m.Cfg.HeadDim, hidden), m.Cfg.HeadDim, hidden)
				}
			}
			s, b := v41ColdCompressorSession(t, m, false)
			st := s.v41State()
			scope := s.beginBackendExecution()
			defer scope.finish()
			scratch := &v41ProjScratch{projectionRoute: st.projectionRoute, compressorNorm: st.compressorNorm, gatherOps: st.gatherOps}
			inputs := [][]float32{make([]float32, hidden), make([]float32, hidden)}
			for row := range inputs {
				for i := range inputs[row] {
					inputs[row][i] = float32((i+row*3)%17-8) * .02
				}
			}
			got, err := m.v41ColdCompressedRows(0, 2, nil, inputs, true, st, scratch)
			if err != nil {
				t.Fatal(err)
			}
			want, err := m.v41CompressedRowsWithDeviceOps(0, 2, nil, inputs, st.denseProjection(), st.compressorNorm, st.gatherOps.compressorPoolFunc())
			if err != nil || !reflect.DeepEqual(got, want) || b.pairs != 0 || len(b.reads) != 0 || len(b.compressor) != 8 {
				t.Fatalf("unqualified packed pair did not retain canonical scalar fallback: %v", err)
			}
		})
	}
	for _, mode := range []string{"gain", "epsilon", "mutated-gain"} {
		t.Run("pool-admission/"+mode, func(t *testing.T) {
			m := v41CompressorTestFixture(t)
			if mode == "gain" {
				m.tensor(layerName(0, "attn.compressor.norm.weight"))[0] = float32(math.NaN())
			} else if mode == "epsilon" {
				m.Cfg.RMSNormEps = 0
			}
			s, b := v41ColdCompressorSession(t, m, false)
			st := s.v41State()
			scope := s.beginBackendExecution()
			defer scope.finish()
			scratch := &v41ProjScratch{projectionRoute: st.projectionRoute, compressorNorm: st.compressorNorm, gatherOps: st.gatherOps}
			inputs := [][]float32{make([]float32, m.Cfg.HiddenSize), make([]float32, m.Cfg.HiddenSize)}
			wantPairs, wantReads, wantCalls := 0, 0, 2
			if mode == "mutated-gain" {
				inputs = append(inputs, inputs...)
				wantPairs, wantReads, wantCalls = 1, 1, 4
				scratch.compressorNorm = func(_ int, input, gain []float32, eps float32) ([]float32, error) {
					pool, err := NewV41CompressorPool(1, len(input))
					if err != nil {
						return nil, err
					}
					output, _, err := pool.PushNormalized(0, input, nil, gain, eps)
					gain[0] = float32(math.NaN())
					return output, err
				}
			}
			_, err := m.v41ColdCompressedRows(0, 2, nil, inputs, true, st, scratch)
			if err == nil || b.pairs != wantPairs || len(b.reads) != wantReads || len(b.compressor) != wantCalls {
				t.Fatalf("invalid pool operand moved past first scalar row: %v", err)
			}
		})
	}
	for _, mode := range []string{"nil", "ordinary", "replaced-same", "wrapped"} {
		t.Run("callback/"+mode, func(t *testing.T) {
			m := v41CompressorTestFixture(t)
			s, b := v41ColdCompressorSession(t, m, false)
			st := s.v41State()
			old := st.denseProjection()
			var calls []string
			switch mode {
			case "nil":
				st.setDenseProjection(nil)
			case "replaced-same":
				st.setDenseProjection(old)
			default:
				st.setDenseProjection(func(l int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if v41CompressorProjectionLeaf(leaf) {
						if rows != 1 {
							t.Fatal("ordinary compressor callback was batched")
						}
						calls = append(calls, leaf)
					}
					if mode == "wrapped" {
						return old(l, leaf, panel, out, in, rows)
					}
					return nil, v41ProjectionDeclined, nil
				})
			}
			s.Prefill([]int{1, 2, 3, 4, 5})
			if b.pairs != 0 || len(b.reads) != 0 {
				t.Fatal("ordinary callback inherited native authority")
			}
			if mode == "ordinary" || mode == "wrapped" {
				if len(calls) != 10 {
					t.Fatalf("scalar callback count=%d", len(calls))
				}
				for i, leaf := range calls {
					if leaf != v41ColdCompressorLeaves[i%2] {
						t.Fatal("scalar callback order changed")
					}
				}
			}
		})
	}
	for _, mode := range []string{"mixed-numeric", "later-recording", "later-cleanup", "transport", "malformed-values", "malformed-tensor", "ownership"} {
		t.Run("failure/"+mode, func(t *testing.T) {
			m := v41CompressorTestFixture(t)
			s, b := v41ColdCompressorSession(t, m, false)
			s.v41State()
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault = mode
			recovered := v41CompressorTestRecover(func() { s.Prefill([]int{1, 2, 3, 4, 5}) })
			err, ok := recovered.(error)
			var operation *V41ProjectionOperationError
			var closed *BackendForwardOperationError
			if !ok || !errors.As(err, &operation) || !errors.As(err, &closed) || !s.BackendSessionClosed() {
				t.Fatalf("selected pair failure did not close with typed projection error: %T %v", recovered, recovered)
			}
			wantLeaf := v41ColdCompressorLeaves[1]
			if mode == "later-recording" || mode == "transport" || mode == "ownership" {
				wantLeaf = v41ColdCompressorLeaves[0]
			}
			if operation.Leaf != wantLeaf {
				t.Fatalf("row/stage precedence selected %s, want %s", operation.Leaf, wantLeaf)
			}
			if mode == "transport" && !errors.Is(err, b.cause) {
				t.Fatal("transport identity lost")
			}
			if (mode == "later-recording" || mode == "later-cleanup") && errors.Is(err, b.cause) {
				t.Fatal("later backend failure masked known earlier arithmetic")
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("failed cold request published partial state")
			}
			b.assertNoTransients(t)
			for output := range b.outputs {
				if !b.released[output] {
					t.Fatal("issued output escaped cleanup")
				}
			}
			if len(b.reads) != 1 || b.pairs != 1 {
				t.Fatal("failed pair did not drain exactly once or replayed")
			}
			phase := v41CompressorTestPhase(t, m, "prefill")
			if phase["compressor_projection_host_calls"] != 0 {
				t.Fatal("selected failure replayed on host")
			}
			calls := len(b.compressor)
			if next := v41CompressorTestRecover(func() { s.Step(6) }); next != s.halFailure || len(b.compressor) != calls {
				t.Fatal("closed owner retried or changed primary failure")
			}
		})
	}
	for _, panicnil := range []string{"0", "1"} {
		for _, kind := range []string{"goexit", "panicnil"} {
			for _, site := range []string{"dispatch", "read", "free"} {
				t.Run("lifecycle/"+panicnil+"/"+kind+"/"+site, func(t *testing.T) {
					t.Setenv("GODEBUG", "panicnil="+panicnil)
					m := v41CompressorTestFixture(t)
					s, b := v41ColdCompressorSession(t, m, false)
					s.v41State()
					before := captureV41ForwardSnapshot(s.v41Forward)
					b.fault = kind + "-" + site
					done := make(chan struct{})
					go func() {
						defer close(done)
						_ = v41CompressorTestRecover(func() { s.Prefill([]int{1, 2, 3}) })
					}()
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Fatal("interrupted compressor retained execution owner")
					}
					if !s.BackendSessionClosed() || s.halFailure == nil {
						t.Fatal("interrupted compressor kept owner usable or lost its failure latch")
					}
					if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
						t.Fatal("interrupted compressor committed partial state")
					}
					b.assertNoTransients(t)
					for output := range b.outputs {
						if !b.released[output] {
							t.Fatal("Goexit prevented a later deferred Free")
						}
					}
					if len(b.reads) != 1 {
						t.Fatalf("interrupted pair drain attempts=%d", len(b.reads))
					}
				})
			}
		}
	}
}
