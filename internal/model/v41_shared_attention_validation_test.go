package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU-owned storage with a real public-entry execution lease. Only attention
// tensors use the outer identity; unrelated seams remain the existing control.
type v41ValidatedAttentionRecorder struct {
	*v41DenseTestBackend
	owner   sync.Mutex
	calls   []v41SharedAttentionTestCall
	uploads int
	live    map[compute.Buffer]bool
}

func (b *v41ValidatedAttentionRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41ValidatedAttentionRecorder) SupportsV41SharedAttention() bool   { return true }
func (b *v41ValidatedAttentionRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 shared attention ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("invalid attention upload")
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.live[out.Buf()] = true
	b.uploads++
	return out
}
func (b *v41ValidatedAttentionRecorder) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	for _, x := range []compute.Tensor{q, kv, sink} {
		if x.Backend() != b || !b.live[x.Buf()] {
			return compute.Tensor{}, errors.New("invalid attention ownership")
		}
	}
	call := v41SharedAttentionTestCall{q: b.Backend.Read(q), kv: b.Backend.Read(kv), sink: b.Backend.Read(sink),
		rows: rows, heads: heads, dim: dim, scale: scale, mode: mode, hasSink: hasSink}
	b.calls = append(b.calls, call)
	values, err := v41SharedAttentionRoundedControl(call)
	if err != nil {
		return compute.Tensor{}, err
	}
	out := compute.NewF32(b, []int{heads, dim}, values)
	b.live[out.Buf()] = true
	return out, nil
}
func (b *v41ValidatedAttentionRecorder) Free(x compute.Tensor) {
	if b.live[x.Buf()] {
		delete(b.live, x.Buf())
		b.Backend.Free(x)
		return
	}
	b.v41DenseTestBackend.Free(x)
}

// One owned source/reader fixture covers the shared single-token continuation
// route and its ordinary-wrapper negative control. This is software evidence;
// runtime is an estimate until the separately authorized real-package run.
// fak-test:runtime medium est=10s lane=default
func TestV41SharedAttentionValidatedContinuation(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("validated attention software witness requires cpu-ref")
	}
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, entry := range []string{"step", "suffix"} {
		t.Run(entry, func(t *testing.T) {
			m := v41ReaderWidthFixture(t)
			newSession := func() (*Session, *v41ValidatedAttentionRecorder) {
				b := &v41ValidatedAttentionRecorder{v41DenseTestBackend: newV41DenseTestBackend(), live: map[compute.Buffer]bool{}}
				s := v41DenseTestSession(t, m, b)
				v41SharedAttentionOnly(s)
				return s, b
			}
			s, b := newSession()
			strict, control := newSession()
			strict.v41State().setSharedAttention(strict.v41State().sharedAttentionFunc())
			binding := s.v41State().sharedAttentionBinding
			core := binding.native.core
			validated, consumed := core.validated, 0
			// Test-only observation happens after the unmodified private entry.
			core.validated = func(layer int, request *v41SharedAttentionRequest, proof *v41SharedAttentionValidation) (v41SharedAttentionResult, error) {
				out, err := validated(layer, request, proof)
				if !proof.used || !proof.accepted || !proof.continuation.used || proof.consume(core, layer, request) {
					t.Fatal("private proof was not consumed exactly once")
				}
				copied, context, other := *proof, *proof.continuation, *request
				copied.used, context.used, copied.continuation = false, false, &context
				if copied.consume(core, layer, &other) {
					t.Fatal("proof transferred to a copied request")
				}
				consumed++
				return out, err
			}
			// Cold positions consume one proof per combined position, and the
			// device preparation then re-scans no row the composer already proved.
			var proofScans, strictScans int
			scans := &proofScans
			v41SharedAttentionPrepareScanProbe = func(elements int) { *scans += elements }
			defer func() { v41SharedAttentionPrepareScanProbe = nil }()
			cold := s.Prefill([]int{1, 2})
			scans = &strictScans
			v41SharedAttentionSessionBits(t, "cold", cold, strict.Prefill([]int{1, 2}))
			v41SharedAttentionPrepareScanProbe = nil
			coldProofs := 0
			for layer := 0; layer < m.Cfg.NumLayers; layer++ {
				if plan, err := m.v41AttentionPlan(layer); err == nil && plan.Ratio > 0 {
					coldProofs += 2
				}
			}
			if coldProofs == 0 || consumed != coldProofs || proofScans != 0 || strictScans == 0 {
				t.Fatalf("cold proofs=%d want %d, prepare row scans proof=%d strict=%d", consumed, coldProofs, proofScans, strictScans)
			}
			v41SharedAttentionSessionBits(t, "step", s.Step(3), strict.Step(3))
			v41SharedAttentionSessionBits(t, "suffix", s.Prefill([]int{4, 5}), strict.Prefill([]int{4, 5}))
			if consumed != coldProofs+3*m.Cfg.NumLayers || len(b.calls) != len(control.calls) || b.uploads != control.uploads ||
				len(b.live) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(strict.v41Forward)) {
				t.Fatal("single-token route changed strict output, work, lifetime, or continuation state")
			}
			func() {
				scope := s.beginBackendExecution()
				defer scope.finish()
				st, scratch := s.v41Forward, &v41ProjScratch{}
				scratch.copySharedAttentionFrom(st)
				plan, err := m.v41AttentionPlan(0)
				if err != nil {
					t.Fatal(err)
				}
				c := newV41SharedAttentionContinuation(m, 0, len(st.history), plan, st.layerState(0), st.attn, scratch)
				if c == nil || !c.live() {
					t.Fatal("owned continuation was not eligible")
				}
				st.setSharedAttention(st.sharedAttentionFunc())
				if c.live() {
					t.Fatal("same-function setter preserved authority")
				}
				st.bindCanonicalSharedAttention(binding)
				s.cacheGeometryEpoch++
				if c.live() {
					t.Fatal("changed epoch preserved authority")
				}
				s.cacheGeometryEpoch--
				m.Cfg.DeepSeekV41.CompressRatios[0] = 1
				if c.live() {
					t.Fatal("changed role plan preserved authority")
				}
				m.Cfg.DeepSeekV41.CompressRatios[0] = 2
			}()
			ordinary, wrapped := s.v41State().sharedAttentionFunc(), 0
			s.v41State().setSharedAttention(func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
				wrapped++
				if request.rowKV == nil || len(request.rowKV.compressed) == 0 {
					t.Fatal("negative control needs an original compressed row")
				}
				rows := *request.rowKV
				rows.compressed = append([][]float32(nil), rows.compressed...)
				rows.compressed[0] = append([]float32(nil), rows.compressed[0]...)
				rows.compressed[0][0] = float32(math.NaN())
				request.rowKV = &rows
				request.idx = append([]int32(nil), request.idx...)
				for i, id := range request.idx {
					if int(id) == len(rows.window) {
						request.idx[i] = -1
					}
				}
				return ordinary(layer, request)
			})
			before, calls, uploads := captureV41ForwardSnapshot(s.v41Forward), len(b.calls), b.uploads
			invoke := func() {
				if entry == "step" {
					s.Step(6)
				} else {
					s.Prefill([]int{6, 7})
				}
			}
			err := recoverError(invoke)
			var selected *V41SharedAttentionOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) ||
				wrapped != 1 || !s.halClosed || s.halFailure == nil || len(b.calls) != calls || b.uploads != uploads || len(b.live) != 0 ||
				consumed != coldProofs+3*m.Cfg.NumLayers || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatalf("masked original row escaped strict validation or retirement: %v", err)
			}
			if retry := recoverError(invoke); retry != s.halFailure || wrapped != 1 || len(b.calls) != calls {
				t.Fatalf("retired owner replayed: %v", retry)
			}
		})
	}
}
