package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage witnesses selected calls and ownership, not physical GPU fences.
// Reuse the existing independent rotary oracle and legacy Read poison checks.
type v41PairedRoPERecorder struct {
	*v41TailRoPETestBackend
	paired, regions, pairBytes, probes int
	groupFault                         string
	shared                             []float32
	owned                              map[compute.Buffer]int
}

func newV41PairedRoPERecorder() *v41PairedRoPERecorder {
	return &v41PairedRoPERecorder{v41TailRoPETestBackend: newV41TailRoPETestBackend(), owned: map[compute.Buffer]int{}}
}

func (b *v41PairedRoPERecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 tail-RoPE ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		panic("unexpected paired RoPE upload")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.owned[y.Buf()] = 0
	if site == "V4.1 tail-RoPE table" {
		b.tables[y.Buf()] = true
		b.tableUploads++
		b.tableBytes += 4 * len(values)
	} else {
		b.transients[y.Buf()] = true
	}
	b.uploadCount++
	b.uploadBytes += 4 * len(values)
	return y
}

func (b *v41PairedRoPERecorder) V41TailRoPEQK(q, kv, table compute.Tensor, heads, headDim, rotaryDim int) (compute.Tensor, compute.Tensor, error) {
	if q.Backend() != b || kv.Backend() != b || table.Backend() != b ||
		!reflect.DeepEqual(q.Shape, []int{heads, headDim}) || !reflect.DeepEqual(kv.Shape, []int{headDim}) || !reflect.DeepEqual(table.Shape, []int{rotaryDim / 2, 2}) {
		panic("unexpected paired RoPE tensor ownership or shape")
	}
	call := v41TailRoPECall{q: append([]float32(nil), b.Backend.Read(q)...), kv: append([]float32(nil), b.Backend.Read(kv)...),
		table: append([]float32(nil), b.Backend.Read(table)...), heads: heads, headDim: headDim, rotaryDim: rotaryDim}
	call.qOut = v41TailRoPEExactOracle(call.q, call.table, headDim, rotaryDim)
	call.kvOut = v41TailRoPEExactOracle(call.kv, call.table, headDim, rotaryDim)
	b.calls = append(b.calls, call)
	qo := compute.NewF32(b, []int{heads, headDim}, append([]float32(nil), call.qOut...))
	ko := compute.NewF32(b, []int{headDim}, append([]float32(nil), call.kvOut...))
	b.owned[qo.Buf()], b.owned[ko.Buf()] = 0, 0
	b.transients[qo.Buf()], b.transients[ko.Buf()] = true, true
	b.outputs[qo.Buf()], b.outputs[ko.Buf()] = "query", "kv"
	return qo, ko, nil
}

// This probe covers unrelated deferred kernels. Pure zero-status reads remain
// valid even when those modules are absent; no such kernel is used by this test.
func (b *v41PairedRoPERecorder) SupportsV41DeferredStatus(int) bool { b.probes++; return false }
func (b *v41PairedRoPERecorder) V41RoutedSwiGLUDeferred(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unexpected deferred SwiGLU")
}
func (b *v41PairedRoPERecorder) V41BF16PublishDeferred(compute.Tensor, uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unexpected deferred BF16 publication")
}

func (b *v41PairedRoPERecorder) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) != 0 || len(rows) != 2 || b.outputs[rows[0].Buf()] != "query" || b.outputs[rows[1].Buf()] != "kv" {
		panic("paired read changed status count or Q/KV order")
	}
	b.paired++
	b.regions += len(rows)
	q, kv := b.Backend.Read(rows[0]), b.Backend.Read(rows[1])
	n := len(q) + len(kv)
	if cap(b.shared) < n {
		b.shared = make([]float32, n)
	}
	b.shared = b.shared[:n]
	copy(b.shared, q)
	copy(b.shared[len(q):], kv)
	b.pairBytes += 4 * n
	// Both returned rows share reusable backing. Free poisons it, so the model
	// must own both results before releasing either device output.
	values := [][]float32{b.shared[:len(q):len(q)], b.shared[len(q):n:n]}
	switch b.groupFault {
	case "transport":
		values[0][0] = float32(math.NaN()) // transport precedes numeric validation
		return nil, values, b.cause
	case "verdict-count":
		return []error{nil}, values, nil
	case "row-count":
		return nil, values[:1], nil
	case "short-query":
		values[0] = values[0][:len(q)-1]
	case "short-kv":
		values[1] = values[1][:len(kv)-1]
	case "query-prefix":
		values[0][0] = math.Float32frombits(math.Float32bits(values[0][0]) ^ 1)
	case "query-and-kv-nan":
		values[0][len(q)-1] = float32(math.NaN())
		values[1][len(kv)-1] = float32(math.NaN())
	case "kv-nan":
		values[1][len(kv)-1] = float32(math.NaN())
	}
	return nil, values, nil
}

func (b *v41PairedRoPERecorder) Free(x compute.Tensor) {
	if _, ok := b.owned[x.Buf()]; ok {
		b.owned[x.Buf()]++
	}
	if b.outputs[x.Buf()] != "" {
		for i := range b.shared {
			b.shared[i] = float32(math.NaN())
		}
	}
	b.v41TailRoPETestBackend.Free(x)
}

// Refs #13668/#13820. The parent performs two ordinary Reads per selected call;
// the candidate groups them while retaining two regions and identical bytes.
// fak-test:runtime medium est=5s lane=default
func TestV41TailRoPEPairedReadback(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" || compute.Default().Caps().DeviceMemory {
		t.Fatal("paired RoPE recorder requires cpu-ref storage")
	}
	t.Run("selected-helper-ownership-and-faults", func(t *testing.T) {
		for _, tc := range []struct{ fault, stage string }{
			{"", ""}, {"transport", "paired readback"}, {"verdict-count", "paired readback"}, {"row-count", "paired readback"},
			{"short-query", "query readback"}, {"short-kv", "kv readback"}, {"query-prefix", "query readback"},
			{"query-and-kv-nan", "query readback"}, {"kv-nan", "kv readback"},
		} {
			t.Run("fault="+tc.fault, func(t *testing.T) {
				b := newV41PairedRoPERecorder()
				b.groupFault = tc.fault
				b.cause = &compute.BackendError{Backend: "cpu-ref-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
				s := &Session{M: &Model{}, Backend: b}
				t.Cleanup(s.Close)
				q := []float32{math.Float32frombits(0x80000000), -7, 3, 11, 2, 9, -5, 1}
				kv := []float32{-2, 4, 6, 8}
				beforeQ, beforeKV := append([]float32(nil), q...), append([]float32(nil), kv...)
				err := v41TailRoPEInPlace(0, q, kv, []float32{0.3}, []float32{0.7}, 2, 4, 2, s.v41TailRoPEFunc())
				if tc.fault == "" {
					if err != nil {
						t.Fatal(err)
					}
					v41TailRoPECheckBits(t, "owned Q", q, v41TailRoPEExactOracle(beforeQ, []float32{0.7, 0.3}, 4, 2))
					v41TailRoPECheckBits(t, "owned KV", kv, v41TailRoPEExactOracle(beforeKV, []float32{0.7, 0.3}, 4, 2))
				} else {
					var selected *V41TailRoPEOperationError
					var closed *BackendForwardOperationError
					if !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Stage != tc.stage || !s.BackendSessionClosed() {
						t.Fatalf("fault=%s lost selected closure/stage: %v", tc.fault, err)
					}
					if tc.fault == "transport" && !errors.Is(err, b.cause) {
						t.Fatal("grouped transport did not precede malformed Q")
					}
					v41TailRoPECheckBits(t, "atomic failed Q", q, beforeQ)
					v41TailRoPECheckBits(t, "atomic failed KV", kv, beforeKV)
				}
				if b.paired != 1 || b.readCount != 0 || b.regions != 2 || b.pairBytes != 4*(len(q)+len(kv)) || b.probes != 0 || len(b.calls) != 1 || len(b.transients) != 0 {
					t.Fatalf("selected call/fence contract: paired=%d reads=%d regions=%d bytes=%d probes=%d calls=%d live=%d", b.paired, b.readCount, b.regions, b.pairBytes, b.probes, len(b.calls), len(b.transients))
				}
				s.Close()
				if len(b.owned) != 5 || len(b.tables) != 0 {
					t.Fatalf("owned allocations=%d tables=%d", len(b.owned), len(b.tables))
				}
				for _, frees := range b.owned {
					if frees != 1 {
						t.Fatalf("owned allocation freed %d times", frees)
					}
				}
			})
		}
	})
	t.Run("full-session-retained-step", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41PairedRoPERecorder()
		s := v41DenseTestSession(t, m, b)
		v41TailRoPEOnly(s)
		legacyBackend := newV41TailRoPETestBackend()
		legacy := v41DenseTestSession(t, m, legacyBackend)
		v41TailRoPEOnly(legacy)
		prefix := []int{1, 2}
		got := s.Prefill(prefix)
		v41TailRoPECheckBits(t, "selected prefill", got, legacy.Prefill(prefix))
		v41GroupedParity(t, got, lastLogits(m.Forward(prefix)), 1e-4)
		if !s.v41IncrementalEligible() || s.v41State().tailRoPE == nil {
			t.Fatal("real full session did not bind selected incremental RoPE")
		}
		for _, token := range []int{3, 4} {
			before := len(b.calls)
			prefix = append(prefix, token)
			got := s.Step(token)
			v41TailRoPECheckBits(t, "selected retained Step", got, legacy.Step(token))
			v41GroupedParity(t, got, lastLogits(m.Forward(prefix)), 1e-4)
			if len(b.calls)-before != m.Cfg.NumLayers || !reflect.DeepEqual(s.v41State().history, prefix) {
				t.Fatal("retained Step replayed a prefix or lost its committed history")
			}
		}
		want := len(prefix) * m.Cfg.NumLayers
		bytes := want * 4 * (m.Cfg.NumHeads + 1) * m.Cfg.HeadDim
		if len(b.calls) != want || b.paired != want || b.readCount != 0 || b.regions != 2*want || b.pairBytes != bytes || len(b.transients) != 0 {
			t.Fatalf("production selected calls=%d paired=%d ordinary=%d regions=%d bytes=%d want=%d/%d", len(b.calls), b.paired, b.readCount, b.regions, b.pairBytes, want, bytes)
		}
		if len(legacyBackend.calls) != want || legacyBackend.readCount != 2*want || legacyBackend.readBytes != bytes {
			t.Fatal("legacy selected control did not retain two Reads and identical bytes")
		}
	})
}
