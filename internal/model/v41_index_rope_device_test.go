package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41IndexRoPERecorder struct {
	*v41InverseRowsTestBackend
	failure      string
	cleanupStage string
	cleanupCalls int
	cleanupArmed bool
	resources    map[compute.Buffer]*v41IndexRoPEResource
}

type v41IndexRoPEResource struct {
	attempts int
	released bool
}

func newV41IndexRoPERecorder() *v41IndexRoPERecorder {
	return &v41IndexRoPERecorder{v41InverseRowsTestBackend: newV41InverseRowsTestBackend(), resources: map[compute.Buffer]*v41IndexRoPEResource{}}
}

func (b *v41IndexRoPERecorder) BeginBatch() {}

func (b *v41IndexRoPERecorder) FlushBatch() {
	if b.cleanupArmed && b.cleanupStage == "flush" {
		b.cleanupCalls++
		panic(errors.New("secondary flush failure"))
	}
}

func (b *v41IndexRoPERecorder) Free(x compute.Tensor) {
	owned := b.resources[x.Buf()]
	if owned != nil {
		owned.attempts++
	}
	if owned != nil && b.cleanupArmed && b.cleanupStage == "free" && b.cleanupCalls == 0 {
		b.cleanupCalls++
		panic(errors.New("secondary free failure"))
	}
	b.v41InverseRowsTestBackend.Free(x)
	if owned != nil {
		owned.released = true
	}
}

func v41IndexRoPECheckResources(t *testing.T, b *v41IndexRoPERecorder, failedFree bool) {
	t.Helper()
	if len(b.resources) == 0 {
		t.Fatal("resource census did not observe any selected allocations")
	}
	unreleased := 0
	for _, resource := range b.resources {
		if resource.attempts != 1 {
			t.Fatalf("owned transient Free attempts=%d, want exactly one", resource.attempts)
		}
		if !resource.released {
			unreleased++
		}
	}
	if (!failedFree && unreleased != 0) || (failedFree && unreleased != b.cleanupCalls) {
		t.Fatalf("resource census releases mismatch: unreleased=%d injected failures=%d", unreleased, b.cleanupCalls)
	}
}

func (b *v41IndexRoPERecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if class != compute.MemoryActivation || dt != compute.F32 {
		return b.v41InverseRowsTestBackend.UploadClass(x, dt, class, site)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.resources[out.Buf()] = &v41IndexRoPEResource{}
	return out
}

func (b *v41IndexRoPERecorder) V41TailRoPERows(x, table compute.Tensor, rows, hd, rd int) (compute.Tensor, error) {
	call := v41InverseRowsCall{x: append([]float32(nil), b.Backend.Read(x)...), table: append([]float32(nil), b.Backend.Read(table)...), rows: rows, headDim: hd, rotaryDim: rd}
	b.calls = append(b.calls, call)
	if x.Backend() != b || table.Backend() != b || !reflect.DeepEqual(x.Shape, []int{rows, hd}) || !reflect.DeepEqual(table.Shape, []int{rd / 2, 2}) {
		return compute.Tensor{}, errors.New("recorder operand geometry")
	}
	if len(b.calls) == b.failCall {
		switch b.failure {
		case "error":
			b.cleanupArmed = true
			return compute.Tensor{}, b.cause
		case "panic":
			b.cleanupArmed = true
			panic(b.cause)
		}
	}
	values := v41TailRoPEExactOracle(call.x, call.table, hd, rd)
	if len(b.calls) == b.failCall {
		switch b.failure {
		case "prefix":
			values[0] = math.Float32frombits(math.Float32bits(values[0]) ^ 1)
		case "nonfinite":
			values[len(values)-1] = float32(math.Inf(1))
		case "length":
			values = values[:len(values)-1]
		}
	}
	out := compute.NewF32(b, []int{rows, hd}, values)
	b.outputs[out.Buf()] = true
	b.resources[out.Buf()] = &v41IndexRoPEResource{}
	return out, nil
}

// fak-test:runtime fast est=20ms lane=default
// fak-test:justify Pins independent nonzero-position arithmetic and atomic publication.
func TestV41IndexRoPEDeviceContract(t *testing.T) {
	m := &Model{Cfg: Config{IndexHeadDim: 8, IndexNHeads: 2, QKRopeHeadDim: 4, RopeTheta: 256}}
	input := []float32{math.Float32frombits(0x80000000), 2, 3, 4, .125, -.75, 1.25, 2.5, 9, 10, 11, 12, -.5, 1.5, 3.25, -.25}
	cos, sin := v41RopeTableForLayer(m.Cfg, 0, 7)
	table := make([]float32, len(cos)*2)
	for i := range cos {
		table[2*i], table[2*i+1] = sin[i], cos[i]
	}
	want := v41TailRoPEExactOracle(input, table, 8, 4)
	for h := 0; h < 2; h++ {
		for i := h*8 + 4; i < h*8+8; i++ {
			want[i] = v41CompressorNormRefCast(want[i])
		}
	}
	t.Run("owned-publication", func(t *testing.T) {
		got := append([]float32(nil), input...)
		var published []float32
		calls := 0
		rotate := func(layer int, values, c, s []float32, heads, hd, rd int) ([]float32, error) {
			calls++
			if layer != 0 || heads != 2 || hd != 8 || rd != 4 {
				t.Fatal("selected query geometry changed")
			}
			v41TailRoPECheckBits(t, "raw query", values, input)
			v41TailRoPECheckBits(t, "cos", c, cos)
			v41TailRoPECheckBits(t, "sin", s, sin)
			published = v41TailRoPEExactOracle(values, table, hd, rd)
			return published, nil
		}
		if err := m.v41IndexRoPEWithDevice(0, 7, got, 2, rotate); err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "staged rotation", got, want)
		if calls != 1 || reflect.DeepEqual(got, input) {
			t.Fatal("nonzero rotation was not invoked exactly once")
		}
		for i := range published {
			published[i] = 99
		}
		v41TailRoPECheckBits(t, "owned caller publication", got, want)
	})
	for _, fault := range []string{"length", "prefix", "nonfinite", "BF16-overflow", "error", "mutated-input"} {
		t.Run(fault, func(t *testing.T) {
			got := append([]float32(nil), input...)
			cause := errors.New("selected query failure")
			calls := 0
			err := m.v41IndexRoPEWithDevice(0, 7, got, 2, func(_ int, x, _, _ []float32, _, hd, rd int) ([]float32, error) {
				calls++
				out := v41TailRoPEExactOracle(x, table, hd, rd)
				switch fault {
				case "length":
					out = out[:len(out)-1]
				case "prefix":
					out[0] = 1
				case "nonfinite":
					out[len(out)-1] = float32(math.NaN())
				case "BF16-overflow":
					out[len(out)-1] = math.MaxFloat32
				case "error":
					return nil, cause
				case "mutated-input":
					x[0] = 123
					return nil, cause
				}
				return out, nil
			})
			var typed *V41ProjectionOperationError
			if err == nil || !errors.As(err, &typed) || typed.Leaf != "indexer.wq_b.weight" || calls != 1 {
				t.Fatalf("selected failure lost its projection contract: %v", err)
			}
			if (fault == "error" || fault == "mutated-input") && !errors.Is(err, cause) {
				t.Fatal("selected cause was discarded")
			}
			v41TailRoPECheckBits(t, "atomic refusal", got, input)
		})
	}
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Exercises production prefill and incremental carriers plus selected fail-close.
func TestV41IndexRoPEDeviceReachability(t *testing.T) {
	t.Run("prefill-step-suffix", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41IndexRoPERecorder()
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		s.v41State().gatherOps.inverseOutput = nil
		if s.v41State().gatherOps.indexRoPEFunc() == nil {
			t.Fatal("available query rows primitive was not bound")
		}
		host := v41DenseTestSession(t, m, newV41DenseTestBackend())
		v41ScoreOnly(host)
		v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
		prefill := len(b.calls)
		if prefill == 0 {
			t.Fatal("prefill skipped selected indexer query rotation")
		}
		v41ScoreTestBits(t, s.Step(4), host.Step(4))
		step := len(b.calls)
		if step <= prefill {
			t.Fatal("incremental step skipped selected query rotation")
		}
		v41ScoreTestBits(t, s.Prefill([]int{5, 6}), host.Prefill([]int{5, 6}))
		if len(b.calls) <= step || len(b.outputs) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
			t.Fatal("suffix rotation skipped, leaked, or changed continuation")
		}
		for _, c := range b.calls {
			if c.rows != m.Cfg.IndexNHeads || c.headDim != m.Cfg.IndexHeadDim || c.rotaryDim != m.Cfg.QKRopeHeadDim {
				t.Fatal("production query used attention geometry")
			}
			for _, x := range c.x {
				if math.Float32bits(x)&0xffff != 0 {
					t.Fatal("production query omitted BF16 staging")
				}
			}
		}
		v41IndexRoPECheckResources(t, b, false)
	})
	for _, fault := range []string{"error", "panic", "length", "prefix", "nonfinite"} {
		t.Run("selected-"+fault, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			b := newV41IndexRoPERecorder()
			s := v41DenseTestSession(t, m, b)
			v41ScoreOnly(s)
			s.v41State().gatherOps.inverseOutput = nil
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.failure, b.failCall = fault, len(b.calls)+1
			b.cause = &compute.BackendError{Backend: "query-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			err := recoverError(func() { s.Step(4) })
			var closed *BackendForwardOperationError
			var projected *V41ProjectionOperationError
			if !errors.As(err, &closed) || closed.Path != "v41-indexer-query-rope" || !errors.As(err, &projected) || projected.Leaf != "indexer.wq_b.weight" || !s.BackendSessionClosed() {
				t.Fatalf("query failure did not close its selected path: %v", err)
			}
			if (fault == "error" || fault == "panic") && !errors.Is(err, b.cause) {
				t.Fatal("backend cause lost")
			}
			if len(b.calls) != b.failCall || len(b.outputs) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("failed query replayed, leaked, or published continuation")
			}
			if retry := recoverError(func() { s.Step(4) }); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatal("closed query retried")
			}
			v41IndexRoPECheckResources(t, b, false)
		})
	}
	t.Run("unlatched-primary-panic", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41IndexRoPERecorder()
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		s.v41State().gatherOps.inverseOutput = nil
		s.Prefill([]int{1, 2, 3})
		before := captureV41ForwardSnapshot(s.v41Forward)
		primary := &BackendForwardOperationError{Backend: "query-recorder", Path: "upstream-selected-operation", Stage: "execution", Cause: ErrV41ForwardStage}
		b.failure, b.failCall, b.cause = "panic", len(b.calls)+1, primary
		if err := recoverError(func() { s.Step(4) }); err != primary || s.halFailure != primary || !s.BackendSessionClosed() {
			t.Fatalf("unlatched primary panic lost identity or closure: %v", err)
		}
		if len(b.calls) != b.failCall || len(b.outputs) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatal("primary panic published continuation or leaked")
		}
		if retry := recoverError(func() { s.Step(4) }); retry != primary || len(b.calls) != b.failCall {
			t.Fatal("primary panic did not prohibit replay")
		}
		v41IndexRoPECheckResources(t, b, false)
	})
	for _, cleanup := range []string{"flush", "free"} {
		t.Run("primary-panic-cleanup-"+cleanup, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			b := newV41IndexRoPERecorder()
			s := v41DenseTestSession(t, m, b)
			v41ScoreOnly(s)
			s.v41State().gatherOps.inverseOutput = nil
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			primary := &BackendForwardOperationError{Backend: "query-recorder", Path: "upstream-selected-operation", Stage: "execution", Cause: ErrV41ForwardStage}
			b.failure, b.failCall, b.cause, b.cleanupStage = "panic", len(b.calls)+1, primary, cleanup
			if err := recoverError(func() { s.Step(4) }); err != primary || s.halFailure != primary || !s.BackendSessionClosed() {
				t.Fatalf("secondary cleanup displaced primary or closure: %v", err)
			}
			if b.cleanupCalls == 0 || len(b.calls) != b.failCall || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("cleanup fault was not exercised or published continuation")
			}
			if retry := recoverError(func() { s.Step(4) }); retry != primary || len(b.calls) != b.failCall {
				t.Fatal("cleanup fault reopened the selected session")
			}
			v41IndexRoPECheckResources(t, b, cleanup == "free")
		})
	}
	t.Run("returned-error-cleanup-free", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41IndexRoPERecorder()
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		s.v41State().gatherOps.inverseOutput = nil
		s.Prefill([]int{1, 2, 3})
		before := captureV41ForwardSnapshot(s.v41Forward)
		primary := &compute.BackendError{Backend: "query-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		b.failure, b.failCall, b.cause, b.cleanupStage = "error", len(b.calls)+1, primary, "free"
		err := recoverError(func() { s.Step(4) })
		var closed *BackendForwardOperationError
		if !errors.Is(err, primary) || !errors.As(err, &closed) || closed.Path != "v41-indexer-query-rope" || s.halFailure != closed || !s.BackendSessionClosed() {
			t.Fatalf("secondary free displaced returned backend error: %v", err)
		}
		if b.cleanupCalls == 0 || len(b.calls) != b.failCall || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatal("returned-error cleanup fault was not exercised or published continuation")
		}
		if retry := recoverError(func() { s.Step(4) }); retry != closed || len(b.calls) != b.failCall {
			t.Fatal("returned backend error plus cleanup fault retried")
		}
		v41IndexRoPECheckResources(t, b, true)
	})
	t.Run("unsupported", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		s := v41DenseTestSession(t, m, newV41DenseTestBackend())
		ops := s.v41State().gatherOps
		if ops.indexRoPEFunc() != nil {
			t.Fatal("unsupported query primitive selected")
		}
		x := make([]float32, m.Cfg.IndexHeadDim)
		for i := range x {
			x[i] = float32(i+1) / 17
		}
		want := append([]float32(nil), x...)
		if err := m.v41IndexRoPE(0, 7, want, 1); err != nil {
			t.Fatal(err)
		}
		if err := m.v41IndexRoPEWithDevice(0, 7, x, 1, ops.indexRoPEFunc()); err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "portable fallback", x, want)
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		before := append([]float32(nil), x...)
		err := m.v41IndexRoPEWithDevice(0, 7, x, 1, ops.indexRoPEFunc())
		if !errors.Is(err, errV41DeviceOnlyHostOp) || !s.BackendSessionClosed() {
			t.Fatalf("missing selected query did not refuse: %v", err)
		}
		v41TailRoPECheckBits(t, "device-only atomic refusal", x, before)
	})
}
