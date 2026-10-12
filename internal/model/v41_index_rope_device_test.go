package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41IndexRoPERecorder struct {
	*v41InverseRowsTestBackend
	failure      string
	exact        bool
	cleanupStage string
	cleanupCalls int
	cleanupArmed bool
	resources    map[compute.Buffer]*v41IndexRoPEResource
	foreign      compute.Buffer
	foreignReads int
	foreignFrees int
}

type v41IndexRoPEResource struct {
	attempts int
	released bool
}

func newV41IndexRoPERecorder() *v41IndexRoPERecorder {
	return &v41IndexRoPERecorder{v41InverseRowsTestBackend: newV41InverseRowsTestBackend(), exact: true, resources: map[compute.Buffer]*v41IndexRoPEResource{}}
}

// The recorder uses the exact host oracle; this marker proves software wiring
// only. It does not qualify any physical backend.
func (b *v41IndexRoPERecorder) V41TailRoPEExactAvailable() bool { return b.exact }

func v41IndexRoPEOnly(s *Session) {
	v41ScoreOnly(s)
	// The exact marker also binds compressed publication. Keep these existing
	// query/inverse ownership witnesses isolated from that third row consumer.
	s.v41State().gatherOps.compressedPublicationRoPE = nil
}

// fak-test:runtime fast est=20ms lane=default
// fak-test:justify Pins exact capability selection and the BF16 subnormal boundary.
func TestV41IndexRoPEExactSelection(t *testing.T) {
	m := &Model{Cfg: Config{IndexHeadDim: 4, IndexNHeads: 1, QKRopeHeadDim: 2, RopeTheta: 256}}
	input := []float32{1, math.Float32frombits(0x80000000), math.Float32frombits(0x00400000), 0}
	for _, mode := range []string{"raw-only", "exact-unavailable", "exact"} {
		t.Run(mode, func(t *testing.T) {
			raw := newV41InverseRowsTestBackend()
			exact := newV41IndexRoPERecorder()
			exact.exact = mode == "exact"
			var backend compute.Backend = exact
			if mode == "raw-only" {
				backend = raw
			}
			s := &Session{M: m, Backend: backend}
			defer s.Close()
			rotate := s.v41IndexRoPEFunc()
			selected := mode == "exact"
			if (rotate != nil) != selected {
				t.Fatalf("query selection=%t, want exact capability=%t", rotate != nil, selected)
			}
			if s.BackendSessionClosed() || s.halFailure != nil {
				t.Fatal("capability absence retired a healthy session")
			}
			ops := &v41DeviceGatherOps{owner: s, indexRoPE: rotate}
			got := append([]float32(nil), input...)
			if err := m.v41IndexRoPEWithDevice(0, 0, got, 1, ops.indexRoPEFunc()); err != nil {
				t.Fatal(err)
			}
			// At position zero, [0x00400000, +0] remains [0x00400000, +0]
			// through separate products/additions and BF16 publication. Flushing
			// the first operand to zero loses 64 BF16 ulps, not a rounding tie.
			v41TailRoPECheckBits(t, "positive BF16 subnormal", got, input)
			if selected {
				if len(exact.calls) != 1 {
					t.Fatal("qualified subnormal query did not dispatch exactly once")
				}
				v41IndexRoPECheckResources(t, exact, false)
				return
			}
			if len(raw.calls) != 0 || len(exact.calls) != 0 || len(exact.resources) != 0 {
				t.Fatal("unqualified query dispatched or allocated device operands")
			}
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			got = append([]float32(nil), input...)
			err := m.v41IndexRoPEWithDevice(0, 0, got, 1, ops.indexRoPEFunc())
			if !errors.Is(err, errV41DeviceOnlyHostOp) || !s.BackendSessionClosed() {
				t.Fatalf("device-only unqualified query did not refuse typed: %v", err)
			}
			v41TailRoPECheckBits(t, "unqualified atomic refusal", got, input)
			if len(raw.calls) != 0 || len(exact.calls) != 0 || len(exact.resources) != 0 {
				t.Fatal("device-only refusal dispatched the unqualified primitive")
			}
		})
	}
}

func (b *v41IndexRoPERecorder) BeginBatch() {}

func (b *v41IndexRoPERecorder) FlushBatch() {
	if b.cleanupArmed && b.cleanupStage == "flush" {
		b.cleanupCalls++
		panic(errors.New("secondary flush failure"))
	}
}

func (b *v41IndexRoPERecorder) Read(x compute.Tensor) []float32 {
	if b.foreign != nil && x.Buf() == b.foreign {
		b.foreignReads++
	}
	return b.v41InverseRowsTestBackend.Read(x)
}

func (b *v41IndexRoPERecorder) Free(x compute.Tensor) {
	if b.foreign != nil && x.Buf() == b.foreign {
		b.foreignFrees++
	}
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
		case "alias-query":
			return x, nil
		case "alias-table":
			return table, nil
		case "foreign":
			out := compute.NewF32(b.Backend, []int{rows, hd}, v41TailRoPEExactOracle(call.x, call.table, hd, rd))
			b.foreign = out.Buf()
			return out, nil
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
		v41IndexRoPEOnly(s)
		s.v41State().gatherOps.inverseOutput = nil
		if s.v41State().gatherOps.indexRoPEFunc() == nil {
			t.Fatal("available query rows primitive was not bound")
		}
		host := v41DenseTestSession(t, m, newV41DenseTestBackend())
		v41IndexRoPEOnly(host)
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
	for _, fault := range []string{"error", "panic", "length", "prefix", "nonfinite", "alias-query", "alias-table", "foreign"} {
		t.Run("selected-"+fault, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			b := newV41IndexRoPERecorder()
			s := v41DenseTestSession(t, m, b)
			v41IndexRoPEOnly(s)
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
			if fault == "foreign" && (b.foreign == nil || b.foreignReads != 0 || b.foreignFrees != 0) {
				t.Fatalf("foreign output was not rejected before read/free: output=%v reads=%d frees=%d", b.foreign != nil, b.foreignReads, b.foreignFrees)
			}
			v41IndexRoPECheckResources(t, b, false)
		})
	}
	t.Run("unlatched-primary-panic", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41IndexRoPERecorder()
		s := v41DenseTestSession(t, m, b)
		v41IndexRoPEOnly(s)
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
			v41IndexRoPEOnly(s)
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
		v41IndexRoPEOnly(s)
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

// fak-test:runtime medium est=2s lane=default
// fak-test:justify Exercises restored ownership of both consumers of the row-RoPE primitive.
func TestV41IndexRoPERestoredGatherOwnership(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("restored query-RoPE software witness requires cpu-ref")
	}
	m := v41IndexerTestFixture(t)
	b := newV41IndexRoPERecorder()
	source := v41DenseTestSession(t, m, b)
	v41IndexRoPEOnly(source)
	// Leave both row-RoPE consumers enabled: the shared carrier must rebind
	// query rotation and inverse attention output to the destination session.
	if ops := source.v41State().gatherOps; ops.indexRoPEFunc() == nil || ops.inverseOutputFunc() == nil {
		t.Fatal("both row-RoPE callbacks must be selected")
	}
	host := v41DenseTestSession(t, m, newV41DenseTestBackend())
	v41IndexRoPEOnly(host)
	v41ScoreTestBits(t, source.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	target := v41DenseTestSession(t, m, b)
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	v41IndexRoPEOnly(target)
	source.Close()
	ops := target.v41State().gatherOps
	if target.v41Forward.callbackOwner != target || ops == nil || ops.owner != target ||
		ops.indexRoPEFunc() == nil || ops.inverseOutputFunc() == nil {
		t.Fatal("restored gather carrier retained the source owner or lost a callback")
	}
	query, inverse := ops.indexRoPE, ops.inverseOutput
	queryCalls, inverseCalls := 0, 0
	ops.indexRoPE = func(layer int, values, cos, sin []float32, heads, hd, rd int) ([]float32, error) {
		queryCalls++
		return query(layer, values, cos, sin, heads, hd, rd)
	}
	ops.inverseOutput = func(layer int, values, cos, sin []float32, heads, hd, rd int) ([]float32, error) {
		inverseCalls++
		return inverse(layer, values, cos, sin, heads, hd, rd)
	}
	v41ScoreTestBits(t, target.Prefill([]int{4}), host.Prefill([]int{4}))
	if queryCalls == 0 || inverseCalls == 0 {
		t.Fatal("restored suffix omitted one row-RoPE consumer")
	}
	priorQuery, priorInverse := queryCalls, inverseCalls
	v41ScoreTestBits(t, target.Step(5), host.Step(5))
	if queryCalls <= priorQuery || inverseCalls <= priorInverse || target.BackendSessionClosed() ||
		len(b.outputs) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
		t.Fatal("restored step lost target ownership, omitted a callback, leaked, or changed continuation")
	}
	v41IndexRoPECheckResources(t, b, false)
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Preserves a wrapped primary across actual query dispatch, retirement and retry.
func TestV41IndexRoPEWrappedPrimary(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("wrapped query-RoPE witness requires cpu-ref")
	}
	for _, restored := range []bool{false, true} {
		owner := "fresh"
		if restored {
			owner = "restored"
		}
		for _, cleanup := range []string{"none", "flush", "free"} {
			t.Run(owner+"/"+cleanup, func(t *testing.T) {
				m := v41IndexerTestFixture(t)
				b := newV41IndexRoPERecorder()
				source := v41DenseTestSession(t, m, b)
				v41IndexRoPEOnly(source)
				source.v41State().gatherOps.inverseOutput = nil
				source.Prefill([]int{1, 2, 3})
				if len(b.calls) == 0 {
					t.Fatal("fixture did not select real query rotation")
				}
				target := source
				if restored {
					snap, err := source.PrefixSnapshot()
					if err != nil {
						t.Fatal(err)
					}
					defer snap.Close()
					target = v41DenseTestSession(t, m, b)
					if err := snap.Restore(target); err != nil {
						t.Fatal(err)
					}
					v41IndexRoPEOnly(target)
					target.v41State().gatherOps.inverseOutput = nil
					source.Close()
				}
				// Teardown disarms only after assertions; a retirement regression must not
				// cause a second injected cleanup panic to abort unrelated subtests.
				t.Cleanup(func() { b.cleanupArmed = false })
				if target.v41State().gatherOps.owner != target || target.v41Forward.callbackOwner != target {
					t.Fatal("selected callback is not bound to current owner")
				}
				before := captureV41ForwardSnapshot(target.v41Forward)
				inner := &BackendForwardOperationError{Backend: "query-recorder", Path: "upstream-selected-operation", Stage: "execution", Cause: ErrV41ForwardStage}
				primary := fmt.Errorf("outer query attribution: %w", inner)
				b.failure, b.failCall, b.cause = "panic", len(b.calls)+1, primary
				if cleanup != "none" {
					b.cleanupStage = cleanup
				}
				first := recoverError(func() { target.Step(4) })
				if first != primary {
					t.Fatalf("first panic lost original wrapper: got %v, want exact %v", first, primary)
				}
				if target.halFailure != primary || !target.BackendSessionClosed() {
					t.Fatalf("retirement lost wrapped primary: latch=%v, closed=%t", target.halFailure, target.BackendSessionClosed())
				}
				if cleanup != "none" && b.cleanupCalls == 0 {
					t.Fatal("secondary cleanup failure was not exercised")
				}
				if len(b.calls) != b.failCall || !reflect.DeepEqual(before, captureV41ForwardSnapshot(target.v41Forward)) {
					t.Fatal("selected failure replayed or changed continuation")
				}
				if retry := recoverError(func() { target.Step(4) }); retry != primary || len(b.calls) != b.failCall {
					t.Fatal("retired owner did not rethrow the exact original wrapper")
				}
				v41IndexRoPECheckResources(t, b, cleanup == "free")
			})
		}
	}
}
