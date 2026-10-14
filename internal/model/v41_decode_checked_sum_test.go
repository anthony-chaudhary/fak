package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This backend records the real Step's transfers and ownership over cpu-ref
// storage. mHC arithmetic reuses the independent scalar test transcription;
// neither these operations nor their counts qualify native GPU throughput.
type v41CheckedSumRecorder struct {
	compute.Backend
	*v41DeferredLedger
	mhc                                            *v41MHCTestBackend
	owner                                          sync.Mutex
	alloc                                          map[compute.Buffer]int
	uploads, uploadBytes, reads, readBytes         int
	postUploads, sumReads, borrowedPosts           int
	sum                                            compute.Buffer
	checked                                        bool
	sumValues, postValues                          []float32
	postOperands                                   [][]float32
	postOutput                                     []float32
	fault, transport                               error
	faultSite, sharedPublish                       int
	nonfinite, postFault, typedPanic, cleanupFault bool
	posts                                          int
}

func newV41CheckedSumRecorder() *v41CheckedSumRecorder {
	b := &v41CheckedSumRecorder{Backend: compute.Default(), mhc: newV41MHCTestBackend(true), alloc: map[compute.Buffer]int{}}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41CheckedSumRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41CheckedSumRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41CheckedSumRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}
func (b *v41CheckedSumRecorder) tensor(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	b.alloc[x.Buf()] = 0
	return x
}
func (b *v41CheckedSumRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(x, dt, compute.MemoryWeights, "")
}
func (b *v41CheckedSumRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("checked-sum fixture requires F32 weights")
	}
	b.uploads++
	b.uploadBytes += 4 * x.Numel()
	if site == "V4.1 mHC post-mix x" {
		b.postUploads++
	}
	out := b.tensor(x.Shape, b.Backend.Read(x))
	if site == "V4.1 shared-add accumulator" {
		b.sum, b.checked = out.Buf(), false
	}
	return out
}
func (b *v41CheckedSumRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	y := b.Backend.MatMul(w, x)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41CheckedSumRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	y := b.Backend.BatchedMatMul(w, x, rows)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41CheckedSumRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	return b.SwiGLUWithLimit(g, u, 0)
}
func (b *v41CheckedSumRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	gate, up := b.Backend.Read(g), b.Backend.Read(u)
	out := make([]float32, len(gate))
	for i, a := range gate {
		c := up[i]
		if limit > 0 {
			a, c = min(a, limit), max(-limit, min(c, limit))
		}
		out[i] = a / (1 + float32(math.Exp(float64(-a)))) * c
	}
	return b.tensor(g.Shape, out)
}
func (b *v41CheckedSumRecorder) AddInPlace(acc, shared compute.Tensor) {
	if b.alloc[acc.Buf()] != 0 || b.alloc[shared.Buf()] != 0 {
		panic("add used released storage")
	}
	b.Backend.AddInPlace(acc, shared)
	if acc.Buf() == b.sum && b.nonfinite {
		b.Backend.Read(acc)[0] = float32(math.Inf(1))
	}
}
func (b *v41CheckedSumRecorder) Read(x compute.Tensor) []float32 {
	b.reads++
	b.readBytes += 4 * x.Numel()
	return b.Backend.Read(x)
}
func (b *v41CheckedSumRecorder) Free(x compute.Tensor) {
	if _, ok := b.alloc[x.Buf()]; ok {
		b.alloc[x.Buf()]++
	}
	delete(b.deferredPending, x.Buf())
	b.Backend.Free(x)
	if b.cleanupFault && x.Buf() == b.sum {
		if b.posts == 2 && !b.postFault && !b.typedPanic {
			panic(b.fault)
		}
		panic("sum cleanup secondary failure")
	}
}
func (b *v41CheckedSumRecorder) SupportsV41BF16Publish(n int) bool { return n > 0 }
func (b *v41CheckedSumRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	p, st, err := b.V41BF16PublishDeferred(x, site)
	if err != nil {
		return p, err
	}
	verdict, _, err := b.V41ReadDeferred([]compute.V41DeferredStatus{st}, nil)
	if err != nil {
		return p, err
	}
	return p, verdict[0]
}
func (b *v41CheckedSumRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if site == compute.V41BF16PublishSharedExpertProjection {
		b.sharedPublish++
		if b.sharedPublish == b.faultSite {
			b.deferredFaultAt, b.deferredFault = b.deferredPublishes+1, b.fault
		}
	}
	p, st, err := b.v41DeferredLedger.V41BF16PublishDeferred(x, site)
	b.alloc[p.Buf()], b.alloc[st.Record.Buf()] = 0, 0
	return p, st, err
}
func (b *v41CheckedSumRecorder) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	for _, row := range rows {
		b.reads++
		b.readBytes += 4 * row.Numel()
		if row.Buf() == b.sum {
			b.sumReads++
			b.deferredReadErr = b.transport
		}
	}
	verdicts, values, err := b.v41DeferredLedger.V41ReadDeferred(statuses, rows)
	if len(rows) == 1 && rows[0].Buf() == b.sum && err == nil {
		b.sumValues = append([]float32(nil), values[0]...)
		b.checked = true
		for _, e := range verdicts {
			b.checked = b.checked && e == nil
		}
		for _, v := range values[0] {
			b.checked = b.checked && finite32(v)
		}
	}
	return verdicts, values, err
}
func (b *v41CheckedSumRecorder) SupportsV41MHC() bool { return true }
func (b *v41CheckedSumRecorder) operand(x compute.Tensor) compute.Tensor {
	if b.alloc[x.Buf()] != 0 {
		panic("mHC used released operand")
	}
	return compute.NewF32(b.mhc, x.Shape, b.Backend.Read(x))
}
func (b *v41CheckedSumRecorder) result(out compute.Tensor, err error) (compute.Tensor, error) {
	if err != nil {
		return compute.Tensor{}, err
	}
	result := b.tensor(out.Shape, b.Backend.Read(out))
	b.mhc.Free(out)
	return result, nil
}
func (b *v41CheckedSumRecorder) V41MHCInitialStreams(x compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	return b.result(b.mhc.V41MHCInitialStreams(b.operand(x), rows, hidden))
}
func (b *v41CheckedSumRecorder) V41MHCSplit(m, s, base compute.Tensor, rows, iters int, eps float32) (compute.Tensor, error) {
	return b.result(b.mhc.V41MHCSplit(b.operand(m), b.operand(s), b.operand(base), rows, iters, eps))
}
func (b *v41CheckedSumRecorder) V41MHCRMSSplit(m, s, p compute.Tensor, rows, hidden, iters int, eps, rmsEps float32) (compute.Tensor, error) {
	return b.result(b.mhc.V41MHCRMSSplit(b.operand(m), b.operand(s), b.operand(p), rows, hidden, iters, eps, rmsEps))
}
func (b *v41CheckedSumRecorder) V41MHCPre(s, k compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	return b.result(b.mhc.V41MHCPre(b.operand(s), b.operand(k), rows, hidden))
}
func (b *v41CheckedSumRecorder) V41MHCPost(x, s, k compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	b.posts++
	if b.sum != nil {
		if !b.checked {
			panic("mHC post preceded the checked finite-only sum read")
		}
		b.postValues = append([]float32(nil), b.Backend.Read(x)...)
		b.postOperands = [][]float32{append([]float32(nil), b.Backend.Read(s)...), append([]float32(nil), b.Backend.Read(k)...)}
		if x.Buf() == b.sum {
			b.borrowedPosts++
		}
		if b.typedPanic {
			panic(b.fault)
		}
		if b.postFault {
			return compute.Tensor{}, b.fault
		}
	}
	out, err := b.result(b.mhc.V41MHCPost(b.operand(x), b.operand(s), b.operand(k), rows, hidden))
	if b.sum != nil && err == nil {
		b.postOutput = append([]float32(nil), b.Backend.Read(out)...)
	}
	return out, err
}

// Refs #12640. This is a software Step-path witness, not measured speedup.
// fak-test:runtime medium est=5s lane=default
func TestV41DecodeCheckedSumPost(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	if b := compute.Default(); b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("requires cpu-ref")
	}
	type fixture struct {
		m      *Model
		s      *Session
		b      *v41CheckedSumRecorder
		before *v41ForwardSnapshot
		peer   *Session
	}
	newLane := func(t *testing.T, fallback, role bool) fixture {
		t.Helper()
		var m *Model
		if role {
			m = v41IncrementalExpertFixture(t, true, false)
		} else {
			m = v41FullStepPatchedNorms(t)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		source := m.NewSession()
		source.Prefill([]int{1, 2})
		snapshot := captureV41ForwardSnapshot(source.v41State())
		source.Close()
		b := newV41CheckedSumRecorder()
		s := v41DenseTestSession(t, m, b)
		s.v41Forward = snapshot.clone().restore()
		st := s.v41State()
		if st.callbackOwner != s || st.gatherOps.owner != s {
			t.Fatal("snapshot callbacks were not rebound")
		}
		var peer *Session
		if fallback {
			// Same kernels and checked shared-add chain, but mHC belongs to a
			// distinct session on this backend. Its row must take the host upload.
			peer = v41DenseTestSession(t, m, b)
			device := *st.mhcDevice
			device.s = peer
			st.mhcDevice = &device
		}
		return fixture{m, s, b, captureV41ForwardSnapshot(st), peer}
	}
	assertReleased := func(t *testing.T, f fixture) {
		t.Helper()
		f.b.cleanupFault = false
		f.s.Close()
		if f.peer != nil {
			f.peer.Close()
		}
		if err := f.m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		for _, frees := range f.b.alloc {
			if frees != 1 {
				t.Fatalf("temporary or cached tensor free count=%d, want 1", frees)
			}
		}
		if len(f.b.deferredPending) != 0 {
			t.Fatal("pending publication records leaked")
		}
		if !f.b.owner.TryLock() {
			t.Fatal("execution owner leaked")
		}
		f.b.owner.Unlock()
	}
	for _, role := range []bool{false, true} {
		name := "plain_step_one_upload_and_exact_parity"
		if role {
			name = "role_step_one_upload_and_exact_parity"
		}
		t.Run(name, func(t *testing.T) {
			old, next := newLane(t, true, role), newLane(t, false, role)
			want, got := old.s.Step(3), next.s.Step(3)
			v41DeferredBitsEqual(t, "Step logits", got, want)
			v41DeferredBitsEqual(t, "unrounded MoE input", next.b.postValues, old.b.postValues)
			v41DeferredBitsEqual(t, "checked MoE read", next.b.postValues, next.b.sumValues)
			v41DeferredBitsEqual(t, "mHC panel", next.b.postOutput, old.b.postOutput)
			if v41BF16Widened(next.b.sumValues) || len(next.b.sumValues) != next.m.Cfg.HiddenSize {
				t.Fatal("fixture did not witness finite non-BF16 sum")
			}
			if !reflect.DeepEqual(old.b.postOperands, next.b.postOperands) || !reflect.DeepEqual(captureV41ForwardSnapshot(old.s.v41State()), captureV41ForwardSnapshot(next.s.v41State())) {
				t.Fatal("mHC operands or committed state changed")
			}
			if old.b.uploads-next.b.uploads != 1 || old.b.uploadBytes-next.b.uploadBytes != 4*next.m.Cfg.HiddenSize || old.b.postUploads != 2 || next.b.postUploads != 1 || next.b.borrowedPosts != 1 || old.b.borrowedPosts != 0 {
				t.Fatalf("uploads=%d→%d bytes=%d→%d post uploads=%d→%d borrowed=%d→%d", old.b.uploads, next.b.uploads, old.b.uploadBytes, next.b.uploadBytes, old.b.postUploads, next.b.postUploads, old.b.borrowedPosts, next.b.borrowedPosts)
			}
			if old.b.reads != next.b.reads || old.b.readBytes != next.b.readBytes || old.b.deferredFences != next.b.deferredFences || !reflect.DeepEqual(old.b.deferredSites, next.b.deferredSites) || old.b.sumReads != 1 || next.b.sumReads != 1 {
				t.Fatal("checked row, read bytes or publication order changed")
			}
			assertReleased(t, old)
			assertReleased(t, next)
		})
	}
	for _, kind := range []string{"earliest_publication", "transport_unknown", "nonfinite_sum", "post_failure", "typed_post_panic", "pending_panel_cleanup_failure"} {
		t.Run(kind, func(t *testing.T) {
			f := newLane(t, false, false)
			primary := errors.New(kind)
			f.b.fault = primary
			switch kind {
			case "earliest_publication":
				f.b.faultSite, f.b.postFault, f.b.cleanupFault = 2, true, true
			case "transport_unknown":
				f.b.faultSite, f.b.transport = 1, primary
				f.b.fault = errors.New("unobserved numeric verdict")
			case "nonfinite_sum":
				f.b.nonfinite = true
			case "post_failure":
				f.b.postFault, f.b.cleanupFault = true, true
			case "pending_panel_cleanup_failure":
				f.b.cleanupFault = true
			case "typed_post_panic":
				f.b.typedPanic = true
				f.b.fault = &BackendForwardOperationError{Backend: "test", Path: "incoming-post", Cause: primary}
			}
			got := recoverError(func() { f.s.Step(3) })
			if got == nil || !f.s.BackendSessionClosed() || !reflect.DeepEqual(captureV41ForwardSnapshot(f.s.v41Forward), f.before) {
				t.Fatalf("failure lost retirement or rollback: %v", got)
			}
			var op *BackendForwardOperationError
			if !errors.As(got, &op) && kind != "pending_panel_cleanup_failure" {
				t.Fatalf("missing operation attribution: %v", got)
			}
			switch kind {
			case "earliest_publication":
				var boundary *V41BF16BoundaryOperationError
				if !errors.Is(got, primary) || !errors.As(got, &boundary) || boundary.Site != "ffn.shared_experts.w3.weight projection" || f.b.posts != 1 {
					t.Fatalf("earliest issued verdict or site changed: %v", got)
				}
			case "transport_unknown":
				if !errors.Is(got, primary) || errors.Is(got, f.b.fault) || op.Path != "v41-shared-add" || f.b.posts != 1 {
					t.Fatalf("transport attribution changed: %v", got)
				}
			case "nonfinite_sum":
				var route *v4RouteError
				if !errors.As(got, &route) || route.Field != "shared_add" || op.Path != "v41-shared-add" || f.b.posts != 1 {
					t.Fatalf("nonfinite sum reached mHC or changed site: %v", got)
				}
			case "post_failure":
				var mhc *V41MHCOperationError
				if !errors.Is(got, primary) || !errors.As(got, &mhc) || op.Path != "v41-mhc-post-mix" || f.b.posts != 2 || f.b.borrowedPosts != 1 {
					t.Fatalf("mHC primary or wrapper changed: %v", got)
				}
			case "pending_panel_cleanup_failure":
				if got != primary || f.b.posts != 2 || f.b.borrowedPosts != 1 || len(f.b.postOutput) == 0 {
					t.Fatalf("pending panel primary or abort changed: %v", got)
				}
			case "typed_post_panic":
				if got != f.b.fault || f.s.halFailure != f.b.fault || f.b.posts != 2 {
					t.Fatalf("typed panic identity changed: %v", got)
				}
			}
			assertReleased(t, f)
		})
	}
	t.Run("cancel_after_resident_publish_rolls_back", func(t *testing.T) {
		f := newLane(t, false, false)
		scope := f.s.beginBackendExecution()
		defer func() {
			if scope.live {
				scope.finish()
			}
		}()
		var scratch v41ProjScratch
		tx, err := f.m.beginV41Step(3, f.s.v41State(), &scratch)
		if err != nil {
			t.Fatal(err)
		}
		if err = tx.advanceLayer(); err != nil {
			t.Fatal(err)
		}
		if f.b.borrowedPosts != 1 || tx.stats.Committed {
			t.Fatal("cancellation did not follow a real resident post")
		}
		row, err := tx.prepareHead()
		if err != nil {
			t.Fatal(err)
		}
		logits, err := tx.headProject(row)
		if err != nil {
			t.Fatal(err)
		}
		_, err = tx.finishHead(logits, func() error { return context.Canceled })
		if !errors.Is(err, context.Canceled) || !tx.closed || tx.stats.Committed || !reflect.DeepEqual(captureV41ForwardSnapshot(f.s.v41Forward), f.before) || scratch.gatherOps != nil || scratch.mhcCarry != nil {
			t.Fatalf("cancellation lost rollback or callback cleanup: %v", err)
		}
		scope.finish()
		assertReleased(t, f)
	})
	t.Run("existing_shared_add_controls", TestV41DecodeSharedAddResidentOwned)
	t.Run("existing_mhc_post_controls", TestV41MHCDeviceSeamRoutesEveryFullOp)

}
