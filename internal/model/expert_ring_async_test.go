package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_async_test.go — the #5627 overlap primitive.
//
// R3/#5614 staged a layer's activated set before its first GEMM and could not report the one number
// it was asked for: the fraction of page-in latency overlapped with compute. Nothing in
// compute.Backend could say when an upload landed, so "issued early" was real and unobservable.
//
// These drive the REAL moeFFN.apply seam (as the R3 tests do) against a backend that advertises
// compute.AsyncUploader, so the fence discipline is exercised where it actually runs rather than by
// calling the ring directly.

// asyncTestFence is a Fence whose landing the test controls. `landed` starts true for a transfer
// that completed underneath earlier work, false for one the demand path must wait on.
type asyncTestFence struct {
	landed  bool
	waits   int
	release func() // applied on Wait, so a SKIPPED fence is observable rather than merely unsafe
}

func (f *asyncTestFence) Done() bool { return f.landed }
func (f *asyncTestFence) Wait() {
	f.waits++
	if !f.landed {
		f.landed = true
		if f.release != nil {
			f.release()
		}
	}
}

// expertAsyncBackend is the order-recording backend plus compute.AsyncUploader. It keys each
// in-flight fence by the uploaded tensor's Buffer identity, so MatMul can detect the one thing this
// extension could get catastrophically wrong: multiplying a weight whose transfer is not yet
// visible. That check is the reason a delayed fence makes the TEST fail instead of quietly
// producing wrong numbers.
type expertAsyncBackend struct {
	*expertOrderRecordingBackend
	landImmediately bool
	inflight        map[compute.Buffer]*asyncTestFence
	fences          []*asyncTestFence
	// unfencedGEMMs counts GEMMs issued against a weight whose own transfer had not landed.
	// It must be 0. Anything else is a data race the ring would hit on a real async backend.
	unfencedGEMMs int
}

func newExpertAsyncBackend(landImmediately bool) *expertAsyncBackend {
	return &expertAsyncBackend{
		expertOrderRecordingBackend: &expertOrderRecordingBackend{Backend: compute.Default()},
		landImmediately:             landImmediately,
		inflight:                    map[compute.Buffer]*asyncTestFence{},
	}
}

func (b *expertAsyncBackend) Name() string { return "cuda-test-async" }

func (b *expertAsyncBackend) UploadAsync(t compute.Tensor, as compute.Dtype) (compute.Tensor, compute.Fence) {
	out := b.expertOrderRecordingBackend.Upload(t, as)
	f := &asyncTestFence{landed: b.landImmediately}
	buf := out.Buf()
	if !b.landImmediately && buf != nil {
		b.inflight[buf] = f
		f.release = func() { delete(b.inflight, buf) }
	}
	b.fences = append(b.fences, f)
	return out, f
}

func (b *expertAsyncBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if buf := w.Buf(); buf != nil {
		if f, pending := b.inflight[buf]; pending && !f.landed {
			b.unfencedGEMMs++
		}
	}
	return b.expertOrderRecordingBackend.MatMul(w, x)
}

func expertAsyncSession(m *Model, ringBytes int64, landImmediately bool) (*Session, *expertAsyncBackend) {
	be := newExpertAsyncBackend(landImmediately)
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
	}, be
}

// The witness R3 could not produce: on a k>1 layer whose activated set is prefetched, the transfers
// that landed underneath earlier work are counted as overlapped, and the fraction is non-zero.
func TestExpertRingPrefetchRecordsOverlapOnAnAsyncBackend(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertAsyncSession(m, perWeight*3*K, true) // every transfer lands before it is demanded
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	if len(be.fences) == 0 {
		t.Fatal("no transfer went through UploadAsync; the async path was never taken and this proves nothing")
	}
	st := s.ExpertRing()
	if st.AsyncOverlapped == 0 {
		t.Fatalf("AsyncOverlapped=0 with %d async transfers issued: a prefetch whose transfers all landed before demand must read as overlap", len(be.fences))
	}
	if got := st.AsyncOverlapFraction(); got <= 0 {
		t.Fatalf("AsyncOverlapFraction=%v, want > 0 (overlapped=%d waited=%d)", got, st.AsyncOverlapped, st.AsyncWaited)
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed", be.unfencedGEMMs)
	}
}

// A synchronous backend must report 0 and stay byte-for-byte unchanged: no fences, no overlap, and
// a fraction of 0 that means "this backend cannot report overlap", not "no overlap was achieved".
func TestExpertRingReportsNoOverlapOnASynchronousBackend(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, _ := expertPrefetchSession(m, perWeight*3*K) // no AsyncUploader
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	st := s.ExpertRing()
	if st.AsyncOverlapped != 0 || st.AsyncWaited != 0 {
		t.Fatalf("synchronous backend reported overlapped=%d waited=%d, want 0/0", st.AsyncOverlapped, st.AsyncWaited)
	}
	if got := st.AsyncOverlapFraction(); got != 0 {
		t.Fatalf("AsyncOverlapFraction=%v on a backend with no fences, want 0", got)
	}
	if st.Prefetched == 0 {
		t.Fatal("nothing was prefetched; the synchronous path did not run the same seam and the comparison is empty")
	}
}

// The safety half, and the one the acceptance gate names: a transfer that has NOT landed must be
// waited on before its weight is multiplied. The backend flags any GEMM against an in-flight
// weight, so a skipped fence fails the test rather than silently producing wrong numbers.
func TestExpertRingAsyncNeverMultipliesBeforeItsFenceLands(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertAsyncSession(m, perWeight*3*K, false) // nothing lands until someone Waits
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	if len(be.fences) == 0 {
		t.Fatal("no transfer went through UploadAsync; the fence discipline was never exercised")
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed — the ring skipped a fence", be.unfencedGEMMs)
	}
	// Every issued transfer must end up settled: an un-awaited fence is device storage the ring
	// still believes is arriving.
	for i, f := range be.fences {
		if !f.landed {
			t.Fatalf("fence %d of %d was never satisfied; its weight is resident but not visible", i, len(be.fences))
		}
	}
	st := s.ExpertRing()
	if st.AsyncWaited == 0 {
		t.Fatalf("AsyncWaited=0 although no transfer landed on its own; the blocking path was not exercised (overlapped=%d)", st.AsyncOverlapped)
	}
}

// The R0 bit-equality claim has to survive the async path: a ring-served GEMM equals the
// synchronous one. Same weights, same input, both orderings — identical logits, not merely close.
func TestExpertRingAsyncIsBitEqualToTheSynchronousPath(t *testing.T) {
	const H, E, K = 256, 8, 4
	for _, land := range []bool{true, false} {
		m := expertPrefetchModel(t, H, E, K)
		perWeight := expertRingWeightBytes(t, m)
		in := expertRingTestInput(H)

		sync, _ := expertPrefetchSession(m, perWeight*3*K)
		wantOut := moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: sync})
		sync.Close()

		async, be := expertAsyncSession(m, perWeight*3*K, land)
		gotOut := moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: async})
		if be.unfencedGEMMs != 0 {
			t.Fatalf("land=%v: %d unfenced GEMM(s)", land, be.unfencedGEMMs)
		}
		async.Close()

		if len(gotOut) != len(wantOut) {
			t.Fatalf("land=%v: async produced %d values, sync %d", land, len(gotOut), len(wantOut))
		}
		for i := range wantOut {
			if gotOut[i] != wantOut[i] {
				t.Fatalf("land=%v: async output differs from the synchronous path at %d: %v != %v",
					land, i, gotOut[i], wantOut[i])
			}
		}
	}
}
