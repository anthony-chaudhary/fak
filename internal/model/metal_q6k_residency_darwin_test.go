//go:build darwin && arm64 && cgo

package model

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

const metalQ6KConcurrencyTimeout = 15 * time.Second

func waitMetalQ6KConcurrency(t *testing.T, done <-chan struct{}, operation string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(metalQ6KConcurrencyTimeout):
		t.Fatalf("%s did not complete within %s", operation, metalQ6KConcurrencyTimeout)
	}
}

func requireMetalQ6KClose(t *testing.T, want, got []float32, context string) {
	t.Helper()
	if cos, rel := cosineAndMaxRel(want, got); cos < 0.9999 || rel > 5e-3 {
		t.Fatalf("%s: cosine=%g max-relative=%g", context, cos, rel)
	}
}

type metalQ6KOrderCloser struct {
	model    *Model
	wantLive int
	closed   bool
}

func (c *metalQ6KOrderCloser) Close() error {
	metalQ4KMu.Lock()
	_, cached := metalQ6KW[c.model]
	metalQ4KMu.Unlock()
	if cached {
		return fmt.Errorf("checkpoint owner closed before Q6_K cache deletion")
	}
	if got := metalgemm.LiveQ6KWeights(); got != c.wantLive {
		return fmt.Errorf("checkpoint owner closed before Q6_K handle release: live=%d want=%d", got, c.wantLive)
	}
	c.closed = true
	return nil
}

func TestMetalQ6KResidencySharesMTPHeadExecutesAndReleases(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ6KWeights()
	qt := randomQ6KTensor(8, qkK, 9061)
	target := &Model{kqw: map[string]*kQuantTensor{"lm_head.weight": qt}}
	draft := &Model{kqw: map[string]*kQuantTensor{"lm_head.weight": qt}}

	var targetWeight, draftWeight *metalgemm.Q6KWeight
	t.Cleanup(func() {
		releaseMetalQ4KResidency(target)
		releaseMetalQ4KResidency(draft)
		if targetWeight != nil {
			targetWeight.Release()
		}
		if draftWeight != nil {
			draftWeight.Release()
		}
	})

	targetWeight = target.metalQ6KWeight("lm_head.weight", qt)
	if targetWeight == nil {
		t.Fatal("target Q6_K upload returned nil")
	}
	firstID := targetWeight.ID()
	if got := metalgemm.LiveQ6KWeights(); got != base+1 {
		t.Fatalf("target upload live=%d want=%d", got, base+1)
	}
	draftWeight = draft.metalQ6KWeight("lm_head.weight", qt)
	if draftWeight == nil || draftWeight == targetWeight {
		t.Fatalf("draft alias=%p target=%p, want distinct handles", draftWeight, targetWeight)
	}
	if draftWeight.ID() != firstID {
		t.Fatalf("draft native id=%d want shared target id=%d", draftWeight.ID(), firstID)
	}
	if got := metalgemm.LiveQ6KWeights(); got != base+1 {
		t.Fatalf("draft alias performed a second upload: live=%d want=%d", got, base+1)
	}

	profiler := NewPhaseProfiler()
	s := &Session{M: draft, MetalQ4K: true, PhaseProfiler: profiler}
	x := randomVecF(qkK, 9062)
	got := make([]float32, qt.out)
	s.kQuantMatRowsIntoDispatch("lm_head.weight", qt, x, got)
	want := make([]float32, qt.out)
	kQuantMatRowsInto(qt, x, want)
	if cos, rel := cosineAndMaxRel(want, got); cos < 0.9999 || rel > 5e-3 {
		t.Fatalf("Q6_K Metal GEMV parity cos=%g rel=%g", cos, rel)
	}
	receipt, err := profiler.MetalExecutionReceipt()
	if err != nil {
		t.Fatalf("Q6_K execution receipt: %v", err)
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt); err != nil {
		t.Fatalf("Q6_K execution receipt validation: %v", err)
	}
	if len(receipt.Events) != 1 || receipt.Events[0].Operation != metalgemm.ExecutionQ6KGEMV ||
		!receipt.Events[0].Committed || !receipt.Events[0].CompletedWait || !receipt.Events[0].HostReadback {
		t.Fatalf("Q6_K GEMV did not produce an exact Metal lifecycle receipt: %+v", receipt.Events)
	}
	if profiler.MetalFallbackCount() != 0 {
		t.Fatalf("Q6_K GEMV recorded %d CPU fallbacks, want zero", profiler.MetalFallbackCount())
	}
	fallback, err := profiler.MetalFallbackReceipt()
	if err != nil || len(fallback.Events) != 0 || fallback.PromisedCPUFallbacks != 0 {
		t.Fatalf("Q6_K fallback receipt=%+v err=%v, want empty", fallback, err)
	}

	closer := &metalQ6KOrderCloser{model: target, wantLive: base + 1}
	target.SetWeightCloser(closer)
	if err := target.CloseWeights(); err != nil {
		t.Fatalf("target CloseWeights: %v", err)
	}
	if !closer.closed || targetWeight.ID() != -1 || draftWeight.ID() != firstID {
		t.Fatalf("target teardown order/alias lifetime: closer=%v targetID=%d draftID=%d", closer.closed, targetWeight.ID(), draftWeight.ID())
	}
	if got := metalgemm.LiveQ6KWeights(); got != base+1 {
		t.Fatalf("target release freed live draft alias: live=%d want=%d", got, base+1)
	}

	// Releasing the final model owner frees the native slot exactly once. Repeated teardown and
	// repeated per-handle Release are both no-ops, and the tombstoned slot is reused.
	releaseMetalQ4KResidency(draft)
	releaseMetalQ4KResidency(draft)
	draftWeight.Release()
	gotLive := metalgemm.LiveQ6KWeights()
	if draftWeight.ID() != -1 || gotLive != base {
		t.Fatalf("final/double release left stale residency: id=%d live=%d base=%d", draftWeight.ID(), gotLive, base)
	}

	reused := metalgemm.UploadQ6K(qt.raw, qt.out, qt.in)
	if reused == nil {
		t.Fatal("Q6_K upload after release returned nil")
	}
	defer reused.Release()
	if reused.ID() != firstID {
		t.Fatalf("released Q6_K slot not reused: id=%d want=%d", reused.ID(), firstID)
	}
}

func TestMetalQ6KConcurrentReleaseReuseNeverAliases(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ6KWeights()
	const out, in = 16, qkK
	first := randomQ6KTensor(out, in, 9161)
	second := randomQ6KTensor(out, in, 9162)
	x := randomVecF(in, 9163)
	wantFirst := make([]float32, out)
	wantSecond := make([]float32, out)
	kQuantMatRowsInto(first, x, wantFirst)
	kQuantMatRowsInto(second, x, wantSecond)

	// Each round races use of the old handle against its final release and immediate tombstone
	// reuse. The operation may run before release or observe the invalidated handle, but it must
	// never read the replacement weight through the reused native id.
	for round := 0; round < 32; round++ {
		weight := metalgemm.UploadQ6K(first.raw, out, in)
		if weight == nil {
			t.Fatalf("round %d: first upload returned nil", round)
		}
		oldID := weight.ID()
		got := make([]float32, out)
		for i := range got {
			got[i] = float32(math.NaN())
		}

		start := make(chan struct{})
		gemvDone := make(chan struct{})
		reusedCh := make(chan *metalgemm.Q6KWeight, 1)
		go func() {
			<-start
			weight.GEMV(x, got)
			close(gemvDone)
		}()
		go func() {
			<-start
			weight.Release()
			reusedCh <- metalgemm.UploadQ6K(second.raw, out, in)
		}()
		close(start)
		waitMetalQ6KConcurrency(t, gemvDone, fmt.Sprintf("round %d GEMV/release race", round))
		var reused *metalgemm.Q6KWeight
		select {
		case reused = <-reusedCh:
		case <-time.After(metalQ6KConcurrencyTimeout):
			t.Fatalf("round %d: release/reupload did not complete", round)
		}
		if reused == nil {
			t.Fatalf("round %d: replacement upload returned nil", round)
		}
		if got[0] == got[0] { // NaN means release won before the operation validated its handle.
			requireMetalQ6KClose(t, wantFirst, got, fmt.Sprintf("round %d old handle aliased replacement", round))
		}
		if gotID := reused.ID(); gotID != oldID {
			t.Fatalf("round %d: tombstone id=%d reused as %d", round, oldID, gotID)
		}
		gotReplacement := make([]float32, out)
		reused.GEMV(x, gotReplacement)
		requireMetalQ6KClose(t, wantSecond, gotReplacement, fmt.Sprintf("round %d replacement weight", round))
		reused.Release()
		if got := metalgemm.LiveQ6KWeights(); got != base {
			t.Fatalf("round %d: live Q6_K weights=%d want baseline %d", round, got, base)
		}
	}

	// Stress independent uploads, aliases, releases, native live-count reads, and slot reuse.
	const workers, rounds = 8, 24
	start := make(chan struct{})
	done := make(chan struct{})
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			<-start
			for round := 0; round < rounds; round++ {
				tensor := first
				if (worker+round)%2 != 0 {
					tensor = second
				}
				weight := metalgemm.UploadQ6K(tensor.raw, out, in)
				if weight == nil {
					errs <- fmt.Errorf("worker %d round %d: upload returned nil", worker, round)
					return
				}
				alias := weight.Share()
				if alias == nil || alias.ID() != weight.ID() {
					errs <- fmt.Errorf("worker %d round %d: invalid shared handle", worker, round)
					weight.Release()
					return
				}
				_ = metalgemm.LiveQ6KWeights()
				weight.Release()
				alias.Release()
			}
		}(worker)
	}
	close(start)
	go func() {
		wg.Wait()
		close(done)
	}()
	waitMetalQ6KConcurrency(t, done, "parallel Q6_K upload/share/release stress")
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got := metalgemm.LiveQ6KWeights(); got != base {
		t.Fatalf("parallel stress left %d live Q6_K weights, want baseline %d", got, base)
	}
}

func TestMetalQ6KResetCannotRetargetStaleHandle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	if live := metalgemm.LiveQ6KWeights(); live != 0 {
		t.Skipf("reset witness requires an empty Q6_K registry, found %d live weights", live)
	}
	const out, in = 16, qkK
	first := randomQ6KTensor(out, in, 9191)
	second := randomQ6KTensor(out, in, 9192)
	x := randomVecF(in, 9193)
	stale := metalgemm.UploadQ6K(first.raw, out, in)
	if stale == nil {
		t.Fatal("stale fixture upload returned nil")
	}
	oldID := stale.ID()
	metalgemm.ResetQ4K()
	if got := stale.ID(); got != -1 {
		t.Fatalf("pre-reset handle id=%d after reset, want invalid", got)
	}

	replacement := metalgemm.UploadQ6K(second.raw, out, in)
	if replacement == nil {
		t.Fatal("post-reset replacement upload returned nil")
	}
	defer replacement.Release()
	if got := replacement.ID(); got != oldID {
		t.Fatalf("reset tombstone id=%d reused as %d", oldID, got)
	}
	stale.Release() // Must not release the replacement that now occupies the old native id.
	if got := metalgemm.LiveQ6KWeights(); got != 1 {
		t.Fatalf("stale release changed replacement residency: live=%d want=1", got)
	}
	want := make([]float32, out)
	kQuantMatRowsInto(second, x, want)
	got := make([]float32, out)
	replacement.GEMV(x, got)
	requireMetalQ6KClose(t, want, got, "replacement after stale pre-reset release")
	replacement.Release()
	if got := metalgemm.LiveQ6KWeights(); got != 0 {
		t.Fatalf("reset alias witness left %d live Q6_K weights", got)
	}
}

func TestMetalQ6KGraphEncodeReleaseRetainsBoundWeight(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ6KWeights()
	const P, out, in = 2, 16, qkK
	tensor := randomQ6KTensor(out, in, 9171)
	x := randomVecF(P*in, 9172)
	want := make([]float32, P*out)
	kQuantMatRowsIntoBatch(tensor, x, P, want)

	type encodeResult struct {
		result *metalgemm.GraphResult
		err    error
	}
	for round := 0; round < 32; round++ {
		weight := metalgemm.UploadQ6K(tensor.raw, out, in)
		if weight == nil {
			t.Fatalf("round %d: upload returned nil", round)
		}
		graph, err := metalgemm.BeginProjectionGraph(x, nil, nil, P, in)
		if err != nil {
			weight.Release()
			t.Fatalf("round %d: begin graph: %v", round, err)
		}

		start := make(chan struct{})
		encoded := make(chan encodeResult, 1)
		released := make(chan struct{})
		go func() {
			<-start
			result, err := graph.EncodeQ6K(weight)
			encoded <- encodeResult{result: result, err: err}
		}()
		go func() {
			<-start
			weight.Release()
			close(released)
		}()
		close(start)

		var encodedResult encodeResult
		select {
		case encodedResult = <-encoded:
		case <-time.After(metalQ6KConcurrencyTimeout):
			graph.Free()
			t.Fatalf("round %d: graph encode/release race timed out", round)
		}
		waitMetalQ6KConcurrency(t, released, fmt.Sprintf("round %d graph release", round))
		if encodedResult.err == nil {
			got, receipt, err := graph.FinishRead(encodedResult.result)
			if err != nil {
				graph.Free()
				t.Fatalf("round %d: finish encoded graph: %v", round, err)
			}
			if !receipt.Committed || !receipt.CompletedWait {
				graph.Free()
				t.Fatalf("round %d: incomplete graph receipt: %+v", round, receipt)
			}
			requireMetalQ6KClose(t, want, got[0], fmt.Sprintf("round %d graph result after release", round))
		}
		graph.Free()
		if got := weight.ID(); got != -1 {
			t.Fatalf("round %d: released graph weight id=%d", round, got)
		}
		if got := metalgemm.LiveQ6KWeights(); got != base {
			t.Fatalf("round %d: graph race left %d live Q6_K weights, want baseline %d", round, got, base)
		}
	}
}

func TestMetalQ6KReversedDuplicateBatchReleaseDoesNotDeadlock(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ6KWeights()
	const H, I = qkK, qkK
	gate0 := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 9181).raw, I, H)
	gate1 := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 9182).raw, I, H)
	up0 := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 9183).raw, I, H)
	up1 := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 9184).raw, I, H)
	down0 := metalgemm.UploadQ6K(randomQ6KTensor(H, I, 9185).raw, H, I)
	down1 := metalgemm.UploadQ6K(randomQ6KTensor(H, I, 9186).raw, H, I)
	if gate0 == nil || gate1 == nil || up0 == nil || up1 == nil || down0 == nil || down1 == nil {
		if gate0 != nil {
			gate0.Release()
		}
		if gate1 != nil {
			gate1.Release()
		}
		if up0 != nil {
			up0.Release()
		}
		if up1 != nil {
			up1.Release()
		}
		if down0 != nil {
			down0.Release()
		}
		if down1 != nil {
			down1.Release()
		}
		t.Fatal("batch fixture upload returned nil")
	}
	defer gate0.Release()
	defer gate1.Release()
	defer up0.Release()
	defer up1.Release()

	x := randomVecF(H, 9187)
	forwardGate := []*metalgemm.Q4KWeight{gate0, gate1, gate0}
	forwardUp := []*metalgemm.Q4KWeight{up0, up1, up0}
	forwardDown := []*metalgemm.Q6KWeight{down0, down1, down0}
	reverseGate := []*metalgemm.Q4KWeight{gate1, gate0, gate1}
	reverseUp := []*metalgemm.Q4KWeight{up1, up0, up1}
	reverseDown := []*metalgemm.Q6KWeight{down1, down0, down1}
	wantForward := make([]float32, len(forwardDown)*H)
	wantReverse := make([]float32, len(reverseDown)*H)
	if !metalgemm.FusedMLPQ6DownBatch(forwardGate, forwardUp, forwardDown, x, wantForward) ||
		!metalgemm.FusedMLPQ6DownBatch(reverseGate, reverseUp, reverseDown, x, wantReverse) {
		t.Fatal("sequential batch reference declined")
	}

	// Both calls use every shared q4k.m scratch family. Repeating a simultaneous start makes
	// cross-call scratch corruption observable as numerical divergence, while the execution
	// mutex guarantees that each completed call owns scratch until its final host readback.
	for round := 0; round < 32; round++ {
		forward := make([]float32, len(wantForward))
		reverse := make([]float32, len(wantReverse))
		start := make(chan struct{})
		done := make(chan struct{})
		var forwardOK, reverseOK bool
		var concurrent sync.WaitGroup
		concurrent.Add(2)
		go func() {
			defer concurrent.Done()
			<-start
			forwardOK = metalgemm.FusedMLPQ6DownBatch(forwardGate, forwardUp, forwardDown, x, forward)
		}()
		go func() {
			defer concurrent.Done()
			<-start
			reverseOK = metalgemm.FusedMLPQ6DownBatch(reverseGate, reverseUp, reverseDown, x, reverse)
		}()
		close(start)
		go func() {
			concurrent.Wait()
			close(done)
		}()
		waitMetalQ6KConcurrency(t, done, fmt.Sprintf("concurrent Q4_K/Q6_K output round %d", round))
		if !forwardOK || !reverseOK {
			t.Fatalf("round %d: concurrent batches declined: forward=%v reverse=%v", round, forwardOK, reverseOK)
		}
		requireMetalQ6KClose(t, wantForward, forward, fmt.Sprintf("round %d forward concurrent batch", round))
		requireMetalQ6KClose(t, wantReverse, reverse, fmt.Sprintf("round %d reverse concurrent batch", round))
	}

	start := make(chan struct{})
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(4)
	go func() {
		defer wg.Done()
		<-start
		metalgemm.FusedMLPQ6DownBatch(forwardGate, forwardUp, forwardDown, x, make([]float32, len(forwardDown)*H))
	}()
	go func() {
		defer wg.Done()
		<-start
		metalgemm.FusedMLPQ6DownBatch(reverseGate, reverseUp, reverseDown, x, make([]float32, len(reverseDown)*H))
	}()
	go func() {
		defer wg.Done()
		<-start
		down0.Release()
	}()
	go func() {
		defer wg.Done()
		<-start
		down1.Release()
	}()
	close(start)
	go func() {
		wg.Wait()
		close(done)
	}()
	waitMetalQ6KConcurrency(t, done, "reversed/duplicate Q6_K batches with releases")
	if got := down0.ID(); got != -1 {
		t.Fatalf("first released batch weight id=%d", got)
	}
	if got := down1.ID(); got != -1 {
		t.Fatalf("second released batch weight id=%d", got)
	}
	if got := metalgemm.LiveQ6KWeights(); got != base {
		t.Fatalf("batch/release race left %d live Q6_K weights, want baseline %d", got, base)
	}
}
