package modelengine

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// TestNativeSchedulerBatchesLanesAndFreesCancelled is the acceptance-#4 witness:
// the SAME abi.LifecycleEngine the per-request in-kernel engine implements also fits
// the continuous-batching shape. Three lanes are admitted and advanced by ONE shared
// StepBatch loop; cancelling one lane mid-run frees it (terminal context.Canceled +
// KV reclaim) WITHOUT disturbing the other two, which decode to completion.
func TestNativeSchedulerBatchesLanesAndFreesCancelled(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	defer s.Close()

	ctx := context.Background()
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}
	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := s.Admit(ctx, c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	const cancelIdx = 1
	const readBeforeCancel = 2

	// Drain the two survivor lanes fully in their own goroutines.
	counts := make([]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		if i == cancelIdx {
			continue
		}
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for range r.Tokens() {
				counts[i]++
			}
		}(i, r)
	}

	// Cancel the middle lane after reading a couple of tokens.
	cr := reqs[cancelIdx]
	got := 0
	for range cr.Tokens() {
		got++
		if got == readBeforeCancel {
			cr.Cancel()
			break
		}
	}
	for range cr.Tokens() { // drain residual so its lane retires
		got++
	}

	wg.Wait()

	receipt := s.SharedWorkReceipt()
	if receipt.Steps == 0 || receipt.Panels == 0 || receipt.MACs == 0 {
		t.Fatalf("scheduler admitted a batch but did not receipt shared model work: %+v", receipt)
	}
	// Survivors decode to completion, unaffected by the cancellation.
	for i := range reqs {
		if i == cancelIdx {
			continue
		}
		if counts[i] != genTokens {
			t.Fatalf("survivor lane %d streamed %d tokens, want %d", i, counts[i], genTokens)
		}
		res, err := reqs[i].Result()
		if err != nil {
			t.Fatalf("survivor lane %d Result: %v", i, err)
		}
		if res == nil || res.Status != abi.StatusOK {
			t.Fatalf("survivor lane %d result = %+v, want StatusOK", i, res)
		}
	}

	// The cancelled lane stopped early, ended Canceled, and reclaimed its slot.
	if got >= genTokens {
		t.Fatalf("cancelled lane did not stop early: streamed %d of %d", got, genTokens)
	}
	res, err := cr.Result()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled lane err = %v, want context.Canceled", err)
	}
	if res != nil {
		t.Fatalf("cancelled lane result = %+v, want nil", res)
	}
	ln, ok := cr.(*schedLane)
	if !ok {
		t.Fatalf("Admit returned %T, want *schedLane", cr)
	}
	if !ln.Reclaimed() {
		t.Fatal("cancelled lane did not signal KV reclaim")
	}
}

func TestNativeSchedulerReportsDeterministicCachePhaseLatency(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	defer s.Close()

	var clockMu sync.Mutex
	now := time.Unix(0, 0)
	s.now = func() time.Time {
		clockMu.Lock()
		defer clockMu.Unlock()
		current := now
		now = now.Add(2 * time.Millisecond)
		return current
	}

	req, err := s.Admit(context.Background(), inlineCall("search_flights", `{"from":"SFO"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	for range req.Tokens() {
	}
	if _, err := req.Result(); err != nil {
		t.Fatalf("Result: %v", err)
	}

	receipt := s.CachePhaseLatencyReceipt()
	if got, want := len(receipt.Phases), 3; got != want {
		t.Fatalf("phase cardinality = %d, want %d", got, want)
	}
	if got := receipt.Phases[0]; got.Phase != modelperfobs.CachePipelinePhasePrefill || got.Observations != 1 || got.Total != 2*time.Millisecond {
		t.Fatalf("prefill bucket = %+v, want one 2ms observation", got)
	}
	decode := receipt.Phases[1]
	if decode.Phase != modelperfobs.CachePipelinePhaseDecode || decode.Observations == 0 {
		t.Fatalf("decode bucket = %+v, want bounded non-empty decode observations", decode)
	}
	if decode.Total != time.Duration(decode.Observations)*2*time.Millisecond {
		t.Fatalf("decode total = %s, want %d deterministic 2ms observations", decode.Total, decode.Observations)
	}
	if receipt.Observations != receipt.Phases[0].Observations+decode.Observations || receipt.Total != receipt.Phases[0].Total+decode.Total {
		t.Fatalf("unlabeled receipt does not reconcile with known phases: %+v", receipt)
	}
}

// TestNativeSchedulerDecodeStableUIDOrder proves decode batch composition is a function
// of the stable request UID (schedLane.seqNo) and not of mutable scheduler slice position.
// Promotion appends, preemption removes from the middle, and readmission appends an OLDER
// victim behind a lane admitted after it, so slice order is not a function of the UID. When
// two ranks of a tensor-parallel deployment disagree about which request holds which batch
// slot, MoE expert and route selection diverge across ranks and a batched decode step
// disagrees with itself. This is a correctness invariant, not a speed item: no latency,
// throughput or token-rate outcome is claimed.
//
// Port of the stable-UID decode ordering convention from sgl-project/mini-sglang at pinned
// revision 9a91cfafe754aa85daee49998176275667eb58f2 (MIT), referenced at decode.py:32-35;
// the upstream file is not vendored.
//
// Batch composition is read from beforeModelExecute, which the scheduler invokes once per
// lane, in execLanes order, in the same loop that builds the BatchSession Seqs handed to
// StepBatch — so the observed sequence IS the batch position, and any reordering anywhere
// between admission and the batch call turns this red rather than being masked by a
// reordered internal slice.
func TestNativeSchedulerDecodeStableUIDOrder(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 8, BlockTokens: 8})
	defer s.Close()

	s.mu.Lock()
	first := decodeUIDOrderLane(t, s, []int{1, 2, 3})
	victim := decodeUIDOrderLane(t, s, []int{4, 5, 6})
	promoted := decodeUIDOrderLane(t, s, []int{7, 8, 9})
	s.lanes = append(s.lanes, first, victim)

	// Preemption-driven removal: the same two statements enforcePreemptionLocked makes
	// (preemptLaneLocked, then drop the victim from the running set), so the victim keeps
	// its restored-state round trip instead of being simulated.
	idx := -1
	for i, ln := range s.lanes {
		if ln == victim {
			idx = i
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		t.Fatal("victim lane was not running")
	}
	if err := s.preemptLaneLocked(victim); err != nil {
		s.mu.Unlock()
		t.Fatalf("preempt victim lane: %v", err)
	}
	s.lanes = append(s.lanes[:idx], s.lanes[idx+1:]...)

	// Promotion of a waiting lane: appended to the running set behind the older survivor.
	s.lanes = append(s.lanes, promoted)

	// Readmission of the preempted lane: appended AFTER the newer promoted lane, which is
	// exactly how slice order stops agreeing with the stable UID.
	s.readmitPreemptedLocked()
	sliceSeq := make([]int64, 0, len(s.lanes))
	for _, ln := range s.lanes {
		sliceSeq = append(sliceSeq, ln.seqNo)
	}
	stats := s.preemptStats
	s.mu.Unlock()

	if stats.Preemptions != 1 || stats.Readmitted != 1 || len(sliceSeq) != 3 {
		t.Fatalf("churn stats=%+v running=%d, want one preemption, one readmit, three running", stats, len(sliceSeq))
	}
	if sliceSeq[0] < sliceSeq[1] && sliceSeq[1] < sliceSeq[2] {
		t.Fatalf("churn left the running set ascending by UID (slice seqNo=%v), so this no longer exercises reordering", sliceSeq)
	}

	var batch []*schedLane
	s.mu.Lock()
	s.beforeModelExecute = func(kind nativeSchedulerEventKind, ln *schedLane) {
		if kind == nativeSchedulerEventDecode {
			batch = append(batch, ln)
		}
	}
	s.mu.Unlock()

	if _, idle, _ := s.runIteration(false); idle {
		t.Fatal("churned scheduler reported idle with three decode lanes")
	}

	s.mu.Lock()
	running := make([]int64, 0, len(s.lanes))
	for _, ln := range s.lanes {
		running = append(running, ln.seqNo)
	}
	s.mu.Unlock()

	if len(batch) != 3 {
		t.Fatalf("decode batch composed %d positions, want 3", len(batch))
	}
	got := make([]int64, 0, len(batch))
	for _, ln := range batch {
		got = append(got, ln.seqNo)
	}
	want := []int64{first.seqNo, victim.seqNo, promoted.seqNo}
	for i := 1; i < len(got); i++ {
		if got[i-1] >= got[i] {
			t.Fatalf("decode batch order = %v, want ascending stable request UID %v (running-set slice order was %v)", got, want, running)
		}
	}
	if got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("decode batch membership = %v, want the three admitted lanes %v", got, want)
	}
}

// TestNativeSchedulerImportedLanesCarryStableUID closes the hole that made the ordering
// above silently inert on the native P/D decode path: AdmitImported published lanes
// straight into the waiting queue without taking a stable admission identity, so every
// imported lane shared the zero key. With a shared key, decode-batch ordering collapses
// back to arrival position, and the preemption victim rule (mostRecentPreemptibleLaneLocked,
// which reads seqNo too) cannot order its candidates.
func TestNativeSchedulerImportedLanesCarryStableUID(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	defer s.Close()
	// Consume the start Once so admission never launches the run loop and the queued
	// imported lanes stay under this test's control.
	s.started.Do(func() {})

	ctx := context.Background()
	for _, prompt := range [][]int{{1, 2}, {3, 4}, {5, 6}} {
		sess := m.NewSession()
		logits := sess.Prefill(prompt)
		if _, err := s.AdmitImported(ctx, nil, ImportedSequence{Session: sess, Logits: logits, Prompt: prompt}); err != nil {
			t.Fatalf("AdmitImported %v: %v", prompt, err)
		}
	}

	s.mu.Lock()
	if len(s.waiting) != 3 {
		s.mu.Unlock()
		t.Fatalf("imported waiting lanes = %d, want 3", len(s.waiting))
	}
	imported := append([]*schedLane(nil), s.waiting...)
	// Churn: publish the imported lanes newest-first, so arrival position and stable UID
	// order disagree before stepOnce normalizes the batch.
	s.lanes = []*schedLane{imported[2], imported[1], imported[0]}
	s.waiting = nil
	seq := make([]int64, len(imported))
	for i, ln := range imported {
		seq[i] = ln.seqNo
	}
	s.mu.Unlock()

	for i, uid := range seq {
		if uid == 0 {
			t.Fatalf("imported lane %d has no stable admission UID (seqNo=%d); the shared zero key collapses decode-batch ordering", i, uid)
		}
		if i > 0 && seq[i-1] >= uid {
			t.Fatalf("imported lane stable UIDs = %v, want distinct ascending admission identities", seq)
		}
	}

	var batch []*schedLane
	s.mu.Lock()
	s.beforeModelExecute = func(kind nativeSchedulerEventKind, ln *schedLane) {
		if kind == nativeSchedulerEventDecode {
			batch = append(batch, ln)
		}
	}
	s.mu.Unlock()

	if _, idle, _ := s.runIteration(false); idle {
		t.Fatal("scheduler reported idle with three imported decode lanes")
	}
	if len(batch) != 3 {
		t.Fatalf("imported decode batch composed %d positions, want 3", len(batch))
	}
	for i, ln := range batch {
		if ln.seqNo != seq[i] {
			t.Fatalf("imported decode batch position %d holds UID %d, want %v ascending by stable admission identity", i, ln.seqNo, seq)
		}
	}
}

// decodeUIDOrderLane mints one admitted-shaped decode lane carrying the scheduler's own
// stable request UID, so batch ordering is asserted against a real admission identity.
// The caller must already hold s.mu: the lane is published into the same lock domain the
// scheduler composes a batch under.
func decodeUIDOrderLane(t *testing.T, s *NativeScheduler, prompt []int) *schedLane {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	sess := s.newLaneSession(false, NativeSessionFresh)
	logits := sess.Prefill(prompt)
	s.seqNo++
	return &schedLane{
		sched: s, ctx: ctx, cancel: cancel, sess: sess, logits: logits,
		prompt: append([]int(nil), prompt...), promptLen: len(prompt),
		state:  schedLaneDecode,
		seqNo:  s.seqNo,
		tokens: make(chan abi.EngineToken, 1),
		done:   make(chan struct{}),
		putCtx: ctx,
	}
}
