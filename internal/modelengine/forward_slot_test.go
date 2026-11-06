package modelengine

// forward_slot_test.go — Continual Engine phase 0 witnesses for ForwardSlot.
// Every result claimed here is [SW-VERIFIED] class only: synthetic in-memory
// checkpoint, no hardware, no performance claim.

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestForwardSlotInstallRevertRoundTrip witnesses the whole-old/whole-new
// round trip: install a candidate over the incumbent (Base=1), then revert
// back to the incumbent as a NEW strictly-monotonic generation. [SW-VERIFIED]
func TestForwardSlotInstallRevertRoundTrip(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	sess := m.NewSession()
	defer sess.Close()
	sess.Prefill([]int{1, 2, 3})

	var candidateCalls atomic.Int64
	candidate := ForwardImpl{
		Name: "candidate",
		Step: func(s *model.Session, id int) []float32 {
			candidateCalls.Add(1)
			return s.Step(id)
		},
		StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			candidateCalls.Add(1)
			return bs.StepBatch(ids)
		},
	}

	slot := NewForwardSlot(DefaultForwardImpl())
	if got := slot.Generation(); got != 1 {
		t.Fatalf("initial generation = %d, want 1", got)
	}

	rec, err := slot.Install(ForwardDelta{Base: 1, Impl: candidate, Reason: "round-trip"})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if rec.Outcome != ForwardSwapPromoted || rec.From != 1 || rec.To != 2 || rec.Impl != "candidate" {
		t.Fatalf("install receipt = %+v, want promoted 1->2 candidate", rec)
	}

	g := slot.Load()
	if g.Generation != 2 || g.Impl.Name != "candidate" {
		t.Fatalf("Load after install = gen %d name %q, want gen 2 candidate", g.Generation, g.Impl.Name)
	}
	if out := g.Impl.Step(sess, 5); len(out) != m.Cfg.VocabSize || candidateCalls.Load() != 1 {
		t.Fatalf("candidate Step: len=%d calls=%d, want vocab-sized logits and 1 call", len(out), candidateCalls.Load())
	}

	rev, err := slot.Revert()
	if err != nil {
		t.Fatalf("Revert: %v", err)
	}
	if rev.Outcome != ForwardSwapReverted || rev.From != 2 || rev.To != 3 || rev.Impl != "incumbent" {
		t.Fatalf("revert receipt = %+v, want reverted 2->3 incumbent", rev)
	}
	g2 := slot.Load()
	if g2.Generation != 3 || g2.Impl.Name != "incumbent" {
		t.Fatalf("Load after revert = gen %d name %q, want gen 3 incumbent", g2.Generation, g2.Impl.Name)
	}
	if out := g2.Impl.Step(sess, 5); len(out) != m.Cfg.VocabSize || candidateCalls.Load() != 1 {
		t.Fatalf("incumbent Step after revert: len=%d candidateCalls=%d, want vocab-sized logits and no candidate calls", len(out), candidateCalls.Load())
	}

	rec2, err := slot.Revert()
	if !errors.Is(err, ErrForwardSlotNoLastGood) {
		t.Fatalf("second Revert error = %v, want ErrForwardSlotNoLastGood", err)
	}
	if rec2.Outcome != ForwardSwapRefused || rec2.From != 3 || rec2.To != 3 || rec2.Impl != "" {
		t.Fatalf("second revert receipt = %+v, want refused 3->3 empty impl", rec2)
	}
	if st := slot.Stats(); st.Installs != 1 || st.Reverts != 1 || st.Rejects != 1 {
		t.Fatalf("Stats = %+v, want installs=1 reverts=1 rejects=1", st)
	}
}

// TestForwardSlotInstallFailClosedBaseMismatch witnesses the fail-closed CAS
// gate: a delta whose Base is not the live generation is refused, never
// merged, and leaves the live generation untouched. [SW-VERIFIED]
func TestForwardSlotInstallFailClosedBaseMismatch(t *testing.T) {
	slot := NewForwardSlot(DefaultForwardImpl())

	rec, err := slot.Install(ForwardDelta{Base: 7, Impl: DefaultForwardImpl(), Reason: "stale base"})
	if !errors.Is(err, ErrForwardSlotBaseMismatch) {
		t.Fatalf("Install error = %v, want ErrForwardSlotBaseMismatch", err)
	}
	if rec.Outcome != ForwardSwapRefused || rec.From != 1 || rec.To != 1 || rec.Impl != "" {
		t.Fatalf("refusal receipt = %+v, want refused 1->1 empty impl", rec)
	}
	g := slot.Load()
	if g.Generation != 1 || g.Impl.Name != "incumbent" {
		t.Fatalf("live generation after refused install = gen %d name %q, want gen 1 incumbent", g.Generation, g.Impl.Name)
	}
	if st := slot.Stats(); st.Rejects != 1 || st.Installs != 0 || st.Reverts != 0 {
		t.Fatalf("Stats = %+v, want rejects=1 installs=0 reverts=0", st)
	}
}

// TestForwardSlotInstallVerifyGateRefuses witnesses the optional Verify gate:
// a failing hook refuses the swap (wrapped sentinel, cause carried in the
// receipt) while a passing hook promotes. [SW-VERIFIED]
func TestForwardSlotInstallVerifyGateRefuses(t *testing.T) {
	slot := NewForwardSlot(DefaultForwardImpl())
	hookErr := errors.New("synthetic verify failure")

	rec, err := slot.Install(ForwardDelta{
		Base:   1,
		Impl:   DefaultForwardImpl(),
		Verify: func(candidate, incumbent ForwardImpl) error { return hookErr },
	})
	if !errors.Is(err, ErrForwardSlotVerify) || !errors.Is(err, hookErr) {
		t.Fatalf("Install error = %v, want wrapped ErrForwardSlotVerify carrying the hook error", err)
	}
	if rec.Outcome != ForwardSwapRefused || rec.From != 1 || rec.To != 1 {
		t.Fatalf("refusal receipt = %+v, want refused 1->1", rec)
	}
	if g := slot.Load(); g.Generation != 1 || g.Impl.Name != "incumbent" {
		t.Fatalf("live generation after verify refusal = gen %d name %q, want gen 1 incumbent", g.Generation, g.Impl.Name)
	}
	if st := slot.Stats(); st.Rejects != 1 || st.Installs != 0 {
		t.Fatalf("Stats = %+v, want rejects=1 installs=0", st)
	}

	rec2, err := slot.Install(ForwardDelta{
		Base: 1,
		Impl: ForwardImpl{Name: "verified", Step: DefaultForwardImpl().Step, StepBatch: DefaultForwardImpl().StepBatch},
		Verify: func(candidate, incumbent ForwardImpl) error {
			if candidate.Name != "verified" || incumbent.Name != "incumbent" {
				return errors.New("unexpected candidate/incumbent pair")
			}
			return nil
		},
		Reason: "gate passed",
	})
	if err != nil {
		t.Fatalf("Install with passing Verify: %v", err)
	}
	if rec2.Outcome != ForwardSwapPromoted || rec2.From != 1 || rec2.To != 2 || rec2.Impl != "verified" {
		t.Fatalf("promote receipt = %+v, want promoted 1->2 verified", rec2)
	}
	if st := slot.Stats(); st.Installs != 1 || st.Rejects != 1 {
		t.Fatalf("Stats = %+v, want installs=1 rejects=1", st)
	}
}

// TestForwardSlotInFlightContinuity witnesses the whole-old/whole-new contract:
// a generation record loaded BEFORE a swap keeps producing the OLD behavior
// for the rest of its step, while a fresh Load sees the NEW behavior.
// [SW-VERIFIED]
func TestForwardSlotInFlightContinuity(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	sess := m.NewSession()
	defer sess.Close()
	sess.Prefill([]int{1, 2, 3})

	sentinel := make([]float32, m.Cfg.VocabSize)
	for i := range sentinel {
		sentinel[i] = 1234.5
	}
	candidate := ForwardImpl{
		Name:      "sentinel",
		Step:      func(s *model.Session, id int) []float32 { return sentinel },
		StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 { return nil },
	}

	slot := NewForwardSlot(DefaultForwardImpl())
	held := slot.Load() // gen 1 incumbent, held across the swap
	if _, err := slot.Install(ForwardDelta{Base: 1, Impl: candidate, Reason: "behavior change"}); err != nil {
		t.Fatalf("Install: %v", err)
	}

	oldOut := held.Impl.Step(sess, 5) // old generation: real forward, session advances
	if reflect.DeepEqual(oldOut, sentinel) {
		t.Fatal("held pre-swap generation produced the new behavior")
	}
	newOut := slot.Load().Impl.Step(sess, 5) // new generation: sentinel, session untouched
	if !reflect.DeepEqual(newOut, sentinel) {
		t.Fatal("fresh Load after swap did not produce the new behavior")
	}
	oldOut2 := held.Impl.Step(sess, 5) // held generation still old on its next step
	if reflect.DeepEqual(oldOut2, sentinel) {
		t.Fatal("held pre-swap generation changed behavior mid-hold")
	}
	if len(oldOut) != m.Cfg.VocabSize || len(oldOut2) != m.Cfg.VocabSize {
		t.Fatalf("held-generation logits len = %d/%d, want %d", len(oldOut), len(oldOut2), m.Cfg.VocabSize)
	}
}

// TestForwardSlotGenerationsStrictlyMonotonic witnesses that generations only
// increase across interleaved installs and reverts (no ABA) and that receipts
// chain From == previous To. [SW-VERIFIED]
func TestForwardSlotGenerationsStrictlyMonotonic(t *testing.T) {
	slot := NewForwardSlot(DefaultForwardImpl())
	first, ok := slot.LastReceipt()
	if !ok || first.To != 1 || first.Outcome != ForwardSwapPromoted {
		t.Fatalf("constructor receipt = %+v ok=%v, want promoted 0->1", first, ok)
	}
	prevTo := first.To

	impl := func(name string) ForwardImpl {
		def := DefaultForwardImpl()
		return ForwardImpl{Name: name, Step: def.Step, StepBatch: def.StepBatch}
	}
	ops := []struct {
		do   func() (ForwardSwapReceipt, error)
		want string
	}{
		{do: func() (ForwardSwapReceipt, error) { return slot.Install(ForwardDelta{Base: 1, Impl: impl("a")}) }, want: "a"},
		{do: func() (ForwardSwapReceipt, error) { return slot.Install(ForwardDelta{Base: 2, Impl: impl("b")}) }, want: "b"},
		{do: slot.Revert, want: "a"},
		{do: func() (ForwardSwapReceipt, error) { return slot.Install(ForwardDelta{Base: 4, Impl: impl("c")}) }, want: "c"},
		// Second revert: prev holds gen 4 ("a"), so the promoted new generation
		// carries "a" again — never a flip-flop back through a rejected gen.
		{do: slot.Revert, want: "a"},
	}
	names := map[uint64]string{}
	for i, op := range ops {
		rec, err := op.do()
		if err != nil {
			t.Fatalf("op %d: %v", i, err)
		}
		if rec.From != prevTo {
			t.Fatalf("op %d receipt From=%d, want previous To=%d (chain broken)", i, rec.From, prevTo)
		}
		if rec.To <= rec.From {
			t.Fatalf("op %d receipt To=%d not greater than From=%d", i, rec.To, rec.From)
		}
		if rec.Impl != op.want {
			t.Fatalf("op %d promoted impl = %q, want %q", i, rec.Impl, op.want)
		}
		names[rec.To] = rec.Impl
		prevTo = rec.To
	}
	wantNames := map[uint64]string{2: "a", 3: "b", 4: "a", 5: "c", 6: "a"}
	for gen, want := range wantNames {
		if got := names[gen]; got != want {
			t.Fatalf("generation %d impl = %q, want %q", gen, got, want)
		}
	}
	if st := slot.Stats(); st.Installs != 3 || st.Reverts != 2 || st.Rejects != 0 {
		t.Fatalf("Stats = %+v, want installs=3 reverts=2 rejects=0", st)
	}
}

// fwdSchedFixture builds a standalone NativeScheduler over the synthetic
// checkpoint with n lanes already in the decode state, mirroring
// deferredLifecycleFixture. The run goroutine is never started; iterations are
// driven synchronously by the test.
func fwdSchedFixture(t *testing.T, prompts ...[]int) (*NativeScheduler, []*schedLane) {
	t.Helper()
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	s.started.Do(func() {})
	lanes := make([]*schedLane, 0, len(prompts))
	for i, p := range prompts {
		ctx, cancel := context.WithCancel(context.Background())
		sess := m.NewSession()
		ln := &schedLane{
			sched: s, ctx: ctx, cancel: cancel,
			sess: sess, logits: sess.Prefill(p),
			prompt: append([]int(nil), p...), promptLen: len(p),
			state: schedLaneDecode, seqNo: int64(i + 1),
			tokens: make(chan abi.EngineToken, 4), done: make(chan struct{}),
		}
		lanes = append(lanes, ln)
	}
	s.lanes = lanes
	t.Cleanup(func() {
		for _, ln := range lanes {
			if !ln.terminal {
				ln.Cancel()
				s.runIteration(true)
			}
		}
		s.Close()
	})
	return s, lanes
}

// driveLanes runs runIteration until every lane is terminal, draining each
// lane's token stream concurrently (finish closes the channel). donated=true
// routes the single-lane case through stepOnce's serial path; donated=false
// routes it through stepSolo.
func driveLanes(t *testing.T, s *NativeScheduler, lanes []*schedLane, donated bool) [][]int {
	t.Helper()
	var mu sync.Mutex
	toks := make([][]int, len(lanes))
	drained := make(chan struct{}, len(lanes))
	for i, ln := range lanes {
		go func(i int, ln *schedLane) {
			defer func() { drained <- struct{}{} }()
			for tok := range ln.tokens {
				mu.Lock()
				toks[i] = append(toks[i], tok.ID)
				mu.Unlock()
			}
		}(i, ln)
	}
	const capIters = 8 * genTokens
	for i := 0; i < capIters; i++ {
		allTerminal := true
		for _, ln := range lanes {
			if !ln.terminal {
				allTerminal = false
				break
			}
		}
		if allTerminal {
			break
		}
		s.runIteration(donated)
	}
	for _, ln := range lanes {
		if !ln.terminal {
			t.Fatalf("lane seqNo=%d did not reach terminal within %d iterations", ln.seqNo, capIters)
		}
	}
	for range lanes {
		<-drained
	}
	mu.Lock()
	defer mu.Unlock()
	return toks
}

// TestForwardSlotSchedulerDecodeRoutesThroughSlot is the wiring + parity
// witness. [SW-VERIFIED] Seam: runIteration on a synthetic NativeScheduler,
// the same seam nativesched_deferred_lifecycle_test.go drives. donated=true
// exercises stepOnce (single-lane Step path and the shared-batch StepBatch
// path); donated=false exercises stepSolo. All three wired decode-forward
// sites load the slot once per step, so a counting wrapper installed over
// DefaultForwardImpl() observes every user-path forward and must leave the
// token stream bit-identical to the baseline (default-impl) run.
func TestForwardSlotSchedulerDecodeRoutesThroughSlot(t *testing.T) {
	var stepCalls, batchCalls atomic.Int64
	def := DefaultForwardImpl()
	wrapper := ForwardImpl{
		Name: "counting-wrapper",
		Step: func(sess *model.Session, id int) []float32 {
			stepCalls.Add(1)
			return def.Step(sess, id)
		},
		StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			batchCalls.Add(1)
			return def.StepBatch(bs, ids)
		},
	}

	// Baseline: default impl, stepSolo fast path (donated=false).
	baseSolo, baseLanes := fwdSchedFixture(t, []int{1, 2, 3})
	baseToks := driveLanes(t, baseSolo, baseLanes, false)
	if len(baseToks[0]) != genTokens {
		t.Fatalf("baseline solo tokens = %d, want %d", len(baseToks[0]), genTokens)
	}

	// Wrapper installed, single-lane path via stepOnce (donated=true).
	solo, soloLanes := fwdSchedFixture(t, []int{1, 2, 3})
	rec, err := solo.fwd.Install(ForwardDelta{Base: 1, Impl: wrapper, Reason: "witness routing"})
	if err != nil || rec.Outcome != ForwardSwapPromoted {
		t.Fatalf("wrapper install: rec=%+v err=%v", rec, err)
	}
	soloToks := driveLanes(t, solo, soloLanes, true)
	// A lane that publishes genTokens tokens performs exactly genTokens-1
	// decode forwards: the bootstrap token comes from prefill logits, and the
	// final published token is retired from the last forward's logits without
	// a further forward.
	if stepCalls.Load() != int64(genTokens-1) || batchCalls.Load() != 0 {
		t.Fatalf("solo routing: stepCalls=%d batchCalls=%d, want %d/0", stepCalls.Load(), batchCalls.Load(), genTokens-1)
	}
	if !reflect.DeepEqual(soloToks[0], baseToks[0]) {
		t.Fatalf("decode parity vs baseline failed on solo path: %v vs %v", soloToks[0], baseToks[0])
	}
	if got := solo.fwd.Generation(); got != 2 {
		t.Fatalf("scheduler slot generation = %d, want 2", got)
	}

	// Revert restores the incumbent on the scheduler's slot.
	rrev, err := solo.fwd.Revert()
	if err != nil || rrev.Outcome != ForwardSwapReverted {
		t.Fatalf("scheduler revert: rec=%+v err=%v", rrev, err)
	}
	if g := solo.fwd.Load(); g.Generation != 3 || g.Impl.Name != "incumbent" {
		t.Fatalf("post-revert slot = gen %d name %q, want gen 3 incumbent", g.Generation, g.Impl.Name)
	}
	if st := solo.fwd.Stats(); st.Installs != 1 || st.Reverts != 1 {
		t.Fatalf("scheduler slot Stats = %+v, want installs=1 reverts=1", st)
	}

	// Batch path: two lanes share ONE StepBatch per step (donated=true).
	baseBatch, baseBatchLanes := fwdSchedFixture(t, []int{1, 2, 3}, []int{4, 5, 6})
	baseBatchToks := driveLanes(t, baseBatch, baseBatchLanes, true)
	if len(baseBatchToks[0]) != genTokens || len(baseBatchToks[1]) != genTokens {
		t.Fatalf("baseline batch tokens = %d/%d, want %d/%d", len(baseBatchToks[0]), len(baseBatchToks[1]), genTokens, genTokens)
	}

	batch, batchLanes := fwdSchedFixture(t, []int{1, 2, 3}, []int{4, 5, 6})
	if _, err := batch.fwd.Install(ForwardDelta{Base: 1, Impl: wrapper, Reason: "witness batch routing"}); err != nil {
		t.Fatalf("batch wrapper install: %v", err)
	}
	batchToks := driveLanes(t, batch, batchLanes, true)
	// Same off-by-one as the solo path: genTokens-1 shared steps; only the
	// wrapper-installed run counts (the baseline run ran on the default impl).
	if batchCalls.Load() != int64(genTokens-1) {
		t.Fatalf("batch routing: batchCalls=%d, want %d (one StepBatch per step)", batchCalls.Load(), genTokens-1)
	}
	if stepCalls.Load() != int64(genTokens-1) {
		t.Fatalf("batch routing leaked into the serial path: stepCalls=%d, want %d", stepCalls.Load(), genTokens-1)
	}
	if !reflect.DeepEqual(batchToks[0], baseBatchToks[0]) || !reflect.DeepEqual(batchToks[1], baseBatchToks[1]) {
		t.Fatalf("decode parity vs baseline failed on batch path: %v/%v vs %v/%v", batchToks[0], batchToks[1], baseBatchToks[0], baseBatchToks[1])
	}
}
