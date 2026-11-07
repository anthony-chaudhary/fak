//go:build !(darwin && arm64 && cgo) && !(vulkan && (windows || linux) && cgo)

package modelengine

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// This host provider executes the real model dispatcher and scheduler ownership
// paths with synthetic logits. It is not a CUDA, physical paged-KV, numerical
// parity or performance witness. Native override builds keep their own dispatch.
type boundedQwenBackend struct {
	compute.Backend
	vocab       int
	path        string
	batches     [][]int
	scalarCalls int
	failAt      int
	failure     string
	onBatch     func(int)
	knownKV     map[compute.KVStore]bool
}

func (b *boundedQwenBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	kv := b.Backend.NewKV(cfg)
	if b.knownKV == nil {
		b.knownKV = make(map[compute.KVStore]bool)
	}
	b.knownKV[kv] = true
	return kv
}

func (b *boundedQwenBackend) Qwen35IndependentDecodeCohortOwned(kvs []compute.KVStore, states [][]compute.Qwen35SequenceState) bool {
	var ranges [][2]uintptr
	for i, kv := range kvs {
		if !b.knownKV[kv] {
			return false
		}
		for _, state := range states[i] {
			for _, tensor := range []compute.Tensor{state.Conv, state.Recurrent} {
				if tensor.Buf() == nil {
					continue
				}
				data := b.Backend.Read(tensor)
				if len(data) == 0 {
					return false
				}
				lo := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
				hi := lo + uintptr(len(data))*4
				for _, prior := range ranges {
					if lo < prior[1] && prior[0] < hi {
						return false
					}
				}
				ranges = append(ranges, [2]uintptr{lo, hi})
			}
		}
	}
	return true
}

func (*boundedQwenBackend) Name() string { return "bounded-qwen-host-test" }
func (*boundedQwenBackend) Qwen35GDNPath() string {
	return model.Qwen35GDNCapabilityIdentity
}
func (b *boundedQwenBackend) Qwen35IndependentDecodeBatchPath() string { return b.path }

func (b *boundedQwenBackend) Qwen35GDNDecode(
	input, qkv, z, beta, alpha, conv, aLog, bias, norm, output, convState, recurrentState compute.Tensor,
	keyHeads, valueHeads, keyDim, valueDim, kernel int, epsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	b.scalarCalls++
	return compute.NewF32(b.Backend, []int{input.Numel()}, make([]float32, input.Numel())), convState, recurrentState, nil
}

func (b *boundedQwenBackend) Qwen35IndependentDecodeBatch(req compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	ids := make([]int, len(req.Lanes))
	for i, lane := range req.Lanes {
		ids[i] = lane.TokenIDs[0]
	}
	b.batches = append(b.batches, ids)
	call := len(b.batches)
	if b.onBatch != nil {
		b.onBatch(call)
	}
	if call == b.failAt {
		switch b.failure {
		case "panic":
			panic("cohort-forward-panic")
		case "goexit":
			runtime.Goexit()
		case "decline":
			return compute.Qwen35IndependentDecodeBatchResult{}, &compute.Qwen35IndependentDecodeBatchError{
				Stage: "host-preflight", Cause: errors.New("unmutated cohort decline"),
			}
		}
	}
	flat := make([]float32, len(ids)*b.vocab)
	for i, lane := range req.Lanes {
		flat[i*b.vocab+(ids[i]+1)%b.vocab] = 1
		attention := 0
		for _, layer := range lane.Layers {
			if layer.Linear {
				continue
			}
			width := lane.NumKVHeads * lane.HeadDim
			row := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
			lane.KV.AppendKV(attention, row, row, row, lane.StartPos)
			b.Backend.Free(row)
			attention++
		}
	}
	if call == b.failAt && b.failure == "mutated_error" {
		return compute.Qwen35IndependentDecodeBatchResult{}, &compute.Qwen35IndependentDecodeBatchError{
			Stage: "host-regression", Mutated: true, Cause: errors.New("cohort mutation failed"),
		}
	}
	return compute.Qwen35IndependentDecodeBatchResult{
		Logits: compute.NewF32(b.Backend, []int{len(ids), b.vocab}, flat),
		Lanes:  len(ids), SharedProjectionPanels: 3, ProjectionMACs: int64(len(ids)) * 7,
	}, nil
}

func boundedQwenLane(t *testing.T, s *NativeScheduler, b *boundedQwenBackend) *schedLane {
	t.Helper()
	sess, err := s.m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.seqNo++
	logits := make([]float32, b.vocab)
	logits[int(s.seqNo)+2] = 1
	ln := &schedLane{
		sched: s, ctx: ctx, cancel: cancel, sess: sess, logits: logits,
		state: schedLaneDecode, seqNo: s.seqNo, putCtx: ctx,
		tokens: make(chan abi.EngineToken, 8), done: make(chan struct{}),
	}
	s.lanes = append(s.lanes, ln)
	return ln
}

func boundedQwenFixture(t *testing.T, n int) (*NativeScheduler, *boundedQwenBackend, []*schedLane) {
	t.Helper()
	m := model.NewSynthetic(model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
		LayerTypes:          []string{"linear_attention", "full_attention"},
		LinearConvKernelDim: 3, LinearKeyHeadDim: 8, LinearNumKeyHeads: 2,
		LinearValueHeadDim: 8, LinearNumValueHeads: 4, FullAttentionInterval: 2,
	})
	s := NewNativeScheduler(m)
	s.started.Do(func() {}) // This fixture owns the serialized executor.
	s.recorder = enginestep.New(128)
	b := &boundedQwenBackend{Backend: compute.Default(), vocab: m.Cfg.VocabSize, path: compute.Qwen35IndependentDecodeBatchPath}
	lanes := make([]*schedLane, n)
	for i := range lanes {
		lanes[i] = boundedQwenLane(t, s, b)
	}
	// Running-set order is deliberately different from stable admission UID.
	for i, j := 0, len(s.lanes)-1; i < j; i, j = i+1, j-1 {
		s.lanes[i], s.lanes[j] = s.lanes[j], s.lanes[i]
	}
	t.Cleanup(func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, ln := range s.lanes {
			ln.finish(nil, context.Canceled)
		}
	})
	return s, b, lanes
}

func boundedQwenStep(s *NativeScheduler, iteration uint64) {
	s.executor.Lock()
	defer s.executor.Unlock()
	s.stepOnce(append([]*schedLane(nil), s.lanes...), iteration)
}

func boundedQwenWidths(b *boundedQwenBackend) []int {
	if len(b.batches) == 0 {
		return nil
	}
	out := make([]int, len(b.batches))
	for i := range out {
		out[i] = len(b.batches[i])
	}
	return out
}

type boundedQwenRetireContext struct {
	context.Context
	err func() error
}

func (c *boundedQwenRetireContext) Err() error { return c.err() }

func boundedQwenRetirementUnwind(t *testing.T, boundary, failure string) {
	t.Helper()
	count := 2
	if boundary == "terminal" {
		count = 1
	}
	s, b, lanes := boundedQwenFixture(t, count)
	first, owner := lanes[0], lanes[0].sess
	owners := make([]*model.Session, len(lanes))
	for i, ln := range lanes {
		owners[i] = ln.sess
	}
	closed := make(map[*model.Session]int)
	s.closeSession = func(sess *model.Session) {
		closed[sess]++
		sess.Close()
		if sess == owner {
			if failure == "goexit" {
				runtime.Goexit()
			}
			panic("publication retirement panic")
		}
	}
	defer func() { s.closeSession = func(sess *model.Session) { sess.Close() } }()
	if boundary == "terminal" {
		if !first.stageDeferredToken() {
			t.Fatal("failed to stage terminal control")
		}
		first.deferredToken.done = true
	} else {
		// The first Err after forwarding is inside publishDecodeSession. The
		// second is retirement after its real session was restored. Cancellation
		// is deterministic at that boundary, not a wall-clock race.
		afterForward, queries := false, 0
		ctx := first.ctx
		first.ctx = &boundedQwenRetireContext{Context: ctx, err: func() error {
			if afterForward {
				queries++
				if queries == 2 && boundary == "before_select" {
					first.cancel()
				}
			}
			return ctx.Err()
		}}
		b.onBatch = func(int) { afterForward = true }
		if boundary == "during_select" {
			first.tokens = make(chan abi.EngineToken)
			s.now = func() time.Time {
				if queries == 2 {
					first.cancel()
				}
				return time.Unix(100, 0)
			}
		}
	}
	finished := make(chan struct{})
	var raised any
	returned := false
	go func() {
		defer close(finished)
		defer func() { raised = recover() }()
		boundedQwenStep(s, 0)
		returned = true
	}()
	<-finished // The parent subprocess deadline contains a leaked-mutex regression.
	if returned || (failure == "panic" && raised != "publication retirement panic") || (failure == "goexit" && raised != nil) {
		t.Fatalf("retirement outcome returned=%t panic=%v", returned, raised)
	}
	if !s.mu.TryLock() {
		t.Fatal("retirement abandoned the scheduler mutex")
	}
	s.mu.Unlock()
	if len(closed) != count || closed[owner] != 1 {
		t.Fatalf("borrowed owners not retired exactly once: closes=%v", closed)
	}
	for i, ln := range lanes {
		if closed[owners[i]] != 1 {
			t.Fatalf("lane %d owner retired %d times", i, closed[owners[i]])
		}
		select {
		case <-ln.done:
		default:
			t.Fatalf("lane %d terminal edge stranded", i)
		}
		if result, err := ln.Result(); result != nil || err == nil || ln.inflightSess != nil || ln.sess != nil {
			t.Fatalf("lane %d reported success or retained an owner: result=%v err=%v", i, result, err)
		}
		if i == 0 && ln.Reclaimed() {
			t.Fatal("interrupted retirement reported reclaimed")
		}
	}
}

// The six contained retirement controls add about six seconds under -race,
// including the race runtime's normal process-exit delay.
// fak-test:runtime fast est=8s lane=default
func TestNativeSchedulerBoundedDecodeCohorts(t *testing.T) {
	for _, boundary := range []string{"before_select", "during_select", "terminal"} {
		for _, failure := range []string{"panic", "goexit"} {
			name := "retire_" + boundary + "_" + failure
			t.Run(name, func(t *testing.T) {
				const childKey = "FAK_TEST_COHORT_RETIREMENT_CHILD"
				if os.Getenv(childKey) == name {
					boundedQwenRetirementUnwind(t, boundary, failure)
					return
				}
				// A broken cleanup can strand the fixture's mutex and its cleanup.
				// Isolate that regression instead of force-unlocking another owner.
				executable, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				cmd := exec.CommandContext(ctx, executable, "-test.v", "-test.run=^TestNativeSchedulerBoundedDecodeCohorts$/^"+name+"$")
				cmd.Env = append(os.Environ(), childKey+"="+name)
				cmd.WaitDelay = time.Second
				output, err := cmd.CombinedOutput()
				t.Logf("retirement child:\n%s", output)
				if err != nil {
					t.Fatalf("retirement child failed: %v (deadline=%v)\n%s", err, ctx.Err(), output)
				}
			})
		}
	}
	for _, mode := range []string{"full", "unbuffered", "aliased"} {
		t.Run("outbox_"+mode, func(t *testing.T) {
			s, b, lanes := boundedQwenFixture(t, 9)
			switch mode {
			case "full":
				lanes[0].tokens = make(chan abi.EngineToken, 1)
				lanes[0].tokens <- abi.EngineToken{ID: -99}
			case "unbuffered":
				lanes[0].tokens = make(chan abi.EngineToken)
			case "aliased":
				original := lanes[8].tokens
				lanes[8].tokens = lanes[0].tokens
				defer func() { lanes[8].tokens = original }()
			}
			var offered []int
			modelDone, finished := make(chan struct{}, 1), make(chan struct{})
			base := DefaultForwardImpl()
			s.fwd = NewForwardSlot(ForwardImpl{Name: "observe-outbox", Step: base.Step, StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
				offered = append(offered, len(ids))
				out := base.StepBatch(bs, ids)
				select {
				case modelDone <- struct{}{}:
				default:
				}
				return out
			}})
			var raised any
			go func() {
				defer close(finished)
				defer func() { raised = recover() }()
				boundedQwenStep(s, 0)
			}()
			// Model work must precede release of the obstructed publication.
			select {
			case <-modelDone:
			case <-finished:
				select {
				case <-modelDone:
				default:
					t.Fatalf("no completed model forward: %v", raised)
				}
			case <-time.After(time.Second):
				t.Fatal("model forward blocked before publication")
			}
			if mode != "aliased" {
				<-lanes[0].tokens
			}
			<-finished
			if raised != nil || !reflect.DeepEqual(offered, []int{9}) || len(b.batches) != 0 {
				t.Fatalf("backpressure changed forwarding: offered=%v batches=%v panic=%v", offered, b.batches, raised)
			}
			for i, ln := range lanes {
				if !reflect.DeepEqual(ln.gen, []int{i + 3}) {
					t.Fatalf("lane %d gen=%v", i, ln.gen)
				}
			}
		})
	}
	for _, tc := range []struct {
		n      int
		widths []int
	}{{1, nil}, {2, []int{2}}, {8, []int{8}}, {9, []int{7, 2}}, {16, []int{8, 8}}, {17, []int{8, 7, 2}}} {
		t.Run(fmt.Sprintf("width_%d", tc.n), func(t *testing.T) {
			s, b, lanes := boundedQwenFixture(t, tc.n)
			for iteration := 0; iteration < 2; iteration++ {
				if tc.n == 1 {
					boundedQwenStep(s, uint64(iteration))
				} else {
					// Real outer iteration observes the offered cohort separately.
					s.executor.Lock()
					s.runIteration(true)
					s.executor.Unlock()
				}
			}
			wantWidths := append(append([]int(nil), tc.widths...), tc.widths...)
			if !reflect.DeepEqual(boundedQwenWidths(b), wantWidths) {
				t.Fatalf("model call widths=%v want=%v", boundedQwenWidths(b), wantWidths)
			}
			for i, ln := range lanes {
				if ln.terminal || ln.inflightSess != nil || ln.sess == ln.statsSess || ln.sess.SequencePosition() != 2 || len(ln.gen) != 2 {
					t.Fatalf("lane %d not advanced/restored exactly twice: gen=%v", i, ln.gen)
				}
				if tc.n > 1 && !reflect.DeepEqual(ln.gen, []int{3 + i, 4 + i}) {
					t.Fatalf("lane %d output/continuation=%v", i, ln.gen)
				}
			}
			if tc.n > 1 {
				var ids []int
				for _, group := range b.batches[:len(tc.widths)] {
					ids = append(ids, group...)
				}
				for i, id := range ids {
					if id != i+3 {
						t.Fatalf("unstable UID order: %v", ids)
					}
				}
				snap := s.recorder.Snapshot(0, "")
				if snap.Cohorts.Steps != 2 || snap.Cohorts.MaxLanes != float64(tc.n) ||
					snap.Decode[enginestep.PathBatched].Steps != uint64(len(wantWidths)) ||
					s.sharedPanels != uint64(len(wantWidths)*3) || s.sharedMACs != uint64(tc.n*2*7) {
					t.Fatalf("offered/executed receipts disagree: %+v panels=%d macs=%d", snap, s.sharedPanels, s.sharedMACs)
				}
			}
		})
	}

	t.Run("unsupported_unsplit_and_truthful_fallback", func(t *testing.T) {
		s, b, lanes := boundedQwenFixture(t, 9)
		b.path = "wrong"
		var offered []int
		base := DefaultForwardImpl()
		s.fwd = NewForwardSlot(ForwardImpl{Name: "observe", Step: base.Step, StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			offered = append(offered, len(ids))
			return base.StepBatch(bs, ids)
		}})
		boundedQwenStep(s, 0)
		if !reflect.DeepEqual(offered, []int{9}) || len(b.batches) != 0 || b.scalarCalls != 9 {
			t.Fatalf("unsupported dispatch changed: offered=%v batches=%v scalar=%d", offered, boundedQwenWidths(b), b.scalarCalls)
		}
		snap := s.recorder.Snapshot(0, "")
		if snap.Decode[enginestep.PathSerial].Tokens != 9 || snap.Decode[enginestep.PathBatched].Tokens != 0 {
			t.Fatalf("false shared execution: %+v", snap.Decode)
		}
		for _, ln := range lanes {
			if ln.sess.SequencePosition() != 1 || len(ln.gen) != 1 {
				t.Fatal("fallback lost or replayed a lane")
			}
		}
	})

	for _, where := range []string{"before_first", "during_earlier", "before_later_submit"} {
		t.Run("cancel_"+where, func(t *testing.T) {
			s, b, lanes := boundedQwenFixture(t, 9)
			canceled := lanes[8]
			switch where {
			case "before_first":
				canceled.Cancel()
			case "during_earlier":
				b.onBatch = func(call int) {
					if call == 1 {
						if lanes[7].inflightSess != nil || lanes[8].inflightSess != nil {
							t.Fatal("borrowed a future group's owner")
						}
						canceled.Cancel()
					}
				}
			case "before_later_submit":
				s.beforeModelExecute = func(_ nativeSchedulerEventKind, ln *schedLane) {
					if ln == lanes[7] {
						canceled.Cancel()
					}
				}
			}
			boundedQwenStep(s, 0)
			if !canceled.terminal || !errors.Is(canceled.err, context.Canceled) || len(canceled.gen) != 0 {
				t.Fatal("canceled lane forwarded or published")
			}
			for _, ln := range lanes[:8] {
				if ln.terminal || ln.inflightSess != nil || ln.sess.SequencePosition() != 1 || len(ln.gen) != 1 {
					t.Fatal("survivor replayed, skipped or leaked")
				}
			}
			if where != "before_first" && (!reflect.DeepEqual(boundedQwenWidths(b), []int{7}) || b.scalarCalls != 1) {
				t.Fatalf("later cancellation not rechecked: widths=%v scalar=%d", boundedQwenWidths(b), b.scalarCalls)
			}
		})
	}

	t.Run("ragged_retirement_and_arrival", func(t *testing.T) {
		s, b, lanes := boundedQwenFixture(t, 9)
		boundedQwenStep(s, 0)
		lanes[0].deferredToken.done = true
		arrival := boundedQwenLane(t, s, b)
		boundedQwenStep(s, 1)
		if !lanes[0].terminal || len(lanes[0].gen) != 2 || arrival.sess.SequencePosition() != 1 {
			t.Fatal("ragged terminal token or arrival lost")
		}
		if !reflect.DeepEqual(boundedQwenWidths(b), []int{7, 2, 7, 2}) {
			t.Fatalf("ragged/arrival groups=%v", boundedQwenWidths(b))
		}
		for _, ln := range lanes[1:] {
			if ln.sess.SequencePosition() != 2 || len(ln.gen) != 2 {
				t.Fatal("continuing lane skipped or replayed")
			}
		}
	})

	t.Run("one_generation_per_iteration", func(t *testing.T) {
		s, _, _ := boundedQwenFixture(t, 9)
		base := DefaultForwardImpl()
		oldCalls, newCalls := 0, 0
		next := ForwardImpl{Name: "new", Step: base.Step, StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			newCalls++
			return base.StepBatch(bs, ids)
		}}
		s.fwd = NewForwardSlot(ForwardImpl{Name: "old", Step: base.Step, StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			oldCalls++
			if oldCalls == 1 {
				if _, err := s.fwd.Install(ForwardDelta{Base: s.fwd.Generation(), Impl: next}); err != nil {
					t.Fatal(err)
				}
			}
			return base.StepBatch(bs, ids)
		}})
		boundedQwenStep(s, 0)
		if oldCalls != 2 || newCalls != 0 {
			t.Fatalf("mixed generations old=%d new=%d", oldCalls, newCalls)
		}
		boundedQwenStep(s, 1)
		if oldCalls != 2 || newCalls != 2 {
			t.Fatalf("next iteration missed new generation old=%d new=%d", oldCalls, newCalls)
		}
	})

	t.Run("later_unmutated_decline", func(t *testing.T) {
		s, b, lanes := boundedQwenFixture(t, 9)
		b.failAt, b.failure = 2, "decline"
		boundedQwenStep(s, 0)
		if !reflect.DeepEqual(boundedQwenWidths(b), []int{7, 2}) || b.scalarCalls != 2 {
			t.Fatalf("decline replayed completed work: widths=%v scalar=%d", boundedQwenWidths(b), b.scalarCalls)
		}
		for _, ln := range lanes {
			if ln.terminal || ln.inflightSess != nil || ln.sess.SequencePosition() != 1 || len(ln.gen) != 1 {
				t.Fatal("decline lost or replayed a lane")
			}
		}
		snap := s.recorder.Snapshot(0, "")
		if snap.Decode[enginestep.PathBatched].Tokens != 7 || snap.Decode[enginestep.PathSerial].Tokens != 2 || s.sharedPanels != 3 || s.sharedMACs != 49 {
			t.Fatalf("decline receipts misstate execution: %+v", snap.Decode)
		}
	})

	for _, failure := range []string{"mutated_error", "panic", "goexit", "observer_panic", "cleanup_panic", "cleanup_goexit"} {
		t.Run("later_group_"+failure, func(t *testing.T) {
			s, b, lanes := boundedQwenFixture(t, 9)
			var err error
			s.elastic, err = newNativeElasticLedger(32, 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, ln := range lanes {
				s.elastic.laneStarted(ln, 1)
			}
			b.failAt, b.failure = 2, failure
			if failure == "cleanup_panic" || failure == "cleanup_goexit" {
				b.failure = "panic"
			}
			if failure == "observer_panic" {
				s.beforeModelExecute = func(_ nativeSchedulerEventKind, ln *schedLane) {
					if ln == lanes[7] {
						panic("cohort-observer-panic")
					}
				}
			}
			closed := make(map[*model.Session]int)
			s.closeSession = func(sess *model.Session) {
				closed[sess]++
				sess.Close()
				if failure == "cleanup_goexit" {
					runtime.Goexit()
				}
				if failure == "cleanup_panic" {
					panic("cleanup-panic-must-not-replace-forward")
				}
			}
			defer func() { s.closeSession = func(sess *model.Session) { sess.Close() } }()
			owners := []*model.Session{lanes[7].sess, lanes[8].sess}
			done := make(chan struct{})
			var raised any
			returned := false
			go func() {
				defer close(done)
				defer func() { raised = recover() }()
				boundedQwenStep(s, 0)
				returned = true
			}()
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("abnormal model exit did not unwind")
			}
			if returned || (failure != "goexit" && raised == nil) {
				t.Fatalf("abnormal exit reported success: returned=%t panic=%v", returned, raised)
			}
			if (failure == "cleanup_panic" || failure == "cleanup_goexit") && raised != "cohort-forward-panic" {
				t.Fatalf("cleanup replaced initiating panic: %v", raised)
			}
			for _, ln := range lanes[:7] {
				if ln.terminal || ln.inflightSess != nil || ln.sess.SequencePosition() != 1 || len(ln.gen) != 1 {
					t.Fatal("completed group replayed, retired or lost")
				}
			}
			for i, ln := range lanes[7:] {
				select {
				case <-ln.done:
				default:
					t.Fatal("borrowed owner terminal edge left open")
				}
				if _, open := <-ln.tokens; open {
					t.Fatal("failed lane token edge left open")
				}
				if !ln.terminal || ln.err == nil || ln.inflightSess != nil || ln.sess != nil || len(ln.gen) != 0 || closed[owners[i]] != 1 {
					t.Fatalf("failed lane leaked/published: lane=%+v closes=%d", ln, closed[owners[i]])
				}
				if got, want := ln.Reclaimed(), failure != "cleanup_panic" && failure != "cleanup_goexit"; got != want {
					t.Fatalf("cleanup completion claim=%t want=%t", got, want)
				}
			}
			wantWidths := []int{7, 2}
			if failure == "observer_panic" {
				wantWidths = []int{7}
			}
			if !reflect.DeepEqual(boundedQwenWidths(b), wantWidths) || b.scalarCalls != 0 {
				t.Fatalf("replayed after partial progress: widths=%v scalar=%d", boundedQwenWidths(b), b.scalarCalls)
			}
			if len(s.elastic.lanes) != 7 || s.elastic.snapshot().CachedReleases != 2 {
				t.Fatal("abnormal cleanup stranded an elastic reservation")
			}
		})
	}
}
