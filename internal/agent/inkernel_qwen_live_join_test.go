package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The coordinator fixture executes native dense host arithmetic, not Qwen GPU
// qualification. The separate HTTP witness covers scoped hybrid prefix restore.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelQwenLiveJoin(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m, coalesceReadyHook: func() {}}
	prompts := [][]int{{1, 2, 3}, {1, 2, 4, 5}}
	budgets := []int{4, 7}
	want := make([][]int, len(prompts))
	for index := range prompts {
		lane, sink := newDecodeLaneOver(m, prompts[index], budgets[index], 1)
		inKernelDecodeSerial(context.Background(), lane)
		want[index] = append([]int(nil), sink.toks...)
		lane.s.Close()
	}

	firstForward, releaseArrival := make(chan struct{}), make(chan struct{})
	peerStillDecoding, releasePeer := make(chan struct{}), make(chan struct{})
	var arrivalOnce, peerOnce sync.Once
	release := func() {
		arrivalOnce.Do(func() { close(releaseArrival) })
		peerOnce.Do(func() { close(releasePeer) })
	}
	type counters struct {
		panels int
		macs   int64
	}
	actual := make(map[*model.Session]counters)
	calls, sharedForwards := 0, 0
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		calls++
		if calls == 2 {
			// One native forward has completed; B cannot be part of the
			// initial cohort and its prefill has not begun yet.
			close(firstForward)
			<-releaseArrival
		}
		if calls == 5 {
			close(peerStillDecoding)
			<-releasePeer
		}
		out := bs.StepBatchActive(ids, active)
		panels, macs := bs.LastStepSharedPanels(), bs.LastStepMACs()
		count := 0
		for index, on := range active {
			if on {
				count++
				value := actual[bs.Seqs[index]]
				value.panels += panels
				value.macs += macs
				actual[bs.Seqs[index]] = value
			}
		}
		if count > 1 {
			sharedForwards++
		}
		// These are the actual native BatchSession counters, not fabricated
		// shared-panel credit. The enclosing iterator must not step twice.
		return out, panels, macs, true
	})
	type answer struct {
		result  inKernelGenerateResult
		tokens  []int
		session *model.Session
		err     error
	}
	done := []chan answer{make(chan answer, 1), make(chan answer, 1)}
	closed := make([]atomic.Int32, len(done))
	var workers sync.WaitGroup
	t.Cleanup(func() {
		release()
		workers.Wait()
		restoreProbe()
	})
	run := func(index int) {
		workers.Add(1)
		go func() {
			defer workers.Done()
			var lane *decodeLane
			var sink *batchLaneSink
			result, err := p.runCoalescedGenerate(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
				lane, sink = newDecodeLaneOver(m, prompts[index], budgets[index], 1)
				defer func() { lane.s.Close(); closed[index].Add(1) }()
				_, err := coalescedDecode(ctx, lane)
				return inKernelGenerateResult{gen: lane.gen}, err
			})
			done[index] <- answer{result: result, tokens: sink.toks, session: lane.s, err: err}
		}()
	}
	wait := func(signal <-chan struct{}, label string) {
		t.Helper()
		select {
		case <-signal:
		case <-time.After(2 * time.Second):
			t.Fatalf("did not reach %s", label)
		}
	}
	run(0)
	wait(firstForward, "completed first forward")
	run(1)
	awaitCoalescedQueue(t, p, 1)
	arrivalOnce.Do(func() { close(releaseArrival) })
	wait(peerStillDecoding, "late request still decoding")
	answers := make([]answer, len(done))
	select {
	case answers[0] = <-done[0]:
	case <-time.After(time.Second):
		t.Error("completed leader remained blocked behind the live late request")
	}
	if sharedForwards == 0 {
		t.Error("late request never joined a forward while the leader was active")
	}
	if closed[0].Load() != 1 || closed[1].Load() != 0 {
		t.Errorf("early retirement closes=%d/%d, want 1/0", closed[0].Load(), closed[1].Load())
	}
	peerOnce.Do(func() { close(releasePeer) })
	for index := range done {
		if answers[index].session != nil {
			continue
		}
		select {
		case answers[index] = <-done[index]:
		case <-time.After(2 * time.Second):
			t.Fatalf("request %d did not complete", index)
		}
	}
	for index, answer := range answers {
		if answer.err != nil || !reflect.DeepEqual(answer.tokens, want[index]) || closed[index].Load() != 1 {
			t.Errorf("request %d tokens=%v want=%v error=%v closes=%d", index, answer.tokens, want[index], answer.err, closed[index].Load())
		}
		receipt := answer.result.batchReceipt
		if receipt.CohortID == 0 || receipt.CohortID != answers[0].result.batchReceipt.CohortID || receipt.CohortSize != 2 {
			t.Errorf("request %d receipt=%+v, want same live cohort with peak size 2", index, receipt)
		}
		if expected := actual[answer.session]; receipt.SharedPanels != expected.panels || receipt.SharedMACs != expected.macs {
			t.Errorf("request %d participation=%+v, actual forwards=%+v", index, receipt, expected)
		}
		if receipt.SessionCloses != uint32(index+1) {
			t.Errorf("request %d closes receipt=%d, want %d", index, receipt.SessionCloses, index+1)
		}
	}
	p.coalesceMu.Lock()
	pending, running := len(p.coalesceReady), p.coalesceRunning
	p.coalesceMu.Unlock()
	if pending != 0 || running {
		t.Errorf("completed coordinator pending=%d running=%t", pending, running)
	}
}

// Unknown panics must reach the originating caller, not escape the asynchronous
// drainer or replay a forward whose sessions have already advanced.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelQwenLiveJoinPanicClosesBeforeHandoff(t *testing.T) {
	for _, phase := range []string{"shared_forward", "preparation"} {
		t.Run(phase, func(t *testing.T) {
			m := model.NewSynthetic(tinyCfg())
			ready, failReady := make(chan struct{}), make(chan struct{})
			releaseFailure := make(chan struct{})
			var readyOnce, failureOnce sync.Once
			release := func() {
				readyOnce.Do(func() { close(ready) })
				failureOnce.Do(func() { close(releaseFailure) })
			}
			p := &InKernelPlanner{m: m, coalesceReadyHook: func() { <-ready }}
			primary := errors.New("unknown Qwen coordinator panic")
			forwarded := make(map[*model.Session]int)
			calls := 0
			restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
				calls++
				for index, on := range active {
					if on {
						forwarded[bs.Seqs[index]]++
					}
				}
				out := bs.StepBatchActive(ids, active)
				if phase == "shared_forward" && calls == 1 {
					close(failReady)
					<-releaseFailure
					panic(primary) // Both active sessions have already advanced.
				}
				return out, bs.LastStepSharedPanels(), bs.LastStepMACs(), true
			})
			type answer struct {
				result       inKernelGenerateResult
				err          error
				panic        any
				closedAtExit int32
			}
			done := []chan answer{make(chan answer, 1), make(chan answer, 1), make(chan answer, 1)}
			closed := make([]atomic.Int32, len(done))
			requests := make([]*inKernelCoalesceRequest, len(done))
			sessions := make([]*model.Session, len(done))
			var workers sync.WaitGroup
			t.Cleanup(func() {
				release()
				workers.Wait()
				restoreProbe()
			})
			run := func(index int) {
				workers.Add(1)
				go func() {
					var out answer
					defer func() {
						out.panic = recover()
						out.closedAtExit = closed[0].Load() + closed[1].Load()
						done[index] <- out
						workers.Done()
					}()
					out.result, out.err = p.runCoalescedGenerate(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
						requests[index] = ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest)
						budget := 4
						if index == 2 {
							budget = 2
						}
						lane, _ := newDecodeLaneOver(m, []int{1, 2, 3 + index}, budget, 1)
						sessions[index] = lane.s
						defer func() { lane.s.Close(); closed[index].Add(1) }()
						if phase == "preparation" && index == 1 {
							close(failReady)
							<-releaseFailure
							panic(primary)
						}
						_, err := coalescedDecode(ctx, lane)
						return inKernelGenerateResult{gen: lane.gen}, err
					})
				}()
			}
			run(0)
			awaitCoalescedQueue(t, p, 1)
			run(1)
			awaitCoalescedQueue(t, p, 2)
			readyOnce.Do(func() { close(ready) })
			select {
			case <-failReady:
			case <-time.After(2 * time.Second):
				t.Fatal("selected cohort did not reach injected failure")
			}
			run(2)
			awaitCoalescedQueue(t, p, 1)
			failureOnce.Do(func() { close(releaseFailure) })
			answers := make([]answer, len(done))
			for index := range done {
				select {
				case answers[index] = <-done[index]:
				case <-time.After(2 * time.Second):
					t.Fatalf("request %d stranded by cohort panic", index)
				}
			}
			cause := 0
			if phase == "preparation" {
				cause = 1
			}
			for index := 0; index < 2; index++ {
				if index == cause {
					if answers[index].panic != primary || answers[index].closedAtExit != 2 {
						t.Errorf("cause panic=%v closes-at-relay=%d, want exact sentinel after both closes", answers[index].panic, answers[index].closedAtExit)
					}
				} else if answers[index].panic != nil || answers[index].err == nil ||
					answers[index].err.Error() != "agent: Qwen cohort aborted; no request replay permitted" {
					t.Errorf("peer panic=%v error=%v, want terminal abort", answers[index].panic, answers[index].err)
				}
				wantForwards := 0
				if phase == "shared_forward" {
					wantForwards = 1
				}
				if got := forwarded[sessions[index]]; got != wantForwards {
					t.Errorf("request %d forwards=%d, want %d without replay", index, got, wantForwards)
				}
			}
			if got := answers[2]; got.panic != nil || got.err != nil || got.result.gen != 2 || got.result.batchReceipt.CohortID == requests[0].receipt.CohortID {
				t.Errorf("queued successor did not survive handoff: %+v", got)
			}
			for index, req := range requests {
				if closed[index].Load() != 1 {
					t.Errorf("request %d closed %d times", index, closed[index].Load())
				}
				select {
				case <-req.receiptReady:
				default:
					t.Errorf("request %d lacks cleanup receipt", index)
				}
			}
			p.coalesceMu.Lock()
			pending, running := len(p.coalesceReady), p.coalesceRunning
			p.coalesceMu.Unlock()
			if pending != 0 || running {
				t.Errorf("panic handoff retained queue=%d running=%t", pending, running)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := p.devMu.LockContext(ctx); err != nil {
				t.Errorf("panic handoff retained device gate: %v", err)
			} else {
				p.devMu.Unlock()
			}
		})
	}
}
