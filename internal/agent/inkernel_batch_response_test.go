package agent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelCoalescedLeaderReturnsBeforeLaterCohort proves that electing a
// caller to drain the planner queue does not make that caller wait for work in
// a later cohort after its own result and receipt are complete.
func TestInKernelCoalescedLeaderReturnsBeforeLaterCohort(t *testing.T) {
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		out := make([][]float32, len(ids))
		lanes := 0
		for i, on := range active {
			if !on {
				continue
			}
			out[i] = bs.Seqs[i].Step(ids[i])
			lanes++
		}
		return out, 1, int64(lanes), true
	})
	t.Cleanup(restoreProbe)

	newPlanner := func(enabled bool) *InKernelPlanner {
		cfg := tinyConcurrencyConfig()
		cfg.EOSTokenID = -1
		cfg.LayerTypes = []string{"linear_attention"}
		cfg.LinearConvKernelDim = 3
		cfg.LinearKeyHeadDim = 8
		cfg.LinearNumKeyHeads = 2
		cfg.LinearValueHeadDim = 8
		cfg.LinearNumValueHeads = 4
		m := model.NewSynthetic(cfg)
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-qwen-cohort-response", true, nil, false)
		p.maxNew = 3
		p.batchDecode = enabled
		p.metal = enabled // synthetic witness for the production Metal gate
		return p
	}

	messages := make([][]Message, inKernelDecodeCohortMax+1)
	for i := range messages {
		messages[i] = []Message{{Role: RoleUser, Content: "cohort response"}}
	}
	serial, err := newPlanner(false).Complete(context.Background(), messages[0], nil)
	if err != nil {
		t.Fatalf("serial reference: %v", err)
	}

	p := newPlanner(true)
	leaderAtGate := make(chan struct{})
	releaseDrain := make(chan struct{})
	var gateOnce, releaseDrainOnce sync.Once
	p.coalesceReadyHook = func() {
		gateOnce.Do(func() { close(leaderAtGate) })
		<-releaseDrain
	}
	secondStarted := make(chan struct{})
	releaseSecond := make(chan struct{})
	var secondOnce, releaseSecondOnce sync.Once
	p.coalesceBatchHook = func(n int) {
		if n == 1 {
			secondOnce.Do(func() { close(secondStarted) })
			<-releaseSecond
		}
	}

	type answer struct {
		completion *Completion
		err        error
	}
	answers := make([]answer, len(messages))
	returned := make([]chan struct{}, len(messages))
	for i := range returned {
		returned[i] = make(chan struct{})
	}
	t.Cleanup(func() {
		releaseDrainOnce.Do(func() { close(releaseDrain) })
		releaseSecondOnce.Do(func() { close(releaseSecond) })
		deadline := time.NewTimer(2 * time.Second)
		defer deadline.Stop()
		for _, done := range returned {
			select {
			case <-done:
			case <-deadline.C:
				return
			}
		}
	})
	go func() {
		answers[0].completion, answers[0].err = p.Complete(context.Background(), messages[0], nil)
		close(returned[0])
	}()

	select {
	case <-leaderAtGate:
	case <-time.After(2 * time.Second):
		t.Fatal("elected caller did not reach the deterministic ready gate")
	}
	for i := 1; i < len(messages); i++ {
		go func(i int) {
			answers[i].completion, answers[i].err = p.Complete(context.Background(), messages[i], nil)
			close(returned[i])
		}(i)
	}
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for {
		p.coalesceMu.Lock()
		ready := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if ready == len(messages) {
			break
		}
		select {
		case <-deadline.C:
			t.Fatalf("ready queue length never reached %d; got %d", len(messages), ready)
		default:
			time.Sleep(time.Millisecond)
		}
	}
	releaseDrainOnce.Do(func() { close(releaseDrain) })

	select {
	case <-secondStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("later singleton cohort did not start")
	}

	leaderReturned := false
	select {
	case <-returned[0]:
		leaderReturned = true
	case <-time.After(2 * time.Second):
	}
	releaseSecondOnce.Do(func() { close(releaseSecond) })
	for i := range returned {
		select {
		case <-returned[i]:
		case <-time.After(2 * time.Second):
			t.Fatalf("request %d did not finish after releasing the later cohort", i)
		}
	}
	if !leaderReturned {
		t.Fatal("elected caller waited for a later cohort after its own cohort completed")
	}
	leader := answers[0]
	if leader.err != nil || leader.completion == nil || leader.completion.InKernelBatch == nil {
		t.Fatalf("leader completion=%#v err=%v", leader.completion, leader.err)
	}
	if got := leader.completion.InKernelBatch.CohortSize; got != inKernelDecodeCohortMax {
		t.Fatalf("leader cohort size=%d, want %d", got, inKernelDecodeCohortMax)
	}
	if leader.completion.Message.Content != serial.Message.Content || leader.completion.FinishReason != serial.FinishReason {
		t.Fatalf("leader=%q/%s, serial=%q/%s", leader.completion.Message.Content, leader.completion.FinishReason, serial.Message.Content, serial.FinishReason)
	}
	singletons := 0
	leaderCohort := 0
	for i, got := range answers {
		if got.err != nil || got.completion == nil || got.completion.InKernelBatch == nil {
			t.Fatalf("completion[%d]=%#v err=%v", i, got.completion, got.err)
		}
		if got.completion.Message.Content != serial.Message.Content || got.completion.FinishReason != serial.FinishReason {
			t.Fatalf("completion[%d]=%q/%s, serial=%q/%s", i, got.completion.Message.Content, got.completion.FinishReason, serial.Message.Content, serial.FinishReason)
		}
		switch got.completion.InKernelBatch.CohortSize {
		case 1:
			singletons++
		case inKernelDecodeCohortMax:
			leaderCohort++
		default:
			t.Fatalf("completion[%d] cohort size=%d, want 1 or %d", i, got.completion.InKernelBatch.CohortSize, inKernelDecodeCohortMax)
		}
	}
	if singletons != 1 || leaderCohort != inKernelDecodeCohortMax {
		t.Fatalf("cohort membership: singleton=%d leader-cohort=%d, want 1/%d", singletons, leaderCohort, inKernelDecodeCohortMax)
	}
}
