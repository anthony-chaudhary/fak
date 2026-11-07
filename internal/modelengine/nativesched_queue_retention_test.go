package modelengine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeSchedulerQueueReleasesDiscardedPointers(t *testing.T) {
	newCanceled := func() *schedLane {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		return &schedLane{ctx: ctx, cancel: cancel,
			prompt: make([]int, 32), logits: make([]float32, 32),
			tokens: make(chan abi.EngineToken, 1), done: make(chan struct{})}
	}
	for _, queue := range []string{"waiting", "preempted", "running"} {
		t.Run(queue, func(t *testing.T) {
			retired := newCanceled()
			backing := []*schedLane{retired}
			s := &NativeScheduler{}
			switch queue {
			case "waiting":
				s.waiting = backing
				s.promoteWaitingLocked(0)
				if len(s.waiting) != 0 {
					t.Fatal("cancelled lane was not removed")
				}
			case "preempted":
				s.preempted = backing
				s.dropCanceledPreemptedLocked()
				if len(s.preempted) != 0 {
					t.Fatal("cancelled lane was not removed")
				}
			case "running":
				retired.finish(nil, retired.ctx.Err())
				s.lanes = backing
				s.runIteration(false)
				if len(s.lanes) != 0 {
					t.Fatal("terminal lane was not removed")
				}
			}
			if !retired.terminal || !retired.Reclaimed() {
				t.Fatal("cancelled lane was not reclaimed")
			}
			if backing[0] != nil {
				t.Fatal("queue backing array retains retired lane and its request buffers")
			}
			if len(retired.prompt) != 32 || len(retired.logits) != 32 {
				t.Fatal("caller-owned lane buffers were mutated instead of releasing container references")
			}
		})
	}
	for _, queue := range []string{"waiting", "preempted"} {
		t.Run(queue+"/retained_order", func(t *testing.T) {
			first := &schedLane{ctx: context.Background(), seqNo: 1}
			second := &schedLane{ctx: context.Background(), seqNo: 2}
			backing := []*schedLane{newCanceled(), first, newCanceled(), second, newCanceled()}
			s := &NativeScheduler{}
			var kept []*schedLane
			if queue == "waiting" {
				s.lanes = []*schedLane{{ctx: context.Background()}}
				s.waiting = backing
				s.promoteWaitingLocked(1)
				kept = s.waiting
			} else {
				s.preempted = backing
				s.dropCanceledPreemptedLocked()
				kept = s.preempted
			}
			if len(kept) != 2 || kept[0] != first || kept[1] != second {
				t.Fatal("retained lane identity or order changed")
			}
			for _, ptr := range backing[len(kept):] {
				if ptr != nil {
					t.Fatal("discarded range retains lane reference")
				}
			}
		})
	}
}
