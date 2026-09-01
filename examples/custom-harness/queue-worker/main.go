package main

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func main() {
	job := Job{
		JobID: "job-demo-001", RunID: "run-demo-001", InputID: "input-demo-001",
		Events: []harnesskit.Envelope{
			{Version: "1.0", RunID: "run-demo-001", Sequence: 1, EventID: "event-demo-001", Type: harnesskit.EventRunStarted},
			{Version: "1.0", RunID: "run-demo-001", Sequence: 2, EventID: "event-demo-002", Type: harnesskit.EventRunCompleted},
		},
	}
	queue := newMemoryQueue(job)
	store := newMemoryStore()
	worker := Worker{Queue: queue, Store: store, Capacity: 2, Lease: 30 * time.Second, MaxAttempts: 3}
	if err := worker.Poll(context.Background()); err != nil {
		panic(err)
	}
	fmt.Printf("acked=%d input_effects=%d event_effects=%d cursor=%d\n", len(queue.acked), store.inputEffects, len(store.eventEffects), store.cursor[job.RunID])
}
