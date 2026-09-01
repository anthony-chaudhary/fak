package main

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func envelope(runID, eventID string, sequence uint64) harnesskit.Envelope {
	return harnesskit.Envelope{Version: "1.0", RunID: runID, Sequence: sequence, EventID: eventID, Type: harnesskit.EventRunCompleted}
}

func workerFor(q Queue, s Store) Worker {
	return Worker{Queue: q, Store: s, Capacity: 2, Lease: time.Minute, MaxAttempts: 2}
}

func TestRedeliveryDoesNotDuplicateInputJobOrEventEffects(t *testing.T) {
	job := Job{JobID: "job-1", RunID: "run-1", InputID: "input-1", Events: []harnesskit.Envelope{envelope("run-1", "event-1", 1)}}
	q := newMemoryQueue(job)
	s := newMemoryStore()
	w := workerFor(q, s)

	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	// A broker may redeliver after the semantic commit when its Ack was lost.
	q.ready = append(q.ready, Delivery{Receipt: "receipt-redelivery", Job: job, Attempts: 2})
	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}

	if s.inputEffects != 1 || !reflect.DeepEqual(s.eventEffects, []string{"event-1"}) {
		t.Fatalf("duplicate effects: inputs=%d events=%v", s.inputEffects, s.eventEffects)
	}
	if got := s.cursor[job.RunID]; got != 1 {
		t.Fatalf("cursor=%d, want 1", got)
	}
	if len(q.acked) != 2 {
		t.Fatalf("acks=%d, want original and redelivery", len(q.acked))
	}
}

func TestDuplicateInputAcrossJobsAndDuplicateEventAreIdempotent(t *testing.T) {
	first := Job{JobID: "job-1", RunID: "run-1", InputID: "input-shared", Events: []harnesskit.Envelope{envelope("run-1", "event-shared", 1)}}
	second := Job{JobID: "job-2", RunID: "run-1", InputID: "input-shared", Events: []harnesskit.Envelope{envelope("run-1", "event-shared", 1), envelope("run-1", "event-2", 2)}}
	q := newMemoryQueue(first, second)
	s := newMemoryStore()
	if err := workerFor(q, s).Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.inputEffects != 1 {
		t.Fatalf("input effects=%d, want 1", s.inputEffects)
	}
	if !reflect.DeepEqual(s.eventEffects, []string{"event-shared", "event-2"}) {
		t.Fatalf("event effects=%v", s.eventEffects)
	}
}

func TestRetryThenSuccess(t *testing.T) {
	job := Job{JobID: "job-retry", RunID: "run-retry", InputID: "input-retry", Events: []harnesskit.Envelope{envelope("run-retry", "event-retry", 1)}}
	q := newMemoryQueue(job)
	s := newMemoryStore()
	s.failRemaining["event-retry"] = 1
	w := workerFor(q, s)

	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(q.acked) != 0 || len(q.ready) != 1 || q.ready[0].Attempts != 2 {
		t.Fatalf("first delivery: acked=%v ready=%v", q.acked, q.ready)
	}
	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(q.acked) != 1 || len(s.eventEffects) != 1 {
		t.Fatalf("after retry: acked=%v effects=%v", q.acked, s.eventEffects)
	}
}

func TestPoisonDeliveryDeadLetters(t *testing.T) {
	job := Job{JobID: "job-poison", RunID: "run-poison", InputID: "input-poison", Events: []harnesskit.Envelope{envelope("run-poison", "event-poison", 1)}}
	q := newMemoryQueue(job)
	s := newMemoryStore()
	s.failRemaining["event-poison"] = 99
	w := workerFor(q, s)

	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := w.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(q.dead) != 1 || len(q.acked) != 0 || len(q.ready) != 0 {
		t.Fatalf("dead=%v acked=%v ready=%v", q.dead, q.acked, q.ready)
	}
	if len(s.eventEffects) != 0 {
		t.Fatalf("poison event produced effects: %v", s.eventEffects)
	}
}

func TestBoundedCapacity(t *testing.T) {
	q := newMemoryQueue(
		Job{JobID: "j1", RunID: "r1", InputID: "i1"},
		Job{JobID: "j2", RunID: "r2", InputID: "i2"},
		Job{JobID: "j3", RunID: "r3", InputID: "i3"},
	)
	s := newMemoryStore()
	if err := workerFor(q, s).Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if q.lastReceiveCredit != 2 || len(q.ready) != 1 || len(q.acked) != 2 {
		t.Fatalf("credit=%d ready=%d acked=%d", q.lastReceiveCredit, len(q.ready), len(q.acked))
	}
}

func TestCancellationDoesNotAck(t *testing.T) {
	q := newMemoryQueue(Job{JobID: "job-cancel", RunID: "run-cancel", InputID: "input-cancel"})
	s := newMemoryStore()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := workerFor(q, s).Poll(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v, want canceled", err)
	}
	if len(q.acked) != 0 {
		t.Fatalf("canceled delivery acknowledged: %v", q.acked)
	}
}
