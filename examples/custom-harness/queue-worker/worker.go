package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Job is the builder-owned transport record. Its IDs remain stable across retries.
type Job struct {
	JobID   string
	RunID   string
	InputID string
	Events  []harnesskit.Envelope
}

// Delivery is a leased queue message. Attempts starts at one and increases on redelivery.
type Delivery struct {
	Receipt  string
	Job      Job
	Attempts int
}

// Queue is the minimal adapter a worker needs from a builder-selected broker.
type Queue interface {
	Receive(context.Context, int, time.Duration) ([]Delivery, error)
	Ack(context.Context, Delivery) error
	Retry(context.Context, Delivery, error) error
	DeadLetter(context.Context, Delivery, error) error
}

// Store owns the domain projection and semantic commit boundary. ApplyEvent must
// atomically persist its effect, event idempotency key, and exclusive cursor.
type Store interface {
	BeginJob(context.Context, Job) (alreadyCommitted bool, err error)
	ApplyEvent(context.Context, Job, harnesskit.Envelope) error
	CommitJob(context.Context, Job) error
}

type Worker struct {
	Queue       Queue
	Store       Store
	Capacity    int
	Lease       time.Duration
	MaxAttempts int
}

// Poll consumes at most Capacity deliveries. Broker acknowledgement happens only
// after all semantic effects and cursors for a job have committed.
func (w Worker) Poll(ctx context.Context) error {
	if w.Capacity < 1 || w.MaxAttempts < 1 {
		return errors.New("capacity and max attempts must be positive")
	}
	deliveries, err := w.Queue.Receive(ctx, w.Capacity, w.Lease)
	if err != nil {
		return err
	}
	for _, delivery := range deliveries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := w.process(ctx, delivery.Job); err != nil {
			if delivery.Attempts >= w.MaxAttempts {
				if dlqErr := w.Queue.DeadLetter(ctx, delivery, err); dlqErr != nil {
					return fmt.Errorf("dead-letter %s: %w", delivery.Job.JobID, dlqErr)
				}
			} else if retryErr := w.Queue.Retry(ctx, delivery, err); retryErr != nil {
				return fmt.Errorf("retry %s: %w", delivery.Job.JobID, retryErr)
			}
			continue
		}
		if err := w.Queue.Ack(ctx, delivery); err != nil {
			return fmt.Errorf("ack %s: %w", delivery.Job.JobID, err)
		}
	}
	return nil
}

func (w Worker) process(ctx context.Context, job Job) error {
	if job.JobID == "" || job.RunID == "" || job.InputID == "" {
		return errors.New("job_id, run_id, and input_id are required")
	}
	alreadyCommitted, err := w.Store.BeginJob(ctx, job)
	if err != nil || alreadyCommitted {
		return err
	}
	for _, event := range job.Events {
		if err := ctx.Err(); err != nil {
			return err
		}
		if event.RunID != job.RunID {
			return fmt.Errorf("event %s belongs to run %s, want %s", event.EventID, event.RunID, job.RunID)
		}
		if err := w.Store.ApplyEvent(ctx, job, event); err != nil {
			return err
		}
	}
	return w.Store.CommitJob(ctx, job)
}

// memoryQueue is a deterministic example transport, not a production broker.
type memoryQueue struct {
	mu                sync.Mutex
	ready             []Delivery
	acked             []string
	dead              []Delivery
	lastReceiveCredit int
}

func newMemoryQueue(jobs ...Job) *memoryQueue {
	q := &memoryQueue{}
	for i, job := range jobs {
		q.ready = append(q.ready, Delivery{Receipt: fmt.Sprintf("receipt-%d", i+1), Job: job, Attempts: 1})
	}
	return q
}

func (q *memoryQueue) Receive(ctx context.Context, max int, _ time.Duration) ([]Delivery, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.lastReceiveCredit = max
	if max > len(q.ready) {
		max = len(q.ready)
	}
	out := append([]Delivery(nil), q.ready[:max]...)
	q.ready = q.ready[max:]
	return out, nil
}

func (q *memoryQueue) Ack(_ context.Context, d Delivery) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.acked = append(q.acked, d.Receipt)
	return nil
}

func (q *memoryQueue) Retry(_ context.Context, d Delivery, _ error) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	d.Attempts++
	q.ready = append(q.ready, d)
	return nil
}

func (q *memoryQueue) DeadLetter(_ context.Context, d Delivery, _ error) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.dead = append(q.dead, d)
	return nil
}

// memoryStore demonstrates the builder-owned transaction boundary. A real store
// would perform each method's documented writes in database transactions.
type memoryStore struct {
	mu            sync.Mutex
	inputs        map[string]bool
	jobs          map[string]bool
	events        map[string]bool
	cursor        map[string]uint64
	inputEffects  int
	eventEffects  []string
	failRemaining map[string]int
}

func newMemoryStore() *memoryStore {
	return &memoryStore{
		inputs:        map[string]bool{},
		jobs:          map[string]bool{},
		events:        map[string]bool{},
		cursor:        map[string]uint64{},
		failRemaining: map[string]int{},
	}
}

func (s *memoryStore) BeginJob(ctx context.Context, job Job) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.jobs[job.JobID] {
		return true, nil
	}
	if !s.inputs[job.InputID] {
		s.inputs[job.InputID] = true
		s.inputEffects++
	}
	return false, nil
}

func (s *memoryStore) ApplyEvent(ctx context.Context, job Job, event harnesskit.Envelope) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.events[event.EventID] {
		return nil
	}
	if s.failRemaining[event.EventID] > 0 {
		s.failRemaining[event.EventID]--
		return fmt.Errorf("project event %s", event.EventID)
	}
	if event.Sequence <= s.cursor[job.RunID] {
		return nil
	}
	// These writes model one atomic transaction: effect + event key + cursor.
	s.eventEffects = append(s.eventEffects, event.EventID)
	s.events[event.EventID] = true
	s.cursor[job.RunID] = event.Sequence
	return nil
}

func (s *memoryStore) CommitJob(ctx context.Context, job Job) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[job.JobID] = true
	return nil
}
