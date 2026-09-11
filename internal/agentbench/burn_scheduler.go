package agentbench

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type burnSteadyConfig struct {
	Client          *http.Client
	Endpoint        string
	Model           string
	Corpus          *burnInCorpus
	Sink            *eventSink
	Concurrency     int
	QueueDepth      int
	AdmissionCutoff time.Time
	DrainDeadline   time.Time
	Now             func() time.Time
	Wait            func(context.Context, string, int, time.Duration) error
}

type burnSteadyRequestReceipt struct {
	SessionID     string      `json:"session_id"`
	Turn          int         `json:"turn"`
	RequestID     string      `json:"request_id"`
	ReleasedAt    time.Time   `json:"released_at"`
	EnqueuedAt    time.Time   `json:"enqueued_at"`
	DequeuedAt    time.Time   `json:"dequeued_at"`
	DispatchedAt  time.Time   `json:"dispatched_at"`
	FirstOutputAt time.Time   `json:"first_output_at"`
	ProgressAt    []time.Time `json:"progress_at,omitempty"`
	RejectedAt    time.Time   `json:"rejected_at,omitempty"`
	EndedAt       time.Time   `json:"ended_at"`
	UsageKnown    bool        `json:"usage_known"`
	ObservedModel string      `json:"observed_model,omitempty"`
	FinishReason  string      `json:"finish_reason,omitempty"`
	Status        string      `json:"status"`
	Error         string      `json:"error,omitempty"`
}

type burnSteadyReceipt struct {
	Accepted            int                        `json:"accepted"`
	Completed           int                        `json:"completed"`
	RejectedAfterCutoff int                        `json:"rejected_after_cutoff"`
	PeakInFlight        int                        `json:"peak_in_flight"`
	PeakQueued          int                        `json:"peak_queued"`
	PeakWaiting         int                        `json:"peak_waiting"`
	Requests            []burnSteadyRequestReceipt `json:"requests"`
	Errors              []string                   `json:"errors,omitempty"`
}

type burnSteadyJob struct {
	session  burnInSession
	turn     normalTurn
	released time.Time
	enqueued time.Time
	result   chan burnSteadyRequestReceipt
}

// runBurnSteady schedules one turn per session at a time. A session cannot
// release turn N+1 until turn N has reached a terminal state and completed its
// recorded tool wait; those waits never retain an HTTP concurrency slot.
func runBurnSteady(ctx context.Context, cfg burnSteadyConfig) burnSteadyReceipt {
	var result burnSteadyReceipt
	if cfg.Client == nil || cfg.Corpus == nil || cfg.Endpoint == "" || cfg.Model == "" || cfg.Concurrency < 1 || cfg.QueueDepth < 1 || cfg.QueueDepth > 2*cfg.Concurrency || cfg.AdmissionCutoff.IsZero() || cfg.DrainDeadline.Before(cfg.AdmissionCutoff) {
		result.Errors = append(result.Errors, "burn steady scheduler configuration incomplete")
		return result
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	wait := cfg.Wait
	if wait == nil {
		wait = func(ctx context.Context, _ string, _ int, delay time.Duration) error {
			timer := time.NewTimer(delay)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
				return nil
			}
		}
	}
	drainFor := cfg.DrainDeadline.Sub(now())
	if drainFor <= 0 {
		result.Errors = append(result.Errors, "burn steady drain deadline elapsed before scheduling")
		return result
	}
	schedulerCtx, cancelScheduler := context.WithTimeout(ctx, drainFor)
	defer cancelScheduler()

	jobs := make(chan burnSteadyJob)
	queueSlots := make(chan struct{}, cfg.QueueDepth)
	var stateMu sync.Mutex
	sequence, waiting, queued, httpInFlight := 0, 0, 0, 0
	appendReceipt := func(request burnSteadyRequestReceipt) {
		stateMu.Lock()
		defer stateMu.Unlock()
		result.Requests = append(result.Requests, request)
		switch request.Status {
		case "completed":
			result.Completed++
		case "rejected":
			result.RejectedAfterCutoff++
		}
		if request.Error != "" && request.Status != "rejected" {
			result.Errors = append(result.Errors, request.Error)
		}
	}
	httpDelta := func(delta int) {
		stateMu.Lock()
		defer stateMu.Unlock()
		httpInFlight += delta
		result.PeakInFlight = max(result.PeakInFlight, httpInFlight)
	}
	var workers sync.WaitGroup
	for worker := 0; worker < cfg.Concurrency; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for job := range jobs {
				dequeued := now()
				stateMu.Lock()
				queued--
				sequence++
				seq := sequence
				stateMu.Unlock()
				<-queueSlots
				planned := plannedRequest{sequence: seq, rung: job.session.Area, session: job.session.ID, turn: job.turn.Ordinal}
				if !dequeued.Before(cfg.AdmissionCutoff) || schedulerCtx.Err() != nil {
					rejected := burnSteadyRequestReceipt{SessionID: job.session.ID, Turn: job.turn.Ordinal, RequestID: requestID(planned), ReleasedAt: job.released, EnqueuedAt: job.enqueued, DequeuedAt: dequeued, RejectedAt: dequeued, EndedAt: dequeued, Status: "rejected"}
					appendReceipt(rejected)
					job.result <- rejected
					continue
				}
				stateMu.Lock()
				result.Accepted++
				stateMu.Unlock()
				request := runBurnSteadyRequest(schedulerCtx, cfg, job.session, job.turn, seq, job.released, job.enqueued, dequeued, now, httpDelta)
				appendReceipt(request)
				job.result <- request
			}
		}()
	}
	var sessions sync.WaitGroup
	for _, session := range cfg.Corpus.Sessions {
		session := session
		sessions.Add(1)
		go func() {
			defer sessions.Done()
			for turnIndex, turn := range session.Turns {
				released := now()
				stateMu.Lock()
				waiting++
				result.PeakWaiting = max(result.PeakWaiting, waiting)
				stateMu.Unlock()
				select {
				case queueSlots <- struct{}{}:
				case <-schedulerCtx.Done():
					stateMu.Lock()
					waiting--
					stateMu.Unlock()
					return
				}
				enqueued := now()
				stateMu.Lock()
				waiting--
				queued++
				result.PeakQueued = max(result.PeakQueued, queued)
				stateMu.Unlock()
				answer := make(chan burnSteadyRequestReceipt, 1)
				job := burnSteadyJob{session: session, turn: turn, released: released, enqueued: enqueued, result: answer}
				select {
				case jobs <- job:
				case <-schedulerCtx.Done():
					stateMu.Lock()
					queued--
					stateMu.Unlock()
					<-queueSlots
					return
				}
				var request burnSteadyRequestReceipt
				select {
				case request = <-answer:
				case <-schedulerCtx.Done():
					return
				}
				if request.Status == "rejected" {
					continue
				}
				delay := normalToolWait((turnIndex % 12) + 1)
				if delay > 0 {
					if err := wait(schedulerCtx, session.ID, turnIndex+1, delay); err != nil {
						stateMu.Lock()
						result.Errors = append(result.Errors, fmt.Sprintf("session %s turn %d tool wait: %v", session.ID, turnIndex+1, err))
						stateMu.Unlock()
						return
					}
				}
			}
		}()
	}
	sessions.Wait()
	close(jobs)
	workers.Wait()
	return result
}

func runBurnSteadyRequest(ctx context.Context, cfg burnSteadyConfig, session burnInSession, turn normalTurn, sequence int, releasedAt, enqueuedAt, dequeuedAt time.Time, now func() time.Time, httpDelta func(int)) burnSteadyRequestReceipt {
	planned := plannedRequest{sequence: sequence, rung: session.Area, session: session.ID, turn: turn.Ordinal}
	receipt := burnSteadyRequestReceipt{SessionID: session.ID, Turn: turn.Ordinal, RequestID: requestID(planned), ReleasedAt: releasedAt, EnqueuedAt: enqueuedAt, DequeuedAt: dequeuedAt, Status: "failed"}
	if cfg.Sink != nil {
		started := receipt.ReleasedAt
		if err := cfg.Sink.start(planned, cfg.Model, started); err != nil {
			receipt.Error, receipt.EndedAt = err.Error(), now()
			return receipt
		}
		if err := cfg.Sink.phase(planned, cfg.Model, "release", receipt.ReleasedAt); err != nil {
			receipt.Error, receipt.EndedAt = err.Error(), now()
			return receipt
		}
		if err := cfg.Sink.phase(planned, cfg.Model, "enqueue", now()); err != nil {
			receipt.Error, receipt.EndedAt = err.Error(), now()
			return receipt
		}
	}
	requestCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	receipt.DispatchedAt = now()
	if cfg.Sink != nil {
		if err := cfg.Sink.phase(planned, cfg.Model, "dispatch", receipt.DispatchedAt); err != nil {
			receipt.Error, receipt.EndedAt = err.Error(), now()
			return receipt
		}
	}
	envelope := requestEnvelope{Model: cfg.Model, Stream: true, StreamOptions: streamOptions{IncludeUsage: true}, Messages: turn.Request.Messages, Tools: turn.Request.Tools, MaxTokens: turn.Request.OutputTokens, Metadata: requestMetadata{Benchmark: "agentbench", Profile: "burn-in", Rung: session.Area, Session: session.ID, Turn: turn.Ordinal, Sequence: sequence, ScoredRequest: true}}
	httpDelta(1)
	defer httpDelta(-1)
	observation, err := dispatchWithProgress(requestCtx, cfg.Client, cfg.Endpoint, envelope, cfg.Model,
		func(at time.Time) error {
			receipt.FirstOutputAt = at
			if cfg.Sink != nil {
				return cfg.Sink.phase(planned, cfg.Model, "first_output", at)
			}
			return nil
		}, nil, func(at time.Time) error {
			receipt.ProgressAt = append(receipt.ProgressAt, at)
			if cfg.Sink != nil {
				return cfg.Sink.phase(planned, cfg.Model, "stream_progress", at)
			}
			return nil
		})
	receipt.EndedAt = now()
	receipt.UsageKnown = observation.usage != nil
	receipt.ObservedModel = observation.model
	receipt.FinishReason = observation.finishReason
	if err != nil {
		receipt.Error = err.Error()
	} else {
		receipt.Status = "completed"
	}
	if cfg.Sink != nil {
		terminal := terminalEvent(planned, cfg.Model, receipt.ReleasedAt, receipt.EndedAt, observation, err, requestCtx.Err(), ctx.Err())
		terminal.Event, terminal.EventType, terminal.Phase = "end", "terminal", "steady"
		if sinkErr := cfg.Sink.finish(terminal); sinkErr != nil && receipt.Error == "" {
			receipt.Error = sinkErr.Error()
		}
	}
	if errors.Is(requestCtx.Err(), context.DeadlineExceeded) && receipt.Error == "" {
		receipt.Error = context.DeadlineExceeded.Error()
	}
	return receipt
}
