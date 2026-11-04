package gateway

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// fak-test:runtime fast est=100ms lane=default
func TestAdmissionRequiresFullFleetReservation(t *testing.T) {
	for _, tc := range []struct {
		name string
		pool int
	}{
		{"offer", 100},
		{"acquire", 100},
		{"acquire-empty", 60},
		{"strict-queue", 100},
		{"weighted-queue", 100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const tokens = 60
			queued := tc.name == "strict-queue" || tc.name == "weighted-queue"
			pool := session.NewPool(tc.pool)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			checked := make(chan struct{}, 2)
			release := make(chan struct{})
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			controllers := make([]*AdmissionController, 2)
			ready := make([]chan struct{}, 2)
			var admittedBefore int64
			for i := range controllers {
				c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 100, MaxWaiting: 1})
				c.SetFleet(pool)
				controllers[i] = c
				if tc.name == "weighted-queue" {
					c.SetOrder(session.WeightedFair)
				}
				if queued {
					if c.Offer(SeqRequest{TraceID: "holder"}) != VerdictAdmitted || c.Offer(SeqRequest{TraceID: "target", Tokens: tokens}) != VerdictQueued || !c.Complete("holder") {
						t.Fatal("could not establish an unfunded queued request")
					}
					ready[i] = make(chan struct{})
					c.waiting[0].ready = ready[i]
				}
				admittedBefore += c.Stats().Admitted
				checks, pauseAt := 0, 2 // empty-set check, then the live fast-path check
				if queued {
					pauseAt = 1 // Schedule performs only the live check
				}
				// Both controllers must observe the same sufficient pool balance before
				// either draws. This existing budget seam only orders the race; the real
				// Pool.Draw and all admission transitions remain under test.
				c.budgets = append(c.budgets, func(batchBudgetSnapshot, SeqRequest) batchBudgetCheck {
					checks++
					if checks == pauseAt {
						checked <- struct{}{}
						select {
						case <-release:
						case <-ctx.Done():
						}
					}
					return batchBudgetCheck{status: batchBudgetAvailable, budget: "reservation_race_fixture"}
				})
			}
			type result struct {
				admitted bool
				lease    *AdmissionLease
				err      error
			}
			results := make(chan result, 2)
			for _, c := range controllers {
				go func() {
					if queued {
						results <- result{admitted: len(c.Schedule()) == 1}
					} else if tc.name == "offer" {
						results <- result{admitted: c.Offer(SeqRequest{TraceID: "target", Tokens: tokens}) == VerdictAdmitted}
					} else {
						lease, err := c.Acquire(ctx, SeqRequest{TraceID: "target", Tokens: tokens})
						results <- result{admitted: lease != nil, lease: lease, err: err}
					}
				}()
			}
			for range controllers {
				select {
				case <-checked:
				case <-ctx.Done():
					t.Fatal("both controllers did not reach the checked-before-draw boundary")
				}
			}
			releaseOnce.Do(func() { close(release) })
			grants := 0
			for range controllers {
				select {
				case got := <-results:
					if got.lease != nil {
						defer got.lease.Release()
					}
					if got.admitted {
						grants++
					} else if !queued && tc.name != "offer" {
						var refusal *AdmissionError
						status, _, _, ok := admissionErrorStatus(got.err)
						if !errors.As(got.err, &refusal) || refusal.Verdict != VerdictShed || refusal.Budget != "fleet_tokens" || !ok || status != http.StatusTooManyRequests {
							t.Fatalf("unfunded Acquire error = %v, want fleet_tokens shed mapped to429", got.err)
						}
					}
				case <-ctx.Done():
					t.Fatal("reservation race did not complete")
				}
			}
			var running, waiting, used, queuedTokens int
			var admitted, shed int64
			for i, c := range controllers {
				st := c.Stats()
				running += st.Running
				waiting += st.Waiting
				used += st.TokensInUse
				queuedTokens += st.QueuedTokens
				admitted += st.Admitted
				shed += st.Shed
				if queued {
					woken := false
					select {
					case <-ready[i]:
						woken = true
					default:
					}
					if woken != (st.Running == 1) {
						t.Fatalf("queued ready=%v while running=%d", woken, st.Running)
					}
				}
			}
			if grants != 1 || running != 1 || used != tokens || admitted-admittedBefore != 1 || pool.Remaining() != tc.pool-tokens {
				t.Fatalf("grants=%d running=%d tokens=%d admitted=%d remaining=%d; only one60-token request may reserve the shared pool", grants, running, used, admitted-admittedBefore, pool.Remaining())
			}
			if queued && (waiting != 1 || queuedTokens != tokens || shed != 0) || !queued && (waiting != 0 || queuedTokens != 0 || shed != 1) {
				t.Fatalf("waiting=%d queued tokens=%d shed=%d after reservation race", waiting, queuedTokens, shed)
			}
		})
	}
}
