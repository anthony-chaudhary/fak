package gateway

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"
)

// fak-test:runtime fast est=2ms lane=default
func TestBeginServedAdmissionOrdersClientDeadlines(t *testing.T) {
	type request struct {
		id       string
		budgetMS string
		priority int
	}
	for _, tc := range []struct {
		name     string
		requests []request
		want     []string
	}{
		{
			name: "equal_priority",
			requests: []request{
				{"none-first", "", 3},
				{"late", "1800000", 3},
				{"none-second", "", 3},
				{"early", "600000", 3},
			},
			want: []string{"early", "late", "none-first", "none-second"},
		},
		{
			name: "higher_priority",
			requests: []request{
				{"urgent-low", "600000", 4},
				{"high", "", 1},
			},
			want: []string{"high", "urgent-low"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 8, AgingRounds: 1_000_000})
			s := &Server{}
			s.SetAdmissionController(ctl)
			running, err := s.beginServedAdmission(context.Background(), servedSessionTurn{traceID: "running"}, nil, nil, 1)
			if err != nil || running == nil {
				t.Fatalf("served blocker = (%v, %v), want lease", running, err)
			}
			type result struct {
				id    string
				lease *AdmissionLease
				err   error
			}
			results := make(chan result, len(tc.requests))
			var cancels []context.CancelFunc
			started, received := 0, 0
			t.Cleanup(func() {
				for _, cancel := range cancels {
					cancel()
				}
				running.Release()
				for received < started {
					select {
					case got := <-results:
						received++
						got.lease.Release()
					case <-time.After(2 * time.Second):
						t.Error("served Acquire did not return during cleanup")
						return
					}
				}
			})
			queued := make(chan struct{}, len(tc.requests)+1)
			arrived := time.Now().Add(-time.Minute)
			ctl.SetClock(func() time.Time {
				select {
				case queued <- struct{}{}:
				default:
				}
				return arrived
			})
			for _, req := range tc.requests {
				base, cancel := context.WithCancel(context.Background())
				cancels = append(cancels, cancel)
				if req.id == "early" {
					// An incoming parent may be tighter than the declared header.
					base, cancel = context.WithDeadline(base, arrived.Add(5*time.Minute))
					cancels = append(cancels, cancel)
				}
				r := httptest.NewRequest("POST", "/v1/chat/completions", nil).WithContext(base)
				if req.budgetMS != "" {
					r.Header.Set("X-Fak-Deadline-Ms", req.budgetMS)
				}
				// Use the real HTTP ingress binder and served adapter, not a
				// pre-filled SeqRequest that could hide missing caller wiring.
				r, cancelDeadline := bindClientDeadline(r, arrived)
				cancels = append(cancels, cancelDeadline)
				started++
				go func(req request, ctx context.Context) {
					lease, err := s.beginServedAdmission(ctx, servedSessionTurn{traceID: req.id, state: SessionState{Priority: req.priority}}, nil, nil, 1)
					results <- result{req.id, lease, err}
				}(req, r.Context())
				select {
				case <-queued:
				case <-time.After(2 * time.Second):
					t.Fatal("served request never recorded a queued decision")
				}
				// Join the real Acquire's initial scheduling round before the
				// next enqueue. No sleeps, fake comparator, or direct promotion.
				ctl.mu.Lock()
				waiting := len(ctl.waiting)
				var queuedReq SeqRequest
				for _, entry := range ctl.waiting {
					if baseTraceID(entry.req.TraceID) == req.id {
						queuedReq = entry.req
					}
				}
				ctl.mu.Unlock()
				if deadline, has := r.Context().Deadline(); has {
					if queuedReq.CreatedAt.IsZero() || queuedReq.DecodeTTL <= 0 || !queuedReq.CreatedAt.Add(queuedReq.DecodeTTL).Equal(deadline) {
						t.Errorf("served deadline for %s = (%v, %v), want exact effective %v", req.id, queuedReq.CreatedAt, queuedReq.DecodeTTL, deadline)
					}
				} else if !queuedReq.CreatedAt.IsZero() || queuedReq.DecodeTTL != 0 {
					t.Errorf("no-deadline request %s acquired deadline metadata: %+v", req.id, queuedReq)
				}
				if waiting != started {
					t.Fatalf("waiting = %d, want %d", waiting, started)
				}
			}
			running.Release()
			for i, want := range tc.want {
				select {
				case got := <-results:
					received++
					got.lease.Release()
					if got.err != nil || got.lease == nil || got.id != want {
						t.Fatalf("served dequeue[%d] = (%q, %v, %v), want %q with lease", i, got.id, got.lease, got.err, want)
					}
				case <-time.After(2 * time.Second):
					t.Fatal("served waiter did not acquire after release")
				}
			}
			if st := ctl.Stats(); st.Running != 0 || st.Waiting != 0 || st.TokensInUse != 0 || st.QueuedTokens != 0 {
				t.Fatalf("final admission accounting = %+v, want zero", st)
			}
		})
	}
}
