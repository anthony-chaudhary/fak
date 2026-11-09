package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=100ms lane=default
func TestReplicaDispatchHedgeLateAbstention(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	for _, kind := range []string{"rejected", "valid", "transport_error"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			t.Cleanup(cancel)
			started, release := make(chan struct{}), make(chan struct{})
			var startOnce, releaseOnce sync.Once
			var hits, capacityCalls, validations atomic.Int32
			var gateTimedOut atomic.Bool
			var endpoints []string
			for _, name := range []string{"a", "b"} {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					hits.Add(1)
					startOnce.Do(func() { close(started) })
					// Only the post-delay capacity callback releases this response.
					// The primary therefore cannot win the early-completion branch.
					select {
					case <-release:
					case <-r.Context().Done():
						return
					}
					if kind == "transport_error" {
						http.Error(w, "fixture transport refusal", http.StatusTeapot)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"model":"fleet-model","choices":[{"message":{"role":"assistant","content":%q},"finish_reason":"stop"}]}`, kind)
				}))
				t.Cleanup(upstream.Close)
				endpoints = append(endpoints, name+"="+upstream.URL)
			}
			// Registered last so any failed assertion unblocks handlers before join.
			t.Cleanup(func() {
				cancel()
				releaseOnce.Do(func() { close(release) })
			})
			planner, err := newProxyPlanner(Config{Provider: "openai"}, "fleet-model", endpoints)
			if err != nil {
				t.Fatal(err)
			}
			router, ok := planner.(*ReplicaDispatch)
			if !ok {
				t.Fatalf("configured planner = %T, want real replica dispatch", planner)
			}
			var receipt HedgeReceipt
			policy := eligibleHedgePolicy(func(got HedgeReceipt) { receipt = got })
			policy.Deadline = 2 * time.Second
			policy.SpareCapacity = func() bool {
				capacityCalls.Add(1)
				select {
				case <-started:
				case <-ctx.Done():
					gateTimedOut.Store(true)
				}
				releaseOnce.Do(func() { close(release) })
				return false
			}
			policy.Validate = func(*agent.Completion) error {
				validations.Add(1)
				if kind == "rejected" {
					return errors.New("fixture rejected output")
				}
				return nil
			}
			router.Hedge = policy
			comp, err := router.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "read only"}}, nil, zeroTemperatureOpt())
			if gateTimedOut.Load() || ctx.Err() != nil {
				t.Fatalf("fixture timed out: gate=%t context=%v", gateTimedOut.Load(), ctx.Err())
			}
			if hits.Load() != 1 || capacityCalls.Load() != 1 || receipt.AbstentionReason != "no_spare_capacity" || receipt.Hedge.Opened || receipt.RealizedMode != "primary_only" {
				t.Fatalf("late abstention not witnessed: hits=%d capacity=%d receipt=%+v", hits.Load(), capacityCalls.Load(), receipt)
			}
			switch kind {
			case "rejected":
				if comp != nil || err == nil || err.Error() != "primary completion after hedge abstention: invalid_completion" || receipt.Primary.TerminalClass != "invalid_completion" || validations.Load() != 1 {
					t.Fatalf("invalid late primary escaped or was revalidated: completion=%+v error=%v class=%s validations=%d", comp, err, receipt.Primary.TerminalClass, validations.Load())
				}
			case "valid":
				if comp == nil || comp.Message.Content != kind || err != nil || receipt.Primary.TerminalClass != "valid" || validations.Load() != 1 {
					t.Fatalf("valid late primary changed: completion=%+v error=%v class=%s validations=%d", comp, err, receipt.Primary.TerminalClass, validations.Load())
				}
			case "transport_error":
				if comp != nil || err == nil || !strings.Contains(err.Error(), "418") || strings.Contains(err.Error(), "invalid_completion") || receipt.Primary.TerminalClass != "error" || validations.Load() != 0 {
					t.Fatalf("transport error changed: completion=%+v error=%v class=%s validations=%d", comp, err, receipt.Primary.TerminalClass, validations.Load())
				}
			}
		})
	}
}
