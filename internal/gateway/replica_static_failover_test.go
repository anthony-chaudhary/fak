package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=8s
// The real HTTP planners must share one request budget across serial placement,
// including immediate credential resends. These controls keep provider-directed
// waits and already-delivered stream content from becoming a failover loophole.
func TestReplicaSerialFailover(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	t.Setenv("FAK_INHANDLER_WAIT_CEILING", "90s")
	messages := []agent.Message{{Role: agent.RoleUser, Content: "serial failover"}}

	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
				w.WriteHeader(http.StatusBadGateway)
			})
			secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
			router := serialFailoverRouter(t, primary, secondary)
			var text strings.Builder
			var comp *agent.Completion
			var err error
			if stream {
				comp, err = router.CompleteStream(context.Background(), func(delta string) error {
					text.WriteString(delta)
					return nil
				}, messages, nil)
			} else {
				comp, err = router.Complete(context.Background(), messages, nil)
			}
			if err != nil || comp == nil || comp.Message.Content != "ok" {
				t.Fatalf("static failover = %+v, %v; HTTP attempts primary=%d secondary=%d", comp, err, first.Load(), second.Load())
			}
			if first.Load() != 1 || second.Load() != 1 {
				t.Fatalf("HTTP attempts = %d/%d, want one per distinct replica", first.Load(), second.Load())
			}
			if stream && text.String() != "ok" {
				t.Fatalf("stream = %q, want ok exactly once", text.String())
			}
		})
	}

	for _, nested := range []bool{false, true} {
		name := "aggregate_attempt_limit"
		if nested {
			name = "nested_dispatch_keeps_parent_budget"
		}
		t.Run(name, func(t *testing.T) {
			reject := func(w http.ResponseWriter, _ *http.Request, _ int32) { w.WriteHeader(http.StatusBadGateway) }
			primary, first := serialFailoverHTTP(t, "primary", reject)
			secondary, second := serialFailoverHTTP(t, "secondary", reject)
			lastHandler := reject
			if nested {
				lastHandler = serialFailoverSuccess
			}
			last, third := serialFailoverHTTP(t, "last", lastHandler)
			var router *ReplicaDispatch
			if nested {
				child := serialFailoverRouter(t, secondary, last)
				router = serialFailoverRouter(t, primary, PlannerReplica{Name: "nested", Planner: child})
			} else {
				router = serialFailoverRouter(t, primary, secondary, last)
			}
			_, err := router.Complete(context.Background(), messages, nil)
			var status *agent.UpstreamStatusError
			if !errors.As(err, &status) || status.Status != http.StatusBadGateway {
				t.Fatalf("budget exhaustion lost final upstream status: %v", err)
			}
			if first.Load() != 1 || second.Load()+third.Load() != 1 || (nested && third.Load() != 0) {
				t.Fatalf("HTTP attempts = %d/%d/%d, want two distinct sends under shared limit 2", first.Load(), second.Load(), third.Load())
			}
		})
	}

	for _, status := range []int{http.StatusMethodNotAllowed, http.StatusServiceUnavailable} {
		name := "nontransient_is_terminal"
		if status == http.StatusServiceUnavailable {
			name = "provider_wait_ceiling_is_terminal"
		}
		t.Run(name, func(t *testing.T) {
			if status == http.StatusServiceUnavailable {
				t.Setenv("FAK_INHANDLER_WAIT_CEILING", "1ms")
			}
			primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
				if status == http.StatusServiceUnavailable {
					w.Header().Set("Retry-After", "120")
				}
				w.WriteHeader(status)
			})
			secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
			_, err := serialFailoverRouter(t, primary, secondary).Complete(context.Background(), messages, nil)
			var upstream *agent.UpstreamStatusError
			if !errors.As(err, &upstream) || upstream.Status != status {
				t.Fatalf("terminal status = %v, want HTTP %d", err, status)
			}
			if status == http.StatusServiceUnavailable {
				var ceiling *agent.RetryCeilingError
				if !errors.As(err, &ceiling) || upstream.RetryAfter != "120" {
					t.Fatalf("provider-directed ceiling was discarded: %v", err)
				}
			}
			if first.Load() != 1 || second.Load() != 0 {
				t.Fatalf("terminal failure retried: HTTP attempts=%d/%d", first.Load(), second.Load())
			}
		})
	}

	t.Run("prior_status_survives_transport_only_replicas", func(t *testing.T) {
		t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverDisconnect)
		last, third := serialFailoverHTTP(t, "last", serialFailoverDisconnect)
		_, err := serialFailoverRouter(t, primary, secondary, last).Complete(context.Background(), messages, nil)
		var status *agent.UpstreamStatusError
		if !errors.As(err, &status) || status.Status != http.StatusServiceUnavailable || first.Load() != 1 || second.Load() != 1 || third.Load() != 1 {
			t.Fatalf("transport-only replicas lost prior status: %v; HTTP attempts=%d/%d/%d", err, first.Load(), second.Load(), third.Load())
		}
	})

	t.Run("handoff_cancellation_keeps_prior_status", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", func(w http.ResponseWriter, r *http.Request, n int32) {
			cancel()
			serialFailoverDisconnect(w, r, n)
		})
		_, err := serialFailoverRouter(t, primary, secondary).Complete(ctx, messages, nil)
		var status *agent.UpstreamStatusError
		if !errors.Is(err, context.Canceled) || !errors.As(err, &status) || status.Status != http.StatusServiceUnavailable || first.Load() != 1 || second.Load() != 1 {
			t.Fatalf("handoff cancellation lost prior status: %v; HTTP attempts=%d/%d", err, first.Load(), second.Load())
		}
	})

	t.Run("newer_body_failure_supersedes_previous_status", func(t *testing.T) {
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.WriteHeader(http.StatusBadGateway)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", "500")
			io.WriteString(w, `{"choices":[]}`)
		})
		_, err := serialFailoverRouter(t, primary, secondary).Complete(context.Background(), messages, nil)
		var status *agent.UpstreamStatusError
		if !errors.Is(err, io.ErrUnexpectedEOF) || errors.As(err, &status) || first.Load() != 1 || second.Load() != 1 {
			t.Fatalf("newer body failure lost precedence: %v; HTTP attempts=%d/%d", err, first.Load(), second.Load())
		}
	})

	t.Run("newer_stream_response_clears_status_before_next_replica", func(t *testing.T) {
		t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Content-Length", "500")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{}}]}\n\n")
		})
		last, third := serialFailoverHTTP(t, "last", serialFailoverDisconnect)
		router := serialFailoverRouter(t, primary, secondary, last).WithPickPolicy(&pickFirstPolicy{})
		_, err := router.CompleteStream(context.Background(), nil, messages, nil)
		var status *agent.UpstreamStatusError
		if !errors.Is(err, io.EOF) || errors.As(err, &status) || first.Load() != 1 || second.Load() != 1 || third.Load() != 1 {
			t.Fatalf("accepted stream response left stale status: %v; HTTP attempts=%d/%d/%d", err, first.Load(), second.Load(), third.Load())
		}
	})

	for _, exhaust := range []bool{false, true} {
		name := "credential_refresh_retained"
		if exhaust {
			name = "credential_resend_consumes_shared_budget"
		}
		t.Run(name, func(t *testing.T) {
			primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, r *http.Request, n int32) {
				if n == 1 {
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				if exhaust {
					w.WriteHeader(http.StatusBadGateway)
					return
				}
				serialFailoverSuccess(w, r, n)
			})
			primary.Planner.(*agent.HTTPPlanner).APIKeyFunc = func() string {
				if first.Load() == 0 {
					return "synthetic-before-refresh"
				}
				return "synthetic-after-refresh"
			}
			secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
			comp, err := serialFailoverRouter(t, primary, secondary).Complete(context.Background(), messages, nil)
			if exhaust {
				var status *agent.UpstreamStatusError
				if !errors.As(err, &status) || status.Status != http.StatusBadGateway {
					t.Fatalf("resend budget lost terminal status: %v", err)
				}
			} else if err != nil || comp == nil || comp.Message.Content != "ok" {
				t.Fatalf("credential refresh failed: %+v, %v", comp, err)
			}
			if first.Load() != 2 || second.Load() != 0 {
				t.Fatalf("credential attempts = %d/%d, want 2/0", first.Load(), second.Load())
			}
		})
	}

	t.Run("single_replica_keeps_local_retry", func(t *testing.T) {
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, r *http.Request, n int32) {
			if n == 1 {
				w.Header().Set("Retry-After", "0")
				w.WriteHeader(http.StatusBadGateway)
				return
			}
			serialFailoverSuccess(w, r, n)
		})
		comp, err := serialFailoverRouter(t, primary).Complete(context.Background(), messages, nil)
		if err != nil || comp == nil || comp.Message.Content != "ok" || first.Load() != 2 {
			t.Fatalf("single replica local retry = %+v, %v; attempts=%d", comp, err, first.Load())
		}
	})

	t.Run("stream_required_recovery_keeps_shared_budget", func(t *testing.T) {
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, r *http.Request, n int32) {
			if n == 1 {
				w.WriteHeader(http.StatusInternalServerError)
				io.WriteString(w, `{"message":"stream required"}`)
				return
			}
			serialFailoverSuccess(w, r, n)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
		comp, err := serialFailoverRouter(t, primary, secondary).Complete(context.Background(), messages, nil)
		if err != nil || comp == nil || comp.Message.Content != "ok" || first.Load() != 2 || second.Load() != 0 {
			t.Fatalf("stream-required recovery = %+v, %v; HTTP attempts=%d/%d", comp, err, first.Load(), second.Load())
		}
	})

	for _, frame := range []struct {
		name, delta, content string
		held                 int32
	}{
		{"content", `{"content":"partial"}`, "partial", 0},
		{"reasoning", `{"reasoning_content":"hidden"}`, "", 1},
		{"tool", `{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}`, "", 1},
	} {
		t.Run("no_retarget_after_stream_"+frame.name, func(t *testing.T) {
			primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Content-Length", "500")
				io.WriteString(w, "data: {\"choices\":[{\"delta\":"+frame.delta+"}]}\n\n")
			})
			secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
			var held atomic.Int32
			ctx := agent.WithStreamHeldActivity(context.Background(), func() { held.Add(1) })
			var text strings.Builder
			_, err := serialFailoverRouter(t, primary, secondary).CompleteStream(ctx, func(delta string) error {
				text.WriteString(delta)
				return nil
			}, messages, nil)
			if !errors.Is(err, io.ErrUnexpectedEOF) || text.String() != frame.content || held.Load() != frame.held || first.Load() != 1 || second.Load() != 0 {
				t.Fatalf("stream replay boundary: text=%q held=%d err=%v HTTP attempts=%d/%d", text.String(), held.Load(), err, first.Load(), second.Load())
			}
		})
	}

	t.Run("caller_cancellation_is_terminal", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		primary, first := serialFailoverHTTP(t, "primary", func(w http.ResponseWriter, _ *http.Request, _ int32) {
			cancel()
			w.WriteHeader(http.StatusBadGateway)
		})
		secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
		_, err := serialFailoverRouter(t, primary, secondary).Complete(ctx, messages, nil)
		if !errors.Is(err, context.Canceled) || first.Load() != 1 || second.Load() != 0 {
			t.Fatalf("cancellation = %v; HTTP attempts=%d/%d", err, first.Load(), second.Load())
		}
	})

	t.Run("shared_deadline_bounds_first_attempt", func(t *testing.T) {
		t.Setenv("FAK_PLANNER_RETRY_BUDGET", "1s")
		release := make(chan struct{})
		defer close(release)
		primary, first := serialFailoverHTTP(t, "primary", func(_ http.ResponseWriter, r *http.Request, _ int32) {
			_, _ = io.Copy(io.Discard, r.Body)
			select {
			case <-r.Context().Done():
			case <-release:
			}
		})
		secondary, second := serialFailoverHTTP(t, "secondary", serialFailoverSuccess)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := serialFailoverRouter(t, primary, secondary).Complete(ctx, messages, nil)
		if !errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil || first.Load() != 1 || second.Load() != 0 {
			t.Fatalf("shared retry deadline = %v, caller=%v; HTTP attempts=%d/%d", err, ctx.Err(), first.Load(), second.Load())
		}
	})
}

func serialFailoverHTTP(t *testing.T, name string, handler func(http.ResponseWriter, *http.Request, int32)) (PlannerReplica, *atomic.Int32) {
	t.Helper()
	hits := new(atomic.Int32)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler(w, r, hits.Add(1))
	}))
	t.Cleanup(server.Close)
	planner := agent.NewHTTPPlanner(server.URL, "fleet-model", "")
	planner.Client = server.Client()
	return PlannerReplica{Name: name, Planner: planner, Endpoint: server.URL}, hits
}

func serialFailoverDisconnect(w http.ResponseWriter, _ *http.Request, _ int32) {
	conn, _, err := w.(http.Hijacker).Hijack()
	if err == nil {
		_ = conn.Close()
	}
}

func serialFailoverRouter(t *testing.T, replicas ...PlannerReplica) *ReplicaDispatch {
	t.Helper()
	router, err := NewReplicaDispatch("fleet-model", replicas)
	if err != nil {
		t.Fatal(err)
	}
	return router.WithPickPolicy(namedPickPolicy(replicas[0].Name))
}

func serialFailoverSuccess(w http.ResponseWriter, r *http.Request, _ int32) {
	var request struct{ Stream bool }
	_ = json.NewDecoder(r.Body).Decode(&request)
	if request.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"model":"fleet-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
}
