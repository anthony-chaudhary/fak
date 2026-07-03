package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"
)

// TestComplete429AccountCapRehomesSeat proves the goal seam: a 429 that classifies as an ACCOUNT
// CAP (session/weekly/usage) rehomes the session ONCE to a permitted sibling seat and completes the
// turn on it — instead of sleeping on the capped seat toward a reset that can be hours away. It also
// proves the boundary the user flagged ("429 is often longer than it seems"): the rehome fires on the
// cap class, NOT on a transient rate_limited throttle, which keeps its seat and rides the short
// backoff. And it proves the no-free-seat path degrades to the existing cap-aware backoff, reporting
// the spent rehome so the give-up is visible rather than silent.
func TestComplete429AccountCapRehomesSeat(t *testing.T) {
	const capped = "sk-ant-oat01-capped-seat"
	const free = "sk-ant-oat01-free-seat"

	// A session-cap 429: the body names the 5h session limit and the unified header relays a reset
	// ~90 minutes out — the exact "longer than it looks" case. isAccountCap429 must see this as a cap.
	sessionCap := anthropicLimitBody("You've hit your session limit · resets 8pm (America/Los_Angeles).")
	capReset := strconv.FormatInt(time.Now().Add(90*time.Minute).Unix(), 10)

	okBody := []byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)

	t.Run("session-cap 429 rehomes to a free sibling seat and completes on it", func(t *testing.T) {
		var mu sync.Mutex
		var gotAuth []string
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			auth := r.Header.Get("Authorization")
			gotAuth = append(gotAuth, auth)
			mu.Unlock()
			// The capped seat keeps 429ing on its cap; only the free sibling seat serves a 200.
			if auth != "Bearer "+free {
				w.Header().Set("Anthropic-Ratelimit-Unified-5h-Reset", capReset)
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write(sessionCap)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(okBody)
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", capped)
		if err != nil {
			t.Fatal(err)
		}
		var reasons []string
		planner.AccountFailoverFunc = func(reason string) (string, bool) {
			reasons = append(reasons, reason)
			return free, true
		}
		var outcomes []string
		planner.AccountFailoverNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

		if _, err := planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools()); err != nil {
			t.Fatalf("complete should succeed after the 429-cap seat rehome: %v", err)
		}
		want := []string{"Bearer " + capped, "Bearer " + free}
		if len(gotAuth) != len(want) {
			t.Fatalf("got %d upstream requests %q, want %d — one capped 429 then one free-seat retry", len(gotAuth), gotAuth, len(want))
		}
		for i := range want {
			if gotAuth[i] != want[i] {
				t.Errorf("request %d Authorization = %q, want %q (the rehome must carry the sibling seat's token)", i, gotAuth[i], want[i])
			}
		}
		if len(reasons) != 1 || reasons[0] != RehomedSeat {
			t.Errorf("failover reason = %v, want exactly one %q (the seat-rehome label, not the org-wall one)", reasons, RehomedSeat)
		}
		if len(outcomes) != 1 || outcomes[0] != RehomedSeat {
			t.Errorf("rehome outcomes = %v, want exactly one %q (confirmed rehome on the 200)", outcomes, RehomedSeat)
		}
	})

	t.Run("transient rate_limited throttle does NOT rehome — keeps its seat", func(t *testing.T) {
		// A plain throttle with no unified-reset header: isAccountCap429 is false, so the seat is
		// kept and the short backoff rides it out. The same seat clears on the next attempt.
		var mu sync.Mutex
		var gotAuth []string
		var n int
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			gotAuth = append(gotAuth, r.Header.Get("Authorization"))
			n++
			first := n == 1
			mu.Unlock()
			if first {
				// Throttle once (no reset header => rate_limited), then serve.
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte("Too many requests"))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(okBody)
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", capped)
		if err != nil {
			t.Fatal(err)
		}
		// Speed up the transient backoff so the test is fast.
		t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "4")
		var failoverCalls int
		planner.AccountFailoverFunc = func(string) (string, bool) { failoverCalls++; return free, true }

		if _, err := planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools()); err != nil {
			t.Fatalf("complete should succeed after a transient throttle: %v", err)
		}
		if failoverCalls != 0 {
			t.Errorf("AccountFailoverFunc called %d times on a transient throttle, want 0 (no rehome — it keeps its seat)", failoverCalls)
		}
		for i, a := range gotAuth {
			if a != "Bearer "+capped {
				t.Errorf("request %d Authorization = %q, want the ORIGINAL seat %q (a throttle must not swap seats)", i, a, "Bearer "+capped)
			}
		}
	})

	t.Run("no free sibling seat => reports spent rehome, then rides the cap-aware backoff", func(t *testing.T) {
		// Every seat is capped: the func reports no free seat. The rehome is reported spent
		// (rehome_seat_unavailable) exactly once, and the loop falls through to the cap-aware wait.
		// A short FAK_CAP_PROBE fallback + a low attempt cap keeps the test fast; the turn still
		// surfaces the cap terminally (no free seat, cap holds), which is the correct give-up.
		var n int
		var mu sync.Mutex
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			n++
			mu.Unlock()
			w.Header().Set("Anthropic-Ratelimit-Unified-5h-Reset", capReset)
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write(sessionCap)
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", capped)
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
		planner.AccountFailoverFunc = func(string) (string, bool) { return "", false }
		var outcomes []string
		planner.AccountFailoverNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

		_, err = planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools())
		if err == nil {
			t.Fatal("want the cap to surface terminally when every seat is capped (no free seat)")
		}
		unavailable := 0
		for _, o := range outcomes {
			if o == RehomeSeatUnavailable {
				unavailable++
			}
			if o == RehomedSeat {
				t.Errorf("outcomes = %v, must not contain %q when no free seat exists", outcomes, RehomedSeat)
			}
		}
		if unavailable != 1 {
			t.Errorf("outcomes = %v, want exactly one %q (spent rehome reported once, not per capped attempt)", outcomes, RehomeSeatUnavailable)
		}
	})
}

// TestIsAccountCap429 pins the discriminator the rehome depends on: a session/weekly/usage cap is a
// rehome-worthy account cap; a transient rate_limited throttle and a non-429 overload are not.
func TestIsAccountCap429(t *testing.T) {
	now := time.Now()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-5h-Reset", strconv.FormatInt(now.Add(30*time.Minute).Unix(), 10))
	if !isAccountCap429(429, anthropicLimitBody("You've hit your session limit · resets 8pm."), h, now) {
		t.Error("session-cap 429 must be an account cap (rehome-worthy)")
	}
	if !isAccountCap429(429, anthropicLimitBody("usage limit reached"), nil, now) {
		t.Error("usage-limit 429 (even with no reset header) must be an account cap")
	}
	if isAccountCap429(429, []byte("Too many requests"), nil, now) {
		t.Error("a transient rate_limited throttle must NOT be an account cap — it keeps its seat")
	}
	if isAccountCap429(statusOverloaded, []byte("overloaded"), nil, now) {
		t.Error("a 529 overload is not a 429 account cap")
	}
	if isAccountCap429(http.StatusServiceUnavailable, []byte("unavailable"), nil, now) {
		t.Error("a 503 is not a 429 account cap")
	}
}
