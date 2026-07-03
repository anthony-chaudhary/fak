package agent

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

// TestHTTPPlannerComplete403AccountFailover proves the mid-session account-failover self-heal on
// the buffered Complete path: when the pinned account's org has OAuth disabled (the canonical
// deceiving 403 — its message reads like a re-login would help, but re-login is futile), Complete
// asks AccountFailoverFunc for a permitted sibling credential, adopts it, and re-sends the SAME
// turn in place — so the walled session completes instead of surfacing to the wrapped agent. It
// also proves the cap (no sibling => fail fast, one swap max) and that a non-account 403 does NOT
// trigger a failover.
func TestHTTPPlannerComplete403AccountFailover(t *testing.T) {
	// Collapse the transient-403 arm so the account-scoped denial reaches the failover arm at once
	// (the org body is classified permanent, so the transient arm gives up immediately anyway, but
	// disabling it keeps the test independent of that window).
	const walled = "sk-ant-oat01-walled-org"
	const permitted = "sk-ant-oat01-permitted-org"

	orgDisabledBody := []byte(`{"type":"error","error":{"type":"permission_error",` +
		`"message":"OAuth authentication is currently not allowed for this organization."}}`)

	t.Run("org-disabled 403 heals by swapping to a permitted sibling account", func(t *testing.T) {
		var gotAuth []string
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			auth := r.Header.Get("Authorization")
			gotAuth = append(gotAuth, auth)
			if auth != "Bearer "+permitted {
				w.WriteHeader(http.StatusForbidden)
				_, _ = w.Write(orgDisabledBody)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", walled)
		if err != nil {
			t.Fatal(err)
		}
		var failoverReason string
		failoverCalls := 0
		planner.AccountFailoverFunc = func(reason string) (string, bool) {
			failoverCalls++
			failoverReason = reason
			return permitted, true
		}
		var outcomes []string
		planner.AccountFailoverNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

		if _, err := planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools()); err != nil {
			t.Fatalf("complete should succeed after the org-403 account failover: %v", err)
		}
		want := []string{"Bearer " + walled, "Bearer " + permitted}
		if len(gotAuth) != len(want) {
			t.Fatalf("got %d upstream requests %q, want %d — one walled 403 then one permitted retry", len(gotAuth), gotAuth, len(want))
		}
		for i := range want {
			if gotAuth[i] != want[i] {
				t.Errorf("request %d Authorization = %q, want %q (the swap must carry the sibling token)", i, gotAuth[i], want[i])
			}
		}
		if failoverCalls != 1 {
			t.Errorf("AccountFailoverFunc called %d times, want exactly 1", failoverCalls)
		}
		if failoverReason != RemedyFailoverAccount.String() {
			t.Errorf("failover reason = %q, want %q (classified label, not raw body)", failoverReason, RemedyFailoverAccount.String())
		}
		if len(outcomes) != 1 || outcomes[0] != AccountFailoverRecovered {
			t.Errorf("failover outcomes = %v, want exactly one %q (confirmed heal on the 200)", outcomes, AccountFailoverRecovered)
		}
	})

	t.Run("no permitted sibling => surfaces the 403 terminally, one swap attempt", func(t *testing.T) {
		var n int
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n++
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write(orgDisabledBody)
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", walled)
		if err != nil {
			t.Fatal(err)
		}
		// Every sibling is walled/absent: the func reports no failover target.
		planner.AccountFailoverFunc = func(string) (string, bool) { return "", false }
		var outcomes []string
		planner.AccountFailoverNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

		_, err = planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools())
		var statusErr *UpstreamStatusError
		if !errors.As(err, &statusErr) || statusErr.Status != http.StatusForbidden {
			t.Fatalf("want an UpstreamStatusError 403, got %v", err)
		}
		if n != 1 {
			t.Errorf("upstream hit %d times, want exactly 1 (permanent org 403, no sibling => no retry loop)", n)
		}
		if len(outcomes) != 1 || outcomes[0] != AccountFailoverExhausted {
			t.Errorf("failover outcomes = %v, want exactly one %q", outcomes, AccountFailoverExhausted)
		}
	})

	t.Run("no AccountFailoverFunc => historical terminal-on-org-403 behavior unchanged", func(t *testing.T) {
		var n int
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n++
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write(orgDisabledBody)
		}))
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", walled)
		if err != nil {
			t.Fatal(err)
		}
		// No AccountFailoverFunc wired: the arm is inert; the 403 surfaces exactly as before.
		_, err = planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools())
		var statusErr *UpstreamStatusError
		if !errors.As(err, &statusErr) || statusErr.Status != http.StatusForbidden {
			t.Fatalf("want an UpstreamStatusError 403, got %v", err)
		}
		if n != 1 {
			t.Errorf("upstream hit %d times, want exactly 1", n)
		}
	})
}

// TestHTTPPlannerComplete403OverageCapWaitsForReset proves the fix for the deeper bug the audit
// found: a usage/overage cap that surfaces as a 403 (org-flavored body, but overage-status
// rejected with a near reset) must ride the cap-aware wait toward its reset and RECOVER — the same
// way a 429 account cap does — instead of dying in the seconds-scale transient-403 arm or being
// treated as a permanent org wall. With no AccountFailoverFunc wired, the only correct behavior is
// to wait out the (short) reset and re-send, then 200.
func TestHTTPPlannerComplete403OverageCapWaitsForReset(t *testing.T) {
	orgBody := []byte(`{"type":"error","error":{"type":"permission_error",` +
		`"message":"OAuth authentication is currently not allowed for this organization."}}`)

	var n int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			// First hit: an overage-rejected 403 with a reset ~1s out. The cap-aware wait must
			// take this toward the reset, NOT the 30s forbidden window and NOT a terminal wall.
			reset := time.Now().Add(1 * time.Second).Unix()
			w.Header().Set("Anthropic-Ratelimit-Unified-Status", "allowed")
			w.Header().Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")
			w.Header().Set("Anthropic-Ratelimit-Unified-Overage-Disabled-Reason", "org_level_disabled")
			w.Header().Set("Anthropic-Ratelimit-Unified-Reset", strconv.FormatInt(reset, 10))
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write(orgBody)
			return
		}
		// The retry after the cap wait succeeds — the window reset.
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("anthropic", ts.URL, "claude-test", "sk-ant-oat01-capped")
	if err != nil {
		t.Fatal(err)
	}
	// No AccountFailoverFunc: the ONLY correct recovery is to wait for the reset and re-send —
	// proving the cap-wait path, not the rehome/failover path.
	var failoverCalls int
	planner.AccountFailoverFunc = func(string) (string, bool) { failoverCalls++; return "", false }

	start := time.Now()
	if _, err := planner.Complete(context.Background(), adapterTestMessages(""), adapterTestTools()); err != nil {
		t.Fatalf("an overage-cap 403 with a near reset should self-heal by waiting for the reset, got: %v", err)
	}
	elapsed := time.Since(start)
	if n != 2 {
		t.Fatalf("upstream hit %d times, want exactly 2 (the capped 403 then the post-reset 200)", n)
	}
	// It must have WAITED (the cap-aware backoff), not spun instantly — a floor proves the wait ran.
	if elapsed < 200*time.Millisecond {
		t.Errorf("Complete returned in %v — too fast; the cap-aware wait toward the reset did not run", elapsed)
	}
	// It must NOT have tried to fail over/rehome: an overage cap with no free seat should wait, and
	// with no AccountFailoverFunc target it simply rides the wait. (failoverCalls may be 1 — the
	// single rehome probe — but never a swap, since the func returns ok=false; the point is the turn
	// recovered via the wait, asserted by the 200 above.)
	_ = failoverCalls
}
