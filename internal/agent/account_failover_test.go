package agent

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
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
