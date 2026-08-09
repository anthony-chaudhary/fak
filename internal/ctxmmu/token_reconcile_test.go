package ctxmmu

import "testing"

func intp(v int) *int { return &v }

func TestTokenReconcilerEndToEndConservesKnownUsage(t *testing.T) {
	r := NewTokenReconciler()
	forecast := TokenCounts{UncachedInput: 100, CachedInput: 400, Output: 80}
	if err := r.Admit("req-7", "profile-3", forecast); err != nil {
		t.Fatal(err)
	}
	if err := r.BeginAttempt("req-7", "profile-3"); err != nil {
		t.Fatal(err)
	}
	if err := r.Observe("req-7", "profile-3", TokenUsage{UncachedInput: intp(90), CachedInput: intp(410), Output: intp(70)}); err != nil {
		t.Fatal(err)
	}
	got, err := r.Complete("req-7", "profile-3", "success")
	if err != nil {
		t.Fatal(err)
	}
	if got.RequestID != "req-7" || got.ProfileID != "profile-3" || got.Attempts != 1 {
		t.Fatalf("identity not carried end to end: %+v", got)
	}
	if got.Released != (TokenCounts{UncachedInput: 10, CachedInput: 0, Output: 10}) {
		t.Fatalf("released = %+v", got.Released)
	}
	if *got.Observed.UncachedInput+got.Released.UncachedInput != got.Reserved.UncachedInput {
		t.Fatal("uncached reservation not conserved")
	}
	if *got.Observed.Output+got.Released.Output != got.Reserved.Output {
		t.Fatal("output reservation not conserved")
	}
	if *got.ForecastError.CachedInput != 10 || got.ReuseExpectationAccuracy == nil {
		t.Fatalf("missing reconciliation metrics: %+v", got)
	}
	read, ok := r.Read("req-7")
	if !ok || read.Status != "success" {
		t.Fatalf("readback failed: %+v %v", read, ok)
	}
}

func TestTokenReconcilerStreamingCancellationRetryAndProviderError(t *testing.T) {
	t.Run("streaming", func(t *testing.T) {
		r := NewTokenReconciler()
		mustAdmitAttempt(t, r, "stream")
		if err := r.Observe("stream", "p", TokenUsage{Output: intp(3)}); err != nil {
			t.Fatal(err)
		}
		if err := r.Observe("stream", "p", TokenUsage{Output: intp(4)}); err != nil {
			t.Fatal(err)
		}
		got, err := r.Complete("stream", "p", "success")
		if err != nil {
			t.Fatal(err)
		}
		if *got.Observed.Output != 7 {
			t.Fatalf("stream output = %d", *got.Observed.Output)
		}
	})
	t.Run("cancel preserves unknown", func(t *testing.T) {
		r := NewTokenReconciler()
		mustAdmitAttempt(t, r, "cancel")
		if err := r.Observe("cancel", "p", TokenUsage{Output: intp(2)}); err != nil {
			t.Fatal(err)
		}
		got, err := r.Complete("cancel", "p", "cancelled")
		if err != nil {
			t.Fatal(err)
		}
		if got.Observed.CachedInput != nil || got.ForecastError.CachedInput != nil || got.ReuseExpectationAccuracy != nil {
			t.Fatalf("missing provider detail treated as zero: %+v", got)
		}
	})
	t.Run("retry aggregates attempts", func(t *testing.T) {
		r := NewTokenReconciler()
		mustAdmitAttempt(t, r, "retry")
		if err := r.Observe("retry", "p", TokenUsage{Output: intp(2)}); err != nil {
			t.Fatal(err)
		}
		if err := r.BeginAttempt("retry", "p"); err != nil {
			t.Fatal(err)
		}
		if err := r.Observe("retry", "p", TokenUsage{Output: intp(5)}); err != nil {
			t.Fatal(err)
		}
		got, err := r.Complete("retry", "p", "success")
		if err != nil {
			t.Fatal(err)
		}
		if got.Attempts != 2 || *got.Observed.Output != 7 {
			t.Fatalf("retry not reconciled: %+v", got)
		}
	})
	t.Run("provider error releases reservation", func(t *testing.T) {
		r := NewTokenReconciler()
		mustAdmitAttempt(t, r, "error")
		got, err := r.Complete("error", "p", "provider_error")
		if err != nil {
			t.Fatal(err)
		}
		if got.Released != got.Reserved || got.Observed.Output != nil {
			t.Fatalf("provider error accounting = %+v", got)
		}
	})
}

func TestTokenReconcilerRejectsIdentityDrift(t *testing.T) {
	r := NewTokenReconciler()
	if err := r.Admit("r", "p1", TokenCounts{}); err != nil {
		t.Fatal(err)
	}
	if err := r.BeginAttempt("r", "p2"); err == nil {
		t.Fatal("profile drift admitted")
	}
}

func mustAdmitAttempt(t *testing.T, r *TokenReconciler, id string) {
	t.Helper()
	if err := r.Admit(id, "p", TokenCounts{UncachedInput: 10, CachedInput: 20, Output: 10}); err != nil {
		t.Fatal(err)
	}
	if err := r.BeginAttempt(id, "p"); err != nil {
		t.Fatal(err)
	}
}
