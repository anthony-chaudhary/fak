package launchguard

import (
	"os"
	"sync"
	"testing"
	"time"
)

func testGuard(t *testing.T, now *time.Time, alive func(int) bool) *Guard {
	t.Helper()
	g, err := New(Config{Root: t.TempDir(), MaxAttempts: 3, Window: time.Minute, InitialBackoff: time.Second, MaxBackoff: 4 * time.Second, Jitter: 200 * time.Millisecond, StaleAfter: 5 * time.Second, Now: func() time.Time { return *now }, JitterValue: func(time.Duration) time.Duration { return 100 * time.Millisecond }, ProcessAlive: alive, PID: 42})
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestAdmitBoundsAttemptsBacksOffAndRequiresReset(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	g := testGuard(t, &now, func(int) bool { return true })
	for attempt, wantDelay := range []time.Duration{1100 * time.Millisecond, 2100 * time.Millisecond, 4100 * time.Millisecond} {
		d, err := g.Admit("service:alpha")
		if err != nil {
			t.Fatal(err)
		}
		if d.Outcome != Admitted || d.Attempts != attempt+1 {
			t.Fatalf("attempt %d: %+v", attempt+1, d)
		}
		if got := d.RetryAt.Sub(now); got != wantDelay {
			t.Fatalf("attempt %d delay=%s want %s", attempt+1, got, wantDelay)
		}
		if err := g.Complete("service:alpha", false); err != nil {
			t.Fatal(err)
		}
		if attempt < 2 {
			b, err := g.Admit("service:alpha")
			if err != nil {
				t.Fatal(err)
			}
			if b.Outcome != Backoff {
				t.Fatalf("attempt %d immediate=%+v", attempt+1, b)
			}
			if err := g.Complete("service:alpha", false); err != nil {
				t.Fatal(err)
			}
			now = d.RetryAt
		}
	}
	now = now.Add(time.Hour)
	q, err := g.Admit("service:alpha")
	if err != nil {
		t.Fatal(err)
	}
	if q.Outcome != Quarantined || !q.Quarantine {
		t.Fatalf("quarantine=%+v", q)
	}
	if err := g.Complete("service:alpha", false); err != nil {
		t.Fatal(err)
	}
	now = now.Add(24 * time.Hour)
	q, _ = g.Admit("service:alpha")
	if q.Outcome != Quarantined {
		t.Fatalf("quarantine reset silently: %+v", q)
	}
	if err := g.Complete("service:alpha", false); err != nil {
		t.Fatal(err)
	}
	if err := g.Reset("service:alpha"); err != nil {
		t.Fatal(err)
	}
	d, err := g.Admit("service:alpha")
	if err != nil || d.Outcome != Admitted || d.Attempts != 1 {
		t.Fatalf("after reset=%+v err=%v", d, err)
	}
}

func TestConcurrentAdmissionHasOneOwner(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	g := testGuard(t, &now, func(int) bool { return true })
	const n = 32
	start := make(chan struct{})
	out := make(chan Outcome, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			d, err := g.Admit("agent:crash-rsi")
			if err != nil {
				t.Errorf("admit: %v", err)
				return
			}
			out <- d.Outcome
		}()
	}
	close(start)
	wg.Wait()
	close(out)
	admitted, duplicate := 0, 0
	for got := range out {
		if got == Admitted {
			admitted++
		} else if got == DuplicateActive {
			duplicate++
		} else {
			t.Fatalf("unexpected outcome %q", got)
		}
	}
	if admitted != 1 || duplicate != n-1 {
		t.Fatalf("admitted=%d duplicate=%d", admitted, duplicate)
	}
}

func TestStaleOwnerRecoveryIsTyped(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 10, 0, time.UTC)
	g := testGuard(t, &now, func(int) bool { return false })
	id := Identity("service:stale")
	if err := writeJSON(g.ownerPath(id), owner{PID: 99, StartedAt: now.Add(-time.Minute)}); err != nil {
		t.Fatal(err)
	}
	old := now.Add(-time.Minute)
	if err := os.Chtimes(g.ownerPath(id), old, old); err != nil {
		t.Fatal(err)
	}
	d, err := g.Admit("service:stale")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != StaleRecovered {
		t.Fatalf("outcome=%+v", d)
	}
}

func TestIdentityIsScrubbed(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	g := testGuard(t, &now, func(int) bool { return true })
	raw := "secret-bearing/service/path"
	if _, err := g.Admit(raw); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(g.c.Root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Name() == raw || contains(entry.Name(), "secret") {
			t.Fatalf("raw identity leaked in %q", entry.Name())
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
