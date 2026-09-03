package launchguard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time      { c.mu.Lock(); defer c.mu.Unlock(); return c.now }
func (c *fakeClock) Add(d time.Duration) { c.mu.Lock(); c.now = c.now.Add(d); c.mu.Unlock() }

func newTestGuard(t *testing.T, clock *fakeClock) *Guard {
	t.Helper()
	g, err := New(Config{
		Dir: t.TempDir(), MaxAttempts: 3, Window: time.Minute,
		BaseBackoff: time.Second, MaxBackoff: 8 * time.Second,
		StaleAfter: 10 * time.Second, Clock: clock.Now,
		Jitter:   func(d time.Duration) time.Duration { return d + 250*time.Millisecond },
		PIDAlive: func(pid int) bool { return pid == 100 }, PID: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestStableIdentityStoresOnlyHash(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	const identity = "secret-service-name"
	d, lease, err := g.Admit(identity)
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != Admitted || lease == nil {
		t.Fatalf("decision = %+v, lease=%v", d, lease)
	}
	if d.Identity == identity || len(d.Identity) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST sha256 hex width is a fixed 64-character invariant
		t.Fatalf("identity = %q", d.Identity)
	}
	entries, err := os.ReadDir(g.cfg.Dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Name() != d.Identity+".json" && entry.Name() != d.Identity+".owner" {
			t.Fatalf("unexpected state file %q", entry.Name())
		}
		data, err := os.ReadFile(filepath.Join(g.cfg.Dir, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if stringContains(string(data), identity) {
			t.Fatalf("state leaks identity: %s", data)
		}
	}
	if err := lease.Finish(true); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentAdmissionHasOneWinner(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	const workers = 32
	start := make(chan struct{})
	results := make(chan Outcome, workers)
	leases := make(chan *Lease, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			d, l, err := g.Admit("same")
			if err != nil {
				t.Errorf("Admit: %v", err)
				return
			}
			results <- d.Outcome
			if l != nil {
				leases <- l
			}
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	close(leases)
	counts := map[Outcome]int{}
	for outcome := range results {
		counts[outcome]++
	}
	if counts[Admitted] != 1 || counts[DuplicateActive] != workers-1 {
		t.Fatalf("outcomes = %#v", counts)
	}
	for lease := range leases {
		if err := lease.Finish(true); err != nil {
			t.Fatal(err)
		}
	}
}

func TestBackoffQuarantineStatusAndReset(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	for attempt := 1; attempt <= 3; attempt++ {
		d, lease, err := g.Admit("worker")
		if err != nil {
			t.Fatal(err)
		}
		if d.Outcome != Admitted {
			t.Fatalf("attempt %d: %+v", attempt, d)
		}
		if err := lease.Finish(false); err != nil {
			t.Fatal(err)
		}
		if attempt < 3 {
			d, lease, err = g.Admit("worker")
			if err != nil {
				t.Fatal(err)
			}
			want := time.Duration(1<<(attempt-1))*time.Second + 250*time.Millisecond
			if d.Outcome != Backoff || d.RetryAfter != want || lease != nil {
				t.Fatalf("attempt %d backoff = %+v lease=%v, want %v", attempt, d, lease, want)
			}
			clock.Add(want)
		}
	}
	d, lease, err := g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != Quarantined || lease != nil {
		t.Fatalf("quarantine = %+v lease=%v", d, lease)
	}
	status, err := g.Inspect("worker")
	if err != nil {
		t.Fatal(err)
	}
	if !status.Quarantined || status.Attempts != 3 || status.Active {
		t.Fatalf("status = %+v", status)
	}
	clock.Add(2 * time.Minute)
	d, _, err = g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != Quarantined {
		t.Fatalf("quarantine silently expired: %+v", d)
	}
	if err := g.Reset("worker"); err != nil {
		t.Fatal(err)
	}
	status, err = g.Inspect("worker")
	if err != nil {
		t.Fatal(err)
	}
	if status.Quarantined || status.Attempts != 0 {
		t.Fatalf("reset status = %+v", status)
	}
}

func TestSuccessClearsBudget(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	_, lease, err := g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Finish(true); err != nil {
		t.Fatal(err)
	}
	status, err := g.Inspect("worker")
	if err != nil {
		t.Fatal(err)
	}
	if status.Attempts != 0 || !status.LastFailure.IsZero() {
		t.Fatalf("status = %+v", status)
	}
}

func TestStaleOwnerRecoveryRequiresDeadPIDAndAge(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	id := StableIdentity("worker")
	writeOwner := func(pid int, age time.Duration) {
		t.Helper()
		o := owner{PID: pid, Token: "old", CreatedAt: clock.Now().Add(-age)}
		data, _ := json.Marshal(o)
		if err := os.WriteFile(g.ownerPath(id), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	writeOwner(200, 5*time.Second)
	d, lease, err := g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != DuplicateActive || lease != nil {
		t.Fatalf("young dead owner = %+v", d)
	}

	if err := os.WriteFile(g.ownerPath(id), []byte(`{"pid":100,"token":"live","created_at":"1970-01-01T00:00:00Z"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	d, lease, err = g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != DuplicateActive || lease != nil {
		t.Fatalf("live old owner = %+v", d)
	}

	writeOwner(200, 20*time.Second)
	d, lease, err = g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if d.Outcome != StaleRecovered || lease == nil {
		t.Fatalf("stale recovery = %+v lease=%v", d, lease)
	}
	if err := lease.Finish(true); err != nil {
		t.Fatal(err)
	}
}

func TestResetRefusesActiveIdentity(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	g := newTestGuard(t, clock)
	_, lease, err := g.Admit("worker")
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Reset("worker"); err == nil {
		t.Fatal("Reset succeeded while active")
	}
	if err := lease.Finish(true); err != nil {
		t.Fatal(err)
	}
}

func stringContains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
