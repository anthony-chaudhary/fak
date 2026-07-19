package dispatchtick

import (
	"errors"
	"testing"
	"time"
)

type fakeWarmShell struct {
	id      int
	healthy bool
	closed  bool
}

func (s *fakeWarmShell) Healthy() bool { return s.healthy }
func (s *fakeWarmShell) Close() error  { s.closed = true; return nil }

type fakeShellSpawner struct {
	calls  int
	shells []*fakeWarmShell
	err    error
}

func (f *fakeShellSpawner) spawn(key string) (WarmShell, error) {
	if f.err != nil {
		return nil, f.err
	}
	f.calls++
	s := &fakeWarmShell{id: f.calls, healthy: true}
	f.shells = append(f.shells, s)
	return s, nil
}

func newTestShellRack(t *testing.T, capacity int, maxIdle time.Duration) (*ShellRack, *fakeShellSpawner) {
	t.Helper()
	sp := &fakeShellSpawner{}
	p, err := NewShellRack(capacity, maxIdle, sp.spawn)
	if err != nil {
		t.Fatalf("NewShellRack: %v", err)
	}
	return p, sp
}

func TestShellRackCheckoutReusesWarmShell(t *testing.T) {
	p, sp := newTestShellRack(t, 2, time.Minute)

	first, err := p.Checkout("claude/lane-a")
	if err != nil {
		t.Fatalf("first checkout: %v", err)
	}
	if first.Warm || !first.Racked {
		t.Fatalf("first checkout should be a racked cold spawn, got warm=%v racked=%v", first.Warm, first.Racked)
	}
	p.Return(first)
	if got := p.IdleCount("claude/lane-a"); got != 1 {
		t.Fatalf("idle count after return = %d, want 1", got)
	}

	second, err := p.Checkout("claude/lane-a")
	if err != nil {
		t.Fatalf("second checkout: %v", err)
	}
	if !second.Warm {
		t.Fatalf("second checkout should reuse the warm shell")
	}
	if second.Shell != first.Shell {
		t.Fatalf("warm reuse handed back a different shell")
	}
	if sp.calls != 1 {
		t.Fatalf("spawner ran %d times, want 1 (reuse must not spawn)", sp.calls)
	}
	st := p.Stats()
	if st.WarmReuses != 1 || st.ColdSpawns != 1 {
		t.Fatalf("stats = %+v, want WarmReuses=1 ColdSpawns=1", st)
	}
}

func TestShellRackRetiresUnhealthyShellAndReplaces(t *testing.T) {
	p, sp := newTestShellRack(t, 2, time.Minute)

	lease, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	p.Return(lease)
	sp.shells[0].healthy = false // shell dies while idle

	replacement, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout after unhealthy: %v", err)
	}
	if replacement.Warm {
		t.Fatalf("unhealthy shell must not be reused")
	}
	if replacement.Shell == lease.Shell {
		t.Fatalf("unhealthy shell was handed back")
	}
	if !sp.shells[0].closed {
		t.Fatalf("unhealthy shell was not closed on retirement")
	}
	if sp.calls != 2 {
		t.Fatalf("spawner ran %d times, want 2 (retire + replace)", sp.calls)
	}
	if st := p.Stats(); st.UnhealthyRetired != 1 {
		t.Fatalf("stats = %+v, want UnhealthyRetired=1", st)
	}
	if got := p.LiveCount(); got != 1 {
		t.Fatalf("live count = %d, want 1 (retired slot freed)", got)
	}
}

func TestShellRackReturnClosesUnhealthyShell(t *testing.T) {
	p, sp := newTestShellRack(t, 2, time.Minute)

	lease, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	sp.shells[0].healthy = false // shell dies while checked out
	p.Return(lease)
	if p.IdleCount("k") != 0 {
		t.Fatalf("unhealthy shell was retained on return")
	}
	if !sp.shells[0].closed {
		t.Fatalf("unhealthy shell was not closed on return")
	}
	if got := p.LiveCount(); got != 0 {
		t.Fatalf("live count = %d, want 0", got)
	}
}

func TestShellRackFullRackForcesOverflowColdSpawn(t *testing.T) {
	p, sp := newTestShellRack(t, 1, time.Minute)

	held, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	overflow, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("overflow checkout: %v", err)
	}
	if overflow.Racked {
		t.Fatalf("checkout on a full rack must hand back an unracked overflow shell")
	}
	if sp.calls != 2 {
		t.Fatalf("spawner ran %d times, want 2 (full rack forces a cold spawn)", sp.calls)
	}
	if st := p.Stats(); st.OverflowSpawns != 1 {
		t.Fatalf("stats = %+v, want OverflowSpawns=1", st)
	}

	p.Return(overflow) // overflow shell is closed, never retained
	if !sp.shells[1].closed {
		t.Fatalf("overflow shell was not closed on return")
	}
	if p.IdleCount("k") != 0 {
		t.Fatalf("overflow shell was retained in the warm set")
	}
	if st := p.Stats(); st.OverflowClosed != 1 {
		t.Fatalf("stats = %+v, want OverflowClosed=1", st)
	}

	p.Return(held) // racked shell IS retained; the warm set stays bounded at cap
	if p.IdleCount("k") != 1 || p.LiveCount() != 1 {
		t.Fatalf("idle=%d live=%d after returns, want 1/1", p.IdleCount("k"), p.LiveCount())
	}
}

func TestShellRackMaxIdleRetirement(t *testing.T) {
	p, sp := newTestShellRack(t, 2, 30*time.Second)
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	now := base
	p.now = func() time.Time { return now }

	lease, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	p.Return(lease)

	now = base.Add(31 * time.Second) // idle past MaxIdle
	fresh, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout after expiry: %v", err)
	}
	if fresh.Warm {
		t.Fatalf("expired shell must not be reused")
	}
	if !sp.shells[0].closed {
		t.Fatalf("expired shell was not closed")
	}
	if st := p.Stats(); st.IdleRetired != 1 {
		t.Fatalf("stats = %+v, want IdleRetired=1", st)
	}
}

func TestShellRackPruneSweepsExpiredIdle(t *testing.T) {
	p, sp := newTestShellRack(t, 4, 30*time.Second)
	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	now := base
	p.now = func() time.Time { return now }

	a, _ := p.Checkout("a")
	b, _ := p.Checkout("b")
	p.Return(a)
	now = base.Add(20 * time.Second)
	p.Return(b) // b is 20s fresher than a

	now = base.Add(45 * time.Second) // a idle 45s (expired), b idle 25s (fresh)
	if got := p.Prune(); got != 1 {
		t.Fatalf("Prune retired %d, want 1", got)
	}
	if !sp.shells[0].closed || sp.shells[1].closed {
		t.Fatalf("Prune closed the wrong shell (a closed=%v b closed=%v)", sp.shells[0].closed, sp.shells[1].closed)
	}
	if p.IdleCount("a") != 0 || p.IdleCount("b") != 1 {
		t.Fatalf("idle counts a=%d b=%d, want 0/1", p.IdleCount("a"), p.IdleCount("b"))
	}
}

func TestShellRackKeyIsolation(t *testing.T) {
	p, sp := newTestShellRack(t, 4, time.Minute)

	a, err := p.Checkout("claude/lane-a")
	if err != nil {
		t.Fatalf("checkout a: %v", err)
	}
	p.Return(a)

	b, err := p.Checkout("codex/lane-b")
	if err != nil {
		t.Fatalf("checkout b: %v", err)
	}
	if b.Warm || b.Shell == a.Shell {
		t.Fatalf("checkout for a different key must not reuse another key's shell")
	}
	back, err := p.Checkout("claude/lane-a")
	if err != nil {
		t.Fatalf("checkout a again: %v", err)
	}
	if !back.Warm || back.Shell != a.Shell {
		t.Fatalf("same-key checkout should reuse the warm shell")
	}
	if sp.calls != 2 {
		t.Fatalf("spawner ran %d times, want 2", sp.calls)
	}
}

func TestShellRackSpawnErrorReleasesSlot(t *testing.T) {
	p, sp := newTestShellRack(t, 1, time.Minute)
	sp.err = errors.New("spawn refused")
	if _, err := p.Checkout("k"); err == nil {
		t.Fatalf("checkout should surface the spawn error")
	}
	if got := p.LiveCount(); got != 0 {
		t.Fatalf("live count = %d after failed spawn, want 0 (slot released)", got)
	}
	sp.err = nil
	lease, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout after recovery: %v", err)
	}
	if !lease.Racked {
		t.Fatalf("recovered checkout should occupy the racked slot")
	}
}

func TestShellRackCloseClosesIdleAndRefuses(t *testing.T) {
	p, sp := newTestShellRack(t, 2, time.Minute)
	lease, err := p.Checkout("k")
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	p.Return(lease)
	p.Close()
	if !sp.shells[0].closed {
		t.Fatalf("Close did not close the idle shell")
	}
	if _, err := p.Checkout("k"); err == nil {
		t.Fatalf("checkout on a closed rack must refuse")
	}
}

func TestShellRackRejectsBadConfig(t *testing.T) {
	if _, err := NewShellRack(0, time.Minute, func(string) (WarmShell, error) { return nil, nil }); err == nil {
		t.Fatalf("capacity 0 must be rejected")
	}
	if _, err := NewShellRack(1, time.Minute, nil); err == nil {
		t.Fatalf("nil spawn func must be rejected")
	}
}
