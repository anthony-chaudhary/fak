package projectionspine

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/supervisionpolicy"
)

func setupBenchAuthority(tb testing.TB) (string, func()) {
	tb.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatalf("net.Listen failed: %v", err)
	}

	authority, err := NewAuthority(1001, "bench-session", 1, "transcript:start")
	if err != nil {
		_ = listener.Close()
		tb.Fatalf("NewAuthority failed: %v", err)
	}

	serveDone := make(chan struct{})
	go func() {
		_ = authority.Serve(listener)
		close(serveDone)
	}()

	cleanup := func() {
		_ = listener.Close()
		<-serveDone
	}
	return listener.Addr().String(), cleanup
}

func BenchmarkProjectionSpine(b *testing.B) {
	addr, cleanup := setupBenchAuthority(b)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	budget := supervisionpolicy.Budget{MaxRestarts: 5, Window: time.Minute}
	now := time.Unix(1_800_000_000, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proj, err := Attach(ctx, addr)
		if err != nil {
			b.Fatalf("Attach failed: %v", err)
		}

		decision := DecideProjectionFailure(proj.Snapshot, "bench-view", 1, nil, now, budget)
		if decision.Action != supervisionpolicy.ActionReattach {
			_ = proj.Close()
			b.Fatalf("unexpected decision action: %v", decision.Action)
		}

		if err := proj.Close(); err != nil {
			b.Fatalf("Close failed: %v", err)
		}

		repl, err := Attach(ctx, addr)
		if err != nil {
			b.Fatalf("replacement Attach failed: %v", err)
		}
		if err := repl.Close(); err != nil {
			b.Fatalf("replacement Close failed: %v", err)
		}
	}
}

func TestBenchmarkProjectionSpineSanity(t *testing.T) {
	addr, cleanup := setupBenchAuthority(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	proj, err := Attach(ctx, addr)
	if err != nil {
		t.Fatalf("Attach failed: %v", err)
	}
	if proj.Snapshot.SessionID != "bench-session" {
		t.Fatalf("unexpected session ID: %s", proj.Snapshot.SessionID)
	}
	if err := proj.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	repl, err := Attach(ctx, addr)
	if err != nil {
		t.Fatalf("replacement Attach failed: %v", err)
	}
	if repl.Snapshot.SessionID != "bench-session" {
		t.Fatalf("unexpected replacement session ID: %s", repl.Snapshot.SessionID)
	}
	if err := repl.Close(); err != nil {
		t.Fatalf("replacement Close failed: %v", err)
	}
}
