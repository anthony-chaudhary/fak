package l3server

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/config"
)

func TestL3ServerLifecycle(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 2
	cfg.MaxMemoryGB = 1

	srv, err := NewServer(&cfg)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}

	if srv.Status() != StatusStopped {
		t.Fatalf("expected StatusStopped, got %v", srv.Status())
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Start(ctx); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	if srv.Status() != StatusRunning {
		t.Fatalf("expected StatusRunning, got %v", srv.Status())
	}

	time.Sleep(5 * time.Millisecond)
	if srv.Uptime() <= 0 {
		t.Fatalf("expected positive uptime, got %v", srv.Uptime())
	}

	if srv.ShardManager() == nil {
		t.Fatal("expected non-nil ShardManager")
	}

	if srv.MetricsCollector() == nil {
		t.Fatal("expected non-nil MetricsCollector")
	}

	if srv.Version() == "" {
		t.Fatal("expected non-empty version string")
	}

	if err := srv.Stop(ctx); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	if srv.Status() != StatusStopped {
		t.Fatalf("expected StatusStopped after stop, got %v", srv.Status())
	}
}

func TestL3ServerDefaultConfig(t *testing.T) {
	srv, err := NewServer(nil)
	if err != nil {
		t.Fatalf("NewServer with nil config failed: %v", err)
	}
	if srv == nil {
		t.Fatal("expected non-nil server")
	}
}

// TestL3ServerStartStopIdempotent is the fak#13518 lifecycle witness: repeated
// Start/Stop cycles must not panic with "close of closed channel". Before the
// fix the second Stop re-closed each shard's single-shot quit/done channels.
func TestL3ServerStartStopIdempotent(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 2
	cfg.MaxMemoryGB = 1

	srv, err := NewServer(&cfg)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}

	ctx := context.Background()
	for cycle := 0; cycle < 3; cycle++ {
		if err := srv.Start(ctx); err != nil {
			t.Fatalf("cycle %d: Start failed: %v", cycle, err)
		}
		if srv.Status() != StatusRunning {
			t.Fatalf("cycle %d: expected StatusRunning, got %v", cycle, srv.Status())
		}
		if err := srv.Stop(ctx); err != nil {
			t.Fatalf("cycle %d: Stop failed: %v", cycle, err)
		}
		if srv.Status() != StatusStopped {
			t.Fatalf("cycle %d: expected StatusStopped, got %v", cycle, srv.Status())
		}
	}

	// A redundant Stop on an already-stopped server is a no-op.
	if err := srv.Stop(ctx); err != nil {
		t.Fatalf("redundant Stop failed: %v", err)
	}
}

// TestL3ServerRepeatedStartStop pins #13518: repeated Stop and a
// Start-Stop-Start-Stop cycle must not panic with "close of closed channel",
// and a restarted server must serve from a freshly provisioned shard manager.
func TestL3ServerRepeatedStartStop(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 2
	cfg.MaxMemoryGB = 1

	srv, err := NewServer(&cfg)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	step := func(name string, fn func() error) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("%s panicked: %v", name, r)
			}
		}()
		if err := fn(); err != nil {
			t.Fatalf("%s failed: %v", name, err)
		}
	}
	start := func() error { return srv.Start(ctx) }
	stop := func() error { return srv.Stop(ctx) }

	for cycle := 0; cycle < 3; cycle++ {
		step("Start", start)
		if srv.Status() != StatusRunning {
			t.Fatalf("cycle %d: expected StatusRunning, got %v", cycle, srv.Status())
		}
		key := []byte(fmt.Sprintf("restart-%d", cycle))
		srv.ShardManager().Set(key, []byte("v"), 0)
		if _, ok := srv.ShardManager().Get(key); !ok {
			t.Fatalf("cycle %d: Get after Start missed", cycle)
		}
		step("Stop", stop)
		step("second Stop", stop)
		if srv.Status() != StatusStopped {
			t.Fatalf("cycle %d: expected StatusStopped, got %v", cycle, srv.Status())
		}
	}
}
