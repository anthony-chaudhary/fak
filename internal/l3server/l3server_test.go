package l3server

import (
	"context"
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
