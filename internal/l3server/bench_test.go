package l3server

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/config"
)

func TestBenchmarkRegistration(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 1
	cfg.MaxMemoryGB = 1
	srv, err := NewServer(&cfg)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	if srv.Version() == "" {
		t.Fatal("expected non-empty version")
	}
}

func BenchmarkL3ServerInitialization(b *testing.B) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 2
	cfg.MaxMemoryGB = 1

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv, err := NewServer(&cfg)
		if err != nil {
			b.Fatalf("NewServer failed: %v", err)
		}
		_ = srv.Version()
	}
}

func BenchmarkL3ServerStartStop(b *testing.B) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 2
	cfg.MaxMemoryGB = 1

	srv, err := NewServer(&cfg)
	if err != nil {
		b.Fatalf("NewServer failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := srv.Start(ctx); err != nil {
			cancel()
			b.Fatalf("Start failed: %v", err)
		}
		if err := srv.Stop(ctx); err != nil {
			cancel()
			b.Fatalf("Stop failed: %v", err)
		}
		cancel()
	}
}

func BenchmarkL3ServerStatus(b *testing.B) {
	cfg := config.DefaultConfig()
	cfg.NumShards = 1
	cfg.MaxMemoryGB = 1

	srv, err := NewServer(&cfg)
	if err != nil {
		b.Fatalf("NewServer failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = srv.Status()
		_ = srv.Uptime()
	}
}
