package gateway

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func benchmarkFakReadServer(b *testing.B, root string, cache bool) *Server {
	b.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})
	agent.RegisterReadEngine(root)
	if cache {
		v := vdso.New(vdso.DefaultCacheSize)
		v.SetGranularity(vdso.Resource)
		abi.RegisterFastPath(1, v)
		abi.RegisterEmitter(v)
	}
	srv, err := New(Config{EngineID: "fakread", Model: "m", VDSO: cache})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(srv.Close)
	return srv
}

func BenchmarkFakRead(b *testing.B) {
	root := b.TempDir()
	path := filepath.Join(root, "fixture.bin")
	data := make([]byte, 64<<10)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(data)))

	b.Run("direct_os_readfile", func(b *testing.B) {
		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := os.ReadFile(path); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("cold_guarded_read", func(b *testing.B) {
		srv := benchmarkFakReadServer(b, root, false)
		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, env, err := srv.fakRead(context.Background(), path, "bench-cold", ""); err != nil || env == nil || env.Status != "OK" {
				b.Fatalf("read failed: err=%v env=%+v", err, env)
			}
		}
	})
	b.Run("verified_fresh_reuse", func(b *testing.B) {
		srv := benchmarkFakReadServer(b, root, true)
		if _, env, err := srv.fakRead(context.Background(), path, "bench-hit", ""); err != nil || env == nil || env.Status != "OK" {
			b.Fatalf("prime failed: err=%v env=%+v", err, env)
		}
		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, env, err := srv.fakRead(context.Background(), path, "bench-hit", ""); err != nil || env == nil || env.Status != "OK" {
				b.Fatalf("read failed: err=%v env=%+v", err, env)
			}
		}
	})
}
