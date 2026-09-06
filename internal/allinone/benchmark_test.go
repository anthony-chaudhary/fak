package allinone

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
)

var (
	sinkTopology      *TopologySpec
	sinkHealthResp    HealthResponse
	sinkBool          bool
	sinkChildProcMap  map[string]ChildProcessInfo
	sinkChildProcInfo ChildProcessInfo
	sinkMemoryEntries []MemoryEntry
	sinkBytes         []byte
)

func createBenchmarkLockFile(b testing.TB, numComponents int) string {
	b.Helper()
	dir := b.TempDir()
	lockPath := filepath.Join(dir, "harness.lock.json")

	comps := make([]map[string]any, numComponents)
	for i := 0; i < numComponents; i++ {
		comps[i] = map[string]any{
			"id":       fmt.Sprintf("bench-mcp-%03d", i),
			"version":  "1.0.0",
			"digest":   fmt.Sprintf("sha256:%064x", i+1),
			"source":   fmt.Sprintf("bin/service-%d", i),
			"provider": "mcp",
			"provides": []string{"query", "mutate"},
		}
	}

	doc := map[string]any{
		"schema": "fak.harness-product-lock/v2",
		"id":     "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"platforms": []map[string]string{
			{"os": runtime.GOOS, "arch": runtime.GOARCH},
		},
		"budget": map[string]int{
			"context_tokens": 8192,
			"memory_mib":     1024,
			"workers":        4,
		},
		"components": comps,
		"assets": []map[string]any{
			{
				"kind":  "memory",
				"id":    "bench-journal",
				"value": "in-memory",
			},
		},
	}

	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		b.Fatalf("marshal benchmark lock: %v", err)
	}
	if err := os.WriteFile(lockPath, data, 0600); err != nil {
		b.Fatalf("write benchmark lock file: %v", err)
	}
	return lockPath
}

// BenchmarkSupervisor_DryRunTopology measures deployment topology resolution and lock validation.
func BenchmarkSupervisor_DryRunTopology(b *testing.B) {
	b.Run("SingleComponent", func(b *testing.B) {
		lockPath := createBenchmarkLockFile(b, 1)
		sup, err := NewSupervisor(Config{
			LockPath: lockPath,
			Addr:     "127.0.0.1:0",
			Engine:   "mock",
		})
		if err != nil {
			b.Fatalf("NewSupervisor: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkTopology, err = sup.DryRunTopology()
			if err != nil {
				b.Fatalf("DryRunTopology: %v", err)
			}
		}
	})

	b.Run("MultiComponent_N=10", func(b *testing.B) {
		lockPath := createBenchmarkLockFile(b, 10)
		sup, err := NewSupervisor(Config{
			LockPath: lockPath,
			Addr:     "127.0.0.1:0",
			Engine:   "mock",
		})
		if err != nil {
			b.Fatalf("NewSupervisor: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkTopology, err = sup.DryRunTopology()
			if err != nil {
				b.Fatalf("DryRunTopology: %v", err)
			}
		}
	})
}

// BenchmarkSupervisor_BootstrapLifecycle measures end-to-end supervisor startup, address binding, and shutdown.
func BenchmarkSupervisor_BootstrapLifecycle(b *testing.B) {
	lockPath := createBenchmarkLockFile(b, 2)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sup, err := NewSupervisor(Config{
			LockPath: lockPath,
			Addr:     "127.0.0.1:0",
			Engine:   "mock",
		})
		if err != nil {
			b.Fatalf("NewSupervisor: %v", err)
		}
		if err := sup.Start(ctx); err != nil {
			b.Fatalf("Start: %v", err)
		}
		if sup.Addr() == "" {
			b.Fatal("empty bound address")
		}
		if !sup.IsHealthy() {
			b.Fatal("supervisor not healthy")
		}
		if err := sup.Shutdown(ctx); err != nil {
			b.Fatalf("Shutdown: %v", err)
		}
	}
}

// BenchmarkSupervisor_SessionDispatch measures agent wire session execution throughput.
func BenchmarkSupervisor_SessionDispatch(b *testing.B) {
	lockPath := createBenchmarkLockFile(b, 1)
	ctx := context.Background()

	sup, err := NewSupervisor(Config{
		LockPath: lockPath,
		Addr:     "127.0.0.1:0",
		Engine:   "mock",
	})
	if err != nil {
		b.Fatalf("NewSupervisor: %v", err)
	}
	if err := sup.Start(ctx); err != nil {
		b.Fatalf("Start: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(ctx)
	}()

	b.Run("DirectHandler_DefaultTool", func(b *testing.B) {
		reqBody := []byte(`{"goal":"benchmark session dispatch","max_turns":1}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest(http.MethodPost, "/v1/fak/agent/sessions", bytes.NewReader(reqBody))
			rec := httptest.NewRecorder()
			sup.handleAgentWire(rec, req)
			if rec.Code != http.StatusOK {
				b.Fatalf("status = %d, want 200", rec.Code)
			}
			sinkBytes = rec.Body.Bytes()
		}
	})

	b.Run("DirectHandler_ExplicitTool", func(b *testing.B) {
		reqBody := []byte(`{"goal":"benchmark explicit tool","tool":"mcp__bench-mcp-000__query","args":{"key":"val"}}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest(http.MethodPost, "/v1/fak/agent/sessions", bytes.NewReader(reqBody))
			rec := httptest.NewRecorder()
			sup.handleAgentWire(rec, req)
			if rec.Code != http.StatusOK {
				b.Fatalf("status = %d, want 200", rec.Code)
			}
			sinkBytes = rec.Body.Bytes()
		}
	})

	b.Run("NetworkWire_POST", func(b *testing.B) {
		client := &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     30 * time.Second,
			},
		}
		url := "http://" + sup.Addr() + "/v1/fak/agent/sessions"
		reqBody := `{"goal":"network wire test","max_turns":1}`

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			resp, err := client.Post(url, "application/json", strings.NewReader(reqBody))
			if err != nil {
				b.Fatalf("POST session: %v", err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				b.Fatalf("status = %d, want 200", resp.StatusCode)
			}
		}
	})
}

// BenchmarkSupervisor_MemoryJournal measures memory entry recording and persistence.
func BenchmarkSupervisor_MemoryJournal(b *testing.B) {
	b.Run("Append_InMemory", func(b *testing.B) {
		j, err := newMemoryJournal("in-memory")
		if err != nil {
			b.Fatalf("newMemoryJournal: %v", err)
		}
		defer j.Close()
		entry := MemoryEntry{
			Timestamp: time.Now(),
			Type:      "session.start",
			SessionID: "sess-bench-1",
			Goal:      "measure append throughput",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := j.Append(entry); err != nil {
				b.Fatalf("Append: %v", err)
			}
		}
	})

	b.Run("Append_Durable", func(b *testing.B) {
		file := filepath.Join(b.TempDir(), "journal.jsonl")
		j, err := newMemoryJournal(file)
		if err != nil {
			b.Fatalf("newMemoryJournal: %v", err)
		}
		defer j.Close()
		entry := MemoryEntry{
			Timestamp: time.Now(),
			Type:      "call",
			SessionID: "sess-bench-2",
			Tool:      "mcp__echo",
			Data:      json.RawMessage(`{"status":"ok"}`),
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := j.Append(entry); err != nil {
				b.Fatalf("Append: %v", err)
			}
		}
	})

	b.Run("Flush_Durable", func(b *testing.B) {
		file := filepath.Join(b.TempDir(), "journal-flush.jsonl")
		j, err := newMemoryJournal(file)
		if err != nil {
			b.Fatalf("newMemoryJournal: %v", err)
		}
		defer j.Close()
		entry := MemoryEntry{
			Timestamp: time.Now(),
			Type:      "step",
			SessionID: "sess-bench-3",
		}
		_ = j.Append(entry)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := j.Flush(); err != nil {
				b.Fatalf("Flush: %v", err)
			}
		}
	})

	b.Run("EntriesSnapshot", func(b *testing.B) {
		j, err := newMemoryJournal("in-memory")
		if err != nil {
			b.Fatalf("newMemoryJournal: %v", err)
		}
		defer j.Close()
		for k := 0; k < 100; k++ {
			_ = j.Append(MemoryEntry{
				Timestamp: time.Now(),
				Type:      "call",
				SessionID: fmt.Sprintf("sess-%d", k),
			})
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkMemoryEntries = j.Entries()
		}
	})
}

// BenchmarkSupervisor_TrackChildProcesses measures aggregating runtime state across managed child processes.
func BenchmarkSupervisor_TrackChildProcesses(b *testing.B) {
	for _, count := range []int{1, 5, 20} {
		b.Run(fmt.Sprintf("N=%d", count), func(b *testing.B) {
			lockPath := createBenchmarkLockFile(b, count)
			ctx := context.Background()
			sup, err := NewSupervisor(Config{
				LockPath: lockPath,
				Addr:     "127.0.0.1:0",
				Engine:   "mock",
			})
			if err != nil {
				b.Fatalf("NewSupervisor: %v", err)
			}
			if err := sup.Start(ctx); err != nil {
				b.Fatalf("Start: %v", err)
			}
			defer func() {
				_ = sup.Shutdown(ctx)
			}()

			// Register mock process supervisors for each component in broker
			for i := 0; i < count; i++ {
				id := fmt.Sprintf("bench-mcp-%03d", i)
				procSup := mcpbroker.NewProcessSupervisor(mcpbroker.ServerConfig{
					ID:   id,
					Name: id,
				}, sup.Broker())
				_ = sup.Broker().RegisterSupervisor(procSup)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkChildProcMap = sup.TrackChildProcesses()
			}
		})
	}
}

// BenchmarkSupervisor_ChildProcessStatus measures point lookup of a single child process by ID.
func BenchmarkSupervisor_ChildProcessStatus(b *testing.B) {
	lockPath := createBenchmarkLockFile(b, 5)
	ctx := context.Background()
	sup, err := NewSupervisor(Config{
		LockPath: lockPath,
		Addr:     "127.0.0.1:0",
		Engine:   "mock",
	})
	if err != nil {
		b.Fatalf("NewSupervisor: %v", err)
	}
	if err := sup.Start(ctx); err != nil {
		b.Fatalf("Start: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(ctx)
	}()

	targetID := "bench-mcp-002"
	procSup := mcpbroker.NewProcessSupervisor(mcpbroker.ServerConfig{
		ID:   targetID,
		Name: targetID,
	}, sup.Broker())
	_ = sup.Broker().RegisterSupervisor(procSup)

	b.Run("ExistingProcess", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkChildProcInfo, sinkBool = sup.ChildProcessStatus(targetID)
		}
	})

	b.Run("MissingProcess", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkChildProcInfo, sinkBool = sup.ChildProcessStatus("non-existent-id")
		}
	})
}

// BenchmarkSupervisor_ChildProcessTracking_Subprocess measures tracking a live OS subprocess.
func BenchmarkSupervisor_ChildProcessTracking_Subprocess(b *testing.B) {
	if os.Getenv(testHelperEnv) == "1" && os.Getenv("MOCK_SERVER_ID") == testServerID {
		runStructuredMCPHelper()
		os.Exit(0)
	}

	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}

	dir := b.TempDir()
	lockPath := filepath.Join(dir, "harness.lock.json")
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "platforms": [
    {"os": ` + jsonQuote(runtime.GOOS) + `, "arch": ` + jsonQuote(runtime.GOARCH) + `}
  ],
  "components": [
    {
      "id": ` + jsonQuote(testServerID) + `,
      "version": "1.0.0",
      "digest": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
      "source": ` + jsonQuote(exe) + `,
      "provider": "mcp",
      "provides": ["query_structured"],
      "adapters": ["-test.run=TestStructuredMCPHelper"]
    }
  ]
}`
	if err := os.WriteFile(lockPath, []byte(lockContent), 0600); err != nil {
		b.Fatalf("write lock file: %v", err)
	}

	sup, err := NewSupervisor(Config{
		LockPath: lockPath,
		Addr:     "127.0.0.1:0",
		Mock:     true,
	})
	if err != nil {
		b.Fatalf("NewSupervisor: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		b.Fatalf("Start: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	b.Run("TrackRunningSubprocess", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkChildProcInfo, sinkBool = sup.ChildProcessStatus(testServerID)
		}
	})

	b.Run("TrackAllIncludingSubprocess", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkChildProcMap = sup.TrackChildProcesses()
		}
	})
}

// BenchmarkHealth_Snapshot measures HealthAggregator snapshotting across varying subsystem counts.
func BenchmarkHealth_Snapshot(b *testing.B) {
	b.Run("MandatorySubsystems_N=4", func(b *testing.B) {
		agg := NewHealthAggregator()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkHealthResp = agg.Snapshot()
		}
	})

	b.Run("Scale_N=16", func(b *testing.B) {
		agg := NewHealthAggregator()
		for i := 0; i < 12; i++ {
			agg.SetStatus(fmt.Sprintf("subsystem-%d", i), true, "")
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkHealthResp = agg.Snapshot()
		}
	})

	b.Run("Scale_N=64", func(b *testing.B) {
		agg := NewHealthAggregator()
		for i := 0; i < 60; i++ {
			agg.SetStatus(fmt.Sprintf("subsystem-%d", i), true, "")
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkHealthResp = agg.Snapshot()
		}
	})
}

// BenchmarkHealth_IsHealthy measures zero-allocation health predicate evaluation.
func BenchmarkHealth_IsHealthy(b *testing.B) {
	agg := NewHealthAggregator()
	b.Run("AllHealthy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool = agg.IsHealthy()
		}
	})

	b.Run("Degraded", func(b *testing.B) {
		agg.SetStatus(SubsystemMCPBroker, false, "unhealthy")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool = agg.IsHealthy()
		}
	})
}

// BenchmarkHealth_SetStatus measures subsystem health status mutation.
func BenchmarkHealth_SetStatus(b *testing.B) {
	agg := NewHealthAggregator()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.SetStatus(SubsystemMCPBroker, i%2 == 0, "simulated error")
	}
}

// BenchmarkHealth_HTTPHandler measures /healthz HTTP handler execution and JSON response encoding.
func BenchmarkHealth_HTTPHandler(b *testing.B) {
	b.Run("Healthy_200", func(b *testing.B) {
		agg := NewHealthAggregator()
		handler := agg.Handler()
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				b.Fatalf("status = %d, want 200", rec.Code)
			}
			sinkBytes = rec.Body.Bytes()
		}
	})

	b.Run("Degraded_503", func(b *testing.B) {
		agg := NewHealthAggregator()
		agg.SetStatus(SubsystemInference, false, "engine connection timeout")
		handler := agg.Handler()
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusServiceUnavailable {
				b.Fatalf("status = %d, want 503", rec.Code)
			}
			sinkBytes = rec.Body.Bytes()
		}
	})
}

// BenchmarkHealth_ConcurrentEvaluation measures concurrent health checks under concurrent mutations.
func BenchmarkHealth_ConcurrentEvaluation(b *testing.B) {
	agg := NewHealthAggregator()
	var stop int32
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		step := 0
		for atomic.LoadInt32(&stop) == 0 {
			agg.SetStatus(SubsystemMemoryStore, step%2 == 0, "")
			step++
			runtime.Gosched()
		}
	}()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			sinkHealthResp = agg.Snapshot()
			sinkBool = agg.IsHealthy()
		}
	})

	atomic.StoreInt32(&stop, 1)
	wg.Wait()
}

// TestBenchmarkOperationsSanity ensures all benchmark operations execute cleanly without panics.
func TestBenchmarkOperationsSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	if bFlag := flag.Lookup("test.bench"); bFlag != nil && bFlag.Value.String() != "" {
		t.Skip("skipping benchmark sanity: benchmarks are executed directly via -bench flag")
	}
	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkSupervisor_DryRunTopology", BenchmarkSupervisor_DryRunTopology},
		{"BenchmarkSupervisor_BootstrapLifecycle", BenchmarkSupervisor_BootstrapLifecycle},
		{"BenchmarkSupervisor_SessionDispatch", BenchmarkSupervisor_SessionDispatch},
		{"BenchmarkSupervisor_MemoryJournal", BenchmarkSupervisor_MemoryJournal},
		{"BenchmarkSupervisor_TrackChildProcesses", BenchmarkSupervisor_TrackChildProcesses},
		{"BenchmarkSupervisor_ChildProcessStatus", BenchmarkSupervisor_ChildProcessStatus},
		{"BenchmarkSupervisor_ChildProcessTracking_Subprocess", BenchmarkSupervisor_ChildProcessTracking_Subprocess},
		{"BenchmarkHealth_Snapshot", BenchmarkHealth_Snapshot},
		{"BenchmarkHealth_IsHealthy", BenchmarkHealth_IsHealthy},
		{"BenchmarkHealth_SetStatus", BenchmarkHealth_SetStatus},
		{"BenchmarkHealth_HTTPHandler", BenchmarkHealth_HTTPHandler},
		{"BenchmarkHealth_ConcurrentEvaluation", BenchmarkHealth_ConcurrentEvaluation},
	}

	for _, tc := range benchmarks {
		t.Run(tc.name, func(t *testing.T) {
			res := testing.Benchmark(tc.fn)
			if res.N <= 0 {
				t.Fatalf("%s failed to execute any iterations: %+v", tc.name, res)
			}
		})
	}
}
