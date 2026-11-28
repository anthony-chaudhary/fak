// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunIndex(t *testing.T) {
	ctx := context.Background()

	t.Run("upsert compressed payloads, narrow column query, and payload roundtrip", func(t *testing.T) {
		tmp := t.TempDir()
		indexPath := filepath.Join(tmp, "runs_index.json.gz")
		store, err := NewRunIndexStore(RunIndexOptions{
			IndexPath:  indexPath,
			ResultsDir: filepath.Join(tmp, "results"),
		})
		if err != nil {
			t.Fatalf("NewRunIndexStore: %v", err)
		}

		p1 := map[string]any{"detail": "run-alpha-metrics", "latency_samples": []float64{10, 12, 14}}
		p2 := map[string]any{"detail": "run-beta-metrics", "latency_samples": []float64{20, 22, 24}}
		if err := store.Upsert(ctx, RunIndexRow{
			Namespace:     "prod",
			RunID:         "run-1",
			Epoch:         1,
			Model:         "qwen3",
			Status:        "Succeeded",
			CreatedUnixMS: 1000,
			TotalRequests: 200,
			ErrorRate:     0.01,
			ThroughputTPS: 180.5,
			TTFTP99MS:     45.0,
		}, p1); err != nil {
			t.Fatalf("Upsert 1: %v", err)
		}
		if err := store.Upsert(ctx, RunIndexRow{
			Namespace:     "prod",
			RunID:         "run-2",
			Epoch:         2,
			Model:         "qwen3",
			Status:        "Succeeded",
			CreatedUnixMS: 2000,
			TotalRequests: 400,
			ErrorRate:     0.0,
			ThroughputTPS: 310.0,
			TTFTP99MS:     38.5,
		}, p2); err != nil {
			t.Fatalf("Upsert 2: %v", err)
		}
		if err := store.Upsert(ctx, RunIndexRow{
			Namespace:     "staging",
			RunID:         "run-3",
			Epoch:         1,
			Model:         "llama",
			Status:        "Failed",
			CreatedUnixMS: 1500,
			ThroughputTPS: 90.0,
		}, map[string]any{"detail": "failed"}); err != nil {
			t.Fatalf("Upsert 3: %v", err)
		}

		rows, src, err := store.List(ctx, RunIndexFilter{
			Namespace:  "prod",
			Model:      "qwen3",
			SortBy:     "throughput_tps",
			Descending: true,
		})
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		if src != IndexSourceFile {
			t.Fatalf("source = %q, want %q", src, IndexSourceFile)
		}
		if len(rows) != 2 || rows[0].RunID != "run-2" || rows[1].RunID != "run-1" {
			t.Fatalf("unexpected sorted rows: %+v", rows)
		}

		var decoded map[string]any
		if err := DecompressRunPayload(rows[0], &decoded); err != nil {
			t.Fatalf("DecompressRunPayload: %v", err)
		}
		if decoded["detail"] != "run-beta-metrics" {
			t.Fatalf("decoded detail = %v, want run-beta-metrics", decoded["detail"])
		}
	})

	t.Run("missing or corrupt index falls back fail-safe to disk scan and skips processing dirs", func(t *testing.T) {
		tmp := t.TempDir()
		indexPath := filepath.Join(tmp, "runs_index.json.gz")
		resultsDir := filepath.Join(tmp, "results")

		readyDir := filepath.Join(resultsDir, "prod", "run-ready")
		if err := os.MkdirAll(readyDir, 0o755); err != nil {
			t.Fatal(err)
		}
		rawReady, _ := json.Marshal(map[string]any{
			"namespace":      "prod",
			"run_id":         "run-ready",
			"epoch":          5,
			"model":          "qwen3",
			"status":         "Succeeded",
			"total_requests": 100,
			"throughput_tps": 250.25,
			"ttft_p99_ms":    41.0,
			"payload":        map[string]any{"ok": true},
		})
		if err := os.WriteFile(filepath.Join(readyDir, "summary.json"), rawReady, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := WriteReadyMarker(readyDir, ReadyMarker{}); err != nil {
			t.Fatal(err)
		}

		// Mid-export directory with processing marker (and no ready marker) must be skipped.
		inflightDir := filepath.Join(resultsDir, "prod", "run-inflight")
		if err := WriteProcessingMarker(inflightDir); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(inflightDir, "summary.json"), []byte(`{"namespace":"prod","run_id":"run-inflight","epoch":6}`), 0o644); err != nil {
			t.Fatal(err)
		}

		// Corrupt JSON file in another directory must not abort the scan.
		corruptDir := filepath.Join(resultsDir, "prod", "run-badjson")
		if err := os.MkdirAll(corruptDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(corruptDir, "summary.json"), []byte("{not-json"), 0o644); err != nil {
			t.Fatal(err)
		}

		store, err := NewRunIndexStore(RunIndexOptions{
			IndexPath:  indexPath,
			ResultsDir: resultsDir,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Case 1: absent index file.
		rows, src, err := store.List(ctx, RunIndexFilter{})
		if err != nil {
			t.Fatalf("List with missing index: %v", err)
		}
		if src != IndexSourceDiskFallback {
			t.Fatalf("source = %q, want %q", src, IndexSourceDiskFallback)
		}
		if len(rows) != 1 || rows[0].RunID != "run-ready" {
			t.Fatalf("fallback rows = %+v, want only run-ready", rows)
		}

		// Case 2: corrupt index file (not valid gzip).
		if err := os.WriteFile(indexPath, []byte("corrupt-not-gzip-bytes"), 0o644); err != nil {
			t.Fatal(err)
		}
		rows, src, err = store.List(ctx, RunIndexFilter{})
		if err != nil {
			t.Fatalf("List with corrupt index: %v", err)
		}
		if src != IndexSourceDiskFallback || len(rows) != 1 || rows[0].RunID != "run-ready" {
			t.Fatalf("corrupt fallback = (%+v, %q), want 1 ready row from disk_fallback", rows, src)
		}

		// Rebuild restores the healthy compressed index file.
		rebuilt, err := store.RebuildFromDisk(ctx)
		if err != nil || rebuilt != 1 {
			t.Fatalf("RebuildFromDisk = (%d, %v), want (1, nil)", rebuilt, err)
		}
		rows, src, err = store.List(ctx, RunIndexFilter{})
		if err != nil || src != IndexSourceFile || len(rows) != 1 {
			t.Fatalf("post-rebuild List = (%+v, %q, %v)", rows, src, err)
		}
	})
}
