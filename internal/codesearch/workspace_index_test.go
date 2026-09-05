package codesearch

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestSharedTrigramIndex(t *testing.T) {
	// 1. Prepare an index of 5,000 synthetic files.
	const numFiles = 5000
	docs := make(map[string]string, numFiles)
	for i := 0; i < numFiles; i++ {
		path := fmt.Sprintf("pkg/service%04d/handler.go", i)
		content := fmt.Sprintf("package service\n\n// Service handler %d\nfunc HandleRequest%d(ctx Context) Response {\n\ttoken := \"tok_%04d_val\"\n\treturn dispatch(token)\n}\n", i, i, i)
		docs[path] = content
	}

	ResetSharedIndices()
	const wsRoot = "virtual/shared/workspace"
	wix := SharedWorkspaceIndex(wsRoot, WithDocuments(docs))
	if err := wix.EnsureBuilt(); err != nil {
		t.Fatalf("failed to build shared index: %v", err)
	}
	if got := wix.DocCount(); got != numFiles {
		t.Fatalf("DocCount = %d, want %d", got, numFiles)
	}

	// Record initial file read count (0 for in-memory docs).
	initialFileReads := wix.FileReads()

	// 2. Launch 10 concurrent workers searching for tokens and regexes simultaneously.
	const numWorkers = 10
	const searchesPerWorker = 30
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	errCh := make(chan error, numWorkers*searchesPerWorker)
	workerLatencies := make([]time.Duration, numWorkers)

	startAll := time.Now()
	for wID := 0; wID < numWorkers; wID++ {
		go func(workerID int) {
			defer wg.Done()
			workerIndex := SharedWorkspaceIndex(wsRoot)
			if workerIndex != wix {
				errCh <- fmt.Errorf("worker %d: shared index instance mismatch", workerID)
				return
			}

			var searchTime time.Duration
			for s := 0; s < searchesPerWorker; s++ {
				targetID := (workerID*searchesPerWorker + s) % numFiles
				if s%2 == 0 {
					// Literal substring search
					query := fmt.Sprintf("tok_%04d_val", targetID)
					sStart := time.Now()
					hits := workerIndex.Search(query)
					searchTime += time.Since(sStart)
					if len(hits) != 1 {
						errCh <- fmt.Errorf("worker %d search %d: Search(%q) = %d hits, want 1", workerID, s, query, len(hits))
						return
					}
					expectedPath := fmt.Sprintf("pkg/service%04d/handler.go", targetID)
					if hits[0].Path != expectedPath {
						errCh <- fmt.Errorf("worker %d search %d: path = %q, want %q", workerID, s, hits[0].Path, expectedPath)
						return
					}
				} else {
					// Regex search
					pattern := fmt.Sprintf(`HandleRequest%d\(.*Context`, targetID)
					sStart := time.Now()
					hits, err := workerIndex.SearchRegexp(pattern)
					searchTime += time.Since(sStart)
					if err != nil {
						errCh <- fmt.Errorf("worker %d search %d: SearchRegexp(%q) err: %v", workerID, s, pattern, err)
						return
					}
					if len(hits) != 1 {
						errCh <- fmt.Errorf("worker %d search %d: SearchRegexp(%q) = %d hits, want 1", workerID, s, pattern, len(hits))
						return
					}
				}
			}
			workerLatencies[workerID] = searchTime
		}(wID)
	}

	wg.Wait()
	totalDuration := time.Since(startAll)
	close(errCh)

	for err := range errCh {
		t.Errorf("worker error: %v", err)
	}

	// 3. Verifies 0 redundant file reads (all served from shared in-memory index).
	redundantReads := wix.FileReads() - initialFileReads
	if redundantReads != 0 {
		t.Fatalf("redundant file reads during search: got %d, want 0", redundantReads)
	}

	// 4. Measures total CPU time / latency (<2ms per worker search).
	totalSearches := numWorkers * searchesPerWorker
	avgLatency := totalDuration / time.Duration(totalSearches)
	t.Logf("Executed %d searches across %d concurrent workers in %v (avg latency: %v per search)",
		totalSearches, numWorkers, totalDuration, avgLatency)

	for wID, wDur := range workerLatencies {
		wAvg := wDur / time.Duration(searchesPerWorker)
		t.Logf("Worker %d: %d searches in %v (avg %v per search)", wID, searchesPerWorker, wDur, wAvg)
		if wAvg >= 2*time.Millisecond {
			t.Errorf("worker %d average latency %v exceeded 2ms threshold", wID, wAvg)
		}
	}
	if avgLatency >= 2*time.Millisecond {
		t.Errorf("overall average search latency %v exceeded 2ms threshold", avgLatency)
	}

	// 5. Asserts memory footprint <10MB.
	memBytes := wix.MemoryBytes()
	const maxMemoryBytes = 10 * 1024 * 1024 // 10MB
	t.Logf("WorkspaceIndex estimated memory footprint: %d bytes (%.2f MB)", memBytes, float64(memBytes)/(1024*1024))
	if memBytes <= 0 {
		t.Errorf("expected positive memory footprint, got %d", memBytes)
	}
	if memBytes >= maxMemoryBytes {
		t.Errorf("index memory footprint %d bytes (%.2f MB) exceeded 10MB threshold",
			memBytes, float64(memBytes)/(1024*1024))
	}
}

func TestWorkspaceIndexDiskScanAndZeroReadOnSearch(t *testing.T) {
	tempDir := t.TempDir()
	for i := 0; i < 20; i++ {
		sub := filepath.Join(tempDir, fmt.Sprintf("sub%d", i))
		if err := os.MkdirAll(sub, 0755); err != nil {
			t.Fatal(err)
		}
		fPath := filepath.Join(sub, "code.go")
		content := fmt.Sprintf("package sub%d\nfunc Fn%d() { println(\"needle_%02d_end\") }\n", i, i, i)
		if err := os.WriteFile(fPath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	wix := NewWorkspaceIndex(tempDir)
	if err := wix.EnsureBuilt(); err != nil {
		t.Fatalf("EnsureBuilt failed: %v", err)
	}
	if wix.DocCount() != 20 {
		t.Fatalf("DocCount = %d, want 20", wix.DocCount())
	}

	readsAfterBuild := wix.FileReads()
	if readsAfterBuild != 20 {
		t.Fatalf("FileReads after build = %d, want 20", readsAfterBuild)
	}

	// Perform queries; file reads must NOT increment.
	for i := 0; i < 20; i++ {
		hits := wix.Search(fmt.Sprintf("needle_%02d_end", i))
		if len(hits) != 1 {
			t.Errorf("Search needle_%02d_end: got %d hits, want 1", i, len(hits))
		}
		regHits, err := wix.SearchRegexp(fmt.Sprintf(`Fn%d\(\)`, i))
		if err != nil || len(regHits) != 1 {
			t.Errorf("SearchRegexp Fn%d: got %d hits, err=%v", i, len(regHits), err)
		}
	}

	if extraReads := wix.FileReads() - readsAfterBuild; extraReads != 0 {
		t.Errorf("extra file reads during search = %d, want 0", extraReads)
	}
}

func TestWorkspaceIndexUpdateFile(t *testing.T) {
	docs := map[string]string{
		"a.go": "package main\nfunc Alpha() { }\n",
	}
	wix := NewWorkspaceIndexFromDocs(docs)
	if len(wix.Search("Alpha")) != 1 {
		t.Fatal("expected match for Alpha")
	}
	if len(wix.Search("Beta")) != 0 {
		t.Fatal("expected no match for Beta")
	}

	wix.UpdateFile("b.go", "package main\nfunc Beta() { }\n")
	if len(wix.Search("Beta")) != 1 {
		t.Fatal("expected match for Beta after UpdateFile")
	}
	if wix.DocCount() != 2 {
		t.Fatalf("DocCount = %d, want 2", wix.DocCount())
	}
}

func TestWorkspaceIndexInvalidRegex(t *testing.T) {
	wix := NewWorkspaceIndexFromDocs(map[string]string{"x.go": "package x\n"})
	_, err := wix.SearchRegexp("[unterminated")
	if err == nil {
		t.Fatal("expected error for invalid regex pattern, got nil")
	}
}
