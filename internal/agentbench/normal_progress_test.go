package agentbench

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNormalRunPersistsProgressDuringBackendStall(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	repo, _ = filepath.EvalSymlinks(repo)
	entered := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
		case "/v1/fak/tokenize":
			select {
			case <-entered:
			default:
				close(entered)
			}
			select {
			case <-r.Context().Done():
			case <-release:
			}
		default:
			http.NotFound(w, r)
		}
	}))
	ctx, cancel := context.WithCancel(context.Background())
	defer func() {
		cancel()
		close(release)
		server.CloseClientConnections()
		server.Close()
	}()
	out := t.TempDir()
	done := make(chan error, 1)
	go func() { _, err := runNormal(ctx, &bytes.Buffer{}, repo, server.URL, "fixture-model", out); done <- err }()
	select {
	case <-entered:
	case err := <-done:
		t.Fatalf("normal run ended before native context encoding: %v", err)
	case <-time.After(30 * time.Second):
		t.Fatal("normal run did not enter native context encoding")
	}

	summaryPath := filepath.Join(out, "summary.json")
	firstBody, firstInfo := waitNormalProgressSnapshot(t, summaryPath, time.Second)
	if !bytes.Contains(firstBody, []byte(`"status":"in_progress"`)) || !bytes.Contains(firstBody, []byte(`"phase":"context-freeze"`)) {
		t.Fatalf("initial durable context-freeze checkpoint=%s", firstBody)
	}
	deadline := time.Now().Add(5200 * time.Millisecond)
	refreshed := false
	for time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
		body, err := os.ReadFile(summaryPath)
		info, statErr := os.Stat(summaryPath)
		if err == nil && statErr == nil && (info.ModTime().After(firstInfo.ModTime()) || !bytes.Equal(body, firstBody)) {
			refreshed = true
			break
		}
	}
	if !refreshed {
		t.Fatal("normal run emitted no durable progress refresh within five seconds while tokenizer request remained in flight")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("normal run did not stop after cancellation")
	}
	terminal, err := os.ReadFile(summaryPath)
	if err != nil || !strings.Contains(string(terminal), `"overall_verdict":"INCOMPLETE"`) {
		t.Fatalf("terminal cancellation checkpoint missing: %s err=%v", terminal, err)
	}
}

func waitNormalProgressSnapshot(t *testing.T, path string, timeout time.Duration) ([]byte, os.FileInfo) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		body, readErr := os.ReadFile(path)
		info, statErr := os.Stat(path)
		if readErr == nil && statErr == nil && len(body) > 0 {
			return body, info
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("no immediate durable progress checkpoint at %s", path)
	return nil, nil
}
