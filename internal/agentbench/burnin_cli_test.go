package agentbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

// fak-test:runtime medium est=25s lane=default
func TestBurnInCLIRunsAndEvaluatesLeakOracle(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	addBurnCorpusMaterial(t, repo)

	normalEnc := &normalCorpusEncoder{}
	burnEnc := &burnMaterialEncoder{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []any{map[string]any{
					"id":             "fixture-model",
					"context_length": 32768,
					"fak_capabilities": map[string]any{
						"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"},
					},
				}},
			})
		case "/v1/fak/tokenize":
			var req referenceRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			raw, _ := json.Marshal(req)
			text := string(raw)
			lower := strings.ToLower(text)
			isBurnIn := strings.Contains(text, "Pinned shared repository foundation:") || (req.OutputTokens == 0 && len(text) > 8000 && strings.Contains(lower, "tool") && strings.Contains(lower, "repository"))
			var enc referenceEncoding
			var err error
			if isBurnIn {
				enc, err = burnEnc.Encode(r.Context(), req)
			} else {
				enc, err = normalEnc.Encode(r.Context(), req)
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			enc.RendererID = "renderer"
			enc.TokenizerID = "tokenizer"
			_ = json.NewEncoder(w).Encode(enc)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"progress"}}]}`+"\n\n")
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			_, _ = io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":14000,"completion_tokens":2,"total_tokens":14002}}`+"\n\ndata: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	origTaskRunner := burnInCLITaskRunner
	t.Cleanup(func() { burnInCLITaskRunner = origTaskRunner })

	t.Run("passes end-to-end with clean runtime", func(t *testing.T) {
		burnInCLITaskRunner = func(_ context.Context, opts taskrun.Options) (taskrun.Receipt, error) {
			tasks := make([]taskrun.TaskReceipt, opts.TaskLimit)
			for i := range tasks {
				tasks[i] = taskrun.TaskReceipt{ID: fmt.Sprintf("task-%d", i), Passed: true, Accepted: true, ExternalPassed: true}
			}
			return taskrun.Receipt{Tasks: tasks, ConfiguredConcurrency: opts.Concurrency}, nil
		}
		var stdout, stderr bytes.Buffer
		code := RunCLI(context.Background(), &stdout, &stderr, []string{
			"--profile", "burn-in",
			"--repo", repo,
			"--endpoint", server.URL + "/v1/chat/completions",
			"--model", "fixture-model",
			"--out", t.TempDir(),
			"--duration", "5s",
			"--burnin-epochs", "1",
			"--json",
		})
		if strings.Contains(stderr.String(), "UNSUPPORTED_INCOMPLETE") {
			t.Fatalf("burn-in still returned UNSUPPORTED_INCOMPLETE: %s", stderr.String())
		}
		if code != 0 {
			t.Fatalf("RunCLI exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
		}
		var receipt burnInReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("decode receipt: %v\nstdout=%s", err, stdout.String())
		}
		if !receipt.Qualified || receipt.Status != "PASSED" || receipt.FullEpochs != 1 {
			t.Fatalf("unexpected burn-in receipt: %+v", receipt)
		}
		if !receipt.LeakOracle.Passed || receipt.LeakOracle.Verdict != "PASS" || len(receipt.LeakOracle.EpochDeltas) != 1 {
			t.Fatalf("unexpected leak oracle report: %+v", receipt.LeakOracle)
		}
	})

	t.Run("flags synthetic goroutine leak breach", func(t *testing.T) {
		leakGate := make(chan struct{})
		t.Cleanup(func() { close(leakGate) })

		burnInCLITaskRunner = func(_ context.Context, opts taskrun.Options) (taskrun.Receipt, error) {
			for i := 0; i < 4; i++ {
				go func() { <-leakGate }()
			}
			tasks := make([]taskrun.TaskReceipt, opts.TaskLimit)
			for i := range tasks {
				tasks[i] = taskrun.TaskReceipt{ID: fmt.Sprintf("task-%d", i), Passed: true, Accepted: true, ExternalPassed: true}
			}
			return taskrun.Receipt{Tasks: tasks, ConfiguredConcurrency: opts.Concurrency}, nil
		}
		var stdout, stderr bytes.Buffer
		code := RunCLI(context.Background(), &stdout, &stderr, []string{
			"--profile", "burn-in",
			"--repo", repo,
			"--endpoint", server.URL + "/v1/chat/completions",
			"--model", "fixture-model",
			"--out", t.TempDir(),
			"--duration", "5s",
			"--burnin-epochs", "1",
			"--burnin-max-goroutine-delta", "1",
			"--json",
		})
		if code == 0 {
			t.Fatalf("expected non-zero exit on synthetic leak breach, got 0; stdout=%s", stdout.String())
		}
		var receipt burnInReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("decode receipt: %v\nstdout=%s", err, stdout.String())
		}
		if receipt.Qualified || receipt.Status != "INCOMPLETE" {
			t.Fatalf("leaking burn-in qualified: %+v", receipt)
		}
		if receipt.LeakOracle.Passed || receipt.LeakOracle.Verdict != "FAIL" || receipt.LeakOracle.MaxGoroutineDelta < 4 || len(receipt.LeakOracle.Breaches) == 0 {
			t.Fatalf("leak oracle did not flag synthetic goroutine breach: %+v", receipt.LeakOracle)
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestBurnInLeakOracleEvaluatesHeapAndFDBounds(t *testing.T) {
	t.Run("heap growth breach fails", func(t *testing.T) {
		bounds := burnInLeakBounds{MaxGoroutineDelta: 8, MaxHeapGrowthBytes: 1024, MaxFDDelta: 4}
		report := newBurnInLeakReport(bounds, burnInRuntimeSample{Goroutines: 10, HeapAlloc: 4096, OpenFDs: -1, FDStatus: "skipped_non_linux"})
		delta := report.recordEpoch(1, burnInRuntimeSample{Goroutines: 10, HeapAlloc: 8192, OpenFDs: -1, FDStatus: "skipped_non_linux"})
		if delta.Verdict != "FAIL" || delta.HeapDeltaBytes != 4096 || delta.FDStatus != "skipped_non_linux" {
			t.Fatalf("expected heap breach with non-linux FD skip, got %+v", delta)
		}
		finalBreaches := report.finalize(burnInRuntimeSample{Goroutines: 10, HeapAlloc: 8192, OpenFDs: -1, FDStatus: "skipped_non_linux"})
		if report.Passed || report.Verdict != "FAIL" || len(finalBreaches) == 0 || report.MaxHeapDeltaBytes != 4096 {
			t.Fatalf("expected final heap breach, got report=%+v breaches=%v", report, finalBreaches)
		}
	})

	t.Run("linux fd breach fails", func(t *testing.T) {
		bounds := burnInLeakBounds{MaxGoroutineDelta: 8, MaxHeapGrowthBytes: 64 << 20, MaxFDDelta: 2}
		report := newBurnInLeakReport(bounds, burnInRuntimeSample{Goroutines: 10, HeapAlloc: 4096, OpenFDs: 12, FDStatus: "measured"})
		delta := report.recordEpoch(1, burnInRuntimeSample{Goroutines: 10, HeapAlloc: 4096, OpenFDs: 17, FDStatus: "measured"})
		if delta.Verdict != "FAIL" || delta.FDDelta != 5 || len(delta.Breaches) != 1 {
			t.Fatalf("expected linux FD breach, got %+v", delta)
		}
		report.finalize(burnInRuntimeSample{Goroutines: 10, HeapAlloc: 4096, OpenFDs: 17, FDStatus: "measured"})
		if report.Passed || report.Verdict != "FAIL" || report.MaxFDDelta != 5 {
			t.Fatalf("expected final linux FD breach, got %+v", report)
		}
	})
}
