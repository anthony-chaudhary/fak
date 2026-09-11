package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

func TestBurnInEpochRunsRealReplayAndScenarioCoverage(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	encoder := &normalCorpusEncoder{}
	normal, err := buildNormalCorpus(context.Background(), encoder, repo)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := buildBurnInCorpus(context.Background(), encoder, normal, repo)
	if err != nil {
		t.Fatal(err)
	}
	if corpus.Schema == "" || corpus.Revision != normal.Revision || len(corpus.Sessions) != 8 {
		t.Fatalf("burn corpus identity/sessions = %+v", corpus)
	}
	for _, session := range corpus.Sessions {
		if len(session.Turns) != 32 || session.Turns[31].PromptTokens+session.Turns[31].OutputTokens > 32768 {
			t.Fatalf("session %s is modulo-reused or exceeds 32K: turns=%d last=%+v", session.ID, len(session.Turns), session.Turns[len(session.Turns)-1])
		}
	}

	server, replayCalls := burnEpochEndpoint(t)
	defer server.Close()
	reference, err := discoverNativeReference(context.Background(), http.DefaultClient, server.URL, "fixture-model")
	if err != nil {
		t.Fatal(err)
	}
	plan, _ := profileplan.Build(profileplan.Options{Profile: "burn-in", PreferredConcurrency: 4})
	spec := burnInSpec(plan, 1)
	out := t.TempDir()
	now := time.Date(2026, 9, 10, 8, 0, 0, 0, time.UTC)
	tasksCalled := 0
	receipt := runBurnInEpoch(context.Background(), burnInEpochConfig{
		Client: http.DefaultClient, Endpoint: server.URL + "/v1/chat/completions", Model: "fixture-model", OutDir: out,
		Corpus: corpus, Reference: reference, Now: func() time.Time { return now },
		TaskOptions: taskrun.Options{Endpoint: server.URL, Model: "fixture-model", Concurrency: 4},
		RunTasks: func(_ context.Context, opts taskrun.Options) (taskrun.Receipt, error) {
			tasksCalled++
			tasks := make([]taskrun.TaskReceipt, spec.Tasks)
			for i := range tasks {
				tasks[i] = taskrun.TaskReceipt{ID: fmt.Sprintf("task-%d", i), Passed: true, Accepted: true, ExternalPassed: true}
			}
			return taskrun.Receipt{Tasks: tasks, ConfiguredConcurrency: opts.Concurrency}, nil
		},
	}, spec, now.Add(2*time.Hour))
	if !receipt.Complete || receipt.Partial || receipt.SteadyRequests != 256 || receipt.ControlRequests != 24 || replayCalls.Load() != 280 || receipt.TaskAttempts != 6 || receipt.AcceptedTasks != 6 || tasksCalled != 1 {
		t.Fatalf("real epoch geometry/acceptance = %+v replay=%d taskCalls=%d", receipt, replayCalls.Load(), tasksCalled)
	}
	if receipt.AreasVisited != 4 || len(receipt.AreaTransitions) < 4 || !receipt.SourceEditRereadPassed || !receipt.IntentionalCancellationObserved || !receipt.RecoveryPassed {
		t.Fatalf("area/edit/cancel coverage = %+v", receipt)
	}
	if receipt.PressureStatus != "unknown" || !reflect.DeepEqual(receipt.PressureCohorts, []int{50, 80, 100, 120}) {
		t.Fatalf("unknown physical occupancy was not replaced by explicit fixture pressure cohorts: %+v", receipt)
	}
	if receipt.PrimingRequests > 8 || receipt.ReplayRetries > 8 || len(receipt.Errors) != 0 {
		t.Fatalf("epoch budgets/errors = %+v", receipt)
	}
	for _, transition := range receipt.AreaTransitions {
		if transition.Area == "" || transition.BeforeSHA256 == "" || transition.AfterSHA256 == "" {
			t.Fatalf("unversioned area transition: %+v", transition)
		}
	}
	artifact, err := os.ReadFile(receipt.EventsPath)
	if err != nil || len(artifact) == 0 || receipt.EventsDigest != fmt.Sprintf("%x", sha256.Sum256(artifact)) || !strings.Contains(string(artifact), "intentional_cancel") || !strings.Contains(string(artifact), "recovery") {
		t.Fatalf("durable epoch lifecycle artifact invalid: bytes=%d receipt=%+v err=%v", len(artifact), receipt, err)
	}

	before := replayCalls.Load()
	cutoff := now
	partial := runBurnInEpoch(context.Background(), burnInEpochConfig{Client: http.DefaultClient, Endpoint: server.URL + "/v1/chat/completions", Model: "fixture-model", OutDir: t.TempDir(), Corpus: corpus, Reference: reference, Now: func() time.Time { return cutoff }, RunTasks: func(context.Context, taskrun.Options) (taskrun.Receipt, error) {
		t.Fatal("tasks admitted at cutoff")
		return taskrun.Receipt{}, nil
	}}, burnInSpec(plan, 2), cutoff)
	if !partial.Partial || partial.Complete || replayCalls.Load() != before {
		t.Fatalf("cutoff admitted new work: %+v calls before/after=%d/%d", partial, before, replayCalls.Load())
	}
}

func burnEpochEndpoint(t *testing.T) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var replay atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
		case "/v1/fak/tokenize":
			var req referenceRequest
			json.NewDecoder(r.Body).Decode(&req)
			ids := make([]int, min(31744, 14000+len(req.Messages)*500))
			json.NewEncoder(w).Encode(referenceEncoding{ModelID: "fixture-model", RendererID: "renderer", TokenizerID: "tokenizer", TokenIDs: ids, PromptTokens: len(ids), ContextWindowTokens: 32768, ReservedOutputTokens: req.OutputTokens, RenderedSHA256: strings.Repeat("a", 64)})
		case "/v1/chat/completions":
			replay.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"progress"}}]}`+"\n\n")
			w.(http.Flusher).Flush()
			io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":14000,"completion_tokens":2,"total_tokens":14002}}`+"\n\ndata: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	return server, &replay
}

func TestBurnInEpochExecutesSemanticControls(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	encoder := &normalCorpusEncoder{}
	normal, err := buildNormalCorpus(context.Background(), encoder, repo)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := buildBurnInCorpus(context.Background(), encoder, normal, repo)
	if err != nil {
		t.Fatal(err)
	}

	// The four advertised areas must be four distinct encoded source histories, and
	// extended turns must carry measured, growing history rather than replay turn 12.
	areaDigests := map[string]string{}
	for _, session := range corpus.Sessions {
		first := burnRequestDigest(session.Turns[0].Request)
		if prior, ok := areaDigests[session.Area]; ok && prior != first {
			continue
		}
		areaDigests[session.Area] = first
		grewBeforeCap := false
		for turn := 12; turn < len(session.Turns); turn++ {
			prev, next := session.Turns[turn-1], session.Turns[turn]
			if burnRequestDigest(prev.Request) == burnRequestDigest(next.Request) || next.Encoding.RenderedSHA256 == prev.Encoding.RenderedSHA256 || next.PromptTokens < prev.PromptTokens {
				t.Fatalf("session %s turn %d repeats instead of carrying distinct measured history: prev=%d next=%d", session.ID, turn+1, prev.PromptTokens, next.PromptTokens)
			}
			grewBeforeCap = grewBeforeCap || next.PromptTokens > prev.PromptTokens
		}
		if !grewBeforeCap && session.Turns[11].PromptTokens < 31744 {
			t.Fatalf("session %s never grew toward its measured 32K envelope", session.ID)
		}
	}
	if len(areaDigests) != 4 {
		t.Fatalf("burn corpus has %d distinct source areas, want A/B/C/D", len(areaDigests))
	}
	seenDigest := map[string]bool{}
	for area, digest := range areaDigests {
		if seenDigest[digest] {
			t.Fatalf("area %s relabels another area's source bytes (%s)", area, digest)
		}
		seenDigest[digest] = true
	}

	server, observation := semanticBurnEndpoint(t)
	defer server.Close()
	reference, err := discoverNativeReference(context.Background(), http.DefaultClient, server.URL, "fixture-model")
	if err != nil {
		t.Fatal(err)
	}
	plan, _ := profileplan.Build(profileplan.Options{Profile: "burn-in", PreferredConcurrency: 4})
	spec := burnInSpec(plan, 1)
	now := time.Now()
	receipt := runBurnInEpoch(context.Background(), burnInEpochConfig{
		Client: http.DefaultClient, Endpoint: server.URL + "/v1/chat/completions", Model: "fixture-model", OutDir: t.TempDir(), Corpus: corpus, Reference: reference,
		RunTasks: func(context.Context, taskrun.Options) (taskrun.Receipt, error) {
			tasks := make([]taskrun.TaskReceipt, spec.Tasks)
			for i := range tasks {
				tasks[i] = taskrun.TaskReceipt{ID: fmt.Sprint(i), Passed: true, Accepted: true, ExternalPassed: true}
			}
			return taskrun.Receipt{Tasks: tasks}, nil
		},
	}, spec, now.Add(time.Hour))
	observation.mu.Lock()
	defer observation.mu.Unlock()
	if observation.peak < spec.Concurrency {
		t.Fatalf("steady requests ran sequentially: peak=%d want >=%d", observation.peak, spec.Concurrency)
	}
	if !observation.cancelReleased || !observation.recovered {
		t.Fatalf("cancel/recovery not observed by backend: %+v", observation)
	}
	if len(observation.pressureBytes) != 4 || !(observation.pressureBytes[0] < observation.pressureBytes[1] && observation.pressureBytes[1] < observation.pressureBytes[2] && observation.pressureBytes[2] < observation.pressureBytes[3]) {
		t.Fatalf("pressure cohorts did not send increasing 50/80/100/120 fixture volumes: %v", observation.pressureBytes)
	}
	if observation.editVersion == "" || observation.rereadVersion != observation.editVersion {
		t.Fatalf("source edit/reread did not carry identical changed bytes: edit=%q reread=%q", observation.editVersion, observation.rereadVersion)
	}
	body, err := os.ReadFile(receipt.EventsPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) < 2*(spec.Sessions*spec.TurnsPerSession+spec.Controls) || !strings.Contains(string(body), `"stage":"dispatched"`) || !strings.Contains(string(body), `"stage":"terminal"`) {
		t.Fatalf("lifecycle is not durably paired per request: rows=%d body=%s", len(lines), body)
	}
}

type burnWireObservation struct {
	mu                         sync.Mutex
	inflight, peak             int
	cancelReleased, recovered  bool
	pressureBytes              []int
	editVersion, rereadVersion string
}

func semanticBurnEndpoint(t *testing.T) (*httptest.Server, *burnWireObservation) {
	t.Helper()
	o := &burnWireObservation{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
			return
		}
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if r.URL.Path == "/v1/fak/tokenize" {
			json.NewEncoder(w).Encode(map[string]any{"model_id": "fixture-model", "renderer_id": "renderer", "tokenizer_id": "tokenizer", "token_ids": []int{1, 2, 3}, "prompt_tokens": 3, "context_window_tokens": 32768, "reserved_output_tokens": 128, "rendered_sha256": strings.Repeat("a", 64)})
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		raw, _ := json.Marshal(req)
		metadata, _ := req["metadata"].(map[string]any)
		rung, _ := metadata["rung"].(string)
		o.mu.Lock()
		o.inflight++
		if o.inflight > o.peak {
			o.peak = o.inflight
		}
		switch {
		case strings.HasPrefix(rung, "pressure-"):
			o.pressureBytes = append(o.pressureBytes, len(raw))
		case rung == "source-edit":
			o.editVersion = burnSourceVersion(raw)
		case rung == "source-reread":
			o.rereadVersion = burnSourceVersion(raw)
		case rung == "recovery":
			o.recovered = true
		}
		o.mu.Unlock()
		defer func() { o.mu.Lock(); o.inflight--; o.mu.Unlock() }()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"first"}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		if rung == "cancel" {
			<-r.Context().Done()
			o.mu.Lock()
			o.cancelReleased = true
			o.mu.Unlock()
			return
		}
		time.Sleep(10 * time.Millisecond)
		io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`+"\n\ndata: [DONE]\n\n")
	}))
	return server, o
}

func burnSourceVersion(raw []byte) string {
	const marker = "burn-source-version:"
	text := string(raw)
	start := strings.Index(text, marker)
	if start < 0 {
		return ""
	}
	start += len(marker)
	end := start
	for end < len(text) && ((text[end] >= '0' && text[end] <= '9') || (text[end] >= 'a' && text[end] <= 'f')) {
		end++
	}
	if end-start != 64 {
		return ""
	}
	return text[start:end]
}
