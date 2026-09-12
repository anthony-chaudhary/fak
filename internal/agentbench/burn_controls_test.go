package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBurnControlsExerciseActualSemanticMaterial(t *testing.T) {
	ctx := context.Background()
	if deadline, ok := t.Deadline(); ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, deadline)
		defer cancel()
	}
	repo := normalCorpusRepo(t, true)
	addBurnCorpusMaterial(t, repo)
	normal, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{}, repo)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := buildBurnInCorpus(context.Background(), &burnMaterialEncoder{}, normal, repo)
	if err != nil {
		t.Fatal(err)
	}
	server, wire := burnControlEndpoint(t)
	defer server.Close()
	reference, err := discoverNativeReference(context.Background(), http.DefaultClient, server.URL, "fixture-model")
	if err != nil {
		t.Fatal(err)
	}
	events, err := os.Create(filepath.Join(t.TempDir(), "controls.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	sink := &eventSink{file: events, terminal: map[int]lifecycleEvent{}, inFlight: map[string]int{}, peak: map[string]int{}}
	validated := false
	now := time.Now().UTC()
	receipt := runBurnControls(ctx, burnControlConfig{Client: http.DefaultClient, Endpoint: server.URL + "/v1/chat/completions", Model: "fixture-model", Reference: reference, Corpus: corpus, Sink: sink, AdmissionCutoff: now.Add(time.Hour), DrainDeadline: now.Add(time.Hour + 90*time.Second), Now: time.Now, ValidateSource: func(_ context.Context, e burnSourceControlEvidence) (bool, error) {
		validated = true
		if len(e.Changed) == 0 || !stringEqualBytes(e.Changed, e.Reread) || e.ChangedSHA256 != digestBytes(e.Changed) || e.RereadSHA256 != digestBytes(e.Reread) {
			return false, nil
		}
		return true, nil
	}})
	_ = events.Close()
	if receipt.Accepted != 24 || receipt.Completed != 24 || receipt.RejectedAfterCutoff != 0 || len(receipt.Requests) != 24 || len(receipt.Errors) != 0 {
		t.Fatalf("control geometry=%+v", receipt)
	}
	if receipt.PhysicalOccupancyStatus != "UNKNOWN" || !validated || !receipt.SourceValidationPassed || receipt.SourceValidation.ChangedSHA256 != receipt.SourceValidation.RereadSHA256 {
		t.Fatalf("source/occupancy receipt=%+v", receipt)
	}
	select {
	case <-wire.cancelReleased:
	case <-ctx.Done():
		t.Fatal("backend did not observe cancellation before the test deadline")
	}
	if !receipt.Cancellation.Observed || receipt.Cancellation.BackendReleased || !receipt.Cancellation.RecoveryPassed || !wire.recovered.Load() {
		t.Fatalf("cancel/recovery receipt=%+v wire=%+v", receipt.Cancellation, wire)
	}
	kinds := map[string][][]byte{}
	pressure := []int{}
	for _, request := range receipt.Requests {
		if request.Kind == "" || request.RequestID == "" || request.ReleasedAt.IsZero() || request.DispatchedAt.IsZero() || request.FirstOutputAt.IsZero() || request.EndedAt.IsZero() {
			t.Fatalf("incomplete control lifecycle: %+v", request)
		}
		if request.Kind == "cancel" {
			if request.UsageKnown || request.Error == "" {
				t.Fatalf("canceled stream fabricated terminal usage or lost cancellation: %+v", request)
			}
		} else if request.ObservedModel != "fixture-model" || !request.UsageKnown || request.Error != "" {
			t.Fatalf("completed control lacks model/usage evidence: %+v", request)
		}
		kinds[request.Kind] = append(kinds[request.Kind], wire.body(request.Kind))
		if strings.HasPrefix(request.Kind, "pressure-") {
			pressure = append(pressure, request.FixtureBytes)
		}
	}
	for _, kind := range []string{"reuse", "fork", "new-area", "no-share", "source-edit", "source-reread", "cancel", "recovery", "pressure-revisit"} {
		if len(kinds[kind]) == 0 {
			t.Fatalf("missing semantic control %s: %v", kind, kinds)
		}
	}
	if equalFirstBodies(kinds, "reuse", "fork", "new-area", "no-share") {
		t.Fatal("semantic controls differed only by labels")
	}
	sort.Ints(pressure)
	if len(pressure) != 4 || !(pressure[0] < pressure[1] && pressure[1] < pressure[2] && pressure[2] < pressure[3]) {
		t.Fatalf("pressure fixture volumes=%v", pressure)
	}
	body, err := os.ReadFile(events.Name())
	if err != nil || !strings.Contains(string(body), "first_output") || !strings.Contains(string(body), "terminal") {
		t.Fatalf("durable lifecycle missing: %s err=%v", body, err)
	}
}

type burnControlWire struct {
	mu             sync.Mutex
	bodies         map[string][]byte
	cancelReleased chan struct{}
	recovered      atomic.Bool
}

func (w *burnControlWire) body(id string) []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]byte(nil), w.bodies[id]...)
}
func burnControlEndpoint(t *testing.T) (*httptest.Server, *burnControlWire) {
	t.Helper()
	state := &burnControlWire{bodies: map[string][]byte{}, cancelReleased: make(chan struct{})}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
			return
		}
		var raw json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&raw)
		if r.URL.Path == "/v1/fak/tokenize" {
			var req referenceRequest
			_ = json.Unmarshal(raw, &req)
			n := min(30000, 13000+len(raw)/3)
			ids := make([]int, n)
			_ = json.NewEncoder(w).Encode(referenceEncoding{ModelID: "fixture-model", RendererID: "renderer", TokenizerID: "tokenizer", TokenIDs: ids, PromptTokens: n, ContextWindowTokens: 32768, ReservedOutputTokens: req.OutputTokens, RenderedSHA256: digestBytes(raw)})
			return
		}
		var envelope map[string]any
		_ = json.Unmarshal(raw, &envelope)
		metadata, _ := envelope["metadata"].(map[string]any)
		rung := fmt.Sprint(metadata["rung"])
		state.mu.Lock()
		state.bodies[rung] = append([]byte(nil), raw...)
		state.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n")
		w.(http.Flusher).Flush()
		if strings.Contains(rung, "cancel") {
			<-r.Context().Done()
			close(state.cancelReleased)
			return
		}
		if strings.Contains(rung, "recovery") {
			state.recovered.Store(true)
		}
		fmt.Fprint(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n")
	}))
	return server, state
}
func digestBytes(v []byte) string       { s := sha256.Sum256(v); return fmt.Sprintf("%x", s) }
func stringEqualBytes(a, b []byte) bool { return string(a) == string(b) }
func equalFirstBodies(groups map[string][][]byte, names ...string) bool {
	var first string
	for i, n := range names {
		if len(groups[n]) == 0 {
			return false
		}
		value := string(groups[n][0])
		if i == 0 {
			first = value
		} else if value != first {
			return false
		}
	}
	return true
}
