package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// fakeLlamaSlot is a llama-server upstream with one slot: each chat completion
// becomes the slot's next task, and GET /slots reports that task's id and prompt.
type fakeLlamaSlot struct {
	mu         sync.Mutex
	task       int
	prompt     int
	cached     int  // prompt_tokens_details.cached_tokens of the next completion
	busy       bool // /slots reports is_processing
	slots      int  // slots reported; 0 means 1
	skewPrompt int  // added to n_prompt_tokens in /slots
	noSlots    bool // /slots answers 404
	completes  atomic.Int32
	slotReads  atomic.Int32
}

func (f *fakeLlamaSlot) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/chat/completions":
		var req struct {
			Messages []agent.Message `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		f.mu.Lock()
		f.task++
		f.prompt = estimateMessageContentTokens(req.Messages)
		prompt, cached := f.prompt, f.cached
		f.mu.Unlock()
		f.completes.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "halo",
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": prompt, "completion_tokens": 1, "prompt_tokens_details": map[string]any{"cached_tokens": cached}},
		})
	case "/slots":
		defer f.slotReads.Add(1)
		f.mu.Lock()
		defer f.mu.Unlock()
		if f.noSlots {
			http.NotFound(w, r)
			return
		}
		n := max(f.slots, 1)
		out := make([]map[string]any, n)
		for i := range out {
			out[i] = map[string]any{"id": i, "id_task": f.task, "is_processing": f.busy, "n_prompt_tokens": f.prompt + f.skewPrompt}
		}
		_ = json.NewEncoder(w).Encode(out)
	default:
		http.NotFound(w, r)
	}
}

// otherClient runs a task on the slot without going through the gateway.
func (f *fakeLlamaSlot) otherClient() {
	f.mu.Lock()
	f.task++
	f.prompt = 300
	f.mu.Unlock()
}

func llamaSlotHaloServer(t *testing.T, f *fakeLlamaSlot) *Server {
	t.Helper()
	srv := haloPiDeadlineServer(t)
	up := httptest.NewServer(f)
	t.Cleanup(up.Close)
	srv.planner = agent.NewHTTPPlanner(up.URL+"/v1", "halo", "")
	return srv
}

func postHaloChat(t *testing.T, srv *Server, model string, messages []agent.Message, budget string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(ChatRequest{Model: model, Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if budget != "" {
		r.Header.Set(deadlineadmit.HeaderStainlessTimeout, budget)
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, r)
	return rec
}

// awaitResidencyRead waits until the off-path /slots read after a served turn
// has completed, so an assertion never races the record.
func awaitResidencyRead(t *testing.T, f *fakeLlamaSlot, reads int32) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for f.slotReads.Load() < reads {
		if time.Now().After(deadline) {
			t.Fatalf("/slots reads = %d, want %d", f.slotReads.Load(), reads)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// awaitResidencyEntry waits until a served turn's residency evidence is recorded.
func awaitResidencyEntry(t *testing.T, srv *Server) {
	t.Helper()
	mem := srv.metrics.deadlineResidencyMem()
	deadline := time.Now().Add(10 * time.Second)
	for {
		mem.mu.Lock()
		n := len(mem.entries)
		mem.mu.Unlock()
		if n > 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("served turn left no residency evidence")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func isColdRefusal(rec *httptest.ResponseRecorder) bool {
	return rec.Code == http.StatusServiceUnavailable &&
		rec.Header().Get(deadlineadmit.HeaderReject) == string(deadlineadmit.ReasonDeadlineInfeasible)
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=300ms lane=default
func TestDeadlineAdmissionCreditsEngineProvenSlotResidency(t *testing.T) {
	f := &fakeLlamaSlot{}
	srv := llamaSlotHaloServer(t, f)
	turn1, turn2 := haloPiTurns()

	if rec := postHaloChat(t, srv, "halo", turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d body=%s", rec.Code, rec.Body.String())
	}
	awaitResidencyEntry(t, srv)
	f.mu.Lock()
	f.cached = 48_000
	f.mu.Unlock()
	rec := postHaloChat(t, srv, "halo", turn2, "600")
	if rec.Code != http.StatusOK || f.completes.Load() != 2 {
		t.Fatalf("warm turn 2: status=%d reject=%q completions=%d; want admission on the slot llama-server reports unchanged",
			rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), f.completes.Load())
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=1s lane=default
func TestDeadlineAdmissionRefusesUnprovenSlotResidency(t *testing.T) {
	turn1, turn2 := haloPiTurns()
	for _, tc := range []struct {
		name       string
		unrecorded bool // turn 1 must leave no evidence
		setup      func(f *fakeLlamaSlot)
		between    func(t *testing.T, srv *Server, f *fakeLlamaSlot)
		model2     string
	}{
		{name: "other_client_task_since", between: func(_ *testing.T, _ *Server, f *fakeLlamaSlot) { f.otherClient() }},
		{name: "slot_busy_at_admission", between: func(_ *testing.T, _ *Server, f *fakeLlamaSlot) {
			f.mu.Lock()
			f.busy = true
			f.mu.Unlock()
		}},
		{name: "several_slots", unrecorded: true, setup: func(f *fakeLlamaSlot) { f.slots = 2 }},
		{name: "slot_prompt_differs", unrecorded: true, setup: func(f *fakeLlamaSlot) { f.skewPrompt = 1 }},
		{name: "no_slots_endpoint", unrecorded: true, setup: func(f *fakeLlamaSlot) { f.noSlots = true }},
		{name: "planner_replaced", between: func(_ *testing.T, srv *Server, _ *fakeLlamaSlot) {
			old := srv.planner.(*agent.HTTPPlanner)
			srv.planner = agent.NewHTTPPlanner(old.BaseURL, "halo", "")
		}},
		{name: "concurrent_in_flight", between: func(t *testing.T, srv *Server, _ *fakeLlamaSlot) {
			_, release, ok := admitHaloTurn(t, srv, context.Background(), []agent.Message{{Role: "user", Content: "x"}}, "")
			if !ok {
				t.Fatal("concurrent admission refused")
			}
			t.Cleanup(release)
		}},
		{name: "different_model", model2: "other-model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeLlamaSlot{}
			if tc.setup != nil {
				tc.setup(f)
			}
			srv := llamaSlotHaloServer(t, f)
			if rec := postHaloChat(t, srv, "halo", turn1, ""); rec.Code != http.StatusOK {
				t.Fatalf("turn 1: status=%d body=%s", rec.Code, rec.Body.String())
			}
			if tc.unrecorded {
				awaitResidencyRead(t, f, 1)
			} else {
				awaitResidencyEntry(t, srv)
			}
			if tc.between != nil {
				tc.between(t, srv, f)
			}
			model2 := tc.model2
			if model2 == "" {
				model2 = "halo"
			}
			done := f.completes.Load()
			rec := postHaloChat(t, srv, model2, turn2, "600")
			if !isColdRefusal(rec) || f.completes.Load() != done {
				t.Fatalf("turn 2: status=%d reject=%q; want cold refusal before inference",
					rec.Code, rec.Header().Get(deadlineadmit.HeaderReject))
			}
		})
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=300ms lane=default
func TestDeadlineAdmissionClearsResidencyTheEngineContradicts(t *testing.T) {
	f := &fakeLlamaSlot{}
	srv := llamaSlotHaloServer(t, f)
	turn1, turn2 := haloPiTurns()
	if rec := postHaloChat(t, srv, "halo", turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d", rec.Code)
	}
	awaitResidencyEntry(t, srv)
	mem := srv.metrics.deadlineResidencyMem()
	stale := deadlinePrefixChain([]agent.Message{{Role: "user", Content: "stale"}})
	mem.mu.Lock()
	e, ok := mem.entries[deadlinePrefixChain(turn1)[len(turn1)-1]]
	mem.entries[stale[0]] = e
	mem.mu.Unlock()
	if !ok {
		t.Fatal("turn 1 left no residency evidence")
	}

	// The credited turn finds nothing cached: the engine contradicts the credit.
	if rec := postHaloChat(t, srv, "halo", turn2, "600"); rec.Code != http.StatusOK {
		t.Fatalf("credited turn 2: status=%d", rec.Code)
	}
	mem.mu.Lock()
	_, kept := mem.entries[stale[0]]
	mem.mu.Unlock()
	if kept {
		t.Fatal("residency evidence survived an engine-contradicted credit")
	}
}
