package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

type softDeadlineDispatch struct {
	slot   int
	cache  bool
	stream bool
}

type softDeadlineReadGate struct {
	entered chan struct{}
	open    chan struct{}
	once    sync.Once
}

func (g *softDeadlineReadGate) release() { g.once.Do(func() { close(g.open) }) }

type softDeadlineUpstream struct {
	mu        sync.Mutex
	slots     [4]agent.LlamaSlotTask
	nextTask  int
	dispatch  []softDeadlineDispatch
	gate      *softDeadlineReadGate
	postGate  *softDeadlineReadGate // holds only the next chat POST, before it is recorded
	duplicate bool
}

func (f *softDeadlineUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/props":
		_, _ = w.Write([]byte(`{"total_slots":4}`))
	case "/slots":
		f.mu.Lock()
		gate := f.gate
		f.mu.Unlock()
		if gate != nil {
			select {
			case gate.entered <- struct{}{}:
			default:
			}
			select {
			case <-gate.open:
			case <-r.Context().Done():
				return
			}
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		var slots []map[string]any
		for id, slot := range f.slots {
			slots = append(slots, map[string]any{"id": id, "id_task": slot.IDTask, "is_processing": slot.Processing, "n_prompt_tokens": slot.PromptTokens})
		}
		if f.duplicate {
			slots = append(slots, slots[0])
		}
		_ = json.NewEncoder(w).Encode(slots)
	case "/v1/chat/completions":
		var req struct {
			Messages []agent.Message `json:"messages"`
			Slot     *int            `json:"id_slot"`
			Cache    bool            `json:"cache_prompt"`
			Stream   bool            `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		postGate := f.postGate
		f.postGate = nil
		f.mu.Unlock()
		if postGate != nil {
			postGate.entered <- struct{}{}
			select {
			case <-postGate.open:
			case <-r.Context().Done():
				return
			}
		}
		id, sent := 0, -1
		if req.Slot != nil {
			id, sent = *req.Slot, *req.Slot
		}
		if id < 0 || id >= len(f.slots) {
			http.Error(w, "invalid slot", http.StatusBadRequest)
			return
		}
		prompt := estimateMessageContentTokens(req.Messages)
		f.mu.Lock()
		cached := min(prompt, f.slots[id].PromptTokens)
		f.nextTask++
		f.slots[id] = agent.LlamaSlotTask{IDTask: f.nextTask, PromptTokens: prompt}
		f.dispatch = append(f.dispatch, softDeadlineDispatch{slot: sent, cache: req.Cache, stream: req.Stream})
		f.mu.Unlock()
		usage := map[string]any{"prompt_tokens": prompt, "completion_tokens": 1, "prompt_tokens_details": map[string]any{"cached_tokens": cached}}
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			chunk, _ := json.Marshal(map[string]any{"model": "halo", "choices": []any{map[string]any{"delta": map[string]any{"content": "ok"}, "finish_reason": "stop"}}, "usage": usage})
			_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "halo", "choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}, "usage": usage})
	default:
		http.NotFound(w, r)
	}
}

func softDeadlineServer(t *testing.T) (*Server, *agent.HTTPPlanner, *softDeadlineUpstream) {
	t.Helper()
	f := &softDeadlineUpstream{}
	up := httptest.NewServer(f)
	t.Cleanup(up.Close)
	p := agent.NewHTTPPlanner(up.URL+"/v1", "halo", "")
	p.LlamaSlotAffinity, p.LlamaSoftSlot = true, true
	primeSoftDeadlinePlanner(t, p, f)
	srv := haloPiDeadlineServer(t)
	srv.planner = p
	// Match this admission-policy control's 600-second client envelope. The
	// normal buffered default is 60 seconds and correctly loses warm credit.
	srv.firstTokenWatchdog = func() time.Duration { return 600 * time.Second }
	return srv, p, f
}

func primeSoftDeadlinePlanner(t *testing.T, p *agent.HTTPPlanner, f *softDeadlineUpstream) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for {
		if _, err := p.Complete(ctx, []agent.Message{{Role: "user", Content: "discovery"}}, nil); err != nil {
			t.Fatal(err)
		}
		f.mu.Lock()
		last := f.dispatch[len(f.dispatch)-1]
		ready := last.slot >= 0 && last.cache
		if ready {
			f.dispatch = nil
		}
		f.mu.Unlock()
		if ready {
			break
		}
	}
}

func softDeadlineRequest(t *testing.T, messages []agent.Message, budget string, stream bool) *http.Request {
	t.Helper()
	body, err := json.Marshal(ChatRequest{Model: "halo", Messages: messages, Stream: stream})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if budget != "" {
		r.Header.Set(deadlineadmit.HeaderStainlessTimeout, budget)
	}
	return r
}

func softDeadlineCold(t *testing.T, srv *Server, f *softDeadlineUpstream) int {
	t.Helper()
	cold, _ := haloPiTurns()
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, softDeadlineRequest(t, cold, "", false))
	if w.Code != http.StatusOK {
		t.Fatalf("cold status=%d: %s", w.Code, w.Body.String())
	}
	// Uses the same existing memory API on parent and candidate. The parent
	// cannot record a four-slot dispatch, so it fails behaviorally here.
	deadline := time.Now().Add(2 * time.Second)
	for {
		mem := srv.metrics.deadlineResidencyMem()
		mem.mu.Lock()
		ready := len(mem.entries) > 0
		mem.mu.Unlock()
		if ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("four-slot completion left no residency evidence")
		}
		time.Sleep(5 * time.Millisecond)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.dispatch) != 1 || f.dispatch[0].slot < 0 {
		t.Fatalf("cold dispatch was not a real soft pin: %+v", f.dispatch)
	}
	return f.dispatch[0].slot
}

// All symbols predate the fix, so native parent overlay remains executable.
// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime medium est=7s lane=default
func TestDeadlineAdmissionPreparedSoftSlotResidency(t *testing.T) {
	for _, name := range []string{"buffered", "streamed", "other_slot_evicted", "shorter_watchdog", "selected_slot_evicted", "prompt_changed", "busy_wait_slot", "soft_disabled", "explicit_slot", "explicit_other_slot", "cache_disabled", "shaped_prefix", "shaping_loses_soft_slot", "shaping_grows_tail", "static_authorization_override", "dynamic_authorization_override", "duplicate_slot"} {
		t.Run(name, func(t *testing.T) {
			srv, p, f := softDeadlineServer(t)
			slot := softDeadlineCold(t, srv, f)
			wantCredit := name == "buffered" || name == "streamed" || name == "other_slot_evicted"
			shapedPromptTokens := 0
			f.mu.Lock()
			switch name {
			case "shorter_watchdog":
				srv.firstTokenWatchdog = func() time.Duration { return 60 * time.Second }
			case "other_slot_evicted":
				f.slots[(slot+1)%4].IDTask++
			case "selected_slot_evicted":
				f.slots[slot].IDTask++
			case "prompt_changed":
				f.slots[slot].PromptTokens++
			case "busy_wait_slot":
				f.slots[slot].Processing = true
				p.LlamaSoftSlotPolicy = agent.SoftSlotPolicyWait
			case "soft_disabled":
				p.LlamaSoftSlot = false
			case "explicit_slot":
				p.ExtraBody = json.RawMessage(fmt.Sprintf(`{"id_slot":%d}`, slot))
			case "explicit_other_slot":
				p.ExtraBody = json.RawMessage(fmt.Sprintf(`{"id_slot":%d}`, (slot+1)%4))
			case "cache_disabled":
				p.ExtraBody = json.RawMessage(`{"cache_prompt":false}`)
			case "shaped_prefix":
				p.CoherenceShaper = func(messages []agent.Message) []agent.Message {
					out := append([]agent.Message(nil), messages...)
					out[0].Name = "changed-after-ingress"
					return out
				}
			case "shaping_loses_soft_slot":
				p.CoherenceShaper = func(messages []agent.Message) []agent.Message { return messages[:1] }
			case "shaping_grows_tail":
				p.CoherenceShaper = func(messages []agent.Message) []agent.Message {
					out := append(append([]agent.Message(nil), messages...), agent.Message{Role: "user", Content: strings.Repeat("x", 400_000)})
					// Capture the real gateway-prepared input, including gateway-added
					// content, then apply the planner's following stabilization seam.
					prepared, _ := agent.StabilizePromptPrefix(out)
					shapedPromptTokens = estimateMessageContentTokens(prepared)
					return out
				}
			case "static_authorization_override":
				p.ExtraHeaders = map[string]string{"Authorization": "Bearer distinct-static-tenant"}
			case "dynamic_authorization_override":
				p.ExtraHeadersFunc = func() map[string]string { return map[string]string{"Authorization": "Bearer distinct-dynamic-tenant"} }
			case "duplicate_slot":
				f.duplicate = true
			}
			f.mu.Unlock()
			_, warm := haloPiTurns()
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, softDeadlineRequest(t, warm, "600", name == "streamed"))
			f.mu.Lock()
			defer f.mu.Unlock()
			if wantCredit {
				if w.Code != http.StatusOK || len(f.dispatch) != 2 || f.dispatch[1].slot != slot || !f.dispatch[1].cache || f.dispatch[1].stream != (name == "streamed") {
					t.Fatalf("credit was not bound to dispatched slot %d: status=%d calls=%+v body=%s", slot, w.Code, f.dispatch, w.Body.String())
				}
			} else if !isColdRefusal(w) || len(f.dispatch) != 1 {
				t.Fatalf("unproven warm prefix dispatched: status=%d reject=%q calls=%+v body=%s", w.Code, w.Header().Get(deadlineadmit.HeaderReject), f.dispatch, w.Body.String())
			}
			if name == "shaping_grows_tail" {
				x, err := deadlineadmit.ReadEvidence(bytes.NewReader(w.Body.Bytes()))
				if err != nil || x == nil || x.DeclaredBudgetMS == nil || *x.DeclaredBudgetMS != 600000 || x.RemainingBudgetMS <= 0 || x.RemainingBudgetMS > 600000 || shapedPromptTokens <= estimateMessageContentTokens(warm) || x.PromptTokens != int64(shapedPromptTokens) || x.CachedTokens != 0 || x.OutputCeiling != 0 || x.InFlight != 0 {
					t.Fatalf("prepared soft-slot refusal lost actual shaped inputs: %+v err=%v", x, err)
				}
			}
		})
	}
}

func softDeadlineGate(t *testing.T, f *softDeadlineUpstream) *softDeadlineReadGate {
	t.Helper()
	g := &softDeadlineReadGate{entered: make(chan struct{}, 16), open: make(chan struct{})}
	t.Cleanup(g.release)
	f.mu.Lock()
	f.gate = g
	f.mu.Unlock()
	// No request is active: this first blocked reader is the ordinary poller.
	// The next reader will be the warm call's live proof, not a guessed sleep.
	softDeadlineWaitRead(t, g)
	return g
}

func softDeadlineWaitRead(t *testing.T, g *softDeadlineReadGate) {
	t.Helper()
	select {
	case <-g.entered:
	case <-time.After(15 * time.Second):
		t.Fatal("expected bounded /slots read did not arrive")
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime medium est=2s lane=default
func TestDeadlineAdmissionSoftSlotReservationLifecycle(t *testing.T) {
	for _, cancelWarm := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel=%v", cancelWarm), func(t *testing.T) {
			srv, p, f := softDeadlineServer(t)
			p.LlamaSoftSlotPolicy = agent.SoftSlotPolicyWait
			slot := softDeadlineCold(t, srv, f)
			// The canceled proof is held at its bounded /slots read. The surviving
			// proof is held at its POST instead: the reservation spans dispatch, and
			// a held /slots read would expire before a loaded host serves the peer.
			var gate *softDeadlineReadGate
			if cancelWarm {
				gate = softDeadlineGate(t, f)
			} else {
				gate = &softDeadlineReadGate{entered: make(chan struct{}, 1), open: make(chan struct{})}
				t.Cleanup(gate.release)
				f.mu.Lock()
				f.postGate = gate
				f.mu.Unlock()
			}
			_, warm := haloPiTurns()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := softDeadlineRequest(t, warm, "600", false).WithContext(ctx)
			w := httptest.NewRecorder()
			done := make(chan struct{})
			go func() { defer close(done); srv.Handler().ServeHTTP(w, r) }()
			softDeadlineWaitRead(t, gate)
			if cancelWarm {
				cancel()
			} else {
				other := httptest.NewRecorder()
				srv.Handler().ServeHTTP(other, softDeadlineRequest(t, warm, "", false))
				f.mu.Lock()
				if other.Code != http.StatusOK || len(f.dispatch) != 2 || f.dispatch[1].slot == slot {
					t.Errorf("wait policy queued on reserved slot %d: status=%d calls=%+v", slot, other.Code, f.dispatch)
				}
				f.mu.Unlock()
			}
			gate.release()
			select {
			case <-done:
			case <-time.After(15 * time.Second):
				t.Fatal("warm request did not release")
			}
			if !cancelWarm {
				f.mu.Lock()
				calls := append([]softDeadlineDispatch(nil), f.dispatch...)
				f.mu.Unlock()
				if w.Code != http.StatusOK || len(calls) != 3 || calls[2].slot != slot || !calls[2].cache {
					t.Fatalf("reserved warm turn status=%d calls=%+v: %s", w.Code, calls, w.Body.String())
				}
			}
			if cancelWarm {
				f.mu.Lock()
				if len(f.dispatch) != 1 {
					t.Errorf("canceled proof dispatched: %+v", f.dispatch)
				}
				f.mu.Unlock()
				retry := httptest.NewRecorder()
				srv.Handler().ServeHTTP(retry, softDeadlineRequest(t, warm, "", false))
				f.mu.Lock()
				defer f.mu.Unlock()
				if retry.Code != http.StatusOK || len(f.dispatch) != 2 || f.dispatch[1].slot != slot {
					t.Fatalf("canceled reservation leaked: status=%d calls=%+v", retry.Code, f.dispatch)
				}
			}
			if got := srv.metrics.deadlineEstimator().InFlight(); got != 0 {
				t.Fatalf("deadline reservation leaked %d active requests", got)
			}
		})
	}
}

// Same planner pointer and coincident engine counters are not route identity.
// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=500ms lane=default
func TestDeadlineAdmissionSoftSlotProofDoesNotCrossRoute(t *testing.T) {
	srv, p, original := softDeadlineServer(t)
	slot := softDeadlineCold(t, srv, original)
	if slot != 1 {
		t.Fatalf("fixture cold slot=%d, want deterministic LRU slot 1 after discovery slot 0", slot)
	}
	replacement := &softDeadlineUpstream{}
	up := httptest.NewServer(replacement)
	t.Cleanup(up.Close)
	p.BaseURL = up.URL + "/v1"
	primeSoftDeadlinePlanner(t, p, replacement)
	original.mu.Lock()
	replacement.mu.Lock()
	replacement.slots = original.slots
	replacement.mu.Unlock()
	original.mu.Unlock()
	_, warm := haloPiTurns()
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, softDeadlineRequest(t, warm, "600", false))
	replacement.mu.Lock()
	defer replacement.mu.Unlock()
	if !isColdRefusal(w) || len(replacement.dispatch) != 0 {
		t.Fatalf("cross-route counters earned credit: status=%d reject=%q calls=%+v", w.Code, w.Header().Get(deadlineadmit.HeaderReject), replacement.dispatch)
	}
}
