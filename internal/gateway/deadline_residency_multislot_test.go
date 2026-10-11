package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// fakeLlamaSlots is a llama-server upstream run with --parallel 4: a chat
// completion runs on the slot its id_slot names (else slot 0) and becomes that
// slot's next task; GET /slots reports every slot's last task and prompt.
type fakeLlamaSlots struct {
	mu        sync.Mutex
	nextID    int
	task      [4]int
	prompt    [4]int
	busy      [4]bool
	sent      []int // id_slot of each completion, -1 when none
	cacheSeen bool  // a prepared request used the discovered cache_prompt hint
}

func (f *fakeLlamaSlots) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/props":
		_ = json.NewEncoder(w).Encode(map[string]any{"total_slots": len(f.task)})
	case "/v1/chat/completions":
		var req struct {
			Messages    []agent.Message `json:"messages"`
			IDSlot      *int            `json:"id_slot"`
			CachePrompt bool            `json:"cache_prompt"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		f.mu.Lock()
		slot, sent := 0, -1
		if req.IDSlot != nil {
			slot, sent = *req.IDSlot, *req.IDSlot
		}
		f.sent = append(f.sent, sent)
		f.cacheSeen = req.CachePrompt
		f.nextID++
		f.task[slot] = f.nextID
		f.prompt[slot] = estimateMessageContentTokens(req.Messages)
		prompt := f.prompt[slot]
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "halo",
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": prompt, "completion_tokens": 1, "prompt_tokens_details": map[string]any{"cached_tokens": prompt}},
		})
	case "/slots":
		f.mu.Lock()
		defer f.mu.Unlock()
		out := make([]map[string]any, len(f.task))
		for i := range out {
			out[i] = map[string]any{"id": i, "id_task": f.task[i], "is_processing": f.busy[i], "n_prompt_tokens": f.prompt[i]}
		}
		_ = json.NewEncoder(w).Encode(out)
	default:
		http.NotFound(w, r)
	}
}

// slotSent returns the id_slot completion n (0-based) carried.
func (f *fakeLlamaSlots) slotSent(t *testing.T, n int) int {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if n >= len(f.sent) {
		t.Fatalf("completion %d not served (%d served)", n, len(f.sent))
	}
	return f.sent[n]
}

// runOnSlot runs another client's task on slot, evicting what it held.
func (f *fakeLlamaSlots) runOnSlot(slot int) {
	f.mu.Lock()
	f.nextID++
	f.task[slot] = f.nextID
	f.prompt[slot] = 300
	f.mu.Unlock()
}

func multiSlotHaloServer(t *testing.T, f *fakeLlamaSlots, softSlot bool) *Server {
	t.Helper()
	srv := haloPiDeadlineServer(t)
	up := httptest.NewServer(f)
	t.Cleanup(up.Close)
	p := agent.NewHTTPPlanner(up.URL+"/v1", "halo", "")
	p.LlamaSlotAffinity = true
	p.LlamaSoftSlot = softSlot
	// This policy fixture intentionally matches its 600-second client budget.
	// The separate prepared-call regression retains the shorter-watchdog refusal.
	srv.firstTokenWatchdog = func() time.Duration { return 600 * time.Second }
	// Prime real asynchronous capability publication outside the gateway, so
	// warmup cannot publish a different conversation's residency evidence.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for {
		if _, err := p.Complete(ctx, []agent.Message{{Role: "user", Content: "warmup"}}, nil); err != nil {
			t.Fatalf("prime multi-slot capability: %v", err)
		}
		f.mu.Lock()
		ready := f.cacheSeen && (!softSlot || f.sent[len(f.sent)-1] >= 0)
		f.mu.Unlock()
		if ready {
			break
		}
	}
	srv.planner = p
	return srv
}

func servedCount(f *fakeLlamaSlots) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sent)
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=500ms lane=default
func TestDeadlineAdmissionCreditsPinnedMultiSlotResidency(t *testing.T) {
	f := &fakeLlamaSlots{}
	srv := multiSlotHaloServer(t, f, true)
	turn1, turn2 := haloPiTurns()

	n := servedCount(f)
	if rec := postHaloChat(t, srv, "halo", turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d body=%s", rec.Code, rec.Body.String())
	}
	slot := f.slotSent(t, n)
	if slot < 0 {
		t.Fatal("turn 1 carried no id_slot")
	}
	awaitResidencyEntry(t, srv)
	rec := postHaloChat(t, srv, "halo", turn2, "600")
	if rec.Code != http.StatusOK || servedCount(f) != n+2 {
		t.Fatalf("warm turn 2: status=%d reject=%q; want admission on the pinned slot llama-server reports unchanged",
			rec.Code, rec.Header().Get(deadlineadmit.HeaderReject))
	}
	if got := f.slotSent(t, n+1); got != slot {
		t.Fatalf("credited turn 2 sent id_slot=%d, want the verified slot %d", got, slot)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=1s lane=default
func TestDeadlineAdmissionRefusesUnprovenMultiSlotResidency(t *testing.T) {
	turn1, turn2 := haloPiTurns()
	for _, tc := range []struct {
		name     string
		softSlot bool
		between  func(f *fakeLlamaSlots, slot int)
	}{
		{name: "unpinned", softSlot: false},
		{name: "slot_evicted", softSlot: true, between: func(f *fakeLlamaSlots, slot int) { f.runOnSlot(slot) }},
		{name: "slot_busy", softSlot: true, between: func(f *fakeLlamaSlots, slot int) {
			f.mu.Lock()
			f.busy[slot] = true
			f.mu.Unlock()
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeLlamaSlots{}
			srv := multiSlotHaloServer(t, f, tc.softSlot)
			n := servedCount(f)
			if rec := postHaloChat(t, srv, "halo", turn1, ""); rec.Code != http.StatusOK {
				t.Fatalf("turn 1: status=%d body=%s", rec.Code, rec.Body.String())
			}
			slot := f.slotSent(t, n)
			if tc.softSlot {
				awaitResidencyEntry(t, srv)
			}
			if tc.between != nil {
				tc.between(f, slot)
			}
			rec := postHaloChat(t, srv, "halo", turn2, "600")
			if !isColdRefusal(rec) || servedCount(f) != n+1 {
				t.Fatalf("turn 2: status=%d reject=%q; want cold refusal before inference",
					rec.Code, rec.Header().Get(deadlineadmit.HeaderReject))
			}
		})
	}
}
