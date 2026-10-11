package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// LlamaSlotTask is what llama-server's GET /slots reports about one slot: the
// last task it ran, whether it is running one now, and that task's prompt length.
// A task id that has not moved proves no other request has used the slot.
type LlamaSlotTask struct {
	ID           int
	IDTask       int
	Processing   bool
	PromptTokens int
}

func (p *HTTPPlanner) readLlamaSlotTasks(ctx context.Context, timeout time.Duration) ([]LlamaSlotTask, bool) {
	if p == nil || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return nil, false
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	base := strings.TrimSuffix(strings.TrimRight(strings.TrimSpace(p.BaseURL), "/"), "/v1")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	if err != nil {
		return nil, false
	}
	if key := p.effectiveAPIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := llamaSlotHTTPClient.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	var slots []struct {
		ID           int  `json:"id"`
		IDTask       *int `json:"id_task"`
		IsProcessing bool `json:"is_processing"`
		NPrompt      int  `json:"n_prompt_tokens"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&slots) != nil {
		return nil, false
	}
	out := make([]LlamaSlotTask, 0, len(slots))
	for _, s := range slots {
		if s.IDTask == nil {
			return nil, false
		}
		out = append(out, LlamaSlotTask{ID: s.ID, IDTask: *s.IDTask, Processing: s.IsProcessing, PromptTokens: s.NPrompt})
	}
	return out, true
}

// LlamaSingleSlot reads the upstream's /slots within timeout. ok is false unless
// the upstream reports exactly one slot carrying an id_task: with several slots
// the slot an unpinned request lands on is not known before it runs.
func (p *HTTPPlanner) LlamaSingleSlot(ctx context.Context, timeout time.Duration) (LlamaSlotTask, bool) {
	if p == nil || !p.LlamaSlotAffinity || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return LlamaSlotTask{}, false
	}
	// OpenAI compatibility alone does not identify a llama-server. Reuse the
	// capability learned by opt-in, asynchronous /props discovery; an unknown
	// or multi-slot upstream cannot supply single-slot residency evidence.
	if llamaSlotStateFor(p.BaseURL).slots.Load() != 1 {
		return LlamaSlotTask{}, false
	}
	slots, ok := p.readLlamaSlotTasks(ctx, timeout)
	if !ok || len(slots) != 1 {
		return LlamaSlotTask{}, false
	}
	return slots[0], true
}

// LlamaSlot reads the upstream's /slots within timeout and returns the slot with
// id. ok is false unless the planner opted into soft slot choice, /props
// discovery already found a llama-server, and /slots lists that slot with an
// id_task.
func (p *HTTPPlanner) LlamaSlot(ctx context.Context, timeout time.Duration, id int) (LlamaSlotTask, bool) {
	if p == nil || !p.LlamaSoftSlot || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" ||
		llamaSlotStateFor(p.BaseURL).slots.Load() <= 0 {
		return LlamaSlotTask{}, false
	}
	slots, ok := p.readLlamaSlotTasks(ctx, timeout)
	if !ok {
		return LlamaSlotTask{}, false
	}
	for _, s := range slots {
		if s.ID == id {
			return s, true
		}
	}
	return LlamaSlotTask{}, false
}

// LlamaSlotPin carries one chat turn's llama-server slot between the gateway's
// deadline admission and the upstream request. A pin built with a slot >= 0 sends
// the turn to that slot, the one whose residency admission verified; Served
// reports the id_slot the request actually carried.
type LlamaSlotPin struct {
	want int

	mu     sync.Mutex
	served int
	sent   bool
}

// NewLlamaSlotPin returns a pin that sends the turn to slot, or, for slot < 0,
// only records the slot the turn was sent to.
func NewLlamaSlotPin(slot int) *LlamaSlotPin {
	return &LlamaSlotPin{want: slot, served: -1}
}

// Served returns the id_slot the turn's upstream request carried; ok is false
// when no request was sent or it named no slot.
func (p *LlamaSlotPin) Served() (int, bool) {
	if p == nil {
		return -1, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.served, p.sent && p.served >= 0
}

type llamaSlotPinCtxKey struct{}

// ContextWithLlamaSlotPin attaches pin to the chat requests made under ctx.
func ContextWithLlamaSlotPin(ctx context.Context, pin *LlamaSlotPin) context.Context {
	return context.WithValue(ctx, llamaSlotPinCtxKey{}, pin)
}

// withContextLlamaSlotPin appends ctx's slot pin to opts without touching the
// caller's backing array.
func withContextLlamaSlotPin(ctx context.Context, opts []SampleOpt) []SampleOpt {
	pin, _ := ctx.Value(llamaSlotPinCtxKey{}).(*LlamaSlotPin)
	if pin == nil {
		return opts
	}
	out := make([]SampleOpt, 0, len(opts)+1)
	out = append(out, opts...)
	return append(out, func(sp *SampleParams) { sp.llamaSlotPin = pin })
}

// apply sends the turn to the pinned slot, replacing a different soft slot
// choice (whose release runs now, as the request will not use that slot), and
// records the id_slot the request carries.
func (p *LlamaSlotPin) apply(extra json.RawMessage, release func()) (json.RawMessage, func()) {
	if p == nil {
		return extra, release
	}
	obj := map[string]json.RawMessage{}
	if len(extra) > 0 && json.Unmarshal(extra, &obj) != nil {
		return extra, release
	}
	served := -1
	if raw, set := obj["id_slot"]; set && json.Unmarshal(raw, &served) != nil {
		served = -1
	}
	if p.want >= 0 && served != p.want {
		raw, _ := json.Marshal(p.want)
		obj["id_slot"] = raw
		if out, err := json.Marshal(obj); err == nil {
			if release != nil {
				release()
				release = nil
			}
			extra, served = out, p.want
		}
	}
	p.mu.Lock()
	p.served, p.sent = served, true
	p.mu.Unlock()
	return extra, release
}
