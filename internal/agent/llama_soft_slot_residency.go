package agent

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const llamaSoftResidencyReadTimeout = 250 * time.Millisecond

// LlamaSoftSlotResidency binds a completed, measured prompt to its actual wire
// prefix, credential and named llama-server slot. Its contents are engine-owned.
type LlamaSoftSlotResidency struct {
	slot     int
	base     string
	url      string
	task     LlamaSlotTask
	envelope [sha256.Size]byte
	prefix   [sha256.Size]byte
	auth     [sha256.Size]byte
}

// LlamaSoftSlotTurn carries observation and an optional deadline verdict from
// ingress to the final prepared HTTP call. It never changes planner fields.
type LlamaSoftSlotTurn struct {
	planner  *HTTPPlanner
	base     string
	previous *LlamaSoftSlotResidency
	check    func(context.Context, []Message, LlamaSlotTask, bool) error
	mu       sync.Mutex
	latest   <-chan *LlamaSoftSlotResidency
}

type llamaSoftResidencyKey struct{}

// WithLlamaSoftSlotResidency observes the actual soft-slot dispatch. When check
// is nonnil, every POST attempt must pass it, even if preparation supplied no
// usable slot or proof. This lets the caller retain its ordinary cold verdict.
// Explicit operator pins/cache disabling and unknown upstreams remain untouched.
// Effective extra headers are ineligible: their routing/authentication meaning
// is not inferred or copied to the slots endpoint.
func (p *HTTPPlanner) WithLlamaSoftSlotResidency(ctx context.Context, previous *LlamaSoftSlotResidency, check func(context.Context, []Message, LlamaSlotTask, bool) error) (context.Context, *LlamaSoftSlotTurn) {
	if !p.softResidencyEligible() {
		return ctx, nil
	}
	t := &LlamaSoftSlotTurn{planner: p, base: p.BaseURL, previous: previous, check: check}
	return context.WithValue(ctx, llamaSoftResidencyKey{}, t), t
}

func (p *HTTPPlanner) softResidencyEligible() bool {
	if p == nil || !p.LlamaSlotAffinity || !p.LlamaSoftSlot || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" || llamaSlotStateFor(p.BaseURL).slots.Load() <= 0 {
		return false
	}
	var extra map[string]json.RawMessage
	if len(p.ExtraBody) > 0 && json.Unmarshal(p.ExtraBody, &extra) != nil {
		return false
	}
	if _, explicit := extra["id_slot"]; explicit {
		return false
	}
	if raw, explicit := extra["cache_prompt"]; explicit {
		var enabled bool
		if json.Unmarshal(raw, &enabled) != nil || !enabled {
			return false
		}
	}
	return true
}

// Record publishes only a healthy caller's measured completion. The bounded
// off-path read finishes before releasing the local slot hold. A failed call
// may be observed internally, but cannot become evidence without this handoff.
func (t *LlamaSoftSlotTurn) Record(promptTokens int, remember func(*LlamaSoftSlotResidency)) {
	if t == nil || promptTokens <= 0 || remember == nil {
		return
	}
	t.mu.Lock()
	done := t.latest
	t.mu.Unlock()
	if done == nil {
		return
	}
	go func() {
		if proof := <-done; proof != nil && proof.task.PromptTokens == promptTokens {
			remember(proof)
		}
	}()
}

type llamaSoftSlotLease struct {
	state  *llamaSoftSlots
	slot   int
	key    uint64
	free   func()
	closed bool // protected by state.mu
}

func (l *llamaSoftSlotLease) reserve() bool {
	if l == nil {
		return false
	}
	l.state.mu.Lock()
	defer l.state.mu.Unlock()
	if l.closed || l.state.inflight[l.slot] != 1 || l.state.slotOwner[l.slot] != l.key {
		return false
	}
	if held := l.state.reserved[l.slot]; held != nil && held != l {
		return false
	}
	if l.state.reserved == nil {
		l.state.reserved = make(map[int]*llamaSoftSlotLease)
	}
	l.state.reserved[l.slot] = l
	return true
}

func (l *llamaSoftSlotLease) release() {
	if l == nil {
		return
	}
	l.state.mu.Lock()
	if l.closed {
		l.state.mu.Unlock()
		return
	}
	l.closed = true
	if l.state.reserved[l.slot] == l {
		delete(l.state.reserved, l.slot)
	}
	l.state.mu.Unlock()
	l.free()
}

type llamaPreparedSlot struct {
	turn   *LlamaSoftSlotTurn
	lease  *llamaSoftSlotLease
	url    string
	auth   [sha256.Size]byte
	done   chan *LlamaSoftSlotResidency
	once   sync.Once
	last   *LlamaSoftSlotResidency
	apiKey string
	ctx    context.Context
}

func (c *upstreamCall) bindSoftResidency(ctx context.Context, p *HTTPPlanner, lease *llamaSoftSlotLease) {
	t, _ := ctx.Value(llamaSoftResidencyKey{}).(*LlamaSoftSlotTurn)
	if t == nil {
		return
	}
	b := &llamaPreparedSlot{turn: t, lease: lease, url: c.url, auth: sha256.Sum256([]byte(c.apiKey)), done: make(chan *LlamaSoftSlotResidency, 1), ctx: ctx}
	c.slotResidency = b
	c.slotRelease = b.release
	t.mu.Lock()
	t.latest = b.done
	t.mu.Unlock()
}

// admitSoftResidency runs after retry backoff, immediately before each POST.
// A changed target, credential, body or unusable slot still crosses check with
// no credit. Local ownership does not exclude foreign upstream clients; callers
// must retain their bounded cold-overrun envelope. A foreign client can replace
// a completed task before its observation, even with an equal prompt length;
// this local hold is not a server-side task lease.
func (c *upstreamCall) admitSoftResidency(ctx context.Context, p *HTTPPlanner) error {
	b := c.slotResidency
	if b == nil {
		return nil
	}
	b.last = nil
	var task LlamaSlotTask
	proven := false
	if b.lease != nil && p == b.turn.planner && p.BaseURL == b.turn.base && p.softResidencyEligible() && c.url == b.url && len(c.extraHeaders) == 0 && sha256.Sum256([]byte(c.apiKey)) == b.auth {
		envelope, chain, ok := llamaPreparedPrefix(c.body, b.lease.slot)
		if ok {
			b.apiKey = c.apiKey
			b.last = &LlamaSoftSlotResidency{slot: b.lease.slot, base: b.turn.base, url: c.url, envelope: envelope, prefix: chain[len(chain)-1], auth: b.auth}
			prev := b.turn.previous
			if prev != nil && prev.base == b.turn.base && prev.url == c.url && prev.slot == b.lease.slot && prev.envelope == envelope && prev.auth == b.auth && llamaPreparedContains(chain, prev.prefix) && b.lease.reserve() {
				task, ok = readLlamaNamedSlot(ctx, b.turn.base, c.apiKey, b.lease.slot)
				proven = ok && !task.Processing && task.IDTask == prev.task.IDTask && task.PromptTokens == prev.task.PromptTokens
			}
		}
	}
	if b.turn.check != nil {
		var prepared struct {
			Messages []Message `json:"messages"`
		}
		if json.Unmarshal(c.body, &prepared) != nil {
			proven = false
		}
		if err := b.turn.check(ctx, prepared.Messages, task, proven); err != nil {
			b.last = nil
			return err
		}
	}
	return nil
}

func (b *llamaPreparedSlot) release() {
	b.once.Do(func() {
		// Do not hold ordinary uncredited requests against the wait policy while
		// they execute. At completion, concurrent local users simply prevent a
		// new proof instead of changing their existing routing behavior.
		if b.last == nil || b.ctx.Err() != nil || !b.lease.reserve() {
			b.lease.release()
			close(b.done)
			return
		}
		proof := *b.last
		go func() {
			defer b.lease.release()
			defer close(b.done)
			task, ok := readLlamaNamedSlot(context.Background(), b.turn.base, b.apiKey, proof.slot)
			if ok && !task.Processing && task.PromptTokens > 0 {
				proof.task = task
				b.done <- &proof
			}
		}()
	})
}

func llamaPreparedContains(chain [][sha256.Size]byte, prefix [sha256.Size]byte) bool {
	for _, sum := range chain {
		if sum == prefix {
			return true
		}
	}
	return false
}

// Hash the actual serialized message prefix and remaining request envelope.
// Only transport streaming fields are excluded; other changed options may
// conservatively lose credit rather than silently alter a prompt template.
func llamaPreparedPrefix(body []byte, slot int) ([sha256.Size]byte, [][sha256.Size]byte, bool) {
	var obj map[string]json.RawMessage
	var messages []json.RawMessage
	var id int
	var cache bool
	if json.Unmarshal(body, &obj) != nil || json.Unmarshal(obj["id_slot"], &id) != nil || id != slot || json.Unmarshal(obj["cache_prompt"], &cache) != nil || !cache || json.Unmarshal(obj["messages"], &messages) != nil || len(messages) == 0 {
		return [sha256.Size]byte{}, nil, false
	}
	delete(obj, "messages")
	delete(obj, "stream")
	delete(obj, "stream_options")
	head, err := json.Marshal(obj)
	if err != nil {
		return [sha256.Size]byte{}, nil, false
	}
	chain := make([][sha256.Size]byte, len(messages))
	var previous [sha256.Size]byte
	for i, message := range messages {
		h := sha256.New()
		h.Write(previous[:])
		h.Write(message)
		copy(chain[i][:], h.Sum(nil))
		previous = chain[i]
	}
	return sha256.Sum256(head), chain, true
}

func readLlamaNamedSlot(ctx context.Context, base, apiKey string, id int) (LlamaSlotTask, bool) {
	ctx, cancel := context.WithTimeout(ctx, llamaSoftResidencyReadTimeout)
	defer cancel()
	base = strings.TrimSuffix(strings.TrimRight(strings.TrimSpace(base), "/"), "/v1")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	if err != nil {
		return LlamaSlotTask{}, false
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := llamaSlotHTTPClient.Do(req)
	if err != nil {
		return LlamaSlotTask{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return LlamaSlotTask{}, false
	}
	var slots []struct {
		ID     *int  `json:"id"`
		Task   *int  `json:"id_task"`
		Busy   *bool `json:"is_processing"`
		Prompt *int  `json:"n_prompt_tokens"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&slots) != nil {
		return LlamaSlotTask{}, false
	}
	seen := make(map[int]bool, len(slots))
	var task LlamaSlotTask
	found := false
	for _, s := range slots {
		if s.ID == nil || *s.ID < 0 || seen[*s.ID] {
			return LlamaSlotTask{}, false
		}
		seen[*s.ID] = true
		if *s.ID == id {
			if s.Task == nil || s.Busy == nil || s.Prompt == nil || *s.Task < 0 || *s.Prompt <= 0 {
				return LlamaSlotTask{}, false
			}
			task, found = LlamaSlotTask{IDTask: *s.Task, Processing: *s.Busy, PromptTokens: *s.Prompt}, true
		}
	}
	return task, found
}
