package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// llama_soft_slot.go — soft per-conversation llama-server slot choice.
//
// The hard pin (id_slot = crc32(shared prefix) % slots) restored prefix reuse but serialized
// every same-prefix turn onto one slot and cost 3x throughput (llama_slot_affinity.go). The
// soft choice pins only to an IDLE slot: a conversation's next turn goes back to the slot
// that holds its KV when that slot is free, a new conversation takes the least recently used
// idle slot (so it does not evict a live conversation's KV), and when every slot is busy no
// id_slot is sent and llama-server's own choice applies. A busy slot is never named, so no
// turn queues behind another.
//
// Idle is read from a background snapshot of llama-server's GET /slots (is_processing) plus
// this process's own in-flight pins, which cover the window before llama-server marks a
// just-sent request. llama-server answers /slots from its task loop, so the call can wait out
// a whole prefill batch; polling it off the request path keeps that wait out of every turn.
// llama-server does not report which slot served an unpinned request, so the conversation
// map only learns from pins it made itself.

const (
	llamaSlotsColdReadTimeout = 250 * time.Millisecond
	llamaSlotsPollTimeout     = 2 * time.Second
	llamaSlotsPollInterval    = 200 * time.Millisecond
	llamaSlotsSnapshotMaxAge  = 2 * time.Second
	llamaSlotsPollIdleStop    = 30 * time.Second
	llamaSoftSlotMaxConvs     = 4096
)

// Soft slot policies. LRU moves a conversation whose own slot is busy to the least recently
// used idle slot. Wait never moves a known conversation: it pins its own slot even when busy,
// and llama-server hands a freed slot to a task pinned to it first. On strix2 (6 agents, 4
// slots, private llama-server) LRU's moves cut warm-turn reuse from 99.3% to 82.8%.
const (
	SoftSlotPolicyLRU  = "lru"
	SoftSlotPolicyWait = "wait"
)

// ValidateSoftSlotPolicy rejects a policy outside the closed vocabulary. Empty means lru.
func ValidateSoftSlotPolicy(policy string) error {
	switch policy {
	case "", SoftSlotPolicyLRU, SoftSlotPolicyWait:
		return nil
	}
	return fmt.Errorf("unknown llama soft slot policy %q (want %s or %s)", policy, SoftSlotPolicyLRU, SoftSlotPolicyWait)
}

// Soft slot outcomes, the closed vocabulary of LlamaSoftSlotCounts.
const (
	SoftSlotSticky      = "sticky"      // the conversation's own slot was idle and was pinned
	SoftSlotWaited      = "waited"      // policy wait: the conversation's own slot was busy and was pinned anyway
	SoftSlotAssigned    = "assigned"    // new (or displaced) conversation pinned to the LRU idle slot
	SoftSlotBusy        = "busy"        // no idle slot; id_slot omitted, upstream chooses
	SoftSlotUnavailable = "unavailable" // no fresh /slots snapshot; id_slot omitted
)

// LlamaSoftSlotOutcomes lists the outcome vocabulary in render order.
func LlamaSoftSlotOutcomes() []string {
	return []string{SoftSlotSticky, SoftSlotWaited, SoftSlotAssigned, SoftSlotBusy, SoftSlotUnavailable}
}

var llamaSoftSlotCounts = map[string]*atomic.Uint64{
	SoftSlotSticky:      {},
	SoftSlotWaited:      {},
	SoftSlotAssigned:    {},
	SoftSlotBusy:        {},
	SoftSlotUnavailable: {},
}

// LlamaSoftSlotCounts reports the process-wide soft slot decisions by outcome.
func LlamaSoftSlotCounts() map[string]uint64 {
	out := make(map[string]uint64, len(llamaSoftSlotCounts))
	for k, v := range llamaSoftSlotCounts {
		out[k] = v.Load()
	}
	return out
}

type softConv struct {
	slot int
	used time.Time
}

type llamaSoftSlots struct {
	mu        sync.Mutex
	convs     map[uint64]softConv
	slotOwner map[int]uint64
	slotUsed  map[int]time.Time
	inflight  map[int]int
	// released records when this process's last pin on a slot ended: a snapshot taken
	// before that instant may show the slot busy with our own finished request.
	released map[int]time.Time

	snapIDs    []int
	snapBusy   map[int]bool
	snapAt     time.Time
	polling    bool
	lastDemand time.Time
}

var llamaSoftSlotRegistry sync.Map

func llamaSoftSlotsFor(base string) *llamaSoftSlots {
	if v, ok := llamaSoftSlotRegistry.Load(base); ok {
		return v.(*llamaSoftSlots)
	}
	v, _ := llamaSoftSlotRegistry.LoadOrStore(base, &llamaSoftSlots{
		convs:     map[uint64]softConv{},
		slotOwner: map[int]uint64{},
		slotUsed:  map[int]time.Time{},
		inflight:  map[int]int{},
		released:  map[int]time.Time{},
	})
	return v.(*llamaSoftSlots)
}

// conversationKey identifies a conversation by its system text and first user turn: stable
// across that conversation's turns, distinct between agents that share a system prompt.
func conversationKey(messages []Message) (uint64, bool) {
	h := fnv.New64a()
	var user bool
	for _, m := range messages {
		switch m.Role {
		case RoleSystem, "developer":
			_, _ = io.WriteString(h, m.Role)
			_, _ = io.WriteString(h, m.Content)
		case RoleUser:
			_, _ = io.WriteString(h, "user")
			_, _ = io.WriteString(h, m.Content)
			user = true
		}
		if user {
			break
		}
	}
	return h.Sum64(), user
}

// readLlamaBusySlots returns the slot ids llama-server reports as processing, and all ids.
func readLlamaBusySlots(base, apiKey string, timeout time.Duration) (ids []int, busy map[int]bool, ok bool) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	base = strings.TrimSuffix(strings.TrimRight(strings.TrimSpace(base), "/"), "/v1")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	if err != nil {
		return nil, nil, false
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := llamaSlotHTTPClient.Do(req)
	if err != nil {
		return nil, nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, nil, false
	}
	var slots []struct {
		ID           int  `json:"id"`
		IsProcessing bool `json:"is_processing"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&slots) != nil || len(slots) == 0 {
		return nil, nil, false
	}
	busy = make(map[int]bool, len(slots))
	for _, s := range slots {
		ids = append(ids, s.ID)
		busy[s.ID] = s.IsProcessing
	}
	sort.Ints(ids)
	return ids, busy, true
}

// pick chooses the slot for one turn, or -1 to leave the choice to llama-server. A non-nil
// release must be called when the upstream request ends.
func (st *llamaSoftSlots) pick(base, apiKey, policy string, messages []Message, now time.Time) (int, func(), string) {
	key, ok := conversationKey(messages)
	if !ok {
		return -1, nil, ""
	}
	st.mu.Lock()
	st.lastDemand = now
	// A stale snapshot (the poller exits after llamaSlotsPollIdleStop without demand) is as
	// good as none: re-read it, or the first turn after an idle spell would go unpinned.
	cold := st.snapAt.IsZero() || now.Sub(st.snapAt) > llamaSlotsSnapshotMaxAge
	st.startPollLocked(base, apiKey)
	st.mu.Unlock()
	if cold {
		// One bounded synchronous read so the first turn after start can pin.
		if ids, busy, ok := readLlamaBusySlots(base, apiKey, llamaSlotsColdReadTimeout); ok {
			st.storeSnapshot(ids, busy, time.Now())
		}
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.snapAt.IsZero() || now.Sub(st.snapAt) > llamaSlotsSnapshotMaxAge {
		return -1, nil, SoftSlotUnavailable
	}
	ids, busy, snapAt := st.snapIDs, st.snapBusy, st.snapAt
	idle := func(id int) bool {
		if st.inflight[id] > 0 {
			return false
		}
		rel, ok := st.released[id]
		return !busy[id] || (ok && !snapAt.After(rel))
	}
	outcome := SoftSlotAssigned
	slot := -1
	c, had := st.convs[key]
	owned := had && st.slotOwner[c.slot] == key
	switch {
	case owned && idle(c.slot):
		slot, outcome = c.slot, SoftSlotSticky
	case owned && policy == SoftSlotPolicyWait:
		slot, outcome = c.slot, SoftSlotWaited
	default:
		for _, id := range ids {
			if !idle(id) {
				continue
			}
			if slot < 0 || st.slotUsed[id].Before(st.slotUsed[slot]) {
				slot = id
			}
		}
	}
	if slot < 0 {
		delete(st.convs, key)
		return -1, nil, SoftSlotBusy
	}
	if prev, owned := st.slotOwner[slot]; owned && prev != key {
		delete(st.convs, prev)
	}
	st.slotOwner[slot] = key
	st.slotUsed[slot] = now
	st.convs[key] = softConv{slot: slot, used: now}
	st.inflight[slot]++
	st.trimLocked()
	var once sync.Once
	release := func() {
		once.Do(func() {
			st.mu.Lock()
			if st.inflight[slot] > 0 {
				st.inflight[slot]--
			}
			st.released[slot] = time.Now()
			st.mu.Unlock()
		})
	}
	return slot, release, outcome
}

func (st *llamaSoftSlots) storeSnapshot(ids []int, busy map[int]bool, at time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if at.After(st.snapAt) {
		st.snapIDs, st.snapBusy, st.snapAt = ids, busy, at
	}
}

// startPollLocked runs one background /slots poller per upstream while turns keep arriving;
// it exits after llamaSlotsPollIdleStop without demand.
func (st *llamaSoftSlots) startPollLocked(base, apiKey string) {
	if st.polling {
		return
	}
	st.polling = true
	go func() {
		t := time.NewTicker(llamaSlotsPollInterval)
		defer t.Stop()
		for range t.C {
			st.mu.Lock()
			if time.Since(st.lastDemand) > llamaSlotsPollIdleStop {
				st.polling = false
				st.mu.Unlock()
				return
			}
			st.mu.Unlock()
			taken := time.Now()
			if ids, busy, ok := readLlamaBusySlots(base, apiKey, llamaSlotsPollTimeout); ok {
				st.storeSnapshot(ids, busy, taken)
			}
		}
	}()
}

func (st *llamaSoftSlots) trimLocked() {
	if len(st.convs) <= llamaSoftSlotMaxConvs {
		return
	}
	var oldest uint64
	var oldestAt time.Time
	for k, c := range st.convs {
		if oldestAt.IsZero() || c.used.Before(oldestAt) {
			oldest, oldestAt = k, c.used
		}
	}
	delete(st.convs, oldest)
}

// withLlamaSoftSlot adds id_slot for an opted-in planner when soft choice picks an idle
// slot. It returns extra unchanged (and a nil release) otherwise, and never overrides an
// id_slot the operator set through ExtraBody.
func (p *HTTPPlanner) withLlamaSoftSlot(extra json.RawMessage, messages []Message) (json.RawMessage, func()) {
	if p == nil || !p.LlamaSoftSlot || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return extra, nil
	}
	if st := llamaSlotStateFor(p.BaseURL); st.slots.Load() <= 0 {
		st.maybeProbe(p.BaseURL, p.effectiveAPIKey())
		return extra, nil
	}
	obj := map[string]json.RawMessage{}
	if len(extra) > 0 && json.Unmarshal(extra, &obj) != nil {
		return extra, nil
	}
	if _, set := obj["id_slot"]; set {
		return extra, nil
	}
	slot, release, outcome := llamaSoftSlotsFor(p.BaseURL).pick(p.BaseURL, p.effectiveAPIKey(), p.LlamaSoftSlotPolicy, messages, time.Now())
	if c, ok := llamaSoftSlotCounts[outcome]; ok {
		c.Add(1)
	}
	if slot < 0 {
		return extra, nil
	}
	raw, _ := json.Marshal(slot)
	obj["id_slot"] = raw
	out, err := json.Marshal(obj)
	if err != nil {
		release()
		return extra, nil
	}
	return out, release
}
