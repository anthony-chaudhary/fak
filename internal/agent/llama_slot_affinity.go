package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// llama_slot_affinity.go — advisory llama-server prompt-cache hint.
//
// The pin this hint replaced (id_slot = crc32(shared prefix) % total_slots) lost 3.00x fleet
// throughput: [HW-WITNESSED] 2026-10-05 on a Strix Halo (strix1), four concurrent 128-token
// requests aggregated 41.65 tok/s straight at the llama-server upstream versus 13.89 tok/s
// through the fak front door, with a per-slot latency staircase of 7.37 -> 22.86 -> 29.85 ->
// 36.86 s — every turn sharing a system prompt hashed onto one slot, and llama-server serves
// one request per slot.
//
// The pin was originally introduced for the opposite trade (commit cbd40151e): with a
// parent-first fan-out of N=4, prefix reuse went 1/4 -> 4/4 for the children and the slowest
// child fell 62s -> 4s. Throughput beat that latency win here, so the pin is gone.
//
// Shipped behavior: the hint is advisory — it adds cache_prompt:true to a discovered
// llama-server upstream and emits no id_slot, so the upstream's own slot selection applies and
// concurrent same-prefix turns are not serialized. fak therefore no longer guarantees prefix
// affinity, and slot choice is the upstream's. An operator who wants the old hard pin can
// still set id_slot explicitly through ExtraBody. total_slots still gates the hint, and any
// key the operator already set through ExtraBody or guided decode wins.
//
// Opt-in: only a planner with LlamaSlotAffinity set ever probes. The zero value — every
// generic OpenAI-compatible client, hosted provider, and test fake upstream — sends no
// extra /props request; `fak serve` sets it for its proxy upstream (--llama-slot-affinity).
//
// Safety: the hint is added ONLY after the upstream's /props answered with total_slots
// (a llama-server fact). Until then — and for vLLM, SGLang, or hosted OpenAI, which never
// answer it — the body is unchanged, because an unknown field can be rejected. A fak
// gateway also answers /props with total_slots (its admission cap, tagged
// fak_total_slots_source); that answer is not a llama-server slot and is ignored. Discovery
// runs in a background goroutine and never blocks a request. Keys the operator already set
// through ExtraBody or guided decode win.

const (
	llamaSlotProbeTimeout = 2 * time.Second
	llamaSlotRetryAfter   = time.Minute
)

type llamaSlotState struct {
	slots     atomic.Int64 // >0 once discovered
	probing   atomic.Bool
	nextProbe atomic.Int64 // unix nanos; 0 = probe now
}

// llamaSlotRegistry caches discovered slot counts per upstream base URL: a server
// property shared by every planner that targets it.
var llamaSlotRegistry sync.Map

// llamaSlotHTTPClient is the bounded discovery client; tests may replace it.
var llamaSlotHTTPClient = &http.Client{Timeout: llamaSlotProbeTimeout}

// llamaPropsURL maps an OpenAI-compatible base (http://h:8080/v1) to llama-server's
// root /props endpoint.
func llamaPropsURL(base string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	base = strings.TrimSuffix(base, "/v1")
	return base + "/props"
}

func llamaSlotStateFor(base string) *llamaSlotState {
	if v, ok := llamaSlotRegistry.Load(base); ok {
		return v.(*llamaSlotState)
	}
	v, _ := llamaSlotRegistry.LoadOrStore(base, &llamaSlotState{})
	return v.(*llamaSlotState)
}

// maybeProbe starts at most one background /props discovery per upstream.
func (st *llamaSlotState) maybeProbe(base, apiKey string) {
	if st.slots.Load() > 0 || time.Now().UnixNano() < st.nextProbe.Load() {
		return
	}
	if !st.probing.CompareAndSwap(false, true) {
		return
	}
	go func() {
		defer st.probing.Store(false)
		n := probeLlamaTotalSlots(base, apiKey)
		if n > 0 {
			st.slots.Store(int64(n))
			return
		}
		st.nextProbe.Store(time.Now().Add(llamaSlotRetryAfter).UnixNano())
	}()
}

func probeLlamaTotalSlots(base, apiKey string) int {
	ctx, cancel := context.WithTimeout(context.Background(), llamaSlotProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, llamaPropsURL(base), nil)
	if err != nil {
		return 0
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := llamaSlotHTTPClient.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
	}
	var props struct {
		TotalSlots     int    `json:"total_slots"`
		FakSlotsSource string `json:"fak_total_slots_source"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&props) != nil {
		return 0
	}
	if props.FakSlotsSource != "" {
		return 0
	}
	return props.TotalSlots
}

// withLlamaSlotAffinity returns extra with cache_prompt:true added when the planner opted
// in and the upstream is a discovered llama-server; otherwise extra unchanged (an opted-in
// planner kicks discovery off in the background). It never writes id_slot, so slot choice
// stays with the upstream and concurrent turns keep batching (see the file doc).
func (p *HTTPPlanner) withLlamaSlotAffinity(extra json.RawMessage) json.RawMessage {
	if p == nil || !p.LlamaSlotAffinity || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return extra
	}
	st := llamaSlotStateFor(p.BaseURL)
	slots := st.slots.Load()
	if slots <= 0 {
		st.maybeProbe(p.BaseURL, p.effectiveAPIKey())
		return extra
	}
	obj := map[string]json.RawMessage{}
	if len(extra) > 0 && json.Unmarshal(extra, &obj) != nil {
		return extra
	}
	if _, set := obj["cache_prompt"]; !set {
		obj["cache_prompt"] = json.RawMessage("true")
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return extra
	}
	return out
}
