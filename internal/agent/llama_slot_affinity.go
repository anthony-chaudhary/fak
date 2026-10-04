package agent

import (
	"context"
	"encoding/json"
	"hash/crc32"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// llama_slot_affinity.go — prefix-affine slot pinning for llama-server upstreams.
//
// llama-server keeps one KV cache per slot. Without a hint it assigns an idle slot, so N
// subagents that share a parent's system prompt and tool catalog land on N different
// slots and each re-prefills the shared prefix. Pinning id_slot = crc32(stable prefix) %
// total_slots routes every request with the same prefix to the same slot, where
// cache_prompt:true reuses it. Measured on a Strix appliance: parent-first N=4 reuse went
// from 1/4 to 4/4 children.
//
// Safety: the hint is added ONLY after the upstream's /props answered with total_slots
// (a llama-server fact). Until then — and for vLLM, SGLang, or hosted OpenAI, which never
// answer it — the body is unchanged, because an unknown field can be rejected. Discovery
// runs in a background goroutine and never blocks a request. Keys the operator already set
// through ExtraBody or guided decode win. FAK_LLAMA_SLOT_AFFINITY=0 disables it.

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

func llamaSlotAffinityEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_LLAMA_SLOT_AFFINITY"))) {
	case "0", "off", "false", "no":
		return false
	}
	return true
}

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
		TotalSlots int `json:"total_slots"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&props) != nil {
		return 0
	}
	return props.TotalSlots
}

// llamaPrefixSlot hashes the stable request prefix — system/developer text plus the tool
// catalog, after prompt-prefix stabilization — to a slot in [0, slots).
func llamaPrefixSlot(messages []Message, tools []ToolDef, slots int64) int64 {
	h := crc32.NewIEEE()
	for _, m := range messages {
		if m.Role != RoleSystem && m.Role != "developer" {
			break
		}
		_, _ = h.Write([]byte(m.Content))
		_, _ = h.Write([]byte{0})
	}
	if len(tools) > 0 {
		if raw, err := json.Marshal(tools); err == nil {
			_, _ = h.Write(raw)
		}
	}
	return int64(h.Sum32()) % slots
}

// withLlamaSlotAffinity returns extra with id_slot and cache_prompt added when the
// upstream is a discovered llama-server; otherwise extra unchanged (and discovery kicked
// off in the background).
func (p *HTTPPlanner) withLlamaSlotAffinity(extra json.RawMessage, messages []Message, tools []ToolDef) json.RawMessage {
	if p == nil || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" || !llamaSlotAffinityEnabled() {
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
	if _, set := obj["id_slot"]; !set {
		slot, _ := json.Marshal(llamaPrefixSlot(messages, tools, slots))
		obj["id_slot"] = slot
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
