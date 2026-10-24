package agent

import (
	"encoding/json"
	"strings"
)

// llama_prompt_progress.go asks a discovered llama-server for per-batch prefill reports (#13781).
//
// stream_prompt_progress.go re-arms the content-progress deadline on a rising
// prompt_progress.processed count, but only when the request carries return_progress:true.
// llama-server emits nothing while it prefills unless asked, so without this hint a cold prompt
// whose prefill outlasts the window (default 300s) is cut as a no-progress stall:
// [HW-WITNESSED] 2026-10-10 strix1 llama-server b1-dff6004, Qwen3.8-27B, a cold 104,020-token
// streamed prompt through fak serve at default flags was cut at 300.09s; with the hint it
// returned 200, finish_reason=stop, after 485.98s.
//
// return_progress is a llama-server field, so it rides the cache_prompt gate
// (llama_slot_affinity.go): an opted-in OpenAI-provider planner whose upstream answered /props
// with llama-server total_slots, on a streamed request only. An operator-set value wins.

// RecordLlamaServerSlots seeds the per-upstream llama-server discovery that gates the
// cache_prompt and return_progress hints, for a caller that already read a llama-server /props
// answer (the gateway's startup context-window probe), so a fresh gateway's first request is
// not sent before discovery. totalSlots must come from a real llama-server: a fak gateway's
// /props (tagged fak_total_slots_source) is not one.
func RecordLlamaServerSlots(base string, totalSlots int) {
	if totalSlots <= 0 || strings.TrimSpace(base) == "" {
		return
	}
	llamaSlotStateFor(base).slots.Store(int64(totalSlots))
}

// withLlamaPromptProgress returns extra with return_progress:true added for a streamed request
// to a discovered llama-server upstream; otherwise extra unchanged. It never probes:
// withLlamaSlotAffinity, called first, owns discovery.
func (p *HTTPPlanner) withLlamaPromptProgress(extra json.RawMessage, stream bool) json.RawMessage {
	if !stream || p == nil || !p.LlamaSlotAffinity || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return extra
	}
	if llamaSlotStateFor(p.BaseURL).slots.Load() <= 0 {
		return extra
	}
	obj := map[string]json.RawMessage{}
	if len(extra) > 0 && json.Unmarshal(extra, &obj) != nil {
		return extra
	}
	if _, set := obj["return_progress"]; set {
		return extra
	}
	obj["return_progress"] = json.RawMessage("true")
	out, err := json.Marshal(obj)
	if err != nil {
		return extra
	}
	return out
}
