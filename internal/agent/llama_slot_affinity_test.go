package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func slotTestBody(t *testing.T, p *HTTPPlanner, msgs []Message, tools []ToolDef) map[string]json.RawMessage {
	t.Helper()
	call, err := p.prepareUpstream(msgs, tools, true)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(call.body, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

func waitSlotsDiscovered(t *testing.T, base string, want int64) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if llamaSlotStateFor(base).slots.Load() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("slot discovery for %s never reached %d", base, want)
}

// TestLlamaSlotAffinityPinsSharedPrefixToOneSlot: once the upstream /props reports
// total_slots, every request sharing a system prompt + tool catalog carries the same
// id_slot (= crc32(prefix) % slots) and cache_prompt:true, so sibling subagents reuse one
// slot's KV. Before discovery the body is untouched (discovery never blocks a request).
//
// fak-test:runtime fast est=300ms
func TestLlamaSlotAffinityPinsSharedPrefixToOneSlot(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/props" {
			_, _ = w.Write([]byte(`{"total_slots":4}`))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(ts.Close)
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: ts.URL + "/v1", ModelID: "m", LlamaSlotAffinity: true}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "read_file"}}}
	parent := []Message{{Role: RoleSystem, Content: "shared harness prompt"}, {Role: RoleUser, Content: "coordinate"}}
	child := []Message{{Role: RoleSystem, Content: "shared harness prompt"}, {Role: RoleUser, Content: "subtask 3"}}

	if body := slotTestBody(t, p, parent, tools); body["id_slot"] != nil || body["cache_prompt"] != nil {
		t.Fatalf("hint sent before /props discovery: %s %s", body["id_slot"], body["cache_prompt"])
	}
	waitSlotsDiscovered(t, p.BaseURL, 4)

	a := slotTestBody(t, p, parent, tools)
	b := slotTestBody(t, p, child, tools)
	want, _ := json.Marshal(llamaPrefixSlot(parent, tools, 4))
	if string(a["id_slot"]) != string(want) || string(b["id_slot"]) != string(want) {
		t.Fatalf("id_slot parent=%s child=%s, want both %s", a["id_slot"], b["id_slot"], want)
	}
	if string(a["cache_prompt"]) != "true" {
		t.Fatalf("cache_prompt=%s want true", a["cache_prompt"])
	}

	// An operator-set id_slot (ExtraBody) wins over the hint.
	p.ExtraBody = json.RawMessage(`{"id_slot":3}`)
	if got := slotTestBody(t, p, parent, tools)["id_slot"]; string(got) != "3" {
		t.Fatalf("operator id_slot overridden: %s", got)
	}
}

// TestLlamaSlotAffinityOmittedForNonLlamaUpstream: an upstream without /props total_slots
// (vLLM, hosted OpenAI) never receives the llama-only fields.
//
// fak-test:runtime fast est=300ms
func TestLlamaSlotAffinityOmittedForNonLlamaUpstream(t *testing.T) {
	probed := make(chan struct{}, 4)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/props" {
			probed <- struct{}{}
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(ts.Close)
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: ts.URL + "/v1", ModelID: "m", LlamaSlotAffinity: true}
	msgs := []Message{{Role: RoleSystem, Content: "s"}, {Role: RoleUser, Content: "u"}}
	_ = slotTestBody(t, p, msgs, nil)
	select {
	case <-probed:
	case <-time.After(5 * time.Second):
		t.Fatal("/props never probed")
	}
	for !llamaSlotStateFor(p.BaseURL).probing.CompareAndSwap(false, false) {
		time.Sleep(10 * time.Millisecond)
	}
	if body := slotTestBody(t, p, msgs, nil); body["id_slot"] != nil || body["cache_prompt"] != nil {
		t.Fatalf("llama-only fields sent to a non-llama upstream: %s %s", body["id_slot"], body["cache_prompt"])
	}
}

// TestLlamaSlotAffinityOffByDefaultNeverProbes: a planner that did not opt in sends no
// /props request, so a generic OpenAI-compatible upstream (and every test fake that counts
// requests or asserts the path) sees only the chat call.
//
// fak-test:runtime fast est=50ms
func TestLlamaSlotAffinityOffByDefaultNeverProbes(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected upstream request %s %s", r.Method, r.URL.Path)
		http.NotFound(w, r)
	}))
	t.Cleanup(ts.Close)
	p := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: ts.URL + "/v1", ModelID: "m"}
	msgs := []Message{{Role: RoleSystem, Content: "s"}, {Role: RoleUser, Content: "u"}}
	if body := slotTestBody(t, p, msgs, nil); body["id_slot"] != nil || body["cache_prompt"] != nil {
		t.Fatalf("llama-only fields sent without opt-in: %s %s", body["id_slot"], body["cache_prompt"])
	}
	if _, started := llamaSlotRegistry.Load(p.BaseURL); started {
		t.Fatal("slot discovery started for a planner that did not opt in")
	}
}

// TestLlamaSlotAffinityIgnoresFakGatewayProps: a fak gateway answers /props with its
// admission cap as total_slots, tagged fak_total_slots_source. That is not a llama-server
// slot, so an opted-in planner pointed at a fak gateway never pins.
//
// fak-test:runtime fast est=50ms
func TestLlamaSlotAffinityIgnoresFakGatewayProps(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"total_slots":4,"fak_total_slots_source":"gateway AdmissionPolicy.MaxNumSeqs"}`))
	}))
	t.Cleanup(ts.Close)
	if got := probeLlamaTotalSlots(ts.URL+"/v1", ""); got != 0 {
		t.Fatalf("fak gateway /props read as %d llama slots, want 0", got)
	}
}
