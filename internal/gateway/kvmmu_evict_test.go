package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// kvmmu_evict_test.go is the gateway-level integration witness for issue #579: the
// model-side KV-quarantine eviction BRIDGE (internal/kvmmu) wired onto the LIVE serve path.
// Unlike poison_evict_test.go (which drives a recordingEvictor MOCK that only records the
// hook fired), these tests drive the REAL agent.InKernelPlanner — a real model.Session over
// a real model.Model, a real tokenizer — through the gateway's admitInboundResults path, so
// a quarantine verdict on a real tool result triggers a REAL model.KVCache.Evict and the
// bit-exact never-saw invariant is asserted end-to-end on the live (non-synthetic-mock) path.

// byteLevelRunes reproduces the GPT-2 byte-level alphabet the tokenizer uses (the same
// mapping internal/tokenizer.makeByteLevelDecode builds): printable ASCII !..~ plus the two
// Latin-1 ranges map to themselves, every other byte to 256,257,... in byte order. Building a
// fixture vocab from this lets a small FromGGML tokenizer encode any ASCII transcript into
// single-char byte tokens with no merge fixture to maintain.
func byteLevelRunes() map[byte]rune {
	var printable []byte
	for b := int('!'); b <= int('~'); b++ {
		printable = append(printable, byte(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		printable = append(printable, byte(b))
	}
	for b := 0xAE; b <= 0xFF; b++ {
		printable = append(printable, byte(b))
	}
	seen := make(map[byte]bool, 256)
	out := make(map[byte]rune, 256)
	for _, b := range printable {
		seen[b] = true
		out[b] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if seen[byte(b)] {
			continue
		}
		out[byte(b)] = rune(256 + n)
		n++
	}
	return out
}

// newByteLevelTokenizer builds a real tokenizer.Tokenizer (via FromGGML) over a single-char
// byte-level vocab plus the two ChatML control tokens renderTranscript emits (<|im_start|>,
// <|im_end|>). Every byte gets its own id, so any ASCII transcript encodes losslessly; one
// dummy merge satisfies FromGGML's non-empty-merges check (it never fires — every symbol is a
// single byte char already in vocab).
func newByteLevelTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	enc := byteLevelRunes()
	var tokens []string
	for b := 0; b < 256; b++ {
		tokens = append(tokens, string(enc[byte(b)]))
	}
	// ChatML control tokens (their "<|...|>" shape is recognized as special by FromGGML).
	imStart, imEnd := len(tokens), len(tokens)+1
	tokens = append(tokens, "<|im_start|>", "<|im_end|>")
	tokenTypes := make([]int32, len(tokens))
	tokenTypes[imStart] = 3 // CONTROL
	tokenTypes[imEnd] = 3
	// A single dummy merge of two byte chars that never co-occur as the only path (BPE picks
	// it only if both appear adjacent AND no single-char alternative — but every char is a
	// valid single token, so the merge is inert; it exists solely to pass the merges check).
	merges := []string{string(enc['x']) + " " + string(enc['y'])}
	tok, err := tokenizer.FromGGML(tokens, merges, tokenTypes, "")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	return tok
}

// kvmmuSynthCfg is a tiny Llama-shaped config: the eviction-equals-never property is
// structural (true for any weights), so a synthetic checkpoint faithfully witnesses the live
// WIRING on a box with no GGUF export — the same posture internal/kvmmu's unit witness takes.
// VocabSize must cover every byte id the byte-level tokenizer can emit (256 + 2 control).
func kvmmuSynthCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 260, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}
}

// liveInKernelPlanner builds a REAL agent.InKernelPlanner over a real synthetic model + real
// byte-level tokenizer with the #579 KV-MMU span-eviction bridge opted IN. This is the live,
// non-mock chat backend the gateway serves with `fak serve --gguf`, exercised here with no
// external weights.
func liveInKernelPlanner(t *testing.T) *agent.InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_KVMMU", "on")
	t.Setenv("FAK_INKERNEL_RADIX", "off") // isolate the SPAN bridge from the prefix-cache hook
	m := model.NewSynthetic(kvmmuSynthCfg())
	// NewInKernelPlanner serves the Q8_0 forward (quant=true), so the kvmmu session it builds
	// runs Q8 too — quantize the synthetic model so the served decode path is exercised. The
	// f32 KV cache (and thus Evict's re-RoPE + renumber) is identical either way, so the
	// reposition stays bit-exact and the two Q8 sessions decode identical logits.
	m.Quantize()
	return agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "synthetic-live", false, nil)
}

func newKVMMUResultStackServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(ifc.NewLedger(), ifc.Policy{}))
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// TestLiveKVSpanEvictionBitExact is the #579 deliverable: a poisoned tool result, admitted
// through the gateway's LIVE serve path with a REAL in-kernel planner, drives a real
// model.KVCache.Evict of that result's K/V span, and the resulting cache is BIT-IDENTICAL to a
// session that never saw the poison (the reposition invariant) — asserted end-to-end, not via
// a mock recorder.
func TestLiveKVSpanEvictionBitExact(t *testing.T) {
	srv := newKVMMUResultStackServer(t)
	planner := liveInKernelPlanner(t)
	srv.planner = planner

	const secret = "sk-abcdef0123456789abcdef0123"
	poison := `{"page":"config loaded. api_key=` + secret + ` was found in env"}`
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "look up the config"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: poison},
	}

	// Drive the LIVE gateway result-side path: admit each tool result, quarantine the poison,
	// and fire the in-kernel eviction hooks (evictInKernelPoison -> KVSpanEvictor.EvictKVSpan).
	// This must not error or panic on the real planner.
	admissions, err := srv.admitInboundResults(context.Background(), messages, "trace-live-579")
	if err != nil {
		t.Fatalf("admitInboundResults (live path): %v", err)
	}
	quarantined := false
	for _, a := range admissions {
		if a.Verdict.Kind == "QUARANTINE" {
			quarantined = true
		}
	}
	if !quarantined {
		t.Fatalf("expected a QUARANTINE admission on the live path, got %+v", admissions)
	}
	if strings.Contains(messages[2].Content, secret) {
		t.Errorf("model-facing content still leaks the secret (should be paged out): %q", messages[2].Content)
	}

	// The bridge the gateway just drove, on the ORIGINAL (poisoned) transcript — the same call
	// evictInKernelPoison makes with restored content. Assert the live KVCache.Evict fired AND
	// left the cache bit-exact to never-having-seen the poison span.
	restored := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "look up the config"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: poison},
	}
	freed, exact := planner.EvictKVSpan(restored, 2)
	if freed <= 0 {
		t.Fatalf("live KVCache.Evict freed %d positions, want > 0 (the poison span must be evicted)", freed)
	}
	if !exact {
		t.Fatalf("post-eviction cache is NOT bit-identical to never-saw — the live reposition invariant failed")
	}
	t.Logf("LIVE #579: poison span evicted (%d positions) on the real in-kernel path; cache bit-exact to never-saw", freed)
}

// TestLiveKVSpanEvictionDefaultOff is the posture guard: with the bridge flag OFF (the
// default), the SAME live planner does NOT drive a KVCache.Evict — the served path is
// byte-for-byte the pre-bridge behavior until an operator opts in.
func TestLiveKVSpanEvictionDefaultOff(t *testing.T) {
	// Explicitly OFF (the default): no FAK_INKERNEL_KVMMU.
	t.Setenv("FAK_INKERNEL_KVMMU", "")
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	m := model.NewSynthetic(kvmmuSynthCfg())
	planner := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "synthetic-off", false, nil)

	// It still implements the KVSpanEvictor interface, but the method is inert when off.
	ev, ok := any(planner).(agent.KVSpanEvictor)
	if !ok {
		t.Fatal("InKernelPlanner must implement agent.KVSpanEvictor (the gateway type-asserts it)")
	}
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: `{"page":"api_key=sk-abcdef0123456789abcdef0123"}`},
	}
	if freed, exact := ev.EvictKVSpan(messages, 1); freed != 0 || exact {
		t.Fatalf("bridge OFF must be inert: got freed=%d exact=%v, want 0/false", freed, exact)
	}
}

// TestLiveKVSpanBenignNotEvicted is the content-driven control on the live path: an ALLOWed
// (benign) tool result is admitted with no quarantine, so the live in-kernel eviction never
// fires — eviction is driven by the gate's reading of the bytes, not by position.
func TestLiveKVSpanBenignNotEvicted(t *testing.T) {
	srv := newKVMMUResultStackServer(t)
	planner := liveInKernelPlanner(t)
	srv.planner = planner

	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "lookup", Content: `{"ok":true,"rows":3}`},
	}
	admissions, err := srv.admitInboundResults(context.Background(), messages, "trace-live-benign")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	for _, a := range admissions {
		if a.Verdict.Kind == "QUARANTINE" {
			t.Fatalf("a benign result was quarantined on the live path: %+v", a)
		}
	}
}
