package modelroute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFreeTierQualityGateRejectsLowQuality pins the gate's closed deny set: the
// witnessed low-quality Groq tier, non-chat endpoints, sub-scale toys, and
// non-tool models are all refused with a reason, while every curated allowlist
// entry passes. Without this test a future edit could silently re-admit a
// free-but-worthless upstream into the pool.
func TestFreeTierQualityGateRejectsLowQuality(t *testing.T) {
	reject := []struct {
		name          string
		upstream      string
		tools         bool
		contextTokens int
	}{
		{"groq compound routing slug", GroqCompoundModel, true, 0},
		{"whisper endpoint", "whisper-large-v3", true, 0},
		{"tts endpoint", "tts-1-hd", true, 0},
		{"embedding endpoint", "text-embed-3-large", true, 0},
		{"1b toy", "llama-3.2-1b-instruct:free", true, 0},
		{"tiny slug", "some-tiny-chat:free", true, 0},
		{"no tools", "qwen/qwen3.6-27b", false, 0},
		{"small pinned context", "qwen/qwen3.6-27b", true, 4096},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			if reason := FreeQualityReason(tc.upstream, tc.tools, tc.contextTokens); reason == "" {
				t.Errorf("upstream %q should be rejected by the quality gate, passed", tc.upstream)
			}
			if IsHighQualityFree(tc.upstream, tc.tools, tc.contextTokens) {
				t.Errorf("IsHighQualityFree(%q) = true, want false", tc.upstream)
			}
		})
	}

	// The allowlist itself must clear the gate — curation can never ship a model
	// its own gate refuses.
	for _, m := range DefaultFreeTierModels() {
		if reason := FreeQualityReason(m.Upstream, m.Tools, m.ContextTokens); reason != "" {
			t.Errorf("curated model %q (%s) fails its own gate: %s", m.Upstream, m.RoutedID, reason)
		}
	}
}

// TestFreeTierDenyTokensAreWholeToken pins the tokenizer subtlety the gate rests
// on: "11b" must NOT match the "1b" toy rule and "qwen3.6-27b" must tokenize
// clean — a substring match here would ban legitimate quality models.
func TestFreeTierDenyTokensAreWholeToken(t *testing.T) {
	if !IsHighQualityFree("meta-llama/llama-3.3-70b-instruct:free", true, 0) {
		t.Error("70b-class free model must pass the gate (no deny token)")
	}
	if !IsHighQualityFree(GroqQwen36Model, true, 0) {
		t.Errorf("%q must pass the gate", GroqQwen36Model)
	}
}

// TestFreeTierRosterValidatesAndResolves is the structural witness for the
// free-tier roster: it validates under BOTH gates (roster shape +
// ValidateFreeTierRoster), every curated id resolves to its witnessed upstream,
// and the roster is CLOSED — an unbound id (including the known low-quality
// slug) is a fail-loud error, never a silent pass-through that puts an
// unvetted upstream on the wire under a free seat's credential.
func TestFreeTierRosterValidatesAndResolves(t *testing.T) {
	r := DefaultFreeTierRoster()
	if err := ValidateFreeTierRoster(r); err != nil {
		t.Fatalf("free-tier roster fails quality validation: %v", err)
	}
	want := map[string]struct{ account, upstream string }{
		FreeCoding:    {FreeSeatGroq, GroqQwen36Model},
		FreeReasoning: {FreeSeatGemini, "gemini-3-pro"},
		FreeFlash:     {FreeSeatGeminiGCP, "google/gemini-3.5-flash"},
		FreeAuto:      {FreeSeatOpenRouter, "openrouter/auto"},
		FreeOx:        {FreeSeatOpenCode, OpenCodeGoOxAlphaModel},
	}
	for id, w := range want {
		tg, err := r.Resolve(id)
		if err != nil {
			t.Fatalf("resolve %q: %v", id, err)
		}
		if tg.Account != w.account || tg.UpstreamModel != w.upstream {
			t.Errorf("resolve %q = %s/%s, want %s/%s", id, tg.Account, tg.UpstreamModel, w.account, w.upstream)
		}
		if !tg.Remote() {
			t.Errorf("free-tier target %q must be remote (it leaves the box; the residency floor reads this)", id)
		}
	}
	for _, b := range r.Bindings {
		up := b.UpstreamModel
		if up == "" {
			up = b.Model
		}
		for _, tok := range freeUpstreamTokens(up) {
			if freeDenyTokens[tok] {
				t.Errorf("binding %q carries deny token %q — low-quality model in the free pool", b.Model, tok)
			}
		}
	}
	// Closed pool: unvetted ids fail loud instead of passing through.
	for _, id := range []string{"groq-compound", "mystery-model", "llama-3.2-1b-instruct:free"} {
		if tg, err := r.Resolve(id); err == nil {
			t.Errorf("unbound id %q must not resolve in the closed free-tier roster, got %+v", id, tg)
		}
	}
}

// TestFreeTierRosterRejectsLowQualityEdit proves the roster gate bites: an
// operator edit binding the low-quality Groq tier fails ValidateFreeTierRoster
// (while still passing plain Roster.Validate — the shape is fine, the QUALITY
// is not), so bad curation fails at --check time, never at dispatch.
func TestFreeTierRosterRejectsLowQualityEdit(t *testing.T) {
	r := DefaultFreeTierRoster()
	r.Bindings = append(r.Bindings, Binding{Model: "free-cheap", Account: FreeSeatGroq, UpstreamModel: GroqCompoundModel})
	if err := r.Validate(); err != nil {
		t.Fatalf("shape should still be valid (the failure must be quality, not shape): %v", err)
	}
	if err := ValidateFreeTierRoster(r); err == nil {
		t.Fatal("binding groq/compound must fail free-tier quality validation")
	}
}

// TestFreeTierFallbackChainsStayInPool pins the exhaustion contract: every
// chain is non-empty, head is the id itself, every link resolves in the roster,
// and no link carries a deny token — degradation walks to another
// HIGH-QUALITY free seat, never off the vetted pool.
func TestFreeTierFallbackChainsStayInPool(t *testing.T) {
	r := DefaultFreeTierRoster()
	for _, m := range DefaultFreeTierModels() {
		chain := FreeFallbackChain(m.RoutedID)
		if len(chain) == 0 {
			t.Fatalf("no fallback chain for %q", m.RoutedID)
		}
		if chain[0] != m.RoutedID {
			t.Errorf("chain for %q must head at itself, got %q", m.RoutedID, chain[0])
		}
		for _, link := range chain {
			tg, err := r.Resolve(link)
			if err != nil {
				t.Errorf("chain link %q (from %q) does not resolve: %v", link, m.RoutedID, err)
			}
			if reason := FreeQualityReason(tg.UpstreamModel, true, 0); reason != "" {
				t.Errorf("chain link %q upstream %q fails gate: %s", link, tg.UpstreamModel, reason)
			}
		}
	}
	if FreeFallbackChain("no-such-model") != nil {
		t.Error("unknown routed id must yield nil chain (fail loud, never a silent seat)")
	}
}

// TestFreeTierManifestDecisions is the semantic witness for the free-tier
// routing policy: hard and destructive work pins to free-reasoning (the floor
// that never drops), ordinary medium-and-up work goes to free-coding, cheap /
// read / search and short interactive turns go to free-flash, and the rest
// fail-closes to free-coding. Each case asserts BOTH rule and primary so a
// reorder that changes which rule fires is caught.
func TestFreeTierManifestDecisions(t *testing.T) {
	m := DefaultFreeTierManifest()
	if err := m.Validate(); err != nil {
		t.Fatalf("free-tier manifest invalid: %v", err)
	}
	cases := []struct {
		name    string
		subject Subject
		primary string
		rule    string
	}{
		{
			name:    "high-complexity work -> free-reasoning",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			primary: FreeReasoning, rule: "free-ultrahard-reasoning",
		},
		{
			name:    "high-complexity read escalates over the read rule",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file", Complexity: ComplexityHigh},
			primary: FreeReasoning, rule: "free-ultrahard-reasoning",
		},
		{
			name:    "delete tool call -> free-reasoning (floor)",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_branch"},
			primary: FreeReasoning, rule: "free-destructive-delete",
		},
		{
			name:    "tiny low-complexity delete still -> free-reasoning",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_file", Complexity: ComplexityLow, Latency: LatencyInteractive},
			primary: FreeReasoning, rule: "free-destructive-delete",
		},
		{
			name:    "drop tool call -> free-reasoning",
			subject: Subject{Aspect: AspectToolCall, Tool: "drop_table"},
			primary: FreeReasoning, rule: "free-destructive-drop",
		},
		{
			name:    "medium work -> free-coding (band rule ahead of cheap)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityMedium},
			primary: FreeCoding, rule: "free-normal-coding",
		},
		{
			name:    "medium read stays on coding, not flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file", Complexity: ComplexityMedium},
			primary: FreeCoding, rule: "free-normal-coding",
		},
		{
			name:    "low-complexity aspect -> free-flash",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityLow},
			primary: FreeFlash, rule: "free-cheap-flash",
		},
		{
			name:    "read tool call -> free-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file"},
			primary: FreeFlash, rule: "free-read-flash",
		},
		{
			name:    "search tool call -> free-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb"},
			primary: FreeFlash, rule: "free-search-flash",
		},
		{
			name:    "short interactive turn -> free-flash",
			subject: Subject{Aspect: AspectRequest, PromptTokens: 512, Latency: LatencyInteractive},
			primary: FreeFlash, rule: "free-short-interactive",
		},
		{
			name:    "unmatched work -> free-coding default",
			subject: Subject{Aspect: AspectQuery},
			primary: FreeCoding, rule: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := m.Route(tc.subject)
			if got := d.Plan.Primary(); got != tc.primary {
				t.Errorf("primary = %q, want %q (matched rule %q)", got, tc.primary, d.RuleName)
			}
			if d.RuleName != tc.rule {
				t.Errorf("rule = %q, want %q", d.RuleName, tc.rule)
			}
		})
	}
}

// TestFreeHighQualityPresetFileDecisions binds the shipped preset file
// (examples/routing-presets/free-high-quality.json) to the same decisions the
// constructor encodes: the file is data-only, so without this test a hand-edit
// could silently re-route a dev aspect (most dangerously drop a destructive
// call to flash) while the round-trip test still passes.
func TestFreeHighQualityPresetFileDecisions(t *testing.T) {
	path := filepath.Join("..", "..", "examples", "routing-presets", "free-high-quality.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse free-high-quality preset: %v", err)
	}
	for _, tc := range []struct {
		subject Subject
		primary string
		rule    string
	}{
		{Subject{Aspect: AspectStep, Complexity: ComplexityHigh}, FreeReasoning, "free-ultrahard-reasoning"},
		{Subject{Aspect: AspectToolCall, Tool: "delete_file", Complexity: ComplexityLow}, FreeReasoning, "free-destructive-delete"},
		{Subject{Aspect: AspectStep, Complexity: ComplexityMedium}, FreeCoding, "free-normal-coding"},
		{Subject{Aspect: AspectToolCall, Tool: "read_file"}, FreeFlash, "free-read-flash"},
		{Subject{Aspect: AspectRequest, PromptTokens: 512, Latency: LatencyInteractive}, FreeFlash, "free-short-interactive"},
		{Subject{Aspect: AspectQuery}, FreeCoding, ""},
	} {
		d := m.Route(tc.subject)
		if got := d.Plan.Primary(); got != tc.primary {
			t.Errorf("subject %+v: primary = %q, want %q (rule %q)", tc.subject, got, tc.primary, d.RuleName)
		}
		if d.RuleName != tc.rule {
			t.Errorf("subject %+v: rule = %q, want %q", tc.subject, d.RuleName, tc.rule)
		}
	}
}

// TestFreeHighQualityRosterFileParses binds the shipped free roster example
// (examples/model-accounts.free-high-quality.json): it must parse, pass the
// quality gate, and resolve every routed id the preset file can emit — the
// manifest/roster pair ships consistent, or not at all.
func TestFreeHighQualityRosterFileParses(t *testing.T) {
	rosterPath := filepath.Join("..", "..", "examples", "model-accounts.free-high-quality.json")
	raw, err := os.ReadFile(rosterPath)
	if err != nil {
		t.Fatalf("read %s: %v", rosterPath, err)
	}
	r, err := ParseRoster(raw)
	if err != nil {
		t.Fatalf("shipped free roster must parse and validate: %v", err)
	}
	if err := ValidateFreeTierRoster(r); err != nil {
		t.Fatalf("shipped free roster must pass the quality gate: %v", err)
	}
	manifestPath := filepath.Join("..", "..", "examples", "routing-presets", "free-high-quality.json")
	mraw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read %s: %v", manifestPath, err)
	}
	m, err := ParseManifest(mraw)
	if err != nil {
		t.Fatalf("parse free-high-quality preset: %v", err)
	}
	ids := map[string]bool{m.Default.Primary(): true}
	for _, rl := range m.Rules {
		for _, mem := range rl.Plan.Members {
			ids[mem.Model] = true
		}
		if rl.Plan.Scout != "" {
			ids[rl.Plan.Scout] = true
		}
	}
	for id := range ids {
		tg, err := r.Resolve(id)
		if err != nil {
			t.Fatalf("preset emits %q but the free roster cannot resolve it: %v", id, err)
		}
		if strings.TrimSpace(tg.UpstreamModel) == "" {
			t.Errorf("routed id %q resolves to an empty upstream", id)
		}
	}
}
