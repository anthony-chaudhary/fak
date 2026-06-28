package sessionreset

import (
	"strings"
	"testing"
)

// sampleTranscript is a long-ish session with a clear durable fact, ephemera that
// should evaporate, a system preamble, an objective, and a recent tail.
func sampleTranscript() []Msg {
	return []Msg{
		{Role: "system", Content: "You are a helpful coding assistant for the fak repo."},
		{Role: "user", Content: "Help me add a budget-triggered session reset to fak."},
		{Role: "assistant", Content: "Sure. Let me look at the session package."},
		{Role: "user", Content: "I prefer concise answers."},                   // durable
		{Role: "user", Content: "it's 3pm and the build is currently running"}, // turn — drops
		{Role: "assistant", Content: "Got it."},
		{Role: "user", Content: "Now wire the gateway hook."}, // latest ask
	}
}

// TestDurabilityFactsKeepsDurableDropsEphemera proves the core human-like move: the
// stated preference survives, the timestamp/progress line evaporates — reusing the
// shipped ctxmmu prior, not a fork.
func TestDurabilityFactsKeepsDurableDropsEphemera(t *testing.T) {
	p, ok := durabilityFacts{}.Contribute(Input{Messages: sampleTranscript()})
	if !ok {
		t.Fatal("durabilityFacts declined on a transcript with a durable fact")
	}
	if !strings.Contains(p.Text, "I prefer concise answers") {
		t.Fatalf("durable preference not kept: %q", p.Text)
	}
	if strings.Contains(p.Text, "3pm") || strings.Contains(p.Text, "currently running") {
		t.Fatalf("ephemeral line leaked into carryover: %q", p.Text)
	}
	if p.Meta["durable"] != "1" {
		t.Fatalf("durable count = %q, want 1", p.Meta["durable"])
	}
}

// TestTaskDistillExtractsObjectiveAndLatest proves the deterministic recap names the
// opening objective and the latest ask.
func TestTaskDistillExtractsObjectiveAndLatest(t *testing.T) {
	p, ok := taskDistill{}.Contribute(Input{Messages: sampleTranscript()})
	if !ok {
		t.Fatal("taskDistill declined on a transcript with user turns")
	}
	if !strings.Contains(p.Text, "budget-triggered session reset") {
		t.Fatalf("objective missing from recap: %q", p.Text)
	}
	if !strings.Contains(p.Text, "wire the gateway hook") {
		t.Fatalf("latest request missing from recap: %q", p.Text)
	}
}

// TestWarmPrefixDescribesStablePrefix proves the prefix contributor fires on a system
// preamble and prices it, while honestly marking live KV reuse deferred.
func TestWarmPrefixDescribesStablePrefix(t *testing.T) {
	p, ok := warmPrefix{}.Contribute(Input{Messages: sampleTranscript()})
	if !ok {
		t.Fatal("warmPrefix declined despite a system preamble")
	}
	if p.Order != 0 {
		t.Fatalf("warm prefix Order = %d, want 0 (top of seed)", p.Order)
	}
	if p.Meta["live_kv_reuse"] != "deferred" {
		t.Fatalf("warm prefix must honestly mark live KV reuse deferred, got %q", p.Meta["live_kv_reuse"])
	}
	// No system preamble -> declines.
	noSys := Input{Messages: []Msg{{Role: "user", Content: "hi"}}}
	if _, ok := (warmPrefix{}).Contribute(noSys); ok {
		t.Fatal("warmPrefix should decline with no system preamble")
	}
}

// TestWarmPrefixStampFlipsToLive proves the #916 stamp flip: with no live splicer wired the
// warm_prefix part honestly marks live_kv_reuse "deferred"; once a concrete same-model warm-KV
// mover is wired (MarkLiveKVReuse(true), the session.WarmKVStore path), the SAME contributor
// flips the stamp to "live" — the descriptor tracks the live wiring instead of being hardcoded.
func TestWarmPrefixStampFlipsToLive(t *testing.T) {
	// Default (unwired) is deferred.
	if got := LiveKVReuseStamp(); got != LiveKVReuseDeferred {
		t.Fatalf("default stamp = %q, want %q", got, LiveKVReuseDeferred)
	}
	p, ok := warmPrefix{}.Contribute(Input{Messages: sampleTranscript()})
	if !ok || p.Meta["live_kv_reuse"] != LiveKVReuseDeferred {
		t.Fatalf("unwired warm prefix stamp = %q, want %q", p.Meta["live_kv_reuse"], LiveKVReuseDeferred)
	}

	// Wire a live splicer -> the stamp flips to live, and the rendered text reflects it.
	MarkLiveKVReuse(true)
	defer MarkLiveKVReuse(false) // restore the global so other tests see the default
	if got := LiveKVReuseStamp(); got != LiveKVReuseLive {
		t.Fatalf("wired stamp = %q, want %q", got, LiveKVReuseLive)
	}
	live, ok := warmPrefix{}.Contribute(Input{Messages: sampleTranscript()})
	if !ok {
		t.Fatal("warmPrefix declined after MarkLiveKVReuse")
	}
	if live.Meta["live_kv_reuse"] != LiveKVReuseLive {
		t.Fatalf("wired warm prefix stamp = %q, want %q (the live splice flips deferred->live)", live.Meta["live_kv_reuse"], LiveKVReuseLive)
	}
	if !strings.Contains(live.Text, "wired") {
		t.Fatalf("wired warm prefix text should announce the live splice, got %q", live.Text)
	}

	// Flipping back restores the honest deferred default.
	MarkLiveKVReuse(false)
	if got := LiveKVReuseStamp(); got != LiveKVReuseDeferred {
		t.Fatalf("after reset stamp = %q, want %q", got, LiveKVReuseDeferred)
	}
}

// TestVerbatimTailKeepsLastTurnsOldestFirst proves the tail is the last N messages in
// chronological order.
func TestVerbatimTailKeepsLastTurnsOldestFirst(t *testing.T) {
	p, ok := verbatimTail{N: 2}.Contribute(Input{Messages: sampleTranscript()})
	if !ok {
		t.Fatal("verbatimTail declined on a non-empty transcript")
	}
	if !strings.Contains(p.Text, "wire the gateway hook") {
		t.Fatalf("tail missing the last user turn: %q", p.Text)
	}
	// "Got it." (assistant) precedes "Now wire..." (user) in the source, so oldest-first
	// the assistant line appears before the user line in the rendered tail.
	gotIdx := strings.Index(p.Text, "Got it.")
	wireIdx := strings.Index(p.Text, "wire the gateway hook")
	if gotIdx < 0 || wireIdx < 0 || gotIdx > wireIdx {
		t.Fatalf("tail not oldest-first: %q", p.Text)
	}
}

// TestBuildSeedIsDeterministicAndOrdered proves the fold is reproducible and renders
// parts by Order (warm prefix first at 0, verbatim tail last at 90).
func TestBuildSeedIsDeterministicAndOrdered(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	a := BuildSeed(in)
	b := BuildSeed(in)
	if a.Recap != b.Recap {
		t.Fatal("BuildSeed is not deterministic: two folds differ")
	}
	if a.Recap == "" {
		t.Fatal("BuildSeed produced an empty recap on a rich transcript")
	}
	// The default four contributors should all fire on this transcript.
	if len(a.Parts) < 4 {
		t.Fatalf("expected >=4 parts from the default contributors, got %d", len(a.Parts))
	}
	// Order: warm_prefix (0) appears before verbatim_tail (90) in the rendered recap.
	prefixIdx := strings.Index(a.Recap, "Stable prefix retained")
	tailIdx := strings.Index(a.Recap, "Most recent exchange")
	if prefixIdx < 0 || tailIdx < 0 || prefixIdx > tailIdx {
		t.Fatalf("parts not rendered in Order: prefix@%d tail@%d", prefixIdx, tailIdx)
	}
	// The continuation header frames it as a recap, not a fresh instruction.
	if !strings.HasPrefix(a.Recap, "[continuation of a prior session") {
		t.Fatalf("recap missing the continuation header")
	}
}

// stubContributor is a third-party registrant proving the open extension seam: a new
// carryover item joins the fold without editing the core.
type stubContributor struct{}

func (stubContributor) Name() string { return "third_party_stub" }
func (stubContributor) Contribute(Input) (Part, bool) {
	return Part{Name: "third_party_stub", Order: 25, Text: "Custom item from a 3rd-party plugin."}, true
}

// TestThirdPartyRegisterJoinsTheFold proves Register is the open seam — a new
// contributor's text lands in the seed with no change to BuildSeed.
func TestThirdPartyRegisterJoinsTheFold(t *testing.T) {
	before := len(Registered())
	Register(stubContributor{})
	if len(Registered()) != before+1 {
		t.Fatalf("Register did not add the contributor: %d -> %d", before, len(Registered()))
	}
	seed := BuildSeed(Input{Messages: sampleTranscript()})
	if !strings.Contains(seed.Recap, "Custom item from a 3rd-party plugin") {
		t.Fatalf("third-party contribution missing from seed: %q", seed.Recap)
	}
	// It sits between durability_facts (10) and task_distill (20)... actually Order 25
	// is after task_distill (20) and before verbatim_tail (90): assert it precedes tail.
	stubIdx := strings.Index(seed.Recap, "Custom item")
	tailIdx := strings.Index(seed.Recap, "Most recent exchange")
	if stubIdx < 0 || stubIdx > tailIdx {
		t.Fatalf("third-party Order not honored: stub@%d tail@%d", stubIdx, tailIdx)
	}
}
