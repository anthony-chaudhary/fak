package modelroute

import (
	"encoding/json"
	"testing"
)

// Two providers chosen to exercise a REAL difference on every witnessed axis
// (guarding the #2934 confusion risk: a pairing that differs on nothing makes the
// witness trivial). Anthropic serves the structured tool_calls channel at the
// frontier price and keys its prefix on the Claude tokenizer; a local Ollama Qwen
// serves the Hermes <tool_call> TEXT dialect for free and keys its prefix on the
// Qwen tokenizer. So the switch flips the dialect (structured->hermes), resets the
// prefix (different tokenizer), AND moves the cost — all three at once.
var (
	epAnthropic = Endpoint{
		Model:     "claude-sonnet",
		Provider:  "anthropic",
		Dialect:   DialectStructured,
		PrefixKey: "anthropic/claude-tokenizer",
		Price:     Price{In: 3, Out: 15},
	}
	epOllamaQwen = Endpoint{
		Model:     "qwen2.5:1.5b",
		Provider:  "ollama",
		Dialect:   DialectHermes,
		PrefixKey: "local/qwen-tokenizer",
		Price:     Price{In: 0, Out: 0},
	}
)

// TestWitnessSwitchCrossProvider is the acceptance witness: switching provider
// mid-session emits a witness recording prefix-reuse status, dialect renorm, and
// cost delta, and it is correct across two real providers.
func TestWitnessSwitchCrossProvider(t *testing.T) {
	w := WitnessSwitch(epAnthropic, epOllamaQwen)

	// Prefix: the tokenizers differ, so the owned prefix is RESET — and the reason
	// must name it as an intentional re-encode, not a silent cache bust.
	if w.Prefix != PrefixReset {
		t.Fatalf("prefix: got %q, want %q", w.Prefix, PrefixReset)
	}
	if w.PrefixReused() {
		t.Fatal("PrefixReused() = true on a cross-tokenizer switch")
	}
	if w.PrefixReason == "" {
		t.Fatal("PrefixReason is empty; a reset must explain why")
	}

	// Dialect: structured -> hermes is a real renormalization.
	if !w.Renormalized {
		t.Fatalf("Renormalized = false for %s -> %s", w.DialectFrom, w.DialectTo)
	}
	if w.DialectFrom != DialectStructured || w.DialectTo != DialectHermes {
		t.Fatalf("dialect renorm: got %s->%s, want %s->%s", w.DialectFrom, w.DialectTo, DialectStructured, DialectHermes)
	}

	// Cost delta: frontier -> free is a large SAVING (negative delta) in both
	// directions. to - from = 0 - 3 and 0 - 15.
	if w.CostDeltaIn != -3 {
		t.Fatalf("CostDeltaIn: got %v, want -3", w.CostDeltaIn)
	}
	if w.CostDeltaOut != -15 {
		t.Fatalf("CostDeltaOut: got %v, want -15", w.CostDeltaOut)
	}
}

// TestWitnessSwitchReversePricier is the same two providers in the OTHER direction:
// switching from the free local Qwen back to Anthropic re-normalizes the dialect
// the other way and reports the cost delta as a PREMIUM (positive), so the witness
// is symmetric and the sign is meaningful.
func TestWitnessSwitchReversePricier(t *testing.T) {
	w := WitnessSwitch(epOllamaQwen, epAnthropic)

	if w.Prefix != PrefixReset {
		t.Fatalf("prefix: got %q, want %q", w.Prefix, PrefixReset)
	}
	if w.DialectFrom != DialectHermes || w.DialectTo != DialectStructured {
		t.Fatalf("dialect renorm: got %s->%s, want %s->%s", w.DialectFrom, w.DialectTo, DialectHermes, DialectStructured)
	}
	if w.CostDeltaIn != 3 || w.CostDeltaOut != 15 {
		t.Fatalf("cost delta: got in=%v out=%v, want in=3 out=15", w.CostDeltaIn, w.CostDeltaOut)
	}
}

// TestWitnessSwitchPreservedSameFamily guards the confusion risk from the other
// side: a same-family retier (Anthropic Sonnet -> Anthropic Haiku) shares a prefix
// key and the structured dialect, so the prefix is PRESERVED and there is NO
// renorm — only the cost moves. This is what makes "reset" legible as a decision,
// not a default: not every switch resets.
func TestWitnessSwitchPreservedSameFamily(t *testing.T) {
	haiku := Endpoint{
		Model:     "claude-haiku",
		Provider:  "anthropic",
		Dialect:   DialectStructured,
		PrefixKey: "anthropic/claude-tokenizer", // same tokenizer family as sonnet
		Price:     Price{In: 0.25, Out: 1.25},
	}
	w := WitnessSwitch(epAnthropic, haiku)

	if !w.PrefixReused() || w.Prefix != PrefixPreserved {
		t.Fatalf("prefix: got %q (reused=%v), want %q", w.Prefix, w.PrefixReused(), PrefixPreserved)
	}
	if w.Renormalized {
		t.Fatal("Renormalized = true on a same-dialect retier")
	}
	if w.CostDeltaOut != 1.25-15 {
		t.Fatalf("CostDeltaOut: got %v, want %v", w.CostDeltaOut, 1.25-15)
	}
}

// TestWitnessSwitchEmptyPrefixKeyResets: an endpoint that declares no prefix key
// cannot prove reuse, so the switch is a reset (fail-closed), even if the other
// endpoint has a key.
func TestWitnessSwitchEmptyPrefixKeyResets(t *testing.T) {
	unknown := epOllamaQwen
	unknown.PrefixKey = ""
	w := WitnessSwitch(epAnthropic, unknown)
	if w.Prefix != PrefixReset {
		t.Fatalf("prefix with empty key: got %q, want %q", w.Prefix, PrefixReset)
	}
	// Same-key check must not treat two empty keys as a match.
	both := WitnessSwitch(unknown, Endpoint{Model: "x", Dialect: DialectHermes})
	if both.Prefix != PrefixReset {
		t.Fatalf("two empty prefix keys must not preserve: got %q", both.Prefix)
	}
}

// TestSwitchWitnessDigestStableAndJSON: the witness is content-addressable (same
// switch -> same digest, a different switch -> a different digest) and its emitted
// JSON round-trips carrying the three witnessed facts.
func TestSwitchWitnessDigestStableAndJSON(t *testing.T) {
	a := WitnessSwitch(epAnthropic, epOllamaQwen)
	b := WitnessSwitch(epAnthropic, epOllamaQwen)
	if a.Digest() == "" {
		t.Fatal("Digest() is empty")
	}
	if a.Digest() != b.Digest() {
		t.Fatalf("digest not stable: %s vs %s", a.Digest(), b.Digest())
	}
	if c := WitnessSwitch(epOllamaQwen, epAnthropic); c.Digest() == a.Digest() {
		t.Fatal("a different switch must have a different digest")
	}

	var got SwitchWitness
	if err := json.Unmarshal([]byte(a.JSON()), &got); err != nil {
		t.Fatalf("witness JSON does not round-trip: %v", err)
	}
	if got.Prefix != a.Prefix || got.Renormalized != a.Renormalized || got.CostDeltaOut != a.CostDeltaOut {
		t.Fatalf("round-tripped witness lost a fact: %+v vs %+v", got, a)
	}
}

// TestEndpointValidate: WitnessSwitch never fails, but Endpoint.Validate lets a
// caller refuse a misconfigured switch — an empty model or an unknown dialect.
func TestEndpointValidate(t *testing.T) {
	if err := epAnthropic.Validate(); err != nil {
		t.Fatalf("valid endpoint rejected: %v", err)
	}
	if err := (Endpoint{Dialect: DialectStructured}).Validate(); err == nil {
		t.Fatal("empty model accepted")
	}
	if err := (Endpoint{Model: "m", Dialect: "made-up"}).Validate(); err == nil {
		t.Fatal("unknown dialect accepted")
	}
}
