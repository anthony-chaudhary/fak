package toon

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// bigTabular is a uniform array of flat objects large enough to clear the default size gate
// (rows ≥ R_min, bytes ≥ B_min) and win decisively on tokens — the canonical FIRE shape.
const bigTabular = `[
	{"id":"n1","verdict":"fresh","kind":"snippet","score":0.91},
	{"id":"n2","verdict":"withheld","kind":"summary","score":0.42},
	{"id":"n3","verdict":"fresh","kind":"fact","score":0.77},
	{"id":"n4","verdict":"stale","kind":"qa","score":0.13},
	{"id":"n5","verdict":"fresh","kind":"snippet","score":0.55},
	{"id":"n6","verdict":"withheld","kind":"summary","score":0.28},
	{"id":"n7","verdict":"fresh","kind":"fact","score":0.66},
	{"id":"n8","verdict":"stale","kind":"qa","score":0.04}
]`

// TestDecideFires is the positive case: on the shape TOON targets, with no blocking signal,
// Decide returns Encode:true and a strictly positive token saving.
func TestDecideFires(t *testing.T) {
	payload := fromJSON(t, bigTabular)
	d := Decide(payload, DecideInput{})
	if !d.Encode {
		t.Fatalf("expected FIRE on the canonical tabular shape, got SKIP(%s)\n%s", d.Reason, d)
	}
	if d.Reason != "" {
		t.Errorf("a firing decision must carry no reason, got %q", d.Reason)
	}
	if d.Eligibility != 1.0 {
		t.Errorf("uniform table should be fully eligible, got %v", d.Eligibility)
	}
	if !(d.TOONTokens < d.JSONTokens) {
		t.Errorf("fire requires a real saving: toon=%d json=%d", d.TOONTokens, d.JSONTokens)
	}
}

// TestDecideForcesEachSkipReason forces EXACTLY one reason per row: every input but the one
// under test is arranged to pass, so the returned reason is unambiguous. This is the
// closed-vocabulary DoD ("every SkipReason has a test that forces it").
func TestDecideForcesEachSkipReason(t *testing.T) {
	cases := []struct {
		name    string
		payload any
		in      DecideInput
		want    SkipReason
	}{
		{
			// TOON is input-only; a good tabular payload still skips when it is an output schema.
			name:    "output_direction",
			payload: fromJSON(t, bigTabular),
			in:      DecideInput{OutputDirection: true},
			want:    ReasonOutputDirection,
		},
		{
			// Deeply nested object: eligibility 0.0 < τ.
			name:    "tabular_eligibility_low",
			payload: fromJSON(t, `{"a":{"b":{"c":{"d":1}}}}`),
			in:      DecideInput{},
			want:    ReasonTabularEligibilityLow,
		},
		{
			// Fully eligible but a single row — below the default R_min row floor.
			name:    "payload_too_small",
			payload: fromJSON(t, `[{"a":1,"b":2}]`),
			in:      DecideInput{},
			want:    ReasonPayloadTooSmall,
		},
		{
			// Winning shape, but the span is already in a cached prefix.
			name:    "cache_prefix_resident",
			payload: fromJSON(t, bigTabular),
			in:      DecideInput{CacheResident: true},
			want:    ReasonCachePrefixResident,
		},
		{
			// Winning shape, but the span head is volatile.
			name:    "volatile_span",
			payload: fromJSON(t, bigTabular),
			in:      DecideInput{Volatile: true},
			want:    ReasonVolatileSpan,
		},
		{
			// Winning shape, but the model is unfit and no primer closes the gap.
			name:    "model_toon_unfit",
			payload: fromJSON(t, bigTabular),
			in:      DecideInput{ModelFitnessKnown: true, ModelFitness: 0.2},
			want:    ReasonModelToonUnfit,
		},
		{
			// Go ints (NOT the encoding/json-native domain): Decode normalizes them to float64,
			// so Decode(Encode(payload)) != payload — the round-trip witness fails. Thresholds
			// are relaxed so only the round-trip gate can fire.
			name: "roundtrip_lossy",
			payload: []any{
				map[string]any{"a": 1, "b": 2},
				map[string]any{"a": 3, "b": 4},
			},
			in:   DecideInput{MinRows: 1, MinBytes: 1, TabularEligibilityMin: 0.01},
			want: ReasonRoundtripLossy,
		},
		{
			// A single-row table: the one-time header costs as much as it saves, so the real
			// tokenizer shows no net win. Size gate relaxed so the net-token gate is reached.
			name:    "net_tokens_nonpositive",
			payload: fromJSON(t, `[{"a":1,"b":2}]`),
			in:      DecideInput{MinRows: 1, MinBytes: 1},
			want:    ReasonNetTokensNonPositive,
		},
	}

	seen := map[SkipReason]bool{}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := Decide(c.payload, c.in)
			if d.Encode {
				t.Fatalf("expected SKIP(%s), got FIRE\n%s", c.want, d)
			}
			if d.Reason != c.want {
				t.Fatalf("expected reason %q, got %q\n%s", c.want, d.Reason, d)
			}
			if !d.Reason.Known() {
				t.Fatalf("returned reason %q is not in the closed vocabulary", d.Reason)
			}
		})
		seen[c.want] = true
	}

	// Every reason in the closed set must be covered by a forcing row above.
	for _, r := range KnownSkipReasons() {
		if !seen[r] {
			t.Errorf("no forcing test covers skip reason %q", r)
		}
	}
}

// TestDecideNeverFiresAtALoss is THE safety property: across a randomized corpus of payloads,
// inputs, and tokenizers, Decide never returns Encode:true when TOONTokens >= JSONTokens. It
// also proves the two structural guarantees — exactly one of {Encode, a known Reason} holds —
// and that the corpus actually exercises the FIRE path (else the proof would be vacuous).
func TestDecideNeverFiresAtALoss(t *testing.T) {
	tokenizers := []struct {
		name string
		fn   func([]byte) int
	}{
		{"default_bytes_over_4", nil},
		{"per_byte", func(b []byte) int { return len(b) }},
		{"word_count", func(b []byte) int { return len(strings.Fields(string(b))) }},
		{"structural_weighted", func(b []byte) int {
			// Weights structural punctuation heavier — a deliberately different profile so the
			// invariant is proven under more than one cost model.
			n := 0
			for _, c := range b {
				switch c {
				case '{', '}', '[', ']', ':', ',', '"':
					n += 2
				case ' ', '\n', '\t':
					// free
				default:
					n++
				}
			}
			return (n + 3) / 4
		}},
	}

	r := rand.New(rand.NewSource(0x70074e))
	const iterations = 4000
	fires := 0

	for i := 0; i < iterations; i++ {
		payload := genPayload(r, 0)
		in := genInput(r)
		for _, tk := range tokenizers {
			in.Tokenizer = tk.fn
			d := Decide(payload, in)

			// Exactly one outcome: Encode xor a reason.
			if d.Encode && d.Reason != "" {
				t.Fatalf("iter %d [%s]: FIRE must carry no reason, got %q\npayload=%#v", i, tk.name, d.Reason, payload)
			}
			if !d.Encode {
				if d.Reason == "" {
					t.Fatalf("iter %d [%s]: SKIP must carry a reason\npayload=%#v", i, tk.name, payload)
				}
				if !d.Reason.Known() {
					t.Fatalf("iter %d [%s]: reason %q outside the closed vocabulary\npayload=%#v", i, tk.name, d.Reason, payload)
				}
				continue
			}

			// THE INVARIANT.
			if d.TOONTokens >= d.JSONTokens {
				t.Fatalf("iter %d [%s]: FIRED AT A LOSS — toon=%d >= json=%d\npayload=%#v\ninput=%+v",
					i, tk.name, d.TOONTokens, d.JSONTokens, payload, in)
			}
			fires++
		}
	}

	if fires == 0 {
		t.Fatalf("corpus never fired — the safety proof is vacuous; loosen the generator")
	}
	t.Logf("randomized corpus: %d fires across %d iterations × %d tokenizers", fires, iterations, len(tokenizers))
}

// TestKnownSkipReasonsClosed pins the closed vocabulary: the enumerator lists every constant
// exactly once, each is Known(), and no stray token creeps in.
func TestKnownSkipReasonsClosed(t *testing.T) {
	want := []SkipReason{
		ReasonTabularEligibilityLow, ReasonPayloadTooSmall, ReasonNetTokensNonPositive,
		ReasonCachePrefixResident, ReasonVolatileSpan, ReasonModelToonUnfit,
		ReasonRoundtripLossy, ReasonOutputDirection,
	}
	got := KnownSkipReasons()
	if len(got) != len(want) {
		t.Fatalf("KnownSkipReasons length = %d, want %d", len(got), len(want))
	}
	set := map[SkipReason]int{}
	for _, r := range got {
		if !r.Known() {
			t.Errorf("enumerated reason %q reports Known()==false", r)
		}
		set[r]++
	}
	for _, r := range want {
		if set[r] != 1 {
			t.Errorf("reason %q appears %d times in KnownSkipReasons, want exactly 1", r, set[r])
		}
	}
	if SkipReason("NOT_A_REASON").Known() {
		t.Error("an invented token must not be Known()")
	}
	if (SkipReason("")).Known() {
		t.Error("the empty reason (the FIRE sentinel) must not be Known()")
	}
}

// TestDecisionString sanity-checks the audit render for both outcomes.
func TestDecisionString(t *testing.T) {
	fire := Decide(fromJSON(t, bigTabular), DecideInput{})
	if !strings.HasPrefix(fire.String(), "toon: FIRE ") {
		t.Errorf("fire render = %q", fire.String())
	}
	skip := Decide(fromJSON(t, `{"a":{"b":{"c":1}}}`), DecideInput{})
	if !strings.Contains(skip.String(), "SKIP(TABULAR_ELIGIBILITY_LOW)") {
		t.Errorf("skip render = %q", skip.String())
	}
}

// --- randomized-corpus generators (deterministic under a fixed seed) ---

// genPayload builds a random encoding/json-native value: containers are map[string]any /
// []any, numbers are float64, plus string/bool/nil — the exact domain the codec round-trips,
// so the FIRE path is reachable. Depth is bounded so generation terminates.
func genPayload(r *rand.Rand, depth int) any {
	if depth >= 3 {
		return genScalar(r)
	}
	switch r.Intn(6) {
	case 0, 1:
		// Uniform array of flat objects (the tabular-eligible win shape) — variable size.
		rows := r.Intn(9) // 0..8
		fieldCount := 1 + r.Intn(4)
		fields := make([]string, fieldCount)
		for i := range fields {
			fields[i] = fmt.Sprintf("f%d", i)
		}
		arr := make([]any, rows)
		for i := range arr {
			obj := make(map[string]any, fieldCount)
			for _, f := range fields {
				obj[f] = genScalar(r)
			}
			arr[i] = obj
		}
		return arr
	case 2:
		// Nested object.
		obj := map[string]any{}
		for i := 0; i < 1+r.Intn(3); i++ {
			obj[fmt.Sprintf("k%d", i)] = genPayload(r, depth+1)
		}
		return obj
	case 3:
		// Ragged array (differing key sets) -> list fallback.
		return []any{
			map[string]any{"a": genScalar(r)},
			map[string]any{"a": genScalar(r), "b": genScalar(r)},
		}
	case 4:
		// Scalar array.
		n := r.Intn(6)
		arr := make([]any, n)
		for i := range arr {
			arr[i] = genScalar(r)
		}
		return arr
	default:
		return genScalar(r)
	}
}

func genScalar(r *rand.Rand) any {
	switch r.Intn(7) {
	case 0:
		return float64(r.Intn(100000))
	case 1:
		return r.Float64() * 1000
	case 2:
		return r.Intn(2) == 0
	case 3:
		return nil
	case 4:
		// A string that merely LOOKS typed — the codec must quote it; round-trips as a string.
		return []string{"123", "true", "null", "-5", "1e9"}[r.Intn(5)]
	case 5:
		// A string with embedded delimiters/quotes/newlines.
		return []string{"a,b,c", "he said \"hi\"", "line1\nline2", "tab\there", "p|q"}[r.Intn(5)]
	default:
		return []string{"fresh", "withheld", "stale", "snippet", "summary", "héllo 世界"}[r.Intn(6)]
	}
}

// genInput builds a random DecideInput. Blocking signals are biased OFF and thresholds kept
// lenient so the FIRE path is reachable often enough to make the invariant proof non-vacuous;
// the invariant itself must hold for EVERY draw regardless.
func genInput(r *rand.Rand) DecideInput {
	in := DecideInput{
		TabularEligibilityMin: 0.5 + r.Float64()*0.4, // 0.5..0.9
		MinRows:               1 + r.Intn(4),         // 1..4
		MinBytes:              1 + r.Intn(120),       // 1..120
		NetTokenMargin:        r.Intn(7),             // 0..6
		ModelFitnessMin:       DefaultModelFitnessMin,
	}
	if r.Intn(100) < 12 {
		in.CacheResident = true
	}
	if r.Intn(100) < 12 {
		in.Volatile = true
	}
	if r.Intn(100) < 12 {
		in.OutputDirection = true
	}
	if r.Intn(100) < 20 {
		in.ModelFitnessKnown = true
		in.ModelFitness = r.Float64() // 0..1
		in.PrimerLift = r.Float64() * 0.3
	}
	return in
}
