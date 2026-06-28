package modelengine

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // register the Ref resolver (CAS backend)
)

// genResultText is the result-payload shape AFTER #463: it carries the additive
// "generated_text" field alongside the raw ids the byte-level path always returned.
type genResultText struct {
	Tool   string `json:"tool"`
	Engine string `json:"engine"`
	Model  string `json:"model"`
	Tokens []int  `json:"generated_tokens"`
	Text   string `json:"generated_text"`
}

// fakeTok is a hermetic NLTokenizer (no internal/tokenizer dependency) over the
// synthetic checkpoint's 256-id vocab: Encode maps each input byte to an in-vocab id
// and Decode maps each id back to a byte, so a round-trip is well-defined. It records
// call counts so a test can witness that the engine actually drove the NL encode +
// detokenize seams rather than the byte-level fallback.
type fakeTok struct {
	encCalls int
	decCalls int
}

func (f *fakeTok) Encode(text string) ([]int, error) {
	f.encCalls++
	ids := make([]int, 0, len(text))
	for i := 0; i < len(text); i++ {
		ids = append(ids, int(text[i])%200) // < synthetic vocab (256): always in range
	}
	if len(ids) == 0 {
		ids = append(ids, 0)
	}
	return ids, nil
}

func (f *fakeTok) Decode(ids []int) (string, error) {
	f.decCalls++
	b := make([]byte, 0, len(ids))
	for _, id := range ids {
		b = append(b, byte(id%256))
	}
	return string(b), nil
}

func decodeGenText(t *testing.T, ctx context.Context, r *abi.Result) genResultText {
	t.Helper()
	b := refBytes(ctx, r.Payload)
	var g genResultText
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("result payload is not the generated-token JSON: %v (%s)", err, b)
	}
	return g
}

// TestSyscallReturnsTextWhenTokenizerArmed is the #463 witness: once a real NL
// tokenizer is armed (SetTokenizer — the seam the `--gguf` serve boot wires), an
// adjudicated in-kernel completion returns decoded TEXT (generated_text) in addition
// to the raw ids, AND the prompt is NL-tokenized (Encode driven) rather than
// byte-tokenized. This is exactly the gap the issue scopes to the /v1/fak/syscall
// route: "the pure kernel can decode but can't yet speak."
func TestSyscallReturnsTextWhenTokenizerArmed(t *testing.T) {
	ctx := context.Background()
	ft := &fakeTok{}
	e := New()
	e.SetTokenizer(ft)

	r, err := e.Complete(ctx, inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if ft.encCalls == 0 {
		t.Fatalf("prompt was not NL-tokenized: Encode never called (byte-level fallback ran)")
	}
	if r.Meta["detokenized"] != "true" {
		t.Fatalf("meta detokenized = %q, want \"true\"", r.Meta["detokenized"])
	}

	g := decodeGenText(t, ctx, r)
	if len(g.Tokens) != genTokens {
		t.Fatalf("generated %d tokens, want %d", len(g.Tokens), genTokens)
	}
	if g.Text == "" {
		t.Fatalf("generated_text is empty; the syscall route still returns only token ids")
	}
	// The detokenized text must be the tokenizer's own decode of the very ids returned —
	// a real wiring, not a constant. Recompute independently via a fresh tokenizer.
	want, _ := (&fakeTok{}).Decode(g.Tokens)
	if g.Text != want {
		t.Fatalf("generated_text = %q, want tok.Decode(generated_tokens) = %q", g.Text, want)
	}
}

// TestSyscallStaysByteLevelWithoutTokenizer pins the honest-scope guard: with NO
// tokenizer armed (the CI / no-export default), the payload omits generated_text and
// the detokenized Meta flag entirely — byte-identical to the pre-#463 path.
func TestSyscallStaysByteLevelWithoutTokenizer(t *testing.T) {
	ctx := context.Background()
	e := New()

	r, err := e.Complete(ctx, inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, ok := r.Meta["detokenized"]; ok {
		t.Fatalf("byte-level default must not set the detokenized Meta flag")
	}
	g := decodeGenText(t, ctx, r)
	if g.Text != "" {
		t.Fatalf("byte-level default leaked generated_text = %q, want empty", g.Text)
	}
	if len(g.Tokens) != genTokens {
		t.Fatalf("generated %d tokens, want %d", len(g.Tokens), genTokens)
	}
}

// TestSetTokenizerNilIsSafe asserts arming a nil tokenizer is a no-op (the byte path
// stays selected), so a boot that resolves no real tokenizer never wedges the engine.
func TestSetTokenizerNilIsSafe(t *testing.T) {
	ctx := context.Background()
	e := New()
	e.SetTokenizer(nil)

	r, err := e.Complete(ctx, inlineCall("get_user_details", `{"id":1}`))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, ok := r.Meta["detokenized"]; ok {
		t.Fatalf("nil tokenizer must leave the byte-level default; got detokenized flag")
	}
}
