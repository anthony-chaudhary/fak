package grammarintegration

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Keep model compatibility checks with the model integration witness so pure
// grammar tests do not import model and compute through their test closure.
var (
	_ model.LogitMask = (*grammar.CallMask)(nil)
	_ model.LogitMask = (*grammar.JSONSchemaMask)(nil)
	_ model.LogitMask = (*grammar.RegexMask)(nil)
)

type byteVocab struct{}

func (byteVocab) TokenBytes(id int) []byte {
	if id >= 0 && id < 256 {
		return []byte{byte(id)}
	}
	return nil
}

const (
	testEOS   = 256
	testVocab = 257
)

const lookupFreeSchema = `{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`

func mustCompile(t *testing.T, specs ...grammar.ToolSpec) *grammar.CallMask {
	t.Helper()
	m, err := grammar.Compile(specs, byteVocab{}, grammar.CompileOptions{EOS: testEOS})
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return m
}

func inlineCall(tool, argsJSON string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(argsJSON)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// TestCompiledMaskEndToEndDecode is the #2596 done-condition witness: a
// synthetic CPU model plus byte tokenizer emits the exact pinned tool call
// {"name":"lookup","arguments":{"q":"sf"}} through GenerateConstrained under
// FAK_NATIVE_GUIDED_DECODE=1; the emission enters grammar.Rung unchanged and
// defers as well-formed; a malformed control still denies; and with the flag
// off the SAME compiled-in mask is dormant (bit-exact with Session.Generate).
//
// The value "sf" is pinned with a const so every decode step has exactly one
// legal byte: the witness can assert EXACT bytes, and the natural-continuation
// comparison proves the mask (not the model) chose them. Free q:string decoding
// is exercised in the grammar unit tests — its content is model-chosen,
// so an exact-byte assertion there would assert synthetic-weight noise.
// fak-test:runtime fast est=1s
// fak-test:justify why=contract when=always
func TestCompiledMaskEndToEndDecode(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        testVocab,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       testEOS,
	}
	mdl := model.NewSynthetic(cfg)
	mask := mustCompile(t, grammar.ToolSpec{
		Name:   "lookup",
		Schema: []byte(`{"type":"object","properties":{"q":{"type":"string","const":"sf"}},"required":["q"]}`),
	})
	prompt := []int{7, 42, 199}
	const target = `{"name":"lookup","arguments":{"q":"sf"}}`

	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	out := mdl.NewSession().GenerateConstrained(prompt, len(target)+8, &model.DecodeConstraint{Mask: mask})
	if len(out) != len(target)+1 || out[len(out)-1] != testEOS {
		t.Fatalf("constrained decode = %v, want %d envelope bytes + EOS", out, len(target))
	}
	emitted := make([]byte, len(out)-1)
	for i, id := range out[:len(out)-1] {
		emitted[i] = byte(id)
	}
	if string(emitted) != target {
		t.Fatalf("emitted %q, want %q", emitted, target)
	}

	// The mask is load-bearing: the unconstrained greedy continuation differs.
	natural := mdl.NewSession().Generate(prompt, len(out))
	if equalIDs(natural, out) {
		t.Fatalf("unconstrained decode already emits the target; mask not load-bearing")
	}

	// Flag off: the same compiled-in mask is dormant — bit-exact with Generate.
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "0")
	dormant := mdl.NewSession().GenerateConstrained(prompt, 12, &model.DecodeConstraint{Mask: mask})
	plain := mdl.NewSession().Generate(prompt, 12)
	if !equalIDs(dormant, plain) {
		t.Fatalf("flag-off constrained decode diverged: %v vs %v", dormant, plain)
	}

	// The emission enters the Rung unchanged: well-formed => Defer.
	r := grammar.New()
	if err := r.LoadFromJSONSchema("lookup", []byte(lookupFreeSchema)); err != nil {
		t.Fatalf("LoadFromJSONSchema: %v", err)
	}
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(emitted, &call); err != nil {
		t.Fatalf("emitted envelope does not parse: %v", err)
	}
	if call.Name != "lookup" {
		t.Fatalf("emitted name = %q", call.Name)
	}
	if v := r.Adjudicate(context.Background(), inlineCall("lookup", string(call.Arguments))); v.Kind != abi.VerdictDefer {
		t.Fatalf("well-formed emission verdict = %+v, want Defer", v)
	}
	// Malformed control: a missing required param still denies (MISROUTE).
	if v := r.Adjudicate(context.Background(), inlineCall("lookup", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMisroute {
		t.Fatalf("malformed control verdict = %+v, want Deny(MISROUTE)", v)
	}
}

func equalIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
