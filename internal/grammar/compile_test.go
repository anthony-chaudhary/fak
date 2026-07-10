package grammar

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// The compiler's output must satisfy model.LogitMask STRUCTURALLY — the
// production file never imports internal/model; this assertion is the proof.
var _ model.LogitMask = (*CallMask)(nil)

// byteVocab is the test TokenDecoder: ids 0..255 decode to their byte, id 256
// is EOS and decodes to nothing (like a real special token).
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

func mustCompile(t *testing.T, specs ...ToolSpec) *CallMask {
	t.Helper()
	m, err := Compile(specs, byteVocab{}, CompileOptions{EOS: testEOS})
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return m
}

func byteIDs(s string) []int {
	ids := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		ids[i] = int(s[i])
	}
	return ids
}

// admitted runs MaskLogits directly (the FAK_NATIVE_GUIDED_DECODE gate lives in
// model.DecodeConstraint, not in the mask itself) and reports which token ids
// survive at the given emitted prefix.
func admitted(t *testing.T, m *CallMask, prefix string) map[int]bool {
	t.Helper()
	logits := make([]float32, testVocab)
	m.MaskLogits(byteIDs(prefix), logits)
	out := map[int]bool{}
	for id, v := range logits {
		if !math.IsInf(float64(v), -1) {
			out[id] = true
		}
	}
	return out
}

// TestCompileCanonicalOneOfSchema pins the normalization half of #2596: specs
// lower to the canonical oneOf tool-call JSON Schema with byte-deterministic
// output, content-addressed the same way the Rung dedups grammars.
func TestCompileCanonicalOneOfSchema(t *testing.T) {
	lookup := ToolSpec{Name: "lookup", Schema: []byte(lookupFreeSchema)}
	m := mustCompile(t, lookup)
	want := `{"oneOf":[{"type":"object","properties":{"name":{"const":"lookup"},` +
		`"arguments":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}},` +
		`"required":["name","arguments"],"additionalProperties":false}]}`
	if got := string(m.CanonicalSchema()); got != want {
		t.Fatalf("canonical schema:\n got %s\nwant %s", got, want)
	}
	if !json.Valid(m.CanonicalSchema()) {
		t.Fatalf("canonical schema is not valid JSON")
	}

	// An identical duplicate spec dedups to the same single-branch schema.
	if d := mustCompile(t, lookup, lookup); d.Digest() != m.Digest() {
		t.Fatalf("duplicate spec changed digest: %s vs %s", d.Digest(), m.Digest())
	}

	// Spec order does not matter (sorted normalization); a different tool set
	// gets a different digest.
	other := ToolSpec{Name: "get_time", Schema: []byte(`{"properties":{"tz":{"type":"string"}},"required":["tz"]}`)}
	ab := mustCompile(t, lookup, other)
	ba := mustCompile(t, other, lookup)
	if ab.Digest() != ba.Digest() {
		t.Fatalf("spec order changed digest: %s vs %s", ab.Digest(), ba.Digest())
	}
	if ab.Digest() == m.Digest() {
		t.Fatalf("different tool sets share digest %s", m.Digest())
	}
}

// TestCompileUnsupportedFeaturesAreNamed pins the honesty contract: every
// schema feature outside the v0 surface is refused with ErrUnsupportedSchema
// and an error that NAMES the feature — never lowered to a weaker mask.
func TestCompileUnsupportedFeaturesAreNamed(t *testing.T) {
	cases := []struct {
		name, schema, wantSub string
	}{
		{"nested object", `{"properties":{"profile":{"type":"object"}},"required":["profile"]}`, `"profile"`},
		{"array", `{"properties":{"tags":{"type":"array"}},"required":["tags"]}`, `type "array"`},
		{"optional property", `{"properties":{"q":{"type":"string"},"opt":{"type":"string"}},"required":["q"]}`, "optional property"},
		{"pattern keyword", `{"properties":{"q":{"type":"string","pattern":"^a"}},"required":["q"]}`, `keyword "pattern"`},
		{"top-level oneOf", `{"oneOf":[]}`, `top-level keyword "oneOf"`},
		{"additionalProperties true", `{"additionalProperties":true,"properties":{"q":{"type":"string"}},"required":["q"]}`, "additionalProperties"},
		{"enum needing escapes", `{"properties":{"q":{"type":"string","enum":["a\"b"]}},"required":["q"]}`, "escaping"},
		{"integer enum", `{"properties":{"n":{"type":"integer","enum":[1]}},"required":["n"]}`, "enum/const"},
		{"missing type", `{"properties":{"q":{}},"required":["q"]}`, "without a type"},
	}
	for _, tc := range cases {
		_, err := Compile([]ToolSpec{{Name: "t", Schema: []byte(tc.schema)}}, byteVocab{}, CompileOptions{EOS: testEOS})
		if !errors.Is(err, ErrUnsupportedSchema) {
			t.Fatalf("%s: err = %v, want ErrUnsupportedSchema", tc.name, err)
		}
		if !strings.Contains(err.Error(), tc.wantSub) {
			t.Fatalf("%s: error %q does not name the feature (%q)", tc.name, err, tc.wantSub)
		}
	}

	// Plain (non-feature) errors stay plain errors.
	if _, err := Compile(nil, byteVocab{}, CompileOptions{EOS: testEOS}); err == nil {
		t.Fatalf("empty spec list compiled")
	}
	bad := `{"properties":{"q":{"type":"string"}},"required":["q","missing"]}`
	if _, err := Compile([]ToolSpec{{Name: "t", Schema: []byte(bad)}}, byteVocab{}, CompileOptions{EOS: testEOS}); err == nil {
		t.Fatalf("undeclared required property compiled")
	}
	a := ToolSpec{Name: "t", Schema: []byte(lookupFreeSchema)}
	b := ToolSpec{Name: "t", Schema: []byte(`{"properties":{"z":{"type":"string"}},"required":["z"]}`)}
	if _, err := Compile([]ToolSpec{a, b}, byteVocab{}, CompileOptions{EOS: testEOS}); err == nil {
		t.Fatalf("same name with different schemas compiled")
	}
}

// TestCallMaskByteAdmission is the unit half of the #2596 witness: a lookup
// schema requiring q:string compiles into a mask whose admission sets force the
// envelope skeleton, leave string content free (minus escapes), branch the
// name/enum regions, and admit only EOS at a complete envelope. The skeleton
// assertions double as the guideddecode composition pin: if envPre/envSuf ever
// drifted from guideddecode's envelope, the hand-off prefix would stop
// admitting '{'.
func TestCallMaskByteAdmission(t *testing.T) {
	m := mustCompile(t,
		ToolSpec{Name: "lookup", Schema: []byte(lookupFreeSchema)},
		ToolSpec{Name: "get_time", Schema: []byte(`{"properties":{"tz":{"type":"string","enum":["utc","local"]}},"required":["tz"]}`)},
	)

	// Name region branches across both declared tools and nothing else.
	a := admitted(t, m, `{"name":"`)
	if !a['g'] || !a['l'] || a['z'] || a[testEOS] {
		t.Fatalf("name branch admission = %v", a)
	}

	// Skeleton hand-off: guideddecode goes unconstrained (nil) exactly here and
	// the argument FSM must take over with the opening brace alone.
	forced := []struct{ prefix, next string }{
		{`{"name":"lookup","arguments":`, `{`},
		{`{"name":"lookup","arguments":{`, `"`},
		{`{"name":"lookup","arguments":{"q":`, `"`},
		{`{"name":"lookup","arguments":{"q":"sf"`, `}`},
		{`{"name":"lookup","arguments":{"q":"sf"}`, `}`},
	}
	for _, f := range forced {
		a := admitted(t, m, f.prefix)
		if len(a) != 1 || !a[int(f.next[0])] {
			t.Fatalf("at %q admission = %v, want only %q", f.prefix, a, f.next)
		}
	}

	// Free string content: open choice, but never escapes, controls, or EOS;
	// the closing quote is always available (empty content is legal).
	a = admitted(t, m, `{"name":"lookup","arguments":{"q":"`)
	if !a['s'] || !a['"'] || !a[' '] || a['\\'] || a[0x01] || a[testEOS] {
		t.Fatalf("string content admission = %v", a)
	}

	// Enum region: only viable variant bytes ("utc" | "local").
	a = admitted(t, m, `{"name":"get_time","arguments":{"tz":"`)
	if !a['u'] || !a['l'] || a['x'] || a['"'] {
		t.Fatalf("enum branch admission = %v", a)
	}
	a = admitted(t, m, `{"name":"get_time","arguments":{"tz":"u`)
	if len(a) != 1 || !a['t'] {
		t.Fatalf("enum tail admission = %v, want only 't'", a)
	}

	// Complete envelope: EOS and nothing else.
	a = admitted(t, m, `{"name":"lookup","arguments":{"q":"sf"}}`)
	if len(a) != 1 || !a[testEOS] {
		t.Fatalf("complete-envelope admission = %v, want only EOS", a)
	}

	// A dead-end prefix (unreachable under masked decoding) declines to mask,
	// mirroring model.GuidedByteMask.
	if a := admitted(t, m, `garbage`); len(a) != testVocab {
		t.Fatalf("dead-end prefix masked %d ids", testVocab-len(a))
	}
}

// TestCallMaskScalarValues walks a four-scalar tool (boolean, integer, string
// const, number) through every byte of a canonical envelope, then probes the
// number-boundary cases where a value has no delimiter of its own.
func TestCallMaskScalarValues(t *testing.T) {
	calc := `{"type":"object","properties":{` +
		`"flag":{"type":"boolean"},"n":{"type":"integer"},` +
		`"tag":{"type":"string","const":"a"},"x":{"type":"number"}},` +
		`"required":["flag","n","tag","x"]}`
	m := mustCompile(t, ToolSpec{Name: "calc", Schema: []byte(calc)})

	// Canonical (sorted) field order: flag, n, tag, x.
	target := `{"name":"calc","arguments":{"flag":true,"n":-12,"tag":"a","x":3.5}}`
	for i := 0; i < len(target); i++ {
		if a := admitted(t, m, target[:i]); !a[int(target[i])] {
			t.Fatalf("byte %q masked after %q", target[i], target[:i])
		}
	}
	if a := admitted(t, m, target); len(a) != 1 || !a[testEOS] {
		t.Fatalf("complete scalar envelope admission = %v, want only EOS", a)
	}

	// Boolean region: exactly the two literal heads.
	a := admitted(t, m, `{"name":"calc","arguments":{"flag":`)
	if !a['t'] || !a['f'] || a['1'] || a['"'] {
		t.Fatalf("boolean admission = %v", a)
	}

	// Integer boundary: after -12 another digit or the field separator is
	// legal; a fraction dot and a second sign are not.
	a = admitted(t, m, `{"name":"calc","arguments":{"flag":true,"n":-12`)
	if !a['3'] || !a[','] || a['.'] || a['-'] {
		t.Fatalf("integer boundary admission = %v", a)
	}

	// Canonical numbers: no leading zeros — after 0 the integer part is closed.
	a = admitted(t, m, `{"name":"calc","arguments":{"flag":true,"n":0`)
	if a['0'] || a['9'] || !a[','] {
		t.Fatalf("leading-zero admission = %v", a)
	}

	// Number field: the fraction dot IS legal, as is closing the envelope.
	a = admitted(t, m, `{"name":"calc","arguments":{"flag":true,"n":-12,"tag":"a","x":3`)
	if !a['.'] || !a['}'] || !a['0'] || a['-'] {
		t.Fatalf("number boundary admission = %v", a)
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
// is exercised at the byte-admission layer above — its content is model-chosen,
// so an exact-byte assertion there would assert synthetic-weight noise.
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
	mask := mustCompile(t, ToolSpec{
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
	r := New()
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
