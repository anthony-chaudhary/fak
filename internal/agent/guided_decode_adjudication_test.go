package agent

// #929 end-to-end witness: the native masked decode in internal/model emits a
// tool call that ENTERS THE REAL grammar adjudicator UNCHANGED.
//
// #929 shipped the in-kernel sampler sink (internal/model/constraint.go): a
// logit-bias map + a flagged JSON-schema/grammar LogitMask applied before argmax,
// bit-exact-off when inert. Its own package test (constraint_test.go,
// TestSchemaMaskActiveDecodesValidJSON) proves the masked decode stays on the JSON
// path and PARSES — but it stops at parse and notes the adjudication step "lives in
// internal/grammar, exercised by the gateway witnesses". internal/model is tier-1
// and cannot import internal/grammar, so the last clause of #929's acceptance
// criterion 3 — "the emitted call ... ENTERS ADJUDICATION UNCHANGED" — could only be
// asserted by reference there, never witnessed in one process.
//
// internal/agent is the tier-2 seam that imports BOTH (model for the in-kernel
// planner, grammar for the lint/tool rungs), and the issue names internal/agent as
// the SampleParams.ResponseFormat/LogitBias carrier. So this is where the two halves
// meet: drive a real model.GenerateConstrained decode with a schema mask, take the
// tokens it emits, and feed the resulting tool call into the REAL grammar.Rung the
// gateway adjudicates with. It is the cross-layer witness the in-kernel package
// structurally can't author.

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// byteVocabDecodeCfg is the synthetic CPU-resident checkpoint shape the constraint
// tests use (vocab widened to a full byte alphabet so a token id IS a byte) — real
// KV/decode wiring, no weights file. EOSTokenID -1 never stops early, so the full
// masked continuation is emitted.
func byteVocabDecodeCfg() model.Config {
	return model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        256, // token id == byte, so a decode emits raw JSON bytes
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	}
}

// TestNativeMaskedDecodeEntersGrammarAdjudication is the #929 criterion-3 closure: a
// flag-on native masked decode emits the canonical {"name":...,"arguments":{...}}
// tool-call shape, and that exact emitted call enters the real grammar.Rung
// adjudicator UNCHANGED — VerdictDefer (well-formed: nothing to repair, nothing to
// deny). The mask is proven load-bearing (the unconstrained greedy path over the
// same model does NOT produce the valid call), and a malformed control proves the
// Defer is not vacuous (the same rung denies a call missing a required arg). This is
// the one process internal/model could not write: it cannot import internal/grammar.
func TestNativeMaskedDecodeEntersGrammarAdjudication(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1") // arm the flagged schema mask

	// The canonical tool-call shape #907's carrier rides and #929's sink must emit:
	// a tool name plus a required argument the grammar rung enforces.
	target := []byte(`{"name":"lookup","arguments":{"q":"sf"}}`)

	// Compile the constraint to a per-step token mask: at step i only target[i] is
	// allowed. This is the minimal real StepMask the (deferred, [STUB]) grammar->token
	// compiler will emit; here it is hand-built so the witness needs no compiler.
	per := make([]map[int]bool, len(target))
	for i, b := range target {
		per[i] = map[int]bool{int(b): true}
	}
	mask := &model.StepMask{PerStep: per}

	m := model.NewSynthetic(byteVocabDecodeCfg())
	prompt := []int{1, 2, 3}

	// Drive the REAL decode loop (model.Session.GenerateConstrained), not just the
	// sampler — every emitted token must come through the production Prefill/Step path.
	out := m.NewSession().GenerateConstrained(prompt, len(target), &model.DecodeConstraint{Mask: mask})
	emitted := bytesOf(out)
	if string(emitted) != string(target) {
		t.Fatalf("masked decode = %q, want canonical tool-call %q", emitted, target)
	}

	// The mask is load-bearing: the UNCONSTRAINED greedy path over the same model does
	// not produce the valid call, so it is the mask — not the (meaningless) logits —
	// that kept the decode on the schema path.
	natural := bytesOf(m.NewSession().Generate(prompt, len(target)))
	if string(natural) == string(target) {
		t.Fatal("mask vacuous: the unconstrained greedy decode already produced the valid call")
	}

	// The emitted bytes parse as the tool-call envelope and yield the inner arguments.
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(emitted, &call); err != nil {
		t.Fatalf("masked decode did not parse as a tool call: %v", err)
	}
	if call.Name != "lookup" {
		t.Fatalf("parsed tool name = %q, want \"lookup\"", call.Name)
	}

	// THE CLOSURE: feed the emitted call into the same grammar.Rung the gateway
	// adjudicates with. A grammar for "lookup" requiring "q" must DEFER it — the call
	// is well-formed, so it "enters adjudication unchanged" (no Transform, no Deny).
	r := grammar.New()
	if err := r.LoadFromJSONSchema("lookup",
		[]byte(`{"properties":{"q":{"type":"string"}},"required":["q"]}`)); err != nil {
		t.Fatalf("load grammar: %v", err)
	}
	v := r.Adjudicate(context.Background(), inlineToolCall(call.Name, call.Arguments))
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("native masked-decode call did not enter adjudication unchanged: verdict=%v, want Defer", v.Kind)
	}

	// Control: the Defer is not vacuous — the SAME rung denies a call that drops the
	// required "q", so the positive Defer above means the mask genuinely produced a
	// gate-passing call rather than the rung deferring on everything.
	bad := r.Adjudicate(context.Background(), inlineToolCall("lookup", []byte(`{}`)))
	if bad.Kind != abi.VerdictDeny {
		t.Fatalf("control: a call missing the required arg should Deny, got %v — adjudicator not load-bearing", bad.Kind)
	}
}

// bytesOf reinterprets a byte-vocab decode (token id == byte) as the raw bytes.
func bytesOf(ids []int) []byte {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = byte(id)
	}
	return b
}

// inlineToolCall builds a tool call carrying its arguments inline (the request side
// needs no resolver; only the repair TRANSFORM path stores output, which we never hit).
func inlineToolCall(tool string, args []byte) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: args},
	}
}
