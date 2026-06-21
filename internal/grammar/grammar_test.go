package grammar

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	// The repair path (VerdictTransform) calls putJSON, which goes through
	// abi.ActiveResolver(). Blank-import the blob backend so a Resolver is
	// registered by init() and putJSON can store the repaired args.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// resolveRef materializes the bytes a Ref points at, exercising the active
// resolver for non-inline refs (the repaired args may be inline or blob).
func resolveRef(t *testing.T, r abi.Ref) []byte {
	t.Helper()
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatalf("no active resolver registered (blob backend not imported?)")
	}
	b, err := res.Resolve(context.Background(), r)
	if err != nil {
		t.Fatalf("resolve repaired ref: %v", err)
	}
	return b
}

// inlineCall builds a tool call with inline JSON args (no resolver needed for
// the request side; the repair path stores its OUTPUT via putJSON).
func inlineCall(tool, argsJSON string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(argsJSON)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// Unit 53: LoadFromJSONSchema derives a grammar from an MCP-style JSON Schema
// and registers it, growing the unique-grammar count to >=1.
func TestLoadFromJSONSchema(t *testing.T) {
	r := New()
	if got := r.UniqueGrammarCount(); got != 0 {
		t.Fatalf("fresh rung UniqueGrammarCount = %d, want 0", got)
	}
	err := r.LoadFromJSONSchema("create_user",
		[]byte(`{"properties":{"name":{"type":"string"}},"required":["name"]}`))
	if err != nil {
		t.Fatalf("LoadFromJSONSchema: %v", err)
	}
	if got := r.UniqueGrammarCount(); got < 1 {
		t.Fatalf("UniqueGrammarCount = %d, want >= 1", got)
	}
}

// LoadFromJSONSchema surfaces malformed schema JSON as an error (defensive).
func TestLoadFromJSONSchemaBadJSON(t *testing.T) {
	r := New()
	if err := r.LoadFromJSONSchema("bad", []byte(`{not json`)); err == nil {
		t.Fatalf("expected error for malformed schema JSON, got nil")
	}
	if got := r.UniqueGrammarCount(); got != 0 {
		t.Fatalf("UniqueGrammarCount after failed load = %d, want 0", got)
	}
}

// Units 52 & 54: a positional call whose arity matches the grammar is a
// MECHANICAL repair: VerdictTransform / ReasonMisroute, with the positional
// values zipped into the grammar's named params.
func TestAdjudicatePositionalRepairable(t *testing.T) {
	r := New()
	if err := r.LoadFromJSONSchema("create_user",
		[]byte(`{"properties":{"name":{"type":"string"}},"required":["name"]}`)); err != nil {
		t.Fatalf("load: %v", err)
	}

	c := inlineCall("create_user", `{"_positional":["alice"]}`)
	v := r.Adjudicate(context.Background(), c)

	if v.Kind != abi.VerdictTransform {
		t.Fatalf("Kind = %v, want VerdictTransform", v.Kind)
	}
	if v.Reason != abi.ReasonMisroute {
		t.Fatalf("Reason = %v, want ReasonMisroute", v.Reason)
	}
	if v.By != "grammar" {
		t.Fatalf("By = %q, want \"grammar\"", v.By)
	}

	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("Payload type = %T, want abi.TransformPayload", v.Payload)
	}

	got := map[string]any{}
	if err := json.Unmarshal(resolveRef(t, tp.NewArgs), &got); err != nil {
		t.Fatalf("unmarshal repaired args: %v", err)
	}
	want := map[string]any{"name": "alice"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("repaired args = %v, want %v (positional should zip to named)", got, want)
	}

	// Forensics: a repair was counted.
	if repairs, _ := r.Stats(); repairs != 1 {
		t.Fatalf("repairs = %d, want 1", repairs)
	}
}

// Unit 54 (unrepairable): a positional call whose arity does NOT match the
// grammar (3 values vs 1 param) cannot be mechanically zipped, so it is a
// model-fixable refusal: VerdictDeny / ReasonMisroute.
func TestAdjudicatePositionalUnrepairable(t *testing.T) {
	r := New()
	if err := r.LoadFromJSONSchema("create_user",
		[]byte(`{"properties":{"name":{"type":"string"}},"required":["name"]}`)); err != nil {
		t.Fatalf("load: %v", err)
	}

	c := inlineCall("create_user", `{"_positional":["a","b","c"]}`)
	v := r.Adjudicate(context.Background(), c)

	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Kind = %v, want VerdictDeny", v.Kind)
	}
	if v.Reason != abi.ReasonMisroute {
		t.Fatalf("Reason = %v, want ReasonMisroute", v.Reason)
	}
	if v.By != "grammar" {
		t.Fatalf("By = %q, want \"grammar\"", v.By)
	}
	if _, denies := r.Stats(); denies != 1 {
		t.Fatalf("denies = %d, want 1", denies)
	}
}

// Unit 55: FAIL-OPEN. A tool with NO grammar loaded is not adjudicable here, so
// the rung defers (never over-refuses).
func TestAdjudicateNoGrammarDefers(t *testing.T) {
	r := New()
	c := inlineCall("unknown_tool", `{"_positional":["a","b"]}`)
	v := r.Adjudicate(context.Background(), c)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("Kind = %v, want VerdictDefer (fail-open)", v.Kind)
	}
	if v.By != "grammar" {
		t.Fatalf("By = %q, want \"grammar\"", v.By)
	}
	// Fail-open must never count as a refusal.
	if repairs, denies := r.Stats(); repairs != 0 || denies != 0 {
		t.Fatalf("Stats = (%d,%d), want (0,0) for fail-open defer", repairs, denies)
	}
}

// A well-formed named call (every required param present) has nothing to prove:
// VerdictDefer.
func TestAdjudicateWellFormedDefers(t *testing.T) {
	r := New()
	if err := r.LoadFromJSONSchema("create_user",
		[]byte(`{"properties":{"name":{"type":"string"}},"required":["name"]}`)); err != nil {
		t.Fatalf("load: %v", err)
	}
	c := inlineCall("create_user", `{"name":"alice"}`)
	v := r.Adjudicate(context.Background(), c)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("Kind = %v, want VerdictDefer for well-formed call", v.Kind)
	}
	if v.By != "grammar" {
		t.Fatalf("By = %q, want \"grammar\"", v.By)
	}
	if repairs, denies := r.Stats(); repairs != 0 || denies != 0 {
		t.Fatalf("Stats = (%d,%d), want (0,0) for well-formed defer", repairs, denies)
	}
}

// Argument-NAME-alias repair: the model used synonym keys (from/to) for the
// required canonical params (from_currency/to_currency). The rung renames them
// in-syscall (VerdictTransform / ReasonMisroute) so the call never bounces back
// as an error the model must spend a turn to fix.
func TestAdjudicateAliasRepair(t *testing.T) {
	r := New()
	r.Add("convert_currency", Grammar{
		Params: []Param{
			{Name: "from_currency", Type: "string", Required: true},
			{Name: "to_currency", Type: "string", Required: true},
			{Name: "amount", Type: "number", Required: true},
		},
		Aliases: map[string]string{"from": "from_currency", "to": "to_currency"},
	})

	c := inlineCall("convert_currency", `{"from":"USD","to":"EUR","amount":240}`)
	v := r.Adjudicate(context.Background(), c)

	if v.Kind != abi.VerdictTransform {
		t.Fatalf("Kind = %v, want VerdictTransform (alias repair)", v.Kind)
	}
	if v.Reason != abi.ReasonMisroute {
		t.Fatalf("Reason = %v, want ReasonMisroute", v.Reason)
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("Payload type = %T, want abi.TransformPayload", v.Payload)
	}
	got := map[string]any{}
	if err := json.Unmarshal(resolveRef(t, tp.NewArgs), &got); err != nil {
		t.Fatalf("unmarshal repaired args: %v", err)
	}
	want := map[string]any{"from_currency": "USD", "to_currency": "EUR", "amount": float64(240)}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("repaired args = %v, want %v (aliases should rename to canonical)", got, want)
	}
	if repairs, _ := r.Stats(); repairs != 1 {
		t.Fatalf("repairs = %d, want 1", repairs)
	}
}

// A canonical (already well-formed) call must NOT trigger an alias rename — the
// repair only fires when an alias actually closes a well-formedness gap.
func TestAdjudicateAliasNoFalseRepair(t *testing.T) {
	r := New()
	r.Add("convert_currency", Grammar{
		Params:  []Param{{Name: "from_currency", Type: "string", Required: true}},
		Aliases: map[string]string{"from": "from_currency"},
	})
	c := inlineCall("convert_currency", `{"from_currency":"USD"}`)
	if v := r.Adjudicate(context.Background(), c); v.Kind != abi.VerdictDefer {
		t.Fatalf("Kind = %v, want VerdictDefer (already well-formed)", v.Kind)
	}
	if repairs, _ := r.Stats(); repairs != 0 {
		t.Fatalf("repairs = %d, want 0 (no false repair)", repairs)
	}
}

// Unit 57: DEDUP. Adding the SAME grammar (identical content => identical
// digest) for two different tool names yields a single canonical entry, so
// UniqueGrammarCount stays 1. Both tool names route to that one grammar.
func TestAddDedup(t *testing.T) {
	r := New()
	g := Grammar{Params: []Param{{Name: "name", Type: "string", Required: true}}}

	r.Add("create_user", g)
	r.Add("make_account", g)

	if got := r.UniqueGrammarCount(); got != 1 {
		t.Fatalf("UniqueGrammarCount = %d, want 1 (identical grammars dedupe)", got)
	}

	// Both tool names resolve to the shared grammar: a well-formed call to
	// either defers, a positional repair on either transforms.
	for _, tool := range []string{"create_user", "make_account"} {
		c := inlineCall(tool, `{"name":"bob"}`)
		if v := r.Adjudicate(context.Background(), c); v.Kind != abi.VerdictDefer {
			t.Fatalf("tool %q well-formed: Kind = %v, want VerdictDefer", tool, v.Kind)
		}
	}
}
