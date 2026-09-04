package preflight

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// inlineCall builds a ToolCall with inline Args (no resolver needed).
func inlineCall(tool string, body string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(body)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// Unit 47 — rung 0: inline Args that are NOT valid JSON => Deny / Malformed.
func TestRung0MalformedJSONDenied(t *testing.T) {
	l := New()
	v := l.Adjudicate(context.Background(), inlineCall("anything", "{bad"))

	if v.Kind != abi.VerdictDeny {
		t.Fatalf("rung-0 bad JSON: got verdict kind %d, want VerdictDeny (%d)", v.Kind, abi.VerdictDeny)
	}
	if v.Reason != abi.ReasonMalformed {
		t.Fatalf("rung-0 bad JSON: got reason %d (%s), want ReasonMalformed", v.Reason, abi.ReasonName(v.Reason))
	}
	if v.By != "preflight" {
		t.Fatalf("rung-0 bad JSON: got By=%q, want preflight", v.By)
	}
}

// Unit 48 — rung 1: schema-typed validation. A wrong-typed required field is
// Denied/Malformed; a well-formed call with the right type passes (Defer).
func TestRung1SchemaTypeCheck(t *testing.T) {
	l := New()
	l.SetSchema("search_flights", Schema{Required: map[string]FieldType{"origin": TypeString}})

	// {"origin":123} — number where a string is required => Deny Malformed.
	bad := l.Adjudicate(context.Background(), inlineCall("search_flights", `{"origin":123}`))
	if bad.Kind != abi.VerdictDeny {
		t.Fatalf("rung-1 wrong type: got kind %d, want VerdictDeny", bad.Kind)
	}
	if bad.Reason != abi.ReasonMalformed {
		t.Fatalf("rung-1 wrong type: got reason %s, want MALFORMED", abi.ReasonName(bad.Reason))
	}

	// {"origin":"SFO"} — correct type, well-formed => Defer (rung has nothing to prove).
	good := l.Adjudicate(context.Background(), inlineCall("search_flights", `{"origin":"SFO"}`))
	if good.Kind != abi.VerdictDefer {
		t.Fatalf("rung-1 well-formed: got kind %d, want VerdictDefer (%d)", good.Kind, abi.VerdictDefer)
	}
}

// Unit 48b — a required field that is MISSING is also caught at rung 1.
func TestRung1MissingRequiredFieldDenied(t *testing.T) {
	l := New()
	l.SetSchema("search_flights", Schema{Required: map[string]FieldType{"origin": TypeString}})

	v := l.Adjudicate(context.Background(), inlineCall("search_flights", `{"dest":"JFK"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("rung-1 missing field: got kind %d reason %s, want Deny/MALFORMED", v.Kind, abi.ReasonName(v.Reason))
	}
}

// Unit 49 — ordering: a rung-0 (unparseable) failure produces a negative row with
// RungFailed==0 and RungPassed==-1 (rung 1 was never reached).
func TestRung0FailureNeverReachesRung1(t *testing.T) {
	l := New()
	// Install a schema so that IF rung-1 were (incorrectly) reached, RungFailed
	// would be 1, not 0. The unparseable args must short-circuit at rung 0.
	l.SetSchema("search_flights", Schema{Required: map[string]FieldType{"origin": TypeString}})

	v := l.Adjudicate(context.Background(), inlineCall("search_flights", "{bad"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("expected Deny on unparseable rung-0, got kind %d", v.Kind)
	}

	rows := l.Negatives()
	if len(rows) != 1 {
		t.Fatalf("expected exactly 1 negative row, got %d", len(rows))
	}
	row := decodeRow(t, rows[0])
	if row.RungFailed != 0 {
		t.Fatalf("ordering: rung_failed=%d, want 0 (failed at rung 0, never reached rung 1)", row.RungFailed)
	}
	if row.RungPassed != -1 {
		t.Fatalf("ordering: rung_passed=%d, want -1 (no rung was passed before the rung-0 catch)", row.RungPassed)
	}
}

// Unit 50 — Negatives() returns >=1 row after a catch; the row carries all the
// labeled fields (call_hash, rung_passed, rung_failed, verdict=="deny", reason).
func TestNegativesRowFields(t *testing.T) {
	l := New()
	l.SetSchema("search_flights", Schema{Required: map[string]FieldType{"origin": TypeString}})

	// One rung-1 catch (wrong type) so the row is the schema-failure shape.
	l.Adjudicate(context.Background(), inlineCall("search_flights", `{"origin":123}`))

	rows := l.Negatives()
	if len(rows) < 1 {
		t.Fatalf("Negatives(): got %d rows, want >=1", len(rows))
	}

	// Unmarshal into a generic map to assert the field KEYS are present.
	var generic map[string]json.RawMessage
	if err := json.Unmarshal(rows[0], &generic); err != nil {
		t.Fatalf("negative row is not valid JSON: %v", err)
	}
	for _, key := range []string{"call_hash", "rung_passed", "rung_failed", "verdict", "reason"} {
		if _, ok := generic[key]; !ok {
			t.Fatalf("negative row missing field %q; row=%s", key, rows[0])
		}
	}

	row := decodeRow(t, rows[0])
	if row.Verdict != "deny" {
		t.Fatalf("negative row verdict=%q, want \"deny\"", row.Verdict)
	}
	if row.Reason != "MALFORMED" {
		t.Fatalf("negative row reason=%q, want MALFORMED", row.Reason)
	}
	if row.RungPassed != 0 || row.RungFailed != 1 {
		t.Fatalf("rung-1 catch row: rung_passed=%d rung_failed=%d, want 0/1", row.RungPassed, row.RungFailed)
	}
	if row.CallHash == "" {
		t.Fatalf("negative row call_hash is empty")
	}
}

// Unit 51 — CatchRate() is correct after a mix of good + bad calls. Also asserts
// a fully well-formed call with no schema => VerdictDefer.
func TestCatchRateMix(t *testing.T) {
	l := New()
	l.SetSchema("search_flights", Schema{Required: map[string]FieldType{"origin": TypeString}})
	ctx := context.Background()

	// 2 bad calls (both caught):
	l.Adjudicate(ctx, inlineCall("search_flights", "{bad"))         // rung-0 catch
	l.Adjudicate(ctx, inlineCall("search_flights", `{"origin":1}`)) // rung-1 catch

	// 3 good calls (none caught):
	//  - well-formed + schema satisfied  => Defer
	good := l.Adjudicate(ctx, inlineCall("search_flights", `{"origin":"SFO"}`))
	if good.Kind != abi.VerdictDefer {
		t.Fatalf("schema-satisfied call: got kind %d, want Defer", good.Kind)
	}
	//  - well-formed, no schema for this tool => Defer (the rung has nothing to prove)
	noSchema := l.Adjudicate(ctx, inlineCall("weather", `{"city":"NYC"}`))
	if noSchema.Kind != abi.VerdictDefer {
		t.Fatalf("no-schema well-formed call: got kind %d, want VerdictDefer", noSchema.Kind)
	}
	if noSchema.By != "preflight" {
		t.Fatalf("defer verdict By=%q, want preflight", noSchema.By)
	}
	//  - empty args (no body) is well-formed too => Defer
	empty := l.Adjudicate(ctx, inlineCall("weather", ""))
	if empty.Kind != abi.VerdictDefer {
		t.Fatalf("empty-args call: got kind %d, want VerdictDefer", empty.Kind)
	}

	caught, total, rate := l.CatchRate()
	if total != 5 {
		t.Fatalf("CatchRate total=%d, want 5", total)
	}
	if caught != 2 {
		t.Fatalf("CatchRate caught=%d, want 2", caught)
	}
	if want := 2.0 / 5.0; rate != want {
		t.Fatalf("CatchRate rate=%v, want %v", rate, want)
	}
}

// TestCatchRateZeroTotal — fresh ladder has rate 0 (no divide-by-zero).
func TestCatchRateZeroTotal(t *testing.T) {
	l := New()
	caught, total, rate := l.CatchRate()
	if caught != 0 || total != 0 || rate != 0 {
		t.Fatalf("fresh ladder: caught=%d total=%d rate=%v, want 0/0/0", caught, total, rate)
	}
}

// rowView mirrors the JSON the ladder emits for a negative row.
type rowView struct {
	CallHash   string `json:"call_hash"`
	RungPassed int    `json:"rung_passed"`
	RungFailed int    `json:"rung_failed"`
	Verdict    string `json:"verdict"`
	Reason     string `json:"reason"`
}

func decodeRow(t *testing.T, b []byte) rowView {
	t.Helper()
	var r rowView
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("failed to unmarshal negative row %s: %v", b, err)
	}
	return r
}

// TestAllFieldTypesValidation verifies rung-1 type enforcement across all FieldType definitions.
func TestAllFieldTypesValidation(t *testing.T) {
	tests := []struct {
		name      string
		fieldType FieldType
		validJSON string
		badJSON   string
	}{
		{
			name:      "string",
			fieldType: TypeString,
			validJSON: `{"val":"hello"}`,
			badJSON:   `{"val":123}`,
		},
		{
			name:      "number",
			fieldType: TypeNumber,
			validJSON: `{"val":42.5}`,
			badJSON:   `{"val":"not-a-number"}`,
		},
		{
			name:      "boolean",
			fieldType: TypeBool,
			validJSON: `{"val":true}`,
			badJSON:   `{"val":"true"}`,
		},
		{
			name:      "object",
			fieldType: TypeObject,
			validJSON: `{"val":{"k":"v"}}`,
			badJSON:   `{"val":["not","an","object"]}`,
		},
		{
			name:      "array",
			fieldType: TypeArray,
			validJSON: `{"val":["item1", "item2"]}`,
			badJSON:   `{"val":{"not":"an-array"}}`,
		},
		{
			name:      "any",
			fieldType: TypeAny,
			validJSON: `{"val":"any-value"}`,
			badJSON:   "", // TypeAny accepts anything present
		},
		{
			name:      "unsupported_custom_fallback",
			fieldType: FieldType("custom_type"),
			validJSON: `{"val":"custom"}`,
			badJSON:   "", // unsupported types fall back to true/TypeAny behavior
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			l := New()
			l.SetSchema("test_tool", Schema{Required: map[string]FieldType{"val": tc.fieldType}})
			ctx := context.Background()

			// Valid payload must pass (VerdictDefer)
			good := l.Adjudicate(ctx, inlineCall("test_tool", tc.validJSON))
			if good.Kind != abi.VerdictDefer {
				t.Fatalf("%s valid: got kind %d, want VerdictDefer", tc.name, good.Kind)
			}

			// Invalid payload must be caught (VerdictDeny/ReasonMalformed)
			if tc.badJSON != "" {
				bad := l.Adjudicate(ctx, inlineCall("test_tool", tc.badJSON))
				if bad.Kind != abi.VerdictDeny || bad.Reason != abi.ReasonMalformed {
					t.Fatalf("%s invalid: got kind %d reason %s, want Deny/ReasonMalformed",
						tc.name, bad.Kind, abi.ReasonName(bad.Reason))
				}
			}
		})
	}
}

// TestRung0NonObjectJSONAndNull verifies that non-object JSON payloads and null
// are handled correctly according to the schema specification.
func TestRung0NonObjectJSONAndNull(t *testing.T) {
	ctx := context.Background()
	nonObjects := []string{
		`[1, 2, 3]`,
		`"just a string"`,
		`12345`,
		`true`,
		`   `, // whitespace without valid JSON
	}

	for _, payload := range nonObjects {
		t.Run(payload, func(t *testing.T) {
			l := New()
			v := l.Adjudicate(ctx, inlineCall("tool", payload))
			if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
				t.Fatalf("payload %q: want Deny/Malformed, got %d/%s", payload, v.Kind, abi.ReasonName(v.Reason))
			}
			rows := l.Negatives()
			if len(rows) != 1 {
				t.Fatalf("expected 1 negative row, got %d", len(rows))
			}
			row := decodeRow(t, rows[0])
			if row.RungFailed != 0 || row.RungPassed != -1 {
				t.Fatalf("payload %q: expected rung_passed=-1, rung_failed=0, got passed=%d failed=%d",
					payload, row.RungPassed, row.RungFailed)
			}
		})
	}

	// Null payload with no schema -> Defer
	lNoSchema := New()
	if v := lNoSchema.Adjudicate(ctx, inlineCall("tool", "null")); v.Kind != abi.VerdictDefer {
		t.Fatalf("null payload with no schema: got kind %d, want VerdictDefer", v.Kind)
	}

	// Null payload with schema requiring a field -> caught at rung 1 (missing field)
	lWithSchema := New()
	lWithSchema.SetSchema("tool", Schema{Required: map[string]FieldType{"key": TypeString}})
	v := lWithSchema.Adjudicate(ctx, inlineCall("tool", "null"))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("null payload with required field: got %d/%s, want Deny/Malformed", v.Kind, abi.ReasonName(v.Reason))
	}
	rows := lWithSchema.Negatives()
	if len(rows) != 1 {
		t.Fatalf("expected 1 negative row, got %d", len(rows))
	}
	row := decodeRow(t, rows[0])
	if row.RungFailed != 1 || row.RungPassed != 0 {
		t.Fatalf("null payload with required field: expected rung_passed=0, rung_failed=1, got passed=%d failed=%d",
			row.RungPassed, row.RungFailed)
	}
}

// TestSchemasDeepCopy verifies that Schemas() returns an isolated deep copy that
// cannot mutate the Ladder's internal schema state.
func TestSchemasDeepCopy(t *testing.T) {
	l := New()
	origReq := map[string]FieldType{
		"path":    TypeString,
		"retries": TypeNumber,
	}
	l.SetSchema("fetch", Schema{Required: origReq})

	gotSchemas := l.Schemas()
	wantSchemas := map[string]Schema{
		"fetch": {Required: map[string]FieldType{
			"path":    TypeString,
			"retries": TypeNumber,
		}},
	}
	if !reflect.DeepEqual(gotSchemas, wantSchemas) {
		t.Fatalf("Schemas() = %+v, want %+v", gotSchemas, wantSchemas)
	}

	// Mutate top-level map returned by Schemas()
	gotSchemas["new_tool"] = Schema{Required: map[string]FieldType{"dummy": TypeBool}}
	// Mutate nested Required map returned by Schemas()
	gotSchemas["fetch"].Required["path"] = TypeBool
	gotSchemas["fetch"].Required["extra"] = TypeArray

	// Second call to Schemas() must remain unchanged
	freshSchemas := l.Schemas()
	if !reflect.DeepEqual(freshSchemas, wantSchemas) {
		t.Fatalf("internal schemas were mutated! got %+v, want %+v", freshSchemas, wantSchemas)
	}

	// Verify ladder adjudication still uses original schema types
	ctx := context.Background()
	goodCall := l.Adjudicate(ctx, inlineCall("fetch", `{"path":"/url","retries":3}`))
	if goodCall.Kind != abi.VerdictDefer {
		t.Fatalf("expected Defer on original types, got %d", goodCall.Kind)
	}
	badCall := l.Adjudicate(ctx, inlineCall("fetch", `{"path":true,"retries":3}`))
	if badCall.Kind != abi.VerdictDeny {
		t.Fatalf("expected Deny on path=true (path is TypeString), got %d", badCall.Kind)
	}
}

// TestDefaultLadderRegistrationAndCaps verifies package initialization invariants.
func TestDefaultLadderRegistrationAndCaps(t *testing.T) {
	if Default == nil {
		t.Fatal("Default preflight.Ladder must not be nil")
	}
	if caps := Default.Caps(); caps != nil {
		t.Fatalf("Default.Caps() = %v, want nil", caps)
	}
	// Verify Default can adjudicate properly without panicking
	ctx := context.Background()
	vGood := Default.Adjudicate(ctx, inlineCall("unregistered_tool", `{"ok":true}`))
	if vGood.Kind != abi.VerdictDefer {
		t.Fatalf("Default.Adjudicate well-formed: got kind %d, want VerdictDefer", vGood.Kind)
	}
	vBad := Default.Adjudicate(ctx, inlineCall("unregistered_tool", `{bad`))
	if vBad.Kind != abi.VerdictDeny {
		t.Fatalf("Default.Adjudicate bad JSON: got kind %d, want VerdictDeny", vBad.Kind)
	}
}

// TestNewWithLimitEdgeCases checks non-positive limits fallback to DefaultMaxNegatives.
func TestNewWithLimitEdgeCases(t *testing.T) {
	for _, lim := range []int{0, -1, -100} {
		l := NewWithLimit(lim)
		if l.maxNeg != DefaultMaxNegatives {
			t.Fatalf("NewWithLimit(%d).maxNeg = %d, want DefaultMaxNegatives %d",
				lim, l.maxNeg, DefaultMaxNegatives)
		}
	}
	lStd := New()
	if lStd.maxNeg != DefaultMaxNegatives {
		t.Fatalf("New().maxNeg = %d, want DefaultMaxNegatives %d", lStd.maxNeg, DefaultMaxNegatives)
	}
}

// TestMultipleRequiredFields verifies schema validation when multiple fields of various types are required.
func TestMultipleRequiredFields(t *testing.T) {
	l := New()
	l.SetSchema("batch_op", Schema{Required: map[string]FieldType{
		"id":      TypeString,
		"count":   TypeNumber,
		"dry_run": TypeBool,
		"items":   TypeArray,
		"options": TypeObject,
	}})
	ctx := context.Background()

	// All fields present and correct type
	valid := `{"id":"b1","count":10,"dry_run":false,"items":["a","b"],"options":{"verbose":true}}`
	if v := l.Adjudicate(ctx, inlineCall("batch_op", valid)); v.Kind != abi.VerdictDefer {
		t.Fatalf("multi-field valid: got %d, want VerdictDefer", v.Kind)
	}

	// Missing one field ("dry_run")
	missing := `{"id":"b1","count":10,"items":["a","b"],"options":{"verbose":true}}`
	if v := l.Adjudicate(ctx, inlineCall("batch_op", missing)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("multi-field missing: got %d/%s, want Deny/ReasonMalformed", v.Kind, abi.ReasonName(v.Reason))
	}

	// One field wrong type ("items" as string instead of array)
	wrongType := `{"id":"b1","count":10,"dry_run":false,"items":"not-array","options":{"verbose":true}}`
	if v := l.Adjudicate(ctx, inlineCall("batch_op", wrongType)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("multi-field wrong type: got %d/%s, want Deny/ReasonMalformed", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestConcurrentLadderAdjudicationAndMutation exercises the Ladder under concurrent reads,
// writes, and adjudications to verify thread safety and absence of race conditions.
func TestConcurrentLadderAdjudicationAndMutation(t *testing.T) {
	const goroutines = 30
	const iterations = 50
	l := NewWithLimit(64)

	var wg sync.WaitGroup
	ctx := context.Background()

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			toolName := fmt.Sprintf("tool_%d", id%5)

			for i := 0; i < iterations; i++ {
				// Periodically update or query schema
				if i%10 == 0 {
					l.SetSchema(toolName, Schema{Required: map[string]FieldType{
						"req_id": TypeString,
					}})
				}
				if i%7 == 0 {
					_ = l.Schemas()
				}
				if i%5 == 0 {
					_, _, _ = l.CatchRate()
					_ = l.NegativesLen()
					_ = l.Evicted()
					_ = l.Negatives()
				}

				// Adjudicate mix of good, malformed rung-0, and wrong-type rung-1
				switch i % 3 {
				case 0:
					// Good call
					_ = l.Adjudicate(ctx, inlineCall(toolName, fmt.Sprintf(`{"req_id":"id-%d-%d"}`, id, i)))
				case 1:
					// Malformed JSON (rung-0)
					_ = l.Adjudicate(ctx, inlineCall(toolName, "{unclosed-json"))
				case 2:
					// Wrong type for req_id (rung-1)
					_ = l.Adjudicate(ctx, inlineCall(toolName, fmt.Sprintf(`{"req_id":%d}`, i)))
				}
			}
		}(g)
	}

	wg.Wait()

	caught, total, rate := l.CatchRate()
	if total != int64(goroutines*iterations) {
		t.Fatalf("total calls = %d, want %d", total, goroutines*iterations)
	}
	if caught <= 0 || caught > total {
		t.Fatalf("caught calls = %d out of %d (rate=%.3f)", caught, total, rate)
	}
	if resLen := l.NegativesLen(); resLen > 64 {
		t.Fatalf("NegativesLen = %d, exceeded cap 64", resLen)
	}
}

// TestDogfoodRuntimeExecution simulates a real-world multi-turn tool invocation trace
// from an autonomous agent session, verifying complete telemetry, hard-negative label
// rows, and deterministic rung-ladder progression.
func TestDogfoodRuntimeExecution(t *testing.T) {
	l := NewWithLimit(20)

	// Configure real agent tools
	l.SetSchema("read_file", Schema{Required: map[string]FieldType{
		"path": TypeString,
	}})
	l.SetSchema("write_file", Schema{Required: map[string]FieldType{
		"path":    TypeString,
		"content": TypeString,
	}})
	l.SetSchema("run_command", Schema{Required: map[string]FieldType{
		"command": TypeString,
		"timeout": TypeNumber,
	}})

	ctx := context.Background()

	type step struct {
		name        string
		tool        string
		args        string
		wantVerdict abi.VerdictKind
		wantReason  abi.ReasonCode
		wantPassed  int
		wantFailed  int
	}

	steps := []step{
		{
			name:        "valid file read",
			tool:        "read_file",
			args:        `{"path":"repo/main.go"}`,
			wantVerdict: abi.VerdictDefer,
		},
		{
			name:        "syntax error truncated JSON from LLM",
			tool:        "read_file",
			args:        `{"path":"repo/ma`,
			wantVerdict: abi.VerdictDeny,
			wantReason:  abi.ReasonMalformed,
			wantPassed:  -1,
			wantFailed:  0,
		},
		{
			name:        "type error path is integer",
			tool:        "read_file",
			args:        `{"path":1234}`,
			wantVerdict: abi.VerdictDeny,
			wantReason:  abi.ReasonMalformed,
			wantPassed:  0,
			wantFailed:  1,
		},
		{
			name:        "valid file write",
			tool:        "write_file",
			args:        `{"path":"repo/main.go","content":"package main"}`,
			wantVerdict: abi.VerdictDefer,
		},
		{
			name:        "write missing required content field",
			tool:        "write_file",
			args:        `{"path":"repo/main.go"}`,
			wantVerdict: abi.VerdictDeny,
			wantReason:  abi.ReasonMalformed,
			wantPassed:  0,
			wantFailed:  1,
		},
		{
			name:        "run_command with boolean timeout instead of number",
			tool:        "run_command",
			args:        `{"command":"go test ./...","timeout":false}`,
			wantVerdict: abi.VerdictDeny,
			wantReason:  abi.ReasonMalformed,
			wantPassed:  0,
			wantFailed:  1,
		},
		{
			name:        "unregistered tool with valid json passes through",
			tool:        "agent_status",
			args:        `{"progress":"halfway","done":false}`,
			wantVerdict: abi.VerdictDefer,
		},
		{
			name:        "empty args pass through as well-formed body",
			tool:        "no_arg_tool",
			args:        "",
			wantVerdict: abi.VerdictDefer,
		},
	}

	for _, s := range steps {
		t.Run(s.name, func(t *testing.T) {
			call := inlineCall(s.tool, s.args)
			v := l.Adjudicate(ctx, call)
			if v.Kind != s.wantVerdict {
				t.Fatalf("[%s] got verdict kind %d, want %d", s.name, v.Kind, s.wantVerdict)
			}
			if s.wantVerdict == abi.VerdictDeny && v.Reason != s.wantReason {
				t.Fatalf("[%s] got reason %s, want %s", s.name, abi.ReasonName(v.Reason), abi.ReasonName(s.wantReason))
			}
		})
	}

	caught, total, rate := l.CatchRate()
	if total != 8 {
		t.Fatalf("total calls = %d, want 8", total)
	}
	if caught != 4 {
		t.Fatalf("caught calls = %d, want 4", caught)
	}
	if rate != 0.5 {
		t.Fatalf("catch rate = %f, want 0.5", rate)
	}

	// Verify the negative label rows generated by the dogfood run
	rows := l.Negatives()
	if len(rows) != 4 {
		t.Fatalf("Negatives count = %d, want 4", len(rows))
	}

	// Check the first negative row (the truncated JSON syntax error)
	row0 := decodeRow(t, rows[0])
	if row0.Verdict != "deny" || row0.Reason != "MALFORMED" || row0.RungPassed != -1 || row0.RungFailed != 0 {
		t.Fatalf("row0 = %+v, want deny/MALFORMED/passed=-1/failed=0", row0)
	}

	// Check the second negative row (the type error on path)
	row1 := decodeRow(t, rows[1])
	if row1.Verdict != "deny" || row1.Reason != "MALFORMED" || row1.RungPassed != 0 || row1.RungFailed != 1 {
		t.Fatalf("row1 = %+v, want deny/MALFORMED/passed=0/failed=1", row1)
	}
}

func testIsSubstantiveContractComment(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}
	text := strings.TrimSpace(cg.Text())
	if len(text) < 35 {
		return false
	}
	lower := strings.ToLower(text)

	hasContractMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "assumption:") ||
		strings.Contains(lower, "assumptions:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.Contains(lower, "precondition:") ||
		strings.Contains(lower, "postcondition:") ||
		strings.Contains(lower, "guard:")
	if !hasContractMarker {
		return false
	}

	words := strings.Fields(lower)
	if len(words) < 6 {
		return false
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.4 {
		return false
	}
	return true
}

func testSplitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func testIsTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := testSplitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}

func testIsSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil || len(doc.List) == 0 {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if len(text) < 12 {
		return false
	}
	return !testIsTautologicalDoc(name, text)
}

// TestPreflightMaturityDocumentationAndContracts verifies that internal/preflight meets
// debtlane maturity requirements: substantive contract comments and at least 80% exported
// symbol documentation coverage.
func TestPreflightMaturityDocumentationAndContracts(t *testing.T) {
	files := []string{"preflight.go", "workspace.go"}
	fset := token.NewFileSet()

	for _, filename := range files {
		path := filepath.Join(".", filename)
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("failed to read %s: %v", path, err)
		}

		node, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", path, err)
		}

		// Verify contract comments presence
		contractCommentsCount := 0
		for _, cg := range node.Comments {
			if testIsSubstantiveContractComment(cg) {
				contractCommentsCount++
			}
		}
		if contractCommentsCount == 0 {
			t.Errorf("%s: expected at least one substantive contract comment, got none", filename)
		}

		// Count exported symbols and documentation
		exported := 0
		documented := 0
		var undocumented []string

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					exported++
					if testIsSubstantiveDoc(d.Name.Name, d.Doc) {
						documented++
					} else {
						undocumented = append(undocumented, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							exported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if testIsSubstantiveDoc(s.Name.Name, doc) {
								documented++
							} else {
								undocumented = append(undocumented, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								exported++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if testIsSubstantiveDoc(name.Name, doc) {
									documented++
								} else {
									undocumented = append(undocumented, name.Name)
								}
							}
						}
					}
				}
			}
		}

		ratio := float64(documented) / float64(exported)
		if ratio < 0.8 {
			t.Errorf("%s: documentation coverage %.1f%% (%d/%d) is below 80%% threshold. Undocumented: %v",
				filename, ratio*100, documented, exported, undocumented)
		}
	}
}
