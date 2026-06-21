package preflight

import (
	"context"
	"encoding/json"
	"testing"

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
