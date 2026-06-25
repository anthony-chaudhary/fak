package abi

// shape_freeze_test.go pins the ABI message SHAPE — the field name/type set, IN
// DECLARATION ORDER, of every frozen wire struct — not just the numeric enum
// VALUES that TestABIGoldenFreeze already pins. The two are complementary: the
// enum freeze catches a renumbered Verdict kind; this freeze catches a renamed,
// reordered, removed, or retyped FIELD on ToolCall / Result / Verdict / Ref / …,
// the breaking changes a value-only golden cannot see.
//
// The freeze is ADDITIVE-ONLY, exactly like the types.go contract ("fields are
// only ever ADDED with zero-value defaults, never removed or repurposed"):
//
//   - APPENDING a new field to the end of a struct, or APPENDING a whole new wire
//     struct to the end of the witnessed set, is allowed (the golden is a strict
//     PREFIX of the live shape, so the new tail simply isn't compared).
//   - REORDERING, RENAMING, REMOVING, or RETYPING any existing field fails — the
//     prefix no longer matches the golden line for that position.
//   - A field inserted in the MIDDLE (not appended) shifts every later field and
//     is therefore correctly rejected: only true appends are additive.
//
// The shape is read by reflection over the live structs, so the golden can never
// drift from the code silently — a breaking edit to types.go reddens this test.
// Regenerate after an intentional additive change with UPDATE_GOLDEN=1.

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

// structShape is one frozen wire struct's field listing: the type's name and its
// fields as "Name: type" strings in declaration order (the order reflect.Field
// returns, which is the order they are written in types.go).
type structShape struct {
	Struct string   `json:"struct"`
	Fields []string `json:"fields"`
}

// wireShapeFreeze is the ordered set of structs whose SHAPE is frozen — the syscall
// message envelopes and the value types they embed. The order is itself part of the
// contract (a struct may be appended, never reordered or removed). reflect.TypeOf is
// taken over a zero value of each so no instance state is needed.
func wireShapeFreeze() []structShape {
	wire := []any{
		Ref{}, SpeculationContext{},
		ToolCall{}, Result{},
		Verdict{}, TransformPayload{}, QuarantinePayload{}, WitnessPayload{},
		Completion{}, SubmissionHandle{},
		Event{}, LabelRow{},
	}
	out := make([]structShape, 0, len(wire))
	for _, v := range wire {
		tp := reflect.TypeOf(v)
		shape := structShape{Struct: tp.String()}
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			shape.Fields = append(shape.Fields, f.Name+": "+f.Type.String())
		}
		out = append(out, shape)
	}
	return out
}

// TestABIShapeFreeze enforces the additive-only message-shape contract: the golden
// shape must be a strict PREFIX of the live shape, both in the struct list and in
// each struct's field list. Appending a field or a struct keeps the prefix intact
// (allowed); any reorder/rename/removal/retype breaks the prefix (fails).
func TestABIShapeFreeze(t *testing.T) {
	got := wireShapeFreeze()
	gotJSON, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatalf("marshal shape: %v", err)
	}

	const golden = "testdata/abi_shape_v0.1.golden"
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(golden, gotJSON, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	var want []structShape
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("decode golden: %v", err)
	}

	// Non-vacuity: the freeze must actually cover the core syscall envelopes, so a
	// future refactor that emptied wireShapeFreeze() cannot make this test pass
	// trivially.
	if len(want) < 8 {
		t.Fatalf("shape freeze is degenerate: golden has only %d structs (expected the full wire set)", len(want))
	}

	// Additive-only over the STRUCT list: the golden must be a prefix of the live
	// set. A struct removed or reordered shifts the prefix and fails here; a struct
	// appended to wireShapeFreeze() leaves the golden prefix intact.
	if len(got) < len(want) {
		t.Fatalf("a frozen wire struct was REMOVED: golden froze %d structs, live set has %d — removal breaks the additive-only freeze", len(want), len(got))
	}
	for i := range want {
		if got[i].Struct != want[i].Struct {
			t.Fatalf("wire struct %d changed: golden %q, live %q — reordering/renaming/removing a frozen struct breaks the freeze (only appending a new struct at the end is allowed)", i, want[i].Struct, got[i].Struct)
		}
		wf, gf := want[i].Fields, got[i].Fields
		if len(gf) < len(wf) {
			t.Fatalf("struct %s LOST a field: golden froze %d fields, live struct has %d — removing a field breaks the additive-only freeze", want[i].Struct, len(wf), len(gf))
		}
		for j := range wf {
			if gf[j] != wf[j] {
				t.Fatalf("struct %s field %d changed: golden %q, live %q — reordering/renaming/retyping a field breaks the freeze; only APPENDING a new field at the end is allowed", want[i].Struct, j, wf[j], gf[j])
			}
		}
	}
}
