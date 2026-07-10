package memq

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// TestNarrowSecondAxisComposesWithBudget is the #4019 landing witness: a
// boilerplate-heavy cell that a byte budget would tail-drop WHOLE is instead kept —
// with its query-relevant key field intact — once an opt-in OpNarrow slims it first.
// The folded remainder rides an audit Effect (the OpDedup discipline), the backend
// snapshot is never mutated, and the whole pass is deterministic.
func TestNarrowSecondAxisComposesWithBudget(t *testing.T) {
	ctx := context.Background()
	mk := func() (fixedBackend, map[string]string) {
		attrs := map[string]string{
			"amount":      "refund_fee 25 EUR",                      // query-relevant key field — must survive
			"boilerplate": strings.Repeat("lorem ipsum dolor ", 10), // irrelevant filler — must fold
		}
		seed := []Cell{
			{ID: "wide", Step: 0, Descriptor: "tool: refund_fee 25 EUR charged", Durability: "durable", Attrs: attrs},
			{ID: "small", Step: 1, Descriptor: "note-2", Bytes: 60, Durability: "durable"},
		}
		seed[0].Bytes = cellWidth(seed[0]) // a notes-style cell: its cost IS its metadata width
		return fixedBackend{cells: seed}, attrs
	}

	backend, _ := mk()
	if w := cellWidth(backend.cells[0]); w <= 200 {
		t.Fatalf("fixture cell width = %d, want > the 200-byte budget (non-vacuous precondition failed)", w)
	}

	// Control: WITHOUT narrow, the wide cell is dropped whole.
	control := Query{Intent: "refund fee", Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 200}}}
	rc, err := Run(ctx, backend, control, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if rc.Overflow == nil || len(rc.Overflow.Dropped) != 1 || rc.Overflow.Dropped[0].ID != "wide" {
		t.Fatalf("control run should tail-drop the wide cell whole, got overflow %+v", rc.Overflow)
	}

	// With the width axis chained before the budget, the wide cell is narrowed and KEPT.
	backend2, attrs := mk()
	q := Query{Intent: "refund fee", Ops: []Op{{Kind: OpScan}, {Kind: OpNarrow, Bytes: 120}, {Kind: OpBudget, Bytes: 200}}}
	r, err := Run(ctx, backend2, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Overflow != nil {
		t.Fatalf("narrowed set still overflowed the budget: %+v", r.Overflow)
	}
	if got := ids(r.Working); !reflect.DeepEqual(got, []string{"wide", "small"}) {
		t.Fatalf("working set = %v, want [wide small] (the narrowed cell must survive the budget)", got)
	}
	wide := r.Working[0]
	if wide.Attrs["amount"] != "refund_fee 25 EUR" {
		t.Errorf("the query-relevant key field was pruned: attrs=%v", wide.Attrs)
	}
	if _, ok := wide.Attrs["boilerplate"]; ok {
		t.Error("the irrelevant boilerplate field survived narrowing")
	}
	wantBytes := int64(len(wide.Descriptor) + len("amount") + len("refund_fee 25 EUR"))
	if wide.Bytes != wantBytes {
		t.Errorf("narrowed Bytes = %d, want the projected width %d", wide.Bytes, wantBytes)
	}
	if wide.Bytes > 120 {
		t.Errorf("narrowed Bytes = %d exceeds the 120-byte per-cell cap", wide.Bytes)
	}
	if wide.Descriptor != "tool: refund_fee 25 EUR charged" {
		t.Errorf("descriptor was trimmed though dropping fields sufficed: %q", wide.Descriptor)
	}

	// The folded remainder is audited, dedup-style.
	var eff *Effect
	for i := range r.Effects {
		if r.Effects[i].Kind == OpNarrow {
			eff = &r.Effects[i]
		}
	}
	if eff == nil {
		t.Fatal("no narrow Effect recorded")
	}
	if !reflect.DeepEqual(eff.Cells, []string{"wide"}) {
		t.Errorf("narrow effect cells = %v, want [wide]", eff.Cells)
	}
	if !strings.Contains(eff.Note, "-boilerplate") {
		t.Errorf("narrow effect note does not name the folded field: %q", eff.Note)
	}
	if eff.Applied {
		t.Error("narrow reports Applied=true, but it is a read-only working-set projection")
	}
	if r.Stats.Narrowed != 1 {
		t.Errorf("Stats.Narrowed = %d, want 1", r.Stats.Narrowed)
	}

	// Read-only: the backend snapshot's shared Attrs map still carries the folded field.
	if attrs["boilerplate"] == "" {
		t.Error("narrow mutated the backend's shared Attrs map")
	}

	// Deterministic: an identical re-run yields a deep-equal Result.
	backend3, _ := mk()
	r2, err := Run(ctx, backend3, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r, r2) {
		t.Error("two identical narrow runs diverged — determinism contract broken")
	}
}

// TestNarrowDefaultOffByteIdentical pins the opt-in gate: with Params.NarrowBytes
// unset, render/compact compile the exact op list they compiled before the knob
// existed (no narrow op anywhere); a bytes<=0 narrow is a pass-through; the Stats
// wire shape is unchanged when the op never ran; and validation stays fail-closed on
// a malformed narrow.
func TestNarrowDefaultOffByteIdentical(t *testing.T) {
	ctx := context.Background()
	kindsOf := func(q Query) []string {
		out := make([]string, len(q.Ops))
		for i, op := range q.Ops {
			out[i] = op.Kind
		}
		return out
	}

	// Default off: the compiled pipelines are identical to today's.
	rq := Get0(t, "render").Build(Params{Intent: "x", Budget: 100})
	if got, want := kindsOf(rq), []string{OpScan, OpFilter, OpDedup, OpRank, OpBudget, OpRender}; !reflect.DeepEqual(got, want) {
		t.Fatalf("render with NarrowBytes unset compiled %v, want %v (byte-identical default violated)", got, want)
	}
	cq := Get0(t, "compact").Build(Params{Budget: 100})
	if got, want := kindsOf(cq), []string{OpScan, OpFilter, OpRank, OpBudget, OpConsolidate, OpTombstone}; !reflect.DeepEqual(got, want) {
		t.Fatalf("compact with NarrowBytes unset compiled %v, want %v (byte-identical default violated)", got, want)
	}

	// Opt-in: the narrow op lands immediately before the budget, carrying the cap.
	rqN := Get0(t, "render").Build(Params{Intent: "x", Budget: 100, NarrowBytes: 64})
	if got, want := kindsOf(rqN), []string{OpScan, OpFilter, OpDedup, OpRank, OpNarrow, OpBudget, OpRender}; !reflect.DeepEqual(got, want) {
		t.Fatalf("render with NarrowBytes=64 compiled %v, want %v", got, want)
	}
	if rqN.Ops[4].Bytes != 64 {
		t.Errorf("narrow op bytes = %d, want 64", rqN.Ops[4].Bytes)
	}
	cqN := Get0(t, "compact").Build(Params{Budget: 100, NarrowBytes: 64})
	if got, want := kindsOf(cqN), []string{OpScan, OpFilter, OpRank, OpNarrow, OpBudget, OpConsolidate, OpTombstone}; !reflect.DeepEqual(got, want) {
		t.Fatalf("compact with NarrowBytes=64 compiled %v, want %v", got, want)
	}

	// Executor: a narrow with no cap set (bytes<=0) is a pure pass-through.
	seed := []Cell{
		{ID: "a", Descriptor: "note-a", Bytes: 100, Durability: "durable", Attrs: map[string]string{"k": "v"}},
		{ID: "b", Descriptor: "note-b", Bytes: 100, Durability: "durable"},
	}
	base := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 500}}}
	withZero := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpNarrow}, {Kind: OpBudget, Bytes: 500}}}
	r0, err := Run(ctx, fixedBackend{cells: seed}, base, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	rz, err := Run(ctx, fixedBackend{cells: seed}, withZero, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r0.Working, rz.Working) || !reflect.DeepEqual(r0.Effects, rz.Effects) || rz.Stats.Narrowed != 0 {
		t.Error("a capless narrow changed the run: it must be a pass-through")
	}

	// Wire shape: a run that never narrowed serializes NO narrowed key (omitempty),
	// so a serialized Result is byte-identical to before the op existed.
	b, err := json.Marshal(r0.Stats)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "narrowed") {
		t.Errorf("Stats of an un-narrowed run serialized a narrowed key: %s", b)
	}

	// Fail-closed validation, matching OpBudget's rule.
	if err := Validate(Query{Ops: []Op{{Kind: OpNarrow, Bytes: -1}}}); err == nil {
		t.Error("negative narrow bytes validated but should be refused")
	}
	if err := Validate(Query{Ops: []Op{{Kind: OpNarrow, Bytes: 120}}}); err != nil {
		t.Errorf("well-formed narrow refused: %v", err)
	}
}
