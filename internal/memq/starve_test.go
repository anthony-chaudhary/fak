package memq

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// starveFixedBackend returns exactly its seeded cells, in order, so the cutline is
// predictable — self-contained so this file compiles regardless of sibling fixtures.
type starveFixedBackend struct{ cells []Cell }

func (b starveFixedBackend) Cells(context.Context) ([]Cell, error) { return b.cells, nil }
func (b starveFixedBackend) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, c := range b.cells {
		if c.ID == id {
			return []byte(c.Descriptor), nil
		}
	}
	return nil, ErrSealed
}

// TestStarveCreditRetainsPerenniallyBorderlineCell is the #4021 witness: a non-durable,
// refcount==0 cell perennially at rank budget+1 advances its below-cutline counter each
// pass, is retained on pass K with the ranking otherwise unchanged (a bounded ONE-pass
// credit, counter reset), and the pass is byte-identical across repeats — no RNG.
func TestStarveCreditRetainsPerenniallyBorderlineCell(t *testing.T) {
	ctx := context.Background()
	mem := NewMemStore()
	// Three 100-byte session-class cells; a 200-byte budget puts the third perennially
	// at rank budget+1 (scan order is the working order — fully deterministic).
	for _, s := range []string{"a", "b", "c"} {
		body := []byte(s + strings.Repeat("x", 99))
		if len(body) != 100 {
			t.Fatalf("fixture body = %d bytes, want 100", len(body))
		}
		mem.Add("note", "episode", DurabilitySession, body, false)
	}
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 200, StarveK: 3}}}

	// Passes 1..2: the borderline cell (cell:2) is still dropped; its counter advances.
	for pass := 1; pass <= 2; pass++ {
		res, err := Run(ctx, mem, q, Caps{})
		if err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
		if len(res.Working) != 2 {
			t.Fatalf("pass %d: working = %d cells, want 2 (no credit before pass K)", pass, len(res.Working))
		}
		if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "cell:2" {
			t.Fatalf("pass %d: overflow = %+v, want cell:2 dropped", pass, res.Overflow)
		}
		if res.Starve == nil || res.Starve.Granted != "" {
			t.Fatalf("pass %d: starve = %+v, want counter updates and no grant", pass, res.Starve)
		}
		want := StarveUpdate{ID: "cell:2", Passes: pass}
		if len(res.Starve.Updates) != 1 || res.Starve.Updates[0] != want {
			t.Fatalf("pass %d: updates = %+v, want [%+v]", pass, res.Starve.Updates, want)
		}
		if n := mem.ApplyStarveUpdates(res.Starve.Updates); n != 1 {
			t.Fatalf("pass %d: applied %d update(s), want 1", pass, n)
		}
	}

	// Pass K=3: the credit fires — cell:2 is retained after the kept prefix (its natural
	// rank position), its counter resets, and the overflow verdict reflects the truth.
	res3, err := Run(ctx, mem, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, 0, len(res3.Working))
	for _, c := range res3.Working {
		got = append(got, c.ID)
	}
	if len(got) != 3 || got[0] != "cell:0" || got[1] != "cell:1" || got[2] != "cell:2" {
		t.Fatalf("pass 3 working = %v, want [cell:0 cell:1 cell:2] (ranking otherwise unchanged)", got)
	}
	if res3.Working[2].Attrs[StarveAttr] != "0" {
		t.Fatalf("granted cell counter attr = %q, want %q (one-pass credit resets)", res3.Working[2].Attrs[StarveAttr], "0")
	}
	if res3.Overflow != nil {
		t.Fatalf("pass 3 overflow = %+v, want nil (the granted cell is not dropped)", res3.Overflow)
	}
	if res3.Starve == nil || res3.Starve.Reason != StarveReason || res3.Starve.Granted != "cell:2" {
		t.Fatalf("pass 3 starve = %+v, want a %s grant for cell:2", res3.Starve, StarveReason)
	}
	var note string
	for _, s := range res3.Steps {
		if s.Kind == OpBudget {
			note = s.Note
		}
	}
	if !strings.Contains(note, StarveReason) || !strings.Contains(note, "cell:2") {
		t.Fatalf("pass 3 budget note = %q, want it to carry %s and cell:2", note, StarveReason)
	}

	// Deterministic: repeating the exact same pass yields a byte-identical Result.
	rep, err := Run(ctx, mem, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	j1, err := json.Marshal(res3)
	if err != nil {
		t.Fatal(err)
	}
	j2, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(j1, j2) {
		t.Fatalf("pass 3 not reproducible across repeats:\n%s\nvs\n%s", j1, j2)
	}

	// The credit is ONE pass: after persisting the reset, the next pass drops the cell
	// again and its streak restarts at 1.
	mem.ApplyStarveUpdates(res3.Starve.Updates)
	res4, err := Run(ctx, mem, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res4.Working) != 2 {
		t.Fatalf("pass 4 working = %d cells, want 2 (credit is bounded to one pass)", len(res4.Working))
	}
	if res4.Starve == nil || len(res4.Starve.Updates) != 1 ||
		res4.Starve.Updates[0].Passes != 1 || res4.Starve.Updates[0].Granted {
		t.Fatalf("pass 4 starve = %+v, want streak restarted at 1 with no grant", res4.Starve)
	}
}

// TestStarveDefaultOffByteIdentical pins the default-off fence: with StarveK unset the
// cutline, the report surface, and the wire encodings are all unchanged — even when a
// cell already carries a (stale) counter attr, it never manufactures a credit.
func TestStarveDefaultOffByteIdentical(t *testing.T) {
	ctx := context.Background()
	seed := []Cell{
		{ID: "m1", Descriptor: "note-1", Bytes: 100, Durability: "session"},
		{ID: "m2", Descriptor: "note-2", Bytes: 100, Durability: "session"},
		{ID: "m3", Descriptor: "note-3", Bytes: 100, Durability: "session",
			Attrs: map[string]string{StarveAttr: "99"}},
	}
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 200}}}

	res, err := Run(ctx, starveFixedBackend{cells: seed}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	// Unchanged cutline: today's strict trim, no credit, no starve report.
	if len(res.Working) != 2 {
		t.Fatalf("default-off working = %d cells, want 2", len(res.Working))
	}
	if res.Starve != nil {
		t.Fatalf("default-off emitted a starve report: %+v", res.Starve)
	}
	if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "m3" {
		t.Fatalf("default-off overflow = %+v, want m3 dropped", res.Overflow)
	}
	// The backend's attrs are untouched (Run stays a read; no counter churn when off).
	if seed[2].Attrs[StarveAttr] != "99" {
		t.Fatalf("default-off mutated the backend attr: %q", seed[2].Attrs[StarveAttr])
	}
	// The wire shape is unchanged: an unset StarveK never serializes, so an authored
	// op and the whole Result round-trip byte-identically to the pre-#4021 encoding.
	oj, err := json.Marshal(Op{Kind: OpBudget, Bytes: 200})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(oj), "starve") {
		t.Fatalf("unset starve_k leaked into the Op encoding: %s", oj)
	}
	rj, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(rj), "starve") {
		t.Fatalf("default-off Result encoding carries a starve key: %s", rj)
	}
	// Driver seam: an unset Params.StarveK builds the exact same render query as today.
	d, ok := Get("render")
	if !ok {
		t.Fatal("render driver not registered")
	}
	qj, err := json.Marshal(d.Build(Params{Intent: "x", Budget: 200}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(qj), "starve") {
		t.Fatalf("render driver with unset StarveK emits a starve field: %s", qj)
	}
}
