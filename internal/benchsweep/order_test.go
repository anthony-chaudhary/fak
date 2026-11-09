package benchsweep

import (
	"reflect"
	"testing"
)

// p builds one swept-axis pin.
func p(name, value string) Param { return Param{Name: name, Value: value} }

// c builds one combo with the given server and workload axes.
func c(id string, server, workload []Param) Combo {
	return Combo{ID: id, ServerParams: server, WorkloadParams: workload}
}

// ids projects a combo slice to its ID sequence for order assertions.
func ids(cs []Combo) []string {
	out := make([]string, len(cs))
	for i, x := range cs {
		out[i] = x.ID
	}
	return out
}

// excludedIDs projects an exclusion record to its combo ID sequence.
func excludedIDs(es []ExcludedCombo) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.Combo.ID
	}
	return out
}

// assertContiguous checks the closed group contract: every combo in a group
// carries the group's key, and adjacent groups always differ in key.
func assertContiguous(t *testing.T, plan RestartPlan) {
	t.Helper()
	for gi, g := range plan.Groups {
		if len(g.Combos) == 0 {
			t.Fatalf("group %d is empty", gi)
		}
		for _, x := range g.Combos {
			if x.ConfigKey() != g.Key {
				t.Fatalf("group %d key %q contains combo %s with key %q", gi, g.Key, x.ID, x.ConfigKey())
			}
		}
		if gi > 0 && plan.Groups[gi-1].Key == g.Key {
			t.Fatalf("adjacent groups %d and %d share key %q — not contiguous-minimal", gi-1, gi, g.Key)
		}
	}
}

func TestSortCombos(t *testing.T) {
	t.Run("equal_keys_keep_input_order", func(t *testing.T) {
		in := []Combo{
			c("w2", []Param{p("tp_size", "8")}, []Param{p("prefixes", "16")}),
			c("a1", []Param{p("tp_size", "8")}, []Param{p("prefixes", "4")}),
			c("w1", []Param{p("tp_size", "8")}, []Param{p("prefixes", "8")}),
		}
		got := SortCombos(in)
		want := []string{"w2", "a1", "w1"}
		if !reflect.DeepEqual(ids(got), want) {
			t.Fatalf("stable sort reordered equal-key combos: got %v, want %v", ids(got), want)
		}
	})

	t.Run("interleaved_keys_run_contiguously_in_key_order", func(t *testing.T) {
		in := []Combo{
			c("b1", []Param{p("tp_size", "4")}, nil),
			c("a1", []Param{p("tp_size", "8")}, nil),
			c("b2", []Param{p("tp_size", "4")}, nil),
			c("a2", []Param{p("tp_size", "8")}, nil),
			c("b3", []Param{p("tp_size", "4")}, nil),
		}
		// Key "...4;" sorts before "...8;", so the tp_size=4 run leads.
		got := SortCombos(in)
		want := []string{"b1", "b2", "b3", "a1", "a2"}
		if !reflect.DeepEqual(ids(got), want) {
			t.Fatalf("keys did not group contiguously and stably: got %v, want %v", ids(got), want)
		}
		runs := 0
		for i, x := range got {
			if i == 0 || x.ConfigKey() != got[i-1].ConfigKey() {
				runs++
			}
		}
		if runs != 2 {
			t.Fatalf("expected 2 contiguous key runs, got %d", runs)
		}
	})

	t.Run("key_is_independent_of_param_declaration_order", func(t *testing.T) {
		x := c("x", []Param{p("tp_size", "8"), p("mem_fraction", "0.85")}, nil)
		y := c("y", []Param{p("mem_fraction", "0.85"), p("tp_size", "8")}, nil)
		if x.ConfigKey() != y.ConfigKey() {
			t.Fatalf("declaration order moved the key: %q vs %q", x.ConfigKey(), y.ConfigKey())
		}
		got := SortCombos([]Combo{x, y})
		if !reflect.DeepEqual(ids(got), []string{"x", "y"}) {
			t.Fatalf("equal keys must keep input order: got %v", ids(got))
		}
	})

	t.Run("input_slice_is_not_mutated", func(t *testing.T) {
		in := []Combo{
			c("b1", []Param{p("tp_size", "4")}, nil),
			c("a1", []Param{p("tp_size", "8")}, nil),
		}
		orig := append([]Combo(nil), in...)
		SortCombos(in)
		if !reflect.DeepEqual(in, orig) {
			t.Fatalf("SortCombos mutated its input: got %v, want %v", ids(in), ids(orig))
		}
	})

	t.Run("empty_input_yields_empty_output", func(t *testing.T) {
		if got := SortCombos(nil); len(got) != 0 {
			t.Fatalf("empty input must yield empty output, got %d combos", len(got))
		}
	})
}

func TestRestartPlan(t *testing.T) {
	keyA := EncodeConfigKey([]Param{p("tp_size", "8")})
	keyB := EncodeConfigKey([]Param{p("tp_size", "4")})

	t.Run("contiguous_groups_one_restart_per_distinct_healthy_key", func(t *testing.T) {
		in := []Combo{
			c("b1", []Param{p("tp_size", "4")}, nil),
			c("a1", []Param{p("tp_size", "8")}, nil),
			c("b2", []Param{p("tp_size", "4")}, nil),
			c("a2", []Param{p("tp_size", "8")}, nil),
		}
		plan, err := PlanRestarts(in, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Restarts != 2 {
			t.Fatalf("restart count = %d, want 2 distinct healthy keys", plan.Restarts)
		}
		if len(plan.Groups) != 2 {
			t.Fatalf("group count = %d, want 2", len(plan.Groups))
		}
		assertContiguous(t, plan)
		// Key "...4;" sorts before "...8;", so the tp_size=4 group leads.
		if plan.Groups[0].Key != keyB || !reflect.DeepEqual(ids(plan.Groups[0].Combos), []string{"b1", "b2"}) {
			t.Fatalf("first group wrong: key %q combos %v", plan.Groups[0].Key, ids(plan.Groups[0].Combos))
		}
		if plan.Groups[1].Key != keyA || !reflect.DeepEqual(ids(plan.Groups[1].Combos), []string{"a1", "a2"}) {
			t.Fatalf("second group wrong: key %q combos %v", plan.Groups[1].Key, ids(plan.Groups[1].Combos))
		}
		if len(plan.Excluded) != 0 {
			t.Fatalf("no failed keys, but %d combos excluded", len(plan.Excluded))
		}
	})

	t.Run("failed_key_combos_are_excluded_typed_never_dropped", func(t *testing.T) {
		in := []Combo{
			c("b1", []Param{p("tp_size", "4")}, nil),
			c("a1", []Param{p("tp_size", "8")}, nil),
			c("b2", []Param{p("tp_size", "4")}, nil),
			c("a2", []Param{p("tp_size", "8")}, nil),
			c("b3", []Param{p("tp_size", "4")}, nil),
		}
		plan, err := PlanRestarts(in, []ConfigKey{keyB})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Restarts != 1 {
			t.Fatalf("restart count = %d, want 1 distinct healthy key", plan.Restarts)
		}
		if len(plan.Groups) != 1 || plan.Groups[0].Key != keyA {
			t.Fatalf("healthy groups wrong: %+v", plan.Groups)
		}
		if !reflect.DeepEqual(ids(plan.Groups[0].Combos), []string{"a1", "a2"}) {
			t.Fatalf("healthy group combos wrong: %v", ids(plan.Groups[0].Combos))
		}
		if !reflect.DeepEqual(excludedIDs(plan.Excluded), []string{"b1", "b2", "b3"}) {
			t.Fatalf("excluded record wrong: %v", excludedIDs(plan.Excluded))
		}
		for _, e := range plan.Excluded {
			if e.Key != keyB {
				t.Fatalf("excluded combo %s carries key %q, want failed key %q", e.Combo.ID, e.Key, keyB)
			}
		}
		// Accounting invariant: nothing dropped silently.
		accounted := len(plan.Excluded)
		for _, g := range plan.Groups {
			accounted += len(g.Combos)
		}
		if accounted != len(in) {
			t.Fatalf("accounting broken: %d accounted of %d input combos", accounted, len(in))
		}
	})

	t.Run("every_combo_failed_yields_zero_restart_plan", func(t *testing.T) {
		in := []Combo{
			c("a1", []Param{p("tp_size", "8")}, nil),
			c("b1", []Param{p("tp_size", "4")}, nil),
		}
		plan, err := PlanRestarts(in, []ConfigKey{keyA, keyB})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Restarts != 0 || len(plan.Groups) != 0 {
			t.Fatalf("all keys failed, want zero restarts/groups, got %d/%d", plan.Restarts, len(plan.Groups))
		}
		if !reflect.DeepEqual(excludedIDs(plan.Excluded), []string{"b1", "a1"}) {
			t.Fatalf("excluded record wrong: %v", excludedIDs(plan.Excluded))
		}
	})

	t.Run("each_distinct_key_costs_one_restart", func(t *testing.T) {
		in := []Combo{
			c("c1", []Param{p("tp_size", "2")}, nil),
			c("c2", []Param{p("tp_size", "4")}, nil),
			c("c3", []Param{p("tp_size", "8")}, nil),
		}
		plan, err := PlanRestarts(in, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Restarts != 3 {
			t.Fatalf("restart count = %d, want 3 distinct healthy keys", plan.Restarts)
		}
		assertContiguous(t, plan)
	})

	t.Run("empty_input_is_a_valid_empty_plan", func(t *testing.T) {
		for _, in := range [][]Combo{nil, {}} {
			plan, err := PlanRestarts(in, nil)
			if err != nil {
				t.Fatalf("empty input must not be an error: %v", err)
			}
			if plan.Restarts != 0 || len(plan.Groups) != 0 || len(plan.Excluded) != 0 {
				t.Fatalf("empty input must yield a valid empty plan, got %+v", plan)
			}
		}
	})
}
