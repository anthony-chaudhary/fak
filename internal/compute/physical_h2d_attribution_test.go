package compute

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=100ms
func TestPhysicalH2DAttribution(t *testing.T) {
	// Bind the serialized boundary to the real capacity vocabulary, without
	// substituting its production types in this stdlib-only reducer witness.
	source, err := os.ReadFile("capacity.go")
	if os.IsNotExist(err) {
		source, err = os.ReadFile("internal/compute/capacity.go")
	}
	if err != nil {
		t.Fatal(err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), "capacity.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		spec, ok := node.(*ast.ValueSpec)
		if !ok {
			return true
		}
		typ, ok := spec.Type.(*ast.Ident)
		if !ok || typ.Name != "MemoryClass" {
			return true
		}
		for _, value := range spec.Values {
			literal, ok := value.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				t.Fatal("MemoryClass label is not a string literal")
			}
			label, err := strconv.Unquote(literal.Value)
			if err != nil {
				t.Fatal(err)
			}
			labels[label] = true
		}
		return true
	})
	classes := []string{"unknown", "weights", "kv_cache", "ddr_cache", "offload", "scratchpad", "activation", "signal_buffer"}
	if len(labels) != len(classes) {
		t.Fatalf("MemoryClass vocabulary changed: %v", labels)
	}
	var acc BackendH2DAttribution
	var total uint64
	for i, class := range classes {
		if !labels[class] {
			t.Fatalf("noncanonical MemoryClass %q", class)
		}
		next := total + uint64(i+1)
		if err := acc.record(class, total, next); err != nil {
			t.Fatal(err)
		}
		total = next
	}
	for _, class := range []string{"", "future-class"} {
		if err := acc.record(class, total, total+2); err != nil {
			t.Fatal(err)
		}
		total += 2
	}
	raw, err := json.Marshal(acc)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]uint64
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for i, class := range classes {
		want := uint64(i + 1)
		if class == "unknown" {
			want += 4
		}
		if fields[class] != want {
			t.Fatalf("%s=%d, want %d", class, fields[class], want)
		}
	}
	unchanged := acc
	if err := acc.record("activation", total, total); err != nil || acc != unchanged {
		t.Fatal("empty native interval changed accumulation")
	}
	for _, interval := range [][2]uint64{{10, 9}, {0, ^uint64(0)}} {
		if err := acc.record("weights", interval[0], interval[1]); err == nil || acc != unchanged {
			t.Fatal("reset/overflow accepted or partially applied")
		}
	}
	// Native totals include unattributed sites; direct mapped bytes are a subset.
	after, err := acc.withTotals(total+11, 17)
	if err != nil || after.UnattributedBytes != 11 || after.DirectMappedBytes != 17 {
		t.Fatalf("completion=%+v err=%v", after, err)
	}
	before, err := (BackendH2DAttribution{Weights: 2}).withTotals(5, 3)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := subtractBackendH2DAttribution(before, after)
	if err != nil || delta.validate(total+11-5, true, true) != nil || delta.DirectMappedBytes != 14 || delta.UnattributedBytes != 8 {
		t.Fatalf("nonconserving delta=%+v err=%v", delta, err)
	}
	if _, err := subtractBackendH2DAttribution(after, before); err == nil {
		t.Fatal("decreasing class counters accepted")
	}
	for _, bad := range []BackendH2DAttribution{
		{Weights: 8}, {Weights: 11}, {Weights: 10, DirectMappedBytes: 11},
		{Weights: ^uint64(0), Activation: 1},
	} {
		if err := bad.validate(10, true, true); err == nil {
			t.Fatalf("invalid accounting accepted: %+v", bad)
		}
	}
	zero := BackendH2DAttribution{}
	if zero.validate(0, true, true) != nil || zero.validate(4, false, true) != nil ||
		acc.validate(total, false, true) == nil || zero.validate(0, true, false) == nil {
		t.Fatal("unavailable, observed-zero, and incomplete evidence were conflated")
	}
	// Individually valid endpoints can still have an impossible mapped delta.
	b, _ := (BackendH2DAttribution{Weights: 10}).withTotals(10, 0)
	a, _ := (BackendH2DAttribution{Weights: 12}).withTotals(12, 10)
	d, err := subtractBackendH2DAttribution(b, a)
	if err != nil || d.validate(2, true, true) == nil {
		t.Fatal("mapped interval exceeding native H2D interval accepted")
	}
}
