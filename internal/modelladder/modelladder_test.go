package modelladder

import (
	"path/filepath"
	"testing"
)

// TestLadderSpecsShape pins the deterministic structure of the candidate table:
// four rungs, returned in ascending Order (0..3), with the expected names, the
// dir/hf kind split, and absolute resolved dirs. Present is disk-dependent and
// deliberately not asserted here.
func TestLadderSpecsShape(t *testing.T) {
	specs := LadderSpecs()
	if len(specs) != 4 {
		t.Fatalf("want 4 rungs, got %d", len(specs))
	}
	for i, s := range specs {
		if s.Order != i {
			t.Errorf("rung %d has Order %d, want %d (specs must come back Order-sorted)", i, s.Order, i)
		}
		if !filepath.IsAbs(s.Dir) {
			t.Errorf("rung %d Dir %q is not absolute", i, s.Dir)
		}
	}
	wantName := []string{"SmolLM2-135M", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"}
	for i, w := range wantName {
		if specs[i].Name != w {
			t.Errorf("rung %d Name = %q, want %q", i, specs[i].Name, w)
		}
	}
	// The smallest rung is a fak export dir; the rest are HF safetensors snapshots.
	if specs[0].Kind != "dir" {
		t.Errorf("rung 0 Kind = %q, want %q", specs[0].Kind, "dir")
	}
	for i := 1; i < len(specs); i++ {
		if specs[i].Kind != "hf" {
			t.Errorf("rung %d Kind = %q, want %q", i, specs[i].Kind, "hf")
		}
	}
}

// TestSmallestPresent covers the pure first-present-in-ladder-order selector,
// including the no-rung-present case (must return the zero Spec + false).
func TestSmallestPresent(t *testing.T) {
	tests := []struct {
		name     string
		specs    []Spec
		wantOK   bool
		wantName string
	}{
		{"nil slice", nil, false, ""},
		{"none present", []Spec{{Name: "a"}, {Name: "b"}}, false, ""},
		{"first present wins", []Spec{{Name: "a", Present: true}, {Name: "b", Present: true}}, true, "a"},
		{"skips absent, takes next present", []Spec{{Name: "a"}, {Name: "b", Present: true}}, true, "b"},
		{"last present", []Spec{{Name: "a"}, {Name: "b"}, {Name: "c", Present: true}}, true, "c"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := SmallestPresent(tt.specs)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if ok {
				if got.Name != tt.wantName {
					t.Errorf("Name = %q, want %q", got.Name, tt.wantName)
				}
			} else if got != (Spec{}) {
				t.Errorf("not-present case must return zero Spec, got %+v", got)
			}
		})
	}
}

// TestNewRegistryEmpty confirms the constructor returns a usable, empty,
// initialized cache (a nil map would panic on the first memoizing write).
func TestNewRegistryEmpty(t *testing.T) {
	r := NewRegistry()
	if r == nil {
		t.Fatal("NewRegistry returned nil")
	}
	if r.cache == nil {
		t.Fatal("registry cache map must be initialized, not nil")
	}
	if len(r.cache) != 0 {
		t.Fatalf("new registry cache should be empty, got %d entries", len(r.cache))
	}
}
