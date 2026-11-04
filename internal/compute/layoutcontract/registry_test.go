package layoutcontract

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestRegistryIntegrity(t *testing.T) {
	// The closed family set: every quantized kernel family present in-tree is
	// registered, and nothing else is. A new kernel family landing without a
	// contract entry fails here on purpose.
	want := []string{"MXFP4", "NVFP4", "Q4_K", "Q8_0"}
	got := Families()
	if len(got) != len(want) {
		t.Fatalf("Families() = %v, want exactly %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Families()[%d] = %q, want %q (closed family set drifted)", i, got[i], want[i])
		}
	}

	seen := map[string]bool{}
	for _, c := range All() {
		if seen[c.Family] {
			t.Fatalf("family %s registered twice", c.Family)
		}
		seen[c.Family] = true
		if err := c.Validate(); err != nil {
			t.Errorf("entry %s violates a closed invariant: %v", c.Family, err)
		}
		resolved, err := Get(c.Family)
		if err != nil {
			t.Fatalf("Get(%q) refused a registered family: %v", c.Family, err)
		}
		if !reflect.DeepEqual(resolved, c) {
			t.Errorf("Get(%q) returned a different contract than the registry entry", c.Family)
		}
		if m := Must(c.Family); !reflect.DeepEqual(m, c) {
			t.Errorf("Must(%q) returned a different contract than the registry entry", c.Family)
		}
	}

	for _, bad := range []string{"", "Q5_K", "q4_k", "NVFP4 ", " Q4_K"} {
		if _, err := Get(bad); !errors.Is(err, ErrUnknownFamily) {
			t.Errorf("Get(%q) error = %v, want ErrUnknownFamily", bad, err)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestPinnedSourceRefsResolve(t *testing.T) {
	// Every pinned ref must still point at a file that exists in this tree; a
	// moved or deleted definition site is a stale pin to re-cut, surfaced here
	// instead of discovered by a reader.
	root := filepath.Join("..", "..", "..")
	for _, c := range All() {
		if _, err := os.Stat(filepath.Join(root, c.Source.Path)); err != nil {
			t.Errorf("%s: pinned source %s does not resolve: %v", c.Family, c.Source, err)
		}
		if c.Transform.Present() {
			if _, err := os.Stat(filepath.Join(root, c.Transform.Source.Path)); err != nil {
				t.Errorf("%s: pinned transform source %s does not resolve: %v", c.Family, c.Transform.Source, err)
			}
		}
	}
}
