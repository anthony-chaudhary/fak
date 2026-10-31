package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// The neutral codec was extracted from model/v41. Its authored ownership must
// follow that compatibility surface, without a package-name fallback or a broad
// internal/ prefix absorbing unrelated packages.
// fak-test:runtime fast est=10ms lane=default
func TestV41KVExtractionRetainsExplicitModelOwnership(t *testing.T) {
	root := FindRoot(".")
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatal(err)
	}
	// Exercise the production taxonomy parser without loading unrelated docs.
	cat := &Catalog{
		Root:     root,
		prefixes: map[string]string{},
		exact:    map[string]string{},
		declared: map[string]bool{},
	}
	cat.parseLanes(string(raw))

	wrapper := "internal/model/v41/v41_kv_cache_layout.go"
	owner := cat.ExplicitTreeLaneForPath(wrapper)
	if owner == "" {
		t.Fatalf("compatibility wrapper %q has no explicit owner", wrapper)
	}
	for _, path := range []string{
		"internal/v41kv/v41_kv_cache_layout.go",
		"internal/v41kv/v41_kv_cache_layout_test.go",
	} {
		if got := cat.ExplicitTreeLaneForPath(path); got != owner {
			t.Errorf("extracted codec %q owner = %q, want compatibility wrapper owner %q", path, got, owner)
		}
	}
	if got := cat.ExplicitTreeLaneForPath("internal/v41kv_unowned/probe.go"); got != "" {
		t.Errorf("unregistered neighboring package acquired owner %q", got)
	}
}
