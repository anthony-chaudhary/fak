package market

import (
	"os"
	"testing"
)

// Invariant: Marketplace extensions catalogs must enforce cryptographic signature and compatibility boundaries.
// Guard: Parse refuses descriptors with missing artifact digests or incompatible major versions.

func TestMarketLifecycle(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("testdata/catalog.golden.json")
	if err != nil {
		t.Fatalf("failed reading golden catalog: %v", err)
	}

	compat := map[string]int{"fak-compute": 1}
	catalog, err := Parse(data, compat)
	if err != nil {
		t.Fatalf("failed parsing golden catalog: %v", err)
	}
	if len(catalog.Extensions) == 0 {
		t.Fatal("expected non-empty extensions in parsed catalog")
	}
}
