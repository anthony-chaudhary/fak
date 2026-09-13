package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRouteAliasMutateRoundTrip covers the file-management surface: a --aliases-set
// persists a new alias (the file round-trips), a self-cycle is refused with exit 1 and
// leaves the file byte-unchanged, and removing a missing alias is exit 1.
func TestRouteAliasMutateRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "aliases.json")
	seed := `{
  "version": "fak-alias-registry/1",
  "aliases": [
    {"name": "prod", "target": "small"}
  ]
}
`
	if err := os.WriteFile(path, []byte(seed), 0o600); err != nil {
		t.Fatalf("seed aliases: %v", err)
	}

	// (a) set a new alias; the file round-trips.
	code, out, errs := runRT("--aliases-set", "fast=large", "--aliases-file", path)
	if code != 0 {
		t.Fatalf("--aliases-set exit=%d stderr=%s", code, errs)
	}
	if !strings.Contains(out, `"fast"`) || !strings.Contains(out, `"large"`) {
		t.Fatalf("set output missing new alias:\n%s", out)
	}
	code, out, errs = runRT("--aliases-list", path)
	if code != 0 {
		t.Fatalf("--aliases-list exit=%d stderr=%s", code, errs)
	}
	if !strings.Contains(out, `"fast"`) || !strings.Contains(out, `"large"`) {
		t.Fatalf("round-trip lost the new alias:\n%s", out)
	}

	// (b) a self-cycle is refused and the file is untouched.
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read before: %v", err)
	}
	code, _, errs = runRT("--aliases-set", "a=a", "--aliases-file", path)
	if code != 1 {
		t.Fatalf("self-cycle exit=%d, want 1 (stderr=%s)", code, errs)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read after: %v", err)
	}
	if string(before) != string(after) {
		t.Fatalf("a rejected edit changed the file:\nbefore=%s\nafter=%s", before, after)
	}

	// (c) removing a nonexistent alias is exit 1.
	code, _, errs = runRT("--aliases-remove", "nope", "--aliases-file", path)
	if code != 1 {
		t.Fatalf("remove-nonexistent exit=%d, want 1 (stderr=%s)", code, errs)
	}
}