package opttarget

import "testing"

// TestDiscoverIgnoresOwnDocExample is the regression witness for the self-
// annotation false positive: DiscoverDir over the opttarget package's OWN
// directory must NOT harvest annotationTag's doc-comment example (a string const
// carrying `// fak:opttarget ... dir=<higher|lower> ...`) as a malformed target.
// Before the int-literal guard, `fak opt discover` reported a spurious
// "malformed annotation" on internal/opttarget/discover.go on every run. The
// package declares no annotated INT tunable of its own, so the result is empty
// and error-free.
func TestDiscoverIgnoresOwnDocExample(t *testing.T) {
	got, err := DiscoverDir(".")
	if err != nil {
		t.Fatalf("DiscoverDir(.) over opttarget's own package errored on its doc example: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("DiscoverDir(.) harvested %d targets from the opttarget package, want 0: %+v", len(got), got)
	}
}
