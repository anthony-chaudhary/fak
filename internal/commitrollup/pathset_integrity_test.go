package commitrollup

import "testing"

// TestAssertPathsetFailsClosedOnUnsafePaths pins the security-relevant
// fail-closed contract of the path-integrity check that guards every path-scoped
// commit: an "actual" pathset containing a .git-writing path, a "../" traversal,
// an absolute path, or a ':'-bearing path (Windows drive / NTFS ADS) must make
// the assertion FAIL — never silently pass.
//
// The mechanism is subtle and worth locking down: AssertPathset discards
// normalizePaths' reason, and normalizePaths returns nil on the FIRST invalid
// path, so a single unsafe entry collapses the whole actual set to empty and the
// expected paths all read as Missing => mismatch. A refactor to
// "skip-invalid-and-continue" would let the unsafe path ride alongside the
// legitimate ones and silently pass — exactly the regression this guards.
func TestAssertPathsetFailsClosedOnUnsafePaths(t *testing.T) {
	expected := []string{"internal/gateway/a.go"}
	unsafe := []string{
		".git/hooks/pre-commit",   // writing into the repo's own git dir
		".git",                    // the git dir itself
		"../outside.go",           // parent-directory traversal
		"/etc/passwd",             // absolute path
		"C:/Windows/System32/x",   // ':' — Windows drive
		"internal/gateway/a.go:x", // ':' — NTFS alternate data stream on the real file
	}
	for _, bad := range unsafe {
		// The legitimate file is present AND matches; only the unsafe sibling is
		// added. Fail-closed means the assertion still rejects the whole set.
		got := AssertPathset(expected, []string{"internal/gateway/a.go", bad})
		if got.OK {
			t.Fatalf("AssertPathset accepted an actual set containing unsafe path %q: %+v", bad, got)
		}
		if got.Reason != ReasonPathsetMismatch {
			t.Fatalf("unsafe path %q gave reason %q, want %s", bad, got.Reason, ReasonPathsetMismatch)
		}
	}
}

// TestAssertPathsetNormalizesAndDedupsThenMatches is the positive control for
// the guard above: equivalent spellings of the SAME safe paths ("./" prefix,
// backslash separators, duplicates) normalize and dedup to the expected set and
// pass. This proves the failures above are caused specifically by the unsafe
// paths — not by over-strict matching.
func TestAssertPathsetNormalizesAndDedupsThenMatches(t *testing.T) {
	expected := []string{"internal/gateway/a.go", "internal/gateway/b.go"}
	actual := []string{
		"internal/gateway/b.go",
		"./internal/gateway/a.go", // leading "./"
		`internal\gateway\a.go`,   // backslash separators (duplicate of a.go)
	}
	got := AssertPathset(expected, actual)
	if !got.OK {
		t.Fatalf("equivalent spellings of the safe set must match after normalization+dedup: %+v", got)
	}
	if len(got.Missing) != 0 || len(got.Extra) != 0 {
		t.Fatalf("clean match must report no missing/extra, got %+v", got)
	}
}
