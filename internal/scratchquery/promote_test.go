package scratchquery

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPromotableTaxonomyIsClosed(t *testing.T) {
	if len(CandidateKinds) != 5 {
		t.Fatalf("CandidateKinds = %#v, want 5", CandidateKinds)
	}
	want := map[CandidateKind]bool{
		KindUnfinishedSpine:    true,
		KindBlocked:            true,
		KindDiscoveredEdgeCase: true,
		KindDeferredCaveat:     true,
		KindNextCheckableStep:  true,
	}
	for _, k := range CandidateKinds {
		if !want[k] {
			t.Fatalf("unexpected kind %q", k)
		}
	}
	for _, rule := range markerRules {
		if !want[rule.kind] {
			t.Fatalf("marker rule names unknown kind %q", rule.kind)
		}
	}
}

func TestPromotableClassifyByMarker(t *testing.T) {
	cases := []struct {
		name   string
		header string
		want   CandidateKind
		ok     bool
	}{
		{"spine.md", "// unfinished-spine: could not ship the path", KindUnfinishedSpine, true},
		{"blk.md", "# BLOCKED: waiting on peer lane", KindBlocked, true},
		{"edge.md", "// discovered edge case: soak hazard", KindDiscoveredEdgeCase, true},
		{"caveat.md", "// deferred-caveat: not yet witnessed on hardware", KindDeferredCaveat, true},
		{"next.md", "// next step: activate chunked prefill", KindNextCheckableStep, true},
		{"edge-case-thing.md", "plain body", KindDiscoveredEdgeCase, true}, // filename channel
		{"ordinary.md", "just a note about nothing", "", false},
		{"keep.md", "// promote:keep\n// deferred-caveat: internal only", "", false},
	}
	for _, tc := range cases {
		kind, ok := Classify(Artifact{Name: tc.name, Path: tc.name, Header: tc.header})
		if ok != tc.ok || kind != tc.want {
			t.Fatalf("Classify(%s) = (%q,%v), want (%q,%v)", tc.name, kind, ok, tc.want, tc.ok)
		}
	}
}

func TestPromotableScanFindsFlagsAndSorts(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "z-last.md", "// deferred-caveat: not yet\n")
	writeScratch(t, root, "a-first.md", "// unfinished-spine: path blocked\n")
	writeScratch(t, root, "b-edge.md", "// discovered-edge-case: found while working\n")
	writeScratch(t, root, "note.md", "ordinary note, no marker\n")

	res, err := ScanPromotable(root, "session-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Candidates) != 3 {
		t.Fatalf("candidates = %#v, want 3", res.Candidates)
	}
	// Sorted by (kind, session, file): deferred-caveat < discovered-edge-case < unfinished-spine.
	if res.Candidates[0].Kind != KindDeferredCaveat || res.Candidates[0].Path != "z-last.md" {
		t.Fatalf("first = %#v", res.Candidates[0])
	}
	if res.Candidates[1].Kind != KindDiscoveredEdgeCase {
		t.Fatalf("second = %#v", res.Candidates[1])
	}
	if res.Candidates[2].Kind != KindUnfinishedSpine {
		t.Fatalf("third = %#v", res.Candidates[2])
	}
	if res.Candidates[0].Session != "session-1" {
		t.Fatalf("session not attached: %#v", res.Candidates[0])
	}
}

func TestPromotableScanIsReadOnlyOnMissing(t *testing.T) {
	res, err := ScanPromotable(filepath.Join(t.TempDir(), "absent"), "s")
	if err != nil {
		t.Fatalf("missing scratchpad errored: %v", err)
	}
	if len(res.Candidates) != 0 {
		t.Fatalf("candidates = %#v, want empty", res.Candidates)
	}
	// The directory must still not exist: the scan never creates or deletes.
	if _, statErr := os.Stat(filepath.Join(t.TempDir(), "absent")); !os.IsNotExist(statErr) {
		t.Fatalf("scan mutated the filesystem: %v", statErr)
	}
}

func TestPromotableScanSkipsBinary(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "blob.bin", "// unfinished-spine in a binary\x00tail\n")
	writeScratch(t, root, "real.md", "// blocked by lane\n")

	res, err := ScanPromotable(root, "s")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range res.Candidates {
		if c.Path == "blob.bin" {
			t.Fatalf("binary artifact surfaced as a candidate: %#v", c)
		}
	}
	if len(res.Candidates) != 1 || res.Candidates[0].Path != "real.md" {
		t.Fatalf("candidates = %#v, want only real.md", res.Candidates)
	}
	if res.Skipped != 1 {
		t.Fatalf("skipped = %d, want 1", res.Skipped)
	}
}

func TestPromotableScanDeterministic(t *testing.T) {
	root := t.TempDir()
	writeScratch(t, root, "x.md", "// todo: finish\n")
	writeScratch(t, root, "y.md", "// blocked by lane\n")
	first, err := ScanPromotable(root, "s")
	if err != nil {
		t.Fatal(err)
	}
	second, err := ScanPromotable(root, "s")
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Candidates) != len(second.Candidates) {
		t.Fatalf("nondeterministic length")
	}
	for i := range first.Candidates {
		if first.Candidates[i] != second.Candidates[i] {
			t.Fatalf("nondeterministic at %d: %#v vs %#v", i, first.Candidates[i], second.Candidates[i])
		}
	}
}
