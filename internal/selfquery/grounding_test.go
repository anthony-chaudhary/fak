package selfquery

import "testing"

// TestScanPatchGroundsImportsAgainstIndex is the #3363 witness (its "first checkable
// step"): a candidate patch that adds an import of a package the index does NOT
// contain must come back UNGROUNDED, while an added import that resolves to a real
// declared lane comes back PRESENT. It proves both the gap (ScanPatch decides
// reference existence pre-apply, from the index, not from any agent's narration) and
// that the index carries enough to decide.
func TestScanPatchGroundsImportsAgainstIndex(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// A unified diff adding a new Go file that imports one REAL package (internal/gateway
	// — a lane the fixture's dos.toml declares) alongside one FABRICATED package
	// (internal/doesnotexist — no lane, no card).
	diff := "diff --git a/internal/example/use.go b/internal/example/use.go\n" +
		"--- /dev/null\n" +
		"+++ b/internal/example/use.go\n" +
		"@@ -0,0 +1,6 @@\n" +
		"+package example\n" +
		"+\n" +
		"+import (\n" +
		"+\t\"github.com/anthony-chaudhary/fak/internal/gateway\"\n" +
		"+\t\"github.com/anthony-chaudhary/fak/internal/doesnotexist\"\n" +
		"+)\n"

	rep := cat.ScanPatch(diff)

	byRef := map[string]GroundingRef{}
	for _, r := range rep.References {
		byRef[r.Ref] = r
	}

	real, ok := byRef["github.com/anthony-chaudhary/fak/internal/gateway"]
	if !ok {
		t.Fatalf("the real import was not extracted from the diff; references = %+v", rep.References)
	}
	if real.Verdict != GroundingPresent {
		t.Errorf("a real declared-lane import must resolve PRESENT, got %q (%s)", real.Verdict, real.Detail)
	}

	fake, ok := byRef["github.com/anthony-chaudhary/fak/internal/doesnotexist"]
	if !ok {
		t.Fatalf("the fabricated import was not extracted from the diff; references = %+v", rep.References)
	}
	if fake.Verdict != GroundingUngrounded {
		t.Errorf("a fabricated import must be UNGROUNDED, got %q (%s)", fake.Verdict, fake.Detail)
	}

	if rep.Ungrounded != 1 {
		t.Errorf("want exactly 1 ungrounded reference, got %d", rep.Ungrounded)
	}
	if rep.Grounded {
		t.Error("a patch carrying an ungrounded reference must not read as grounded")
	}
}
