package modelroute

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAccidentalCorpusPairsHaveDeterministicGroundTruth(t *testing.T) {
	fixtures := AccidentalCorpus()
	if err := SelfCheckAccidentalCorpus(fixtures); err != nil {
		t.Fatalf("accidental corpus selfcheck: %v", err)
	}
	if len(fixtures) != 24 {
		t.Fatalf("fixtures = %d, want 24 (12 clean/corrupt pairs)", len(fixtures))
	}
	manifest, err := BuildAccidentalCorpusManifest(fixtures)
	if err != nil {
		t.Fatalf("build manifest: %v", err)
	}
	if manifest.Schema != AccidentalCorpusManifestSchema || manifest.Pairs != 12 || manifest.Corrupt != 12 || manifest.Clean != 12 {
		t.Fatalf("manifest summary = %+v", manifest)
	}
	if len(manifest.ClassSizes) != 12 || len(manifest.Rows) != len(fixtures) {
		t.Fatalf("manifest classes=%d rows=%d", len(manifest.ClassSizes), len(manifest.Rows))
	}
	for _, row := range manifest.Rows {
		if row.BundleDigest == "" || row.Witness.Command == "" || row.Author.Family == row.Auditor.Family {
			t.Errorf("incomplete manifest row %+v", row)
		}
	}
	if _, err := json.Marshal(manifest); err != nil {
		t.Fatalf("manifest is not machine-readable: %v", err)
	}
}

func TestAccidentalCorpusLabelFlipBreaksSelfCheck(t *testing.T) {
	fixtures := AccidentalCorpus()
	fixtures[0].Corrupt = !fixtures[0].Corrupt
	if err := SelfCheckAccidentalCorpus(fixtures); err == nil || !strings.Contains(err.Error(), "label/witness contract diverged") {
		t.Fatalf("flipped label selfcheck = %v, want label/witness failure", err)
	}
}

func TestAccidentalCorpusCoversRequestedFailureClasses(t *testing.T) {
	want := []AccidentalFailureClass{
		AccidentalIncompleteDoneCondition, AccidentalWrongEdgeCase, AccidentalSwallowedError,
		AccidentalStaleConsumer, AccidentalMissingFailureTest, AccidentalRaceLostUpdate,
		AccidentalPartialRename, AccidentalBuildPoison, AccidentalDocsCLIDrift,
		AccidentalOverBroadRewrite, AccidentalRevertedSafetyCheck, AccidentalCleanHardRefactor,
	}
	seen := map[AccidentalFailureClass]map[bool]bool{}
	for _, fixture := range AccidentalCorpus() {
		if seen[fixture.Class] == nil {
			seen[fixture.Class] = map[bool]bool{}
		}
		seen[fixture.Class][fixture.Corrupt] = true
	}
	for _, class := range want {
		if !seen[class][false] || !seen[class][true] {
			t.Errorf("class %s missing clean/corrupt pair", class)
		}
	}
}
