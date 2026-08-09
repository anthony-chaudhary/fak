package devindex

import (
	"strings"
	"testing"
)

func TestPreStateZeroIsUnverified(t *testing.T) {
	var state PreState
	if got := state.String(); got != "UNVERIFIED" {
		t.Fatalf("zero state=%q", got)
	}
}

func TestSourceSurfacePublishesGapsAndRefusals(t *testing.T) {
	root := repoRootForSurface(t)
	surface, err := ExtractVerbSurface(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(surface.Leaves) < vsVerbFloor {
		t.Fatalf("only %d rows", len(surface.Leaves))
	}
	rendered := string(surface.Markdown())
	if !strings.Contains(rendered, "unverified rows:") || !strings.Contains(rendered, "| REFUSES |") {
		t.Fatal("render omits gap/refusal contract")
	}
	refusalRows := 0
	sourceOnly := 0
	for _, leaf := range surface.Leaves {
		if len(leaf.Pre.Codes) > 0 {
			refusalRows++
		}
		if !leaf.InHelp {
			sourceOnly++
		}
	}
	if refusalRows == 0 {
		t.Fatal("REFUSES extraction found no rows")
	}
	// This is the failure-matched witness from #5934: source discovers paths help omits.
	// If help reaches parity later, update this assertion to name that landing SHA.
	if sourceOnly == 0 {
		t.Fatal("source/help drift set is empty; record the parity SHA before changing this witness")
	}
}

func TestReasonLexiconRejectsOrdinaryWords(t *testing.T) {
	got := refusalCodesInString("ALLOW nope OFF_TRUNK and LOCK_BUSY")
	joined := strings.Join(got, ",")
	if joined != "LOCK_BUSY,OFF_TRUNK" {
		t.Fatalf("codes=%q", joined)
	}
}
