package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInkling2ResolveSpecDraftAutoTakesAdjacentDFlash2(t *testing.T) {
	withWeights := t.TempDir()
	if err := os.WriteFile(filepath.Join(withWeights, inkling2DFlash2Weights), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	dspark := t.TempDir()
	if err := os.WriteFile(filepath.Join(dspark, inkling2DSparkWeightsFile), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ kind, mtp, dflash, want string }{
		{inkling2DraftAuto, "", withWeights, inkling2DraftDFlash},
		{inkling2DraftAuto, "", t.TempDir(), inkling2DraftLookup},
		{inkling2DraftAuto, dspark, withWeights, inkling2DraftDSpark},
		{inkling2DraftLookup, "", withWeights, inkling2DraftLookup},
	} {
		if got := inkling2ResolveSpecDraft(c.kind, c.mtp, c.dflash); got != c.want {
			t.Errorf("resolve(%q, %q, %q) = %q, want %q", c.kind, c.mtp, c.dflash, got, c.want)
		}
	}
}
