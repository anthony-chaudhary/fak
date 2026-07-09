package selfquery

import "testing"

func TestScratchFalsePresent(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Dump leaves so we see what trees exist.
	if cat.dev != nil {
		for _, l := range cat.dev.Leaves {
			t.Logf("LEAF name=%q tree=%q", l.Name, l.Tree)
		}
	}
	// Dump card DetailRefs.
	for _, fc := range cat.Cards(PlaneAll) {
		t.Logf("CARD name=%q detailref=%q", fc.Name, fc.DetailRef)
	}

	for _, p := range []string{
		"github.com/anthony-chaudhary/fak/internal",
		"github.com/anthony-chaudhary/fak/docs",
		"github.com/anthony-chaudhary/fak/cmd",
		"github.com/anthony-chaudhary/fak/internal/ctxpl",
		"github.com/anthony-chaudhary/fak/internal/gatewayX",
	} {
		v, d := cat.resolveImport(p)
		t.Logf("RESOLVE %-55s => %-10s %s", p, v, d)
	}
}
