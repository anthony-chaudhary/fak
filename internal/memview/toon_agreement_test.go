package memview_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memview"
	"github.com/anthony-chaudhary/fak/internal/toon"
)

// TestTOONAgreesWithGeneralCodec pins the seam described in both packages' docs: on the
// flat-Surface case memview.EncodeTOON was built for, the standalone encoder and the
// general internal/toon codec produce BYTE-FOR-BYTE identical output. memview stays the
// documented flat-subset special case; toon is the lossless generalization. If either
// side drifts (a quoting rule, the two-space indent, the header grammar, the comma
// delimiter), this fails — the guarantee is enforced, not just asserted in prose.
//
// This test lives in memview (tier 2) and imports toon (tier 1): a downward edge, and it
// is a _test.go file, which architest's importer does not even scan.
func TestTOONAgreesWithGeneralCodec(t *testing.T) {
	// A memview Surface: a lowercase-ident title, a fixed field set, plain string cells —
	// the exact shape every Surface in the codebase produces.
	surface, err := memview.NewSurface("notes", []string{"id", "verdict"}, []memview.Row{
		{"n1", "fresh"},
		{"n2", "withheld"},
	})
	if err != nil {
		t.Fatal(err)
	}
	fromMemview := memview.EncodeTOON(surface)

	// The equivalent JSON value for the general codec: an object with one uniform-array
	// field named for the Surface title, each row an object of field->cell. toon sorts
	// keys, so the fields are given already sorted to match the Surface's column order.
	value := map[string]any{
		"notes": []any{
			map[string]any{"id": "n1", "verdict": "fresh"},
			map[string]any{"id": "n2", "verdict": "withheld"},
		},
	}
	fromToon, err := toon.Encode(value, toon.Options{})
	if err != nil {
		t.Fatal(err)
	}

	if string(fromMemview) != string(fromToon) {
		t.Fatalf("memview.EncodeTOON and toon.Encode disagree on the flat case:\n"+
			"memview:\n%s\ntoon:\n%s", fromMemview, fromToon)
	}
}
