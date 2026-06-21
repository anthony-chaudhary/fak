package toollint

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/preflight"
)

// The supported-type set TL006 checks against MUST equal the JSON-Schema subset the
// pre-flight ladder actually honors (preflight.typeOK). They live in two packages —
// this test is the pin that makes them fail together if either drifts. If preflight
// adds a FieldType, this test fails until supportedSchemaTypes learns it; if
// toollint invents a type preflight does not honor, it fails too.
func TestSupportedSchemaTypesMatchesPreflight(t *testing.T) {
	want := map[string]bool{
		string(preflight.TypeString): true,
		string(preflight.TypeNumber): true,
		string(preflight.TypeBool):   true,
		string(preflight.TypeObject): true,
		string(preflight.TypeArray):  true,
		string(preflight.TypeAny):    true, // "" — intentional any
	}
	if len(want) != len(supportedSchemaTypes) {
		t.Fatalf("type-set size drift: toollint has %d, preflight subset has %d", len(supportedSchemaTypes), len(want))
	}
	for ty := range want {
		if !supportedSchemaTypes[ty] {
			t.Fatalf("preflight honors type %q but toollint does not list it as supported", ty)
		}
	}
	for ty := range supportedSchemaTypes {
		if !want[ty] {
			t.Fatalf("toollint lists %q as supported but preflight does not honor it", ty)
		}
	}
}
