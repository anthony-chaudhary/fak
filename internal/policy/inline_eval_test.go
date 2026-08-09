package policy

import (
	"reflect"
	"testing"
)

func TestInlineEvalManifestValidationAndRoundTrip(t *testing.T) {
	m := Manifest{Version: Version, InlineEval: []InlineEvalSpec{{Interp: " Perl ", Flags: []string{"-e", "-E"}}}}
	p, err := m.ToPolicy()
	if err != nil {
		t.Fatal(err)
	}
	if got := p.InlineEval; len(got) != 1 || got[0].Interp != "perl" || !reflect.DeepEqual(got[0].Flags, []string{"-e", "-E"}) {
		t.Fatalf("compiled inline_eval = %#v", got)
	}
	if got := FromPolicy(p).InlineEval; len(got) != 1 || got[0].Interp != "perl" {
		t.Fatalf("round-trip inline_eval = %#v", got)
	}

	bad := []Manifest{
		{Version: Version, InlineEval: []InlineEvalSpec{{Flags: []string{"-e"}}}},
		{Version: Version, InlineEval: []InlineEvalSpec{{Interp: "perl"}}},
		{Version: Version, InlineEval: []InlineEvalSpec{{Interp: "perl", Flags: []string{""}}}},
	}
	for i, m := range bad {
		if _, err := m.ToPolicy(); err == nil {
			t.Fatalf("bad spec %d accepted", i)
		}
	}
}
