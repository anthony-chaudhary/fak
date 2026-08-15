package rolloutmode

import "testing"

// TestVocabularyIsClosed: exactly four rungs, in ladder order, and nothing else
// is a rung — including the three retired private spellings of the applied rung
// (dispatchtick's `default`, memorycotravel's `live`) and the zero value. If a
// leaf ever reintroduces a fourth spelling this is where it fails.
func TestVocabularyIsClosed(t *testing.T) {
	want := []Mode{"off", "shadow", "canary", "on"}
	got := Modes()
	if len(got) != len(want) {
		t.Fatalf("Modes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Modes()[%d] = %q, want %q (ladder order is off -> shadow -> canary -> on)", i, got[i], want[i])
		}
		if !want[i].Valid() {
			t.Fatalf("%q must be Valid", want[i])
		}
	}
	for _, bad := range []Mode{"", "default", "live", "ON", "Off", " off", "off ", "canary-lite", "enabled", "true"} {
		if bad.Valid() {
			t.Errorf("%q must NOT be a rung: the vocabulary is closed", bad)
		}
	}
}

// TestModesReturnsACopy: the shared vocabulary cannot be mutated by a caller that
// writes into the slice it got back.
func TestModesReturnsACopy(t *testing.T) {
	m := Modes()
	m[0] = "tampered"
	if Modes()[0] != Off {
		t.Fatalf("Modes() leaked the shared ladder: first rung is now %q", Modes()[0])
	}
	if !Off.Valid() {
		t.Fatalf("Off stopped being a rung after a caller mutated a Modes() result")
	}
}

// TestParseUnsetResolvesToTheCallersFallback: an empty string is "unset", not a
// bad value — it resolves to the caller's posture with ok=true so a leaf does not
// warn about a flag nobody set.
func TestParseUnsetResolvesToTheCallersFallback(t *testing.T) {
	for _, fb := range []Mode{Off, Shadow, On} {
		m, ok := Parse("", fb)
		if m != fb || !ok {
			t.Errorf("Parse(\"\", %q) = (%q,%v), want (%q,true)", fb, m, ok, fb)
		}
	}
}

// TestParseRecognizesEveryRung: each rung parses to itself under the full ladder,
// whatever the caller's fallback is.
func TestParseRecognizesEveryRung(t *testing.T) {
	for _, want := range Modes() {
		m, ok := Parse(string(want), Off)
		if m != want || !ok {
			t.Errorf("Parse(%q, off) = (%q,%v), want (%q,true)", want, m, ok, want)
		}
	}
}

// TestUnknownValueFallsTheCALLERSWay is the #6090 confusion risk made a test: the
// shared parser must NOT hardcode a direction. A routing guard fails toward off
// (never route on a typo); a deliberately on-by-default curate fails toward on
// (never silently disable itself on a typo). Both get ok=false, so either can warn.
func TestUnknownValueFallsTheCALLERSWay(t *testing.T) {
	cases := []struct {
		name     string
		fallback Mode
	}{
		{"guard fails closed", Off},
		{"co-travel fails to the safe observe rung", Shadow},
		{"on-by-default curate fails open", On},
	}
	for _, c := range cases {
		m, ok := Parse("bogus", c.fallback)
		if m != c.fallback {
			t.Errorf("%s: Parse(\"bogus\", %q) = %q, want the caller's fallback %q", c.name, c.fallback, m, c.fallback)
		}
		if ok {
			t.Errorf("%s: Parse(\"bogus\", %q) reported ok=true; an unrecognized value must always be reportable", c.name, c.fallback)
		}
	}
	// The retired spellings are unknown values, not aliases: the shared parser
	// never silently accepts them. A leaf that must keep one accepts it at its own
	// seam (memorycotravel maps the legacy env `live` before parsing).
	for _, retired := range []string{"default", "live"} {
		if m, ok := Parse(retired, Off); ok || m != Off {
			t.Errorf("Parse(%q, off) = (%q,%v), want (off,false): retired spellings are not rungs", retired, m, ok)
		}
	}
}

// TestParseInRejectsRungsALeafDoesNotImplement: a correctly spelled rung outside
// the leaf's subset is refused exactly like gibberish — a curate over a prompt
// body has no narrow scope to canary into, so `canary` is not a valid answer there.
func TestParseInRejectsRungsALeafDoesNotImplement(t *testing.T) {
	curate := []Mode{Off, Shadow, On} // promptmmu's subset: no canary rung
	if m, ok := ParseIn("canary", curate, On); ok || m != On {
		t.Fatalf("ParseIn(\"canary\", {off,shadow,on}, on) = (%q,%v), want (on,false)", m, ok)
	}
	for _, want := range curate {
		if m, ok := ParseIn(string(want), curate, On); m != want || !ok {
			t.Errorf("ParseIn(%q, curate, on) = (%q,%v), want (%q,true)", want, m, ok, want)
		}
	}
	if m, ok := ParseIn("", curate, On); m != On || !ok {
		t.Fatalf("ParseIn(\"\", curate, on) = (%q,%v), want (on,true)", m, ok)
	}
}

// TestRungSemantics: Computes/Applies describe the rung, and an unrecognized mode
// does neither — a typo can never make a leaf compute or apply.
func TestRungSemantics(t *testing.T) {
	cases := []struct {
		m        Mode
		computes bool
		applies  bool
	}{
		{Off, false, false},
		{Shadow, true, false},
		{Canary, true, true},
		{On, true, true},
		{Mode(""), false, false},
		{Mode("default"), false, false},
		{Mode("live"), false, false},
	}
	for _, c := range cases {
		if got := c.m.Computes(); got != c.computes {
			t.Errorf("%q.Computes() = %v, want %v", c.m, got, c.computes)
		}
		if got := c.m.Applies(); got != c.applies {
			t.Errorf("%q.Applies() = %v, want %v", c.m, got, c.applies)
		}
	}
}

// TestIn is the membership test a leaf's subset admission shares with ParseIn.
func TestIn(t *testing.T) {
	if !In(Shadow, []Mode{Off, Shadow}) {
		t.Errorf("Shadow must be in {off,shadow}")
	}
	if In(Canary, []Mode{Off, Shadow}) {
		t.Errorf("Canary must not be in {off,shadow}")
	}
	if In(Off, nil) {
		t.Errorf("nothing is a member of an empty allowed set")
	}
}

// TestStringIsTheWireSpelling: the rung renders as its own literal, so a JSON or
// log surface reads the same word the vocabulary defines.
func TestStringIsTheWireSpelling(t *testing.T) {
	if Off.String() != "off" || Shadow.String() != "shadow" || Canary.String() != "canary" || On.String() != "on" {
		t.Fatalf("rung spellings drifted: %q %q %q %q", Off, Shadow, Canary, On)
	}
}
