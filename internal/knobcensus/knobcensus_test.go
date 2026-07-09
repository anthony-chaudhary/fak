package knobcensus

import (
	"encoding/json"
	"testing"
)

const fixtureRoot = "testdata/repo"

// TestScanFixture pins the walker against a known fixture tree: three INTENT
// knobs, three HOUSEKEEPING knobs (one folded from #2199's context inventory),
// and none of the excluded plumbing/output knobs.
func TestScanFixture(t *testing.T) {
	census, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if census.Intent != 3 {
		t.Errorf("Intent = %d, want 3", census.Intent)
	}
	if census.Housekeeping != 3 {
		t.Errorf("Housekeeping = %d, want 3", census.Housekeeping)
	}
	if len(census.Knobs) != 6 {
		t.Fatalf("len(Knobs) = %d, want 6: %+v", len(census.Knobs), census.Knobs)
	}

	got := map[string]Knob{}
	for _, k := range census.Knobs {
		got[k.Key()] = k
	}
	for _, want := range []struct {
		key     string
		verdict Verdict
	}{
		{"flag:account", Intent},
		{"flag:account-refresh", Intent}, // strong-intent token overrides "refresh"
		{"env:FAK_GOAL_OBJECTIVE", Intent},
		{"flag:session-cooldown-ttl", Housekeeping},
		{"flag:ctx-view-budget", Housekeeping}, // folded from #2199, not re-derived
		{"env:FAK_GUARD_AUTO_REFRESH", Housekeeping},
	} {
		k, ok := got[want.key]
		if !ok {
			t.Errorf("missing expected knob %q", want.key)
			continue
		}
		if k.Verdict != want.verdict {
			t.Errorf("%q verdict = %q, want %q", want.key, k.Verdict, want.verdict)
		}
		if k.Disposition != k.Verdict.Disposition() || k.OwnerEpic != k.Verdict.OwnerEpic() {
			t.Errorf("%q disposition/epic disagree with verdict: %+v", want.key, k)
		}
		if k.File == "" || k.Line == 0 {
			t.Errorf("%q missing file:line provenance: %+v", want.key, k)
		}
	}
	// The over-match guards: none of these gate user behavior; none may appear.
	for _, bad := range []string{"flag:json", "flag:root", "env:FAK_ADDR"} {
		if _, ok := got[bad]; ok {
			t.Errorf("walker over-matched a non-behavior knob: %q", bad)
		}
	}
}

// TestScanDeterministic is the "run the verb twice → identical output" witness at
// the walker level: two scans of the same tree marshal to byte-identical JSON.
func TestScanDeterministic(t *testing.T) {
	a, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan a: %v", err)
	}
	b, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan b: %v", err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("scan not deterministic:\n a=%s\n b=%s", ja, jb)
	}
}

// TestVerdictDisposition pins the closed vocabulary → disposition/owner mapping.
func TestVerdictDisposition(t *testing.T) {
	if Intent.Disposition() != "promote" || Intent.OwnerEpic() != "#2208" {
		t.Errorf("INTENT should promote under #2208, got %s / %s", Intent.Disposition(), Intent.OwnerEpic())
	}
	if Housekeeping.Disposition() != "automate" || Housekeeping.OwnerEpic() != "#2198" {
		t.Errorf("HOUSEKEEPING should automate under #2198, got %s / %s", Housekeeping.Disposition(), Housekeeping.OwnerEpic())
	}
}
