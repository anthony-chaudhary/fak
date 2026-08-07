package provenance

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureCase is one committed live-version scenario. The fixture file — not this
// source — is the case list, so the known-positive and the adversarial banners are
// committed evidence a reviewer can read and extend without touching Go.
type fixtureCase struct {
	Name  string       `json:"name"`
	Kind  string       `json:"kind"` // "positive" | "adversarial" | "boundary"
	Pin   string       `json:"pin"`
	Probe VersionProbe `json:"probe"`
	// SubstringTrap marks a case whose RAW banner literally contains the pin — the
	// exact input on which the historical `strings.Contains(banner, pin)` check
	// returned a false green. The test asserts both halves: the naive check really
	// would have passed, and this contract really refuses.
	SubstringTrap bool   `json:"substring_trap,omitempty"`
	WantState     string `json:"want_state"`
	WantSatisfied bool   `json:"want_satisfied"`
	WantLive      string `json:"want_live,omitempty"`
}

type fixtureFile struct {
	Cases []fixtureCase `json:"cases"`
}

const fixtureDir = "testdata/toolversion"

func loadFixture(t *testing.T) fixtureFile {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir, "banners.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f fixtureFile
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("fixture carries no cases: the contract would pass vacuously")
	}
	return f
}

// TestToolVersionFixtures drives every committed banner through the contract and
// checks the typed state, the satisfied gate, and the normalized live reading.
// It covers the six required scenarios — exact match, path confusion, longer-token
// confusion, no-version output, missing executable, failing executable — plus the
// constraint and invalid pin kinds.
func TestToolVersionFixtures(t *testing.T) {
	f := loadFixture(t)
	for _, tc := range f.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			w := VerifyToolVersion(tc.Pin, tc.Probe)
			if got := w.State.String(); got != tc.WantState {
				t.Errorf("state = %q want %q (reason: %s)", got, tc.WantState, w.Reason)
			}
			if w.Satisfied() != tc.WantSatisfied {
				t.Errorf("Satisfied() = %v want %v (reason: %s)", w.Satisfied(), tc.WantSatisfied, w.Reason)
			}
			if w.Live != tc.WantLive {
				t.Errorf("live = %q want %q", w.Live, tc.WantLive)
			}
			// Criterion 4: the complete raw self-report is retained as evidence in
			// EVERY branch, including each refusal.
			if w.Raw != tc.Probe.Raw {
				t.Errorf("raw self-report not retained: got %q want %q", w.Raw, tc.Probe.Raw)
			}
			if w.Reason == "" {
				t.Error("every verdict must name its reason")
			}
		})
	}
}

// TestSubstringCheckWouldHaveLied is the anti-vacuity proof. For every trap case
// the fixture marks, the old banner-substring test genuinely returns true — so
// these inputs are not strawmen — and the parsed contract still refuses. If the
// contract ever regressed to substring matching, this test fails.
func TestSubstringCheckWouldHaveLied(t *testing.T) {
	f := loadFixture(t)
	traps := 0
	for _, tc := range f.Cases {
		if !tc.SubstringTrap {
			continue
		}
		traps++
		if !strings.Contains(tc.Probe.Raw, tc.Pin) {
			t.Errorf("%s: fixture claims a substring trap, but the banner does not contain the pin %q — the case proves nothing", tc.Name, tc.Pin)
		}
		if w := VerifyToolVersion(tc.Pin, tc.Probe); w.Satisfied() {
			t.Errorf("%s: banner substring satisfied pin %q — the identity comparison regressed to Contains", tc.Name, tc.Pin)
		}
	}
	if traps == 0 {
		t.Fatal("no substring-trap fixture: the adversarial half of the contract is untested")
	}
}

// TestKnownPositiveCanSucceed proves the signal is not stuck refusing everything:
// a real, fully matching self-report must reach VersionMatch. A gate that can only
// refuse is as useless as one that only passes.
func TestKnownPositiveCanSucceed(t *testing.T) {
	f := loadFixture(t)
	for _, tc := range f.Cases {
		if tc.Kind == "positive" && tc.WantSatisfied {
			if w := VerifyToolVersion(tc.Pin, tc.Probe); w.Satisfied() {
				return
			}
		}
	}
	t.Fatal("no committed known-positive fixture reaches VersionMatch: the contract cannot succeed")
}

// TestDistinctRefusalStates — criterion 3: measured absence, command failure,
// versionless output, and mismatch are FOUR different states, never collapsed.
func TestDistinctRefusalStates(t *testing.T) {
	seen := map[string]bool{}
	f := loadFixture(t)
	for _, tc := range f.Cases {
		seen[VerifyToolVersion(tc.Pin, tc.Probe).State.String()] = true
	}
	for _, want := range []string{"match", "mismatch", "absent", "probe_failed", "no_version", "constraint_pin", "pin_invalid"} {
		if !seen[want] {
			t.Errorf("state %q is never exercised by the committed fixtures", want)
		}
	}
}

// TestConstraintPinNeverExactMatched — criterion 7. A constraint pin is typed
// separately and refused, even when its own text would trivially substring-match
// the banner, and even when the live version genuinely satisfies the range.
func TestConstraintPinNeverExactMatched(t *testing.T) {
	probe := VersionProbe{Tool: "mytool", Raw: "mytool version 1.5.0\n", Found: true}
	for _, pin := range []string{">=1.2.0", "^1.2.3", "~1.2", "1.2.*", "1.x", ">=1.0 <2.0", "1.x || 2.x", "*"} {
		if k := ClassifyPin(pin); k != PinConstraint {
			t.Errorf("ClassifyPin(%q) = %v want constraint", pin, k)
		}
		w := VerifyToolVersion(pin, probe)
		if w.State != VersionConstraintPin {
			t.Errorf("pin %q: state = %v want constraint_pin", pin, w.State)
		}
		if w.Satisfied() || w.Established() {
			t.Errorf("pin %q: a constraint must not be satisfied or established by the exact witness", pin)
		}
	}
	// The live version literally IS 1.5.0 and the range >=1.2.0 genuinely holds —
	// the point is that this witness does not decide ranges, so it refuses instead
	// of guessing.
	if w := VerifyToolVersion(">=1.2.0", probe); w.Satisfied() {
		t.Error("a satisfiable range must still be refused by the exact-version witness")
	}
}

// TestParseToolVersionTyping covers the parser directly: the v-prefix and
// punctuation normalization it must accept, and the name-fused / non-numeric
// tokens it must reject rather than mine for digits.
func TestParseToolVersionTyping(t *testing.T) {
	ok := map[string]string{
		"1.2.3": "1.2.3", "v20.11.1": "20.11.1", "V2.0": "2.0",
		"  1.2.3  ": "1.2.3", "(2.43.0)": "2.43.0", "1.2.3-rc1": "1.2.3-rc1",
		"1.2.3+build7": "1.2.3-build7", "14": "14", "01.2": "1.2",
	}
	for in, want := range ok {
		v, good := ParseToolVersion(in)
		if !good {
			t.Errorf("ParseToolVersion(%q) rejected, want %q", in, want)
			continue
		}
		if got := v.String(); got != want {
			t.Errorf("ParseToolVersion(%q).String() = %q want %q", in, got, want)
		}
	}
	for _, in := range []string{"", "tool-1.2.3", "go1.22.3", "latest", "1..2", "1.2.", "v", "abc", "1.2.x"} {
		if v, good := ParseToolVersion(in); good {
			t.Errorf("ParseToolVersion(%q) accepted as %q, want rejection", in, v.String())
		}
	}
}

// TestEqualIsArityStrict — criterion 6, at the comparison layer: a longer token
// never satisfies a shorter pin, in either direction, with or without padding.
func TestEqualIsArityStrict(t *testing.T) {
	pairs := [][2]string{
		{"1.2.3", "1.2.30"}, {"1.2.3", "1.2.3.4"}, {"1.2.3", "1.2"},
		{"1.2", "1.2.0"}, {"1.2.3", "1.2.3-rc1"},
	}
	for _, p := range pairs {
		a, okA := ParseToolVersion(p[0])
		b, okB := ParseToolVersion(p[1])
		if !okA || !okB {
			t.Fatalf("fixture parse failed for %v", p)
		}
		if a.Equal(b) || b.Equal(a) {
			t.Errorf("%q and %q must not compare equal", p[0], p[1])
		}
	}
	same, _ := ParseToolVersion("v1.2.3")
	other, _ := ParseToolVersion("1.2.3")
	if !same.Equal(other) {
		t.Error("normalized equal versions must compare equal")
	}
}

// TestReceiptGolden pins the machine-readable receipt for the accepted case and
// the refusal case. Regenerate with FAK_UPDATE_GOLDEN=1 go test ./internal/provenance.
func TestReceiptGolden(t *testing.T) {
	f := loadFixture(t)
	byName := map[string]fixtureCase{}
	for _, tc := range f.Cases {
		byName[tc.Name] = tc
	}
	accepted, okA := byName["exact-match-known-positive"]
	refused, okR := byName["path-confusion"]
	if !okA || !okR {
		t.Fatal("fixture must carry both the accepted and the refusal receipt cases")
	}
	doc := map[string]json.RawMessage{
		"accepted": json.RawMessage(VerifyToolVersion(accepted.Pin, accepted.Probe).Receipt()),
		"refused":  json.RawMessage(VerifyToolVersion(refused.Pin, refused.Probe).Receipt()),
	}
	got, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	got = append(got, '\n')
	path := filepath.Join(fixtureDir, "receipt.json")
	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatalf("update golden: %v", err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden receipt: %v", err)
	}
	if !bytes.Equal(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), got) {
		t.Errorf("receipt drifted from the committed golden.\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}
