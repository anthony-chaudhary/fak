package corelocks

import (
	"strings"
	"testing"
)

// TestShippedFixtureParses proves the embedded declaration is well-formed and
// names only members of the closed vocabulary.
func TestShippedFixtureParses(t *testing.T) {
	tax, err := LoadFixture()
	if err != nil {
		t.Fatalf("shipped fixture must parse, got error: %v", err)
	}
	if len(tax.Classes) == 0 {
		t.Fatal("shipped fixture declared no classes")
	}
	for _, c := range tax.Classes {
		if !knownClasses[c.Name] {
			t.Errorf("class %q is not in the known vocabulary", c.Name)
		}
		if !knownReasons[c.Reason] {
			t.Errorf("class %q raises unknown reason %q", c.Name, c.Reason)
		}
	}
	// All five required classes must be present.
	want := []string{"hard-self", "serial-core", "soft-contract", "shadow-learn", "open-leaf"}
	have := map[string]bool{}
	for _, n := range tax.ClassNames() {
		have[n] = true
	}
	for _, n := range want {
		if !have[n] {
			t.Errorf("required class %q missing from shipped fixture", n)
		}
	}
}

// TestClassifyHardSelf is the required witness: a path under
// internal/adjudicator/** maps to hard-self with the self-modify reason.
func TestClassifyHardSelf(t *testing.T) {
	tax, err := LoadFixture()
	if err != nil {
		t.Fatalf("fixture parse: %v", err)
	}
	class, reason := tax.Classify("internal/adjudicator/decide.go")
	if class != "hard-self" {
		t.Errorf("internal/adjudicator/decide.go: class = %q, want %q", class, "hard-self")
	}
	if reason != "CORE_SELF_MODIFY" {
		t.Errorf("internal/adjudicator/decide.go: reason = %q, want %q", reason, "CORE_SELF_MODIFY")
	}
}

// TestClassifyOpenLeaf is the required witness: an ordinary leaf path no glob
// claims falls through to open-leaf with no reason.
func TestClassifyOpenLeaf(t *testing.T) {
	tax, err := LoadFixture()
	if err != nil {
		t.Fatalf("fixture parse: %v", err)
	}
	class, reason := tax.Classify("internal/somewidget/foo.go")
	if class != ClassOpenLeaf {
		t.Errorf("internal/somewidget/foo.go: class = %q, want %q", class, ClassOpenLeaf)
	}
	if reason != "" {
		t.Errorf("internal/somewidget/foo.go: reason = %q, want empty (a leaf raises no lock)", reason)
	}
}

// TestClassifySerialAndContract covers the other declared classes end to end.
func TestClassifySerialAndContract(t *testing.T) {
	tax, err := LoadFixture()
	if err != nil {
		t.Fatalf("fixture parse: %v", err)
	}
	cases := []struct {
		path       string
		wantClass  string
		wantReason string
	}{
		{"dos.toml", "serial-core", "CORE_SERIAL_REQUIRED"},
		{"internal/resume/engine.go", "serial-core", "CORE_SERIAL_REQUIRED"},
		{"internal/canon/canon.go", "soft-contract", "CORE_CONTRACT_WITNESS_MISSING"},
	}
	for _, tc := range cases {
		class, reason := tax.Classify(tc.path)
		if class != tc.wantClass || reason != tc.wantReason {
			t.Errorf("Classify(%q) = (%q, %q), want (%q, %q)",
				tc.path, class, reason, tc.wantClass, tc.wantReason)
		}
	}
}

// TestMostSpecificGlobWins proves the longest matching glob decides, so a
// deeper declaration overrides a shallower one.
func TestMostSpecificGlobWins(t *testing.T) {
	data := `
[[class]]
name   = "soft-contract"
reason = "CORE_CONTRACT_WITNESS_MISSING"
globs  = ["internal/**"]

[[class]]
name   = "hard-self"
reason = "CORE_SELF_MODIFY"
globs  = ["internal/adjudicator/**"]
`
	tax, err := Parse([]byte(data))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// The deeper internal/adjudicator/** glob must beat the broad internal/**.
	class, reason := tax.Classify("internal/adjudicator/decide.go")
	if class != "hard-self" || reason != "CORE_SELF_MODIFY" {
		t.Errorf("most-specific glob should win: got (%q, %q), want (hard-self, CORE_SELF_MODIFY)", class, reason)
	}
	// A non-adjudicator internal path falls to the broad class.
	class, reason = tax.Classify("internal/other/x.go")
	if class != "soft-contract" || reason != "CORE_CONTRACT_WITNESS_MISSING" {
		t.Errorf("broad glob fallback: got (%q, %q), want (soft-contract, CORE_CONTRACT_WITNESS_MISSING)", class, reason)
	}
}

func TestParseRejectsUnknownClass(t *testing.T) {
	data := `
[[class]]
name   = "totally-made-up"
reason = "CORE_SELF_MODIFY"
globs  = ["internal/x/**"]
`
	if _, err := Parse([]byte(data)); err == nil {
		t.Fatal("Parse must reject an unknown class name")
	} else if !strings.Contains(err.Error(), "unknown lock class") {
		t.Errorf("error should name the unknown-class fault, got: %v", err)
	}
}

func TestParseRejectsUnknownReason(t *testing.T) {
	data := `
[[class]]
name   = "hard-self"
reason = "CORE_NOT_A_REAL_TOKEN"
globs  = ["internal/x/**"]
`
	if _, err := Parse([]byte(data)); err == nil {
		t.Fatal("Parse must reject an unknown reason token")
	} else if !strings.Contains(err.Error(), "unknown reason token") {
		t.Errorf("error should name the unknown-reason fault, got: %v", err)
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	cases := []struct {
		name string
		data string
	}{
		{"missing name", "[[class]]\nreason = \"CORE_SELF_MODIFY\"\nglobs = [\"internal/x/**\"]\n"},
		{"no tables", "# just a comment\n"},
		{"unknown key", "[[class]]\nname = \"hard-self\"\nbogus = \"x\"\nglobs = [\"internal/x/**\"]\n"},
		{"key outside table", "name = \"hard-self\"\n"},
		{"unterminated string", "[[class]]\nname = \"hard-self\nglobs = [\"internal/x/**\"]\n"},
		{"non-class table", "[server]\nname = \"x\"\n"},
		{"missing globs on locking class", "[[class]]\nname = \"hard-self\"\nreason = \"CORE_SELF_MODIFY\"\n"},
		{"empty glob", "[[class]]\nname = \"hard-self\"\nreason = \"CORE_SELF_MODIFY\"\nglobs = [\"\"]\n"},
		{"duplicate class", "[[class]]\nname = \"hard-self\"\nreason = \"CORE_SELF_MODIFY\"\nglobs = [\"a/**\"]\n[[class]]\nname = \"hard-self\"\nreason = \"CORE_SELF_MODIFY\"\nglobs = [\"b/**\"]\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse([]byte(tc.data)); err == nil {
				t.Errorf("Parse(%q) should have failed but did not", tc.name)
			}
		})
	}
}

// TestUnclassifiedWhenNoReasonWiring proves a locking class declared without a
// reason returns CORE_LOCK_UNCLASSIFIED rather than an empty reason.
func TestUnclassifiedWhenNoReasonWiring(t *testing.T) {
	data := `
[[class]]
name   = "serial-core"
reason = ""
globs  = ["internal/x/**"]
`
	tax, err := Parse([]byte(data))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	class, reason := tax.Classify("internal/x/y.go")
	if class != "serial-core" {
		t.Errorf("class = %q, want serial-core", class)
	}
	if reason != ReasonUnclassified {
		t.Errorf("reason = %q, want %q", reason, ReasonUnclassified)
	}
}

// TestPathNormalization proves backslashes and ./ prefixes cannot dodge a glob.
func TestPathNormalization(t *testing.T) {
	tax, err := LoadFixture()
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	for _, p := range []string{
		`internal\adjudicator\decide.go`,
		`./internal/adjudicator/decide.go`,
		`internal/adjudicator`,
	} {
		class, _ := tax.Classify(p)
		if class != "hard-self" {
			t.Errorf("Classify(%q) = %q, want hard-self (normalization should bind it)", p, class)
		}
	}
}
