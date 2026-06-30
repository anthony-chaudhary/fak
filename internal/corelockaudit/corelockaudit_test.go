package corelockaudit

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/corelocks"
)

// loadTaxonomy loads the shipped corelocks fixture or fails the test.
func loadTaxonomy(t *testing.T) *corelocks.Taxonomy {
	t.Helper()
	tax, err := corelocks.LoadFixture()
	if err != nil {
		t.Fatalf("corelocks.LoadFixture: %v", err)
	}
	return tax
}

// findingFor returns the finding for a given class, or fails if absent.
func findingFor(t *testing.T, r Report, class string) Finding {
	t.Helper()
	for _, f := range r.Findings {
		if f.Class == class {
			return f
		}
	}
	t.Fatalf("no finding for class %q in report; got %+v", class, r.Findings)
	return Finding{}
}

// TestAuditClassesEachLockClass drives a path under EACH declared class through
// the fold and asserts the class, reason token, and advisory verdict the audit
// must report. This is the per-class fixture coverage the ticket requires.
func TestAuditClassesEachLockClass(t *testing.T) {
	tax := loadTaxonomy(t)

	cases := []struct {
		name        string
		path        string
		wantClass   string
		wantReason  string
		wantVerdict Verdict
		wantWitness bool
	}{
		{
			name:        "hard-self under adjudicator",
			path:        "internal/adjudicator/decide.go",
			wantClass:   "hard-self",
			wantReason:  "CORE_SELF_MODIFY",
			wantVerdict: VerdictWarn,
			wantWitness: true,
		},
		{
			name:        "hard-self under corelocks itself",
			path:        "internal/corelocks/corelocks.go",
			wantClass:   "hard-self",
			wantReason:  "CORE_SELF_MODIFY",
			wantVerdict: VerdictWarn,
			wantWitness: true,
		},
		{
			name:        "serial-core dos.toml",
			path:        "dos.toml",
			wantClass:   "serial-core",
			wantReason:  "CORE_SERIAL_REQUIRED",
			wantVerdict: VerdictWarn,
			wantWitness: true,
		},
		{
			name:        "serial-core under resume",
			path:        "internal/resume/state.go",
			wantClass:   "serial-core",
			wantReason:  "CORE_SERIAL_REQUIRED",
			wantVerdict: VerdictWarn,
			wantWitness: true,
		},
		{
			name:        "soft-contract under canon",
			path:        "internal/canon/schema.go",
			wantClass:   "soft-contract",
			wantReason:  "CORE_CONTRACT_WITNESS_MISSING",
			wantVerdict: VerdictWarn,
			wantWitness: true,
		},
		{
			name:        "shadow-learn under rsiloop is advisory ok",
			path:        "internal/rsiloop/loop.go",
			wantClass:   "shadow-learn",
			wantReason:  "CORE_LOCK_UNCLASSIFIED",
			wantVerdict: VerdictOK,
			wantWitness: false,
		},
		{
			name:        "ordinary leaf is open-leaf ok",
			path:        "internal/corelockaudit/corelockaudit.go",
			wantClass:   "open-leaf",
			wantReason:  "",
			wantVerdict: VerdictOK,
			wantWitness: false,
		},
		{
			name:        "unrelated leaf is open-leaf ok",
			path:        "cmd/fak/info.go",
			wantClass:   "open-leaf",
			wantReason:  "",
			wantVerdict: VerdictOK,
			wantWitness: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := Audit(tax, []string{tc.path})
			if r.Changed != 1 {
				t.Fatalf("Changed = %d, want 1", r.Changed)
			}
			f := findingFor(t, r, tc.wantClass)
			if f.ReasonToken != tc.wantReason {
				t.Errorf("reason = %q, want %q", f.ReasonToken, tc.wantReason)
			}
			if f.Verdict != tc.wantVerdict {
				t.Errorf("verdict = %q, want %q", f.Verdict, tc.wantVerdict)
			}
			if f.LockID != tc.wantClass {
				t.Errorf("lock_id = %q, want %q", f.LockID, tc.wantClass)
			}
			gotWitness := len(f.RequiredWitnesses) > 0
			if gotWitness != tc.wantWitness {
				t.Errorf("has witness = %v, want %v (witnesses=%v)", gotWitness, tc.wantWitness, f.RequiredWitnesses)
			}
			if tc.wantWitness && f.SourceNote == "" {
				t.Errorf("warn finding missing source note")
			}
		})
	}
}

// TestAuditWarnIsNonFailing proves the central first-phase property: a warning
// is measurement-only. The fold returns no error (it cannot — Audit has no error
// return), produces verdict=warn, and the report's OK() stays true because no
// finding crossed into refuse.
func TestAuditWarnIsNonFailing(t *testing.T) {
	tax := loadTaxonomy(t)
	r := Audit(tax, []string{"internal/adjudicator/decide.go"})

	f := findingFor(t, r, "hard-self")
	if f.Verdict != VerdictWarn {
		t.Fatalf("verdict = %q, want warn", f.Verdict)
	}
	if r.Warnings != 1 {
		t.Errorf("Warnings = %d, want 1", r.Warnings)
	}
	if r.Refusals != 0 {
		t.Errorf("Refusals = %d, want 0", r.Refusals)
	}
	if !r.OK() {
		t.Errorf("OK() = false; a warn must NOT fail the audit in this phase")
	}
}

// TestAuditGroupsAndSorts confirms paths are grouped by class, deduplicated, and
// that findings + paths are sorted for determinism.
func TestAuditGroupsAndSorts(t *testing.T) {
	tax := loadTaxonomy(t)
	r := Audit(tax, []string{
		"internal/adjudicator/z.go",
		"internal/adjudicator/a.go",
		"internal/adjudicator/a.go", // duplicate, must collapse
		"cmd/fak/info.go",
		"dos.toml",
	})

	// Findings sorted by class name: hard-self, open-leaf, serial-core.
	gotClasses := []string{}
	for _, f := range r.Findings {
		gotClasses = append(gotClasses, f.Class)
	}
	wantClasses := []string{"hard-self", "open-leaf", "serial-core"}
	if !reflect.DeepEqual(gotClasses, wantClasses) {
		t.Errorf("finding classes = %v, want %v", gotClasses, wantClasses)
	}

	hs := findingFor(t, r, "hard-self")
	wantPaths := []string{"internal/adjudicator/a.go", "internal/adjudicator/z.go"}
	if !reflect.DeepEqual(hs.Paths, wantPaths) {
		t.Errorf("hard-self paths = %v, want %v (sorted, deduped)", hs.Paths, wantPaths)
	}
	// Changed counts every non-empty input including the duplicate (5 inputs).
	if r.Changed != 5 {
		t.Errorf("Changed = %d, want 5", r.Changed)
	}
}

// TestAuditEmptyAndNil checks the degenerate inputs: nil taxonomy and empty
// path set both yield a well-formed, passing, empty report.
func TestAuditEmptyAndNil(t *testing.T) {
	if r := Audit(nil, []string{"anything"}); len(r.Findings) != 0 || !r.OK() {
		t.Errorf("nil taxonomy: got %+v, want empty passing report", r)
	}
	tax := loadTaxonomy(t)
	r := Audit(tax, []string{"", "   "})
	if len(r.Findings) != 0 || r.Changed != 0 || !r.OK() {
		t.Errorf("blank paths: got %+v, want empty passing report", r)
	}
}

// TestReportJSONDeterministic asserts the JSON bytes are stable: the same report
// encodes identically across calls, and the document round-trips back to an
// equal report.
func TestReportJSONDeterministic(t *testing.T) {
	tax := loadTaxonomy(t)
	paths := []string{
		"internal/adjudicator/decide.go",
		"dos.toml",
		"cmd/fak/info.go",
		"internal/canon/schema.go",
		"internal/rsiloop/loop.go",
	}
	r := Audit(tax, paths)

	a, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	b, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON (2nd): %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("JSON not deterministic:\n%s\n---\n%s", a, b)
	}

	var round Report
	if err := json.Unmarshal(a, &round); err != nil {
		t.Fatalf("round-trip unmarshal: %v", err)
	}
	if !reflect.DeepEqual(round, r) {
		t.Errorf("round-trip mismatch:\n got %+v\nwant %+v", round, r)
	}
}

// TestReportJSONExactBytes pins the exact JSON for a known input so a schema or
// ordering change is caught.
func TestReportJSONExactBytes(t *testing.T) {
	tax := loadTaxonomy(t)
	r := Audit(tax, []string{"cmd/fak/info.go", "internal/adjudicator/decide.go"})
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}

	want := `{
  "findings": [
    {
      "lock_id": "hard-self",
      "class": "hard-self",
      "paths": [
        "internal/adjudicator/decide.go"
      ],
      "reason_token": "CORE_SELF_MODIFY",
      "required_witnesses": [
        "dos commit-audit HEAD",
        "dos review origin/main..HEAD"
      ],
      "verdict": "warn",
      "source_note": "core-lock taxonomy/internal/adjudicator: a self-modifying surface; the diff-witnessed commit audit must confirm the claim."
    },
    {
      "lock_id": "open-leaf",
      "class": "open-leaf",
      "paths": [
        "cmd/fak/info.go"
      ],
      "reason_token": "",
      "required_witnesses": [],
      "verdict": "ok",
      "source_note": ""
    }
  ],
  "changed": 2,
  "warnings": 1,
  "refusals": 0
}`
	if string(got) != want {
		t.Errorf("JSON bytes mismatch:\n got:\n%s\nwant:\n%s", got, want)
	}
}

// TestStringRender exercises the human render for a warn finding: it must name
// the class, the reason, and the witness to clear it.
func TestStringRender(t *testing.T) {
	tax := loadTaxonomy(t)
	r := Audit(tax, []string{"internal/adjudicator/decide.go", "cmd/fak/info.go"})
	s := r.String()
	for _, want := range []string{
		"core-lock audit:",
		"WARN", "hard-self", "CORE_SELF_MODIFY",
		"witness to clear:", "dos commit-audit HEAD",
		"OK", "open-leaf",
	} {
		if !contains(s, want) {
			t.Errorf("String() missing %q; got:\n%s", want, s)
		}
	}
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || (len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// TestSplitChanged checks the git-output parser in isolation (no git needed):
// dedupe, trim, sort, drop blanks, handle CRLF.
func TestSplitChanged(t *testing.T) {
	raw := "b/two.go\r\na/one.go\n\n  a/one.go  \nc/three.go\n"
	got := splitChanged(raw)
	want := []string{"a/one.go", "b/two.go", "c/three.go"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitChanged = %v, want %v", got, want)
	}
}

// TestChangedPathsEmptyRef confirms the I/O layer rejects an empty ref before
// shelling to git.
func TestChangedPathsEmptyRef(t *testing.T) {
	if _, err := ChangedPaths(".", ""); err == nil {
		t.Errorf("ChangedPaths with empty ref: want error, got nil")
	}
}
