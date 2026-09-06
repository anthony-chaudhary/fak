package workerenvelope

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// TestParseValidFixture proves a well-formed shipped envelope round-trips
// through Parse (decode + validate) and preserves every field.
func TestParseValidFixture(t *testing.T) {
	const fixture = `{
		"status": "shipped",
		"issue": 1795,
		"commit_sha": "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90",
		"tests_run": ["go test ./internal/workerenvelope/"],
		"witness": "commit c99f5c02"
	}`

	r, err := Parse([]byte(fixture))
	if err != nil {
		t.Fatalf("Parse of valid fixture failed: %v", err)
	}
	if r.Status != StatusShipped {
		t.Errorf("status = %q, want %q", r.Status, StatusShipped)
	}
	if r.Issue != 1795 {
		t.Errorf("issue = %d, want 1795", r.Issue)
	}
	if r.CommitSHA != "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90" {
		t.Errorf("commit_sha = %q, unexpected", r.CommitSHA)
	}
	if len(r.TestsRun) != 1 || r.TestsRun[0] != "go test ./internal/workerenvelope/" {
		t.Errorf("tests_run = %v, unexpected", r.TestsRun)
	}
	if r.Witness != "commit c99f5c02" {
		t.Errorf("witness = %q, unexpected", r.Witness)
	}
	if r.Blocker != "" {
		t.Errorf("blocker = %q, want empty", r.Blocker)
	}
}

// TestParseShortSHA proves a 7-char short SHA is accepted on a shipped result.
func TestParseShortSHA(t *testing.T) {
	const fixture = `{"status":"shipped","issue":42,"commit_sha":"c99f5c0","witness":"log: run.log"}`
	if _, err := Parse([]byte(fixture)); err != nil {
		t.Fatalf("short-SHA shipped fixture should validate, got: %v", err)
	}
}

// TestValidateBlockedAndNotYet proves the two non-shipped statuses validate
// when they name a blocker.
func TestValidateBlockedAndNotYet(t *testing.T) {
	for _, st := range []Status{StatusBlocked, StatusNotYet} {
		r := Result{Status: st, Issue: 7, Blocker: "peer WIP breaks internal/model build"}
		if err := r.Validate(); err != nil {
			t.Errorf("%s result with a blocker should validate, got: %v", st, err)
		}
	}
}

// TestValidateMalformed drives the malformed fixtures: each must fail, and the
// error must name the field that broke the contract.
func TestValidateMalformed(t *testing.T) {
	cases := []struct {
		name    string
		r       Result
		wantSub string // substring the error must contain
	}{
		{
			name:    "shipped missing witness",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "c99f5c02"},
			wantSub: "witness",
		},
		{
			name:    "shipped missing commit_sha",
			r:       Result{Status: StatusShipped, Issue: 1, Witness: "log: run.log"},
			wantSub: "commit_sha",
		},
		{
			name:    "shipped carries a blocker",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "c99f5c02", Witness: "commit c99f5c02", Blocker: "flaky"},
			wantSub: "must not carry a blocker",
		},
		{
			name:    "blocked missing blocker",
			r:       Result{Status: StatusBlocked, Issue: 1},
			wantSub: "requires a blocker",
		},
		{
			name:    "not_yet missing blocker",
			r:       Result{Status: StatusNotYet, Issue: 1},
			wantSub: "requires a blocker",
		},
		{
			name:    "issue <= 0",
			r:       Result{Status: StatusBlocked, Issue: 0, Blocker: "x"},
			wantSub: "issue must be > 0",
		},
		{
			name:    "negative issue",
			r:       Result{Status: StatusBlocked, Issue: -5, Blocker: "x"},
			wantSub: "issue must be > 0",
		},
		{
			name:    "bad sha shape (too short)",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "abc12", Witness: "w"},
			wantSub: "hex sha",
		},
		{
			name:    "bad sha shape (non-hex)",
			r:       Result{Status: StatusBlocked, Issue: 1, CommitSHA: "zzzzzzz", Blocker: "x"},
			wantSub: "hex sha",
		},
		{
			name:    "unknown status",
			r:       Result{Status: Status("done"), Issue: 1},
			wantSub: "invalid status",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.r.Validate()
			if err == nil {
				t.Fatalf("expected validation error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// TestParseRejectsMalformedJSON proves Parse surfaces a decode error (distinct
// from a validation error) for syntactically broken JSON.
func TestParseRejectsMalformedJSON(t *testing.T) {
	_, err := Parse([]byte(`{"status": "shipped", "issue":`))
	if err == nil {
		t.Fatal("expected decode error for truncated JSON, got nil")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Errorf("error %q should mention decode", err.Error())
	}
}

// TestParseMalformedFixtureFails is the ticket's explicit witness: a malformed
// (well-formed JSON, contract-violating) fixture parses through JSON but fails
// Validate.
func TestParseMalformedFixtureFails(t *testing.T) {
	const fixture = `{"status":"shipped","issue":1795,"commit_sha":"c99f5c02"}` // no witness
	if _, err := Parse([]byte(fixture)); err == nil {
		t.Fatal("malformed fixture (shipped without witness) should fail Parse")
	}
}

func TestStatusInvariants(t *testing.T) {
	validStatuses := []Status{StatusShipped, StatusBlocked, StatusNotYet}
	for _, st := range validStatuses {
		if !st.valid() {
			t.Errorf("expected valid() == true for %q", st)
		}
	}

	invalidStatuses := []Status{
		"",
		"done",
		"Shipped",
		"SHIPPED",
		"blocked ",
		" not_yet",
		"pending",
		"aborted",
	}
	for _, st := range invalidStatuses {
		if st.valid() {
			t.Errorf("expected valid() == false for %q", st)
		}
	}
}

func TestLooksLikeSHAInvariants(t *testing.T) {
	cases := []struct {
		input string
		want  bool
	}{
		{"", false},
		{"123456", false}, // 6 chars, too short
		{"1234567", true}, // 7 chars, min bound
		{"abcdef0", true},
		{"ABCDEF0", true},
		{"c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90", true}, // 40 chars, max bound
		{"C99F5C02A1B2C3D4E5F60718293A4B5C6D7E8F90", true},
		{"c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f901", false}, // 41 chars, too long
		{"123456g", false},  // non-hex char 'g'
		{"123456z", false},  // non-hex char 'z'
		{"123456 ", false},  // trailing space
		{" 1234567", false}, // leading space
		{"1234-567", false}, // dash
	}
	for _, tc := range cases {
		got := looksLikeSHA(tc.input)
		if got != tc.want {
			t.Errorf("looksLikeSHA(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestDispatchContractInvariants(t *testing.T) {
	t.Run("empty commit_sha on shipped fails", func(t *testing.T) {
		r := Result{
			Status:    StatusShipped,
			Issue:     1,
			CommitSHA: "",
			Witness:   "commit c99f5c02",
		}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "requires a commit_sha") {
			t.Fatalf("expected requires a commit_sha error, got: %v", err)
		}
	})

	t.Run("whitespace commit_sha on shipped fails", func(t *testing.T) {
		r := Result{
			Status:    StatusShipped,
			Issue:     1,
			CommitSHA: "       ",
			Witness:   "commit c99f5c02",
		}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "commit_sha") {
			t.Fatalf("expected commit_sha error, got: %v", err)
		}
	})

	t.Run("whitespace witness on shipped fails", func(t *testing.T) {
		r := Result{
			Status:    StatusShipped,
			Issue:     1,
			CommitSHA: "c99f5c02",
			Witness:   "  \t \n ",
		}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "requires a witness") {
			t.Fatalf("expected witness error, got: %v", err)
		}
	})

	t.Run("whitespace blocker on shipped passes", func(t *testing.T) {
		r := Result{
			Status:    StatusShipped,
			Issue:     1,
			CommitSHA: "c99f5c02",
			Witness:   "commit c99f5c02",
			Blocker:   "   ",
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("expected whitespace blocker to count as empty on shipped, got error: %v", err)
		}
	})

	t.Run("whitespace blocker on blocked fails", func(t *testing.T) {
		r := Result{
			Status:  StatusBlocked,
			Issue:   1,
			Blocker: "   \t",
		}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "requires a blocker") {
			t.Fatalf("expected blocker error, got: %v", err)
		}
	})

	t.Run("whitespace blocker on not_yet fails", func(t *testing.T) {
		r := Result{
			Status:  StatusNotYet,
			Issue:   1,
			Blocker: " \n ",
		}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "requires a blocker") {
			t.Fatalf("expected blocker error, got: %v", err)
		}
	})

	t.Run("blocked with optional valid commit_sha passes", func(t *testing.T) {
		r := Result{
			Status:    StatusBlocked,
			Issue:     1,
			CommitSHA: "c99f5c02",
			Blocker:   "pipeline failure",
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("expected blocked with valid commit_sha to pass, got: %v", err)
		}
	})
}

func TestParseFailClosedInvariants(t *testing.T) {
	cases := []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{"empty payload", []byte{}, "decode"},
		{"null payload", []byte("null"), "invalid status"},
		{"numeric payload", []byte("12345"), "decode"},
		{"array payload", []byte("[]"), "decode"},
		{"trailing syntax error", []byte(`{"status":"shipped"} trailing`), "decode"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse(tc.data)
			if err == nil {
				t.Fatalf("expected Parse to fail, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func countNonEmptyLines(b []byte) int {
	scanner := bufio.NewScanner(strings.NewReader(string(b)))
	n := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			n++
		}
	}
	return n
}

func checkFormulaicComment(cg *ast.CommentGroup) (bool, bool) {
	if cg == nil {
		return false, false
	}
	text := strings.TrimSpace(cg.Text())
	lower := strings.ToLower(text)

	hasMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.HasPrefix(lower, "invariant") ||
		strings.HasPrefix(lower, "guard") ||
		strings.HasPrefix(lower, "contract") ||
		strings.HasPrefix(lower, "fail-closed")

	if !hasMarker {
		return false, false
	}

	words := strings.Fields(lower)
	if len(words) <= 3 {
		return true, true
	}

	keywordCount := 0
	for _, w := range words {
		cleaned := strings.Trim(w, ":,.-()")
		if cleaned == "invariant" || cleaned == "contract" || cleaned == "guard" ||
			cleaned == "fail-closed" || cleaned == "precondition" || cleaned == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

func isSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if text == "" {
		return false
	}
	words := strings.Fields(text)
	if len(words) < 3 {
		return false
	}
	firstWord := strings.Trim(strings.ToLower(words[0]), ":,.-()")
	nameLower := strings.ToLower(name)
	if firstWord == nameLower && len(words) <= 4 {
		fillers := map[string]bool{
			"is": true, "a": true, "the": true, "an": true, "for": true, "of": true,
		}
		meaningful := 0
		for _, w := range words[1:] {
			wl := strings.ToLower(strings.Trim(w, ":,.-()"))
			if !fillers[wl] && wl != nameLower {
				meaningful++
			}
		}
		if meaningful < 2 {
			return false
		}
	}
	return true
}

func TestCommentHygieneAndNoFormulaicNoise(t *testing.T) {
	fset := token.NewFileSet()
	filename := "workerenvelope.go"

	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed to read %s: %v", filename, err)
	}

	node, err := parser.ParseFile(fset, filename, content, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", filename, err)
	}

	codeLines := countNonEmptyLines(content)
	commentLines := 0
	formulaicCount := 0
	hasFiller := false

	for _, cg := range node.Comments {
		for _, c := range cg.List {
			commentLines += strings.Count(c.Text, "\n") + 1
		}
		isForm, isFill := checkFormulaicComment(cg)
		if isForm {
			formulaicCount++
			t.Logf("%s: detected formulaic comment: %q", filename, strings.TrimSpace(cg.Text()))
		}
		if isFill {
			hasFiller = true
		}
	}

	commentRatio := float64(commentLines) / float64(codeLines)
	if codeLines > 30 && commentRatio > 0.35 {
		t.Errorf("%s: comment bloat ratio %.2f exceeds 0.35 (comments: %d, code: %d)",
			filename, commentRatio, commentLines, codeLines)
	}

	if formulaicCount > 0 || hasFiller {
		t.Errorf("%s: formulaic comments detected: count=%d, filler=%v",
			filename, formulaicCount, hasFiller)
	}

	exported := 0
	documented := 0
	var undocumented []string

	for _, decl := range node.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if ast.IsExported(d.Name.Name) {
				exported++
				if isSubstantiveDoc(d.Name.Name, d.Doc) {
					documented++
				} else {
					undocumented = append(undocumented, d.Name.Name)
				}
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if ast.IsExported(s.Name.Name) {
						exported++
						doc := s.Doc
						if doc == nil {
							doc = d.Doc
						}
						if isSubstantiveDoc(s.Name.Name, doc) {
							documented++
						} else {
							undocumented = append(undocumented, s.Name.Name)
						}
					}
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if ast.IsExported(name.Name) {
							exported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if isSubstantiveDoc(name.Name, doc) {
								documented++
							} else {
								undocumented = append(undocumented, name.Name)
							}
						}
					}
				}
			}
		}
	}

	if exported > 0 {
		ratio := float64(documented) / float64(exported)
		if ratio < 0.90 {
			t.Errorf("%s: documented exports ratio %.2f < 0.90 (undocumented: %v)", filename, ratio, undocumented)
		}
	}
}
