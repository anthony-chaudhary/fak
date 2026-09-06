package guardcorpus

import (
	"bufio"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// planted is a synthetic session covering every fold branch: an allow, an
// explained deny (reason+witness), a quarantine, a blank-reason deny (honesty
// hole), an unknown verdict, a witnessless block, and a child crash.
func planted() []journal.Row {
	return []journal.Row{
		{Seq: 1, TSUnixNano: 100, Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", By: "floor", ArgsLabel: "path=main.go"},
		{Seq: 2, TSUnixNano: 200, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "floor", Witness: "rm -rf /", ArgsLabel: "command=rm"},
		{Seq: 3, TSUnixNano: 300, Kind: "QUARANTINE", Tool: "WebFetch", Verdict: "QUARANTINE", Reason: "SECRET_DISCOVERED", By: "secretgate", Witness: "sk-redacted-claim"},
		{Seq: 4, TSUnixNano: 400, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "", By: "floor"},            // blank reason on deny
		{Seq: 5, TSUnixNano: 500, Kind: "DECIDE", Tool: "Edit", Verdict: "WEIRD", By: "advmodel"},                  // unknown verdict
		{Seq: 6, TSUnixNano: 600, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "OFF_TRUNK", By: "gitgate"}, // witnessless block (reason but no witness)
		{Seq: 7, TSUnixNano: 700, Kind: "CHILD_CRASH", Tool: "claude", Reason: "SIGNAL_CRASH"},
	}
}

func TestFoldCountsHonestyAndOutcome(t *testing.T) {
	meta := SessionMeta{TraceID: "t1", Agent: "claude-code", HostClass: "desktop", PolicyDigest: "sha256:abc", ChainVerified: true}
	rec, ex := Fold(meta, planted())

	if rec.Schema != SessionSchema {
		t.Fatalf("schema = %q", rec.Schema)
	}
	// 6 decision rows (allow, deny, quarantine, blank-deny, unknown, witnessless-deny); the crash is not a decision.
	if rec.ToolCalls != 6 {
		t.Fatalf("tool_calls = %d, want 6", rec.ToolCalls)
	}
	if rec.ByVerdict["DENY"] != 3 || rec.ByVerdict["ALLOW"] != 1 || rec.ByVerdict["QUARANTINE"] != 1 {
		t.Fatalf("by_verdict = %v", rec.ByVerdict)
	}
	if rec.ByGate["floor"] != 3 || rec.ByGate["secretgate"] != 1 || rec.ByGate["gitgate"] != 1 {
		t.Fatalf("by_gate = %v", rec.ByGate)
	}
	if rec.HonestyHoles.BlankReasonOnDeny != 1 {
		t.Fatalf("blank_reason_on_deny = %d, want 1", rec.HonestyHoles.BlankReasonOnDeny)
	}
	if rec.HonestyHoles.UnknownVerdict != 1 {
		t.Fatalf("unknown_verdict = %d, want 1", rec.HonestyHoles.UnknownVerdict)
	}
	if rec.HonestyHoles.WitnesslessBlock != 1 {
		t.Fatalf("witnessless_block = %d, want 1", rec.HonestyHoles.WitnesslessBlock)
	}
	if rec.HonestyHoles.ChildCrash != 1 {
		t.Fatalf("child_crash = %d, want 1", rec.HonestyHoles.ChildCrash)
	}
	if rec.Outcome != OutcomeCrashed {
		t.Fatalf("outcome = %q, want %q", rec.Outcome, OutcomeCrashed)
	}
	if rec.StartedUnixNano != 100 || rec.EndedUnixNano != 700 {
		t.Fatalf("span = [%d,%d], want [100,700]", rec.StartedUnixNano, rec.EndedUnixNano)
	}
	// Policy attribution (Gap-2) rides every record and every example.
	if rec.PolicyDigest != "sha256:abc" {
		t.Fatalf("record policy_digest = %q", rec.PolicyDigest)
	}
	// Every source row that represents a decision or crash emits one example.
	if len(ex) != len(planted())-1 { // the unknown verdict is counted as an honesty hole, not training data
		t.Fatalf("examples = %d, want one for every eligible planted row", len(ex))
	}
	for _, e := range ex {
		if e.Schema != ExampleSchema {
			t.Fatalf("example schema = %q", e.Schema)
		}
		if e.PolicyDigest != "sha256:abc" {
			t.Fatalf("example missing policy attribution: %+v", e)
		}
	}
}

func TestFoldDeterministicAndPure(t *testing.T) {
	meta := SessionMeta{TraceID: "t1", PolicyDigest: "d"}
	rows := planted()
	r1, e1 := Fold(meta, rows)
	r2, e2 := Fold(meta, rows)
	b1, _ := json.Marshal(struct {
		R SessionRecord
		E []Example
	}{r1, e1})
	b2, _ := json.Marshal(struct {
		R SessionRecord
		E []Example
	}{r2, e2})
	if string(b1) != string(b2) {
		t.Fatalf("fold is not deterministic:\n%s\n%s", b1, b2)
	}
}

func TestFoldRateLimitExitIsNotACrash(t *testing.T) {
	rows := []journal.Row{
		{Seq: 1, TSUnixNano: 10, Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", By: "floor"},
		{Seq: 2, TSUnixNano: 20, Kind: "CHILD_CRASH", Tool: "claude", Reason: "usage_limit reached"},
	}
	rec, _ := Fold(SessionMeta{}, rows)
	if rec.HonestyHoles.ChildCrash != 0 {
		t.Fatalf("rate-limit exit counted as crash: %d", rec.HonestyHoles.ChildCrash)
	}
	if rec.Outcome != OutcomeRateLimited {
		t.Fatalf("outcome = %q, want %q", rec.Outcome, OutcomeRateLimited)
	}
}

func TestFoldAllowSampleIsBounded(t *testing.T) {
	var rows []journal.Row
	for i := 0; i < maxAllowExamples+5; i++ {
		rows = append(rows, journal.Row{Seq: uint64(i + 1), TSUnixNano: int64(i + 1), Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", By: "floor"})
	}
	rec, ex := Fold(SessionMeta{}, rows)
	if rec.ToolCalls != maxAllowExamples+5 {
		t.Fatalf("tool_calls = %d, want %d", rec.ToolCalls, maxAllowExamples+5)
	}
	if len(ex) != maxAllowExamples {
		t.Fatalf("allow examples = %d, want capped at %d", len(ex), maxAllowExamples)
	}
}

// TestFoldAdvisoryIsNotAnUnknownVerdict locks the finding that surfaced when the
// fold ran over this host's real journals: a TOOL_DEFINITION_PRUNED / ADVISORY
// row (the dominant real verdict) is legitimate and must NOT count as an
// unknown-verdict honesty hole (guardrsi's KnownVerdicts miscounts it — see the
// package doc). A regression here would re-introduce that false honesty signal.
func TestFoldAdvisoryIsNotAnUnknownVerdict(t *testing.T) {
	rows := []journal.Row{
		{Seq: 1, TSUnixNano: 1, Kind: "TOOL_DEFINITION_PRUNED", Tool: "ReportFindings", Verdict: "ADVISORY", Reason: "DEFAULT_DENY", By: "tool-definition-pruner"},
	}
	rec, _ := Fold(SessionMeta{}, rows)
	if rec.HonestyHoles.UnknownVerdict != 0 {
		t.Fatalf("ADVISORY counted as unknown_verdict: %d (would replicate the guardrsi miscount)", rec.HonestyHoles.UnknownVerdict)
	}
	if rec.ByVerdict["ADVISORY"] != 1 {
		t.Fatalf("ADVISORY not recorded in by_verdict: %v", rec.ByVerdict)
	}
}

// TestGoldenCorpusRoundTrip is the regression harness (GUARD-SESSION-DATASET-PLAN
// A5): fold the committed input journal fixture and assert the produced dataset
// matches the committed golden byte-for-byte. A fold change that silently flips a
// record's counts, honesty holes, or example set fails here. Regenerate the
// golden intentionally with FAK_UPDATE_GOLDEN=1 when a change is deliberate.
func TestGoldenCorpusRoundTrip(t *testing.T) {
	meta := SessionMeta{TraceID: "sess-golden", Agent: "claude-code", HostClass: "test", PolicyDigest: "sha256:golden", ChainVerified: true}
	rows := readJournalFixture(t, filepath.Join("testdata", "session.journal.jsonl"))
	rec, ex := Fold(meta, rows)

	got := marshalJSONL(t, append([]any{rec}, examplesAsAny(ex)...))
	goldenPath := filepath.Join("testdata", "corpus.golden.jsonl")
	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(goldenPath, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("updated golden %s", goldenPath)
		return
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden (regenerate with FAK_UPDATE_GOLDEN=1): %v", err)
	}
	wantStr := strings.ReplaceAll(string(want), "\r\n", "\n")
	if got != wantStr {
		t.Fatalf("corpus golden mismatch — a fold change flipped the dataset.\n--- got ---\n%s\n--- want ---\n%s", got, wantStr)
	}
}

func readJournalFixture(t testing.TB, path string) []journal.Row {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []journal.Row
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r journal.Row
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("fixture parse: %v", err)
		}
		rows = append(rows, r)
	}
	return rows
}

func examplesAsAny(ex []Example) []any {
	out := make([]any, len(ex))
	for i := range ex {
		out[i] = ex[i]
	}
	return out
}

func marshalJSONL(t *testing.T, rows []any) string {
	t.Helper()
	var b strings.Builder
	for _, r := range rows {
		j, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(j)
		b.WriteByte('\n')
	}
	return b.String()
}

func TestFoldEmptyIsCleanZero(t *testing.T) {
	rec, ex := Fold(SessionMeta{TraceID: "empty"}, nil)
	if rec.ToolCalls != 0 || rec.Outcome != OutcomeClean || len(ex) != 0 {
		t.Fatalf("empty fold = %+v / %d examples", rec, len(ex))
	}
	if rec.ByVerdict != nil || rec.ByReason != nil || rec.ByGate != nil {
		t.Fatalf("empty fold should nil out empty maps: %+v", rec)
	}
}

func TestFoldPreservesRowOrdering(t *testing.T) {
	rows := []journal.Row{
		{Seq: 1, TSUnixNano: 100, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "FIRST"},
		{Seq: 2, TSUnixNano: 200, Kind: "QUARANTINE", Tool: "Bash", Verdict: "QUARANTINE", Reason: "SECOND"},
		{Seq: 3, TSUnixNano: 300, Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "THIRD"},
		{Seq: 4, TSUnixNano: 400, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "FOURTH"},
	}
	meta := SessionMeta{TraceID: "order-test", PolicyDigest: "p1"}
	rec, examples := Fold(meta, rows)

	if rec.StartedUnixNano != 100 || rec.EndedUnixNano != 400 {
		t.Fatalf("unexpected timestamps: [%d, %d]", rec.StartedUnixNano, rec.EndedUnixNano)
	}
	if len(examples) != 4 {
		t.Fatalf("expected 4 examples, got %d", len(examples))
	}
	expectedReasons := []string{"FIRST", "SECOND", "THIRD", "FOURTH"}
	for i, ex := range examples {
		if ex.Reason != expectedReasons[i] {
			t.Errorf("example %d reason = %q, want %q", i, ex.Reason, expectedReasons[i])
		}
	}
}

func TestNormalizeVerdictInvariants(t *testing.T) {
	cases := []struct {
		verdict string
		kind    string
		want    string
	}{
		{"allow", "", "ALLOW"},
		{" DENY ", "", "DENY"},
		{"", "deny", "DENY"},
		{"", "result_deny", "DENY"},
		{"", "quarantine", "QUARANTINE"},
		{"", "decide", ""},
		{"", "tool_definition_pruned", ""},
		{"", "", ""},
	}
	for _, tc := range cases {
		got := normalizeVerdict(tc.verdict, tc.kind)
		if got != tc.want {
			t.Errorf("normalizeVerdict(%q, %q) = %q, want %q", tc.verdict, tc.kind, got, tc.want)
		}
	}
}

func TestRateLimitClassInvariants(t *testing.T) {
	cases := []struct {
		row  journal.Row
		want string
	}{
		{journal.Row{Reason: "session_limit exceeded"}, "session_limit"},
		{journal.Row{Witness: "Weekly_Limit reached"}, "weekly_limit"},
		{journal.Row{ArgsLabel: "Usage_Limit hit"}, "usage_limit"},
		{journal.Row{Reason: "worker rate_limited by upstream"}, "rate_limited"},
		{journal.Row{Witness: "hit rate limit"}, "rate_limited"},
		{journal.Row{ArgsLabel: "ratelimit backoff"}, "rate_limited"},
		{journal.Row{Reason: "rate_limit_exit detected"}, "rate_limited"},
		{journal.Row{Reason: "SIGSEGV", Witness: "segfault", ArgsLabel: "proc"}, ""},
	}
	for i, tc := range cases {
		got := rateLimitClass(tc.row)
		if got != tc.want {
			t.Errorf("case %d: rateLimitClass(%+v) = %q, want %q", i, tc.row, got, tc.want)
		}
	}
}

func TestExampleFieldPreservation(t *testing.T) {
	meta := SessionMeta{TraceID: "t-preserve", PolicyDigest: "d-preserve"}
	r := journal.Row{
		Tool:      "WebFetch",
		ArgsLabel: "url=https://example.com",
		Reason:    "UNSAFE_EGRESS",
		By:        "egressgate",
		Taint:     "taint:external",
		Witness:   "blocked external host",
	}
	ex := exampleFrom(meta, r, "DENY", "DENY")
	if ex.Schema != ExampleSchema || ex.TraceID != meta.TraceID || ex.PolicyDigest != meta.PolicyDigest {
		t.Fatalf("metadata not preserved: %+v", ex)
	}
	if ex.Tool != r.Tool || ex.ArgsLabel != r.ArgsLabel || ex.Reason != r.Reason ||
		ex.By != r.By || ex.Taint != r.Taint || ex.Witness != r.Witness ||
		ex.Kind != "DENY" || ex.Verdict != "DENY" {
		t.Fatalf("journal fields not preserved verbatim: %+v", ex)
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
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

func testIsSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
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

func testReferencesName(expr ast.Expr, name string) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return true
	})
	return found
}

func testIsSubstantiveBenchmark(fn *ast.FuncDecl) bool {
	if fn.Body == nil || len(fn.Body.List) == 0 {
		return false
	}
	hasLoopOrRun := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if hasLoopOrRun {
			return false
		}
		switch stmt := n.(type) {
		case *ast.ForStmt:
			if stmt.Cond != nil && testReferencesName(stmt.Cond, "N") {
				hasLoopOrRun = true
			}
		case *ast.RangeStmt:
			if testReferencesName(stmt.X, "N") {
				hasLoopOrRun = true
			}
		case *ast.CallExpr:
			if sel, ok := stmt.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Run" {
				hasLoopOrRun = true
			}
		}
		return true
	})
	return hasLoopOrRun
}

// TestCommentHygieneAndNoFormulaicNoise verifies that guardcorpus enforces clean,
// self-documenting code without formulaic comment noise, excessive comment bloat,
// or missing exported symbol documentation.
func TestCommentHygieneAndNoFormulaicNoise(t *testing.T) {
	fset := token.NewFileSet()
	files := []string{"guardcorpus.go"}

	for _, filename := range files {
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
					if testIsSubstantiveDoc(d.Name.Name, d.Doc) {
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
							if testIsSubstantiveDoc(s.Name.Name, doc) {
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
								if testIsSubstantiveDoc(name.Name, doc) {
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

	benchPath := "bench_test.go"
	benchContent, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", benchPath, err)
	}
	benchNode, err := parser.ParseFile(fset, benchPath, benchContent, 0)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", benchPath, err)
	}

	hasScanBenchmark := false
	for _, decl := range benchNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "BenchmarkFoldPlanted" {
			if testIsSubstantiveBenchmark(fn) {
				hasScanBenchmark = true
			}
		}
	}
	if !hasScanBenchmark {
		t.Errorf("bench_test.go must define a substantive BenchmarkFoldPlanted with a b.N loop")
	}
}
