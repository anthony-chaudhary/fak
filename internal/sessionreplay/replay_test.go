package sessionreplay

import (
	"bufio"
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

var (
	benchVerdictSink Verdict
	benchFixtureSink Fixture
	benchBytesSink   []byte
)

const goldenFixture = "regime_conditioned_turn.json"

func TestReplayRegressionFreezesRegimeVerdict(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}
	if f.ActiveRegime != "plan" {
		t.Fatalf("golden active_regime = %q, want plan", f.ActiveRegime)
	}
	frozen := Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"}
	if !f.Expect.Equal(frozen) {
		t.Fatalf("golden Expect = %s, want frozen %s", f.Expect, frozen)
	}

	got, err := Replay(f)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if !got.Equal(f.Expect) {
		t.Fatalf("replay verdict = %s, want frozen %s", got, f.Expect)
	}
}

func TestReplayIsRegimeConditioned(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}

	planVerdict, err := Replay(f)
	if err != nil {
		t.Fatalf("replay under plan: %v", err)
	}

	f.ActiveRegime = "autonomous"
	autoVerdict, err := Replay(f)
	if err != nil {
		t.Fatalf("replay under autonomous: %v", err)
	}

	if autoVerdict.Equal(planVerdict) {
		t.Fatalf("verdict did not change with regime: plan=%s autonomous=%s", planVerdict, autoVerdict)
	}
	if autoVerdict.Kind != "ALLOW" {
		t.Fatalf("autonomous regime verdict = %s, want ALLOW", autoVerdict)
	}
}

func TestFixtureRoundTripsAndCaptures(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}
	b, err := f.Marshal()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseFixture(b)
	if err != nil {
		t.Fatalf("re-parse: %v", err)
	}
	if got.Schema != SchemaV1 || got.Turn.Tool != f.Turn.Tool ||
		got.ActiveRegime != f.ActiveRegime || !got.Expect.Equal(f.Expect) {
		t.Fatalf("round-trip drifted:\n got %+v\nwant %+v", got, f)
	}

	captured := Capture(f.Turn, f.ActiveRegime, f.Expect)
	if captured.Schema != SchemaV1 || captured.ActiveRegime != f.ActiveRegime || !captured.Expect.Equal(f.Expect) {
		t.Fatalf("Capture built an unexpected fixture: %+v", captured)
	}
}

func TestReplayRefusesUnknownRegime(t *testing.T) {
	f := Capture(
		DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/report.txt"}`)},
		"no-such-regime",
		Verdict{Kind: "DENY"},
	)
	if _, err := Replay(f); err == nil {
		t.Fatal("replay under an unknown regime returned nil error, want a refusal")
	}
}

func TestInvariants_SchemaAndValidation(t *testing.T) {
	f := Capture(DecisionInputs{Tool: "Read"}, "plan", Verdict{Kind: "ALLOW"})
	f.Schema = "fak.sessionreplay.invalid"
	if _, err := Replay(f); err == nil {
		t.Fatal("expected error for unsupported schema, got nil")
	}

	for _, badTool := range []string{"", "   ", "\t\n"} {
		fBadTool := Capture(DecisionInputs{Tool: badTool}, "plan", Verdict{Kind: "ALLOW"})
		if _, err := Replay(fBadTool); err == nil {
			t.Errorf("expected error for empty tool %q, got nil", badTool)
		}
	}

	fNilArgs := Capture(DecisionInputs{Tool: "Read", Args: nil}, "plan", Verdict{Kind: "ALLOW"})
	v, err := Replay(fNilArgs)
	if err != nil {
		t.Fatalf("Replay failed with empty args: %v", err)
	}
	if v.Kind != "ALLOW" {
		t.Errorf("expected ALLOW for Read in plan regime, got %s", v)
	}

	badJSON := []byte(`{"schema":"fak.sessionreplay.v1","turn":{"tool":"Read"},"active_regime":"plan","expect":{"kind":"ALLOW"},"unexpected_extra":"val"}`)
	if _, err := ParseFixture(badJSON); err == nil {
		t.Fatal("expected ParseFixture to reject unknown fields, got nil error")
	}

	mismatchSchema := []byte(`{"schema":"fak.sessionreplay.v99","turn":{"tool":"Read"},"active_regime":"plan","expect":{"kind":"ALLOW"}}`)
	if _, err := ParseFixture(mismatchSchema); err == nil {
		t.Fatal("expected ParseFixture to reject mismatched schema, got nil error")
	}

	if _, err := ParseFixture([]byte(`{not valid json`)); err == nil {
		t.Fatal("expected ParseFixture to reject invalid json syntax, got nil error")
	}

	if _, err := LoadFixture("testdata/non_existent_file.json"); err == nil {
		t.Fatal("expected LoadFixture to fail on missing file, got nil error")
	}
}

func TestInvariants_RegimeManifestResolution(t *testing.T) {
	for _, bad := range []string{"", "   ", "\t"} {
		if _, err := regimeManifest(bad); err == nil {
			t.Errorf("expected error for empty regime %q, got nil", bad)
		}
	}

	for _, name := range policy.PresetNames() {
		b, err := regimeManifest(name)
		if err != nil {
			t.Errorf("expected preset %q to resolve, got error: %v", name, err)
		}
		if len(b) == 0 {
			t.Errorf("expected non-empty manifest for preset %q", name)
		}
	}

	for alias, expectedPreset := range regimeToPreset {
		b, err := regimeManifest(alias)
		if err != nil {
			t.Errorf("expected alias %q to resolve, got error: %v", alias, err)
		}
		expectedBytes, err := policy.PresetManifest(expectedPreset)
		if err != nil {
			t.Fatalf("failed to get manifest for preset %q: %v", expectedPreset, err)
		}
		if !bytes.Equal(b, expectedBytes) {
			t.Errorf("alias %q manifest bytes did not match preset %q", alias, expectedPreset)
		}
	}

	if _, err := regimeManifest("non-existent-regime-name"); err == nil {
		t.Fatal("expected error for unknown regime, got nil")
	}
}

func TestInvariants_VerdictProjection(t *testing.T) {
	kinds := []struct {
		kind abi.VerdictKind
		want string
	}{
		{abi.VerdictAllow, "ALLOW"},
		{abi.VerdictDeny, "DENY"},
		{abi.VerdictTransform, "TRANSFORM"},
		{abi.VerdictQuarantine, "QUARANTINE"},
		{abi.VerdictRequireWitness, "REQUIRE_WITNESS"},
		{abi.VerdictDefer, "DEFER"},
		{abi.VerdictIndeterminate, "INDETERMINATE"},
		{abi.VerdictKind(88), "KIND_88"},
	}
	for _, tc := range kinds {
		got := verdictKindName(tc.kind)
		if got != tc.want {
			t.Errorf("verdictKindName(%d) = %q, want %q", tc.kind, got, tc.want)
		}
	}

	allowVerdict := abi.Verdict{Kind: abi.VerdictAllow, Reason: abi.ReasonNone}
	pAllow := projectVerdict(allowVerdict)
	if pAllow.Kind != "ALLOW" || pAllow.Reason != "" {
		t.Errorf("unexpected projectVerdict for allow: %+v", pAllow)
	}

	denyVerdict := abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}
	pDeny := projectVerdict(denyVerdict)
	if pDeny.Kind != "DENY" || pDeny.Reason != "POLICY_BLOCK" {
		t.Errorf("unexpected projectVerdict for deny: %+v", pDeny)
	}

	v1 := Verdict{Kind: "ALLOW"}
	if v1.String() != "ALLOW" {
		t.Errorf("expected 'ALLOW', got %q", v1.String())
	}
	v2 := Verdict{Kind: "DENY", Reason: "POLICY_BLOCK"}
	if v2.String() != "DENY/POLICY_BLOCK" {
		t.Errorf("expected 'DENY/POLICY_BLOCK', got %q", v2.String())
	}

	if !v1.Equal(Verdict{Kind: "ALLOW"}) {
		t.Errorf("expected v1.Equal to be true for identical verdict")
	}
	if v1.Equal(v2) {
		t.Errorf("expected v1.Equal(v2) to be false")
	}
	if v2.Equal(Verdict{Kind: "DENY", Reason: "DIFFERENT_REASON"}) {
		t.Errorf("expected v2.Equal to be false for differing reason")
	}
}

func TestCommentHygieneAndNoFormulaicNoise(t *testing.T) {
	fset := token.NewFileSet()
	files := []string{"fixture.go", "replay.go"}

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

		exportedCount := 0
		documentedCount := 0
		var undocumented []string

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					exportedCount++
					if isSubstantiveDoc(d.Name.Name, d.Doc) {
						documentedCount++
					} else {
						undocumented = append(undocumented, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							exportedCount++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if isSubstantiveDoc(s.Name.Name, doc) {
								documentedCount++
							} else {
								undocumented = append(undocumented, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								exportedCount++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if isSubstantiveDoc(name.Name, doc) {
									documentedCount++
								} else {
									undocumented = append(undocumented, name.Name)
								}
							}
						}
					}
				}
			}
		}

		if exportedCount > 0 {
			ratio := float64(documentedCount) / float64(exportedCount)
			if ratio < 0.90 {
				t.Errorf("%s: documented exports ratio %.2f < 0.90 (undocumented: %v)", filename, ratio, undocumented)
			}
		}
	}
}

func countNonEmptyLines(b []byte) int {
	scanner := bufio.NewScanner(strings.NewReader(string(b)))
	lines := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			lines++
		}
	}
	return lines
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

func isSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil || len(doc.List) == 0 {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if len(text) < 12 {
		return false
	}
	return !isTautologicalDoc(name, text)
}

func splitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func isTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := splitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}

func BenchmarkReplay_Plan(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

func BenchmarkReplay_Autonomous(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}
	f.ActiveRegime = "autonomous"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

func BenchmarkReplay_DiverseTurns(b *testing.B) {
	fixtures := []Fixture{
		Capture(
			DecisionInputs{Tool: "Read", Args: []byte(`{"path":"workspace/notes.txt"}`)},
			"plan",
			Verdict{Kind: "ALLOW"},
		),
		Capture(
			DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/notes.txt","content":"hello"}`)},
			"plan",
			Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"},
		),
		Capture(
			DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/notes.txt","content":"hello"}`)},
			"autonomous",
			Verdict{Kind: "ALLOW"},
		),
		Capture(
			DecisionInputs{Tool: "bash", Args: []byte(`{"command":"ls -la"}`)},
			"plan",
			Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"},
		),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i%len(fixtures)]
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

func BenchmarkParseFixture(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("read golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f, err := ParseFixture(raw)
		if err != nil {
			b.Fatalf("parse fixture: %v", err)
		}
		benchFixtureSink = f
	}
}

func BenchmarkMarshalFixture(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := f.Marshal()
		if err != nil {
			b.Fatalf("marshal fixture: %v", err)
		}
		benchBytesSink = data
	}
}

func BenchmarkCapture(b *testing.B) {
	turn := DecisionInputs{
		Tool: "Write",
		Args: []byte(`{"path":"workspace/report.txt","content":"shipped"}`),
	}
	expect := Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := Capture(turn, "plan", expect)
		benchFixtureSink = f
	}
}
