package hooks

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func conceptAdmissionFixture(t *testing.T, line string, treatment string, headHas bool) *StagedDiff {
	t.Helper()
	return conceptAdmissionFixtureRoots(t, line, treatment, headHas, "cache")
}

// conceptAdmissionFixtureRoots is conceptAdmissionFixture with the family's roots chosen by
// the caller, so a case can exercise a root other than the default "cache" one.
func conceptAdmissionFixtureRoots(t *testing.T, line string, treatment string, headHas bool, roots ...string) *StagedDiff {
	t.Helper()
	root := t.TempDir()
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	if err := os.MkdirAll(data, 0755); err != nil {
		t.Fatal(err)
	}
	meta := map[string]any{"families": []map[string]any{{"id": "cache", "roots": roots, "ignore": []string{}}}}
	if treatment == "classify" {
		meta["families"].([]map[string]any)[0]["ignore"] = []string{"CacheBurst"}
	}
	mb, _ := json.Marshal(meta)
	os.WriteFile(filepath.Join(data, "_meta.json"), mb, 0600)
	rows := []map[string]any{{"id": "cache-a", "family": "cache", "grounding": "CacheA"}}
	if treatment == "position" {
		rows = append(rows, map[string]any{"id": "cache-burst", "family": "cache", "grounding": "CacheBurst"})
	}
	rb, _ := json.Marshal(map[string]any{"rows": rows})
	os.WriteFile(filepath.Join(data, "rows-cache.json"), rb, 0600)
	d := diffOf(root, map[string][]string{"internal/demo/demo.go": {line}})
	d.ctx = context.Background()
	d.run = func(context.Context, string, ...string) (string, int, error) {
		if headHas {
			return "HEAD:internal/demo/demo.go:CacheBurst", 0, nil
		}
		return "", 1, nil
	}
	return d
}

func TestConceptAdmissionRejectsUntreatedTokenWithRepair(t *testing.T) {
	d := conceptAdmissionFixture(t, "const CacheBurst = 1", "", false)
	start := time.Now()
	got, err := gateConceptAdmission(d)
	elapsed := time.Since(start)
	if err != nil || len(got) != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	detail := got[0].Detail
	for _, want := range []string{"family=cache", "token=CacheBurst", "internal/demo/demo.go:1", "fak concept position", "fak concept classify"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("diagnostic missing %q: %s", want, detail)
		}
	}
	if elapsed > 100*time.Millisecond {
		t.Fatalf("diff-scoped gate latency %s exceeds 100ms fixture budget", elapsed)
	}
}

// The suggested repair must be lane-local: exactly one per-leaf rows shard,
// never the shared glossary or scorecard files a peer may hold mid-flight (#5104).
func TestConceptAdmissionRemedyTouchesExactlyOneFile(t *testing.T) {
	d := conceptAdmissionFixture(t, "const CacheBurst = 1", "", false)
	got, err := gateConceptAdmission(d)
	if err != nil || len(got) != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	detail := got[0].Detail
	start := strings.Index(detail, "`fak concept position")
	if start < 0 {
		t.Fatalf("no position remedy in detail: %s", detail)
	}
	end := strings.Index(detail[start+1:], "`")
	if end < 0 {
		t.Fatalf("unterminated remedy command in detail: %s", detail)
	}
	cmd := detail[start+1 : start+1+end]
	var files []string
	for _, tok := range strings.Fields(cmd) {
		if strings.Contains(tok, "/") || strings.HasSuffix(tok, ".json") || strings.HasSuffix(tok, ".md") {
			files = append(files, tok)
		}
	}
	if len(files) != 1 || files[0] != "rows-demo.json" {
		t.Fatalf("remedy must name exactly the per-leaf rows shard, got %v in %q", files, cmd)
	}
	if strings.Contains(cmd, "--glossary") {
		t.Fatalf("remedy still plans a shared glossary write: %q", cmd)
	}
}

func TestConceptAdmissionTreatmentsPass(t *testing.T) {
	for _, treatment := range []string{"position", "classify"} {
		t.Run(treatment, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixture(t, "const CacheBurst = 1", treatment, false))
			if err != nil || len(got) != 0 {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
	}
}

func TestConceptAdmissionRenameOrExistingUseIsNotAddition(t *testing.T) {
	got, err := gateConceptAdmission(conceptAdmissionFixture(t, "const CacheBurst = 1", "", true))
	if err != nil || len(got) != 0 {
		t.Fatalf("existing/renamed token falsely admitted: %+v %v", got, err)
	}
}

// conceptFindingToken pulls the token= value back out of an admission diagnostic so a
// test can assert on the exact identifier the gate would make the author classify.
func conceptFindingToken(detail string) string {
	_, rest, ok := strings.Cut(detail, "token=")
	if !ok {
		return ""
	}
	tok, _, _ := strings.Cut(rest, " ")
	return tok
}

// A Go escape is one character of the compiled string, so its letter belongs to the escape
// and never to the identifier that follows: "DRIFT\tSESSION" holds SESSION, not tSESSION.
// Tokenizing the raw source text glued the two together and blocked the commit on a symbol
// that exists nowhere in the tree (#5529).
func TestConceptAdmissionDoesNotGlueStringEscapeOntoNextWord(t *testing.T) {
	for _, tc := range []struct {
		name  string
		line  string
		glued string // the invented token the extractor used to report
		real  string // the identifier that is actually in the literal
	}{
		{"tab", `	fmt.Fprintln(stdout, "DRIFT\tAGE_H\tSESSION\tLEAVES")`, "tSESSION", "SESSION"},
		{"newline", `	fmt.Fprintf(stdout, "header\nSlotBudget\n")`, "nSlotBudget", "SlotBudget"},
		{"carriage return", `	fmt.Fprintf(stdout, "header\rSlotLease")`, "rSlotLease", "SlotLease"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "session", "slot"))
			if err != nil {
				t.Fatal(err)
			}
			var tokens []string
			for _, f := range got {
				tok := conceptFindingToken(f.Detail)
				tokens = append(tokens, tok)
				// A token that reappears in the source only with a backslash welded to its
				// front was manufactured by the extractor, not written by the author.
				if strings.Contains(tc.line, `\`+tok) {
					t.Errorf("token %q absorbed the Go escape ahead of it in %s", tok, tc.line)
				}
			}
			for _, tok := range tokens {
				if tok == tc.glued {
					t.Errorf("gate reports invented token %q", tc.glued)
				}
			}
			// The identifier inside the literal must still be discovered: dropping string
			// literals wholesale would silence the artifact and the real symbol together.
			found := false
			for _, tok := range tokens {
				if tok == tc.real {
					found = true
				}
			}
			if !found {
				t.Errorf("real identifier %q no longer discovered; tokens=%v", tc.real, tokens)
			}
		})
	}
}

// Backslashes in comments and raw strings are ordinary characters, not escapes, so the
// words around them must survive intact -- blanking a would-be escape there would eat the
// leading letter of a real identifier (Users -> sers) and lose it from the corpus.
func TestConceptAdmissionKeepsLiteralBackslashWordsInCommentsAndRawStrings(t *testing.T) {
	for _, tc := range []struct{ name, line string }{
		{"comment", `	// checkpoints land under C:\Users\Public\SessionSlot on Windows`},
		{"raw string", "	root := `C:\\Users\\Public\\SessionSlot`"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "session", "user"))
			if err != nil {
				t.Fatal(err)
			}
			var tokens []string
			for _, f := range got {
				tokens = append(tokens, conceptFindingToken(f.Detail))
			}
			for _, want := range []string{"Users", "SessionSlot"} {
				found := false
				for _, tok := range tokens {
					if tok == want {
						found = true
					}
				}
				if !found {
					t.Errorf("word %q around a literal backslash was lost; tokens=%v", want, tokens)
				}
			}
		})
	}
}

func TestConceptAdmissionRegisteredEarliestCommitGate(t *testing.T) {
	for _, g := range PreCommitGates() {
		if g.Name == "CONCEPT_ADMISSION" {
			return
		}
	}
	t.Fatal("CONCEPT_ADMISSION is not registered in pre-commit")
}

func gitFixture(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func TestConceptAdmissionCommittedRangeRepeatsCandidateDiff(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init")
	gitFixture(t, root, "config", "user.email", "fixture@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	os.MkdirAll(data, 0755)
	os.WriteFile(filepath.Join(data, "_meta.json"), []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`), 0600)
	os.WriteFile(filepath.Join(data, "rows-cache.json"), []byte(`{"rows":[{"id":"cache-a","family":"cache","grounding":"CacheA"}]}`), 0600)
	os.MkdirAll(filepath.Join(root, "internal", "demo"), 0755)
	os.WriteFile(filepath.Join(root, "internal", "demo", "demo.go"), []byte("package demo\nconst CacheA=1\n"), 0600)
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "base")
	base := gitFixture(t, root, "rev-parse", "HEAD")
	os.WriteFile(filepath.Join(root, "internal", "demo", "demo.go"), []byte("package demo\nconst CacheA=1\nconst CacheBurst=2\n"), 0600)
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "candidate")
	tip := gitFixture(t, root, "rev-parse", "HEAD")
	d, err := ReadRangeDiff(root, base, tip)
	if err != nil {
		t.Fatal(err)
	}
	got, err := CheckConceptAdmission(d)
	if err != nil || len(got) != 1 {
		t.Fatalf("range gate got=%+v err=%v", got, err)
	}
}
