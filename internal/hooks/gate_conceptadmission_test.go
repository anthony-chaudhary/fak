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

// conceptFindingTokens lists the identifiers a run of the gate would make the author
// position or classify, in report order.
func conceptFindingTokens(fs []Finding) []string {
	var tokens []string
	for _, f := range fs {
		tokens = append(tokens, conceptFindingToken(f.Detail))
	}
	return tokens
}

func conceptTokensContain(tokens []string, want string) bool {
	for _, tok := range tokens {
		if tok == want {
			return true
		}
	}
	return false
}

// A file name written in prose or in a literal is a path, not a symbol. The extractor used
// to take the stem, drop the extension at the following dot, and offer the remainder as an
// identifier the file introduces -- blocking the commit on a name that is declared nowhere
// in the tree (#5533). The gate's own wording ("introduced at") is the claim that is false.
func TestConceptAdmissionDoesNotReadAQuotedFileNameAsAnIntroducedSymbol(t *testing.T) {
	for _, tc := range []struct {
		name     string
		line     string
		invented string // the path stem the extractor used to report as a symbol
	}{
		{"test file named in a header comment", `// NOT VERIFIED END TO END: guard_split_close_test.go witnesses the generated command line`, "guard_split_close_test"},
		{"source file named in a header comment", `// the sibling rule lives in guard_relaunch_files.go`, "guard_relaunch_files"},
		{"data file inside a string literal", `	b, _ := os.ReadFile("rows-gatekeeper.json")`, "gatekeeper"},
		{"doc file named in a trailing comment", `	const x = 1 // the shape is written up in docs/notes/preflight-gateway.md`, "gateway"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "guard", "gate"))
			if err != nil {
				t.Fatal(err)
			}
			tokens := conceptFindingTokens(got)
			if conceptTokensContain(tokens, tc.invented) {
				t.Errorf("gate reports path stem %q as an introduced symbol; tokens=%v", tc.invented, tokens)
			}
		})
	}
}

// The path rule keys off the dot and the extension that follows it, so it must not reach a
// declaration that merely reads like a file stem, nor a selector in code position -- `x.json`
// there is a field read, and suppressing its receiver would lose a real symbol.
func TestConceptAdmissionStillReportsSymbolsThatOnlyLookLikeFileNames(t *testing.T) {
	for _, tc := range []struct{ name, line, want string }{
		{"declaration spelled like a stem", `const guardSplitCloseTest = 1`, "guardSplitCloseTest"},
		{"selector in code position", `	raw := gatekeeperCfg.json`, "gatekeeperCfg"},
		{"unknown extension is not an extension", `	const gateBurst = 1 // see gateBurst.notes for the rest`, "gateBurst"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "guard", "gate"))
			if err != nil {
				t.Fatal(err)
			}
			tokens := conceptFindingTokens(got)
			if !conceptTokensContain(tokens, tc.want) {
				t.Errorf("real symbol %q was suppressed by the path rule; tokens=%v", tc.want, tokens)
			}
		})
	}
}

// A test entry point named in prose is a reference into the _test.go files this gate
// deliberately never scans (see the _test.go skip in the scan loop), so the quoting file
// cannot be the file that introduces it. Naming the test that witnesses the code above it
// is good practice and used to cost a permanent corpus row (#5533, the #5411 set).
func TestConceptAdmissionDoesNotReadATestEntryPointNamedInProseAsAnIntroducedSymbol(t *testing.T) {
	for _, tc := range []struct{ name, line, invented string }{
		{"test function in a header comment", `// TestGuardSelfTightenOverlayPathIsAgentWritable witnesses this end to end.`, "TestGuardSelfTightenOverlayPathIsAgentWritable"},
		{"benchmark in a trailing comment", `	const x = 1 // BenchmarkGatewayDecode measures it`, "BenchmarkGatewayDecode"},
		{"test name inside a command literal", `	args := []string{"-run", "TestGuardSplitClosesTheWindow"}`, "TestGuardSplitClosesTheWindow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "guard", "gate"))
			if err != nil {
				t.Fatal(err)
			}
			tokens := conceptFindingTokens(got)
			if conceptTokensContain(tokens, tc.invented) {
				t.Errorf("gate reports referenced test entry point %q as introduced here; tokens=%v", tc.invented, tokens)
			}
		})
	}
}

// The test-entry-point rule is a statement about prose, not about the prefix: a non-test
// file may legally declare a function whose name starts with Test, and a word that merely
// begins with those letters is an ordinary identifier. Both must still be admitted.
func TestConceptAdmissionStillReportsTestPrefixedSymbolsDeclaredInCode(t *testing.T) {
	for _, tc := range []struct{ name, line, want string }{
		{"declared in a non-test file", `func TestGuardHarness(t *testing.T) {`, "TestGuardHarness"},
		{"prefix without the entry-point shape", `// the TestableGuard wrapper is documented here`, "TestableGuard"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixtureRoots(t, tc.line, "", false, "guard", "gate"))
			if err != nil {
				t.Fatal(err)
			}
			tokens := conceptFindingTokens(got)
			if !conceptTokensContain(tokens, tc.want) {
				t.Errorf("real symbol %q was suppressed by the test-entry-point rule; tokens=%v", tc.want, tokens)
			}
		})
	}
}

// The printed remedy has to be sufficient on its own. `fak concept position` / `classify`
// write the worktree corpus, this gate reads the index, and `fak commit --path` stages only
// the paths it is given -- so a remedy that does not name the corpus files leaves the author
// running the prescribed command and being refused identically (#5534).
func TestConceptAdmissionPrintedRemedyStagesItsOwnCorpus(t *testing.T) {
	d := conceptAdmissionFixture(t, "const CacheBurst = 1", "", false)
	findings, err := gateConceptAdmission(d)
	if err != nil || len(findings) != 1 {
		t.Fatalf("got=%+v err=%v", findings, err)
	}
	if !strings.Contains(findings[0].Detail, "fak concept position --stage") ||
		!strings.Contains(findings[0].Detail, "fak concept classify --stage") {
		t.Fatalf("printed remedy must make index-aware corpus visibility explicit: %s", findings[0].Detail)
	}
}

func TestConceptAdmissionRemedyNamesTheCorpusFilesTheCommitMustStage(t *testing.T) {
	got, err := gateConceptAdmission(conceptAdmissionFixture(t, "const CacheBurst = 1", "", false))
	if err != nil || len(got) != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	detail := got[0].Detail
	for _, want := range []string{
		"--path tools/concept_disambiguation_scorecard.data/_meta.json",
		"--path tools/concept_disambiguation_scorecard.data/rows-demo.json",
		"index",
	} {
		if !strings.Contains(detail, want) {
			t.Errorf("remedy does not tell the author to stage the corpus (%q missing): %s", want, detail)
		}
	}
}

// The mechanism behind #5534, pinned: a classification that exists only in the worktree is
// invisible to the gate, and staging the same file clears it. This is deliberate -- reading
// the worktree would let a peer's dirty corpus decide another lane's commit -- so the test
// asserts the refusal STAYS, and only the staged write clears it.
func TestConceptAdmissionReadsTheIndexNotTheWorktreeCorpus(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init")
	gitFixture(t, root, "config", "user.email", "fixture@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	if err := os.MkdirAll(data, 0755); err != nil {
		t.Fatal(err)
	}
	meta := filepath.Join(data, "_meta.json")
	if err := os.WriteFile(meta, []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(data, "rows-cache.json"), []byte(`{"rows":[{"id":"cache-a","family":"cache","grounding":"CacheA"}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	demo := filepath.Join(root, "internal", "demo", "demo.go")
	if err := os.MkdirAll(filepath.Dir(demo), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(demo, []byte("package demo\nconst CacheA = 1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "base")

	// The author stages a new symbol and is refused.
	if err := os.WriteFile(demo, []byte("package demo\nconst CacheA = 1\nconst CacheBurst = 2\n"), 0600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "internal/demo/demo.go")
	admit := func(t *testing.T) []Finding {
		t.Helper()
		d, err := ReadStagedDiff(root)
		if err != nil {
			t.Fatal(err)
		}
		got, err := CheckConceptAdmission(d)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := admit(t); len(got) != 1 {
		t.Fatalf("staged new symbol was not refused: %+v", got)
	}

	// The author runs the prescribed `fak concept classify`, which writes the WORKTREE.
	if err := os.WriteFile(meta, []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":["CacheBurst"]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if got := admit(t); len(got) != 1 {
		t.Fatalf("worktree-only classification cleared the gate; the gate must read the index: %+v", got)
	}

	// Staging the corpus file -- what the remedy now tells the author to do -- clears it.
	gitFixture(t, root, "add", "tools/concept_disambiguation_scorecard.data/_meta.json")
	if got := admit(t); len(got) != 0 {
		t.Fatalf("staged classification did not clear the gate: %+v", got)
	}

	// The other half of the remedy is a per-leaf rows shard that does not exist in HEAD.
	// Same rule, and the same test: unstaged it is invisible, staged it clears.
	if err := os.WriteFile(meta, []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "tools/concept_disambiguation_scorecard.data/_meta.json")
	if got := admit(t); len(got) != 1 {
		t.Fatalf("withdrawing the classification did not restore the finding: %+v", got)
	}
	rows := filepath.Join(data, "rows-demo.json")
	if err := os.WriteFile(rows, []byte(`{"rows":[{"id":"cache-burst","family":"cache","grounding":"CacheBurst"}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if got := admit(t); len(got) != 1 {
		t.Fatalf("unstaged new rows shard cleared the gate; the gate must read the index: %+v", got)
	}
	gitFixture(t, root, "add", "tools/concept_disambiguation_scorecard.data/rows-demo.json")
	if got := admit(t); len(got) != 0 {
		t.Fatalf("staged new rows shard did not clear the gate: %+v", got)
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

func TestConceptAdmissionBatchesIndexRowReads(t *testing.T) {
	meta := `{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`
	rows := `{"rows":[{"family":"cache","grounding":"CacheBurst"}]}`
	calls := map[string]int{}
	d := diffOf(t.TempDir(), map[string][]string{"internal/demo/demo.go": {"const CacheBurst = 1"}})
	d.ctx = context.Background()
	d.Treeish = ":"
	d.IndexPaths = []string{
		"tools/concept_disambiguation_scorecard.data/rows-cache.json",
		"tools/concept_disambiguation_scorecard.data/rows-other.json",
	}
	d.run = func(_ context.Context, _ string, args ...string) (string, int, error) {
		verb := strings.Join(args, " ")
		calls[verb]++
		switch args[0] {
		case "show":
			if args[1] == ":tools/concept_disambiguation_scorecard.data/_meta.json" {
				return meta, 0, nil
			}
		case "grep":
			return "tools/concept_disambiguation_scorecard.data/rows-cache.json\x00" + rows + "\n", 0, nil
		}
		return "", 1, nil
	}

	got, err := gateConceptAdmission(d)
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if calls["grep --cached -z -I -e ^ -- tools/concept_disambiguation_scorecard.data/rows-*.json"] != 1 {
		t.Fatalf("batch grep calls = %v, want exactly one", calls)
	}
	for call := range calls {
		if strings.HasPrefix(call, "show :tools/concept_disambiguation_scorecard.data/rows-") {
			t.Fatalf("row shard used per-file git show despite successful batch: %q", call)
		}
	}
}

func TestRegisteredConceptAdmissionBatchesThroughLandsTreeView(t *testing.T) {
	meta := `{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`
	rows := `{"rows":[{"family":"cache","grounding":"CacheBurst"}]}`
	calls := map[string]int{}
	d := diffOf(t.TempDir(), map[string][]string{"internal/demo/demo.go": {"const CacheBurst = 1"}})
	d.ctx = context.Background()
	d.Treeish = ":"
	d.IndexPaths = []string{"tools/concept_disambiguation_scorecard.data/_meta.json", "tools/concept_disambiguation_scorecard.data/rows-cache.json"}
	d.run = func(_ context.Context, _ string, args ...string) (string, int, error) {
		call := strings.Join(args, " ")
		calls[call]++
		switch {
		case args[0] == "rev-parse":
			return "0123456789012345678901234567890123456789\n", 0, nil
		case args[0] == "ls-files" && len(args) > 1 && args[1] == "--stage":
			return "100644 0123456789012345678901234567890123456789 0\ttools/concept_disambiguation_scorecard.data/_meta.json\n", 0, nil
		case args[0] == "show" && strings.Contains(args[1], "_meta.json"):
			return meta, 0, nil
		case args[0] == "grep":
			return "tools/concept_disambiguation_scorecard.data/rows-cache.json\x00" + rows + "\n", 0, nil
		}
		return "", 1, nil
	}
	var concept Gate
	for _, g := range PreCommitGates() {
		if g.Name == "CONCEPT_ADMISSION" {
			concept = g
			break
		}
	}
	got, err := concept.Check(d)
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if calls["grep --cached -z -I -e ^ -- tools/concept_disambiguation_scorecard.data/rows-*.json"] != 1 {
		t.Fatalf("registered LANDS_TREE gate did not use one index batch: %v", calls)
	}
}

func TestStagedFilesMatchingReconstructsMultilineFiles(t *testing.T) {
	d := &StagedDiff{Root: t.TempDir(), Treeish: ":", ctx: context.Background()}
	d.run = func(_ context.Context, _ string, args ...string) (string, int, error) {
		return "a.json\x00{\n" + "a.json\x00  \"rows\": []\n" + "a.json\x00}\n" + "b.json\x00{}", 0, nil
	}
	got, ok := d.stagedFilesMatching("*.json")
	if !ok {
		t.Fatal("batch read unexpectedly fell back")
	}
	if string(got["a.json"]) != "{\n  \"rows\": []\n}\n" || string(got["b.json"]) != "{}" {
		t.Fatalf("reconstructed files = %#v", got)
	}
}

func TestConceptAdmissionReadPathsDiff(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init")
	gitFixture(t, root, "config", "user.email", "test@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	os.MkdirAll(data, 0755)
	os.WriteFile(filepath.Join(data, "_meta.json"), []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`), 0600)
	os.WriteFile(filepath.Join(data, "rows-cache.json"), []byte(`{"rows":[{"id":"cache-a","family":"cache","grounding":"CacheA"}]}`), 0600)
	os.MkdirAll(filepath.Join(root, "internal", "demo"), 0755)
	demoPath := filepath.Join(root, "internal", "demo", "demo.go")
	os.WriteFile(demoPath, []byte("package demo\nconst CacheA=1\n"), 0600)
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "base")

	os.WriteFile(demoPath, []byte("package demo\nconst CacheA=1\nconst CacheBurst=2\n"), 0600)

	d, err := ReadPathsDiff(root, []string{"internal/demo/demo.go"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := CheckConceptAdmission(d)
	if err != nil || len(got) != 1 {
		t.Fatalf("ReadPathsDiff got=%+v err=%v", got, err)
	}
}
