package hooks

import (
	"strings"
	"testing"
)

// diffOf builds a StagedDiff in-package for a gate unit test: each (file, line) becomes an added
// line numbered from 1. Paths populate all three name lists so any gate sees the file.
func diffOf(root string, lines map[string][]string) *StagedDiff {
	d := &StagedDiff{Root: root, AddedByFile: map[string][]AddedLine{}, fileCache: map[string]fileEntry{}}
	for f, ls := range lines {
		for i, t := range ls {
			d.AddedByFile[f] = append(d.AddedByFile[f], AddedLine{File: f, New: i + 1, Text: t})
		}
		d.StagedPaths = append(d.StagedPaths, f)
		d.AddedPaths = append(d.AddedPaths, f)
		d.AddedRenamedPaths = append(d.AddedRenamedPaths, f)
	}
	return d
}

func hasFindingFor(fs []Finding, gate, substr string) bool {
	for _, f := range fs {
		if f.Gate == gate && strings.Contains(f.Detail, substr) {
			return true
		}
	}
	return false
}

func leakIPFixture() string       { return "100" + ".64.0.10" }
func gcpSAFixture() string        { return "svc@proj." + "iam." + "gserviceaccount.com" }
func mslHostFixture() string      { return "msl" + "-build-01" }
func labHostFixture() string      { return "secret" + ".lab" }
func operatorNameFixture() string { return "anth" + "ony" }
func userPathFixture(suffix string) string {
	return `C:\Users\` + operatorNameFixture() + suffix
}

func TestPublicLeak_needleAndRegex(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"docs/a.md": {
			"the node is at " + leakIPFixture() + " today",
			`path ` + userPathFixture(`\go`) + ` is mine`,
			"contact " + gcpSAFixture(),
			"a perfectly clean line",
		},
	})
	f, err := gatePublicLeak(d)
	if err != nil {
		t.Fatal(err)
	}
	if !hasFindingFor(f, "PUBLIC_LEAK", leakIPFixture()) {
		t.Errorf("missed the IP needle; got %+v", f)
	}
	if !hasFindingFor(f, "PUBLIC_LEAK", "GCP service-account email") {
		t.Errorf("missed the GCP regex; got %+v", f)
	}
}

func TestPublicLeak_backslashUsersNeedle(t *testing.T) {
	// The Windows-user needle (case-insensitive substring) must match a Windows path.
	d := diffOf("/r", map[string][]string{"x.txt": {userPathFixture(`\dev`)}})
	f, _ := gatePublicLeak(d)
	if len(f) == 0 {
		t.Fatalf("expected a leak for a Windows user path; got none")
	}
}

func TestPublicLeak_selfReferentialFileExempt(t *testing.T) {
	d := diffOf("/r", map[string][]string{"tools/scrub_public_copy.py": {leakIPFixture() + " in the policy doc"}})
	f, _ := gatePublicLeak(d)
	if len(f) != 0 {
		t.Fatalf("self-referential file must be exempt; got %+v", f)
	}
}

func TestSecretShape_operatorPathButNotPlaceholder(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"a.md": {
			`see ` + userPathFixture(`\go`), // real username -> hit
			`see C:\Users\runner\go`,        // placeholder -> no hit
			"host " + mslHostFixture() + " is internal",
			"box at gpu.example.lab", // example.lab carve-out -> no hit
			"box at gpu." + labHostFixture(),
		},
	})
	f, _ := gateSecretShape(d)
	if !hasFindingFor(f, "SECRET_SHAPE", operatorNameFixture()) {
		t.Errorf("missed real operator path; got %+v", f)
	}
	if hasFindingFor(f, "SECRET_SHAPE", "runner") {
		t.Errorf("placeholder user must not be flagged; got %+v", f)
	}
	if !hasFindingFor(f, "SECRET_SHAPE", mslHostFixture()) {
		t.Errorf("missed msl host; got %+v", f)
	}
	if hasFindingFor(f, "SECRET_SHAPE", "example.lab") {
		t.Errorf("example.lab must be carved out; got %+v", f)
	}
	if !hasFindingFor(f, "SECRET_SHAPE", labHostFixture()) {
		t.Errorf("missed .lab host; got %+v", f)
	}
}

func TestSecretShape_onlyTextExtensions(t *testing.T) {
	// A binary-ish extension is not scanned.
	d := diffOf("/r", map[string][]string{"img.png": {userPathFixture(`\x`)}})
	f, _ := gateSecretShape(d)
	if len(f) != 0 {
		t.Fatalf(".png must not be scanned; got %+v", f)
	}
}

func TestFileAdmission_precedenceAndJunk(t *testing.T) {
	d := &StagedDiff{Root: "/r", AddedByFile: map[string][]AddedLine{}, fileCache: map[string]fileEntry{}}
	d.AddedRenamedPaths = []string{
		"secrets/db.txt",          // SECRET_FILES
		"cmd/dgxbox/main.go",      // PRIVATE_ONLY
		"build/__pycache__/x.pyc", // HARD_JUNK
		"foo.log",                 // SOFT_JUNK (not exempt)
		"internal/x.log",          // SOFT_JUNK but EXEMPT_DATA_DIRS -> clean
		"src/main.go",             // clean
	}
	f, _ := gateFileAdmission(d)
	if !hasFindingFor(f, "FILE_ADMISSION", "secrets dir") {
		t.Errorf("missed secrets/; got %+v", f)
	}
	if !hasFindingFor(f, "FILE_ADMISSION", "private lab GPU-server") {
		t.Errorf("missed dgx; got %+v", f)
	}
	if !hasFindingFor(f, "FILE_ADMISSION", "build artifact") {
		t.Errorf("missed __pycache__; got %+v", f)
	}
	if !hasFindingFor(f, "FILE_ADMISSION", "log / temp") {
		t.Errorf("missed foo.log; got %+v", f)
	}
	for _, bad := range f {
		if bad.File == "internal/x.log" {
			t.Errorf("internal/ .log is exempt; should be clean")
		}
		if bad.File == "src/main.go" {
			t.Errorf("a normal .go file must be clean")
		}
	}
}

func TestDocPlacement_rootMD(t *testing.T) {
	d := diffOf("/r", map[string][]string{})
	d.StagedPaths = []string{"RANDOM-NOTE-2026-06-25.md", "README.md", "docs/x.md"}
	f, _ := gateDocPlacement(d)
	if !hasFindingFor(f, "DOC_PLACEMENT", "RANDOM-NOTE-2026-06-25.md") {
		t.Errorf("a non-allowlisted root .md must be flagged; got %+v", f)
	}
	for _, bad := range f {
		if strings.Contains(bad.Detail, "README.md") {
			t.Errorf("README.md is allowlisted")
		}
		if strings.Contains(bad.Detail, "docs/x.md") {
			t.Errorf("a non-root .md is fine")
		}
	}
}

func TestProvenance_measuredModeledBlockedButCarveoutsPass(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"README.md": {
			"WebVoyager 643-task measured 9.7x speedup",     // ctx+num+measured -> VIOLATION
			"WebVoyager 643-task modeled 9.7x speedup",      // 'modeled' carve-out -> ok
			"a measured 4.1x end-to-end real run",           // 'measured 4.1' carve-out -> ok
			"just a measured number with no modeled family", // measured but no ctx/num -> ok
		},
	})
	f, _ := gateProvenanceLabel(d)
	if len(f) != 1 {
		t.Fatalf("exactly one provenance violation expected, got %d: %+v", len(f), f)
	}
	if !strings.Contains(f[0].Detail, "modeled") { // the fix text mentions modeled
		t.Errorf("violation should carry the fix; got %+v", f[0])
	}
}

func TestCommitMsgVerdict(t *testing.T) {
	cases := []struct {
		msg string
		ok  bool
	}{
		{"feat(safecommit): add the hooks verb", true},
		{"fix: correct the parser", true},
		{"docs: clean up the readme", false},     // 'clean' is not a verb
		{"feat(x): thing without a verb", false}, // 'thing' not a verb
		{"random subject no type", false},
		{"Merge branch 'main'", true}, // exempt prefix
		{"chore: bump deps", true},
		{"unknowntype: add a thing", false}, // not a known type
	}
	for _, c := range cases {
		ok, why := CommitMsgVerdict(c.msg)
		if ok != c.ok {
			t.Errorf("CommitMsgVerdict(%q) = %v (%q), want %v", c.msg, ok, why, c.ok)
		}
	}
}

func TestScanMessageNeedles_skipsTrailersCommentsScissors(t *testing.T) {
	needle := privateAddressNeedle() // a hardcoded AUDIT_NEEDLE
	// A needle inside a DCO/identity trailer is metadata, not a leak -> exempt (this is
	// the fix that lets `git commit -s` with an org-domain sign-off survive the gate).
	if f := ScanMessageNeedles("fix: x\n\nSigned-off-by: A B <a@"+needle+">\n", ""); len(f) != 0 {
		t.Errorf("identity trailer must be exempt; got %+v", f)
	}
	// A needle in a comment line git strips from the final message -> exempt.
	if f := ScanMessageNeedles("fix: x\n\n# note "+needle+"\n", ""); len(f) != 0 {
		t.Errorf("comment line must be exempt; got %+v", f)
	}
	// A needle below git's scissors line -> exempt (the content gate owns that preview).
	scissors := "# ------------------------ >8 ------------------------"
	if f := ScanMessageNeedles("fix: x\n"+scissors+"\n"+needle+"\n", ""); len(f) != 0 {
		t.Errorf("scissors block must be exempt; got %+v", f)
	}
	// A needle in the real prose body IS a leak -> still flagged (no weakening).
	if f := ScanMessageNeedles("fix: x\n\nbody has "+needle+" leak\n", ""); len(f) == 0 {
		t.Error("a needle in the prose body must be flagged")
	}
}

func TestParseUnifiedAddedLines_lineNumbers(t *testing.T) {
	diff := "" +
		"diff --git a/x.md b/x.md\n" +
		"--- a/x.md\n" +
		"+++ b/x.md\n" +
		"@@ -0,0 +1,2 @@\n" +
		"+first line\n" +
		"+second line\n"
	got := parseUnifiedAddedLines(diff)
	lines := got["x.md"]
	if len(lines) != 2 || lines[0].New != 1 || lines[1].New != 2 {
		t.Fatalf("hunk line numbers wrong: %+v", lines)
	}
	if lines[0].Text != "first line" {
		t.Errorf("text wrong: %q", lines[0].Text)
	}
}
