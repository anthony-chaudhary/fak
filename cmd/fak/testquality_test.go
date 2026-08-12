package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testquality"
)

func TestRunTestQualityMalformedBaselineIsHardError(t *testing.T) {
	root := t.TempDir()
	if out, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	testPath := filepath.Join(root, "sample_test.go")
	if err := os.WriteFile(testPath, []byte("package sample\n\nimport \"testing\"\n\nfunc TestSample(t *testing.T) { t.Fatal(\"sentinel\") }\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", root, "add", "sample_test.go").CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "testquality"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "testquality", "baseline.txt"), []byte("bad row\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runTestQuality(&out, &errOut, []string{"--root", root})
	if code == 0 || !strings.Contains(errOut.String(), "line 1") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

// seedTestQualityRepo makes a one-file git repo whose only tracked test is the
// named source, and returns its root. Tracked, because TrackedTestFiles reads the
// git index — a file merely written to disk is invisible to the scanner.
func seedTestQualityRepo(t *testing.T, name, src string) string {
	t.Helper()
	root := t.TempDir()
	if out, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if err := os.WriteFile(filepath.Join(root, name), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", root, "add", name).CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	return root
}

// testQualityJSON is the consumer's view of the report: only the fields a caller
// would branch on, decoded independently of the producer's struct so the test
// fails if the wire names move.
type testQualityJSON struct {
	Schema       string         `json:"schema"`
	Files        int            `json:"files"`
	Total        int            `json:"total"`
	New          int            `json:"new"`
	CountsByCode map[string]int `json:"counts_by_code"`
	Verdict      string         `json:"verdict"`
	Findings     []struct {
		Code string `json:"code"`
		Func string `json:"func"`
	} `json:"findings"`
}

// #5936. `--json` was a stub: it printed nothing and returned 0 no matter how many
// NEW findings the same run reported on stdout, so every machine consumer of the
// ratchet read a growing tree as a clean one. That is precisely the vacuous-green
// shape this verb exists to find, wearing the checker's own uniform — and it is
// worse than a missing flag, because a flag that is merely absent fails loudly.
func TestRunTestQualityJSONCarriesTheGrowingVerdict(t *testing.T) {
	// No baseline file, so the single candidate is NEW: the case where text mode
	// exits 1, and therefore the case --json must not silently disagree with.
	root := seedTestQualityRepo(t, "sample_test.go",
		"package sample\n\nimport \"testing\"\n\nfunc TestVacuous(t *testing.T) { x := 1; _ = x }\n")

	var out, errOut bytes.Buffer
	code := runTestQuality(&out, &errOut, []string{"--root", root, "--json"})

	var rep testQualityJSON
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("--json printed %q, which does not decode as a report: %v", out.String(), err)
	}
	if code != 1 {
		t.Fatalf("exit %d for a report carrying %d NEW finding(s): --json must reach the same verdict as text mode",
			code, rep.New)
	}
	if rep.Verdict != "growing" || rep.New != 1 || rep.Total != 1 || rep.Files != 1 {
		t.Fatalf("report = %+v; want verdict growing, new 1, total 1, files 1", rep)
	}
	if len(rep.Findings) != 1 || rep.Findings[0].Code != testquality.CodeNoAssertion ||
		rep.Findings[0].Func != "TestVacuous" {
		t.Fatalf("findings = %+v; want the one %s on TestVacuous", rep.Findings, testquality.CodeNoAssertion)
	}
	if rep.CountsByCode[testquality.CodeNoAssertion] != 1 {
		t.Fatalf("counts_by_code = %v; want one %s", rep.CountsByCode, testquality.CodeNoAssertion)
	}
	if rep.Schema == "" {
		t.Fatalf("report carries no schema id: %+v", rep)
	}
}

// The other half of the same contract: a floor that already covers the tree must
// report through JSON as "not growing" with exit 0. A verdict field that only ever
// said "growing" would be as useless as one that only ever said nothing.
func TestRunTestQualityJSONCarriesTheNotGrowingVerdict(t *testing.T) {
	root := seedTestQualityRepo(t, "sample_test.go",
		"package sample\n\nimport \"testing\"\n\nfunc TestVacuous(t *testing.T) { x := 1; _ = x }\n")
	if err := os.MkdirAll(filepath.Join(root, "internal", "testquality"), 0755); err != nil {
		t.Fatal(err)
	}
	floor := testquality.FormatBaseline([]testquality.Finding{{
		Code: testquality.CodeNoAssertion, File: "sample_test.go", Func: "TestVacuous",
	}})
	if err := os.WriteFile(filepath.Join(root, "internal", "testquality", "baseline.txt"), floor, 0644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	code := runTestQuality(&out, &errOut, []string{"--root", root, "--json"})

	var rep testQualityJSON
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("--json printed %q, which does not decode as a report: %v", out.String(), err)
	}
	if code != 0 {
		t.Fatalf("exit %d on a tree fully covered by its floor; stderr=%q", code, errOut.String())
	}
	if rep.Verdict != "not_growing" || rep.New != 0 || rep.Total != 1 {
		t.Fatalf("report = %+v; want verdict not_growing, new 0, total 1", rep)
	}
	if rep.Findings == nil {
		t.Fatalf("findings decoded as null, not an empty array: %q", out.String())
	}
}

// --write-baseline must not eat the review record. The baseline header tells its
// reader to regenerate with exactly this command, so a writer that dropped the
// trailing comment block would destroy the only place the tree records WHY a row
// is deliberate — on the one command the file itself recommends.
func TestRunTestQualityWriteBaselineKeepsReviewNotes(t *testing.T) {
	root := seedTestQualityRepo(t, "sample_test.go",
		"package sample\n\nimport \"testing\"\n\nfunc TestVacuous(t *testing.T) { x := 1; _ = x }\n")
	dir := filepath.Join(root, "internal", "testquality")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	const note = "# TestVacuous is a child-process fixture; the parent asserts."
	prior := "TESTQ_NO_ASSERTION\tgone_test.go\tTestGone\t1\n\n## Precision spot-check at SHA deadbeef\n" + note + "\n"
	path := filepath.Join(dir, "baseline.txt")
	if err := os.WriteFile(path, []byte(prior), 0644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	if code := runTestQuality(&out, &errOut, []string{"--root", root, "--write-baseline"}); code != 0 {
		t.Fatalf("exit %d; stderr=%q", code, errOut.String())
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), note) {
		t.Fatalf("regeneration dropped the review notes:\n%s", got)
	}
	if strings.Contains(string(got), "gone_test.go") {
		t.Fatalf("regeneration kept a floor row no longer present in the tree:\n%s", got)
	}
	if _, err := testquality.ParseBaseline(got); err != nil {
		t.Fatalf("regenerated baseline does not parse: %v", err)
	}
}
