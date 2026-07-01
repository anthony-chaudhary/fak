package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const testDurationFixture = "testdata/go-test-json-duration.jsonl"

func TestParseTestDurationLedgerRanksSlowPackagesAndTests(t *testing.T) {
	f, err := os.Open(testDurationFixture)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	ledger, err := parseTestDurationLedger(f, testDurationOptions{
		Source:        "fixture",
		PackageBudget: time.Second,
		TestBudget:    500 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if ledger.Schema != testDurationLedgerSchema {
		t.Fatalf("schema = %q", ledger.Schema)
	}
	if ledger.Summary.Packages != 2 || ledger.Summary.Tests != 3 || ledger.Summary.Findings != 2 {
		t.Fatalf("summary = %+v", ledger.Summary)
	}
	if got, want := ledger.Summary.SlowestPackage, "example.com/fak/internal/foo"; got != want {
		t.Fatalf("slowest package = %q, want %q", got, want)
	}
	if got, want := ledger.Summary.SlowestTest, "example.com/fak/internal/foo::TestSlow"; got != want {
		t.Fatalf("slowest test = %q, want %q", got, want)
	}
	if len(ledger.Packages) != 2 || ledger.Packages[0].Package != "example.com/fak/internal/foo" || ledger.Packages[0].ElapsedMS != 1500 {
		t.Fatalf("packages = %+v", ledger.Packages)
	}
	if ledger.Packages[0].Tests != 2 || ledger.Packages[0].SlowestTest != "TestSlow" || ledger.Packages[0].SlowestTestMS != 1234 {
		t.Fatalf("foo package row = %+v", ledger.Packages[0])
	}
	if len(ledger.Tests) != 3 || ledger.Tests[0].Package != "example.com/fak/internal/foo" || ledger.Tests[0].Test != "TestSlow" || ledger.Tests[0].ElapsedMS != 1234 {
		t.Fatalf("tests = %+v", ledger.Tests)
	}
	gotFindings := []string{ledger.Findings[0].Kind + ":" + ledger.Findings[0].Target, ledger.Findings[1].Kind + ":" + ledger.Findings[1].Target}
	wantFindings := []string{
		"test_over_budget:example.com/fak/internal/foo::TestSlow",
		"package_over_budget:example.com/fak/internal/foo",
	}
	if !reflect.DeepEqual(gotFindings, wantFindings) {
		t.Fatalf("findings = %v, want %v", gotFindings, wantFindings)
	}
}

func TestRunTestDurationsCommandEmitsLedger(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runTest(&out, &errb, []string{
		"durations",
		"--input", testDurationFixture,
		"--package-budget", "1s",
		"--test-budget", "500ms",
	})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}

	var ledger testDurationLedger
	if err := json.Unmarshal(out.Bytes(), &ledger); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if ledger.Schema != testDurationLedgerSchema || ledger.PackageBudgetMS != 1000 || ledger.TestBudgetMS != 500 {
		t.Fatalf("ledger header = %+v", ledger)
	}
	if len(ledger.Findings) != 2 || ledger.Findings[0].Rank != 1 || !strings.Contains(ledger.Findings[0].Action, "slow test") {
		t.Fatalf("findings = %+v", ledger.Findings)
	}
}

func TestPlanTestDurationRunInjectsJSONFlag(t *testing.T) {
	linux, err := planTestDurationRun("linux", "fast")
	if err != nil {
		t.Fatal(err)
	}
	wantLinux := []string{"go", "test", "-json", "-short", "./..."}
	if !reflect.DeepEqual(linux.Argv, wantLinux) {
		t.Fatalf("linux argv = %v, want %v", linux.Argv, wantLinux)
	}

	windows, err := planTestDurationRun("windows", "./cmd/fak")
	if err != nil {
		t.Fatal(err)
	}
	if !windows.ViaWSL || len(windows.Argv) < 7 || windows.Argv[0] != "powershell" ||
		windows.Argv[len(windows.Argv)-2] != "-json" || windows.Argv[len(windows.Argv)-1] != "./cmd/fak" {
		t.Fatalf("windows argv = %+v", windows)
	}
}

func TestRunTestDurationsRunModeWritesLedger(t *testing.T) {
	oldRun := testDurationRunCommand
	t.Cleanup(func() { testDurationRunCommand = oldRun })

	raw, err := os.ReadFile(testDurationFixture)
	if err != nil {
		t.Fatal(err)
	}
	var gotName string
	var gotArgs []string
	testDurationRunCommand = func(name string, args []string, stderr io.Writer) ([]byte, int, error) {
		gotName = name
		gotArgs = append([]string(nil), args...)
		return raw, 0, nil
	}

	outPath := filepath.Join(t.TempDir(), "durations", "ledger.json")
	var out, errb bytes.Buffer
	rc := runTest(&out, &errb, []string{
		"durations",
		"--run", "./cmd/fak",
		"--out", outPath,
		"--package-budget", "1s",
	})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	if gotName == "" || len(gotArgs) == 0 {
		t.Fatalf("fake runner was not called")
	}
	joined := gotName + " " + strings.Join(gotArgs, " ")
	if !strings.Contains(joined, "-json") || !strings.Contains(joined, "./cmd/fak") {
		t.Fatalf("runner command = %q, want go test -json ./cmd/fak", joined)
	}

	var stdoutLedger, fileLedger testDurationLedger
	if err := json.Unmarshal(out.Bytes(), &stdoutLedger); err != nil {
		t.Fatalf("stdout json: %v\n%s", err, out.String())
	}
	fileBytes, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fileBytes, &fileLedger); err != nil {
		t.Fatalf("file json: %v\n%s", err, string(fileBytes))
	}
	if stdoutLedger.Source == "" || !strings.Contains(stdoutLedger.Source, "go test -json") {
		t.Fatalf("source = %q", stdoutLedger.Source)
	}
	if !reflect.DeepEqual(stdoutLedger.Command, fileLedger.Command) || stdoutLedger.Summary != fileLedger.Summary {
		t.Fatalf("stdout/file ledger mismatch:\nstdout=%+v\nfile=%+v", stdoutLedger, fileLedger)
	}
}

func TestRunTestDurationsCheckFailsOnBudgetFindings(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runTest(&out, &errb, []string{
		"duration-ledger",
		"--input", testDurationFixture,
		"--package-budget", "1s",
		"--check",
	})
	if rc != 1 {
		t.Fatalf("rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), `"kind": "package_over_budget"`) {
		t.Fatalf("budget failure should still emit the ledger, got:\n%s", out.String())
	}
}
