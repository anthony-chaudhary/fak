package main

import (
	"bytes"
	"encoding/json"
	"os"
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
