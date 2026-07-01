package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBuildTestShardPlanBalancesByDuration(t *testing.T) {
	ledger := testDurationLedger{
		Schema: testDurationLedgerSchema,
		Source: "fixture-ledger",
		Packages: []testDurationPackage{
			{Package: "example.com/fak/pkg/a", ElapsedMS: 900},
			{Package: "example.com/fak/pkg/b", ElapsedMS: 500},
			{Package: "example.com/fak/pkg/c", ElapsedMS: 400},
			{Package: "example.com/fak/pkg/d", ElapsedMS: 300},
		},
	}

	plan, err := buildTestShardPlan(ledger, testShardOptions{
		ShardCount:    2,
		CommandPrefix: []string{"go", "test", "-short", "-count=1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Schema != testShardPlanSchema || plan.Source != "fixture-ledger" {
		t.Fatalf("plan header = %+v", plan)
	}
	if plan.TotalPackages != 4 || plan.TotalElapsedMS != 2100 || plan.ImbalanceMS != 300 {
		t.Fatalf("plan totals = %+v", plan)
	}
	wantPackages := [][]string{
		{"example.com/fak/pkg/a", "example.com/fak/pkg/d"},
		{"example.com/fak/pkg/b", "example.com/fak/pkg/c"},
	}
	for i := range wantPackages {
		if !reflect.DeepEqual(plan.Shards[i].Packages, wantPackages[i]) {
			t.Fatalf("shard %d packages = %v, want %v", i+1, plan.Shards[i].Packages, wantPackages[i])
		}
		if !reflect.DeepEqual(plan.Shards[i].Command[:4], []string{"go", "test", "-short", "-count=1"}) {
			t.Fatalf("shard %d command prefix = %v", i+1, plan.Shards[i].Command)
		}
	}
}

func TestRunTestShardsCommandReadsDurationLedger(t *testing.T) {
	f, err := os.Open(testDurationFixture)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := parseTestDurationLedger(f, testDurationOptions{
		Source:        "fixture",
		PackageBudget: time.Second,
	})
	if closeErr := f.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "duration-ledger.json")
	var buf bytes.Buffer
	if err := writeIndentedJSONNoEscape(&buf, ledger); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runTest(&out, &errb, []string{
		"shards",
		"--input", path,
		"--shards", "2",
		"--go-arg", "-short",
		"--go-arg", "-count=1",
	})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	var plan testShardPlan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if plan.Schema != testShardPlanSchema || plan.ShardCount != 2 || plan.TotalPackages != 2 {
		t.Fatalf("plan = %+v", plan)
	}
	if len(plan.Shards) != 2 || plan.Shards[0].Packages[0] != "example.com/fak/internal/foo" {
		t.Fatalf("shards = %+v", plan.Shards)
	}
	if !strings.Contains(strings.Join(plan.Shards[0].Command, " "), "go test -short -count=1") {
		t.Fatalf("command missing go args: %+v", plan.Shards[0].Command)
	}
}

func TestRunTestShardsRejectsEmptyLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(path, []byte(`{"schema":"`+testDurationLedgerSchema+`","packages":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := runTest(&out, &errb, []string{"shard-plan", "--input", path})
	if rc != 1 {
		t.Fatalf("rc = %d, want 1; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no package rows") {
		t.Fatalf("stderr = %q", errb.String())
	}
}
