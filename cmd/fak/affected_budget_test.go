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

func TestAffectedBudgetRegressionRedsAndWritesReport(t *testing.T) {
	oldGraph := affectedListGraph
	oldRun := affectedRunGoTest
	oldNow := affectedNow
	t.Cleanup(func() {
		affectedListGraph = oldGraph
		affectedRunGoTest = oldRun
		affectedNow = oldNow
	})

	affectedListGraph = func(root string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{
				"internal/foo/foo.go": "example.com/m/internal/foo",
			}, map[string][]string{
				"example.com/m/cmd/fak": {"example.com/m/internal/foo"},
			}, 2, nil
	}
	var ran []string
	affectedRunGoTest = func(root string, args []string, stdout, stderr io.Writer) (int, error) {
		ran = append([]string(nil), args...)
		return 0, nil
	}
	start := time.Unix(0, 0)
	calls := 0
	affectedNow = func() time.Time {
		calls++
		if calls == 1 {
			return start
		}
		return start.Add(150 * time.Millisecond)
	}

	reportPath := filepath.Join(t.TempDir(), "verify-loop.json")
	var stdout, stderr bytes.Buffer
	code := runAffected(&stdout, &stderr, []string{
		"--file", `internal\foo\foo.go`,
		"--budget", "100ms",
		"--report", reportPath,
	})
	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "GATE_LATENCY_REGRESSION") {
		t.Fatalf("stderr missing GATE_LATENCY_REGRESSION:\n%s", stderr.String())
	}
	wantArgs := []string{"test", "example.com/m/cmd/fak", "example.com/m/internal/foo"}
	if !reflect.DeepEqual(ran, wantArgs) {
		t.Fatalf("go test args = %v, want %v", ran, wantArgs)
	}

	raw, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var rep affectedRunReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, raw)
	}
	if rep.Schema != "fak.verify_loop.v1" || rep.Mode != "incremental" {
		t.Fatalf("report identity = %s/%s, want fak.verify_loop.v1/incremental", rep.Schema, rep.Mode)
	}
	if rep.Verdict != "GATE_LATENCY_REGRESSION" || rep.BudgetMS != 100 || rep.ElapsedMS != 150 {
		t.Fatalf("report verdict/budget/elapsed = %s/%d/%d, want regression/100/150", rep.Verdict, rep.BudgetMS, rep.ElapsedMS)
	}
	if !reflect.DeepEqual(rep.ChangedFiles, []string{"internal/foo/foo.go"}) {
		t.Fatalf("changed files = %v, want slash-normalized path", rep.ChangedFiles)
	}
	if !reflect.DeepEqual(rep.SelectedPackages, []string{"example.com/m/cmd/fak", "example.com/m/internal/foo"}) {
		t.Fatalf("selected packages = %v", rep.SelectedPackages)
	}
}
