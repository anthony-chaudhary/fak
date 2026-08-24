package main

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestParseCloseBatchInput_BareArray(t *testing.T) {
	issues, budget, err := parseCloseBatchInput([]byte(`[101, 102, 103]`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(issues) != 3 || issues[0] != 101 {
		t.Fatalf("want [101 102 103], got %v", issues)
	}
	if budget != defaultCloseBatchBudget {
		t.Fatalf("want the default budget for a bare array, got %+v", budget)
	}
}

func TestParseCloseBatchInput_ObjectWithBudget(t *testing.T) {
	raw := []byte(`{"issues": [1, 2], "budget": {"remaining": 40, "limit": 5000, "reset_unix": 9999}}`)
	issues, budget, err := parseCloseBatchInput(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(issues) != 2 {
		t.Fatalf("want 2 issues, got %v", issues)
	}
	if budget.Remaining != 40 || budget.Limit != 5000 || budget.ResetAtUnix != 9999 {
		t.Fatalf("want the parsed budget, got %+v", budget)
	}
}

func TestParseCloseBatchInput_ObjectWithoutBudget_UsesDefault(t *testing.T) {
	issues, budget, err := parseCloseBatchInput([]byte(`{"issues": [7]}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(issues) != 1 || issues[0] != 7 {
		t.Fatalf("want [7], got %v", issues)
	}
	if budget != defaultCloseBatchBudget {
		t.Fatalf("want the default budget when omitted, got %+v", budget)
	}
}

func TestParseCloseBatchInput_BadJSON(t *testing.T) {
	if _, _, err := parseCloseBatchInput([]byte("not json")); err == nil {
		t.Fatal("want an error for invalid JSON")
	}
}

func TestRunDispatchCloseBatch_FileInput(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	fixture := `{"issues": [1,2,3,4,5,6,7,8,9,10,11,12], "budget": {"remaining": 5000, "limit": 5000, "reset_unix": 9999}}`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchCloseBatch(&stdout, &stderr, []string{"--in", path, "--batch-size", "5", "--now", "1000"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "batch 0:") || !strings.Contains(out, "batch 1:") {
		t.Fatalf("want at least 2 batches rendered, got %q", out)
	}
	if !strings.Contains(out, "cost=5") {
		t.Fatalf("want a batch cost shown, got %q", out)
	}
}

func TestRunDispatchCloseBatch_JSONOutput(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/issues.json"
	if err := os.WriteFile(path, []byte(`[1,2,3]`), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchCloseBatch(&stdout, &stderr, []string{"--in", path, "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"total_issues": 3`) {
		t.Fatalf("want total_issues=3 in the json report, got %s", stdout.String())
	}
}

func TestParseCloseBatchInputFinalMileGate(t *testing.T) {
	accepted := `{"candidates":[{"issue":7,"operator_facing":true,"delivery":{"schema":"fak.work-delivery/v1","id":"issue-7","revision":"cmd/fak@r1+gabc","axes":{"authoring":"recorded","compile_admission":"admitted","verification":"passed","integration":"integrated","release":"ready","activation":"activated","operator_acceptance":"accepted"},"activated":{"revision":"cmd/fak@r1+gabc","build_digest":"sha256:build","config_digest":"sha256:config"},"accepted":{"revision":"cmd/fak@r1+gabc","build_digest":"sha256:build","config_digest":"sha256:config","journey":"TUI changed"}}}]}`
	issues, _, err := parseCloseBatchInput([]byte(accepted))
	if err != nil || !reflect.DeepEqual(issues, []int{7}) {
		t.Fatalf("issues=%v err=%v", issues, err)
	}
	releaseOnly := `{"candidates":[{"issue":8,"operator_facing":true,"delivery":{"schema":"fak.work-delivery/v1","id":"issue-8","revision":"cmd/fak@r1+gabc","axes":{"authoring":"recorded","compile_admission":"admitted","verification":"passed","integration":"integrated","release":"ready"}}}]}`
	if _, _, err := parseCloseBatchInput([]byte(releaseOnly)); err == nil || !strings.Contains(err.Error(), "not activated") {
		t.Fatalf("release-only err=%v", err)
	}
}
