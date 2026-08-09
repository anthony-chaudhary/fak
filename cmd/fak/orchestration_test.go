package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func orchestrationFixture(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("..", "..", "internal", "orchestration", "testdata", name)
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOrchestrationPlanJSONSelfcheckOffline(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task", orchestrationFixture(t, "forced-ultracode.json"), "--json", "--selfcheck",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
	}
	if got.Schema != orchestration.SchemaVersion || got.Resolved.Profile != orchestration.ProfileUltracode {
		t.Fatalf("unexpected plan: %+v", got.Resolved)
	}
	if len(got.Resolved.Explanation) == 0 || !strings.Contains(stderr.String(), "SELFCHECK PASS") || !strings.Contains(stderr.String(), "launched=0") {
		t.Fatalf("missing offline selfcheck/explanation: stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
}

func TestOrchestrationPlanTextReportsEveryDegradation(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task", orchestrationFixture(t, "unsupported-harness.json"), "--capabilities", "unsupported",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	for _, capability := range []string{"concurrency", "task_messaging", "cancellation", "leases", "independent_witness"} {
		if !strings.Contains(stdout.String(), "DEGRADED "+capability) {
			t.Errorf("missing %s degradation:\n%s", capability, stdout.String())
		}
	}
}

func TestOrchestrationPlanStrictRejectsDegradation(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task", orchestrationFixture(t, "unsupported-harness.json"), "--capabilities", "unsupported", "--strict",
	})
	if code != 3 || !strings.Contains(stderr.String(), "rejects degradation") {
		t.Fatalf("exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}
