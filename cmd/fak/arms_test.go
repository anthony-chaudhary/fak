package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

func TestArmsCLIFullLifecycle(t *testing.T) {
	gw, err := gateway.New(gateway.Config{})
	if err != nil {
		t.Fatalf("failed to create gateway: %v", err)
	}
	coord := researcharm.NewCoordinator(8)
	gw.SetResearchArmCoordinator(coord)

	ts := httptest.NewServer(gw.Handler())
	defer ts.Close()

	var stdout, stderr bytes.Buffer

	// 1. fak arms help
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"help"}); code != 0 {
		t.Fatalf("arms help returned %d", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak arms") {
		t.Errorf("expected usage text, got: %s", stdout.String())
	}

	// 2. fak arms status (table mode)
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"status", "--addr", ts.URL}); code != 0 {
		t.Fatalf("arms status returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "RESEARCH PROJECT ARMS") {
		t.Errorf("expected RESEARCH PROJECT ARMS header, got: %s", stdout.String())
	}

	// 3. fak arms status --json
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"status", "--addr", ts.URL, "--json"}); code != 0 {
		t.Fatalf("arms status --json returned %d: %s", code, stderr.String())
	}
	var snap researcharm.Snapshot
	if err := json.Unmarshal(stdout.Bytes(), &snap); err != nil {
		t.Fatalf("failed to unmarshal arms status json: %v", err)
	}

	// 4. fak arms who / traffic
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"who", "--addr", ts.URL}); code != 0 {
		t.Fatalf("arms who returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "No active in-flight requests") {
		t.Errorf("expected no active in-flight requests message, got: %s", stdout.String())
	}

	// 5. fak arms lease acquire
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"lease", "acquire", "--addr", ts.URL, "--arm", "experiment-eval", "--concurrency", "4"}); code != 0 {
		t.Fatalf("lease acquire returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Lease acquired successfully") {
		t.Errorf("expected lease acquired confirmation, got: %s", stdout.String())
	}

	// 6. fak arms lease list
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"lease", "list", "--addr", ts.URL}); code != 0 {
		t.Fatalf("lease list returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "experiment-eval") {
		t.Errorf("expected experiment-eval in lease list, got: %s", stdout.String())
	}

	// 7. fak arms limit
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"limit", "--addr", ts.URL, "--arm", "experiment-eval", "--max-concurrency", "10"}); code != 0 {
		t.Fatalf("arms limit returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Updated concurrency limit") {
		t.Errorf("expected updated concurrency limit confirmation, got: %s", stdout.String())
	}

	// 8. fak arms lease release
	stdout.Reset()
	stderr.Reset()
	if code := runArms(&stdout, &stderr, []string{"lease", "release", "--addr", ts.URL, "--arm", "experiment-eval"}); code != 0 {
		t.Fatalf("lease release returned %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "released successfully") {
		t.Errorf("expected released successfully confirmation, got: %s", stdout.String())
	}
}
