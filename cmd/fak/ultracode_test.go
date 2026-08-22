package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestUltracodeResolvesCanonicalFleetPlan(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runUltracode(&stdout, &stderr, []string{"--task-text", "fan out independent checks", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if got.Resolved.Profile != orchestration.ProfileUltracode {
		t.Fatalf("profile=%q", got.Resolved.Profile)
	}
	if got.Resolved.Budget.MaxWorkers < 2 || !got.Resolved.Leases.Required || !got.Resolved.Witness.Independent || !got.Resolved.Reconcile.Required {
		t.Fatalf("not a bounded witnessed fleet plan: %+v", got.Resolved)
	}
}

func TestUltracodeSelfcheckNeedsNoTaskInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runUltracode(&stdout, &stderr, []string{"--selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "SELFCHECK PASS") || !strings.Contains(stderr.String(), "launched=0") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}

func TestUltracodeRejectsProfileOverride(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runUltracode(&stdout, &stderr, []string{"--profile", "off", "--task-text", "x"}); code != 2 {
		t.Fatalf("code=%d", code)
	}
	if !strings.Contains(stderr.String(), "fixed to ultracode") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}

func TestUltracodeStatusDelegates(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runUltracode(&stdout, &stderr, []string{"status", "unexpected"}); code != 2 {
		t.Fatalf("code=%d", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak orchestration status") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}
