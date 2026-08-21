package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/idempotency"
)

func TestIdempotencyCommandHelper(t *testing.T) {
	if os.Getenv("FAK_IDEMPOTENCY_HELPER") != "1" {
		return
	}
	marker := os.Args[len(os.Args)-1]
	f, err := os.OpenFile(marker, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		os.Exit(91)
	}
	_, _ = f.WriteString("x")
	_ = f.Close()
	fmt.Fprint(os.Stdout, "created issue #1")
	os.Exit(7)
}

func TestIdempotencyCLIBlocksAmbiguousRetryUntilResolve(t *testing.T) {
	t.Setenv("FAK_IDEMPOTENCY_HELPER", "1")
	dir := t.TempDir()
	ledger := filepath.Join(dir, "idem.jsonl")
	marker := filepath.Join(dir, "applied.marker")
	op, token := "issue-create", "cli-response-loss"
	runArgs := []string{
		"run", "--op", op, "--token", token, "--ledger", ledger, "--",
		os.Args[0], "-test.run=^TestIdempotencyCommandHelper$", "--", marker,
	}

	var stdout, stderr bytes.Buffer
	if code := runIdempotency(&stdout, &stderr, runArgs); code != 1 {
		t.Fatalf("ambiguous run exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	message := stderr.String()
	if strings.Contains(strings.ToLower(message), "safe to retry") {
		t.Fatalf("ambiguous CLI output still promises safe retry: %q", message)
	}
	for _, want := range []string{"UNKNOWN_APPLIED", "fak idempotency status", "fak idempotency resolve"} {
		if !strings.Contains(message, want) {
			t.Errorf("ambiguous CLI output lacks %q: %q", want, message)
		}
	}
	assertMarkerCalls(t, marker, 1)

	stdout.Reset()
	stderr.Reset()
	if code := runIdempotency(&stdout, &stderr, runArgs); code != 1 {
		t.Fatalf("blocked retry exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	assertMarkerCalls(t, marker, 1)

	stdout.Reset()
	stderr.Reset()
	statusArgs := []string{"status", "--op", op, "--token", token, "--ledger", ledger, "--json"}
	if code := runIdempotency(&stdout, &stderr, statusArgs); code != 0 {
		t.Fatalf("status exit=%d stderr=%q", code, stderr.String())
	}
	var status struct {
		Found bool              `json:"found"`
		State idempotency.State `json:"state"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status %q: %v", stdout.String(), err)
	}
	if !status.Found || status.State != idempotency.StateUnknownApplied {
		t.Fatalf("status = %+v, want found UNKNOWN_APPLIED", status)
	}

	stdout.Reset()
	stderr.Reset()
	resolveArgs := []string{
		"resolve", "--op", op, "--token", token, "--ledger", ledger,
		"--applied-result", "created issue #1", "--json",
	}
	if code := runIdempotency(&stdout, &stderr, resolveArgs); code != 0 {
		t.Fatalf("resolve exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var resolved struct {
		State idempotency.State `json:"state"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &resolved); err != nil {
		t.Fatalf("decode resolution %q: %v", stdout.String(), err)
	}
	if resolved.State != idempotency.StateApplied {
		t.Fatalf("resolved state=%q, want APPLIED", resolved.State)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runIdempotency(&stdout, &stderr, runArgs); code != 0 {
		t.Fatalf("resolved replay exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stdout.String() != "created issue #1" {
		t.Fatalf("resolved replay stdout=%q", stdout.String())
	}
	assertMarkerCalls(t, marker, 1)
}

func TestIdempotencySelfcheckCoversAmbiguousResolution(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIdempotency(&stdout, &stderr, []string{"selfcheck", "--json"}); code != 0 {
		t.Fatalf("selfcheck exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var verdict struct {
		Pass        bool   `json:"pass"`
		Detail      string `json:"detail"`
		IssuesFiled int    `json:"issues_filed"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &verdict); err != nil {
		t.Fatalf("decode selfcheck %q: %v", stdout.String(), err)
	}
	if !verdict.Pass || verdict.IssuesFiled != 3 {
		t.Fatalf("selfcheck verdict=%+v", verdict)
	}
	for _, want := range []string{"ambiguous apply blocked", "proven-applied read-back replayed"} {
		if !strings.Contains(verdict.Detail, want) {
			t.Errorf("selfcheck detail lacks %q: %q", want, verdict.Detail)
		}
	}
}

func assertMarkerCalls(t *testing.T, path string, want int) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read marker: %v", err)
	}
	if got := len(b); got != want {
		t.Fatalf("mutating command ran %d times, want %d", got, want)
	}
}
