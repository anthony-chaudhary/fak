package main

// session_compact_audit_test.go — the wired `fak session compact-audit` verb end to end
// over the internal/session fixtures (#4763). The classification logic is proven in
// internal/session; this pins the CLI seam: flag parsing, --json/--scrub, exit codes.

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureCorpusRoot() string {
	return filepath.Join("..", "..", "internal", "session", "testdata", "compactaudit")
}

func TestSessionCompactAuditHumanReport(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot()})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"compaction health", "resident context", "append-only", "verdicts:"} {
		if !strings.Contains(got, want) {
			t.Errorf("report missing %q\n%s", want, got)
		}
	}
}

func TestSessionCompactAuditJSONScrubbed(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--json", "--scrub"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	var res struct {
		Root      string `json:"root"`
		Aggregate struct {
			Sessions int `json:"sessions"`
			Fires    int `json:"fires"`
		} `json:"aggregate"`
		Sessions []struct {
			Path string `json:"path"`
			Cwd  string `json:"cwd"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if res.Aggregate.Sessions != 5 {
		t.Errorf("sessions = %d, want 5", res.Aggregate.Sessions)
	}
	if res.Root != "" {
		t.Errorf("scrubbed root = %q, want empty", res.Root)
	}
	for _, s := range res.Sessions {
		if s.Path != "" || s.Cwd != "" {
			t.Errorf("scrubbed session kept path/cwd: %q/%q", s.Path, s.Cwd)
		}
	}
	if strings.Contains(out.String(), "MUST_NOT_LEAK") {
		t.Error("json leaked a prompt body")
	}
}

func TestSessionCompactAuditAggregateOnly(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--json", "--aggregate-only"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	if strings.Contains(out.String(), `"sessions": [`) {
		t.Errorf("--aggregate-only still emitted per-session rows:\n%s", out.String())
	}
}

func TestSessionCompactAuditMissingRoot(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", filepath.Join(t.TempDir(), "does-not-exist")})
	if rc != 1 {
		t.Errorf("rc = %d, want 1 for a missing corpus root; stderr = %s", rc, errb.String())
	}
}

func TestSessionCompactAuditBadSince(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--since", "last-tuesday"})
	if rc != 2 {
		t.Errorf("rc = %d, want 2 for a malformed --since", rc)
	}
}
