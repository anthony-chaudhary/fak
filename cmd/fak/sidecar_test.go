package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captured fleet_sessions.py json payload — the existing contract the sidecar
// census reads. Two sessions across two accounts, one throttled account, plus a
// synthetic _probe row the census must skip.
const fleetSessionsFixture = `{
  "app_version": "test",
  "now": "2026-07-03T00:00:00Z",
  "accounts": [
    {"account": "acct-1", "available": true,  "blocked": false, "throttled": false},
    {"account": "acct-2", "available": false, "blocked": false, "throttled": true, "reset": "01:00Z"}
  ],
  "rows": [
    {"account": "acct-1", "session": "sess-a", "project": "work", "disp": "LIVE"},
    {"account": "acct-2", "session": "sess-b", "project": "work", "disp": "STOPPED_LIMIT"},
    {"account": "acct-1", "session": "probe-x", "project": "_probe", "disp": "LIVE"}
  ]
}`

func writeFixture(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return p
}

// TestSidecarTextFromCensus drives the shell over a captured census payload and
// asserts the sessions/accounts planes fold in, while the un-fed lanes/posture
// planes read UNMEASURED (honest degradation, never a silent green).
func TestSidecarTextFromCensus(t *testing.T) {
	from := writeFixture(t, "census.json", fleetSessionsFixture)
	var stdout, stderr bytes.Buffer
	rc := runSidecar(&stdout, &stderr, []string{"--from", from})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	// Two real sessions (the _probe row is skipped), one live.
	for _, want := range []string{"2 session(s) (1 live)", "sess-a", "sess-b", "acct-1", "acct-2", "throttled"} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q:\n%s", want, out)
		}
	}
	// The un-fed planes degrade honestly.
	if !strings.Contains(out, "UNMEASURED") || !strings.Contains(out, "no lane reading") {
		t.Errorf("lanes/posture did not degrade to UNMEASURED:\n%s", out)
	}
	// probe rows must not leak into the census.
	if strings.Contains(out, "probe-x") {
		t.Errorf("synthetic _probe row leaked into the sidecar:\n%s", out)
	}
}

// TestSidecarSlackBlocksValid proves --slack emits a well-formed Block Kit array
// carrying the same census facts as the text surface.
func TestSidecarSlackBlocksValid(t *testing.T) {
	from := writeFixture(t, "census.json", fleetSessionsFixture)
	var stdout, stderr bytes.Buffer
	rc := runSidecar(&stdout, &stderr, []string{"--from", from, "--slack"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var blocks []any
	if err := json.Unmarshal(stdout.Bytes(), &blocks); err != nil {
		t.Fatalf("--slack did not emit a JSON block array: %v\n%s", err, stdout.String())
	}
	if len(blocks) == 0 {
		t.Fatal("--slack emitted zero blocks")
	}
	// The first block is the header; the payload carries the census facts.
	raw := stdout.String()
	for _, want := range []string{"fak sidecar", "sess-a", "acct-2", "throttled"} {
		if !strings.Contains(raw, want) {
			t.Errorf("slack payload missing %q:\n%s", want, raw)
		}
	}
}

// TestSidecarWithLaneAndPostureReadings feeds all four planes and confirms the
// lane/posture readings render and the pane reports OK (no unmeasured gap).
func TestSidecarWithLaneAndPostureReadings(t *testing.T) {
	from := writeFixture(t, "census.json", fleetSessionsFixture)
	lanes := writeFixture(t, "lanes.json", `{"lanes":[{"lane":"cmd","kind":"cluster","held":true,"owner":"worker-7"},{"lane":"docs","kind":"cluster","held":false}]}`)
	posture := writeFixture(t, "posture.json", `{"cache_posture":"managed","compactions":3,"elisions":1,"sessions_joined":2}`)

	var stdout, stderr bytes.Buffer
	rc := runSidecar(&stdout, &stderr, []string{"--from", from, "--lanes-from", lanes, "--posture-from", posture, "--json"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var pane struct {
		OK         bool `json:"ok"`
		Unmeasured int  `json:"unmeasured"`
		Lanes      struct {
			Held int `json:"held"`
		} `json:"lanes"`
		Posture struct {
			Measured bool `json:"measured"`
		} `json:"posture"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &pane); err != nil {
		t.Fatalf("--json decode: %v\n%s", err, stdout.String())
	}
	if !pane.OK || pane.Unmeasured != 0 {
		t.Errorf("all four planes fed but pane OK=%v unmeasured=%d", pane.OK, pane.Unmeasured)
	}
	if pane.Lanes.Held != 1 {
		t.Errorf("lane held count = %d, want 1", pane.Lanes.Held)
	}
	if !pane.Posture.Measured {
		t.Error("posture fed but not measured")
	}
}

// TestSidecarMutuallyExclusiveSurfaces guards the flag contract.
func TestSidecarMutuallyExclusiveSurfaces(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runSidecar(&stdout, &stderr, []string{"--json", "--slack"}); rc != 2 {
		t.Fatalf("rc=%d, want 2 for --json+--slack", rc)
	}
}
