package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
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

// TestSidecarDefaultSessionCollectorJoinsCrossHarnessDescriptors is the #8292
// spine witness: the DEFAULT collector sees both harness namespaces through the
// in-process census/descriptor join. The deliberately-invalid Python command is
// the no-subprocess fence: any legacy collector launch makes this test fail.
func TestSidecarDefaultSessionCollectorJoinsCrossHarnessDescriptors(t *testing.T) {
	home := t.TempDir()
	const claudeID = "11111111-1111-4111-8111-111111111111"
	const codexID = "22222222-2222-4222-8222-222222222222"
	writeSessionFixture(t, filepath.Join(home, ".claude", "projects", "fixture-project", claudeID+".jsonl"))
	writeSessionFixture(t, filepath.Join(home, ".codex", "sessions", "2026", "08", "20",
		"rollout-2026-08-20T00-00-00-"+codexID+".jsonl"))
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	var stdout, stderr bytes.Buffer
	rc := runSidecar(&stdout, &stderr, []string{"--json", "--python", "definitely-not-a-python-binary"})
	if rc != 0 {
		t.Fatalf("runSidecar rc=%d stderr=%q", rc, stderr.String())
	}
	var pane struct {
		Sessions struct {
			Measured bool `json:"measured"`
			Note     string
			Rows     []struct {
				Session string
				Harness string
			}
		}
	}
	if err := json.Unmarshal(stdout.Bytes(), &pane); err != nil {
		t.Fatalf("decode default sidecar JSON: %v\n%s", err, stdout.String())
	}
	if !pane.Sessions.Measured {
		t.Fatalf("default sessions are unmeasured: note=%q stderr=%q", pane.Sessions.Note, stderr.String())
	}

	got := make(map[string]string, len(pane.Sessions.Rows))
	for _, row := range pane.Sessions.Rows {
		got[row.Session] = row.Harness
	}
	if got[claudeID] != "claude" || got[codexID] != "codex" {
		t.Fatalf("cross-harness descriptor rows = %#v, want claude=%s codex=%s", got, claudeID, codexID)
	}
	if !strings.Contains(pane.Sessions.Note, "NO_NAMESPACE") || !strings.Contains(pane.Sessions.Note, "openai-generic") {
		t.Fatalf("typed unavailable source missing from rendered note %q", pane.Sessions.Note)
	}
}

func TestSidecarSessionDescriptorAdapterContract(t *testing.T) {
	t.Run("generic profile identity", func(t *testing.T) {
		rows, note, err := sidecarRowsFromFleetCensus([]fleetmon.CensusRow{{
			Agent: "custom-harness", Kind: fleetmon.KindSession, Session: "generic-session", Liveness: fleetmon.LivenessLive,
		}})
		if err != nil {
			t.Fatalf("sidecarRowsFromFleetCensus: %v", err)
		}
		if len(rows) != 1 || rows[0].Harness != "custom-harness" || rows[0].Disposition != "live" {
			t.Fatalf("generic descriptor rows = %+v", rows)
		}
		if !strings.Contains(note, "fak.session.descriptor.v1") {
			t.Fatalf("descriptor provenance missing from note %q", note)
		}
	})

	t.Run("measured empty", func(t *testing.T) {
		rows, note, err := sidecarRowsFromFleetCensus(nil)
		if err != nil || len(rows) != 0 || !strings.Contains(note, "measured empty") {
			t.Fatalf("empty census rows=%+v note=%q err=%v", rows, note, err)
		}
	})

	for _, tc := range []struct {
		name string
		rows []fleetmon.CensusRow
	}{
		{
			name: "empty identity",
			rows: []fleetmon.CensusRow{{Agent: "claude", Kind: fleetmon.KindSession, Session: ""}},
		},
		{
			name: "exact id collision",
			rows: []fleetmon.CensusRow{
				{Agent: "claude", Kind: fleetmon.KindSession, Session: "same-id"},
				{Agent: "codex", Kind: fleetmon.KindSession, Session: "same-id"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := sidecarRowsFromFleetCensus(tc.rows); err == nil {
				t.Fatalf("sidecarRowsFromFleetCensus(%+v) succeeded; want fail-closed refusal", tc.rows)
			}
		})
	}
}

func writeSessionFixture(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir session fixture: %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"type":"user"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write session fixture: %v", err)
	}
	now := time.Now()
	if err := os.Chtimes(path, now, now); err != nil {
		t.Fatalf("stamp session fixture: %v", err)
	}
}
