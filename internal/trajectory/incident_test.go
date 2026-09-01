package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const incidentTag = "guard-crash-rsi/a9d4034da07d8ef3"

func TestRunIncidentClassifiesRootsAndLaunchIdentity(t *testing.T) {
	root := t.TempDir()
	prompt := "You are the specially tagged crash-RSI investigation session " + incidentTag + ".\nInvestigate the original crash."
	writeIncidentFixture(t, root, "exec.jsonl", `{"timestamp":"2026-09-01T01:00:00Z","type":"session_meta","payload":{"id":"exec-root","timestamp":"2026-08-31T18:29:00Z","originator":"codex_exec","source":"exec"}}`, prompt, `{"timestamp":"2026-09-01T02:00:00Z","type":"event_msg","payload":{"type":"done"}}`)
	writeIncidentFixture(t, root, "cli.jsonl", `{"timestamp":"2026-08-31T19:25:00Z","type":"session_meta","payload":{"id":"cli-root","timestamp":"2026-08-31T19:25:00Z","originator":"codex_cli_rs","source":"cli"}}`, prompt)
	writeIncidentFixture(t, root, "child.jsonl", `{"timestamp":"2026-08-31T19:26:00Z","type":"session_meta","payload":{"id":"child","timestamp":"2026-08-31T19:26:00Z","originator":"codex_exec","source":{"subagent":{"thread_spawn":{"parent_thread_id":"exec-root","depth":1}}}}}`, prompt)
	writeIncidentFixture(t, root, "quoted.jsonl", `{"timestamp":"2026-09-01T04:00:00Z","type":"session_meta","payload":{"id":"quoted","timestamp":"2026-09-01T04:00:00Z","originator":"codex_exec","source":"exec"}}`, "Investigate the current incident without using the historical launch identity.", `{"timestamp":"2026-09-01T04:01:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Historical evidence quoted `+incidentTag+` here."}]}}`)

	restart := time.Date(2026, 9, 1, 3, 0, 0, 0, time.UTC)
	packet, err := RunIncident(IncidentOptions{Root: root, Tag: incidentTag, Restart: restart, Now: time.Date(2026, 9, 1, 5, 0, 0, 0, time.UTC)})
	if err != nil {
		t.Fatal(err)
	}
	if len(packet.Sessions) != 3 {
		t.Fatalf("sessions=%d want 3: %#v", len(packet.Sessions), packet.Sessions)
	}
	wantSources := []IncidentCount{{Key: "cli", Count: 1}, {Key: "exec", Count: 1}, {Key: "subagent", Count: 1}}
	if got, want := incidentJSON(packet.BySource), incidentJSON(wantSources); got != want {
		t.Fatalf("by_source=%s want %s", got, want)
	}
	if packet.Sessions[0].SessionID != "exec-root" || packet.Sessions[0].Boundary != "before_restart" {
		t.Fatalf("authoritative start not used: %#v", packet.Sessions[0])
	}
	var child IncidentSession
	for _, row := range packet.Sessions {
		if row.SessionID == "child" {
			child = row
		}
	}
	if child.ParentID != "exec-root" || child.RootID != "exec-root" {
		t.Fatalf("child attribution=%#v", child)
	}
	encoded := incidentJSON(packet)
	if strings.Contains(encoded, "Investigate the original crash") {
		t.Fatalf("packet exposed prompt: %s", encoded)
	}

	sum := sha256.Sum256([]byte(prompt))
	byHash, err := RunIncident(IncidentOptions{Root: root, PromptSHA256: hex.EncodeToString(sum[:]), Now: time.Date(2026, 9, 1, 5, 0, 0, 0, time.UTC)})
	if err != nil || len(byHash.Sessions) != 3 {
		t.Fatalf("hash match sessions=%d err=%v", len(byHash.Sessions), err)
	}
}

func TestRunIncidentHonorsBounds(t *testing.T) {
	root := t.TempDir()
	prompt := "launch " + incidentTag
	writeIncidentFixture(t, root, "a.jsonl", `{"timestamp":"2026-08-31T18:29:00Z","type":"session_meta","payload":{"id":"a","timestamp":"2026-08-31T18:29:00Z","source":"exec"}}`, prompt)
	writeIncidentFixture(t, root, "b.jsonl", `{"timestamp":"2026-08-31T18:30:00Z","type":"session_meta","payload":{"id":"b","timestamp":"2026-08-31T18:30:00Z","source":"exec"}}`, prompt)
	packet, err := RunIncident(IncidentOptions{Root: root, Tag: incidentTag, MaxFiles: 1, MaxBytesPerFile: 1024, MaxBytesTotal: 1024, Now: time.Date(2026, 9, 1, 5, 0, 0, 0, time.UTC)})
	if err != nil {
		t.Fatal(err)
	}
	if !packet.Limits.Truncated || packet.Limits.FilesScanned != 1 || len(packet.Sessions) != 1 || packet.Sessions[0].SessionID != "a" {
		t.Fatalf("bounds=%#v sessions=%#v", packet.Limits, packet.Sessions)
	}
}

func writeIncidentFixture(t *testing.T, root, name, meta, prompt string, rest ...string) {
	t.Helper()
	row := map[string]any{"timestamp": "2026-08-31T18:29:01Z", "type": "response_item", "payload": map[string]any{"type": "message", "role": "user", "content": []map[string]string{{"type": "input_text", "text": prompt}}}}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	lines := append([]string{meta, string(encoded)}, rest...)
	if err := os.WriteFile(filepath.Join(root, name), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func incidentJSON(v any) string { data, _ := json.Marshal(v); return string(data) }
