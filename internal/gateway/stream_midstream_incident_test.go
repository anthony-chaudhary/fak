package gateway

// stream_midstream_incident_test.go — #10672: the incident packet on a terminal mid-stream
// failure. When a stream that already produced a first token dies before message end, the
// gateway writes a bounded, content-free incident record: phase, time-to-first-token,
// bytes/events emitted, last-event age, upstream status class, and the bound policy in
// effect. It is the correlated gateway slice an operator needs to answer "what was lost"
// without re-reading a single byte of the conversation.

import (
	"bufio"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// readIncidents reads every JSON line of the incident log, failing the test when the file
// is absent — the pre-fix behavior this file exists to retire.
func readIncidents(t *testing.T, dir string) []map[string]any {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, "incidents.jsonl"))
	if err != nil {
		t.Fatalf("no incident packet was written to %s — a terminal mid-stream failure left no correlated evidence (#10672): %v", dir, err)
	}
	defer f.Close()
	var out []map[string]any
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 8*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("decode incident line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

// TestMidStreamFailureWritesIncidentPacket drives a stream that emits content, then dies
// mid-stream, and asserts the durable incident packet records the correlated slice —
// counts, durations, and classifications only, never the streamed content itself.
func TestMidStreamFailureWritesIncidentPacket(t *testing.T) {
	incDir := t.TempDir()
	t.Setenv("FAK_STREAM_INCIDENT_DIR", incDir)
	const contentMark = "INCIDENT-CONTENT-MARKER "
	planner := newHBProbePlanner("test-model", contentMark, "never", 100*time.Millisecond,
		&agent.UpstreamStalledError{Idle: 61 * time.Second, Kind: "idle"})

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(planner.releasePause)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)
	if resp := tap.waitHead(t, hbBudget, "no HTTP head"); resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	rest := tap.drain(t, 15*time.Second)
	joined := strings.Join(rest, "\n")
	if !strings.Contains(joined, "data: [DONE]") {
		t.Fatalf("errored stream did not terminate with [DONE]: %s", joined)
	}

	incs := readIncidents(t, incDir)
	var pkt map[string]any
	for _, m := range incs {
		if m["wire"] == "openai_chat_completions" {
			pkt = m
			break
		}
	}
	if pkt == nil {
		t.Fatalf("no openai_chat_completions incident in %v", incs)
	}
	if pkt["phase"] != "mid_stream" {
		t.Fatalf("phase = %v, want mid_stream — the failure landed after a first token", pkt["phase"])
	}
	if v, ok := pkt["first_token_ms"].(float64); !ok || v < 0 {
		t.Fatalf("first_token_ms = %v, want a non-negative number", pkt["first_token_ms"])
	}
	if v, ok := pkt["bytes_emitted"].(float64); !ok || v < float64(len(contentMark)) {
		t.Fatalf("bytes_emitted = %v, want >= %d (the first fragment reached the wire)", pkt["bytes_emitted"], len(contentMark))
	}
	if v, ok := pkt["events_emitted"].(float64); !ok || v < 1 {
		t.Fatalf("events_emitted = %v, want >= 1", pkt["events_emitted"])
	}
	if v, ok := pkt["elapsed_ms"].(float64); !ok || v <= 0 {
		t.Fatalf("elapsed_ms = %v, want > 0", pkt["elapsed_ms"])
	}
	if v, ok := pkt["last_event_age_ms"].(float64); !ok || v < 0 {
		t.Fatalf("last_event_age_ms = %v, want a non-negative number", pkt["last_event_age_ms"])
	}
	if pkt["upstream_status_class"] != "stall" {
		t.Fatalf("upstream_status_class = %v, want stall (the failure was a typed stall)", pkt["upstream_status_class"])
	}
	bp, _ := pkt["bound_policy"].(string)
	if !strings.Contains(bp, "max-duration=off") {
		t.Fatalf("bound_policy = %q, want it to record the max-duration policy state (off by default)", bp)
	}
	if v, _ := pkt["cause"].(string); strings.TrimSpace(v) == "" {
		t.Fatalf("cause = %v, want a bounded non-empty cause", pkt["cause"])
	}

	// The packet is counts and classifications ONLY: the streamed content must not appear.
	raw, err := os.ReadFile(filepath.Join(incDir, "incidents.jsonl"))
	if err != nil {
		t.Fatalf("read incidents: %v", err)
	}
	if strings.Contains(string(raw), contentMark) {
		t.Fatal("incident packet carried streamed content — the packet must be content-free")
	}
}

// TestMidStreamIncidentRecordsMaxDurationBound pins the bound-policy half of the packet:
// when FAK_STREAM_MAX_DURATION_S is armed and the bound is what ended the stream, the
// incident names the class and the armed value instead of a bare stall.
func TestMidStreamIncidentRecordsMaxDurationBound(t *testing.T) {
	incDir := t.TempDir()
	t.Setenv("FAK_STREAM_INCIDENT_DIR", incDir)
	t.Setenv("FAK_STREAM_MAX_DURATION_S", "5")
	planner := newHBProbePlanner("test-model", "bound ", "never", 100*time.Millisecond,
		&agent.UpstreamStalledError{Idle: 5 * time.Second, Kind: "max-duration"})

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(planner.releasePause)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)
	if resp := tap.waitHead(t, hbBudget, "no HTTP head"); resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	rest := tap.drain(t, 20*time.Second)
	if !strings.Contains(strings.Join(rest, "\n"), "data: [DONE]") {
		t.Fatalf("errored stream did not terminate with [DONE]: %s", rest)
	}

	incs := readIncidents(t, incDir)
	var pkt map[string]any
	for _, m := range incs {
		if m["wire"] == "openai_chat_completions" {
			pkt = m
			break
		}
	}
	if pkt == nil {
		t.Fatalf("no openai_chat_completions incident in %v", incs)
	}
	if pkt["upstream_status_class"] != "bound:max-stream-duration" {
		t.Fatalf("upstream_status_class = %v, want bound:max-stream-duration — the armed bound ended this stream", pkt["upstream_status_class"])
	}
	bp, _ := pkt["bound_policy"].(string)
	if !strings.Contains(bp, "max-duration=5s") {
		t.Fatalf("bound_policy = %q, want the armed 5s max-duration recorded", bp)
	}
}
