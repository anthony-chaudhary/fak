package gateway

// soft_progress_stall_test.go — #10638: the gateway side of the soft no-progress diagnostic.
// The planner's SoftStallNotify hook (wired onto every reachable upstream in New) writes a
// content-free, public-safe receipt — elapsed-since-progress, soft window, retry attempt — to
// the incident directory BEFORE the hard client-survivable deadline ends the turn. This is the
// durable evidence #10634's offline audit consumes, captured without killing a healthy turn.

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// readSoftStalls reads every JSON line of the soft-stall receipt log.
func readSoftStalls(t *testing.T, dir string) []map[string]any {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, "soft-stalls.jsonl"))
	if err != nil {
		t.Fatalf("no soft-stall receipt was written to %s — the soft deadline left no diagnostic (#10638): %v", dir, err)
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
			t.Fatalf("decode soft-stall line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

// TestOnSoftProgressStallWritesContentFreeReceipt is the core gateway witness: the hook records
// the elapsed silence, the soft window, and the retry attempt, marks the turn as still running,
// and carries NO content-bearing key at all.
func TestOnSoftProgressStallWritesContentFreeReceipt(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_STREAM_INCIDENT_DIR", dir)

	srv := newTestServer(t)
	srv.onSoftProgressStall(agent.SoftProgressStall{
		ElapsedSinceProgress: 42 * time.Second,
		Window:               42 * time.Second,
		RetryAttempt:         3,
	})

	pkts := readSoftStalls(t, dir)
	if len(pkts) != 1 {
		t.Fatalf("wrote %d soft-stall receipts, want exactly 1", len(pkts))
	}
	pkt := pkts[0]
	if pkt["schema"] != "fak-soft-progress-stall/1" {
		t.Fatalf("schema = %v, want fak-soft-progress-stall/1", pkt["schema"])
	}
	if pkt["class"] != "no_progress" {
		t.Fatalf("class = %v, want no_progress", pkt["class"])
	}
	if pkt["phase"] != "soft_no_progress" {
		t.Fatalf("phase = %v, want soft_no_progress", pkt["phase"])
	}
	if pkt["disposition"] != "turn_continues" {
		t.Fatalf("disposition = %v, want turn_continues — the soft strike must not kill the turn", pkt["disposition"])
	}
	if v, ok := pkt["elapsed_since_progress_ms"].(float64); !ok || v != 42000 {
		t.Fatalf("elapsed_since_progress_ms = %v, want 42000", pkt["elapsed_since_progress_ms"])
	}
	if v, ok := pkt["window_ms"].(float64); !ok || v != 42000 {
		t.Fatalf("window_ms = %v, want 42000", pkt["window_ms"])
	}
	if v, ok := pkt["retry_attempt"].(float64); !ok || v != 3 {
		t.Fatalf("retry_attempt = %v, want 3", pkt["retry_attempt"])
	}
	// The receipt must carry ONLY the bounded evidence keys — never a content-bearing field.
	allowed := map[string]bool{
		"schema": true, "class": true, "phase": true, "disposition": true,
		"elapsed_since_progress_ms": true, "window_ms": true, "retry_attempt": true,
	}
	for k := range pkt {
		if !allowed[k] {
			t.Fatalf("soft-stall receipt carries unexpected key %q — must be bounded/content-free", k)
		}
	}
}

// TestOnSoftProgressStallDisabledWithoutIncidentDir pins the off switch: with no incident dir
// configured the hook is a silent no-op, so behavior is unchanged for a gateway nobody wired.
func TestOnSoftProgressStallDisabledWithoutIncidentDir(t *testing.T) {
	t.Setenv("FAK_STREAM_INCIDENT_DIR", "")
	srv := newTestServer(t)
	// Must not panic and must not create anything.
	srv.onSoftProgressStall(agent.SoftProgressStall{ElapsedSinceProgress: time.Second, Window: time.Second, RetryAttempt: 1})
}

// TestNewWiresSoftStallNotifyOntoReachablePlanner proves the platform wiring: New attaches the
// hook beside the other planner notifies, so a reachable HTTPPlanner really does emit the soft
// diagnostic. Without this a correctly-built hook would never run.
func TestNewWiresSoftStallNotifyOntoReachablePlanner(t *testing.T) {
	srv, err := New(Config{
		EngineID: "test",
		Model:    "test-model",
		APIKey:   "test-key",
		Provider: "openai",
		BaseURL:  "http://127.0.0.1:1/v1",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer srv.Close()

	var wired bool
	walkHTTPPlanners(srv.planner, func(hp *agent.HTTPPlanner) {
		if hp.SoftStallNotify == nil {
			t.Fatalf("reachable HTTPPlanner has no SoftStallNotify — #10638 diagnostic is unreachable")
		}
		if hp.RetryNotify == nil {
			t.Fatalf("sanity: RetryNotify missing too, walk missed the planner")
		}
		wired = true
	})
	if !wired {
		t.Fatal("no reachable HTTPPlanner found to inspect — the wiring test proved nothing")
	}
}

// TestNewConfiguredHTTPPlannerCarriesStreamSoftProgressTimeout pins the threading seam for the
// soft diagnostic (#10638): the Config value rides onto the planner VERBATIM, so 0 keeps meaning
// "derive from the hard window" and the negative off switch survives the hop.
func TestNewConfiguredHTTPPlannerCarriesStreamSoftProgressTimeout(t *testing.T) {
	for _, c := range []struct {
		name string
		set  time.Duration
		want time.Duration
	}{
		{"unset (derive from hard)", 0, 0},
		{"an explicit soft window rides through", 45 * time.Second, 45 * time.Second},
		{"the off switch keeps its negative encoding", -1 * time.Second, -1 * time.Second},
	} {
		t.Run(c.name, func(t *testing.T) {
			p, err := newConfiguredHTTPPlanner(Config{
				Provider:                  "openai",
				APIKey:                    "test-key",
				StreamSoftProgressTimeout: c.set,
			}, "m", "https://example.invalid")
			if err != nil {
				t.Fatalf("newConfiguredHTTPPlanner: %v", err)
			}
			if p.StreamSoftProgressTimeout != c.want {
				t.Fatalf("planner StreamSoftProgressTimeout = %s, want %s", p.StreamSoftProgressTimeout, c.want)
			}
		})
	}
}
