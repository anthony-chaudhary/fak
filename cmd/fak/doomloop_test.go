package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestDoomloopTickWiresCorrection drives the full shell path: repeated
// burning-flat ticks must trip DOOM_LOOP, write an accountability-ledger row,
// and (with --correct) queue a reversible re-anchor nudge to the outbox.
func TestDoomloopTickWiresCorrection(t *testing.T) {
	store := t.TempDir()
	var lastOut bytes.Buffer

	// Five observations: effort climbs every tick, progress pinned flat.
	efforts := []string{"10", "20", "30", "40", "50"}
	for i, e := range efforts {
		lastOut.Reset()
		var errb bytes.Buffer
		argv := []string{"tick",
			"--session", "w1",
			"--effort", e,
			"--progress", "7",
			"--alive",
			"--correct",
			"--store", store,
			"--now", dlItoa(int64(1000 + i)),
		}
		if rc := runDoomloop(&lastOut, &errb, strings.NewReader(""), argv); rc != 0 {
			t.Fatalf("tick %d rc=%d stderr=%s", i, rc, errb.String())
		}
	}

	if !strings.Contains(lastOut.String(), string("DOOM_LOOP")) {
		t.Fatalf("final tick did not report DOOM_LOOP:\n%s", lastOut.String())
	}

	// Accountability ledger exists and its last row is a nudge-queued doom loop.
	ledger := filepath.Join(store, "decisions.jsonl")
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	lines := dlNonEmptyLines(string(raw))
	if len(lines) != len(efforts) {
		t.Fatalf("ledger has %d rows, want %d", len(lines), len(efforts))
	}
	var last dlDecision
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatalf("decode last decision: %v", err)
	}
	if last.Verdict != "DOOM_LOOP" {
		t.Fatalf("last verdict = %q, want DOOM_LOOP", last.Verdict)
	}
	if last.Applied != "nudge-queued" {
		t.Fatalf("last applied = %q, want nudge-queued", last.Applied)
	}

	// The reversible correction artifact landed in the outbox.
	outbox := filepath.Join(store, "outbox")
	des, err := os.ReadDir(outbox)
	if err != nil || len(des) == 0 {
		t.Fatalf("expected a queued nudge packet in %s (err=%v, n=%d)", outbox, err, len(des))
	}
	pktRaw, err := os.ReadFile(filepath.Join(outbox, des[0].Name()))
	if err != nil {
		t.Fatalf("read nudge: %v", err)
	}
	var pkt nudgePacket
	if err := json.Unmarshal(pktRaw, &pkt); err != nil {
		t.Fatalf("decode nudge: %v", err)
	}
	if !pkt.Reversible || pkt.Kind != "reanchor" || pkt.Session != "w1" {
		t.Fatalf("nudge packet malformed: %+v", pkt)
	}
}

// TestDoomloopScanFoldsStore verifies scan classifies each stored worker.
func TestDoomloopScanFoldsStore(t *testing.T) {
	store := t.TempDir()
	// Healthy worker: effort and progress both climb.
	for i := 0; i < 4; i++ {
		var o, e bytes.Buffer
		argv := []string{"tick", "--session", "healthy",
			"--effort", dlItoa(int64(10 * (i + 1))), "--progress", dlItoa(int64(i + 1)),
			"--alive", "--store", store, "--now", dlItoa(int64(2000 + i))}
		if rc := runDoomloop(&o, &e, strings.NewReader(""), argv); rc != 0 {
			t.Fatalf("healthy tick rc=%d %s", rc, e.String())
		}
	}
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"scan", "--store", store, "--json"}); rc != 0 {
		t.Fatalf("scan rc=%d %s", rc, e.String())
	}
	if !strings.Contains(o.String(), "healthy") || !strings.Contains(o.String(), "HEALTHY") {
		t.Fatalf("scan output missing healthy worker verdict:\n%s", o.String())
	}
}

// TestDoomloopClassifyStdin verifies the one-shot path over a samples stream.
func TestDoomloopClassifyStdin(t *testing.T) {
	in := strings.Join([]string{
		`{"unix_millis":0,"effort":5,"progress":2,"alive":true}`,
		`{"unix_millis":1,"effort":15,"progress":2,"alive":true}`,
		`{"unix_millis":2,"effort":25,"progress":2,"alive":true}`,
		`{"unix_millis":3,"effort":35,"progress":2,"alive":true}`,
	}, "\n")
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(in), []string{"classify"}); rc != 0 {
		t.Fatalf("classify rc=%d %s", rc, e.String())
	}
	if !strings.Contains(o.String(), "DOOM_LOOP") {
		t.Fatalf("classify did not detect doom loop:\n%s", o.String())
	}
}

func TestDoomloopTickRequiresSession(t *testing.T) {
	var o, e bytes.Buffer
	if rc := runDoomloop(&o, &e, strings.NewReader(""), []string{"tick", "--effort", "1"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for missing --session", rc)
	}
}

func dlNonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

func dlItoa(n int64) string { return strconv.FormatInt(n, 10) }
