package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

func TestRenderStallFingerprint_showsCauseAndTop(t *testing.T) {
	s := stallscan.Sample{
		TotalFaultsPerSec:      489505,
		HardFaultsPerSec:       508,
		DemandZeroFaultsPerSec: 192374,
		TransitionFaultsPerSec: 136422,
		ContextSwitchesPerSec:  24000,
		AvailableMB:            229257,
		TopIO: []stallscan.ProcIO{
			{PID: 6028, Name: "MsMpEng.exe", Ops: 7423},
			{PID: 18456, Name: "AUEPMaster.exe", Ops: 248187},
		},
	}
	v := stallscan.Classify(s, stallscan.DefaultThresholds())
	var buf bytes.Buffer
	renderStallFingerprint(&buf, s, v, 6)
	out := buf.String()
	if !strings.Contains(out, "soft_fault_churn") {
		t.Fatalf("expected cause soft_fault_churn in output:\n%s", out)
	}
	if !strings.Contains(out, "AUEPMaster.exe") {
		t.Fatalf("expected top process in output:\n%s", out)
	}
	// The disk/RAM "not the cause" line must be present — it's the whole point.
	if !strings.Contains(out, "not-the-cause") {
		t.Fatalf("expected not-the-cause line:\n%s", out)
	}
}

func TestStallFingerprint_hasSchemaAndVerdict(t *testing.T) {
	s := stallscan.Sample{TotalFaultsPerSec: 1000, AvailableMB: 229000}
	v := stallscan.Classify(s, stallscan.DefaultThresholds())
	rec := stallFingerprint(s, v)
	if rec["schema"] != "fak.stallscan.v1" {
		t.Fatalf("schema = %v", rec["schema"])
	}
	if _, ok := rec["verdict"]; !ok {
		t.Fatalf("verdict missing from record")
	}
}

// A census in which TWO processes independently cross the reboot high-water: the
// leaking terminal and, behind it, the TermService svchost the single-max verdict
// used to mask (#4614).
func twoCrosserStallSample() stallscan.Sample {
	return stallscan.Sample{
		AvailableMB:       229000,
		SystemHandleTotal: 300000,
		TopHandles: []stallscan.ProcHandles{
			{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054},
			{PID: 9001, Name: "svchost.exe", Handles: 31200},
		},
	}
}

func TestStallFingerprint_rebootBlockCarriesEveryCrosser(t *testing.T) {
	// The --json reboot block must name both drivers, not just the max.
	s := twoCrosserStallSample()
	rec := stallFingerprint(s, stallscan.Classify(s, stallscan.DefaultThresholds()))
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Reboot stallscan.RebootAdvice `json:"reboot"`
	}
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Reboot.Process != "WindowsTerminal.exe" {
		t.Fatalf("headline must stay the worst hog: %+v", back.Reboot)
	}
	if len(back.Reboot.Crossers) != 2 || back.Reboot.Crossers[1].Process != "svchost.exe" {
		t.Fatalf("--json reboot block must carry both crossers, got %+v", back.Reboot.Crossers)
	}
	if !strings.Contains(string(b), `"crossers"`) {
		t.Fatalf("crossers key missing from the emitted JSON:\n%s", b)
	}
}

func TestRenderStallFingerprint_listsSecondaryCrossers(t *testing.T) {
	s := twoCrosserStallSample()
	var buf bytes.Buffer
	renderStallFingerprint(&buf, s, stallscan.Classify(s, stallscan.DefaultThresholds()), 6)
	out := buf.String()
	if !strings.Contains(out, "reboot      : ADVISED") || !strings.Contains(out, "WindowsTerminal.exe pid 4242 at 33054") {
		t.Fatalf("expected the headline reboot line:\n%s", out)
	}
	if !strings.Contains(out, "ALSO (handle_high_water axis): svchost.exe pid 9001 at 31200") {
		t.Fatalf("the second crosser must be listed beneath the headline:\n%s", out)
	}
	// One crosser renders exactly as before: no ALSO line at all.
	var solo bytes.Buffer
	one := stallscan.Sample{AvailableMB: 229000, SystemHandleTotal: 300000,
		TopHandles: []stallscan.ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054}}}
	renderStallFingerprint(&solo, one, stallscan.Classify(one, stallscan.DefaultThresholds()), 6)
	if strings.Contains(solo.String(), "ALSO") {
		t.Fatalf("a lone crosser must not grow an ALSO line:\n%s", solo.String())
	}
}

func TestRunStallscan_usageOnBadFlag(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runStallscan(&out, &errb, []string{"--nope"})
	if rc != 2 {
		t.Fatalf("rc = %d, want 2 (usage)", rc)
	}
}

func TestSortProcIOByOps(t *testing.T) {
	in := []stallscan.ProcIO{{Ops: 1}, {Ops: 99}, {Ops: 5}}
	got := sortProcIOByOps(in)
	if got[0].Ops != 99 || got[2].Ops != 1 {
		t.Fatalf("bad order: %+v", got)
	}
}

func TestBoundStallLogRetainsCompleteNewestRecords(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stallscan.jsonl")
	for i := 0; i < 20; i++ {
		appendStallJSONL(path, map[string]any{"seq": i, "payload": strings.Repeat("x", 40)}, 400)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) > 400 {
		t.Fatalf("bounded log is %d bytes, want <= 400", len(b))
	}
	lines := bytes.Split(bytes.TrimSpace(b), []byte{'\n'})
	if len(lines) == 0 {
		t.Fatal("bounded log is empty")
	}
	for _, line := range lines {
		var record map[string]any
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatalf("partial JSONL record %q: %v", line, err)
		}
	}
	var newest map[string]any
	if err := json.Unmarshal(lines[len(lines)-1], &newest); err != nil {
		t.Fatal(err)
	}
	if newest["seq"] != float64(19) {
		t.Fatalf("newest seq=%v, want 19", newest["seq"])
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "stallscan.jsonl" {
		t.Fatalf("atomic replacement left residue: %v", entries)
	}
}
