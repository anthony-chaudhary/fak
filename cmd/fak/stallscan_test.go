package main

import (
	"bytes"
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
