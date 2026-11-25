package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// runLaneLedgerJSON runs the subcommand with --json and decodes the document into generic maps, so
// the test pins the wire keys rather than the Go struct tags.
func runLaneLedgerJSON(t *testing.T, args ...string) map[string]any {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if code := runBenchLaneLedger(&stdout, &stderr, append(args, "--json")); code != 0 {
		t.Fatalf("runBenchLaneLedger(%v) exit %d; stderr: %s", args, code, stderr.String())
	}
	var doc map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &doc); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, stdout.String())
	}
	return doc
}

func laneLedgerObject(t *testing.T, parent map[string]any, key string) map[string]any {
	t.Helper()
	obj, ok := parent[key].(map[string]any)
	if !ok {
		t.Fatalf("key %q is %T, want object", key, parent[key])
	}
	return obj
}

func laneLedgerNumber(t *testing.T, parent map[string]any, key string) int64 {
	t.Helper()
	n, ok := parent[key].(float64)
	if !ok {
		t.Fatalf("key %q is %T, want number", key, parent[key])
	}
	return int64(n)
}

// TestBenchLaneLedgerSM80HandValues pins the sm_80 R=40 TPB=128 smem=768 sms=108 grid=9 launch
// against values worked out by hand: 12 resident blocks per SM (register-limited), one wave,
// 108*2048 = 221184 lane-slot-waves, and 9 busy SMs out of 108.
func TestBenchLaneLedgerSM80HandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, "--arch", "sm_80", "--sms", "108", "--threads", "128",
		"--regs", "40", "--smem", "768", "--grid", "9", "--kernel", "k_flash_attention")

	for _, key := range []string{"schema", "unit", "arch", "provenance", "launch", "ledger",
		"findings", "register_footprint", "register_findings", "caveats"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("top-level key %q missing", key)
		}
	}
	if doc["schema"] != "fak.lane-ledger/1" || doc["unit"] != "lane-slot-waves" || doc["arch"] != "sm_80" {
		t.Errorf("schema/unit/arch = %v/%v/%v", doc["schema"], doc["unit"], doc["arch"])
	}

	ledger := laneLedgerObject(t, doc, "ledger")
	wantInts := map[string]int64{
		"capacity":             221184,
		"waves":                1,
		"blocks_per_sm":        12,
		"block_slots_unfilled": 1287, // 1*108*12 slots - 9 blocks
	}
	for key, want := range wantInts {
		if got := laneLedgerNumber(t, ledger, key); got != want {
			t.Errorf("ledger.%s = %d, want %d", key, got, want)
		}
	}
	if ledger["limiter"] != "register" {
		t.Errorf("ledger.limiter = %v, want register", ledger["limiter"])
	}
	if ledger["conserved"] != true {
		t.Errorf("ledger.conserved = %v, want true", ledger["conserved"])
	}

	cells := laneLedgerObject(t, ledger, "cells")
	wantCells := map[string]int64{
		"active":            9 * 128,      // 9 blocks x 128 threads
		"partial_warp":      0,            // 128 threads fill whole warps
		"resident_unfilled": 9 * 11 * 128, // 11 free block slots on each of the 9 busy SMs
		"resource_capped":   9 * (2048 - 12*128),
		"unit_idle":         99 * 2048, // 108 - 9 SMs receive no block
	}
	for key, want := range wantCells {
		if got := laneLedgerNumber(t, cells, key); got != want {
			t.Errorf("ledger.cells.%s = %d, want %d", key, got, want)
		}
	}

	findings, ok := doc["findings"].([]any)
	if !ok || len(findings) == 0 {
		t.Fatalf("findings = %v, want a non-empty list", doc["findings"])
	}
	top, _ := findings[0].(map[string]any)
	if top["name"] != "underfill" || laneLedgerNumber(t, top, "lane_slot_waves") != 99*2048 {
		t.Errorf("largest finding = %v, want underfill of 202752 lane-slot-waves", top)
	}

	fp := laneLedgerObject(t, doc, "register_footprint")
	wantFP := map[string]int64{
		"regs_per_sm":       65536,
		"useful":            48 * 1280,
		"rounding_waste":    0,
		"granularity_waste": 3 * 1280,
		"headroom_unused":   0,
		"remainder":         256,
	}
	for key, want := range wantFP {
		if got := laneLedgerNumber(t, fp, key); got != want {
			t.Errorf("register_footprint.%s = %d, want %d", key, got, want)
		}
	}

	regFindings, _ := doc["register_findings"].([]any)
	if len(regFindings) != 1 {
		t.Fatalf("register_findings = %v, want only register_granularity (headroom is off under a register limiter)", regFindings)
	}
	if name, _ := regFindings[0].(map[string]any)["name"].(string); name != "register_granularity" {
		t.Errorf("register finding = %q, want register_granularity", name)
	}

	caveats, _ := doc["caveats"].([]any)
	if len(caveats) != 5 {
		t.Errorf("caveats has %d entries, want 5", len(caveats))
	}
}

// TestBenchLaneLedgerSM120RunsWithCallerSMCount runs an sm_120 launch whose 100 threads leave partial
// warps. 84 SMs * 1536 lanes = 129024; 12 blocks per SM, one wave; 9 blocks * 28 idle lanes = 252.
func TestBenchLaneLedgerSM120RunsWithCallerSMCount(t *testing.T) {
	doc := runLaneLedgerJSON(t, "--arch", "sm_120", "--sms", "84", "--threads", "100", "--regs", "40", "--grid", "9")

	ledger := laneLedgerObject(t, doc, "ledger")
	if got := laneLedgerNumber(t, ledger, "capacity"); got != 129024 {
		t.Errorf("capacity = %d, want 129024", got)
	}
	if ledger["conserved"] != true {
		t.Errorf("conserved = %v, want true", ledger["conserved"])
	}
	cells := laneLedgerObject(t, ledger, "cells")
	if got := laneLedgerNumber(t, cells, "partial_warp"); got != 252 {
		t.Errorf("partial_warp = %d, want 252", got)
	}
	launch := laneLedgerObject(t, doc, "launch")
	if got := laneLedgerNumber(t, launch, "device_sms"); got != 84 {
		t.Errorf("launch.device_sms = %d, want 84", got)
	}
}

func TestBenchLaneLedgerRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{
			name:    "zero registers refused as unmeasured",
			args:    []string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "0", "--grid", "9"},
			wantErr: "registers per thread must come from a measurement",
		},
		{
			name:    "missing sms refused",
			args:    []string{"--arch", "sm_80", "--threads", "128", "--regs", "40", "--grid", "9"},
			wantErr: "--sms is required",
		},
		{
			name:    "zero sms refused",
			args:    []string{"--arch", "sm_80", "--sms", "0", "--threads", "128", "--regs", "40", "--grid", "9"},
			wantErr: "--sms is required",
		},
		{
			name:    "unknown arch refused",
			args:    []string{"--arch", "sm_999", "--sms", "108", "--threads", "128", "--regs", "40", "--grid", "9"},
			wantErr: `unknown --arch "sm_999"`,
		},
		{
			name:    "missing arch refused",
			args:    []string{"--sms", "108", "--threads", "128", "--regs", "40", "--grid", "9"},
			wantErr: "--arch is required",
		},
		{
			name:    "threads over the block ceiling refused by the ledger",
			args:    []string{"--arch", "sm_80", "--sms", "108", "--threads", "2048", "--regs", "40", "--grid", "9"},
			wantErr: "ThreadsPerBlock must be in [1, 1024]",
		},
		{
			name:    "registers over the per-thread ceiling refused by the ledger",
			args:    []string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "256", "--grid", "9"},
			wantErr: "per-thread register ceiling",
		},
		{
			name:    "missing grid refused",
			args:    []string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "40"},
			wantErr: "--grid is required",
		},
		{
			name:    "positional argument refused",
			args:    []string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "40", "--grid", "9", "extra"},
			wantErr: "unexpected argument",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runBenchLaneLedger(&stdout, &stderr, tc.args); code == 0 {
				t.Fatalf("exit 0, want non-zero; stdout: %s", stdout.String())
			}
			if !strings.Contains(stderr.String(), tc.wantErr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), tc.wantErr)
			}
			if stdout.Len() != 0 {
				t.Errorf("refusal wrote to stdout: %q", stdout.String())
			}
		})
	}
}

func TestBenchLaneLedgerHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr,
		[]string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "40", "--smem", "768", "--grid", "9"})
	if code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"conserved=true", "capacity=221184", "unit_idle", "underfill", "caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q:\n%s", want, out)
		}
	}
}
