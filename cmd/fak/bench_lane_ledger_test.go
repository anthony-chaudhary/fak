package main

import (
	"bytes"
	"encoding/json"
	"slices"
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
	assertLaneLedgerCaveats(t, caveats, laneLedgerCaveats)
}

// assertLaneLedgerCaveats requires the reported caveats to be exactly the expected set: every entry a
// non-empty string drawn from want, and every wanted caveat reported.
func assertLaneLedgerCaveats(t *testing.T, got []any, want []string) {
	t.Helper()
	reported := make(map[string]bool, len(got))
	for _, item := range got {
		text, _ := item.(string)
		if text == "" {
			t.Errorf("caveat %v is not a non-empty string", item)
			continue
		}
		if !slices.Contains(want, text) {
			t.Errorf("unexpected caveat %q", text)
		}
		reported[text] = true
	}
	for _, w := range want {
		if !reported[w] {
			t.Errorf("caveat missing: %q", w)
		}
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

// sm80Args is the sm_80 launch of TestBenchLaneLedgerSM80HandValues with a caller-chosen grid and extra flags.
func sm80Args(grid string, extra ...string) []string {
	args := []string{"--arch", "sm_80", "--sms", "108", "--threads", "128", "--regs", "40", "--smem", "768", "--grid", grid}
	return append(args, extra...)
}

// findByString returns the first object in list whose key holds want.
func findByString(t *testing.T, list []any, key, want string) map[string]any {
	t.Helper()
	for _, item := range list {
		if obj, ok := item.(map[string]any); ok && obj[key] == want {
			return obj
		}
	}
	t.Fatalf("no entry with %s=%q in %v", key, want, list)
	return nil
}

// TestBenchLaneLedgerMeasuredHandValues joins a 1000-cycle window to the one-wave sm_80 launch. Each cell is
// its lane-slot-waves times 1000 cycles: active 1152, unit_idle 202752, resident_unfilled 12672 and
// resource_capped 4608. Together they fill 108*2048*1000 with nothing unattributed. The measured 1000000
// active lane-cycles leaves a 152000 shortfall, which the model calls overstated.
func TestBenchLaneLedgerMeasuredHandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "1000000")...)
	m := laneLedgerObject(t, doc, "measured")
	if m["unit"] != "lane-slot-cycles" {
		t.Errorf("measured.unit = %v, want lane-slot-cycles", m["unit"])
	}
	wantInts := map[string]int64{
		"cycles_per_sm":        1000,
		"capacity_lane_cycles": 221184000,
		"unattributed":         0,
		"model_active":         1152000,
		"measured_active":      1000000,
		"active_shortfall":     152000,
	}
	for key, want := range wantInts {
		if got := laneLedgerNumber(t, m, key); got != want {
			t.Errorf("measured.%s = %d, want %d", key, got, want)
		}
	}
	cells := laneLedgerObject(t, m, "cells")
	wantCells := map[string]int64{
		"active":            1152000,
		"partial_warp":      0,
		"resident_unfilled": 12672000,
		"resource_capped":   4608000,
		"unit_idle":         202752000,
	}
	for key, want := range wantCells {
		if got := laneLedgerNumber(t, cells, key); got != want {
			t.Errorf("measured.cells.%s = %d, want %d", key, got, want)
		}
	}
	if m["verdict"] != "model_overstates_active" || m["active_supplied"] != true || m["assumed_equal_waves"] != true {
		t.Errorf("verdict/active_supplied/assumed_equal_waves = %v/%v/%v", m["verdict"], m["active_supplied"], m["assumed_equal_waves"])
	}

	findings, _ := m["findings"].([]any)
	if len(findings) == 0 {
		t.Fatalf("measured.findings is empty")
	}
	if top, _ := findings[0].(map[string]any); top["name"] != "unit_idle" {
		t.Errorf("largest measured finding = %v, want unit_idle", top["name"])
	}
	// The additive findings partition the capacity that is not active, so they must sum to it exactly.
	var additiveSum int64
	for _, item := range findings {
		if f, _ := item.(map[string]any); f["additive"] == true {
			additiveSum += laneLedgerNumber(t, f, "lane_slot_cycles")
		}
	}
	if want := int64(221184000 - 1152000); additiveSum != want {
		t.Errorf("additive findings sum to %d, want capacity minus active = %d", additiveSum, want)
	}
	inactive := findByString(t, findings, "name", "unexplained_inactive")
	if inactive["additive"] != false || laneLedgerNumber(t, inactive, "lane_slot_cycles") != 152000 {
		t.Errorf("unexplained_inactive = %v, want non-additive 152000", inactive)
	}

	caveats, _ := m["caveats"].([]any)
	assertLaneLedgerCaveats(t, caveats, append(append([]string(nil), laneLedgerMeasuredCaveats...), laneLedgerEqualWavesCaveat))
}

// Without active_lane_cycles the active fields stay zero, the verdict is empty, and no unexplained_inactive
// finding is emitted, since nothing was measured to fall short.
func TestBenchLaneLedgerMeasuredWithoutActive(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("9", "--cycles-per-sm", "1000")...)
	m := laneLedgerObject(t, doc, "measured")
	if m["active_supplied"] != false {
		t.Errorf("active_supplied = %v, want false", m["active_supplied"])
	}
	if m["verdict"] != "" {
		t.Errorf("verdict = %q, want empty", m["verdict"])
	}
	if got := laneLedgerNumber(t, m, "active_shortfall"); got != 0 {
		t.Errorf("active_shortfall = %d, want 0", got)
	}
	findings, _ := m["findings"].([]any)
	for _, item := range findings {
		if f, _ := item.(map[string]any); f["name"] == "unexplained_inactive" {
			t.Errorf("unexplained_inactive emitted without active lanes: %v", f)
		}
	}
}

// A measured zero is still a supplied count: --active-lane-cycles 0 gets a verdict, not the empty one.
func TestBenchLaneLedgerMeasuredZeroActiveSupplied(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "0")...)
	m := laneLedgerObject(t, doc, "measured")
	if m["active_supplied"] != true || m["verdict"] != "model_overstates_active" {
		t.Errorf("active_supplied/verdict = %v/%v, want true/model_overstates_active", m["active_supplied"], m["verdict"])
	}
	if got := laneLedgerNumber(t, m, "active_shortfall"); got != 1152000 {
		t.Errorf("active_shortfall = %d, want 1152000", got)
	}
}

// A measured count above the model's active lanes gets an explicit non-additive residual, unexplained_active,
// so the measured active lanes are visible in a row and not only in the verdict.
func TestBenchLaneLedgerMeasuredUnderstatesResidual(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "1200000")...)
	m := laneLedgerObject(t, doc, "measured")
	if m["verdict"] != "model_understates_active" {
		t.Fatalf("verdict = %v, want model_understates_active", m["verdict"])
	}
	findings, _ := m["findings"].([]any)
	residual := findByString(t, findings, "name", "unexplained_active")
	if residual["additive"] != false || laneLedgerNumber(t, residual, "lane_slot_cycles") != 48000 {
		t.Errorf("unexplained_active = %v, want non-additive 48000", residual)
	}
	for _, item := range findings {
		if f, _ := item.(map[string]any); f["name"] == "unexplained_inactive" {
			t.Errorf("unexplained_inactive emitted for an understating verdict: %v", f)
		}
	}
}

// Without --cycles-per-sm the document keeps its original keys: no measured object appears.
func TestBenchLaneLedgerDefaultOmitsMeasured(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("9")...)
	if _, ok := doc["measured"]; ok {
		t.Errorf("default mode emitted a measured key")
	}
}

// TestBenchLaneLedgerMeasuredWaveWeights uses grid 1300, which is two waves: 1296 blocks fill every SM at 12
// blocks each, and the final wave holds 4 blocks on 4 SMs. Full-wave shape: active 165888, resource_capped
// 55296. Final-wave shape: active 512, resident_unfilled 5632, resource_capped 2048, unit_idle 212992. Weights
// 800 and 200 give active 165888*800 + 512*200 = 132812800, and the four cells sum to 221184000 with nothing
// unattributed.
func TestBenchLaneLedgerMeasuredWaveWeights(t *testing.T) {
	doc := runLaneLedgerJSON(t, sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "800,200")...)
	if got := laneLedgerNumber(t, laneLedgerObject(t, doc, "ledger"), "waves"); got != 2 {
		t.Fatalf("ledger.waves = %d, want 2", got)
	}
	m := laneLedgerObject(t, doc, "measured")
	if m["assumed_equal_waves"] != false {
		t.Errorf("assumed_equal_waves = %v, want false when weights are given", m["assumed_equal_waves"])
	}
	cells := laneLedgerObject(t, m, "cells")
	wantCells := map[string]int64{
		"active":            132812800,
		"partial_warp":      0,
		"resident_unfilled": 1126400,
		"resource_capped":   44646400,
		"unit_idle":         42598400,
	}
	for key, want := range wantCells {
		if got := laneLedgerNumber(t, cells, key); got != want {
			t.Errorf("measured.cells.%s = %d, want %d", key, got, want)
		}
	}
	if got := laneLedgerNumber(t, m, "unattributed"); got != 0 {
		t.Errorf("unattributed = %d, want 0", got)
	}
}

func TestBenchLaneLedgerMeasuredRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{
			name:    "zero cycles refused by the measured join",
			args:    sm80Args("9", "--cycles-per-sm", "0"),
			wantErr: "CyclesPerSM must be positive",
		},
		{
			name:    "wave weights without cycles refused",
			args:    sm80Args("9", "--wave-weights", "1000"),
			wantErr: "--wave-weights describes a measured window and needs --cycles-per-sm",
		},
		{
			name:    "tolerance without cycles refused",
			args:    sm80Args("9", "--tolerance", "0.1"),
			wantErr: "--tolerance describes a measured window and needs --cycles-per-sm",
		},
		{
			name:    "exact without cycles refused",
			args:    sm80Args("9", "--exact"),
			wantErr: "--exact describes a measured window and needs --cycles-per-sm",
		},
		{
			name:    "tolerance above one refused",
			args:    sm80Args("9", "--cycles-per-sm", "1000", "--tolerance", "1.5"),
			wantErr: "ActiveTolerance must be a fraction in [0, 1]",
		},
		{
			name:    "negative active lane-cycles refused",
			args:    sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "-1"),
			wantErr: "ActiveLaneCycles must not be negative",
		},
		{
			name:    "active lane-cycles over window capacity refused",
			args:    sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "221184001"),
			wantErr: "exceeds the window capacity",
		},
		{
			name:    "wave weights of the wrong length refused",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "800,100,100"),
			wantErr: "needs one entry per modelled wave",
		},
		{
			name:    "non-integer wave weight refused",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "800,x"),
			wantErr: `"x" is not an integer cycle count`,
		},
		{
			name:    "empty wave weight refused",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "800,"),
			wantErr: "is not an integer cycle count",
		},
		{
			name:    "non-positive wave weight refused",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "800,0"),
			wantErr: "every WaveWeights entry must be positive",
		},
		{
			name:    "sampled wave weights over the window refused, not clamped",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "900,200"),
			wantErr: "sampled WaveWeights sum exceeds CyclesPerSM; refused, not clamped",
		},
		{
			name:    "exact wave weights over the window refused",
			args:    sm80Args("1300", "--cycles-per-sm", "1000", "--wave-weights", "900,200", "--exact"),
			wantErr: "exact WaveWeights sum exceeds CyclesPerSM",
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

func TestBenchLaneLedgerMeasuredHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr,
		sm80Args("9", "--cycles-per-sm", "1000", "--active-lane-cycles", "1000000"))
	if code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"model cells over measured window (lane-slot-cycles)", "verdict=model_overstates_active", "unexplained_inactive", "model-apportioned findings", "unit_idle", "measured caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("measured table missing %q:\n%s", want, out)
		}
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
