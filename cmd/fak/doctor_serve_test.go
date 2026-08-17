package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

const testGiB = int64(1) << 30

// TestServeReadinessISATiers pins the CPU-ISA row classification across the tiers:
// AVX-512/AMX green, AVX2 yellow with an upgrade hint, sub-AVX2 red with a hint,
// and arm64 NEON green.
func TestServeReadinessISATiers(t *testing.T) {
	cases := []struct {
		name       string
		facts      serveHostFacts
		wantStatus string
		wantTier   string
		wantHint   bool
	}{
		{"avx512-green", serveHostFacts{Arch: "amd64", ISA: "avx512"}, sevOK, "Ready", false},
		{"amx-green", serveHostFacts{Arch: "amd64", ISA: "amx"}, sevOK, "Ready", false},
		{"avx2-yellow", serveHostFacts{Arch: "amd64", ISA: "avx2"}, sevWarn, "Marginal", true},
		{"scalar-red", serveHostFacts{Arch: "amd64", ISA: "scalar"}, sevFail, "Unready", true},
		{"sse-red", serveHostFacts{Arch: "amd64", ISA: "sse"}, sevFail, "Unready", true},
		{"none-red", serveHostFacts{Arch: "amd64", ISA: ""}, sevFail, "Unready", true},
		{"neon-green", serveHostFacts{Arch: "arm64", ISA: "neon"}, sevOK, "Ready", false},
		{"arm-noneon-yellow", serveHostFacts{Arch: "arm64", ISA: ""}, sevWarn, "Marginal", true},
		{"unknown-arch-yellow", serveHostFacts{Arch: "riscv64", ISA: ""}, sevWarn, "Marginal", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			row := serveISARow(c.facts)
			if row.Check != "cpu-isa" {
				t.Fatalf("check = %q, want cpu-isa", row.Check)
			}
			if row.Status != c.wantStatus {
				t.Errorf("status = %q, want %q", row.Status, c.wantStatus)
			}
			if row.Tier != c.wantTier {
				t.Errorf("tier = %q, want %q", row.Tier, c.wantTier)
			}
			if (row.Remediation != "") != c.wantHint {
				t.Errorf("remediation present = %v, want %v (got %q)", row.Remediation != "", c.wantHint, row.Remediation)
			}
		})
	}
}

// TestServeReadinessModelFit pins the model-fit row: fits-with-headroom green,
// fits-raw-but-not-headroom yellow, exceeds-free red, plus the two "cannot verify"
// yellow rows (no model size / memory not probeable).
func TestServeReadinessModelFit(t *testing.T) {
	cases := []struct {
		name       string
		facts      serveHostFacts
		wantStatus string
		wantHint   bool
	}{
		{
			name:       "fits-with-headroom-green",
			facts:      serveHostFacts{ModelBytes: 40 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15},
			wantStatus: sevOK,
			wantHint:   false,
		},
		{
			name:       "fits-raw-not-headroom-yellow",
			facts:      serveHostFacts{ModelBytes: 75 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15},
			wantStatus: sevWarn,
			wantHint:   true,
		},
		{
			name:       "exceeds-free-red",
			facts:      serveHostFacts{ModelBytes: 200 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15},
			wantStatus: sevFail,
			wantHint:   true,
		},
		{
			name:       "no-model-size-yellow",
			facts:      serveHostFacts{ModelBytes: 0, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15},
			wantStatus: sevWarn,
			wantHint:   true,
		},
		{
			name:       "mem-unknown-yellow",
			facts:      serveHostFacts{ModelBytes: 40 * testGiB, FreeBytes: 0, MemKnown: false, Headroom: 0.15},
			wantStatus: sevWarn,
			wantHint:   true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			row := serveFitRow(c.facts)
			if row.Check != "model-fit" {
				t.Fatalf("check = %q, want model-fit", row.Check)
			}
			if row.Status != c.wantStatus {
				t.Errorf("status = %q, want %q (finding=%q)", row.Status, c.wantStatus, row.Finding)
			}
			if (row.Remediation != "") != c.wantHint {
				t.Errorf("remediation present = %v, want %v", row.Remediation != "", c.wantHint)
			}
		})
	}
}

// TestServeReadinessNUMA pins the NUMA row: unreadable topology is a benign yellow,
// single node is green with no hint, multi-node is green WITH a placement hint.
func TestServeReadinessNUMA(t *testing.T) {
	cases := []struct {
		name       string
		nodes      int
		wantStatus string
		wantHint   bool
	}{
		{"unreadable-yellow", 0, sevWarn, true},
		{"single-node-green", 1, sevOK, false},
		{"multi-node-green-hint", 4, sevOK, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			row := serveNUMARow(serveHostFacts{NUMANodes: c.nodes})
			if row.Check != "numa-topology" {
				t.Fatalf("check = %q, want numa-topology", row.Check)
			}
			if row.Status != c.wantStatus {
				t.Errorf("status = %q, want %q", row.Status, c.wantStatus)
			}
			if (row.Remediation != "") != c.wantHint {
				t.Errorf("remediation present = %v, want %v", row.Remediation != "", c.wantHint)
			}
			if c.nodes >= 2 && !strings.Contains(row.Remediation, "threadpool") {
				t.Errorf("multi-node hint should mention threadpool sizing, got %q", row.Remediation)
			}
		})
	}
}

// TestBuildServeReadinessRollup checks the whole-report fold: rollup is the worst
// tier, and Findings counts every non-green row (a green multi-node NUMA hint does
// NOT count as a finding).
func TestBuildServeReadinessRollup(t *testing.T) {
	// A red anywhere -> Unready rollup.
	red := buildServeReadiness(serveHostFacts{
		Arch: "amd64", ISA: "avx512",
		ModelBytes: 200 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15,
		NUMANodes: 2,
	})
	if red.Rollup != "Unready" {
		t.Errorf("rollup = %q, want Unready", red.Rollup)
	}
	if red.Findings != 1 { // only the model-fit red; ISA green, NUMA green(+hint)
		t.Errorf("findings = %d, want 1 (multi-node green hint must not count)", red.Findings)
	}
	if len(red.Rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(red.Rows))
	}

	// All green -> Ready, zero findings.
	green := buildServeReadiness(serveHostFacts{
		Arch: "amd64", ISA: "avx512",
		ModelBytes: 40 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15,
		NUMANodes: 1,
	})
	if green.Rollup != "Ready" || green.Findings != 0 {
		t.Errorf("all-green report = {rollup:%q findings:%d}, want {Ready 0}", green.Rollup, green.Findings)
	}

	// A yellow with no red -> Marginal rollup.
	yellow := buildServeReadiness(serveHostFacts{
		Arch: "amd64", ISA: "avx2", // yellow
		ModelBytes: 40 * testGiB, FreeBytes: 80 * testGiB, MemKnown: true, Headroom: 0.15,
		NUMANodes: 1,
	})
	if yellow.Rollup != "Marginal" {
		t.Errorf("rollup = %q, want Marginal", yellow.Rollup)
	}
	if yellow.Findings != 1 {
		t.Errorf("findings = %d, want 1", yellow.Findings)
	}
}

// TestRunServeDoctorJSONExit drives the CLI entrypoint end to end over an injected
// probe-free path (the live probe yields MemKnown=false, so a --model-bytes request
// lands on the "cannot verify" yellow row) and asserts the JSON shape and exit code.
func TestRunServeDoctorJSON(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runServeDoctor(&out, &errb, []string{"--json", "--model-bytes", "40000000000"})
	if rc != 0 {
		t.Fatalf("rc = %d (want 0; live probe cannot red a row), stderr=%s", rc, errb.String())
	}
	var rep serveReadinessReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\noutput=%s", err, out.String())
	}
	if len(rep.Rows) != 4 {
		t.Fatalf("rows = %d, want 4 (three host checks plus durability)", len(rep.Rows))
	}
	if rep.Rollup == "" {
		t.Error("rollup empty")
	}
	if rep.Durability == nil {
		t.Fatal("durability posture missing from serve doctor JSON")
	}
	hasDurability := false
	for _, row := range rep.Rows {
		if row.Check == "session-durability" {
			hasDurability = true
		}
	}
	if !hasDurability {
		t.Fatal("session-durability row missing from serve doctor JSON")
	}
	// model-fit must be the "cannot verify" yellow because the live host probe cannot
	// measure free device VRAM (MemKnown=false).
	var fit serveReadinessRow
	for _, r := range rep.Rows {
		if r.Check == "model-fit" {
			fit = r
		}
	}
	if fit.Status != sevWarn {
		t.Errorf("model-fit status = %q, want warn (mem not probeable), finding=%q", fit.Status, fit.Finding)
	}
}

// TestRunServeDoctorUsageError checks unexpected positional args return exit 2.
func TestRunServeDoctorUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runServeDoctor(&out, &errb, []string{"unexpected"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for unexpected args", rc)
	}
}
