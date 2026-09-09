package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/localadmission"
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

// TestRunServeDoctorJSON drives the CLI entrypoint end to end and asserts the JSON shape and exit code.
func TestRunServeDoctorJSON(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runServeDoctor(&out, &errb, []string{"--json", "--model-bytes", "1024"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	var rep serveReadinessReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\noutput=%s", err, out.String())
	}
	if len(rep.Rows) < 4 {
		t.Fatalf("rows = %d, want at least 4", len(rep.Rows))
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
	var fit serveReadinessRow
	foundFit := false
	for _, r := range rep.Rows {
		if r.Check == "model-fit" {
			fit = r
			foundFit = true
		}
	}
	if !foundFit {
		t.Fatal("model-fit row missing from serve doctor JSON")
	}
	if fit.Status == sevFail {
		t.Errorf("model-fit status = %q (must not fail for 1 KiB model), finding=%q", fit.Status, fit.Finding)
	}
}

// TestRunServeDoctorUsageError checks unexpected positional args return exit 2.
func TestRunServeDoctorUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runServeDoctor(&out, &errb, []string{"unexpected"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for unexpected args", rc)
	}
}

func TestServePressureRow(t *testing.T) {
	if row := servePressureRow(serveHostFacts{Pressure: ""}); row != nil {
		t.Fatalf("expected nil for empty pressure, got %+v", row)
	}

	crit := servePressureRow(serveHostFacts{
		Pressure:   string(localadmission.PressureCritical),
		TotalBytes: 64 * testGiB,
		CompBytes:  16 * testGiB,
		FreeBytes:  4 * testGiB,
	})
	if crit == nil || crit.Status != sevFail || crit.Tier != "Unready" {
		t.Fatalf("critical pressure: got %+v, want fail/Unready", crit)
	}
	if !strings.Contains(crit.Finding, "critical ambient memory pressure") {
		t.Errorf("critical finding missing expected text: %s", crit.Finding)
	}
	if !strings.Contains(crit.Remediation, "reboot, close memory-heavy") {
		t.Errorf("critical remediation missing expected text: %s", crit.Remediation)
	}

	warn := servePressureRow(serveHostFacts{
		Pressure:   string(localadmission.PressureWarning),
		TotalBytes: 64 * testGiB,
		CompBytes:  8 * testGiB,
		FreeBytes:  12 * testGiB,
	})
	if warn == nil || warn.Status != sevWarn || warn.Tier != "Marginal" {
		t.Fatalf("warning pressure: got %+v, want warn/Marginal", warn)
	}
	if !strings.Contains(warn.Finding, "warning ambient memory pressure") {
		t.Errorf("warning finding missing expected text: %s", warn.Finding)
	}
	if !strings.Contains(warn.Remediation, "close background applications") {
		t.Errorf("warning remediation missing expected text: %s", warn.Remediation)
	}

	norm := servePressureRow(serveHostFacts{
		Pressure:   string(localadmission.PressureNormal),
		TotalBytes: 64 * testGiB,
		FreeBytes:  48 * testGiB,
	})
	if norm == nil || norm.Status != sevOK || norm.Tier != "Ready" {
		t.Fatalf("normal pressure: got %+v, want ok/Ready", norm)
	}
	if !strings.Contains(norm.Finding, "normal — nominal pressure") {
		t.Errorf("normal finding missing expected text: %s", norm.Finding)
	}
}

func TestServeReservationsRow(t *testing.T) {
	if row := serveReservationsRow(serveHostFacts{TotalBytes: 0, ReservedBytes: 0, GPULeaseHeld: false}); row != nil {
		t.Fatalf("expected nil when no memory/lease/reservation facts, got %+v", row)
	}

	locked := serveReservationsRow(serveHostFacts{
		TotalBytes:   36 * testGiB,
		GPULeaseHeld: true,
		GPULeasePath: "/tmp/fak-gpu.lease",
	})
	if locked == nil || locked.Status != sevWarn || locked.Tier != "Marginal" {
		t.Fatalf("locked lease: got %+v, want warn/Marginal", locked)
	}
	if !strings.Contains(locked.Finding, "Metal GPU residency lease is currently locked") {
		t.Errorf("locked finding missing expected text: %s", locked.Finding)
	}
	if !strings.Contains(locked.Remediation, "wait for the active serve process") {
		t.Errorf("locked remediation missing expected text: %s", locked.Remediation)
	}

	res := serveReservationsRow(serveHostFacts{
		TotalBytes:    36 * testGiB,
		ReservedBytes: 10 * testGiB,
	})
	if res == nil || res.Status != sevWarn || res.Tier != "Marginal" {
		t.Fatalf("active reservations: got %+v, want warn/Marginal", res)
	}
	if !strings.Contains(res.Finding, "10.0 GiB active local memory reservation") {
		t.Errorf("reservation finding missing expected text: %s", res.Finding)
	}

	clean := serveReservationsRow(serveHostFacts{
		TotalBytes: 36 * testGiB,
	})
	if clean == nil || clean.Status != sevOK || clean.Tier != "Ready" {
		t.Fatalf("clean reservations: got %+v, want ok/Ready", clean)
	}
	if !strings.Contains(clean.Finding, "zero active local memory reservations") {
		t.Errorf("clean finding missing expected text: %s", clean.Finding)
	}
}

func TestResolveDoctorTargetModelPreflight(t *testing.T) {
	gib := float64(int64(1) << 30)

	// Test aliases (which may resolve from local cache or fallback)
	var f1 serveHostFacts
	resolveDoctorTargetModel(&f1, "qwen38", "")
	if f1.ModelName != "qwen38" {
		t.Errorf("model name = %q, want qwen38", f1.ModelName)
	}
	if f1.ModelArm != "resident-q4k" {
		t.Errorf("model arm = %q, want resident-q4k", f1.ModelArm)
	}
	if f1.ModelBytes < 8*testGiB || f1.ModelBytes > 11*testGiB {
		t.Errorf("model bytes = %d, want ~9.12 GiB", f1.ModelBytes)
	}

	var f2 serveHostFacts
	resolveDoctorTargetModel(&f2, "qwen38:27b-q4", "")
	if f2.ModelBytes < 15*testGiB || f2.ModelBytes > 18*testGiB {
		t.Errorf("model bytes = %d, want ~16.3 GiB", f2.ModelBytes)
	}

	var f3 serveHostFacts
	resolveDoctorTargetModel(&f3, "qwen2.5-coder:7b", "")
	if f3.ModelBytes < 4*testGiB || f3.ModelBytes > 6*testGiB {
		t.Errorf("model bytes = %d, want ~4.5 GiB", f3.ModelBytes)
	}

	// Test uncached fallback resolution directly
	var fb1 serveHostFacts
	resolveDoctorTargetModel(&fb1, "nonexistent-model/qwen38", "")
	if fb1.ModelBytes != int64(9.12*gib) {
		t.Errorf("uncached qwen38 bytes = %d, want %d", fb1.ModelBytes, int64(9.12*gib))
	}

	var fb2 serveHostFacts
	resolveDoctorTargetModel(&fb2, "nonexistent-model/qwen38:27b-q4", "")
	if fb2.ModelBytes != int64(16.3*gib) {
		t.Errorf("uncached qwen38:27b-q4 bytes = %d, want %d", fb2.ModelBytes, int64(16.3*gib))
	}

	var fb3 serveHostFacts
	resolveDoctorTargetModel(&fb3, "nonexistent-model/qwen2.5-coder:7b", "")
	if fb3.ModelBytes != int64(4.5*gib) {
		t.Errorf("uncached qwen2.5-coder:7b bytes = %d, want %d", fb3.ModelBytes, int64(4.5*gib))
	}

	fitRow := serveFitRow(serveHostFacts{
		ModelName:  "qwen38",
		ModelArm:   "resident-q4k",
		ModelBytes: f1.ModelBytes,
		FreeBytes:  32 * testGiB,
		MemKnown:   true,
		Headroom:   0.15,
	})
	if fitRow.Status != sevOK {
		t.Errorf("fitRow status = %q, want ok", fitRow.Status)
	}
	if !strings.Contains(fitRow.Finding, `model "qwen38" (`) || !strings.Contains(fitRow.Finding, `[resident-q4k])`) {
		t.Errorf("fitRow finding does not format model name and arm: %s", fitRow.Finding)
	}
}

func TestRunServeDoctorModelFlag(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runServeDoctor(&out, &errb, []string{"--json", "--model", "qwen38"})
	if rc != 0 && rc != 1 {
		t.Fatalf("unexpected rc = %d, stderr=%s", rc, errb.String())
	}
	var rep serveReadinessReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\noutput=%s", err, out.String())
	}
	if rep.Facts.ModelName != "qwen38" {
		t.Errorf("facts.ModelName = %q, want qwen38", rep.Facts.ModelName)
	}
	if rep.Facts.ModelBytes == 0 {
		t.Errorf("facts.ModelBytes should be resolved from model flag, got 0")
	}
	var fitRow *serveReadinessRow
	for _, r := range rep.Rows {
		if r.Check == "model-fit" {
			fitRow = &r
			break
		}
	}
	if fitRow == nil {
		t.Fatal("missing model-fit row")
	}
	if !strings.Contains(fitRow.Finding, `model "qwen38"`) {
		t.Errorf("finding %q missing model name", fitRow.Finding)
	}
}
