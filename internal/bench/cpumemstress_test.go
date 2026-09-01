package bench

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestResolveSystemTestUtilization_ProfilesAndCPUCounts(t *testing.T) {
	tests := []struct {
		name       string
		profile    SystemTestUtilizationProfile
		logicalCPU int
		wantCPU    int
		wantPct    int
	}{
		{name: "shared one CPU", profile: SystemTestProfileSharedDevelopment, logicalCPU: 1, wantCPU: 1, wantPct: 80},
		{name: "shared two CPUs", profile: SystemTestProfileSharedDevelopment, logicalCPU: 2, wantCPU: 1, wantPct: 80},
		{name: "shared larger host", profile: SystemTestProfileSharedDevelopment, logicalCPU: 8, wantCPU: 6, wantPct: 80},
		{name: "remote one CPU", profile: SystemTestProfileRemote, logicalCPU: 1, wantCPU: 1, wantPct: 90},
		{name: "remote two CPUs", profile: SystemTestProfileRemote, logicalCPU: 2, wantCPU: 1, wantPct: 90},
		{name: "remote larger host", profile: SystemTestProfileRemote, logicalCPU: 8, wantCPU: 7, wantPct: 90},
		{name: "dedicated one CPU", profile: SystemTestProfileDedicatedInference, logicalCPU: 1, wantCPU: 1, wantPct: 100},
		{name: "dedicated two CPUs", profile: SystemTestProfileDedicatedInference, logicalCPU: 2, wantCPU: 2, wantPct: 100},
		{name: "dedicated larger host", profile: SystemTestProfileDedicatedInference, logicalCPU: 8, wantCPU: 8, wantPct: 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveSystemTestUtilization(SystemTestUtilizationRequest{Profile: tt.profile}, tt.logicalCPU)
			if err != nil {
				t.Fatalf("ResolveSystemTestUtilization: %v", err)
			}
			if got.Profile != tt.profile || got.CPUPercent != tt.wantPct || got.GPUPercent != tt.wantPct || got.CPUCapacity != tt.wantCPU {
				t.Fatalf("resolved = %+v, want profile=%q cpu_percent=%d gpu_percent=%d capacity=%d", got, tt.profile, tt.wantPct, tt.wantPct, tt.wantCPU)
			}
			if !strings.Contains(got.GPULimitation, "does not enforce or measure GPU") {
				t.Errorf("GPU limitation is not honest: %q", got.GPULimitation)
			}
		})
	}
}

func TestResolveSystemTestUtilization_OverridesAndValidation(t *testing.T) {
	tests := []struct {
		name      string
		req       SystemTestUtilizationRequest
		cpus      int
		want      SystemTestUtilization
		wantError string
	}{
		{name: "zero profile selects shared", cpus: 8, want: SystemTestUtilization{Profile: SystemTestProfileSharedDevelopment, CPUPercent: 80, GPUPercent: 80, CPUCapacity: 6}},
		{name: "explicit percentages", req: SystemTestUtilizationRequest{Profile: SystemTestProfileRemote, CPUPercent: 50, GPUPercent: 35}, cpus: 8, want: SystemTestUtilization{Profile: SystemTestProfileRemote, CPUPercent: 50, GPUPercent: 35, CPUCapacity: 4}},
		{name: "percentages clamp above 100", req: SystemTestUtilizationRequest{CPUPercent: 101, GPUPercent: 1000}, cpus: 8, want: SystemTestUtilization{Profile: SystemTestProfileSharedDevelopment, CPUPercent: 100, GPUPercent: 100, CPUCapacity: 7}},
		{name: "exact concurrency wins last", req: SystemTestUtilizationRequest{Profile: SystemTestProfileSharedDevelopment, CPUPercent: 25, Concurrency: 11}, cpus: 8, want: SystemTestUtilization{Profile: SystemTestProfileSharedDevelopment, CPUPercent: 25, GPUPercent: 80, CPUCapacity: 11}},
		{name: "unknown profile", req: SystemTestUtilizationRequest{Profile: "workstation"}, cpus: 8, wantError: "unknown system-test utilization profile"},
		{name: "negative CPU percent", req: SystemTestUtilizationRequest{CPUPercent: -1}, cpus: 8, wantError: "CPU percent must not be negative"},
		{name: "negative GPU percent", req: SystemTestUtilizationRequest{GPUPercent: -1}, cpus: 8, wantError: "GPU percent must not be negative"},
		{name: "negative concurrency", req: SystemTestUtilizationRequest{Concurrency: -1}, cpus: 8, wantError: "concurrency must not be negative"},
		{name: "invalid CPU count", cpus: 0, wantError: "logical CPU count must be positive"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveSystemTestUtilization(tt.req, tt.cpus)
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("error = %v, want containing %q", err, tt.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveSystemTestUtilization: %v", err)
			}
			if got.Profile != tt.want.Profile || got.CPUPercent != tt.want.CPUPercent || got.GPUPercent != tt.want.GPUPercent || got.CPUCapacity != tt.want.CPUCapacity {
				t.Fatalf("resolved = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// #4371 smoke: a tiny bounded run exercises both arms, records the required
// witness fields (duration, concurrency, alloc size, throughput, latency
// distribution, host before/after, digest), and the artifact self-verifies. Sized
// down so CI stays fast and allocates only a few MiB.
func TestRunCPUMemStress_Smoke(t *testing.T) {
	cfg := CPUMemConfig{
		CPUPercent:  1, // exercise the profile ceiling without saturating the host
		CPUDuration: 60 * time.Millisecond,
		BlockBytes:  64 << 10, // 64 KiB blocks — many ops even in 60ms
		AllocSizes:  []int64{4 << 20, 16 << 20},
		ChunkBytes:  4 << 20,
	}
	rep, err := RunCPUMemStress(context.Background(), cfg)
	if err != nil {
		t.Fatalf("RunCPUMemStress: %v", err)
	}
	if rep.Schema != CPUMemSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, CPUMemSchema)
	}
	if rep.Refused {
		t.Fatalf("smoke run refused unexpectedly: %s", rep.RefusedWhy)
	}

	// CPU arm: the resolved shared-development ceiling is reported and used.
	if rep.Utilization.Profile != SystemTestProfileSharedDevelopment || rep.Utilization.CPUPercent != 1 || rep.Utilization.GPUPercent != 80 {
		t.Fatalf("resolved utilization = %+v", rep.Utilization)
	}
	if rep.CPU.Workers != rep.Utilization.CPUCapacity || rep.Config.Concurrency != rep.Utilization.CPUCapacity {
		t.Errorf("workers/config/resolved mismatch: workers=%d config=%d resolved=%d", rep.CPU.Workers, rep.Config.Concurrency, rep.Utilization.CPUCapacity)
	}
	if rep.NumCPU > 1 && rep.CPU.Workers >= rep.NumCPU {
		t.Errorf("smoke saturated host: workers=%d logical CPUs=%d", rep.CPU.Workers, rep.NumCPU)
	}
	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	for _, field := range []string{`"profile":"shared-development"`, `"cpu_percent":1`, `"gpu_percent":80`, `"gpu_limitation":`} {
		if !strings.Contains(string(raw), field) {
			t.Errorf("JSON report missing resolved utilization field %s: %s", field, raw)
		}
	}
	if rep.CPU.Ops <= 0 {
		t.Fatalf("cpu ops = %d, want > 0 (no hashing happened)", rep.CPU.Ops)
	}
	if rep.CPU.HashesPerSec <= 0 || rep.CPU.ThroughputMBs <= 0 {
		t.Errorf("cpu throughput not derived: hashes/s=%.2f MiB/s=%.2f", rep.CPU.HashesPerSec, rep.CPU.ThroughputMBs)
	}
	if rep.CPU.DurationNS <= 0 {
		t.Errorf("cpu duration = %d, want > 0", rep.CPU.DurationNS)
	}
	// Latency distribution ordering only where the clock can resolve it.
	if !coarseClock() {
		if rep.CPU.P50NS <= 0 || rep.CPU.P99NS < rep.CPU.P50NS || rep.CPU.MaxNS < rep.CPU.P99NS {
			t.Errorf("cpu distribution not ordered: p50=%d p99=%d max=%d", rep.CPU.P50NS, rep.CPU.P99NS, rep.CPU.MaxNS)
		}
	} else {
		t.Log("coarse monotonic clock: cpu distribution ordering not asserted on this host")
	}

	// Memory arms: both slabs touched, alloc size recorded, throughput derived.
	if len(rep.Memory) != 2 {
		t.Fatalf("memory arms = %d, want 2", len(rep.Memory))
	}
	for _, m := range rep.Memory {
		if m.Skipped {
			t.Errorf("slab %d unexpectedly skipped: %s", m.RequestedBytes, m.SkipReason)
			continue
		}
		if m.TouchedBytes <= 0 || m.Pages <= 0 {
			t.Errorf("slab %d not touched: touched=%d pages=%d", m.RequestedBytes, m.TouchedBytes, m.Pages)
		}
		if m.TouchedBytes > m.RequestedBytes {
			t.Errorf("slab %d touched %d > requested %d", m.RequestedBytes, m.TouchedBytes, m.RequestedBytes)
		}
	}

	// Host before/after both captured (runtime fields are always present).
	if rep.HostBefore.HeapSysBytes == 0 || rep.HostAfter.HeapSysBytes == 0 {
		t.Errorf("host snapshots missing: before=%+v after=%+v", rep.HostBefore, rep.HostAfter)
	}

	// Fence + stress-ng guidance travel with the artifact.
	if !strings.Contains(rep.Fence, "NO gate asserted") {
		t.Errorf("fence must travel with the artifact; got %q", rep.Fence)
	}
	if !strings.Contains(rep.StressNGWhen, "stress-ng") {
		t.Errorf("stress-ng guidance must travel with the artifact; got %q", rep.StressNGWhen)
	}

	// The digest is a valid, self-consistent SHA-256 of the report.
	assertDigestSelfConsistent(t, rep)
}

// The load gate must REFUSE a busy host before applying any load, and the refusal
// is a first-class recorded outcome (Refused=true + error), not a silent skip.
func TestCPUMemStress_LoadGateRefuses(t *testing.T) {
	if coarseClock() || readLoadAvgUnavailable() {
		t.Skip("no /proc/loadavg on this host: load gate is unchecked here (verified structurally on Linux/da33)")
	}
	// MaxLoadPerCPU: -0 is impossible; use a tiny positive limit so any real
	// 1-min load per CPU trips it.
	cfg := CPUMemConfig{
		Concurrency:   1,
		CPUDuration:   20 * time.Millisecond,
		AllocSizes:    []int64{1 << 20},
		MaxLoadPerCPU: 0.0000001,
	}
	rep, err := RunCPUMemStress(context.Background(), cfg)
	if err == nil {
		t.Fatalf("expected refusal error from load gate, got nil (load1=%.2f)", rep.HostBefore.Load1)
	}
	if !rep.Refused || !rep.Gates.Load.Tripped {
		t.Errorf("expected Refused+load gate tripped; got refused=%v load=%+v", rep.Refused, rep.Gates.Load)
	}
	if rep.CPU.Ops != 0 {
		t.Errorf("refused run still did %d hash ops — load was applied despite refusal", rep.CPU.Ops)
	}
	// Even a refused run must digest itself (the witness records the refusal).
	assertDigestSelfConsistent(t, rep)
}

// The memory cap must SKIP an oversized slab rather than allocate it (OOM guard),
// and record why. Uses a tiny MaxMemFraction so any real slab exceeds the cap on
// a host where MemAvailable is readable.
func TestCPUMemStress_MemoryCapSkips(t *testing.T) {
	if readMemInfoUnavailable() {
		t.Skip("no /proc/meminfo on this host: memory cap is derived from MemAvailable (verified on Linux/da33)")
	}
	cfg := CPUMemConfig{
		Concurrency:    1,
		CPUDuration:    20 * time.Millisecond,
		AllocSizes:     []int64{1 << 20},
		MaxMemFraction: 1e-12, // cap ~0 bytes → the 1 MiB slab must be skipped
	}
	rep, err := RunCPUMemStress(context.Background(), cfg)
	if err != nil {
		t.Fatalf("RunCPUMemStress: %v", err)
	}
	if len(rep.Memory) != 1 || !rep.Memory[0].Skipped {
		t.Fatalf("expected the slab skipped by the memory cap; got %+v", rep.Memory)
	}
	if !strings.Contains(rep.Memory[0].SkipReason, "memory cap") {
		t.Errorf("skip reason should cite the memory cap; got %q", rep.Memory[0].SkipReason)
	}
}

// The percentile fold is exact over raw samples (no bucket quantization).
func TestPercentiles_Exact(t *testing.T) {
	samples := make([]int64, 1000)
	for i := range samples {
		samples[i] = int64(i + 1) // 1..1000
	}
	p50, p90, p99, max, mean := percentiles(samples)
	if p50 != 501 || p90 != 901 || p99 != 991 || max != 1000 {
		t.Errorf("percentiles = p50 %d p90 %d p99 %d max %d, want 501/901/991/1000", p50, p90, p99, max)
	}
	if mean != 500.5 {
		t.Errorf("mean = %.2f, want 500.5", mean)
	}
	if p50e, _, _, _, _ := percentiles(nil); p50e != 0 {
		t.Errorf("empty input p50 = %d, want 0", p50e)
	}
}

// Artifact writer, env-gated: FAK_CPUMEM_OUT=<path> runs the full-size arm and
// writes the witness JSON (this is how #4371 is run on da33 and the result
// attached to #4367). Set FAK_BENCH_HW to name the host in the artifact.
func TestWriteCPUMemArtifact(t *testing.T) {
	out := os.Getenv("FAK_CPUMEM_OUT")
	if out == "" {
		t.Skip("set FAK_CPUMEM_OUT=<path> to run the full CPU+memory stress arm and write the witness artifact")
	}
	rep, err := RunCPUMemStress(context.Background(), CPUMemConfig{})
	if err != nil {
		// A gate refusal is a legitimate, recordable outcome — still write it.
		if !rep.Refused {
			t.Fatalf("RunCPUMemStress: %v", err)
		}
		t.Logf("run refused by safety gate: %s", rep.RefusedWhy)
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("cpu+mem stress witness written to %s (hash %.0f MiB/s, digest %s…)",
		out, rep.CPU.ThroughputMBs, firstN(rep.Digest, 12))
}

func assertDigestSelfConsistent(t *testing.T, rep *CPUMemReport) {
	t.Helper()
	if len(rep.Digest) != 64 {
		t.Fatalf("digest %q is not a 64-hex SHA-256", rep.Digest)
	}
	if _, err := hex.DecodeString(rep.Digest); err != nil {
		t.Fatalf("digest not hex: %v", err)
	}
	if got := rep.computeDigest(); got != rep.Digest {
		t.Errorf("digest not self-consistent: stored %s, recomputed %s", rep.Digest, got)
	}
}

func readLoadAvgUnavailable() bool {
	_, _, _, ok := readLoadAvg()
	return !ok
}

func readMemInfoUnavailable() bool {
	_, _, _, ok := readMemInfo()
	return !ok
}

func firstN(s string, n int) string {
	if len(s) < n {
		return s
	}
	return s[:n]
}
