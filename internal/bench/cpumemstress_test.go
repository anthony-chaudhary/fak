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

// #4371 smoke: a tiny bounded run exercises both arms, records the required
// witness fields (duration, concurrency, alloc size, throughput, latency
// distribution, host before/after, digest), and the artifact self-verifies. Sized
// down so CI stays fast and allocates only a few MiB.
func TestRunCPUMemStress_Smoke(t *testing.T) {
	cfg := CPUMemConfig{
		Concurrency: 2,
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

	// CPU arm: concurrency echoed, work actually happened, throughput derived.
	if rep.CPU.Workers != 2 {
		t.Errorf("cpu workers = %d, want 2", rep.CPU.Workers)
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
