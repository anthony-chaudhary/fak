package harnessres

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestNetAndGPUProvidersFold(t *testing.T) {
	clock := time.Unix(0, 0)
	s := newSampler(func() time.Time { return clock })
	s.SetNetworkProvider(func() (rx, tx uint64, ok bool) { return 4 << 20, 1 << 20, true })
	s.SetGPUProvider(func() (used, total uint64, ok bool) { return 6 << 30, 24 << 30, true })
	s.foldProc(procSample{haveCPU: true, cpuUser: time.Second}, clock, 3, 0)

	snap := s.Snapshot()
	if !snap.Kernel.HaveNet || snap.Kernel.NetRxBytes != 4<<20 || snap.Kernel.NetTxBytes != 1<<20 {
		t.Fatalf("net = rx %d tx %d (have=%v), want 4MiB/1MiB", snap.Kernel.NetRxBytes, snap.Kernel.NetTxBytes, snap.Kernel.HaveNet)
	}
	if !snap.HaveGPU || snap.GPUVRAMUsedBytes != 6<<30 || snap.GPUVRAMTotalBytes != 24<<30 {
		t.Fatalf("gpu = used %d total %d (have=%v), want 6/24 GiB", snap.GPUVRAMUsedBytes, snap.GPUVRAMTotalBytes, snap.HaveGPU)
	}

	rep := snap.Report()
	if !strings.Contains(rep, "net rx/tx 4.0 MiB/1.0 MiB") {
		t.Errorf("Report missing net: %s", rep)
	}
	if !strings.Contains(rep, "gpu vram 6.0 GiB/24.0 GiB") {
		t.Errorf("Report missing gpu: %s", rep)
	}
	prom := snap.PrometheusText()
	if !strings.Contains(prom, `fak_harness_net_bytes_total{half="kernel",dir="rx"}`) {
		t.Errorf("Prometheus missing net: %s", prom)
	}
	if !strings.Contains(prom, `fak_harness_gpu_vram_bytes{kind="used"}`) {
		t.Errorf("Prometheus missing gpu: %s", prom)
	}

	b, err := snap.MarshalLedgerRow("guard", "openai", "codex", time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["gpu_vram_used_bytes"]; !ok {
		t.Errorf("ledger missing gpu_vram_used_bytes: %s", b)
	}
	kernel := m["kernel"].(map[string]any)
	if _, ok := kernel["net_rx_bytes"]; !ok {
		t.Errorf("ledger kernel missing net_rx_bytes: %s", b)
	}

	// A proxy-path snapshot (no net/gpu observed) omits the axes rather than reporting 0.
	bare, _ := (Snapshot{NumCPU: 1}).MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1, 0))
	if strings.Contains(string(bare), "gpu_vram") || strings.Contains(string(bare), "net_rx") {
		t.Errorf("proxy-path snapshot must omit net/gpu: %s", bare)
	}
}
