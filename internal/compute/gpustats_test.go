package compute

import (
	"context"
	"errors"
	"testing"
)

func TestParseNvidiaSMIStats(t *testing.T) {
	// Two devices, MiB memory + integer utilization, exactly as
	// `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu
	//  --format=csv,noheader,nounits` prints them.
	const out = "0, 1024, 40960, 37\n1, 2048, 40960, 91\n"
	stats, ok := parseNvidiaSMIStats(out)
	if !ok || len(stats) != 2 {
		t.Fatalf("parse ok=%v n=%d, want true/2", ok, len(stats))
	}
	if stats[0].Index != 0 || stats[0].VRAMUsedBytes != 1024<<20 || stats[0].VRAMTotalBytes != 40960<<20 || stats[0].UtilizationPct != 37 {
		t.Errorf("device 0 = %+v, want idx0 used1024MiB total40960MiB util37", stats[0])
	}
	if stats[1].Index != 1 || stats[1].UtilizationPct != 91 {
		t.Errorf("device 1 = %+v, want idx1 util91", stats[1])
	}

	used, total, util, aok := AggregateGPUStats(stats)
	if !aok || used != (1024+2048)<<20 || total != (40960*2)<<20 || util != 91 {
		t.Errorf("aggregate = used %d total %d util %g ok %v, want summed VRAM + max util 91", used, total, util, aok)
	}
}

func TestParseNvidiaSMIStatsSkipsUnreadableRows(t *testing.T) {
	// A MIG/partitioned device can report [N/A] for utilization; that row is
	// skipped, not fabricated as 0. A blank line is ignored.
	const out = "0, 8000, 24000, 55\n1, [N/A], [N/A], [N/A]\n\n"
	stats, ok := parseNvidiaSMIStats(out)
	if !ok || len(stats) != 1 || stats[0].UtilizationPct != 55 {
		t.Fatalf("parse = %+v ok=%v, want the one readable device only", stats, ok)
	}
	// Empty / all-unreadable input reports not-present rather than a bogus zero device.
	if _, ok := parseNvidiaSMIStats("1, [N/A], [N/A], [N/A]\n"); ok {
		t.Errorf("all-unreadable input must report ok=false")
	}
	if _, _, _, ok := AggregateGPUStats(nil); ok {
		t.Errorf("AggregateGPUStats(nil) must report ok=false")
	}
}

func TestHarnessGPUVRAM(t *testing.T) {
	const mib = 1 << 20
	smi := []GPUStat{{Index: 0, VRAMUsedBytes: 6 << 30, VRAMTotalBytes: 24 << 30, UtilizationPct: 73}}

	// Device handle known → preferred; used = total - free, smi ignored.
	if used, total, ok := HarnessGPUVRAM(24<<30, 18<<30, true, smi); !ok || used != 6<<30 || total != 24<<30 {
		t.Errorf("device-known = used %d total %d ok %v, want 6GiB/24GiB from the handle", used, total, ok)
	}
	// Device handle unknown → fall back to the nvidia-smi aggregate.
	if used, total, ok := HarnessGPUVRAM(0, 0, false, smi); !ok || used != 6<<30 || total != 24<<30 {
		t.Errorf("device-unknown = used %d total %d ok %v, want the smi aggregate", used, total, ok)
	}
	// Neither source usable → honest absence.
	if _, _, ok := HarnessGPUVRAM(0, 0, false, nil); ok {
		t.Errorf("no device + no smi must report ok=false (axis stays n/a)")
	}
	// A transient free>total (or negative free) clamps used into [0,total], never underflows.
	if used, total, ok := HarnessGPUVRAM(8*mib, 9*mib, true, nil); !ok || used != 0 || total != 8*mib {
		t.Errorf("free>total = used %d total %d ok %v, want used clamped to 0", used, total, ok)
	}
	if used, _, ok := HarnessGPUVRAM(8*mib, -1, true, nil); !ok || used != 0 {
		t.Errorf("negative free = used %d ok %v, want used clamped to 0", used, ok)
	}
}

func TestNvidiaGPUStatsFailSoft(t *testing.T) {
	// No nvidia-smi (or any exec error) => present=false, honest n/a — never a fabricated 0.
	if _, present := nvidiaGPUStats(func(ctx context.Context, args ...string) (string, error) {
		return "", errors.New("exec: nvidia-smi not found")
	}); present {
		t.Errorf("an nvidia-smi error must yield present=false")
	}
	// A healthy run folds through to present=true.
	stats, present := nvidiaGPUStats(func(ctx context.Context, args ...string) (string, error) {
		return "0, 512, 16384, 12\n", nil
	})
	if !present || len(stats) != 1 || stats[0].UtilizationPct != 12 {
		t.Errorf("healthy run = %+v present=%v, want one device util 12", stats, present)
	}
}
