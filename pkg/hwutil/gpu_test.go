package hwutil

// Fake-probe tests for the nvidia-smi CSV contract: pinned query output in
// the extended nine-field-plus style, including [N/A] placeholders that must
// stay null on the wire.

import (
	"context"
	"errors"
	"strings"
	"testing"
)

const fakeCoreCSV = "GPU-abc,NVIDIA RTX 5070 Ti,PCI:0000:01:00.0,16384,8192,87,245.5,2407,2610,1073741824,536870912,0\n"

// Multi-token throttle sets arrive as one quoted CSV field (the comma is
// inside the value), exactly as nvidia-smi emits them.
const fakeThrottleCSV = "\"HW Slowdown, SW Power Brake\"\n"

func installFakeSMI(t *testing.T, core, throttle string, fail bool) {
	t.Helper()
	orig := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = orig })
	runNvidiaSMI = func(ctx context.Context, args ...string) ([]byte, error) {
		if fail {
			return nil, errors.New("no driver")
		}
		joined := strings.Join(args, " ")
		switch {
		case strings.Contains(joined, "pcie.tx_throughput"):
			return []byte(core), nil
		case strings.Contains(joined, "clocks_event_reasons.active"):
			return nil, errors.New("unsupported key") // legacy spelling fails...
		case strings.Contains(joined, "clocks_throttle_reasons.active"):
			return []byte(throttle), nil // ...modern spelling answers.
		}
		return nil, errors.New("unexpected query: " + joined)
	}
}

func TestNvidiaGPUsFromFakeProbe(t *testing.T) {
	installFakeSMI(t, fakeCoreCSV, fakeThrottleCSV, false)
	gpus := nvidiaGPUs(context.Background())
	if len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	g := gpus[0]
	if g.UUID != "GPU-abc" || g.Name != "NVIDIA RTX 5070 Ti" || g.PCI != "0000:01:00.0" {
		t.Fatalf("identity = %+v", g)
	}
	if *g.MemTotalBytes != 16384*1024*1024 || *g.MemUsedBytes != 8192*1024*1024 {
		t.Fatalf("mem = %v / %v (MiB->bytes)", *g.MemTotalBytes, *g.MemUsedBytes)
	}
	if *g.UtilPct != 87 || *g.PowerW != 245.5 || *g.SMClockMHz != 2407 || *g.MaxSMClockMHz != 2610 {
		t.Fatalf("metrics = %+v", g)
	}
	if *g.PCIETxBytesPerS != 1073741824 || *g.PCIERxBytesPerS != 536870912 {
		t.Fatalf("pcie = %v / %v", *g.PCIETxBytesPerS, *g.PCIERxBytesPerS)
	}
	if *g.ECCUncorrectedTotal != 0 {
		t.Fatalf("ecc = %v", *g.ECCUncorrectedTotal)
	}
	if len(g.ThrottleReasons) != 2 || g.ThrottleReasons[0] != "HW Slowdown" || g.ThrottleReasons[1] != "SW Power Brake" {
		t.Fatalf("throttle = %v", g.ThrottleReasons)
	}
}

func TestNvidiaNAPlaceholdersStayNil(t *testing.T) {
	na := "GPU-x,NVIDIA T4,PCI:0000:00:0e.0,15360,[N/A],[N/A],[N/A s],[N/A],[N/A],[N/A],[N/A],[N/A]\n"
	installFakeSMI(t, na, "Not Active\n", false)
	gpus := nvidiaGPUs(context.Background())
	if len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	g := gpus[0]
	if g.MemUsedBytes != nil || g.UtilPct != nil || g.PowerW != nil || g.SMClockMHz != nil ||
		g.PCIETxBytesPerS != nil || g.ECCUncorrectedTotal != nil {
		t.Fatalf("[N/A] values fabricated: %+v", g)
	}
	if g.MemTotalBytes == nil || *g.MemTotalBytes != 15360*1024*1024 {
		t.Fatalf("readable total lost: %+v", g.MemTotalBytes)
	}
	if len(g.ThrottleReasons) != 0 {
		t.Fatalf("'Not Active' reported as reasons: %v", g.ThrottleReasons)
	}
}

func TestNvidiaGPUsAbsentWhenProbeFails(t *testing.T) {
	installFakeSMI(t, "", "", true)
	if gpus := nvidiaGPUs(context.Background()); len(gpus) != 0 {
		t.Fatalf("failing probe emitted rows: %+v", gpus)
	}
}

func TestNvidiaPercentRangeGuard(t *testing.T) {
	if v := nvidiaPercent("101"); v != nil {
		t.Fatalf("out-of-range percent accepted: %v", *v)
	}
	if v := nvidiaPercent("42.5"); v == nil || *v != 42.5 {
		t.Fatalf("valid percent rejected: %v", v)
	}
}
