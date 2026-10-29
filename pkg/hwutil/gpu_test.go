package hwutil

// Fake-probe tests for the nvidia-smi CSV contract: pinned query output in
// the driver-generation-stable core-field style (no pcie.*_throughput —
// NVML/DCGM-only counters), including [N/A] placeholders that must stay null
// on the wire, the reduced-set fallback retry, and fixture-driven AMD sysfs
// cases over a fake /sys/class/drm tree.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fakeCoreCSV = "GPU-abc,NVIDIA RTX 5070 Ti,PCI:0000:01:00.0,16384,8192,87,245.5,2407,2610,0\n"

// Multi-token throttle sets arrive as one quoted CSV field (the comma is
// inside the value), exactly as nvidia-smi emits them.
const fakeThrottleCSV = "\"HW Slowdown, SW Power Brake\"\n"

// installFakeSMI pins the nvidia-smi seam; when out is non-nil, every argv
// it sees is appended there so tests can assert the exact query contract.
func installFakeSMI(t *testing.T, core, throttle string, fail bool, out *[][]string) {
	t.Helper()
	orig := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = orig })
	runNvidiaSMI = func(ctx context.Context, args ...string) ([]byte, error) {
		if out != nil {
			*out = append(*out, args)
		}
		if fail {
			return nil, errors.New("no driver")
		}
		joined := strings.Join(args, " ")
		switch {
		case strings.Contains(joined, "--query-gpu="+nvidiaCoreFields):
			return []byte(core), nil
		case strings.Contains(joined, "clocks_event_reasons.active"):
			return nil, errors.New("unsupported key") // legacy spelling fails...
		case strings.Contains(joined, "clocks_throttle_reasons.active"):
			return []byte(throttle), nil // ...modern spelling answers.
		}
		return nil, errors.New("unexpected query: " + joined)
	}
}

// TestNvidiaQueryArgvIsStableCoreSet pins the wire contract: the first
// nvidia-smi call carries exactly the stable core field set, and no call may
// ever carry pcie.*_throughput (invalid nvidia-smi fields that zero the whole
// NVIDIA surface on current drivers).
func TestNvidiaQueryArgvIsStableCoreSet(t *testing.T) {
	var calls [][]string
	installFakeSMI(t, fakeCoreCSV, fakeThrottleCSV, false, &calls)
	if gpus := nvidiaGPUs(context.Background()); len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	if len(calls) == 0 {
		t.Fatal("nvidia-smi never invoked")
	}
	want := []string{"--query-gpu=" + nvidiaCoreFields, "--format=csv,noheader,nounits"}
	if got := calls[0]; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("first argv = %v, want %v", got, want)
	}
	for _, c := range calls {
		joined := strings.Join(c, " ")
		if strings.Contains(joined, "pcie.tx_throughput") || strings.Contains(joined, "pcie.rx_throughput") {
			t.Fatalf("pcie.*_throughput queried via nvidia-smi (NVML/DCGM-only): %v", c)
		}
	}
}

func TestNvidiaGPUsFromFakeProbe(t *testing.T) {
	installFakeSMI(t, fakeCoreCSV, fakeThrottleCSV, false, nil)
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
	if g.PCIETxBytesPerS != nil || g.PCIERxBytesPerS != nil {
		t.Fatalf("pcie populated from nvidia-smi (NVML/DCGM-only counters): %+v", g)
	}
	if *g.ECCUncorrectedTotal != 0 {
		t.Fatalf("ecc = %v", *g.ECCUncorrectedTotal)
	}
	if len(g.ThrottleReasons) != 2 || g.ThrottleReasons[0] != "HW Slowdown" || g.ThrottleReasons[1] != "SW Power Brake" {
		t.Fatalf("throttle = %v", g.ThrottleReasons)
	}
}

// TestNvidiaFallbackRetryOnFieldRename pins the resilience contract: a
// driver-era field rename that breaks the core query triggers exactly one
// retry with the reduced essential set, and the reduced row still lands with
// its absent fields left nil.
func TestNvidiaFallbackRetryOnFieldRename(t *testing.T) {
	orig := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = orig })
	mainCalls := 0
	runNvidiaSMI = func(ctx context.Context, args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		switch {
		case strings.Contains(joined, "--query-gpu="+nvidiaCoreFields):
			mainCalls++
			return nil, errors.New(`Field "clocks.current.sm" is not a valid field to query.`)
		case strings.Contains(joined, "--query-gpu="+nvidiaEssentialFields):
			mainCalls++
			return []byte("GPU-fb,NVIDIA RTX 5090 Laptop GPU,PCI:0000:01:00.0,16384,4096,12,45.25\n"), nil
		case strings.Contains(joined, "clocks_event_reasons.active"),
			strings.Contains(joined, "clocks_throttle_reasons.active"):
			return []byte("Not Active\n"), nil
		}
		return nil, errors.New("unexpected query: " + joined)
	}
	gpus := nvidiaGPUs(context.Background())
	if mainCalls != 2 {
		t.Fatalf("main-query invocations = %d, want exactly 2 (core + one reduced retry)", mainCalls)
	}
	if len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	g := gpus[0]
	if g.UUID != "GPU-fb" || g.Name != "NVIDIA RTX 5090 Laptop GPU" || g.PCI != "0000:01:00.0" {
		t.Fatalf("identity = %+v", g)
	}
	if *g.MemTotalBytes != 16384*1024*1024 || *g.MemUsedBytes != 4096*1024*1024 || *g.UtilPct != 12 || *g.PowerW != 45.25 {
		t.Fatalf("metrics = %+v", g)
	}
	if g.SMClockMHz != nil || g.MaxSMClockMHz != nil || g.ECCUncorrectedTotal != nil {
		t.Fatalf("reduced-set row fabricated absent fields: %+v", g)
	}
}

func TestNvidiaNAPlaceholdersStayNil(t *testing.T) {
	na := "GPU-x,NVIDIA T4,PCI:0000:00:0e.0,15360,[N/A],[N/A],[N/A s],[N/A],[N/A],[N/A]\n"
	installFakeSMI(t, na, "Not Active\n", false, nil)
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
	installFakeSMI(t, "", "", true, nil)
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

// amdCard is one fixture card under the fake /sys/class/drm root.
type amdCard struct {
	name  string // "card0"
	pci   string // PCI BDF the device symlink resolves to (base of the target)
	files map[string]string
}

// amdPciDir renders a PCI BDF as a directory name legal on every OS
// (Windows forbids ':' in names); the device symlink's base stays the
// sanitized form, which is what the collector reports as pci.
func amdPciDir(bdf string) string {
	return strings.NewReplacer(":", "_", ".", "_").Replace(bdf)
}

// amdFixture builds a fake /sys/class/drm tree: each card gets a device
// symlink into pci/<dir> holding its metric files; connectors point back at
// their card kobject dir, exactly as sysfs does.
func amdFixture(t *testing.T, cards []amdCard, connectors []string) string {
	t.Helper()
	root := t.TempDir()
	for _, c := range cards {
		devDir := filepath.Join(root, "pci", amdPciDir(c.pci))
		if err := os.MkdirAll(devDir, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, content := range c.files {
			if err := os.WriteFile(filepath.Join(devDir, name), []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		cardDir := filepath.Join(root, c.name)
		if err := os.MkdirAll(cardDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join("..", "pci", amdPciDir(c.pci)), filepath.Join(cardDir, "device")); err != nil {
			t.Skipf("symlinks unavailable: %v", err)
		}
	}
	for _, conn := range connectors {
		connDir := filepath.Join(root, conn)
		if err := os.MkdirAll(connDir, 0o755); err != nil {
			t.Fatal(err)
		}
		cardName := strings.SplitN(conn, "-", 2)[0]
		if err := os.Symlink(".."+string(os.PathSeparator)+cardName, filepath.Join(connDir, "device")); err != nil {
			t.Skipf("symlinks unavailable: %v", err)
		}
	}
	return root
}

func installDRMRoot(t *testing.T, root string) {
	t.Helper()
	orig := drmRootDir
	t.Cleanup(func() { drmRootDir = orig })
	drmRootDir = root
}

// TestAMDNoMetricsEmitsNoRows pins the secondary-defect fix: DRM cards whose
// device exposes no amdgpu metric (platform iGPU PCI functions) and their
// connector nodes yield no rows at all — no all-nil pollution.
func TestAMDNoMetricsEmitsNoRows(t *testing.T) {
	root := amdFixture(t,
		[]amdCard{{name: "card0", pci: "0000:00:02.0"}, {name: "card1", pci: "0000:01:00.0"}},
		[]string{"card0-DP-1", "card1-HDMI-A-1"})
	installDRMRoot(t, root)
	if gpus := amdSysfsgpus(context.Background()); len(gpus) != 0 {
		t.Fatalf("all-nil rows emitted: %+v", gpus)
	}
}

func TestAMDMetricsEmitRow(t *testing.T) {
	root := amdFixture(t,
		[]amdCard{{
			name: "card0",
			pci:  "0000:00:02.0",
			files: map[string]string{
				"gpu_busy_percent":    "42\n",
				"mem_info_vram_total": "8589934592\n",
				"mem_info_vram_used":  "1073741824\n",
				"power":               "65.5\n",
			},
		}},
		nil)
	installDRMRoot(t, root)
	gpus := amdSysfsgpus(context.Background())
	if len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	g := gpus[0]
	wantPCI := amdPciDir("0000:00:02.0")
	if g.UUID != wantPCI || g.Name != "card0 (amdgpu)" || g.PCI != wantPCI {
		t.Fatalf("identity = %+v", g)
	}
	if *g.UtilPct != 42 || *g.MemTotalBytes != 8589934592 || *g.MemUsedBytes != 1073741824 || *g.PowerW != 65.5 {
		t.Fatalf("metrics = %+v", g)
	}
}

// TestAMDConnectorSkippedEvenWithMetrics guards against the connector's
// device symlink inheriting its card's metrics and emitting a duplicate row.
func TestAMDConnectorSkippedEvenWithMetrics(t *testing.T) {
	root := amdFixture(t,
		[]amdCard{{name: "card0", pci: "0000:00:02.0", files: map[string]string{"gpu_busy_percent": "42\n"}}},
		[]string{"card0-DP-1"})
	installDRMRoot(t, root)
	gpus := amdSysfsgpus(context.Background())
	if len(gpus) != 1 {
		t.Fatalf("gpus = %+v", gpus)
	}
	if gpus[0].Name != "card0 (amdgpu)" {
		t.Fatalf("connector leaked a row: %+v", gpus)
	}
}
