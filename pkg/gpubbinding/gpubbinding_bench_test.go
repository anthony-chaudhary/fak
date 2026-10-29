package gpubinding

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkDiagnoseReportConstruction measures closed verdict and report
// construction over a fixed two-device sysfs fixture (one enumerated NVIDIA
// display device plus one expected-but-absent row) — the exact shape companion
// repositories consume through this public seam.
func BenchmarkDiagnoseReportConstruction(b *testing.B) {
	root := b.TempDir()
	dev := filepath.Join(root, "bus", "pci", "devices", "0000_58_00.0")
	if err := os.MkdirAll(dev, 0o755); err != nil {
		b.Fatal(err)
	}
	for k, v := range map[string]string{"vendor": "0x10de", "device": "0x2c05", "class": "0x030000"} {
		if err := os.WriteFile(filepath.Join(dev, k), []byte(v+"\n"), 0o644); err != nil {
			b.Fatal(err)
		}
	}
	opts := Options{SysfsRoot: root, ExpectGPUs: 2}
	var rep Report
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep = Diagnose(opts)
	}
	b.StopTimer()
	if len(rep.Devices) != 2 || rep.Ready {
		b.Fatalf("report = %+v, want 2 devices, not ready", rep.Verdicts)
	}
}
