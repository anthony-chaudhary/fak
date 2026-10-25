package gpubinding

import (
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestDiagnoseSysfsFixture(t *testing.T) {
	root := t.TempDir()
	dev := filepath.Join(root, "bus", "pci", "devices", "0000_58_00.0")
	if err := os.MkdirAll(dev, 0o755); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{"vendor": "0x10de", "device": "0x2c05", "class": "0x030000"} {
		if err := os.WriteFile(filepath.Join(dev, k), []byte(v+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	rep := Diagnose(Options{SysfsRoot: root, ExpectGPUs: 2})
	if len(rep.Devices) != 2 || rep.Devices[0].Verdict != VerdictUnbound || rep.Devices[1].Verdict != VerdictNotEnumerated || rep.Ready {
		t.Fatalf("report = %+v, want GPU_UNBOUND + GPU_NOT_ENUMERATED, not ready", rep.Verdicts)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDiagnoseMissingSysfsClaimsNothing(t *testing.T) {
	rep := Diagnose(Options{SysfsRoot: filepath.Join(t.TempDir(), "absent"), ExpectGPUs: 3})
	if rep.Status != "error" || len(rep.Devices) != 0 || rep.Ready {
		t.Fatalf("status=%s devices=%d ready=%v, want error/0/false", rep.Status, len(rep.Devices), rep.Ready)
	}
}
