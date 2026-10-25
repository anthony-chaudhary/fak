package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func runBindingJSON(t *testing.T, args ...string) (compute.GPUBindingReport, int) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, append([]string{"binding", "--json"}, args...))
	var rep compute.GPUBindingReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("binding --json output is not a report: %v (stderr %q)", err, stderr.String())
	}
	return rep, code
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_BindingUnboundAndMissing(t *testing.T) {
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
	rep, code := runBindingJSON(t, "--sysfs", root, "--expect", "2")
	var got []compute.GPUBindingVerdict
	for _, d := range rep.Devices {
		got = append(got, d.Verdict)
	}
	want := []compute.GPUBindingVerdict{compute.GPUUnbound, compute.GPUNotEnumerated}
	if !reflect.DeepEqual(got, want) || code != 1 || rep.Ready {
		t.Fatalf("verdicts=%v code=%d ready=%v, want %v, exit 1, not ready", got, code, rep.Ready, want)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_BindingMissingSysfsIsError(t *testing.T) {
	rep, code := runBindingJSON(t, "--sysfs", filepath.Join(t.TempDir(), "absent"), "--expect", "3")
	if rep.Status != compute.CUDADiscoveryError || len(rep.Devices) != 0 || code != 1 {
		t.Fatalf("status=%s devices=%d code=%d, want error, 0 devices, exit 1", rep.Status, len(rep.Devices), code)
	}
}
