package compute

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
// Every `type V41*Backend interface` in the package has exactly one probe row,
// so a new op contract cannot ship without showing up in the coverage table.
func TestV41OpCoverageListIsClosed(t *testing.T) {
	re := regexp.MustCompile(`(?m)^type (V41\w*Backend) interface`)
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var declared []string
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(data), -1) {
			declared = append(declared, m[1])
		}
	}
	var listed []string
	seen := map[string]bool{}
	for _, p := range v41OpProbes {
		if seen[p.name] {
			t.Fatalf("duplicate coverage row %s", p.name)
		}
		seen[p.name] = true
		listed = append(listed, p.name)
	}
	sort.Strings(declared)
	sort.Strings(listed)
	if strings.Join(declared, ",") != strings.Join(listed, ",") {
		t.Fatalf("coverage rows drifted from declared V41 contracts\ndeclared: %v\nlisted:   %v", declared, listed)
	}
}

// fak-test:runtime fast est=10ms lane=default
// The host reference serves no V4.1 op on a device.
func TestV41OpCoverageCPUHasNoDeviceOps(t *testing.T) {
	be, ok := Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	r := V41Coverage(be)
	if r.Schema != V41CoverageReportSchema || r.TotalOps != len(v41OpProbes) || r.DeviceOps != 0 {
		t.Fatalf("cpu coverage = schema %q total %d device %d", r.Schema, r.TotalOps, r.DeviceOps)
	}
	for _, row := range r.Ops {
		if row.Status != V41OpAbsent && row.Status != V41OpHostFallback {
			t.Fatalf("cpu row %s status %q", row.Op, row.Status)
		}
	}
	for _, dt := range r.WeightTypes {
		if dt.Device {
			t.Fatalf("cpu reports device dtype %s", dt.Dtype)
		}
	}
	for _, row := range V41OpCoverage(nil) {
		if row.Status != V41OpAbsent {
			t.Fatalf("nil backend row %s status %q", row.Op, row.Status)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
// Report for the artifact under test: the CUDA backend when this binary was
// built with -tags cuda and a device is present, else the host reference.
// FAK_V41_COVERAGE_OUT=<path> writes the JSON receipt (fak-dev build-cuda
// coverage); FAK_CUDA_REQUIRE_DEVICE=1 fails instead of falling back.
func TestV41OpCoverageReport(t *testing.T) {
	be, ok := Lookup("cuda")
	if !ok || be == nil {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_CUDA_REQUIRE_DEVICE=1 but the cuda backend is not registered in this build")
		}
		be, _ = Lookup("cpu-ref")
	}
	r := V41Coverage(be)
	for _, row := range r.Ops {
		t.Logf("V41_COVERAGE %-34s %-13s %s", row.Op, row.Status, row.Probe)
	}
	for _, dt := range r.WeightTypes {
		t.Logf("V41_COVERAGE_DTYPE %-8s device=%v", dt.Dtype, dt.Device)
	}
	t.Logf("V41_COVERAGE backend=%s device_ops=%d/%d", r.Backend, r.DeviceOps, r.TotalOps)
	if path := os.Getenv("FAK_V41_COVERAGE_OUT"); path != "" {
		data, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
