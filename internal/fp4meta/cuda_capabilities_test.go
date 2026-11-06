package fp4meta

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestCUDACapabilitiesAdmitsSM120OnlyBehindCUBLASPolicy(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "nvfp4.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"architecture": "sm100"`), []byte(`"architecture": "sm120"`), 1)
	cases := []struct {
		name       string
		caps       Capabilities
		unverified bool
	}{
		{"defaults", DefaultCapabilities(), true},
		{"sm120 cuBLAS 12.8.3 below floor", CUDACapabilities(120, 120803), true},
		{"sm120 cuBLAS 12.8.4 admitted", CUDACapabilities(120, 120804), false},
		{"sm89 unqualified", CUDACapabilities(89, 130000), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, res, _ := Parse(raw, tc.caps)
			got := res.Outcome == OutcomeAbstain && res.Reason == ReasonHardwareUnverified
			if got != tc.unverified {
				t.Fatalf("result = %+v, want hardware_unverified=%v", res, tc.unverified)
			}
		})
	}
	if caps := CUDACapabilities(120, 120804); len(caps.Hardware) != len(DefaultCapabilities().Hardware)+1 {
		t.Fatalf("hardware = %v", caps.Hardware)
	}
	if caps := CUDACapabilities(100, 120803); len(caps.Hardware) != len(DefaultCapabilities().Hardware) {
		t.Fatalf("duplicate sm100 entry: %v", caps.Hardware)
	}
}
