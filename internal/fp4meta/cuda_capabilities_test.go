package fp4meta

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestCUBLASLtNVFP4PolicyBoundaries(t *testing.T) {
	for _, tc := range []struct {
		sm, version    int
		eligible, skip bool
	}{
		{89, 130800, false, false}, {100, 120802, false, false}, {100, 120803, true, false},
		{120, 120803, false, false}, {120, 120804, true, false}, {103, 120899, false, false},
		{103, 120900, true, false}, {103, 130500, true, false}, {103, 130501, true, true},
		{103, 130601, true, true}, {103, 130602, true, false}, {103, 130699, true, false},
		{103, 130700, true, true}, {103, 130799, true, true}, {103, 130800, true, false},
		{100, 130700, true, false}, {120, 130700, true, false}, {103, -1, false, false},
	} {
		eligible, skip := CUBLASLtNVFP4Policy(tc.sm, tc.version)
		if eligible != tc.eligible || skip != tc.skip {
			t.Errorf("sm=%d version=%d got=(%v,%v) want=(%v,%v)", tc.sm, tc.version, eligible, skip, tc.eligible, tc.skip)
		}
	}
}

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
