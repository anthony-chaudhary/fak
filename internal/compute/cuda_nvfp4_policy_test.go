package compute

import "testing"

// fak-test:runtime fast est=1ms lane=nvfp4-cublaslt-policy-tests
func TestNVFP4CUBLASLtPolicyArchitectureFloors(t *testing.T) {
	tests := []struct {
		name     string
		sm       int
		version  int
		eligible bool
	}{
		{"SM100 below cuBLAS 12.8.3", 100, 120802, false},
		{"SM100 cuBLAS 12.8.3", 100, 120803, true},
		{"SM103 below cuBLAS 12.9.0", 103, 120899, false},
		{"SM103 cuBLAS 12.9.0", 103, 120900, true},
		{"SM120 below cuBLAS 12.8.4", 120, 120803, false},
		{"SM120 cuBLAS 12.8.4", 120, 120804, true},
		{"pre-Blackwell", 89, 130800, false},
		{"unknown SM121", 121, 130800, false},
		{"unknown future architecture", 130, 140000, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			eligible, skip := nvfp4CUBLASLtPolicy(tc.sm, tc.version)
			if eligible != tc.eligible || skip {
				t.Fatalf("policy(%d, %d) = (%v, %v), want (%v, false)",
					tc.sm, tc.version, eligible, skip, tc.eligible)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=nvfp4-cublaslt-policy-tests
func TestNVFP4CUBLASLtPolicySM103Algo66Windows(t *testing.T) {
	tests := []struct {
		name    string
		version int
		skip    bool
	}{
		{"before affected branch", 130500, false},
		{"affected 13.5.1 lower bound", 130501, true},
		{"affected 13.6.1", 130601, true},
		{"fixed 13.6.2", 130602, false},
		{"fixed branch upper edge", 130699, false},
		{"regressed 13.7.0 lower bound", 130700, true},
		{"regressed 13.7 latest patch", 130799, true},
		{"fixed 13.8.0", 130800, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			eligible, skip := nvfp4CUBLASLtPolicy(103, tc.version)
			if !eligible || skip != tc.skip {
				t.Fatalf("policy(103, %d) = (%v, %v), want (true, %v)",
					tc.version, eligible, skip, tc.skip)
			}
		})
	}

	for _, sm := range []int{100, 120} {
		if eligible, skip := nvfp4CUBLASLtPolicy(sm, 130700); !eligible || skip {
			t.Fatalf("policy(%d, 130700) = (%v, %v), affected window is SM103-only",
				sm, eligible, skip)
		}
	}
}
