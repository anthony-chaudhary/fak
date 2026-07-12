package model

import (
	"runtime"
	"testing"
)

// TestQ8DecodeKernelWitness pins the #3176 Q1 witness: Q8DecodeKernel() must report a resolved
// decode inner-kernel tier from the known set for this host arch, and the FUSED flag must be
// self-consistent with that tier (fused only where the fused fast decode GEMV actually fires —
// amd64 + AVX-512). The value is a static host property resolved once at init, so it is stable
// across calls. This is the queryable fact an operator reads instead of guessing from wall-clock
// whether the SIMD decode lane engaged or decode fell back to the reference path.
func TestQ8DecodeKernelWitness(t *testing.T) {
	kernel, fused := Q8DecodeKernel()
	if kernel == "" {
		t.Fatal("Q8DecodeKernel returned an empty tier name")
	}

	// fused is true ONLY on the amd64 AVX-512 fused-row path; every other tier runs the
	// per-row qdot8GEMV (or the scalar reference), so a fused=true with any other tier name
	// would be a mislabeled witness.
	if fused && kernel != "avx512" {
		t.Fatalf("fused decode path reported for tier %q; only avx512 fires qMatRowsRangeFast", kernel)
	}

	known := map[string]bool{"avx512": true, "avx2": true, "scalar": true, "neon": true, "neon-amort": true}
	if !known[kernel] {
		t.Fatalf("Q8DecodeKernel returned unknown tier %q", kernel)
	}

	// Per-arch containment: the tier name must belong to this build's kernel ladder.
	switch runtime.GOARCH {
	case "amd64":
		switch kernel {
		case "avx512", "avx2", "scalar":
		default:
			t.Fatalf("amd64 host reported non-amd64 tier %q", kernel)
		}
		if kernel == "avx512" && !fused {
			t.Fatalf("amd64 avx512 host must report the fused decode path engaged, got fused=false")
		}
		if kernel != "avx512" && fused {
			t.Fatalf("amd64 tier %q must not report fused=true", kernel)
		}
	case "arm64":
		switch kernel {
		case "neon", "neon-amort", "scalar":
		default:
			t.Fatalf("arm64 host reported non-arm64 tier %q", kernel)
		}
		if fused {
			t.Fatal("arm64 never fires the fused decode path (qMatRowsRangeFast declines)")
		}
	}

	// Stable across calls (resolved once at init).
	if k2, f2 := Q8DecodeKernel(); k2 != kernel || f2 != fused {
		t.Fatalf("Q8DecodeKernel not stable: (%q,%v) then (%q,%v)", kernel, fused, k2, f2)
	}
}
