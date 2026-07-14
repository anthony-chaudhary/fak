package compute

import (
	"os"
	"strings"
	"testing"
)

// TestCUDAArchCoverageMatchesFile is the anti-drift guard: the Go cubin table (via KnownCUDAArches)
// must stay byte-equal to internal/compute/cuda_arch.txt, the single source build_cuda.sh reads. If
// someone adds an arch to the file but not the table (or vice versa), coverage classification would
// silently lie about what this build embeds — this test fails first.
func TestCUDAArchCoverageMatchesFile(t *testing.T) {
	raw, err := os.ReadFile("cuda_arch.txt")
	if err != nil {
		t.Fatalf("read cuda_arch.txt: %v", err)
	}
	var fileArches []string
	for _, ln := range strings.Fields(string(raw)) { // Fields splits on any whitespace, drops blanks
		fileArches = append(fileArches, ln)
	}
	got := KnownCUDAArches()
	if len(got) != len(fileArches) {
		t.Fatalf("KnownCUDAArches() = %v (%d), cuda_arch.txt = %v (%d): length drift",
			got, len(got), fileArches, len(fileArches))
	}
	for i := range got {
		if got[i] != fileArches[i] {
			t.Fatalf("arch %d: table %q != file %q (full: %v vs %v)", i, got[i], fileArches[i], got, fileArches)
		}
	}
}

func TestClassifyCUDAArchFullBuild(t *testing.T) {
	cases := []struct {
		major, minor int
		want         CUDACoverage
		why          string
	}{
		{8, 0, CUDANative, "sm_80 cubin, exact"},
		{8, 6, CUDANative, "sm_80 cubin covers 8.6 (same major, minor 6>=0)"},
		{8, 9, CUDANative, "sm_89 cubin, exact"},
		{9, 0, CUDANative, "sm_90 cubin, exact"},
		{10, 0, CUDANative, "sm_100 cubin, exact"},
		{12, 0, CUDANative, "sm_120 cubin, exact"},
		{12, 1, CUDANative, "sm_120 cubin covers 12.1 (same major, minor 1>=0)"},
		{13, 0, CUDAJITFromPTX, "no sm_13x cubin; compute_120 PTX JITs forward onto 13.0"},
		{11, 0, CUDAUncovered, "the gap: no sm_11x cubin, and the 12.0 PTX floor is too high to JIT down to 11.0"},
		{7, 5, CUDAUncovered, "Turing 7.5: no sm_7x cubin, below the 12.0 PTX floor"},
	}
	for _, c := range cases {
		if got := ClassifyCUDAArch(c.major, c.minor); got != c.want {
			t.Errorf("ClassifyCUDAArch(%d,%d) = %v, want %v (%s)", c.major, c.minor, got, c.want, c.why)
		}
	}
}

func TestClassifyCUDAArchSingle(t *testing.T) {
	// The #4184 headline: a single-arch sm_89 build has ONE cubin and NO PTX floor, so a Blackwell
	// sm_100 (10.0) card is UNCOVERED — not the confident bare "sm_100" cuda.go:95 would report.
	if got := ClassifyCUDAArchSingle(8, 9, 10, 0); got != CUDAUncovered {
		t.Errorf("single sm_89 on 10.0 = %v, want uncovered (the #4184 headline)", got)
	}
	// Same single cubin runs natively on its own arch and forward within the major...
	if got := ClassifyCUDAArchSingle(8, 9, 8, 9); got != CUDANative {
		t.Errorf("single sm_89 on 8.9 = %v, want native", got)
	}
	if got := ClassifyCUDAArchSingle(8, 0, 8, 9); got != CUDANative {
		t.Errorf("single sm_80 on 8.9 = %v, want native (minor 9>=0, same major)", got)
	}
	// ...but NOT backward across minor (device minor below the cubin's), and with no PTX to rescue it.
	if got := ClassifyCUDAArchSingle(8, 9, 8, 6); got != CUDAUncovered {
		t.Errorf("single sm_89 on 8.6 = %v, want uncovered (minor 6<9, no PTX floor)", got)
	}
	// A single-arch build never JITs forward: sm_120 single build on a future 13.0 is UNCOVERED,
	// whereas the FULL build's PTX floor would classify 13.0 as jit-from-ptx. This is the one-bit
	// difference between the two build modes.
	if got := ClassifyCUDAArchSingle(12, 0, 13, 0); got != CUDAUncovered {
		t.Errorf("single sm_120 on 13.0 = %v, want uncovered (no PTX floor in a single-arch build)", got)
	}
	if got := ClassifyCUDAArch(13, 0); got != CUDAJITFromPTX {
		t.Errorf("full build on 13.0 = %v, want jit-from-ptx (contrast with single-arch)", got)
	}
}

func TestCUDACoverageString(t *testing.T) {
	for c, want := range map[CUDACoverage]string{
		CUDANative:     "native",
		CUDAJITFromPTX: "jit-from-ptx",
		CUDAUncovered:  "uncovered",
	} {
		if got := c.String(); got != want {
			t.Errorf("CUDACoverage(%d).String() = %q, want %q", c, got, want)
		}
	}
}
