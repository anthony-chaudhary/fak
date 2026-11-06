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
		{10, 3, CUDANative, "sm_103 cubin, exact (B300)"},
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

// TestCudaArchDecodeProgramSelect pins the arch+dtype decode-program table's dispatch contract:
// a covered (arch, dtype) pair selects its specialized entry; an uncovered arch OR dtype falls
// back to the generic FP16 flat-ring program; the generic entry itself is stable; and every
// specialized entry stays inside the static embedded-cubin coverage data with no Blackwell part.
func TestCudaArchDecodeProgramSelect(t *testing.T) {
	t.Run("covered-arch-dtype-selects-specialized", func(t *testing.T) {
		cases := []struct {
			major, minor int
			dt           Dtype
			wantName     string
		}{
			{9, 0, FP8, "decode/sm90-fp8-uniform"}, // Hopper: covered arch + covered dtype
			{8, 9, FP8, "decode/sm89-fp8-uniform"}, // Ada: covered arch + covered dtype
		}
		for _, c := range cases {
			got := SelectCUDADecodeProgram(c.major, c.minor, c.dt)
			if got.Name != c.wantName {
				t.Errorf("SelectCUDADecodeProgram(%d,%d,%s).Name = %q, want %q", c.major, c.minor, c.dt, got.Name, c.wantName)
			}
			if !got.Specialized {
				t.Errorf("SelectCUDADecodeProgram(%d,%d,%s).Specialized = false, want true (table entry)", c.major, c.minor, c.dt)
			}
			if got.DType != c.dt {
				t.Errorf("SelectCUDADecodeProgram(%d,%d,%s).DType = %s, want %s", c.major, c.minor, c.dt, got.DType, c.dt)
			}
		}
	})

	t.Run("uncovered-arch-or-dtype-falls-back-to-generic", func(t *testing.T) {
		cases := []struct {
			major, minor int
			dt           Dtype
			why          string
		}{
			{10, 0, FP8, "Blackwell sm_100: no specialized entry (this port adds no Blackwell kernels)"},
			{10, 3, FP8, "Blackwell sm_103: no specialized entry"},
			{12, 0, FP8, "Blackwell consumer sm_120: no specialized entry"},
			{7, 5, FP8, "Turing 7.5: not in the embedded cubin set, no entry"},
			{13, 0, FP8, "future 13.0: not in the table"},
			{8, 0, FP8, "Ampere sm_80: no FP8 tensor-core lane, no entry"},
			{9, 0, F16, "Hopper arch covered but FP16 has no specialized stage"},
			{9, 0, BF16, "BF16: no specialized stage anywhere"},
			{9, 0, F32, "F32: no specialized stage anywhere"},
		}
		for _, c := range cases {
			got := SelectCUDADecodeProgram(c.major, c.minor, c.dt)
			if got.Specialized {
				t.Errorf("SelectCUDADecodeProgram(%d,%d,%s) = specialized %q, want generic fallback (%s)", c.major, c.minor, c.dt, got.Name, c.why)
			}
			if got.Name != cudaGenericDecodeProgram.Name {
				t.Errorf("SelectCUDADecodeProgram(%d,%d,%s).Name = %q, want the generic %q (%s)", c.major, c.minor, c.dt, got.Name, cudaGenericDecodeProgram.Name, c.why)
			}
		}
	})

	t.Run("generic-path-output-stable", func(t *testing.T) {
		// Regression pin: the generic fallback IS today's decode path. If these fields move, the
		// fallback is no longer behavior-identical and something downstream of dispatch changed.
		got := SelectCUDADecodeProgram(9, 0, F16) // any uncovered pair returns the same entry
		want := CUDADecodeProgram{Name: "decode/fp16-flat-ring", DType: F16, Specialized: false}
		if got != want {
			t.Errorf("generic fallback = %+v, want unchanged %+v", got, want)
		}
	})

	t.Run("table-stays-within-static-coverage", func(t *testing.T) {
		// The static coverage data stays authoritative: a specialized entry may only target an arch
		// this build embeds a native SASS cubin for, and no entry may target a Blackwell part.
		for k, p := range cudaDecodePrograms {
			embedded := false
			for _, c := range cudaEmbeddedArches {
				if c == k.cc {
					embedded = true
					break
				}
			}
			if !embedded {
				t.Errorf("specialized entry %q targets arch {%d,%d} with no embedded cubin", p.Name, k.cc.Major, k.cc.Minor)
			}
			if k.cc.Major >= 10 {
				t.Errorf("specialized entry %q targets Blackwell {%d,%d}; this port adds no Blackwell programs", p.Name, k.cc.Major, k.cc.Minor)
			}
		}
	})
}
