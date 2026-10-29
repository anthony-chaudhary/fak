package compute

import (
	"errors"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1s
func TestParseCUDAQ4KPrefillArm(t *testing.T) {
	cases := []struct {
		in   string
		want CUDAQ4KPrefillArm
		err  error
	}{
		{"", CUDAQ4KPrefillCompensated, nil},
		{"compensated", CUDAQ4KPrefillCompensated, nil},
		{"mmq", CUDAQ4KPrefillMMQ, nil},
		{"fp8lt", CUDAQ4KPrefillFP8LT, nil},
		{"MMQ", "", ErrCUDAQ4KPrefillArmUnknown},
		{"fp8", "", ErrCUDAQ4KPrefillArmUnknown},
		{" mmq", "", ErrCUDAQ4KPrefillArmUnknown},
	}
	for _, tc := range cases {
		got, err := ParseCUDAQ4KPrefillArm(tc.in)
		if !errors.Is(err, tc.err) || got != tc.want {
			t.Errorf("ParseCUDAQ4KPrefillArm(%q) = %q, %v; want %q, %v", tc.in, got, err, tc.want, tc.err)
		}
		var armErr *CUDAQ4KPrefillArmError
		if tc.err != nil && !errors.As(err, &armErr) {
			t.Errorf("ParseCUDAQ4KPrefillArm(%q) error %T is not a *CUDAQ4KPrefillArmError", tc.in, err)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCheckCUDAQ4KPrefillArmArchAndImplementation(t *testing.T) {
	cases := []struct {
		arm CUDAQ4KPrefillArm
		sm  int
		err error
	}{
		{CUDAQ4KPrefillCompensated, 80, nil},
		{CUDAQ4KPrefillCompensated, 120, nil},
		{CUDAQ4KPrefillCompensated, 0, nil},
		{CUDAQ4KPrefillFP8LT, 80, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillFP8LT, 86, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillFP8LT, 89, nil},
		{CUDAQ4KPrefillFP8LT, 90, nil},
		{CUDAQ4KPrefillFP8LT, 91, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillFP8LT, 100, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillFP8LT, 103, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillFP8LT, 120, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillMMQ, 75, ErrCUDAQ4KPrefillArmArch},
		{CUDAQ4KPrefillMMQ, 80, nil},
		{CUDAQ4KPrefillMMQ, 120, nil},
		{"bogus", 120, ErrCUDAQ4KPrefillArmUnknown},
	}
	for _, tc := range cases {
		if err := CheckCUDAQ4KPrefillArm(tc.arm, tc.sm); !errors.Is(err, tc.err) {
			t.Errorf("CheckCUDAQ4KPrefillArm(%q, %d) = %v; want %v", tc.arm, tc.sm, err, tc.err)
		}
	}
	if _, err := resolveCUDAQ4KPrefillArm("fp8lt", 80); !errors.Is(err, ErrCUDAQ4KPrefillArmArch) {
		t.Errorf("resolve fp8lt on sm_80 = %v; want ErrCUDAQ4KPrefillArmArch", err)
	}
}

// TestSetCUDAQ4KPrefillArmRefusalKeepsCurrentArm drives the runtime setter. A refused arm returns
// its typed error and leaves the arm in use unchanged, with or without a CUDA device.
//
// fak-test:runtime fast est=1s
func TestSetCUDAQ4KPrefillArmRefusalKeepsCurrentArm(t *testing.T) {
	before, beforeErr := CUDAQ4KPrefillArmInUse()
	if beforeErr != nil {
		t.Skipf("%s refused at init: %v", CUDAQ4KPrefillEnv, beforeErr)
	}
	t.Cleanup(func() { _ = SetCUDAQ4KPrefillArm(before) })
	if err := SetCUDAQ4KPrefillArm(CUDAQ4KPrefillCompensated); err != nil {
		t.Fatalf("SetCUDAQ4KPrefillArm(compensated) = %v", err)
	}
	for _, arm := range []CUDAQ4KPrefillArm{"bogus", "FP8LT"} {
		var armErr *CUDAQ4KPrefillArmError
		if err := SetCUDAQ4KPrefillArm(arm); !errors.As(err, &armErr) {
			t.Errorf("SetCUDAQ4KPrefillArm(%q) = %v; want *CUDAQ4KPrefillArmError", arm, err)
		}
		if got, err := CUDAQ4KPrefillArmInUse(); got != CUDAQ4KPrefillCompensated || err != nil {
			t.Errorf("after refused %q: arm in use = %q, %v; want compensated", arm, got, err)
		}
	}
}

// TestCUDAQ4KMMQMinColsMatchesKernelSource holds the Go mirror of the small-panel mmq threshold
// to the C dispatch, and the threshold to a column tile the vendored MMQ kernel instantiates.
//
// fak-test:runtime fast est=10ms
func TestCUDAQ4KMMQMinColsMatchesKernelSource(t *testing.T) {
	dispatch, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^#define FCUDA_Q4K_MMQ_MIN_COLS (\d+)$`).FindSubmatch(dispatch)
	if m == nil {
		t.Fatal("FCUDA_Q4K_MMQ_MIN_COLS not defined in cuda/quant_matmul.cuh")
	}
	if got, _ := strconv.Atoi(string(m[1])); got != cudaQ4KMMQMinCols {
		t.Fatalf("FCUDA_Q4K_MMQ_MIN_COLS = %d, Go cudaQ4KMMQMinCols = %d", got, cudaQ4KMMQMinCols)
	}
	mmq, err := os.ReadFile("cuda/q4k_mmq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	tile := regexp.MustCompile(`case\s+` + strconv.Itoa(cudaQ4KMMQMinCols) + `: return launch_mul_mat_q<\s*` + strconv.Itoa(cudaQ4KMMQMinCols) + `,`)
	if !tile.Match(mmq) {
		t.Fatalf("q4k_mmq.cuh instantiates no %d-column MMQ tile", cudaQ4KMMQMinCols)
	}
}
