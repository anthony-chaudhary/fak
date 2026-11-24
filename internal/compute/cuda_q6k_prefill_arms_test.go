package compute

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestResolveCUDAQ6KPrefillArm(t *testing.T) {
	for _, tc := range []struct {
		env     string
		sm      int
		want    CUDAQ6KPrefillArm
		wantErr error
	}{
		{"", 0, CUDAQ6KPrefillDequant, nil},
		{"", 75, CUDAQ6KPrefillDequant, nil},
		{"", 80, CUDAQ6KPrefillMMQ, nil},
		{"", 89, CUDAQ6KPrefillMMQ, nil},
		{"", 120, CUDAQ6KPrefillMMQ, nil},
		{"dequant", 120, CUDAQ6KPrefillDequant, nil},
		{"mmq", 0, CUDAQ6KPrefillMMQ, nil},
		{"mmq", 86, CUDAQ6KPrefillMMQ, nil},
		{"mmq", 75, CUDAQ6KPrefillDequant, ErrCUDAQ6KPrefillArmArch},
		{"cublas", 120, CUDAQ6KPrefillDequant, ErrCUDAQ6KPrefillArmUnknown},
	} {
		got, err := resolveCUDAQ6KPrefillArm(tc.env, tc.sm)
		if got != tc.want || !errors.Is(err, tc.wantErr) || (tc.wantErr == nil) != (err == nil) {
			t.Errorf("resolve(%q, sm_%d) = %q, %v; want %q, %v", tc.env, tc.sm, got, err, tc.want, tc.wantErr)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAQ6KPrefillMMQEligible(t *testing.T) {
	for _, tc := range []struct {
		arm  CUDAQ6KPrefillArm
		dt   Dtype
		p    int
		qual bool
		want bool
	}{
		{CUDAQ6KPrefillMMQ, Q6_K, cudaQ6KMMQMinCols, false, true},
		{CUDAQ6KPrefillMMQ, Q6_K, 512, false, true},
		{CUDAQ6KPrefillMMQ, Q6_K, cudaQ6KMMQMinCols - 1, false, false},
		{CUDAQ6KPrefillMMQ, Q6_K, 1, false, false},
		{CUDAQ6KPrefillMMQ, Q6_K, 512, true, false}, // the strict qualification route owns it
		{CUDAQ6KPrefillMMQ, Q5_K, 512, false, false},
		{CUDAQ6KPrefillMMQ, Q4_K, 512, false, false},
		{CUDAQ6KPrefillDequant, Q6_K, 512, false, false},
	} {
		if got := cudaQ6KPrefillMMQEligible(tc.arm, tc.dt, tc.p, tc.qual); got != tc.want {
			t.Errorf("eligible(%s, %s, P=%d, qual=%v) = %v; want %v", tc.arm, tc.dt, tc.p, tc.qual, got, tc.want)
		}
	}
}

// The Go and native min-cols and declined codes are separate literals; keep them in step.
// fak-test:runtime fast est=5ms lane=default
func TestCUDAQ6KPrefillNativeConstantsAgree(t *testing.T) {
	b, err := os.ReadFile("cuda/q56k_mmq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	for _, marker := range []string{
		"#define FCUDA_Q6K_PREFILL_MMQ_DECLINED 80011",
		"#define FCUDA_Q6K_PREFILL_MMQ_MIN_COLS 8",
		"#define FAK_Q56K_MMQ_MIN_COLS 8",
		"if (g_q6k_prefill_arm != 1 || p < FCUDA_Q6K_PREFILL_MMQ_MIN_COLS)",
		"run<6>(w,x,y,out,in,p,FCUDA_Q6K_PREFILL_MMQ_DECLINED)",
	} {
		if !strings.Contains(src, marker) {
			t.Fatalf("native Q6_K prefill arm missing %q", marker)
		}
	}
	if cudaQ6KMMQMinCols != 8 || cudaQ6KPrefillMMQDeclined != 80011 {
		t.Fatalf("Go min cols %d / declined %d disagree with native 8 / 80011", cudaQ6KMMQMinCols, cudaQ6KPrefillMMQDeclined)
	}
	if cudaQ6KPrefillArmCodes[CUDAQ6KPrefillDequant] != 0 || cudaQ6KPrefillArmCodes[CUDAQ6KPrefillMMQ] != 1 {
		t.Fatal("arm codes disagree with g_q6k_prefill_arm (1 = mmq)")
	}
	// The family must compile without the qualification define: the only guard ahead of the
	// namespace is the one inside fcuda_q56k_mmq_enabled.
	i := strings.Index(src, "namespace fak_q56k_mmq {")
	if i < 0 || strings.Count(src[:i], "#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION") != 1 {
		t.Fatal("Q56K MMQ family is still compiled only under the qualification define")
	}
}

// A default-arm scratch miss must decline natively or surface as an allocation error, never as
// an execution fault that poisons the backend.
// fak-test:runtime fast est=5ms lane=default
func TestCUDAQ6KPrefillMMQScratchMissIsNotAnExecutionFault(t *testing.T) {
	native, err := os.ReadFile("cuda/q56k_mmq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	for _, marker := range []string{
		"return declined && status == FCUDA_SCRATCH_ALLOC_FAILED && !g_cuda_cleanup_status;",
		"return declinable_miss(declined, st) ? declined : st;",
		"const int freed = retire(activation, nullptr, 0);",
		"run_variant<kind,J,fallback>(w,x,y,out,in,p,info,plan,declined)",
	} {
		if !strings.Contains(string(native), marker) {
			t.Fatalf("native Q6_K MMQ scratch-miss decline missing %q", marker)
		}
	}
	panel, err := os.ReadFile("cuda_prompt_panel.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(panel), "cudaScratchAllocFailed      = 80001") || cudaQ6KPrefillMMQScratchMiss != 80001 {
		t.Fatal("Q6_K MMQ scratch-miss code disagrees with cudaScratchAllocFailed")
	}
	dispatch, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(dispatch), "ran && rc == cudaQ6KPrefillMMQScratchMiss {\n\t\t\tpanic(cudaGEMMError(") {
		t.Fatal("BatchedMatMul no longer maps a Q6_K MMQ scratch miss through cudaGEMMError")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAQ6KPrefillMMQScratchBytes(t *testing.T) {
	for _, tc := range []struct{ in, rows, want int }{
		{3584, 8, 8*3584/128*144 + 128*144},
		{18944, 8192, 8192*18944/128*144 + 128*144}, // 18944 is already 512-aligned
		{256, 64, 64*512/128*144 + 128*144},
	} {
		if got := cudaQ6KPrefillMMQScratchBytes(tc.in, tc.rows); got != tc.want {
			t.Errorf("scratch(in=%d, rows=%d) = %d; want %d", tc.in, tc.rows, got, tc.want)
		}
	}
}
