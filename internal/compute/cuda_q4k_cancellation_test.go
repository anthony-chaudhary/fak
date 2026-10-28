package compute

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// This is an adversarial, valid GGUF fixture, not a sampled model weight. Row
// zero decodes to 1024*1*3 - 3072*1 == 0; row one has small positive weights.
// Splitting the zero weight into two activation FMAs can invert the argmax.
func cudaQ4KCancellationFixture() ([]byte, []float32, []float32) {
	raw := make([]byte, 2*q4kSuperBlock)
	for row, scale := range []uint16{0x6400, 0x0040} { // 1024 and 2^-18
		block := raw[row*q4kSuperBlock : (row+1)*q4kSuperBlock]
		binary.LittleEndian.PutUint16(block, scale)
		code := byte(1)
		if row == 0 {
			binary.LittleEndian.PutUint16(block[2:], 0x6a00) // 3072
			code = 3
		}
		for i := 4; i < 12; i++ {
			block[i] = 1
		}
		for i := 12; i < 16; i++ {
			block[i] = 0x11
		}
		for i := 16; i < len(block); i++ {
			block[i] = code | code<<4
		}
	}
	x := make([]float32, q4kSuper)
	for i := range x {
		x[i] = .1
	}
	ref := Default()
	want := ref.Read(ref.MatMul(NewQ4K(ref, []int{2, q4kSuper}, raw), NewF32(ref, []int{q4kSuper}, x)))
	return raw, x, want
}

func checkCUDAQ4KCancellation(t *testing.T, got, want []float32) {
	t.Helper()
	// Match the recorded CUDA Q4_K floor without importing a CUDA-tagged constant.
	const cosineMin = .995
	if len(want) != 2 || want[0] != 0 || !(want[1] > 0) {
		t.Fatalf("invalid cancellation oracle: %v", want)
	}
	if len(got) != 2 || got[0] != 0 || math.IsNaN(float64(got[1])) ||
		math.Abs(float64(got[1]-want[1])) > float64(want[1])*1e-5 ||
		!(cosine(got, want) >= cosineMin) || !(got[1] > got[0]) {
		t.Fatalf("Q4_K dequantized-dot cancellation: got %v, reference %v", got, want)
	}
}

// Executes the exact production per-lane arithmetic with host fmaf. The warp
// reduction is 32 identical lane totals. This is not NVCC or device execution.
// fak-test:runtime medium est=1s lane=optin
func TestCUDAQ4KCancellationHost(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_Q4K_CANCELLATION_HOST") != "1" {
		t.Skip("explicit host C++ arithmetic witness")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	begin := strings.Index(source, "__global__ void k_q4k_gemm(")
	end := strings.Index(source, "// k_q4k_gemm_panel")
	if begin < 0 || end <= begin {
		t.Fatal("production Q4_K decode kernel missing")
	}
	kernel := source[begin:end]
	begin = strings.Index(kernel, "float2 c = cb[group];")
	if begin < 0 {
		t.Fatal("production Q4_K coefficient use missing")
	}
	end = strings.Index(kernel[begin:], "\n      }")
	if end < 0 {
		t.Fatal("production Q4_K inner arithmetic boundary missing")
	}
	fragment := kernel[begin : begin+end]
	program := `
#include <cmath>
#include <cstdio>
struct float2 { float x, y; };
static float lane(float scale, float minimum, int code) {
  float2 cb[8]; float xb[256];
  for (auto &c : cb) c = {scale, minimum};
  for (auto &x : xb) x = 0.1f;
  int lane = 0;
  float sum = 0.0f;
  for (int group = 0; group < 8; ++group) {
` + fragment + `
  }
  return sum;
}
int main() {
  std::printf("%.9g %.9g\n", 32.0f*lane(1024.0f, 3072.0f, 3),
              32.0f*lane(0x1p-18f, 0.0f, 1));
}
`
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "witness.cpp"), filepath.Join(dir, "witness")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production arithmetic: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("execute production arithmetic: %v\n%s", err, out)
	}
	got := make([]float32, 2)
	if _, err := fmt.Fscan(strings.NewReader(string(out)), &got[0], &got[1]); err != nil {
		t.Fatal(err)
	}
	_, _, want := cudaQ4KCancellationFixture()
	checkCUDAQ4KCancellation(t, got, want)
}

// Explicit admission is required; absence of CUDA after opt-in is a failure,
// never a CPU substitution or a skipped hardware pass.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAQ4KCancellationDevice(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_Q4K_CANCELLATION_DEVICE") != "1" {
		t.Skip("requires admitted CUDA hardware execution")
	}
	be, ok := Lookup("cuda")
	if !ok || !be.Caps().DeviceMemory {
		t.Fatal("requested CUDA device backend unavailable")
	}
	raw, x, want := cudaQ4KCancellationFixture()
	w := be.Upload(NewQ4K(Default(), []int{2, q4kSuper}, raw), Q4_K)
	defer be.Free(w)
	for _, rows := range []int{1, 3, 4} { // Decode/small-batch and the panel control.
		t.Run(fmt.Sprint(rows), func(t *testing.T) {
			panel := make([]float32, rows*len(x))
			for row := 0; row < rows; row++ {
				copy(panel[row*len(x):], x)
			}
			input := be.Upload(NewF32(Default(), []int{rows, len(x)}, panel), F32)
			defer be.Free(input)
			var output Tensor
			if rows == 1 {
				output = be.MatMul(w, input)
			} else {
				output = be.BatchedMatMul(w, input, rows)
			}
			defer be.Free(output)
			got := be.Read(output)
			if len(got) != rows*2 {
				t.Fatalf("output length %d, want %d", len(got), rows*2)
			}
			for row := 0; row < rows; row++ {
				checkCUDAQ4KCancellation(t, got[row*2:(row+1)*2], want)
			}
		})
	}
}
