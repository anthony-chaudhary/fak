package compute

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=10s lane=default
// Execute the production F32 V-gather/mask/accumulation region with poisoned
// physical-page tails. Host lanes are summed after this region; CUDA launches,
// shuffle/barrier execution, QK/softmax and hardware performance are not tested.
func TestPagedAttentionVPaddingNaNMask(t *testing.T) {
	b, err := os.ReadFile("cuda/paged_attention_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	start := "  constexpr int V_VEC_SIZE = FAK_PA_MIN(16 / sizeof(scalar_t), BLOCK_SIZE);"
	end := "#pragma unroll\n  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {\n    float acc = accs[i];"
	a, z := strings.Index(source, start), strings.Index(source, end)
	if a < 0 || z <= a || strings.Count(source, start) != 1 || strings.Count(source, end) != 1 {
		t.Fatal("missing or ambiguous production value accumulation region")
	}
	region := source[a:z]
	mask := "v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;"
	if strings.Count(region, mask) != 1 {
		t.Fatal("missing or ambiguous value padding mask for the negative control")
	}
	// Keep the witness on the actual instantiated storage type and geometry.
	for _, launch := range []string{
		"paged_attention_v1_kernel<float, float, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>",
		"paged_attention_v2_kernel<float, float, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>",
	} {
		if !strings.Contains(source, launch) {
			t.Fatalf("review host witness for changed production storage: %s", launch)
		}
	}
	var definitions strings.Builder
	for _, name := range []string{"FAK_PA_BLOCK_SIZE", "FAK_PA_NUM_THREADS", "FAK_PA_PARTITION_SIZE", "FAK_PA_WARP_SIZE"} {
		m := regexp.MustCompile(`(?m)^#define `+name+` ([0-9]+)$`).FindAllString(source, -1)
		if len(m) != 1 {
			t.Fatalf("missing or ambiguous production geometry %s", name)
		}
		definitions.WriteString(m[0] + "\n")
	}
	heads := regexp.MustCompile(`(?m)^\s+FAK_PA_CASE\(([0-9]+)\)$`).FindAllStringSubmatch(source, -1)
	if len(heads) == 0 {
		t.Fatal("missing production head-size dispatch")
	}
	var cases strings.Builder
	for _, h := range heads {
		fmt.Fprintf(&cases, "  if (!check<%s>(mutant)) return 1;\n", h[1])
	}
	actual := fmt.Sprintf(pagedVPaddingFunction, "actual", region)
	mutant := fmt.Sprintf(pagedVPaddingFunction, "unmasked", strings.Replace(region, mask, "v_vec_ptr[j] = v_vec_ptr[j];", 1))
	program := pagedVPaddingPreamble + definitions.String() + actual + mutant + pagedVPaddingControls +
		"int main(int argc, char**) { bool mutant = argc > 1;\n" + cases.String() +
		"  std::puts(\"PASS: production F32 V padding, all dispatched head sizes and partition tails\"); return 0; }\n"
	t.Logf("production header sha256=%x; value region sha256=%x; derived host program sha256=%x", sha256.Sum256(b), sha256.Sum256([]byte(region)), sha256.Sum256([]byte(program)))
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("required host C++ compiler unavailable: ", err)
	}
	dir := t.TempDir()
	src, exe := filepath.Join(dir, "padding.cc"), filepath.Join(dir, "padding")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	if err := os.WriteFile(src, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	run := func(command string, args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, command, args...)
		cmd.WaitDelay = time.Second
		return cmd.CombinedOutput()
	}
	if out, err := run(compiler, "-std=c++17", "-O0", src, "-o", exe); err != nil {
		t.Fatalf("host compile: %v\n%s", err, out)
	}
	if out, err := run(exe); err != nil {
		t.Fatalf("production V padding: %v\n%s", err, out)
	} else {
		if string(out) != "PASS: production F32 V padding, all dispatched head sizes and partition tails\n" {
			t.Fatalf("missing exact production witness success marker: %q", out)
		}
		t.Logf("%s", out)
	}
	out, err := run(exe, "--negative-control")
	if e, ok := err.(*exec.ExitError); !ok || e.ExitCode() != 1 || !strings.Contains(string(out), "nonfinite value with zero-weight padding") {
		t.Fatalf("removing the production mask must fail at accumulation, got %v\n%s", err, out)
	}
	t.Logf("mask-removed negative control failed as required: %s", out)
}

const pagedVPaddingPreamble = `
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#define FAK_PA_MIN(a,b) ((a)<(b)?(a):(b))
#define FAK_PA_DIVIDE_ROUND_UP(a,b) (((a)+(b)-1)/(b))
using scalar_t = float;
using cache_t = float;
template<class T,int N> struct Vec { using Type = std::array<T,N>; };
template<class T> struct FloatVec { using Type = T; };
template<class T> static void from_float(T &out, const T &in) { out = in; }
static void zero(float &v) { v = 0.0f; }
template<size_t N> static float dot(const std::array<float,N>& a, const std::array<float,N>& b) {
  float sum = 0; for (size_t i=0; i<N; i++) sum += a[i]*b[i]; return sum;
}
`

const pagedVPaddingFunction = `
template<int HEAD_SIZE>
static std::vector<float> %s(const std::vector<float>& cache, std::vector<float> weights,
                            const std::vector<int>& pages, int seq_len, int start_block_idx,
                            int end_block_idx, int kv_head_idx) {
  constexpr int BLOCK_SIZE = FAK_PA_BLOCK_SIZE;
  constexpr int NUM_WARPS = FAK_PA_NUM_THREADS / FAK_PA_WARP_SIZE;
  const int num_seq_blocks = FAK_PA_DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int kv_head_stride = HEAD_SIZE * BLOCK_SIZE;
  const int kv_block_stride = 2 * kv_head_stride;
  const float* v_cache = cache.data();
  float* logits = weights.data();
  const int* block_table = pages.data();
  std::vector<float> out(HEAD_SIZE, 0.0f);
  for (int warp_idx=0; warp_idx<NUM_WARPS; warp_idx++) {
    for (int lane=0; lane<FAK_PA_WARP_SIZE; lane++) {
%s
      // Host reference reduction of this region's lane-local accumulators.
      // This does not model CUDA shuffle/barrier or cross-partition reduction.
      for (int i=0; i<NUM_ROWS_PER_THREAD; i++) {
        const int row = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row < HEAD_SIZE) out[row] += accs[i];
      }
    }
  }
  return out;
}
`

const pagedVPaddingControls = `
template<int HEAD_SIZE> static bool check(bool mutant) {
  constexpr int B = FAK_PA_BLOCK_SIZE, P = FAK_PA_PARTITION_SIZE;
  static_assert(B >= 4 && B % 4 == 0 && P % B == 0, "unsupported production geometry");
  std::vector<int> lengths;
  for (int n=1; n<=2*B+1; n++) lengths.push_back(n);
  for (int n : {P-1,P,P+1,P+B-1,P+B,P+B+1,2*P+1}) lengths.push_back(n);
  for (int n : lengths) {
    const int blocks = (n+B-1)/B, physical_blocks = blocks+2;
    std::vector<int> pages(blocks);
    // Reversed, non-identity physical IDs, with unused guard pages.
    for (int i=0; i<blocks; i++) pages[i] = blocks-i;
    std::vector<float> poison(size_t(physical_blocks)*2*HEAD_SIZE*B, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> clean(poison.size(), 0.0f);
    auto offset = [&](int token,int head,int row) {
      return size_t(pages[token/B])*2*HEAD_SIZE*B + head*HEAD_SIZE*B + row*B + token%B;
    };
    for (int t=0; t<n; t++) for (int h=0; h<2; h++) for (int r=0; r<HEAD_SIZE; r++) {
      const float value = float((t%13)-6)/8 + float(h+1)/4 + float(r%7)/16;
      clean[offset(t,h,r)] = poison[offset(t,h,r)] = value;
    }
    // Whole-sequence and real v2 partition windows share the same value loop.
    for (bool partitioned : {false,true}) for (int start=0; start<n; start += partitioned ? P : n) {
      const int end = partitioned ? std::min(n,start+P) : n;
      const int first_block = start/B, last_block = (end+B-1)/B;
      std::vector<float> weights((last_block-first_block)*B,0.0f);
      for (int t=start; t<end; t++) weights[t-start] = float(t%5+1)/2048;
      for (int head=0; head<2; head++) {
        std::vector<double> want(HEAD_SIZE,0.0);
        for (int t=start; t<end; t++) for (int r=0; r<HEAD_SIZE; r++) want[r] += double(weights[t-start])*clean[offset(t,head,r)];
        for (bool poisoned : {false,true}) {
          const auto& input = poisoned ? poison : clean;
          const auto before = input;
          const auto got = mutant ? unmasked<HEAD_SIZE>(input,weights,pages,n,first_block,last_block,head)
                                  : actual<HEAD_SIZE>(input,weights,pages,n,first_block,last_block,head);
          for (int r=0; r<HEAD_SIZE; r++) {
            if (!std::isfinite(got[r])) {
              std::fprintf(stderr,"nonfinite value with zero-weight padding: head_size=%d length=%d start=%d head=%d row=%d poisoned=%d\n",HEAD_SIZE,n,start,head,r,poisoned);
              return false;
            }
            if (std::abs(double(got[r])-want[r]) > 1e-6*std::max(1.0,std::abs(want[r]))) {
              std::fprintf(stderr,"valid V accumulation changed: head_size=%d length=%d row=%d\n",HEAD_SIZE,n,r);
              return false;
            }
          }
          // The mask must affect the gathered vector, never overwrite cache bytes.
          if (std::memcmp(input.data(),before.data(),input.size()*sizeof(float)) != 0) {
            std::fputs("source cache mutated\n",stderr); return false;
          }
        }
      }
    }
  }
  return true;
}
`
