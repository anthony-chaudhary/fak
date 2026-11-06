package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Execute the actual device bodies as serial host calls with zero Q/K and
// initialized shared memory. Every score/reduction is exactly zero, so barrier
// elision is valid only for this layout witness, not arbitrary attention math.
// The independent oracle averages dimension-varying V directly. No CUDA syntax,
// GPU synchronization, nonzero-score numerical parity, or performance is proved.
// fak-test:runtime medium est=5s lane=default
// fak-test:justify Compiles one source-derived C++ layout witness with four hostile variants.
func TestCUDASplitKVPartialLayout(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("host C++ compiler required for split-KV layout witness: ", err)
	}
	source := qwen38PromptAttentionSource(t, "cuda/attention.cuh")
	var constants strings.Builder
	for _, name := range []string{"FLASH_THREADS", "QWEN38_PROMPT_HEAD_DIM", "QWEN38_SPLITKV_PREFIX_THRESHOLD", "QWEN38_SPLITKV_MAX_PARTITIONS"} {
		found := 0
		for _, line := range strings.Split(source, "\n") {
			if strings.HasPrefix(line, "#define "+name+" ") {
				constants.WriteString(line + "\n")
				found++
			}
		}
		if found != 1 {
			t.Fatalf("production constant %s: got %d definitions", name, found)
		}
	}
	actual := qwen38SourceFunction(t, source, "static int qwen38_splitkv_partitions(") + "\n" +
		qwen38SourceFunction(t, source, "__global__ void k_qwen38_causal_attention_panel_hd256_splitkv_partial(") + "\n" +
		qwen38SourceFunction(t, source, "__global__ void k_qwen38_splitkv_lse_merge_hd256(") + "\n"
	run := func(name, body, wantFailure string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			cpp, binary := filepath.Join(dir, "layout.cpp"), filepath.Join(dir, "layout.exe")
			program := constants.String() + splitKVLayoutHost + body + splitKVLayoutMain
			if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", cpp, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile %s: %v\n%s", name, err, out)
			}
			ctxRun, stop := context.WithTimeout(t.Context(), 20*time.Second)
			defer stop()
			out, err := exec.CommandContext(ctxRun, binary).CombinedOutput()
			if wantFailure == "" {
				if err != nil || !strings.Contains(string(out), "split-KV layout controls passed") {
					t.Fatalf("actual layout: %v\n%s", err, out)
				}
			} else {
				exit, ok := err.(*exec.ExitError)
				if !ok || exit.ExitCode() != 1 || !strings.Contains(string(out), wantFailure) {
					t.Fatalf("hostile %s did not fail at %q: %v\n%s", name, wantFailure, err, out)
				}
			}
		})
	}
	run("actual", actual, "")
	mutate := func(s, old, replacement string, count int) string {
		t.Helper()
		if strings.Count(s, old) != count {
			t.Fatalf("hostile mutation binding changed for %q", old)
		}
		return strings.ReplaceAll(s, old, replacement)
	}
	run("partial-local-index", mutate(actual, "slot[2 + d] = acc[k]", "slot[2 + k] = acc[k]", 1), "partial payload ownership")
	run("merge-local-index", mutate(actual, "acc[k] += slot[2 + d]", "acc[k] += slot[2 + k]", 1), "merged dimension average")
	run("empty-local-index", mutate(actual,
		"for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS) slot[2 + d] = 0.f;",
		"int k = 0; for (int d = tid; d < QWEN38_PROMPT_HEAD_DIM; d += FLASH_THREADS, k++) slot[2 + k] = 0.f;", 1), "partial payload ownership")
	run("replicated-scalars", mutate(actual, "if (tid == 0) {", "if (true) {", 2), "partial scalar ownership")
}

const splitKVLayoutHost = `
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
struct HostIndex { int x; };
static HostIndex blockIdx{}, threadIdx{};
float smem[QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS]{};
static_assert(QWEN38_PROMPT_HEAD_DIM == 256 && FLASH_THREADS == 128, "revisit serial fixture geometry");
static void require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
static bool same(float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); }
static bool near(float value, double want) {
  return std::isfinite(value) && std::fabs(double(value) - want) <= 2e-6 * std::max(1.0, std::fabs(want));
}
`

const splitKVLayoutMain = `
static constexpr int HD = QWEN38_PROMPT_HEAD_DIM, SLOT = HD + 2;
static constexpr float GUARD = 123456.f;
static const float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
static unsigned long long dimensions = 0, emptyPartitions = 0, scalarLanes = 0;
static float valueAt(int position, int kvHead, int d) {
  return float(position % 7 - 3) * 0.25f + float(kvHead * 2) + float(d) / 512.f;
}
static void guards(const std::vector<float>& x) {
  require(x.front() == GUARD && x.back() == GUARD, "storage boundary guard");
}
static void panel(int tokens, int prefix, int heads, int kvHeads, int splits, bool routed) {
  if (routed) require(qwen38_splitkv_partitions(prefix, tokens) == splits, "partition routing");
  const int positions = prefix + tokens, rows = tokens * heads;
  std::vector<float> q(size_t(rows) * HD, 0.f), k(size_t(positions) * kvHeads * HD, 0.f);
  std::vector<float> v(k.size());
  for (int j=0; j<positions; ++j) for (int h=0; h<kvHeads; ++h) for (int d=0; d<HD; ++d)
    v[(size_t(j)*kvHeads+h)*HD+d] = valueAt(j,h,d);
  std::vector<float> partialStorage(size_t(rows)*splits*SLOT+2, NAN_VALUE);
  partialStorage.front() = partialStorage.back() = GUARD;
  float *partials = partialStorage.data()+1;
  for (int partition=0; partition<splits; ++partition) for (int row=0; row<rows; ++row) {
    const int token=row/heads, h=row%heads, kvh=h/(heads/kvHeads), positionsHere=prefix+token+1;
    // Independent range counting: classify each key by the declared contiguous
    // floor-boundary interval; do not use the kernel's output or recurrence.
    const int begin=int((long long)partition*positionsHere/splits);
    const int end=int((long long)(partition+1)*positionsHere/splits);
    if (begin==end) ++emptyPartitions;
    float *slot=partials+(size_t(row)*splits+partition)*SLOT;
    blockIdx.x=partition*rows+row;
    // Nonzero lanes first makes accidental replicated scalar stores observable.
    for (int tid=FLASH_THREADS-1; tid>=0; --tid) {
      threadIdx.x=tid;
      std::fill(std::begin(smem),std::end(smem),0.f);
      std::vector<float> before(slot,slot+SLOT);
      k_qwen38_causal_attention_panel_hd256_splitkv_partial(q.data(),k.data(),v.data(),partials,tokens,prefix,heads,kvHeads,1.f,splits);
      if (tid!=0) {
        require(same(slot[0],before[0]) && same(slot[1],before[1]), "partial scalar ownership");
      } else {
        require(slot[0] == (begin==end ? -INFINITY : 0.f) && slot[1] == float(end-begin), "partial scalar value");
      }
      ++scalarLanes;
      for (int d=0; d<HD; ++d) {
        if (d%FLASH_THREADS==tid) {
          double sum=0;
          for (int j=begin; j<end; ++j) sum+=double(v[(size_t(j)*kvHeads+kvh)*HD+d]);
          require(near(slot[2+d],sum), "partial payload ownership");
          ++dimensions;
        } else require(same(slot[2+d],before[2+d]), "partial payload ownership");
      }
      guards(partialStorage);
    }
  }
  std::vector<float> outputStorage(size_t(rows)*HD+2,NAN_VALUE);
  outputStorage.front()=outputStorage.back()=GUARD;
  float *out=outputStorage.data()+1;
  for (int row=0; row<rows; ++row) {
    blockIdx.x=row;
    for (int tid=0; tid<FLASH_THREADS; ++tid) {
      threadIdx.x=tid;
      k_qwen38_splitkv_lse_merge_hd256(partials,out,tokens,heads,splits);
    }
    const int kvh=(row%heads)/(heads/kvHeads), count=prefix+row/heads+1;
    for (int d=0; d<HD; ++d) {
      double sum=0;
      for (int j=0; j<count; ++j) sum+=double(v[(size_t(j)*kvHeads+kvh)*HD+d]);
      require(near(out[size_t(row)*HD+d],sum/count), "merged dimension average");
    }
    guards(outputStorage);
  }
  guards(partialStorage);
}
static void allEmpty() {
  constexpr int splits=4;
  std::vector<float> slots(splits*SLOT,0.f), output(HD+2,NAN_VALUE);
  output.front()=output.back()=GUARD;
  for (int p=0;p<splits;++p) slots[p*SLOT]=-INFINITY;
  blockIdx.x=0;
  for (int tid=0;tid<FLASH_THREADS;++tid) {
    threadIdx.x=tid;
    k_qwen38_splitkv_lse_merge_hd256(slots.data(),output.data()+1,1,1,splits);
  }
  for (int d=0;d<HD;++d) require(output[d+1]==0.f,"all-empty merge");
  guards(output);
}
int main() {
  try {
    // First, nonempty data isolates the original per-lane register-index defect.
    panel(3,5,4,2,4,false); // uneven partitions, multiple rows, GQA mapping
    panel(1,0,2,1,4,false); // three empty leading partitions, one nonempty
    panel(2,0,2,2,1,true);  // single split and no head grouping
    panel(2,1023,4,2,1,true); // immediate threshold predecessor
    panel(2,1024,24,4,2,true); // actual production HD256 head geometry
    panel(2,2048,4,2,3,true);
    panel(1,3072,4,2,4,true);
    allEmpty(); // defensive merge branch, not a valid public causal input
    require(emptyPartitions>0 && dimensions>0 && scalarLanes>0,"fixture coverage");
    std::printf("split-KV layout controls passed; dimensions=%llu empty=%llu lanes=%llu\n", dimensions,emptyPartitions,scalarLanes);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what());
    return 1;
  }
}
`
