package compute

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Executes the literal native host section with only kernel launch syntax replaced,
// and pins the actual Go refusal/fence control flow. This does not compile CUDA/cgo,
// execute either kernel, or establish physical GPU correctness or recovery.
// Unmeasured estimate.
// fak-test:runtime medium est=1s lane=default
func TestCUDADSASelectionCompletion(t *testing.T) {
	t.Parallel()
	t.Run("go-completion-before-fence", func(t *testing.T) {
		raw, err := os.ReadFile("cuda_specialized.go")
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, "cuda_specialized.go", raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		text := func(n ast.Node) string { return string(raw[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]) }
		var body *ast.BlockStmt
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Recv != nil && fn.Name.Name == "DSAIndexSelect" {
				body = fn.Body
			}
		}
		if body == nil {
			t.Fatal("DSAIndexSelect missing")
		}
		fenceAt, declineAt := strings.Index(text(body), "atomic.AddUint64(&c.fenceGen, 1)"), strings.Index(text(body), "if n < 0")
		if fenceAt < 0 || declineAt < 0 || fenceAt < declineAt {
			t.Fatal("nonexecuting native decline advances the readiness fence before returning")
		}
		previous := -1
		for _, step := range []string{
			"if nKeys <= 0 || topK <= 0 || queryPos < 0",
			"if nKeys > 4096",
			"if c.capturing",
			`c.faultLatch.Admit("dsa-index-select")`,
			"nH > math.MaxInt32-255",
			"indexDim > math.MaxInt/F32.Bytes()/nH",
			"indexDim > math.MaxInt/F32.Bytes()/nKeys",
			"tensor.be != c || db.device != 0",
			`db.invalidStateError(name)`,
			"tensor.Dtype != F32 || db.ptr == nil || db.n < 0 || elements > db.n/F32.Bytes()",
			"status := int(C.fak_cuda_dsa_index_select_f32(",
			"if status != 0",
			`c.faultLatch.Observe(DeviceFaultUnknown, "dsa-index-select", status)`,
			"if n < 0",
			"if n != topK",
			"atomic.AddUint64(&c.fenceGen, 1)",
			"return sel",
		} {
			at := strings.Index(text(body), step)
			if at <= previous {
				t.Fatalf("DSA must refuse absent or failed work before publishing a fence at %q", step)
			}
			previous = at
		}
		branches := map[string]*ast.IfStmt{}
		for _, stmt := range body.List {
			if branch, ok := stmt.(*ast.IfStmt); ok {
				branches[text(branch.Cond)] = branch
			}
		}
		for _, condition := range []string{"nKeys <= 0 || topK <= 0 || queryPos < 0", "nKeys > 4096", "n < 0"} {
			branch := branches[condition]
			if branch == nil || branch.Else != nil || len(branch.Body.List) != 1 || text(branch.Body.List[0]) != "return nil" {
				t.Fatalf("nonexecuting path %q must return before the fence", condition)
			}
		}
		capture := branches["c.capturing"]
		if capture == nil || capture.Else != nil || len(capture.Body.List) != 1 || !strings.HasPrefix(text(capture.Body.List[0]), "panic(&CUDAOpError{") || !strings.Contains(text(capture.Body.List[0]), `Site: "dsa-index-capture"`) {
			t.Fatal("capture refusal must not call CUDA, poison the latch, or publish a fence")
		}
		failed := branches["status != 0"]
		if failed == nil || failed.Else != nil || len(failed.Body.List) != 2 || text(failed.Body.List[0]) != `c.faultLatch.Observe(DeviceFaultUnknown, "dsa-index-select", status)` || !strings.HasPrefix(text(failed.Body.List[1]), "panic(&CUDAOpError{") || !strings.Contains(text(failed.Body.List[1]), "Code: status") {
			t.Fatal("native failure must latch and unconditionally panic typed before the fence")
		}
		invalid := branches["n != topK"]
		if invalid == nil || invalid.Else != nil || len(invalid.Body.List) != 1 || !strings.HasPrefix(text(invalid.Body.List[0]), "panic(&CUDAOpError{") {
			t.Fatal("invalid native counts must not be truncated into plausible selections")
		}
		if strings.Count(text(body), "C.fak_cuda_dsa_index_select_f32(") != 1 || strings.Count(text(body), "atomic.AddUint64(&c.fenceGen, 1)") != 1 || strings.Contains(text(body), "C.fcuda_dsa_index_select_f32(") {
			t.Fatal("DSA must have one checked call and one success-only fence")
		}
	})
	t.Run("literal-native-status-and-selection", func(t *testing.T) {
		var compiler string
		for _, name := range []string{"c++", "g++", "clang++"} {
			if path, err := exec.LookPath(name); err == nil {
				compiler = path
				break
			}
		}
		if compiler == "" {
			t.Skip("host C++ compiler unavailable; DSA host-body witness not run")
		}
		raw, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		start := strings.Index(source, "#define DSA_TOPK_MAX_KEYS 4096")
		if start < 0 {
			t.Fatal("DSA host section missing")
		}
		end := strings.Index(source[start:], "// ---- argmax")
		if end < 0 {
			t.Fatal("DSA host section boundary missing")
		}
		body := source[start : start+end]
		// Accept either historical or checked host text so this same witness gives
		// a behavioral RED on the pre-fix body, not merely a missing-symbol error.
		for _, kernel := range []string{"k_dsa_index_score", "k_dsa_index_topk"} {
			prefix := kernel + "<<<"
			at := strings.Index(body, prefix)
			if at < 0 || strings.Count(body, prefix) != 1 {
				t.Fatalf("expected exactly one %s launch", kernel)
			}
			launchEnd := strings.Index(body[at:], ">>>")
			if launchEnd < 0 {
				t.Fatal("unterminated kernel launch")
			}
			config := body[at+len(prefix) : at+launchEnd]
			body = body[:at] + "mock_launch_config(" + config + "), mock_" + kernel + body[at+launchEnd+3:]
		}
		prefix := cudaDSAStatusPrefix
		if strings.Contains(body, `extern "C" int fak_cuda_dsa_index_select_f32(`) {
			prefix += "\n#define CHECKED_DSA 1\n"
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "dsa-status.cpp"), filepath.Join(dir, "dsa-status")
		if err := os.WriteFile(cpp, []byte(prefix+body+cudaDSAStatusMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal DSA host body: %v\n%s", err, output)
		}
		if output, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("DSA completion witness: %v\n%s", err, output)
		} else if !strings.Contains(string(output), "DSAIndexSelect restore device 2 failed: injected 99") {
			t.Fatalf("secondary restoration failure diagnostic missing:\n%s", output)
		}
	})
}

const cudaDSAStatusPrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unordered_map>
#include <vector>
#include <string>
using cudaError_t = int;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorStreamCaptureUnsupported = 900, cudaMemcpyDeviceToHost = 2, g_stream = 17 };
static bool g_capture_open;
static size_t g_host_bytes;
static std::unordered_map<size_t, std::vector<void *>> g_pool;
static std::unordered_map<void *, size_t> g_live;
static int current, set_calls, alloc_calls, launch_calls, copy_calls, fault, restore_fault, bad_count, partial_copy;
static int *selected, *count;
static std::string calls;
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr, "%s\n", why); exit(1); } }
static inline const char *cudaGetErrorString(int status) { static char b[32]; snprintf(b, sizeof(b), "injected %d", status); return b; }
static inline int cudaGetDevice(int *device) { calls += 'g'; if (fault == 1) return 81; *device = current; return 0; }
static inline int cudaSetDevice(int device) {
  calls += ++set_calls == 1 ? 's' : 'r';
  require(device == (set_calls == 1 ? 0 : 2) && set_calls <= 2, "wrong device scope");
  if (set_calls == 1 && fault == 2) return 82;
  if (set_calls == 2 && restore_fault) return 99;
  current = device; return 0;
}
static inline int cudaMalloc(void **out, size_t bytes) {
  calls += 'a'; ++alloc_calls;
  if (fault == alloc_calls + 2) { *out = reinterpret_cast<void *>(1); return 80 + fault; }
  *out = malloc(bytes); require(*out != nullptr, "host fixture allocation failed"); return 0;
}
static inline void *fcuda_malloc(size_t bytes) {
  void *out = nullptr;
  auto it = g_pool.find(bytes);
  if (it != g_pool.end() && !it->second.empty()) { out = it->second.back(); it->second.pop_back(); }
  else if (cudaMalloc(&out, bytes)) return nullptr;
  g_live[out] = bytes; return out;
}
static inline void fcuda_free(void *out) {
  if (!out) return;
  auto it = g_live.find(out);
  require(it != g_live.end(), "DSA freed unowned/failed allocation output");
  g_pool[it->second].push_back(out); g_live.erase(it);
}
static inline void mock_launch_config(int grid, int threads, int shared, int stream) {
  require((grid == 1 || grid == 2) && threads == 256 && shared == 0 && stream == g_stream, "wrong DSA launch configuration");
}
static inline void mock_k_dsa_index_score(const float *q, const float *k, const float *w, double *scores, int keys, int heads, int dim, int pos, float scale) {
  calls += 'l'; ++launch_calls;
  require(q && k && w && scores && keys == 2 && heads == 1 && dim == 1 && pos == 1 && scale == 1, "wrong score launch");
}
static inline void mock_k_dsa_index_topk(const double *scores, int keys, int pos, int top, int *sel, int *n) {
  calls += 'l'; ++launch_calls;
  require(scores && sel && n && keys == 2 && pos == 1 && top == 1, "wrong top-k launch");
  selected = sel; count = n; *sel = 1; *n = bad_count ? 2 : 1;
}
static inline int cudaGetLastError() { calls += 'e'; return fault == launch_calls + 5 ? 80 + fault : 0; }
static inline int cudaMemcpy(void *out, const void *in, size_t bytes, int kind) {
  calls += 'c'; ++copy_calls;
  require(kind == cudaMemcpyDeviceToHost && bytes == sizeof(int) && in == (copy_calls == 1 ? count : selected), "wrong D2H copy");
  if (fault == copy_calls + 7) {
    if (partial_copy) *static_cast<int *>(out) = -101;
    return 80 + fault;
  }
  memcpy(out, in, bytes); return 0;
}
#define CK(call) do { int status = (call); if (status) fprintf(stderr, "logging-only CUDA failure %d\n", status); } while (0)
static void reset(int fail, int restore) {
  require(g_live.empty(), "DSA leaked live scratch");
  for (auto &entry : g_pool) for (void *p : entry.second) free(p);
  g_pool.clear(); current = 2; set_calls = alloc_calls = launch_calls = copy_calls = 0;
  selected = count = nullptr; fault = fail; restore_fault = restore; bad_count = partial_copy = 0;
  g_capture_open = false; g_host_bytes = 7; calls.clear();
}
`

const cudaDSAStatusMain = `
int main() {
  const float q[1] = {1}, keys[2] = {0, 1}, weights[1] = {1};
  // The old wrapper returns count=1 and untouched index=0 when only the second
  // copy fails. That is a causal, unique, but wrong selection for these scores.
  reset(9, 0); int out = 0;
  int returned = fcuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, 1, 1, 1, &out);
  if (returned != -1) fprintf(stderr, "observed count=%d index=%d; expected best index=1, but its D2H copy failed\n", returned, out);
  require(returned == -1,
          "failed index copy fabricated a valid count and zero selection");
#ifdef CHECKED_DSA
  for (int fail = 0; fail <= 9; ++fail) {
    for (int restore = 0; restore <= 1; ++restore) {
      reset(fail, restore); partial_copy = 1; out = -7; int n = -8;
      int status = fak_cuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, 1, 1, 1, &out, &n);
      require(status == (fail ? 80 + fail : restore ? 99 : 0), "lost primary CUDA failure");
      require(out == (status ? -7 : 1) && n == (status ? -8 : 1), "failed DSA published a result");
      require(g_live.empty(), "DSA leaked scratch");
      require(current == (!restore || fail == 1 || fail == 2 ? 2 : 0), "caller device was not restored");
      size_t copied = fail == 0 ? 2 : fail == 9 ? 1 : 0;
      require(g_host_bytes == 7 + copied * sizeof(int), "counted unsuccessful D2H bytes");
      std::string expected = "g";
      if (fail != 1) {
        expected += 's';
        if (fail != 2) {
          expected += 'a';
          if (fail != 3) {
            expected += 'a';
            if (fail != 4) {
              expected += 'a';
              if (fail != 5) {
                expected += "le";
                if (fail != 6) {
                  expected += "le";
                  if (fail != 7) { expected += 'c'; if (fail != 8) expected += 'c'; }
                }
              }
            }
          }
        }
        expected += 'r';
      }
      require(calls == expected, "executed work after failure or changed status-check order");
    }
  }
  reset(0, 0); out = -7; int n = -8; bad_count = 1;
  require(fak_cuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, 1, 1, 1, &out, &n) == cudaErrorInvalidValue && out == -7 && n == -8 && copy_calls == 1 && g_host_bytes == 7 + sizeof(int), "invalid count was clipped into a plausible selection");
  reset(0, 0); g_capture_open = true;
  require(fak_cuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, 1, 1, 1, &out, &n) == cudaErrorStreamCaptureUnsupported && calls.empty(), "capture submitted host-fence work");
  for (int mode = 0; mode < 4; ++mode) {
    reset(0, 0); out = -7; n = -8;
    int status = fak_cuda_dsa_index_select_f32(q, keys, weights, mode == 0 ? 4097 : mode == 1 ? 0 : 2, 1, 1, mode == 2 ? -1 : 1, mode == 3 ? 0 : 1, 1, &out, &n);
    require(status == 0 && n == (mode == 0 ? -1 : 0) && out == -7 && calls.empty() && g_host_bytes == 7, "decline/empty work touched CUDA or fabricated a completion");
  }
  reset(0, 0); out = -7;
  require(fcuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, INT_MAX, 1, 1, &out) == 1 && out == 1, "successful public ABI or causal clamping changed");
  // Warm pool reuse retains ownership and performs no new device allocation.
  current = 2; set_calls = launch_calls = copy_calls = 0; calls.clear(); out = -7;
  require(fcuda_dsa_index_select_f32(q, keys, weights, 2, 1, 1, 1, 1, 1, &out) == 1 && out == 1 && alloc_calls == 3 && g_live.empty(), "warm scratch ownership changed");
#endif
  reset(0, 0);
}
`
