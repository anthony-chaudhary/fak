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

// Pins the Go refusal/fence control flow and executes the literal native host body
// with only the CUDA launch syntax replaced by a deterministic launch stub.
// This is not CUDA/cgo compilation, kernel correctness, or a physical GPU witness.
// Unmeasured estimate.
// fak-test:runtime medium est=1s lane=default
func TestCUDAArgmaxCheckedStatusBoundary(t *testing.T) {
	t.Parallel()
	t.Run("go-refusal-before-fence", func(t *testing.T) {
		raw, err := os.ReadFile("cuda_specialized.go")
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, "cuda_specialized.go", raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		text := func(node ast.Node) string {
			return string(raw[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset])
		}
		var body *ast.BlockStmt
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "Argmax" && fn.Recv != nil {
				body = fn.Body
			}
		}
		if body == nil {
			t.Fatal("CUDA Argmax method missing")
		}
		previous := -1
		for _, step := range []string{
			`c.faultLatch.Admit("argmax")`,
			"db, ok := logits.buf.(*cudaBuf)",
			"db.device != 0",
			`db.invalidStateError("buffer")`,
			"logits.Dtype != F32 || db.ptr == nil",
			"dim <= 0 || elements > (math.MaxInt32-255)/dim",
			"db.n < 0 || elements > db.n/F32.Bytes()",
			"status := int(C.fak_cuda_argmax_f32(",
			"if status != 0",
			`c.faultLatch.Observe(DeviceFaultUnknown, "argmax", status)`,
			"atomic.AddUint64(&c.fenceGen, 1)",
			"return int(id)",
		} {
			at := strings.Index(text(body), step)
			if at <= previous {
				t.Fatalf("Argmax must refuse failed work before publishing a fence at %q", step)
			}
			previous = at
		}
		var admit, device, failed *ast.IfStmt
		var checked, fenced int
		for _, stmt := range body.List {
			if branch, ok := stmt.(*ast.IfStmt); ok {
				if branch.Init != nil && text(branch.Init) == `err := c.faultLatch.Admit("argmax")` {
					admit = branch
				}
				if text(branch.Cond) == "status != 0" {
					failed = branch
				}
				if text(branch.Cond) == "!ok || db == nil || logits.be != c || db.device != 0" {
					device = branch
				}
			}
			if strings.HasPrefix(text(stmt), "status := int(C.fak_cuda_argmax_f32(") {
				checked++
			}
			if text(stmt) == "atomic.AddUint64(&c.fenceGen, 1)" {
				fenced++
			}
		}
		if admit == nil || text(admit.Cond) != "err != nil" || admit.Else != nil || len(admit.Body.List) != 1 || text(admit.Body.List[0]) != "panic(err)" {
			t.Fatal("a poisoned session must panic before any native Argmax work")
		}
		if device == nil || device.Else != nil || len(device.Body.List) != 1 || !strings.HasPrefix(text(device.Body.List[0]), "panic(&CUDAOpError{") {
			t.Fatal("foreign or non-device-0 logits must be refused before the native call")
		}
		if failed == nil || failed.Else != nil || len(failed.Body.List) != 2 || text(failed.Body.List[0]) != `c.faultLatch.Observe(DeviceFaultUnknown, "argmax", status)` {
			t.Fatal("every native failure must observe the latch")
		}
		refusal, ok := failed.Body.List[1].(*ast.ExprStmt)
		if !ok {
			t.Fatal("native Argmax failure must unconditionally panic")
		}
		call, ok := refusal.X.(*ast.CallExpr)
		if !ok || text(call.Fun) != "panic" || len(call.Args) != 1 || !strings.HasPrefix(text(call.Args[0]), "&CUDAOpError{") || !strings.Contains(text(call.Args[0]), "Code: status") {
			t.Fatal("native failure must panic typed, even with a nil latch")
		}
		if checked != 1 || fenced != 1 || strings.Count(text(body), "atomic.AddUint64(&c.fenceGen, 1)") != 1 || strings.Contains(text(body), "C.fcuda_argmax_f32(") || strings.Contains(text(body), "logits.Numel()") || !strings.Contains(text(body), "C.int(elements), &id") {
			t.Fatal("Argmax must use one checked call followed by one success-only fence")
		}
	})
	t.Run("native-status-and-device", func(t *testing.T) {
		var compiler string
		for _, name := range []string{"c++", "g++", "clang++"} {
			if path, err := exec.LookPath(name); err == nil {
				compiler = path
				break
			}
		}
		if compiler == "" {
			t.Skip("host C++ compiler unavailable; checked CUDA Argmax witness not run")
		}
		raw, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		start := strings.Index(source, "static int *g_argmax_idx = nullptr;")
		if start < 0 {
			t.Fatal("Argmax native host section missing")
		}
		end := strings.Index(source[start:], "// ---- AWQ")
		if end < 0 {
			t.Fatal("Argmax native host section boundary missing")
		}
		body := source[start : start+end]
		if !strings.Contains(body, `extern "C" int fak_cuda_argmax_f32(`) {
			t.Fatal("checked CUDA Argmax definition missing")
		}
		launch := "k_argmax<<<1, 256, 0, g_stream>>>(dLogits, n, g_argmax_idx);"
		if strings.Count(body, launch) != 1 {
			t.Fatal("expected one unchanged Argmax kernel launch")
		}
		body = strings.Replace(body, launch, "mock_argmax_launch(dLogits, n, g_argmax_idx, g_stream);", 1)
		dir := t.TempDir()
		cpp := filepath.Join(dir, "argmax-status.cpp")
		if err := os.WriteFile(cpp, []byte(cudaArgmaxStatusPrefix+body+cudaArgmaxStatusMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		bin := filepath.Join(dir, "argmax-status")
		if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal Argmax host body: %v\n%s", err, output)
		}
		if output, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("checked Argmax status witness: %v\n%s", err, output)
		} else if !strings.Contains(string(output), "Argmax restore device 2 failed: injected 76") {
			t.Fatalf("secondary restoration failure lost its diagnostic:\n%s", output)
		}
	})
}

const cudaArgmaxStatusPrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using cudaError_t = int;
const int cudaSuccess = 0, cudaMemcpyDeviceToHost = 2;
static int current, caller, get_error, select_error, alloc_error, launch_error, copy_error, restore_error;
static int set_calls, storage, g_stream = 17;
static size_t g_host_bytes;
static std::string calls;
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "%s\n", why); exit(1); }
}
static int cudaGetDevice(int *device) {
  calls += 'g';
  if (get_error) return get_error;
  *device = current; return 0;
}
static int cudaSetDevice(int device) {
  if (++set_calls == 1) {
    calls += 's'; require(device == 0, "Argmax did not select backend device 0");
    if (select_error) return select_error;
  } else {
    calls += 'r'; require(set_calls == 2 && device == caller, "wrong restoration device/count");
    if (restore_error) return restore_error;
  }
  current = device; return 0;
}
static int cudaMalloc(int **ptr, size_t n) {
  calls += 'a'; require(current == 0 && n == sizeof(int), "wrong Argmax allocation");
  *ptr = &storage; // A failed creator's output must not become persistent scratch.
  return alloc_error;
}
static void mock_argmax_launch(const float *logits, int n, int *idx, int stream) {
  calls += 'l'; require(current == 0 && logits && n == 4 && idx == &storage && stream == g_stream, "wrong Argmax launch");
  storage = 3;
}
static int cudaGetLastError() { calls += 'e'; return launch_error; }
static int cudaMemcpy(void *h, const void *d, size_t n, int kind) {
  calls += 'c'; require(current == 0 && d == &storage && n == sizeof(int) && kind == cudaMemcpyDeviceToHost, "wrong scalar copy");
  if (copy_error) return copy_error;
  *static_cast<int *>(h) = storage; return 0;
}
static const char *cudaGetErrorString(int status) {
  static char message[32]; snprintf(message, sizeof(message), "injected %d", status); return message;
}
`

const cudaArgmaxStatusMain = `
int main() {
  const float logits[4] = {1, 2, 3, 4};
  for (int warm = 0; warm <= 1; ++warm) {
    for (int fault = 0; fault <= 5; ++fault) {
      if (warm && fault == 3) continue; // Warm scratch performs no allocation.
      for (int restore = 0; restore <= 1; ++restore) {
        caller = current = 2; set_calls = 0; calls.clear(); g_host_bytes = 7;
        get_error = fault == 1 ? 71 : 0; select_error = fault == 2 ? 72 : 0;
        alloc_error = fault == 3 ? 73 : 0; launch_error = fault == 4 ? 74 : 0;
        copy_error = fault == 5 ? 75 : 0; restore_error = restore ? 76 : 0;
        g_argmax_idx = warm ? &storage : nullptr;
        int result = -9;
        const int status = fak_cuda_argmax_f32(logits, 4, &result);
        const int primary = get_error ? get_error : select_error ? select_error : alloc_error ? alloc_error : launch_error ? launch_error : copy_error;
        require(status == (primary ? primary : restore_error), "Argmax lost primary native error");
        std::string expected = "g";
        if (!get_error) {
          expected += 's';
          if (!select_error) {
            if (!warm) expected += 'a';
            if (!alloc_error) {
              expected += "le";
              if (!launch_error) expected += 'c';
            }
          }
          expected += 'r';
        }
        require(calls == expected, "Argmax ran work after a failure or changed device/launch/copy order");
        require(result == (status ? -9 : 3), "failed Argmax published an index");
        require(g_host_bytes == 7 + (primary ? 0 : sizeof(int)), "Argmax counted a failed scalar transfer");
        require(g_argmax_idx == (warm || (!get_error && !select_error && !alloc_error) ? &storage : nullptr), "failed allocation published scratch");
        require(current == (!get_error && !select_error && restore ? 0 : caller), "caller device restoration changed");
      }
    }
  }
  // The unchanged public ABI must retain success results but never return token 0 on failure.
  caller = current = 2; set_calls = 0;
  get_error = select_error = alloc_error = launch_error = copy_error = restore_error = 0;
  require(fcuda_argmax_f32(logits, 4) == 3, "legacy ABI changed successful result");
  caller = current = 2; set_calls = 0; copy_error = 75;
  require(fcuda_argmax_f32(logits, 4) == -1, "legacy ABI fabricated a valid token on failure");
}
`
