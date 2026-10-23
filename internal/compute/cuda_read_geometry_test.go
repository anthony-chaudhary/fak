package compute

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Metadata-only admission and production-callsite checks; no CUDA or large allocation.
// Unmeasured estimate.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAReadGeometryAdmission(t *testing.T) {
	t.Parallel()
	t.Run("refusals", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			dtype Dtype
			shape []int
			bytes int
		}{
			{"f16-storage-is-not-f32", F16, []int{4}, 8},
			{"quantized-storage-is-not-f32", Q8_0, []int{32}, 32},
			{"short-allocation", F32, []int{4}, 12},
			{"negative-shape", F32, []int{-1}, 4},
			{"negative-shape-after-zero", F32, []int{0, -1}, 0},
			{"negative-allocation", F32, []int{0}, -1},
			{"shape-product-overflow", F32, []int{math.MaxInt, math.MaxInt}, math.MaxInt},
			{"byte-size-overflow", F32, []int{math.MaxInt/4 + 1}, math.MaxInt},
		} {
			t.Run(tc.name, func(t *testing.T) {
				defer func() {
					got := recover()
					err, ok := got.(*CUDAOpError)
					if !ok || err.Op != "Read" || err.Site != "read-geometry" || err.Class != MemoryActivation {
						t.Fatalf("refusal = %#v, want typed Read geometry error", got)
					}
				}()
				requireCUDAReadGeometry(Tensor{Dtype: tc.dtype, Shape: tc.shape}, tc.bytes, MemoryActivation)
			})
		}
	})
	t.Run("valid-extents", func(t *testing.T) {
		for _, tc := range []struct {
			shape    []int
			bytes    int
			elements int
		}{
			{nil, 4, 1},
			{[]int{2, 3}, 24, 6},
			{[]int{2}, 32, 2},   // a view may use less than the backing allocation
			{[]int{0, 3}, 0, 0}, // empty pointerless views retain their row width
		} {
			elements, bytes := requireCUDAReadGeometry(Tensor{Dtype: F32, Shape: tc.shape}, tc.bytes, MemoryActivation)
			if elements != tc.elements || bytes != tc.elements*4 {
				t.Fatalf("shape %v: got %d elements/%d bytes, want %d/%d", tc.shape, elements, bytes, tc.elements, tc.elements*4)
			}
		}
	})
	t.Run("entrypoint-order", func(t *testing.T) {
		raw, err := os.ReadFile("cuda.go")
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, "cuda.go", raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		var body string
		var readBody *ast.BlockStmt
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if ok && fn.Name.Name == "Read" && fn.Recv != nil && fn.Body != nil {
				readBody = fn.Body
				body = string(raw[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
			}
		}
		previous := -1
		for _, step := range []string{
			"if hb, ok := t.buf.(*hostBuf); ok",
			"if !ok || db == nil || t.be != c",
			"db.invalidStateError(\"buffer\")",
			"elements, nbytes := requireCUDAReadGeometry(t, db.n, db.class)",
			"if db.ptr == nil && (nbytes != 0 || db.n != 0)",
			"if elements > 0",
			"c.faultLatch.Admit(\"read-d2h\")",
			"make([]float32, elements)",
			"C.fak_cuda_read_d2h(",
			"if status != 0",
			"c.faultLatch.Observe(DeviceFaultUnknown, \"read-d2h\", status)",
			"if db.device == 0",
			"atomic.AddUint64(&c.fenceGen, 1)",
		} {
			at := strings.Index(body, step)
			if at <= previous {
				t.Fatalf("Read does not validate before allocation/copy/fence at %q", step)
			}
			previous = at
		}
		if strings.Count(body, "C.size_t(nbytes)") != 1 || strings.Contains(body, "t.Numel()") || strings.Contains(body, "C.fcuda_d2h") {
			t.Fatal("Read must use one checked native copy with the checked byte extent")
		}
		// Pin control flow, not just token ordering: a failed copy must panic before
		// the sole generation update, and a rank read must never reach that update.
		nodeText := func(node ast.Node) string {
			return string(raw[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset])
		}
		var admission, transfer *ast.IfStmt
		for _, stmt := range readBody.List {
			branch, ok := stmt.(*ast.IfStmt)
			if !ok {
				continue
			}
			switch nodeText(branch.Cond) {
			case "elements > 0":
				admission = branch
			case "len(out) > 0":
				transfer = branch
			}
		}
		if admission == nil || admission.Else != nil || len(admission.Body.List) != 1 {
			t.Fatal("only nonempty device reads may enter latch admission")
		}
		admit, ok := admission.Body.List[0].(*ast.IfStmt)
		if !ok || admit.Init == nil || nodeText(admit.Init) != `err := c.faultLatch.Admit("read-d2h")` || nodeText(admit.Cond) != "err != nil" || admit.Else != nil || len(admit.Body.List) != 1 || nodeText(admit.Body.List[0]) != "panic(err)" {
			t.Fatal("poisoned nonempty reads must panic before allocation/copy")
		}
		if transfer == nil || transfer.Else != nil || len(transfer.Body.List) != 3 {
			t.Fatal("nonempty transfer must contain checked call, failure refusal, then device-scoped fence")
		}
		call, ok := transfer.Body.List[0].(*ast.AssignStmt)
		if !ok || !strings.HasPrefix(nodeText(call), "status := int(C.fak_cuda_read_d2h(") {
			t.Fatal("Read must retain the native operation's direct status")
		}
		failure, ok := transfer.Body.List[1].(*ast.IfStmt)
		if !ok || nodeText(failure.Cond) != "status != 0" || failure.Else != nil || len(failure.Body.List) != 2 || nodeText(failure.Body.List[0]) != `c.faultLatch.Observe(DeviceFaultUnknown, "read-d2h", status)` {
			t.Fatal("every native read failure must directly observe the latch")
		}
		refusal, ok := failure.Body.List[1].(*ast.ExprStmt)
		if !ok {
			t.Fatal("native failure must unconditionally panic typed, including with a nil latch")
		}
		panicCall, ok := refusal.X.(*ast.CallExpr)
		if !ok || nodeText(panicCall.Fun) != "panic" || len(panicCall.Args) != 1 || !strings.HasPrefix(nodeText(panicCall.Args[0]), "&CUDAOpError{") {
			t.Fatal("native failure must unconditionally panic a nonnil CUDAOpError")
		}
		fence, ok := transfer.Body.List[2].(*ast.IfStmt)
		if !ok || nodeText(fence.Cond) != "db.device == 0" || fence.Else != nil || len(fence.Body.List) != 1 || nodeText(fence.Body.List[0]) != "atomic.AddUint64(&c.fenceGen, 1)" || strings.Count(body, "atomic.AddUint64(&c.fenceGen, 1)") != 1 {
			t.Fatal("only a successful device-0 copy may advance the backend generation")
		}
	})
}

// Executes the literal checked host body with deterministic CUDA status injection.
// This is not a Go/cgo Read, CUDA compilation, recovery, or physical GPU witness.
// Unmeasured estimate.
// fak-test:runtime medium est=1s lane=default
func TestCUDAReadNativeStatusBoundary(t *testing.T) {
	t.Parallel()
	var compiler string
	for _, name := range []string{"c++", "g++", "clang++"} {
		if path, err := exec.LookPath(name); err == nil {
			compiler = path
			break
		}
	}
	if compiler == "" {
		t.Skip("host C++ compiler unavailable; checked CUDA read witness not run")
	}
	raw, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	start := strings.Index(source, `extern "C" int fak_cuda_read_d2h(`)
	if start < 0 {
		t.Fatal("checked CUDA read definition is missing")
	}
	end := strings.Index(source[start:], "// Device-to-device copies stay")
	if end < 0 {
		t.Fatal("checked CUDA read section boundary is missing")
	}
	dir := t.TempDir()
	cpp := filepath.Join(dir, "read-status.cpp")
	if err := os.WriteFile(cpp, []byte(cudaReadStatusMockPrefix+source[start:start+end]+cudaReadStatusMockMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	bin := filepath.Join(dir, "read-status")
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal CUDA read host body: %v\n%s", err, output)
	}
	output, err := exec.CommandContext(ctx, bin).CombinedOutput()
	if err != nil {
		t.Fatalf("checked CUDA read status witness: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "Read restore device 2 failed: injected 74") {
		t.Fatalf("secondary restoration failure lost its diagnostic:\n%s", output)
	}
}

const cudaReadStatusMockPrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
using cudaError_t = int;
const int cudaSuccess = 0, cudaMemcpyDeviceToHost = 2;
static int current, caller, target, get_error, select_error, copy_error, restore_error;
static int set_calls, copy_calls;
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
    calls += 's'; require(device == target, "wrong requested device");
    if (select_error) return select_error;
  } else {
    calls += 'r';
    require(set_calls == 2 && device == caller, "wrong restoration device/count");
    if (restore_error) return restore_error;
  }
  current = device; return 0;
}
static int cudaMemcpy(void *h, const void *d, size_t n, int kind) {
  calls += 'c'; ++copy_calls;
  require(current == target && kind == cudaMemcpyDeviceToHost, "copy used wrong device/direction");
  if (copy_error) return copy_error;
  memcpy(h, d, n); return 0;
}
static const char *cudaGetErrorString(int status) {
  static char message[32];
  snprintf(message, sizeof(message), "injected %d", status);
  return message;
}
`

const cudaReadStatusMockMain = `
int main() {
  for (int requested = 0; requested <= 1; ++requested) {
    for (int fault = 0; fault < 7; ++fault) {
      caller = current = 2; target = requested;
      get_error = fault == 1 ? 71 : 0;
      select_error = fault == 2 || fault == 5 ? 72 : 0;
      copy_error = fault == 3 || fault == 6 ? 73 : 0;
      restore_error = fault >= 4 ? 74 : 0;
      set_calls = copy_calls = 0; calls.clear(); g_host_bytes = 7;
      const int source[4] = {1, -2, 3, -4};
      int destination[4] = {9, 9, 9, 9};
      const int status = fak_cuda_read_d2h(target, destination, source, sizeof(source));
      const int want = get_error ? get_error : select_error ? select_error : copy_error ? copy_error : restore_error;
      require(status == want, "checked read lost the primary native status");
      require(calls == (get_error ? "g" : select_error ? "gsr" : "gscr"), "wrong capture/select/copy/restore order");
      require(set_calls == (get_error ? 0 : 2), "read did not restore after selection failure");
      require(copy_calls == (get_error || select_error ? 0 : 1), "read copied after failed admission");
      const bool copied = !get_error && !select_error && !copy_error;
      require(g_host_bytes == 7 + (copied ? sizeof(source) : 0), "transfer bytes disagree with successful copy");
      require(copied ? memcmp(destination, source, sizeof(source)) == 0 : destination[0] == 9, "failed copy published data");
      require(current == (restore_error && !select_error ? target : caller), "caller device restoration changed");
    }
  }
}
`
