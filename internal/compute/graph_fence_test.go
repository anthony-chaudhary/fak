package compute

import (
	"context"
	"embed"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

//go:embed cuda/spec_graph.cuh cuda_backend.h
var graphFenceNativeSources embed.FS

// Compile the actual production function against host callbacks. No CUDA SDK,
// driver, device execution or physical fault injection participates in this test.
// fak-test:runtime medium est=3s lane=default
func TestCUDAGraphFenceNativeStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	header, err := graphFenceNativeSources.ReadFile("cuda_backend.h")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cuda_backend.h"), header, 0600); err != nil {
		t.Fatal(err)
	}
	production, err := graphFenceNativeSources.ReadFile("cuda/spec_graph.cuh")
	if err != nil {
		t.Fatal(err)
	}
	body := cudaFunctionBody(t, string(production), "fcuda_graph_end_launch")
	source := `
#include "cuda_backend.h"
#include <cstdio>
using cudaError_t = int;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
using cudaStream_t = void *;
struct cudaGraphExecUpdateResultInfo { int result; };
static constexpr int cudaSuccess = 0;
static constexpr int cudaErrorLaunchFailure = 719;
static cudaGraphExec_t g_exec = nullptr;
static cudaStream_t g_stream = reinterpret_cast<void *>(1);
static bool g_capture_open = false;
static int g_graph_fence_test_fault = 0;
static int launch_status = 0, fence_status = 0;
static int launched = 0, fenced = 0, destroyed = 0;
static cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *graph) {
  *graph = reinterpret_cast<void *>(2); return 0;
}
static cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec, cudaGraph_t, unsigned long long) {
  *exec = reinterpret_cast<void *>(3); return 0;
}
static cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo *info) {
  info->result = 0; return 0;
}
static cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return 0; }
static cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { ++launched; return launch_status; }
static cudaError_t cudaStreamSynchronize(cudaStream_t) { ++fenced; return fence_status; }
static cudaError_t cudaGraphDestroy(cudaGraph_t) { ++destroyed; return 0; }
static cudaError_t cudaGetLastError() { return 0; }
static const char *cudaGetErrorString(cudaError_t) { return "host callback status"; }
extern "C" int fcuda_graph_end_launch(void)
` + body + `
int main() {
  const int statuses[][2] = {{0, 0}, {1, 0}, {0, 719}, {1, 719}};
  for (const auto &status : statuses) {
    launch_status = status[0]; fence_status = status[1];
    g_capture_open = true;
    int before = launched;
    int rc = fcuda_graph_end_launch();
    if (g_capture_open || launched != before + 1 || fenced != launched || destroyed != launched) return 20;
    std::printf("%d\n", rc);
  }
  return 0;
}
`
	cpp := filepath.Join(dir, "graph-fence.cpp")
	if err := os.WriteFile(cpp, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "graph-fence")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", dir, cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production graph fence: %v\n%s", err, out)
	}
	var stderr strings.Builder
	command := exec.CommandContext(ctx, binary)
	command.Stderr = &stderr
	out, err := command.Output()
	if err != nil {
		t.Fatalf("run production graph fence: %v\n%s", err, stderr.String())
	}
	results := strings.Fields(string(out))
	want := []int{0, graphLaunchFailed, graphFenceFailed, graphFenceFailed}
	if len(results) != len(want) {
		t.Fatalf("native status count %d, want %d", len(results), len(want))
	}
	for i, expected := range want {
		code, err := strconv.Atoi(results[i])
		if err != nil || code != expected {
			t.Fatalf("launch/fence case %d: code=%d, parse=%v, want=%d", i, code, err, expected)
		}
		if code == 0 {
			continue
		}
		site := GraphLaunchSite
		if i >= 2 {
			site = GraphFenceSite
		}
		var launch *CUDALaunchError
		if !errors.As(graphEndLaunchError(code), &launch) || launch.Site != site || launch.Code != code {
			t.Fatalf("case %d: native code %d lost typed site %s", i, code, site)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCUDAGraphFenceCodeRaisesTypedLaunchError(t *testing.T) {
	for _, tc := range []struct {
		code int
		site string
	}{
		{graphEndCaptureFailed, GraphLaunchSite},
		{graphInstantiateFailed, GraphLaunchSite},
		{graphLaunchFailed, GraphLaunchSite},
		{graphFenceFailed, GraphFenceSite},
	} {
		var err error = graphEndLaunchError(tc.code)
		var launch *CUDALaunchError
		if !errors.As(err, &launch) || launch.Op != "GraphEndLaunch" || launch.Site != tc.site || launch.Code != tc.code {
			t.Fatalf("code %d: error %#v, want CUDALaunchError op=GraphEndLaunch site=%s code=%d", tc.code, err, tc.site, tc.code)
		}
		var op *CUDAOpError
		if !errors.As(err, &op) || op.Code != tc.code {
			t.Fatalf("code %d: error does not expose its CUDAOpError", tc.code)
		}
	}
}

// cudaFunctionBody returns the brace-balanced body of the first C function named fn in src.
func cudaFunctionBody(t *testing.T, src, fn string) string {
	t.Helper()
	start := strings.Index(src, fn+"(")
	if start < 0 {
		t.Fatalf("function %s not found", fn)
	}
	open := strings.Index(src[start:], "{")
	if open < 0 {
		t.Fatalf("function %s has no body", fn)
	}
	depth := 0
	for i := start + open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start+open : i+1]
			}
		}
	}
	t.Fatalf("function %s body is unbalanced", fn)
	return ""
}

func readCUDASource(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.ReplaceAll(string(b), "\r\n", "\n")
}

// fak-test:runtime fast est=1s
func TestCUDAGraphCaptureABIContract(t *testing.T) {
	header := readCUDASource(t, "cuda_backend.h")
	for name, want := range map[string]int{
		"FCUDA_GRAPH_END_CAPTURE_FAILED": graphEndCaptureFailed,
		"FCUDA_GRAPH_INSTANTIATE_FAILED": graphInstantiateFailed,
		"FCUDA_GRAPH_LAUNCH_FAILED":      graphLaunchFailed,
		"FCUDA_GRAPH_FENCE_FAILED":       graphFenceFailed,
	} {
		m := regexp.MustCompile(`#define ` + name + ` (\d+)`).FindStringSubmatch(header)
		if m == nil {
			t.Fatalf("cuda_backend.h does not define %s", name)
		}
		if got, _ := strconv.Atoi(m[1]); got != want {
			t.Fatalf("%s = %d in cuda_backend.h, Go mirrors %d", name, got, want)
		}
	}

	rawSync := regexp.MustCompile(`cudaStreamSynchronize\(`)
	end := cudaFunctionBody(t, readCUDASource(t, "cuda/spec_graph.cuh"), "fcuda_graph_end_launch")
	if strings.Contains(end, "CK(cudaStreamSynchronize") || !strings.Contains(end, "FCUDA_GRAPH_FENCE_FAILED") {
		t.Fatal("fcuda_graph_end_launch must return FCUDA_GRAPH_FENCE_FAILED from its fence, not only log it")
	}

	gdn := readCUDASource(t, "cuda/qwen35_gdn.cuh")
	if fence := cudaFunctionBody(t, gdn, "qwen35_gdn_fence"); !strings.Contains(fence, "g_capture_open") {
		t.Fatal("qwen35_gdn_fence must skip the synchronize while a capture is open")
	}
	for _, fn := range []string{"fcuda_qwen35_gdn_decode_f32", "qwen35_gdn_drain_after_error"} {
		body := cudaFunctionBody(t, gdn, fn)
		if rawSync.MatchString(body) || !strings.Contains(body, "qwen35_gdn_fence()") {
			t.Fatalf("%s must fence through qwen35_gdn_fence, with no raw stream synchronize inside a capture", fn)
		}
	}
}
