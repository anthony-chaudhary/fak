package compute

import (
	"errors"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

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
