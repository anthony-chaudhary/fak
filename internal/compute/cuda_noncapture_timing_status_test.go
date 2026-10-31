package compute

import (
	"context"
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

// Real profile implementation plus exact changed call-site statements; only
// CUDA APIs and kernel execution are simulated. This is not CUDA/GPU proof.
// fak-test:runtime medium est=2s lane=default
func TestCUDANonCaptureTimingPreservesOperationStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	// Extract complete actual macro calls, including status capture before end.
	wrappers := func(source, name string) []string {
		t.Helper()
		needle := "FCUDA_TS(\"" + name + "\""
		var result []string
		for off := 0; ; {
			at := strings.Index(source[off:], needle)
			if at < 0 {
				break
			}
			at += off
			start := at + len("FCUDA_TS")
			depth, end := 0, -1
			quoted, escaped := false, false
			for i := start; i < len(source); i++ {
				c := source[i]
				if quoted {
					if escaped {
						escaped = false
					} else if c == '\\' {
						escaped = true
					} else if c == '"' {
						quoted = false
					}
					continue
				}
				if c == '"' {
					quoted = true
					continue
				}
				if c == '(' {
					depth++
				}
				if c == ')' {
					depth--
					if depth == 0 {
						end = i + 1
						break
					}
				}
			}
			if end < 0 {
				t.Fatalf("unterminated %s", name)
			}
			result = append(result, source[at:end]+";")
			off = end
		}
		return result
	}
	header := read("cuda_backend.h")
	start := strings.Index(header, "typedef struct fak_cuda_ts_row {")
	stop := strings.Index(header, "} fak_cuda_ts_totals;")
	if start < 0 || stop < start {
		t.Fatal("profile ABI typedefs missing")
	}
	program := cudaNonCaptureTimingHarness + "\n" + header[start:stop+len("} fak_cuda_ts_totals;")] +
		"\n" + read("cuda/kernel_profile.cuh") + "\n"
	launch := regexp.MustCompile(`\bv41_eg_[a-z0-9_]+<<<[^;]+;`)
	count := 0
	for _, tc := range []struct {
		path, name, result string
		want, kind         int
	}{
		{"cuda/q2k_mmq.cuh", "fill", "init_status", 1, 0},
		{"cuda/q3k_mmq.cuh", "fill", "init_status", 1, 0},
		{"cuda/q8_mmq.cuh", "fill", "init_status", 1, 0},
		{"cuda_v41_routed_mmid.cu", "readback", "e", 1, 2},
		{"cuda_flashmla_sparse_decode_owner.cuh", "h2d_upload", "e", 1, 1},
		{"cuda_flashmla_sparse_decode_owner.cuh", "readback", "e", 2, 2},
		{"cuda/v41_embed_gather.cuh", "v41_eg_q5", "launch_status", 1, 3},
		{"cuda/v41_embed_gather.cuh", "v41_eg_q2", "launch_status", 1, 3},
		{"cuda/v41_embed_gather.cuh", "v41_eg_f32", "launch_status", 1, 3},
		{"cuda/v41_embed_gather.cuh", "v41_eg_iq4nl", "launch_status", 1, 3},
		{"cuda/v41_embed_gather.cuh", "v41_eg_iq4nl_row", "launch_status", 1, 3},
		{"cuda/v41_embed_gather.cuh", "v41_eg_iq4nl_poison", "launch_status", 1, 3},
	} {
		calls := wrappers(read(tc.path), tc.name)
		if len(calls) != tc.want {
			t.Fatalf("%s %s calls=%d want=%d", tc.path, tc.name, len(calls), tc.want)
		}
		for _, call := range calls {
			if tc.kind == 3 {
				if len(launch.FindAllString(call, -1)) != 1 || !strings.Contains(call, "launch_status=cudaPeekAtLastError()") {
					t.Fatalf("kernel must capture its status inside actual wrapper: %s", call)
				}
				// The sole transformed span is native kernel execution. Real macro,
				// profile begin/end, and real adjacent status capture remain intact.
				call = launch.ReplaceAllString(call, "mock_launch();")
			}
			program += fmt.Sprintf("\nstatic int route%d() { expected_kind=%d;\n", count, tc.kind) +
				"cudaError_t init_status=-999,e=-999,launch_status=-999;\n" +
				call + "\nreturn " + tc.result + ";\n}\n"
			count++
		}
	}
	if count != 13 {
		t.Fatalf("instrumented site count=%d", count)
	}
	program += "\nusing Route=int(*)();\nstatic Route routes[]={"
	for i := 0; i < count; i++ {
		program += fmt.Sprintf("route%d,", i)
	}
	program += "};\n" + cudaNonCaptureTimingMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "timing.cpp"), filepath.Join(dir, "timing")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile real profile/calls: %v\n%s", err, out)
	}
	for _, enabled := range []string{"0", "1"} {
		cmd := exec.CommandContext(ctx, binary)
		for _, v := range os.Environ() {
			if !strings.HasPrefix(v, "FAK_CUDA_TIMESTAMP_PROFILE=") {
				cmd.Env = append(cmd.Env, v)
			}
		}
		cmd.Env = append(cmd.Env, "FAK_CUDA_TIMESTAMP_PROFILE="+enabled)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("profile=%s status preservation: %v\n%s", enabled, err, out)
		}
	}
}

const cudaNonCaptureTimingHarness = `
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <unordered_map>
#include <vector>
using cudaError_t=int;
using cudaEvent_t=void*;
using cudaStream_t=void*;
using cudaGraphExec_t=void*;
using cudaMemcpyKind=int;
static constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
static bool g_capture_open=false;
static cudaStream_t g_stream=(void*)1,cudaStreamLegacy=(void*)2;
static int last_error=0,operation_error=0,operations=0,event_records=0,fail_record=0,expected_kind=0;
static int clears=0;
static const char *cudaGetErrorString(int) { return "mock"; }
static cudaError_t cudaPeekAtLastError() { return last_error; }
static cudaError_t cudaGetLastError() { ++clears; int e=last_error;last_error=0;return e; }
static cudaError_t cudaEventCreate(cudaEvent_t *e) { *e=(void*)3;return 0; }
static cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
static cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t stream) {
 assert(stream==g_stream);++event_records;
 if(event_records==fail_record) return last_error=700;
 return 0;
}
static cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
static cudaError_t cudaEventElapsedTime(float *ms,cudaEvent_t,cudaEvent_t) { *ms=1;return 0; }
static cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
static cudaError_t cudaDeviceSynchronize() { return 0; }
static cudaError_t cudaGraphLaunch(cudaGraphExec_t,cudaStream_t) { return 0; }
static cudaError_t cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind) { return 0; }
static cudaError_t mock_operation(int kind,cudaStream_t stream) {
 assert(kind==expected_kind && stream==g_stream);
 ++operations;
 // Deliberately simulate an operation setting its own last error; event-end
 // failure may overwrite it. Only the explicit operation result is guaranteed.
 last_error=operation_error;
 return operation_error;
}
static cudaError_t cudaMemsetAsync(void*,int,size_t,cudaStream_t stream) { return mock_operation(0,stream); }
static cudaError_t cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind kind,cudaStream_t stream) { return mock_operation(kind,stream); }
static void mock_launch() { (void)mock_operation(3,g_stream); }
struct block_q8_1_mmq { char value[4]; };
static char storage[32768];
static void *y=storage,*image=storage;
static size_t y_bytes=4096;
static const void *sources[]={storage};
static size_t offsets[]={0},sizes[]={16};
static void *device[14]={storage},*host[10]={storage};
static size_t bytes[14]={16};
static int i=0,blocks=2;
`

const cudaNonCaptureTimingMain = `
int main() {
 for(Route route:routes) {
  for(int op:{0,17}) for(int fail:{0,1,2}) {
   g_ts_pending.clear();g_ts_batch_open=false;
   operation_error=op;fail_record=fail;operations=event_records=clears=0;last_error=0;
   int result=route();
   assert(operations==1 && result==op);
   if(!g_ts_on) assert(event_records==0 && clears==0);
   else {
    assert(event_records==(fail==1?1:2));
    // Real stop-event handling must not consume a previously pending op fault.
    // It may be overwritten by the fake event API; no universal slot guarantee.
    if(op && fail==2) assert(last_error==700 && clears==0);
    if(!op && fail==2) assert(last_error==0 && clears==1);
   }
  }
 }
 return 0;
}
`
