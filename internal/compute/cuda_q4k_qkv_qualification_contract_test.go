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

// Exact native wrapper + existing checked-finish implementation, with recording
// submission/ownership/runtime stubs. This is control flow, not device proof.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles source-extracted C++ qualification and finish helpers.
func TestCUDAQ4KQKVQualificationDrainContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	extract := func(source, sig string) string {
		t.Helper()
		start := strings.Index(source, sig)
		if start < 0 || strings.Count(source, sig) != 1 {
			t.Fatalf("missing/ambiguous function %q", sig)
		}
		open := strings.Index(source[start:], "{") + start
		depth := 0
		for i := open; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatal("unterminated function")
		return ""
	}
	stream := read("cuda/v41_checked_stream.cuh")
	finish := extract(stream, "static int v41_checked_stream_error(") + "\n" + extract(stream, "static int v41_checked_stream_finish(")
	wrapper := read("cuda/q4k_shared_input_qualification.cuh")
	goAdapter := read("cuda_q4k_qkv_qualification.go")
	for _, required := range []string{"t.Backend() != c", "b.device != 0", "b.managed", "b.n != bytes", "b.be != c", "if c.capturing", "CUDAQ4KDecodeArmInUse() != CUDAQ4KDecodeMMVQ", "c.devTrDeviceOnly", "C.fcuda_q4k_qkv_qualification_finish()", "c.releaseTransientBuffers(buffers)", "atomic.AddUint64(&c.fenceGen, 1)", "c.releaseTransientBuffers([]*cudaBuf{scratch})"} {
		if !strings.Contains(goAdapter, required) {
			t.Fatalf("missing adapter ownership boundary %q", required)
		}
	}
	if strings.Index(goAdapter, "if c.capturing") > strings.Index(goAdapter, "c.devTrDeviceOnly") {
		t.Fatal("capture must decline before allocation")
	}
	if strings.Index(goAdapter, "atomic.AddUint64(&c.fenceGen, 1)") < strings.Index(goAdapter, "status = int(C.fcuda_q4k_qkv_qualification_f32(") {
		t.Fatal("readiness published before checked operation")
	}
	// The reused terminal seam must guard both deallocation routes before any
	// allocation is returned to a pool or device free. It covers earlier work too.
	for path, sig := range map[string]string{"cuda/runtime_memory.cuh": `extern "C" void fcuda_free(`, "cuda/runtime_services.cuh": `extern "C" void fcuda_free_on(`} {
		body := extract(read(path), sig)
		if !strings.Contains(body, "if (v41_candidates_stream_uncertain()) return;") {
			t.Fatalf("terminal native storage retention missing in %s", path)
		}
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable")
	}
	dir := t.TempDir()
	cpp, executable := filepath.Join(dir, "qualification.cpp"), filepath.Join(dir, "qualification.exe")
	if err := os.WriteFile(cpp, []byte(q4kQualificationPreamble+finish+wrapper+q4kQualificationMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", executable).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, executable).CombinedOutput(); err != nil {
		t.Fatalf("control-flow witness: %v\n%s", err, output)
	}
}

const q4kQualificationPreamble = `
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#define FAK_CUDA_V41_CHECKED_STREAM
#define FCUDA_Q4K_MMVQ_REFUSED 80006
using cudaError_t=int;
static const int cudaSuccess=0;
static bool g_capture_open=false, g_v41_checked_stream_uncertain=false;
static int g_stream=1, state=0, submit=0, syncStatus=0, ownedStatus=0;
static int submits=0, syncs=0, owned=0;
static int v41_checked_stream_state() { return state; }
static int v41_checked_stream_owned_bytes(const void *,uint64_t) { ++owned; return ownedStatus; }
static int cudaStreamSynchronize(int) { ++syncs; return syncStatus; }
namespace fak_q4k_shared_input {
static bool valid_ranges(const uint8_t *const[3],const float *,float *const[3],const int[3],int,void *,size_t) { return true; }
}
static int fcuda_q4k_qkv_shared_input_f32(const uint8_t *,const uint8_t *,const uint8_t *,const float *,float *,float *,float *,int,int,int,int,void *,size_t) { ++submits; return submit; }
`
const q4kQualificationMain = `
int main() {
    uint8_t w[144]={}; float x[256]={}, y[3]={}; unsigned char scratch[288]={};
    auto run=[&]() { return fcuda_q4k_qkv_qualification_f32(w,w,w,x,y,y+1,y+2,1,1,1,256,scratch,288); };
    g_capture_open=true; assert(run()==80006 && submits==0 && syncs==0 && owned==0); g_capture_open=false;
    ownedStatus=-2; assert(run()==-2 && submits==0 && syncs==0); ownedStatus=0; owned=0;
    assert(run()==0 && submits==1 && syncs==1 && owned==8 && !g_v41_checked_stream_uncertain);
    assert(fcuda_q4k_qkv_qualification_operations()==1);
    submit=70007; assert(run()==70007 && submits==2 && syncs==2 && !g_v41_checked_stream_uncertain);
    submit=0; syncStatus=9; assert(run()==50009 && submits==3 && syncs==3 && g_v41_checked_stream_uncertain);
    // Real finish retains terminal uncertainty rather than reporting a fresh drain.
    assert(fcuda_q4k_qkv_qualification_finish()==-3 && syncs==3);
    assert(fcuda_q4k_qkv_qualification_operations()==1);
    return 0;
}
`
