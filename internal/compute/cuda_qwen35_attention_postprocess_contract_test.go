package compute

import (
	"bytes"
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

// This executes the production header's byte admission and store helpers on the
// host. It does not compile a CUDA kernel, certify CUDA intrinsics, or run a GPU.
// fak-test:runtime fast est=2s lane=default
func TestCUDAQwen35AttentionPostprocessHostContract(t *testing.T) {
	t.Parallel()
	header, err := os.ReadFile("cuda/qwen35_attention_postprocess.cuh")
	if err != nil {
		t.Fatal(err)
	}
	text := string(bytes.ReplaceAll(header, []byte("\r\n"), []byte("\n")))
	cut := strings.Index(text, "__global__ void k_qwen35_attention_postprocess(")
	if cut < 0 {
		t.Fatal("production kernel boundary missing")
	}
	for _, want := range []string{
		"__shared__ float red[256]", "__shared__ volatile float normalized[256]",
		"for (int s = blockDim.x / 2; s > 0; s >>= 1)",
		"double freq = pow(theta, -(2.0 * j) / rotary)",
		"volatile float rotated =", "QOut[dst] = rotated",
		"KRawCache[dst] = normalized[dim]", "__float2half_rn(x)",
		"qwen35_postprocess_bytes(tokens, start, kv_dtype, bytes)",
		"qwen35_postprocess_memory(ptrs, bytes, kv_dtype)",
		"attrs.type != cudaMemoryTypeDevice || attrs.device != 0",
		"capture != cudaStreamCaptureStatusNone", "tokens * 32, 256, 0, g_stream",
		`extern "C" int fak_qwen35_attention_postprocess_f32(`,
		"KRawCache, KCache, VCache, tokens, start, rotary, eps, theta, 0)",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("production source lost contract %q", want)
		}
	}
	if strings.Index(text, "const int memory_status") > strings.Index(text, "FCUDA_TS(") {
		t.Fatal("memory admission moved after launch")
	}
	wrapper, err := os.ReadFile("cuda_qwen35_attention_postprocess.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"case KVPrecisionF16:", "storage, storageCode = F16, 1",
		"case KVPrecisionBF16:", "storage, storageCode = BF16, 2",
		"kv.Kraw[layer].dtype != F32 || kv.K[layer].dtype != storage || kv.V[layer].dtype != storage",
		"dst.len != want || dst.cap < end", "C.int(storageCode)",
		"c.qwen35SequenceSyncLocked()", "errors.Join(selectedErr, drainErr)",
	} {
		if !strings.Contains(string(wrapper), want) {
			t.Errorf("Go wrapper lost contract %q", want)
		}
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; no store/admission execution witness")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "contract.cc")
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	program := qwen35PostprocessHostPreamble + text[:cut] + qwen35PostprocessNativeFailurePreamble +
		qwen35PostprocessCFunction(t, read("cuda_kernels.cu"), "static bool v41_candidates_stream_uncertain(void)") +
		qwen35PostprocessCFunction(t, read("cuda_kernels.cu"), `extern "C" int fak_v41_checked_stream_terminal(void)`) +
		qwen35PostprocessCFunction(t, read("cuda/q56k_mmq.cuh"), `extern "C" void fcuda_q56k_mmq_retirement_failed(int first)`) +
		qwen35PostprocessCFunction(t, read("cuda/runtime_memory.cuh"), `extern "C" void fcuda_free(void *d)`) +
		qwen35PostprocessCFunction(t, text, `extern "C" int fak_qwen35_attention_postprocess_capability(int tokens)`) +
		qwen35PostprocessHostMain
	if err := os.WriteFile(source, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	binary := filepath.Join(dir, "contract")
	cmd := exec.CommandContext(ctx, compiler, "-std=c++17", "-O2", "-fno-fast-math", "-fno-strict-aliasing", "-ffp-contract=off", source, "-o", binary)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("actual production helper compilation: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("actual production helper execution: %v\n%s", err, output)
	} else {
		t.Log(string(output))
	}
}

func qwen35PostprocessCFunction(t *testing.T, source, signature string) string {
	t.Helper()
	start := strings.Index(source, signature)
	if start < 0 {
		t.Fatalf("production function missing: %s", signature)
	}
	open := strings.Index(source[start:], "{")
	if open < 0 {
		t.Fatal("production function body missing")
	}
	depth := 0
	for i := start + open; i < len(source); i++ {
		switch source[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return source[start:i+1] + "\n"
			}
		}
	}
	t.Fatal("production function body incomplete")
	return ""
}

// Execute the actual Go selection/drain/release code with explicit fake native
// callbacks. The companion C++ witness separately executes the real terminal
// marker, terminal query and allocator release. Neither fixture is a GPU test.
// fak-test:runtime medium est=3s lane=default
func TestCUDAQwen35AttentionPostprocessFailedDrainOwnership(t *testing.T) {
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Fatal("Go tool required for extracted-production lifecycle witness")
	}
	extract := func(path string, names ...string) string {
		t.Helper()
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, path, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		for _, name := range names {
			found := false
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if ok && fn.Name.Name == name {
					out.Write(source[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
					out.WriteByte('\n')
					found = true
				}
			}
			if !found {
				t.Fatalf("production Go function missing: %s", name)
			}
		}
		return out.String()
	}
	code := extract("cuda_qwen35_attention_postprocess.go", "qwen35SequenceAttentionPostprocessLocked") +
		extract("cuda_kernels.go", "qwen35SequenceSyncLocked") +
		extract("cuda.go", "cudaQ56KMMVQReleaseBlocked", "releaseTransientBuffers", "qwen35SequenceIQ2FenceLocked") +
		extract("cuda_qwen35_sequence.go", "qwen35SequenceReleaseSinceLocked") +
		extract("cuda_q56k_mmq.go", "cudaQ56KMMQRetirementFailedLocked", "qwen35Q56KMMQFenceLocked")
	// Only cgo spellings become instrumented callbacks/types. Production Go
	// conditionals, publication, error joining and cleanup bodies are unchanged.
	code = strings.NewReplacer("C.fak_qwen35_sequence_sync", "nativeSequenceSync", "C.fcuda_qwen35_iq2_sequence_fence", "nativeQualifiedFence", "C.fak_qwen35_attention_postprocess_capability", "nativeCapability", "C.fak_qwen35_attention_postprocess_typed", "nativeSubmit", "C.fcuda_q56k_mmq_retirement_failed", "nativeRetirementFailed", "C.fak_v41_checked_stream_terminal", "nativeTerminal", "C.fcuda_free", "nativeFree", "C.float", "float32", "C.double", "float64", "C.int", "int").Replace(code)
	dir := t.TempDir()
	path := filepath.Join(dir, "ownership.go")
	if err := os.WriteFile(path, []byte(qwen35PostprocessGoFixture+code+qwen35PostprocessGoMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// Use the actual error definition and complete latch API, not lookalike
	// test structs that could hide a misspelled field or ignored error class.
	args := []string{"run", "-trimpath", "-buildvcs=false", "-p=1", path}
	for _, name := range []string{"cuda_errors.go", "devicefault.go"} {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(dir, name)
		if err := os.WriteFile(dst, bytes.Replace(data, []byte("package compute"), []byte("package main"), 1), 0600); err != nil {
			t.Fatal(err)
		}
		args = append(args, dst)
	}
	cmd := exec.CommandContext(ctx, goTool, args...)
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOMAXPROCS=1")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("extracted production ownership witness: %v\n%s", err, output)
	} else {
		t.Log(string(output))
	}
}

const qwen35PostprocessHostPreamble = `
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#define __device__
#define __forceinline__ inline
// Host shim only. The device witness must separately certify __float2half_rn.
using __half = _Float16;
static __half __float2half_rn(float x) { return (__half)x; }
static unsigned __float_as_uint(float x) { unsigned u; memcpy(&u, &x, 4); return u; }
static std::map<void*, size_t> g_live, g_panel_live;
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr, "%s\n", why); exit(1); } }
`

const qwen35PostprocessNativeFailurePreamble = `
using cudaError_t = int;
static int g_cuda_cleanup_status=0, runtime_calls=0, frees=0, pools=0, split_status=0;
static bool g_v41_router_uncertain=false, g_v41_mmid_uncertain=false, g_capture_open=false;
static void sequence_split_note_failed_fence(cudaError_t e) { split_status=e; }
static bool v41c_pending_contains(void*) { return false; }
static void q4_coeff_cache_evict(const void*) {}
static std::map<void*,size_t> g_managed_live;
static std::map<size_t,std::vector<void*>> g_panel_pool;
static size_t g_panel_live_bytes=0, g_panel_cached_bytes=0;
#define FAK_ALLOC_IDLE_POOL 0
#define FAK_ALLOC_SITE(x) 0
static void fak_alloc_untrack(void*) {}
static void fak_alloc_retag(void*,int,int) {}
static void cudaFree(void*) { ++frees; }
static void fcuda_pool_put(void*,size_t) { ++pools; }
static int g_stream=0;
static const int cudaSuccess=0, cudaStreamCaptureStatusNone=0;
using cudaStreamCaptureStatus=int;
struct cudaDeviceProp { int maxThreadsPerBlock=256; int maxThreadsDim[3]={256,1,1}; size_t sharedMemPerBlock=2048; int maxGridSize[3]={INT_MAX,1,1}; };
static int cudaGetDevice(int *d) { ++runtime_calls; *d=0; return 0; }
static int cudaStreamIsCapturing(int,int *s) { ++runtime_calls; *s=0; return 0; }
static int cudaGetDeviceProperties(cudaDeviceProp*,int) { ++runtime_calls; return 0; }
`

const qwen35PostprocessHostMain = `
int main() {
  void *held=(void*)(uintptr_t)0x100000;
  g_live[held]=4096;
  require(fak_qwen35_attention_postprocess_capability(1)==1, "healthy capability refused");
  fcuda_q56k_mmq_retirement_failed(60700);
  require(g_cuda_cleanup_status==60700 && split_status==60700 && fak_v41_checked_stream_terminal(), "failed drain did not close shared native admission");
  int calls=runtime_calls;
  require(fak_qwen35_attention_postprocess_capability(1)<0 && runtime_calls==calls, "terminal capability queried runtime");
  fcuda_free(held);
  require(g_live.count(held)==1 && pools==0 && frees==0, "failed drain recycled native owner");
  fcuda_q56k_mmq_retirement_failed(60999);
  require(g_cuda_cleanup_status==60700, "later failure replaced first terminal code");
  // Fixture isolation only, not a production recovery path.
  g_cuda_cleanup_status=0; fcuda_free(held);
  require(!g_live.count(held) && pools==1, "healthy cleanup control did not execute");
  g_live.clear();
  for (int dtype = 0; dtype <= 2; ++dtype) {
    size_t bytes[9];
    require(qwen35_postprocess_bytes(7, 5, dtype, bytes), "valid layout refused");
    const size_t expected[9] = {7*6144*4,7*1024*4,7*1024*4,1024,1024,7*6144*4,
        12*1024*4,(size_t)12*1024*(dtype ? 2 : 4),(size_t)12*1024*(dtype ? 2 : 4)};
    const void *ptrs[9];
    for (int i = 0; i < 9; ++i) {
      require(bytes[i] == expected[i], "wrong byte extent");
      ptrs[i] = (void*)(uintptr_t)(0x100000 + i*0x100000);
      g_live[(void*)ptrs[i]] = bytes[i];
    }
    require(qwen35_postprocess_memory(ptrs, bytes, dtype) == 0, "exact owned bytes refused");
    for (int i = 0; i < 9; ++i) {
      --g_live[(void*)ptrs[i]];
      require(qwen35_postprocess_memory(ptrs, bytes, dtype) == -3, "short owned bytes accepted");
      ++g_live[(void*)ptrs[i]];
    }
    for (int out = 5; out < 9; ++out) for (int other = 0; other < out; ++other) {
      const void *saved = ptrs[out]; ptrs[out] = ptrs[other];
      const size_t old = g_live[(void*)ptrs[other]];
      g_live[(void*)ptrs[other]] = bytes[out] + bytes[other];
      require(qwen35_postprocess_memory(ptrs, bytes, dtype) == -4, "output alias accepted");
      g_live[(void*)ptrs[other]] = old; ptrs[out] = saved;
    }
    // Typed K/V exact allocations must not be charged F32-sized extents.
    g_panel_live[(void*)ptrs[7]] = bytes[7]; g_live.erase((void*)ptrs[7]);
    require(qwen35_postprocess_memory(ptrs, bytes, dtype) == 0, "owned panel refused");
    const void *saved = ptrs[7];
    ptrs[7] = (void*)((uintptr_t)ptrs[6] + bytes[6] - 2);
    g_live[(void*)ptrs[7]] = bytes[7];
    require(qwen35_postprocess_memory(ptrs, bytes, dtype) != 0, "prefix overlap accepted");
    ptrs[7] = (void*)((uintptr_t)saved + 1); g_live[(void*)ptrs[7]] = bytes[7];
    require(qwen35_postprocess_memory(ptrs, bytes, dtype) == -3, "misalignment accepted");
    ptrs[7] = (void*)(UINTPTR_MAX - 1); g_live[(void*)ptrs[7]] = bytes[7];
    require(qwen35_postprocess_memory(ptrs, bytes, dtype) == -3, "pointer overflow accepted");
    g_live.clear(); g_panel_live.clear();
  }
  size_t bytes[9];
  require(!qwen35_postprocess_bytes(0,0,0,bytes), "zero tokens");
  require(!qwen35_postprocess_bytes(INT_MAX,0,0,bytes), "token overflow");
  require(!qwen35_postprocess_bytes(1,-1,0,bytes), "negative frontier");
  require(!qwen35_postprocess_bytes(1,INT_MAX/1024,0,bytes), "frontier overflow");
  require(!qwen35_postprocess_bytes(1,0,-1,bytes), "negative dtype");
  require(!qwen35_postprocess_bytes(1,0,3,bytes), "unsupported dtype");
  require(qwen35_postprocess_bytes(1,INT_MAX/1024-1,2,bytes), "bounded maximum frontier");
  const unsigned inputs[] = {0,0x80000000,0x3f800000,0x3f801000,0x3f803000,
      0x3f808000,0x3f818000,0x7f800000,0xff800000,0x7f800001,0xffc12345};
  const unsigned short bf16[] = {0,0x8000,0x3f80,0x3f80,0x3f80,0x3f80,0x3f82,
      0x7f80,0xff80,0x7fc0,0xffc1};
  const unsigned short half[] = {0,0x8000,0x3c00,0x3c00,0x3c02,0x3c04,0x3c0c};
  for (size_t i = 0; i < sizeof(inputs)/sizeof(inputs[0]); ++i) {
    float x; memcpy(&x, &inputs[i], 4);
    unsigned short narrowed[3] = {0xa55a,0xa55a,0xa55a};
    qwen35_postprocess_store(narrowed, 1, x, 2);
    require(narrowed[0] == 0xa55a && narrowed[2] == 0xa55a && narrowed[1] == bf16[i], "BF16 bytes/RNE/NaN/sentinel");
    unsigned raw = 0;
    qwen35_postprocess_store(&raw, 0, x, 0);
    require(raw == inputs[i], "F32 bits changed");
    if (i < sizeof(half)/sizeof(half[0])) {
      qwen35_postprocess_store(narrowed, 1, x, 1);
      require(narrowed[0] == 0xa55a && narrowed[2] == 0xa55a && narrowed[1] == half[i], "host half shim bytes/sentinel");
    }
  }
  puts("actual production byte admission and F32/BF16 stores passed; F16 host shim only, no CUDA execution");
}
`

const qwen35PostprocessGoFixture = `package main
import("errors";"fmt";"math";"sync/atomic";"unsafe")
type Dtype int
const(F32 Dtype=iota; F16; BF16)
type KVPrecision int
const(KVPrecisionF32 KVPrecision=iota; KVPrecisionQ8; KVPrecisionF16; KVPrecisionBF16)
type cudaBuf struct{ptr unsafe.Pointer; graphOncePinned bool}
type Tensor struct{buf *cudaBuf}
func(t Tensor)Buf()*cudaBuf{return t.buf}
type dslice struct{ptr unsafe.Pointer;len,cap int;dtype Dtype}
type config struct{Precision KVPrecision;NumKVHeads,HeadDim int}
type cudaKV struct{be *cudaBackend;cfg config;Kraw,K,V []dslice;pos []int}
type Qwen35SequenceError struct{Stage string;Layer int;Reason string;Cause error}
func(e *Qwen35SequenceError)Error()string{return fmt.Sprintf("%s:%s:%v",e.Stage,e.Reason,e.Cause)}
func(e *Qwen35SequenceError)Unwrap()error{return e.Cause}
type MemoryClass string
const MemoryUnknown MemoryClass="unknown"
type Backend interface{}
type Qwen35GDNKernelError struct{Stage string;Code int}
func(e *Qwen35GDNKernelError)Error()string{return e.Stage}
type Qwen35GDNInvalidStateError struct{Operand string}
func(e *Qwen35GDNInvalidStateError)Error()string{return e.Operand}
type DeviceAllocError struct{Site string}
func(e *DeviceAllocError)Error()string{return e.Site}
type cudaBackend struct{capturing bool;faultLatch *DeviceFaultLatch;fenceGen uint64;transient []*cudaBuf}
var cudaQwen35AttentionPostprocessEnabled=true
var qwen35SequenceIQ2Qualification,cudaQ56KMMQQualification bool
var drainStatus,launchStatus,terminal,drains,submits,allocs,frees,iq,q56 int
func nativeSequenceSync()int{drains++;return drainStatus}
func nativeQualifiedFence()int{if qwen35SequenceIQ2Qualification{iq++}else{q56++};return nativeSequenceSync()}
func nativeRetirementFailed(code int){if terminal==0{terminal=code}}
func nativeTerminal()int{return terminal}
func nativeCapability(int)int{if terminal!=0{return -89101};return 1}
func nativeSubmit(...any)int{submits++;return launchStatus}
func nativeFree(unsafe.Pointer){if terminal!=0{panic("retired after failed drain")};frees++}
func(c *cudaBackend)cf(Tensor)*float32{return nil}
func(c *cudaBackend)validateQwen35SequenceTensor(string,Tensor,bool,...int)error{return nil}
func validateQwen35CausalAttentionPanelGeometry(int,int,int,int,int)error{return nil}
func(c *cudaBackend)qwen35SequenceAllocLocked([]int,string)(Tensor,*cudaBuf,error){allocs++;b:=&cudaBuf{ptr:unsafe.Pointer(new(byte))};c.transient=append(c.transient,b);return Tensor{b},b,nil}
func cudaQ4KAsyncReleaseBlockedLocked()bool{return false}
func cudaQ4KAsyncRequireDrainLocked(*cudaBackend){}
func cudaV41CompressorDeferFreeLocked(*cudaBackend,*cudaBuf)bool{return false}
func cudaV41CompressorPendingIndex(*cudaBuf)int{return -1}
func(c *cudaBackend)graphOnceFreeDependencyLocked(*cudaBuf){}
func require(ok bool,why string){if !ok{panic(why)}}
`

const qwen35PostprocessGoMain = `
func main(){
 for mode:=0;mode<3;mode++{for _,nilLatch:=range []bool{false,true}{for _,failed:=range []bool{false,true}{for _,enqueueOK:=range []bool{false,true}{
  drainStatus,launchStatus,terminal,drains,submits,allocs,frees,iq,q56=0,86007,0,0,0,0,0,0,0
  if failed{drainStatus=60700}
  if enqueueOK{launchStatus=0}
  qwen35SequenceIQ2Qualification,cudaQ56KMMQQualification=mode==1,mode==2
  c:=&cudaBackend{faultLatch:NewDeviceFaultLatch("fixture",1)};if nilLatch{c.faultLatch=nil}
  p:=unsafe.Pointer(new(byte));plane:=dslice{ptr:p,cap:1024,dtype:F32}
  kv:=&cudaKV{be:c,cfg:config{NumKVHeads:4,HeadDim:256},Kraw:[]dslice{plane},K:[]dslice{plane},V:[]dslice{plane}}
  x:=Tensor{&cudaBuf{ptr:p}}
  out,b,used,err:=c.qwen35SequenceAttentionPostprocessLocked(x,x,x,x,x,kv,0,1,0,2,1e-6,1e7)
  if enqueueOK{
   require(used&&err==nil&&out.Buf()!=nil&&b!=nil&&drains==0,"successful enqueue added a fence/failed")
   err=c.qwen35SequenceSyncLocked()
   require((err!=nil)==failed,"final fence outcome lost")
  }else{require(used&&err!=nil&&out.Buf()==nil&&b==nil,"selected failure was replayed/published")}
  require(drains==1&&submits==1&&allocs==1,"unexpected retry/fence/allocation")
  require((iq==1)==(mode==1)&&(q56==1)==(mode==2),"qualification dispatch changed")
  end,positions:=0,0;if enqueueOK{end,positions=1024,1}
  require(len(kv.pos)==positions&&kv.Kraw[0].len==end&&kv.K[0].len==end&&kv.V[0].len==end,"enqueue metadata boundary changed")
  if failed{
   // Exercise actual outer cleanup before asserting retention. On the old
   // common-fence body this visibly frees/detaches the still-uncertain owner.
   c.qwen35SequenceReleaseSinceLocked(0)
   require(terminal==60700&&c.fenceGen==0&&frees==0&&len(c.transient)==1,fmt.Sprintf("failed drain ownership: terminal=%d frees=%d retained=%d fence=%d",terminal,frees,len(c.transient),c.fenceGen))
   if !nilLatch{s:=c.faultLatch.Snapshot();require(s.LastCode==60700&&s.Faults==1&&s.LastClass==DeviceFaultExecution&&s.Refusing(),"actual Go execution latch not observed exactly once")}
   var native *CUDAOpError;require(errors.As(err,&native)&&native.Code==60700,"exact drain error lost")
   if !enqueueOK{require(len(err.(interface{Unwrap()[]error}).Unwrap())==2,"selected error not joined with drain failure")}
   require(len(c.transient)==1&&c.transient[0].ptr!=nil&&frees==0,"outer release detached owner")
   c.faultLatch=NewDeviceFaultLatch("reconstructed",1)
   // Return metadata to a valid frontier solely to isolate terminal admission.
   kv.Kraw[0].len,kv.K[0].len,kv.V[0].len=0,0,0;kv.pos=nil
   _,_,used,err=c.qwen35SequenceAttentionPostprocessLocked(x,x,x,x,x,kv,0,1,0,2,1e-6,1e7)
   require(used&&err!=nil&&drains==1&&submits==1&&allocs==1,"reconstructed latch reopened native stream")
  }else{if enqueueOK{c.qwen35SequenceReleaseSinceLocked(0)};require(terminal==0&&c.fenceGen==1&&frees==1&&len(c.transient)==0,"successful completion did not retire exactly once")}
 }}}}
 fmt.Println("actual error/latch API and extracted production immediate/final-fence/qualification/owner-retention paths passed; fake runtime callbacks, no GPU")
}
`
