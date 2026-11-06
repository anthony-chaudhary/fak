package compute

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var graphOnceQuantHost = flag.Bool("fak-graph-once-quant-host", false, "run source-bound fixed-quant graph control witness; no CUDA arithmetic")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOnceQuantSourceContract(t *testing.T) {
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	quant := readCUDASource(t, "cuda/graph_quant_workspace.cuh")
	body := cudaFunctionBody(t, quant, "fak_graph_once_quant_matmul")
	for _, absent := range []string{"cudaMalloc", "cudaFree", "cudaStreamSynchronize", "cudaMemcpy", "g_q4k_mmvq_y", "g_q8_qX", "g_q8_xScale", "g_q4k_mmvq_launches", "s->next"} {
		if strings.Contains(body, absent) {
			t.Fatalf("record path has forbidden effect %q", absent)
		}
	}
	for _, want := range []string{"operation != s->quant_cursor", "cudaStreamCaptureStatusActive", "cudaPeekAtLastError", "mul_mat_vec_q<1>", "++s->quant_cursor"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing record contract %q", want)
		}
	}
	if strings.Count(body, "cudaGetLastError()") != 4 {
		t.Fatal("each quantizer and dot needs its own accepted-record status")
	}
	rows := map[string]int{}
	for _, match := range cudaTSWrap.FindAllStringSubmatch(quant, -1) {
		rows[match[1]]++
	}
	wantRows := map[string]string{"quantize_q8_1": KernelOpOther, "k_q4k_mmvq_gemv": KernelOpMatvec, "k_q8_quant_act": KernelOpOther, "k_q8_gemm": KernelOpMatmul}
	if len(rows) != len(wantRows) {
		t.Fatalf("unexpected quant timestamp rows: %v", rows)
	}
	for name, class := range wantRows {
		if rows[name] != 1 || KernelOpClass(name) != class || cudaKernelOtherAllowlist[name] != (class == KernelOpOther) {
			t.Fatalf("quant profile row %s", name)
		}
	}
	for name, order := range map[string][]string{
		"fak_graph_once_begin":   {"graph_once_quant_idle(s)", "graph_once_set_position(s, pos)", "cudaStreamBeginCapture"},
		"fak_graph_once_replay":  {"graph_once_quant_idle(s)", "graph_once_set_position(s, pos)", "graph_once_launch(s)", "s->next++"},
		"fak_graph_once_end":     {"s->quant_cursor != s->quant_count", "cudaStreamEndCapture", "s->terminal", "cudaGraphInstantiate", "graph_once_launch(s)"},
		"fak_graph_once_release": {"graph_once_drain(s)", "cudaGraphExecDestroy", "cudaGraphDestroy", "graph_once_quant_retire(s)", "cudaFree(s->position)", "free(s)"},
	} {
		b := cudaFunctionBody(t, graph, name)
		at := 0
		for _, fragment := range order {
			i := strings.Index(b[at:], fragment)
			if i < 0 {
				t.Fatalf("%s lost ordered stage %q", name, fragment)
			}
			at += i + len(fragment)
		}
	}
	retire := cudaFunctionBody(t, quant, "graph_once_quant_retire")
	if strings.Index(retire, "cudaFree") >= strings.Index(retire, "fak_alloc_untrack") || !strings.Contains(retire, "g_cuda_cleanup_status = (int)e") {
		t.Fatal("retirement can retry or forget an uncertain allocation")
	}
	// No new public selector or GDN admission is granted by this native leaf.
	if !strings.Contains(graph, "#include \"graph_quant_workspace.cuh\"") {
		t.Fatal("fixed workspace fragment missing")
	}
}

// Extract actual owner/preflight/transaction bodies. Fake CUDA records control
// effects and ownership only; it never evaluates a GPU dot or certifies replay.
// fak-test:runtime medium est=2s lane=optin
func TestCUDAGraphOnceQuantHostTransaction(t *testing.T) {
	if !*graphOnceQuantHost {
		t.Skip("explicit host control witness not selected; GPU evidence absent")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested host witness requires c++")
	}
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	start, end := strings.Index(graph, "struct fak_graph_once {"), strings.Index(graph, "static fak_graph_once *g_graph_once_capture")
	if start < 0 || end <= start {
		t.Fatal("native owner definition missing")
	}
	definition := func(prefix, name string) string {
		t.Helper()
		begin := strings.Index(graph, prefix+name+"(")
		if begin < 0 {
			t.Fatalf("missing native function %s", name)
		}
		open := strings.Index(graph[begin:], "{")
		return graph[begin:begin+open] + cudaFunctionBody(t, graph, name) + "\n"
	}
	program := graphOnceQuantHarness + graph[start:end] + "\nstatic fak_graph_once *g_graph_once_capture=nullptr;\n" +
		definition("static int ", "graph_once_capture_admit") +
		strings.TrimPrefix(readCUDASource(t, "cuda/alloc_status.h"), "#pragma once\n") +
		readCUDASource(t, "cuda/graph_quant_workspace.cuh")
	for _, name := range []string{"graph_once_drain", "graph_once_fail", "graph_once_set_position", "graph_once_launch"} {
		program += definition("static int ", name)
	}
	for _, name := range []string{"fak_graph_once_create", "fak_graph_once_begin", "fak_graph_once_end", "fak_graph_once_replay", "fak_graph_once_release"} {
		program += definition("extern \"C\" int ", name)
	}
	program += graphOnceQuantMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "quant-owner.cpp"), filepath.Join(dir, "quant-owner")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// Launch expressions are deliberately erased by the recording macro. Unused
	// CUDA-only locals therefore are not meaningful host-witness warnings.
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-variable", "-Wno-unused-parameter", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual owner control: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("actual owner control: %v\n%s", err, out)
	}
}

const graphOnceQuantHarness = `
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
using cudaError_t=int; using cudaGraph_t=void *; using cudaGraphExec_t=void *;
using cudaStreamCaptureStatus=int;
static constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2;
static constexpr int cudaStreamCaptureStatusNone=0,cudaStreamCaptureStatusActive=1,cudaStreamCaptureModeGlobal=0;
static constexpr int FAK_ALLOC_GRAPH_POOL=10,FAK_ALLOC_DEVICE_LOCAL=0;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
namespace fak_q4k_mmvq { struct block_q4_K { char bytes[144]; }; struct block_q8_1 { char bytes[36]; }; }
static bool g_capture_open=false;
static int g_stream=0,g_cuda_cleanup_status=0,last_error=0,capture=0;
static int allocs=0,frees=0,fail_alloc=0,fail_free=0,records=0,fail_record=0,launches=0,instantiates=0;
static std::unordered_map<void *,size_t> g_live;
static std::unordered_set<void *> allocated,tagged;
static std::vector<std::string> events;
static bool v41_candidates_stream_uncertain(){return g_cuda_cleanup_status!=0;}
static int cudaPeekAtLastError(){return last_error;}
static int cudaGetLastError(){int e=last_error;last_error=0;return e;}
static int cudaStreamIsCapturing(int,int *p){*p=capture;return 0;}
static int cudaStreamBeginCapture(int,int){assert(capture==0);capture=1;events.push_back("begin");return 0;}
static int cudaStreamEndCapture(int,void **p){assert(capture==1);capture=0;*p=(void *)1;events.push_back("end");return 0;}
static int cudaStreamSynchronize(int){events.push_back("drain");return 0;}
static int cudaGraphInstantiate(void **p,void *,int){++instantiates;*p=(void *)2;return 0;}
static int cudaGraphLaunch(void *,int){++launches;events.push_back("launch");return 0;}
static int cudaGraphExecDestroy(void *){events.push_back("destroy-exec");return 0;}
static int cudaGraphDestroy(void *){events.push_back("destroy-graph");return 0;}
static int cudaMalloc(void **p,size_t n){++allocs;if(allocs==fail_alloc){*p=nullptr;return last_error=2;}*p=std::malloc(n);assert(*p);allocated.insert(*p);return 0;}
static int cudaFree(void *p){++frees;if(frees==fail_free)return 700;assert(allocated.erase(p)==1);std::free(p);events.push_back("free");return 0;}
static void fak_alloc_track(void *p,int cls,int kind,int,size_t){assert(cls==10&&kind==0&&allocated.count(p));assert(tagged.insert(p).second);}
static void fak_alloc_untrack(void *p){assert(tagged.erase(p)==1);events.push_back("untrack");}
static void record(const char *name){++records;events.push_back(name);if(records==fail_record)last_error=700;}
#define FAK_ALLOC_SITE(name) 1
#define FCUDA_TS(name,grid,...) do { (void)(grid); record(name); } while(false)
`

const graphOnceQuantMain = `
static std::vector<void *> borrowed;
static void *buffer(size_t bytes){void *p=std::malloc(bytes);assert(p);g_live[p]=bytes;borrowed.push_back(p);return p;}
static void reset(){assert(allocated.empty()&&tagged.empty());for(void *p:borrowed)std::free(p);borrowed.clear();g_live.clear();events.clear();allocs=frees=fail_alloc=fail_free=records=fail_record=launches=instantiates=0;g_cuda_cleanup_status=last_error=capture=0;g_capture_open=false;g_graph_once_capture=nullptr;}
static void *owner(){void *o=nullptr;assert(fak_graph_once_create(4,&o)==0);return o;}
static void plan(fak_graph_quant_op (&ops)[2]){
 float *x=(float *)buffer(256*4),*y=(float *)buffer(8*4);
 ops[0]={FAK_GRAPH_QUANT_Q4_K,8,256,buffer(8*144),nullptr,x,y};
 ops[1]={FAK_GRAPH_QUANT_Q8_0,8,256,buffer(8*256),(float *)buffer(8*8*4),x,y};
}
// Fake-context teardown ONLY, after testing refusal/no retry. Not a proposed
// production recovery: no CUDA work or asynchronous pointer users exist here.
static void context_teardown(void *o){auto *s=(fak_graph_once *)o;for(void *p:allocated)std::free(p);allocated.clear();tagged.clear();std::free(s->quant_ops);std::free(s);g_graph_once_capture=nullptr;g_capture_open=false;capture=0;}
int main(){
 reset();void *o=owner();fak_graph_quant_op ops[2];plan(ops);auto *s=(fak_graph_once *)o;
 int before=allocs;
 assert(fak_graph_once_quant_prepare(o,ops,4097,576)==1&&allocs==before);
 auto bad=ops[0];bad.out=INT_MAX;bad.in=512;
 assert(fak_graph_once_quant_prepare(o,&bad,1,576)==1&&allocs==before);
 bad=ops[0];bad.output=(float *)ops[1].weights;
 fak_graph_quant_op alias[2]={bad,ops[1]};
 assert(fak_graph_once_quant_prepare(o,alias,2,576)==1&&allocs==before);
 capture=1;assert(fak_graph_once_quant_prepare(o,ops,2,576)==1&&allocs==before);capture=0;
 assert(fak_graph_once_quant_prepare(o,ops,2,575)==2&&allocs==before&&!s->quant_ops);
 assert(fak_graph_once_quant_prepare(o,ops,2,576)==0&&allocs==before+3&&tagged.size()==3);
 assert(s->quant_bytes[0]==288&&s->quant_bytes[1]==256&&s->quant_bytes[2]==32);
 ops[0].out=1;assert(((fak_graph_quant_op *)s->quant_ops)[0].out==8);
 assert(fak_graph_once_quant_prepare(o,ops,2,9999)==1&&allocs==before+3);
 assert(fak_graph_once_quant_matmul(o,0)==1&&records==0);
 assert(fak_graph_once_begin(o,0)==0);int prior_records=records;
 assert(fak_graph_once_quant_matmul(o,0)==0&&fak_graph_once_quant_matmul(o,1)==0);
 assert(s->quant_cursor==2&&records==prior_records+4&&allocs==before+3);
 assert(fak_graph_once_end(o)==0&&launches==1&&instantiates==1);
 assert(fak_graph_once_replay(o,1)==0&&launches==2&&s->next==2&&records==prior_records+5);
 g_live.erase((void *)ops[1].weights);prior_records=records;
 assert(fak_graph_once_replay(o,2)==1&&launches==2&&records==prior_records);
 assert(fak_graph_once_release(&o)==0&&!o&&tagged.empty()&&allocated.empty());
 auto destroyed=std::find(events.begin(),events.end(),"destroy-graph"),freed=std::find(events.begin(),events.end(),"free");assert(destroyed<freed);
 // Wrong order and omitted records must close capture without initial launch.
 for(int mode=0;mode<2;++mode){reset();o=owner();plan(ops);assert(fak_graph_once_quant_prepare(o,ops,2,576)==0);assert(fak_graph_once_begin(o,0)==0);
  if(mode==0)assert(fak_graph_once_quant_matmul(o,1)==1);else assert(fak_graph_once_quant_matmul(o,0)==0);
  assert(fak_graph_once_end(o)!=0&&!g_capture_open&&capture==0&&launches==0&&instantiates==0);assert(fak_graph_once_release(&o)==0);}
 // A capacity failure retains a releasable partial allocation and consumes only OOM.
 reset();o=owner();plan(ops);fail_alloc=allocs+2;
 assert(fak_graph_once_quant_prepare(o,ops,2,576)==2&&last_error==0&&!g_cuda_cleanup_status&&tagged.size()==1);
 assert(fak_graph_once_release(&o)==0&&allocated.empty()&&tagged.empty());
 // A failed accepted record does not advance the cursor or record its dependent dot.
 reset();o=owner();plan(ops);assert(fak_graph_once_quant_prepare(o,ops,2,576)==0);assert(fak_graph_once_begin(o,0)==0);s=(fak_graph_once *)o;fail_record=records+1;prior_records=records;
 assert(fak_graph_once_quant_matmul(o,0)==3&&records==prior_records+1&&s->quant_cursor==0&&g_cuda_cleanup_status==700);
 assert(fak_graph_once_end(o)!=0&&launches==0);assert(fak_graph_once_release(&o)==1&&o);context_teardown(o);
 // Failed retirement retains exact pointer/accounting and forbids another free.
 reset();o=owner();plan(ops);assert(fak_graph_once_quant_prepare(o,ops,2,576)==0);s=(fak_graph_once *)o;void *first=s->quant_scratch[0];fail_free=frees+1;
 assert(fak_graph_once_release(&o)==2&&o&&s->quant_scratch[0]==first&&tagged.count(first)&&s->quant_error==700);
 int attempts=frees;assert(fak_graph_once_release(&o)==1&&frees==attempts);context_teardown(o);
 reset();return 0;
}
`
