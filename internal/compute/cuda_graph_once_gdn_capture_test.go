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

var graphOnceGDNHost = flag.Bool("fak-graph-once-gdn-host", false, "run actual GDN capture preflight/record control; no CUDA arithmetic")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOnceGDNRecordSourceContract(t *testing.T) {
	source := readCUDASource(t, "cuda/graph_gdn_capture.cuh")
	body := cudaFunctionBody(t, source, "fak_graph_once_gdn_record")
	for _, forbidden := range []string{"cudaMalloc", "cudaFree", "cudaStreamSynchronize", "getenv(", "qwen35_gdn_fence", "g_qwen35_gdn_operations", "fenceGen", "g_qwen35_gdn_decode_register_state_launches", "fcuda_qwen35_gdn_decode_f32", "s->next"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("record-only GDN effect %q", forbidden)
		}
	}
	at := 0
	for _, fragment := range []string{"graph_once_capture_admit(s)", "graph_gdn_preflight(s,r,&g)", "cudaStreamCaptureStatusActive", "cudaPeekAtLastError()", "fak_graph_once_quant_matmul(s,r->quant_operation[i])", "k_qwen35_gdn_fused_in_proj<<<", "k_qwen35_gdn_conv_state<<<", "k_qwen35_gdn_qk_norm<<<", "k_qwen35_gdn_recurrent_gated_norm_regstate<<<", "k_qwen35_gdn_recurrent_gated_norm<<<", "k_qwen35_gdn_out_proj<<<"} {
		i := strings.Index(body[at:], fragment)
		if i < 0 {
			t.Fatalf("missing/reordered GDN record stage %q", fragment)
		}
		at += i + len(fragment)
	}
	if strings.Count(body, "graph_gdn_record_status(s)") != 5 {
		t.Fatal("every dependent GDN stage requires its own accepted-record status")
	}
	preflight := cudaFunctionBody(t, source, "graph_gdn_preflight")
	for _, required := range []string{"r->hidden>INT_MAX-255", "conv>INT_MAX-255", "r->kind[i]>(i<2?2:1)", "r->quant_operation[i]!=-1", "op.kind!=FAK_GRAPH_QUANT_Q4_K", "op.weights!=r->weights[i]", "op.output!=(i==0?r->mixed:r->z)", "graph_quant_overlap(write[i],read[j])", "graph_quant_overlap(write[i],write[j])"} {
		if !strings.Contains(preflight, required) {
			t.Fatalf("missing full preflight %q", required)
		}
	}
	ordinary := cudaFunctionBody(t, readCUDASource(t, "cuda/qwen35_gdn.cuh"), "qwen35_gdn_projection_preflight")
	if !strings.Contains(ordinary, "!g_stream || g_capture_open") || !strings.Contains(ordinary, "cudaStreamCaptureStatusNone") {
		t.Fatal("ordinary packed capture refusal must stay intact")
	}
	for _, name := range []string{"k_qwen35_gdn_fused_in_proj", "k_qwen35_gdn_conv_state", "k_qwen35_gdn_qk_norm", "k_qwen35_gdn_recurrent_gated_norm", "k_qwen35_gdn_recurrent_gated_norm_regstate", "k_qwen35_gdn_out_proj"} {
		if strings.Count(source, "FCUDA_TS(\""+name+"\",") != 1 {
			t.Fatalf("expected one exact native timestamp site %s", name)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOnceGDNManifestSourceContract(t *testing.T) {
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	source := readCUDASource(t, "cuda/graph_gdn_capture.cuh")
	for name, ordered := range map[string][]string{
		"fak_graph_once_begin":      {"GDN_BEGIN", "graph_once_set_position(s, pos)", "cudaStreamBeginCapture"},
		"fak_graph_once_end":        {"GDN_END", "cudaStreamEndCapture", "cudaGraphInstantiate", "graph_once_launch(s)"},
		"fak_graph_once_replay":     {"GDN_REPLAY", "graph_once_set_position(s, pos)", "graph_once_launch(s)"},
		"fak_graph_once_copy_input": {"GDN_COPY", "FCUDA_TS(\"d2d_copy\""},
		"fak_graph_once_release":    {"cudaGraphExecDestroy", "cudaGraphDestroy", "graph_once_quant_retire(s)", "s->gdn_retire(s)", "cudaFree(s->position)"},
	} {
		body := cudaFunctionBody(t, graph, name)
		at := 0
		for _, text := range ordered {
			i := strings.Index(body[at:], text)
			if i < 0 {
				t.Fatalf("%s missing ordered %q", name, text)
			}
			at += i + len(text)
		}
	}
	for _, text := range []string{"count>256", "found->second!=d.extent[i]", "s->quant_ops!=s->gdn_quant_plan",
		"s->gdn_cursor!=s->gdn_count", "operation<=previous_quant", "graph_gdn_manifest_valid(s,plan,count,true)",
		"r!=&plan[s->gdn_cursor].record", "operation!=s->gdn_cursor", "++s->gdn_cursor"} {
		if !strings.Contains(source, text) {
			t.Fatalf("missing manifest contract %q", text)
		}
	}
	for _, name := range []string{"graph_gdn_manifest_check", "fak_graph_once_gdn_record_at"} {
		body := cudaFunctionBody(t, source, name)
		for _, bad := range []string{"cudaMalloc", "cudaFree", "cudaMemcpy", "cudaStreamSynchronize", "calloc(", "free("} {
			if strings.Contains(body, bad) {
				t.Fatalf("%s has forbidden replay/record effect %q", name, bad)
			}
		}
	}
	retire := cudaFunctionBody(t, source, "graph_gdn_manifest_retire")
	if strings.Contains(retire, "cudaFree") || !strings.Contains(retire, "free(s->gdn_plan)") {
		t.Fatal("manifest retirement touched borrowed device memory")
	}
}

// Reuse only the prior record-only CUDA stubs, then extract the actual native
// owner, fixed-quant plan and GDN producer. No independent model substitutes for
// the implementation under test, and no fake kernel evaluates arithmetic.
// fak-test:runtime medium est=2s lane=optin
func TestCUDAGraphOnceGDNHostRecord(t *testing.T) {
	if !*graphOnceGDNHost {
		t.Skip("explicit GDN host record witness not selected; GPU evidence absent")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested GDN host witness needs c++")
	}
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	start, end := strings.Index(graph, "struct fak_graph_once {"), strings.Index(graph, "static fak_graph_once *g_graph_once_capture")
	if start < 0 || end <= start {
		t.Fatal("owner missing")
	}
	gdn := readCUDASource(t, "cuda/qwen35_gdn.cuh")
	lo, hi := strings.Index(gdn, "constexpr int kQwen35GDNDecodeRegisterMaxKeyHeadDim"), strings.Index(gdn, "template <bool kRegisterState>")
	if lo < 0 || hi <= lo {
		t.Fatal("register-state envelope missing")
	}
	// The warp row-dot launch helpers are host-only; FCUDA_TS discards their launch here.
	rowLo, rowHi := strings.Index(gdn, "// GDN row-dot host policy (begin)"), strings.Index(gdn, "// GDN row-dot host policy (end)")
	if rowLo < 0 || rowHi <= rowLo {
		t.Fatal("row-dot host policy missing")
	}
	program := graphOnceQuantHarness + "\n#include <cmath>\n#include <cstdlib>\nusing std::isfinite;\n" + graph[start:end] + "\nstatic fak_graph_once *g_graph_once_capture=nullptr;\n" +
		"static int graph_once_capture_admit(fak_graph_once *s) " + cudaFunctionBody(t, graph, "graph_once_capture_admit") + "\n" +
		"static int qwen35_gdn_threads(int n) " + cudaFunctionBody(t, gdn, "qwen35_gdn_threads") + "\n" + gdn[lo:hi] + gdn[rowLo:rowHi] +
		strings.TrimPrefix(readCUDASource(t, "cuda/alloc_status.h"), "#pragma once\n") +
		readCUDASource(t, "cuda/graph_quant_workspace.cuh") + readCUDASource(t, "cuda/graph_gdn_capture.cuh")
	definition := func(prefix, name string) string {
		begin := strings.Index(graph, prefix+name+"(")
		if begin < 0 {
			t.Fatalf("missing lifecycle %s", name)
		}
		open := strings.Index(graph[begin:], "{")
		return graph[begin:begin+open] + cudaFunctionBody(t, graph, name) + "\n"
	}
	for _, name := range []string{"graph_once_drain", "graph_once_fail", "graph_once_set_position", "graph_once_launch"} {
		program += definition("static int ", name)
	}
	for _, name := range []string{"fak_graph_once_begin", "fak_graph_once_end", "fak_graph_once_replay"} {
		program += definition("extern \"C\" int ", name)
	}
	program += graphOnceGDNMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "gdn-record.cpp"), filepath.Join(dir, "gdn-record")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-variable", "-Wno-unused-parameter", "-Wno-unused-function", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual GDN record source: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("actual GDN record control: %v\n%s", err, out)
	}
}

const graphOnceGDNMain = `
static fak_graph_once owner;
static fak_graph_gdn_record request;
static int position[2];
static uintptr_t address=0x10000;
// Pseudo device addresses are never dereferenced by these record-only stubs.
static void *region(size_t bytes){void *p=(void *)address;g_live[p]=bytes;address+=(bytes+255)/256*256+256;return p;}
static float *floats(size_t n){return (float *)region(n*sizeof(float));}
static void teardown(){for(void *p:allocated)std::free(p);allocated.clear();tagged.clear();std::free(owner.quant_ops);if(owner.gdn_retire)owner.gdn_retire(&owner);owner={};}
static void setup(int packed){
 teardown();g_live.clear();events.clear();address=0x10000;allocs=frees=records=fail_alloc=fail_free=fail_record=last_error=g_cuda_cleanup_status=launches=instantiates=0;
 capture=0;g_capture_open=false;g_graph_once_capture=nullptr;owner.position=position;owner.capacity=4;
 request={};request.hidden=256;request.key_heads=1;request.value_heads=1;request.key_dim=request.value_dim=128;request.conv_kernel=4;request.producer=FAK_GRAPH_GDN_BASELINE_V1;request.register_state=1;request.rms_eps=1e-6f;
 int rows[5]={384,128,1,1,256},cols[5]={256,256,256,256,128};
 for(int i=0;i<5;++i){request.kind[i]=(i<packed?2:1);request.weights[i]=region(request.kind[i]==2?(size_t)rows[i]*cols[i]/256*144:(size_t)rows[i]*cols[i]);if(request.kind[i]==1)request.scales[i]=floats((size_t)rows[i]*cols[i]/32);}
 request.input=floats(256);request.conv_weight=floats(384*4);request.a_log=floats(1);request.dt_bias=floats(1);request.norm=floats(128);
 request.conv_state=floats(384*3);request.recurrent_state=floats(128*128);request.output=floats(256);request.mixed=floats(384);request.z=floats(128);request.b=floats(1);request.a=floats(1);request.conv_out=floats(384);request.q_norm=floats(128);request.k_norm=floats(128);request.core=floats(128);
 request.quant_operation[0]=request.quant_operation[1]=-1;
 if(packed){fak_graph_quant_op ops[2];for(int i=0;i<packed;++i){request.quant_operation[i]=i;ops[i]={FAK_GRAPH_QUANT_Q4_K,rows[i],256,request.weights[i],nullptr,request.input,i==0?request.mixed:request.z};}assert(fak_graph_once_quant_prepare(&owner,ops,packed,288)==0);}
 capture=1;g_capture_open=true;g_graph_once_capture=&owner;
}
static int call(){return fak_graph_once_gdn_record(&owner,&request);}
static void refusal(){int before=allocs;assert(call()==1&&owner.terminal&&records==0&&owner.quant_cursor==0&&allocs==before);}

static void idle(){capture=0;g_capture_open=false;g_graph_once_capture=nullptr;}
static fak_graph_gdn_record second_record(){
 auto r=request;r.input=request.output;r.output=floats(256);
 r.conv_state=floats(384*3);r.recurrent_state=floats(128*128);return r;
}
static void manifest_controls(){
 // Immutable copied descriptors; two private state pairs may reuse transient
 // workspace and chain output-to-input in their fixed capture order.
 setup(0);idle();fak_graph_gdn_record pair[2]={request,second_record()};
 int before=allocs;size_t bytes=2*sizeof(graph_gdn_dependency);
 assert(fak_graph_once_gdn_prepare(&owner,pair,257,bytes)==1&&!owner.gdn_plan&&allocs==before);
 assert(fak_graph_once_gdn_prepare(&owner,pair,2,bytes-1)==2&&!owner.gdn_plan&&allocs==before);
 assert(fak_graph_once_gdn_prepare(&owner,pair,2,bytes)==0&&allocs==before);
 auto *frozen=(const graph_gdn_dependency *)owner.gdn_plan;
 pair[0].hidden=32;assert(frozen[0].record.hidden==256);
 assert(fak_graph_once_gdn_prepare(&owner,pair,2,bytes)==1);
 assert(fak_graph_once_begin(&owner,0)==0);int recorded=records;
 assert(fak_graph_once_gdn_record_at(&owner,0)==0&&fak_graph_once_gdn_record_at(&owner,1)==0);
 assert(owner.gdn_cursor==2&&records==recorded+10&&allocs==before);
 assert(fak_graph_once_end(&owner)==0&&launches==1&&instantiates==1);
 assert(owner.gdn_check(&owner,fak_graph_once::GDN_COPY));
 assert(fak_graph_once_replay(&owner,1)==0&&launches==2&&owner.next==2);
 // Same pointer and sufficient size are not enough: changed capacity declines
 // before input copying/position update, rather than recapturing or fallback.
 auto ptr=(void *)frozen[1].record.weights[0];g_live[ptr]+=16;recorded=records;
 assert(!owner.gdn_check(&owner,fak_graph_once::GDN_COPY));
 assert(fak_graph_once_replay(&owner,2)==1&&records==recorded&&launches==2&&owner.terminal);
 // Full packed record consumes the SAME frozen Q4 operation pair.
 setup(2);idle();assert(fak_graph_once_gdn_prepare(&owner,&request,1,sizeof(graph_gdn_dependency))==0);
 assert(fak_graph_once_begin(&owner,0)==0&&fak_graph_once_gdn_record_at(&owner,0)==0);
 assert(owner.gdn_cursor==1&&owner.quant_cursor==2&&fak_graph_once_end(&owner)==0);
 // Missing, duplicate, reordered or caller-rebound record never launches.
 for(int mode=0;mode<4;++mode){
  setup(0);idle();pair[0]=request;pair[1]=second_record();
  assert(fak_graph_once_gdn_prepare(&owner,pair,2,bytes)==0&&fak_graph_once_begin(&owner,0)==0);
  if(mode==1){assert(fak_graph_once_gdn_record_at(&owner,0)==0);assert(fak_graph_once_gdn_record_at(&owner,0)==1);}
  if(mode==2)assert(fak_graph_once_gdn_record_at(&owner,1)==1);
  if(mode==3)assert(fak_graph_once_gdn_record(&owner,&pair[0])==1);
  assert(fak_graph_once_end(&owner)!=0&&!g_capture_open&&capture==0&&!launches&&!instantiates);
 }
 // Cross-block state privacy and immutable storage, plus non-GDN quant ops.
 for(int mode=0;mode<4;++mode){
  setup(0);idle();pair[0]=request;pair[1]=second_record();
  if(mode==0)pair[1].recurrent_state=pair[0].recurrent_state;
  if(mode==1){pair[1].weights[0]=region(384*256);pair[1].output=(float *)pair[0].weights[0];}
  if(mode>=2){
   fak_graph_quant_op op{FAK_GRAPH_QUANT_Q4_K,1,256,region(144),nullptr,request.input,floats(1)};
   if(mode==2)op.output=request.conv_state;else op.weights=request.recurrent_state;
   assert(fak_graph_once_quant_prepare(&owner,&op,1,288)==0);
  }
  before=allocs;assert(fak_graph_once_gdn_prepare(&owner,pair,2,bytes)==1&&!owner.gdn_plan&&allocs==before&&!records);
 }
 // The manifest freezes quant plan identity: installing a plan afterward cannot
 // evade the combined alias checks already performed at GDN preparation.
 setup(0);idle();assert(fak_graph_once_gdn_prepare(&owner,&request,1,sizeof(graph_gdn_dependency))==0);
 fak_graph_quant_op later{FAK_GRAPH_QUANT_Q4_K,1,256,region(144),nullptr,request.input,request.conv_state};
 assert(fak_graph_once_quant_prepare(&owner,&later,1,288)==0);recorded=records;
 assert(fak_graph_once_begin(&owner,0)==1&&owner.terminal&&records==recorded&&capture==0);
 // Accepted native-record failure does not advance the manifest cursor.
 setup(0);idle();assert(fak_graph_once_gdn_prepare(&owner,&request,1,sizeof(graph_gdn_dependency))==0);
 assert(fak_graph_once_begin(&owner,0)==0);fail_record=records+3;
 assert(fak_graph_once_gdn_record_at(&owner,0)==3&&owner.gdn_cursor==0&&g_cuda_cleanup_status==700);
 assert(fak_graph_once_end(&owner)!=0&&!launches);
 // Fake-context teardown is not production recovery or proof of GPU lifetime.
 teardown();
}

int main(){
 manifest_controls();
 for(int packed=0;packed<=2;++packed){setup(packed);int before=allocs;assert(call()==0&&!owner.terminal&&records==5+2*packed&&owner.quant_cursor==packed&&allocs==before);}
 // Actual F32 weight matrices, not Q8 weights with F32 activations.
 setup(0);{
  int rows[5]={384,128,1,1,256},cols[5]={256,256,256,256,128};
  for(int i=0;i<5;++i){request.kind[i]=0;request.scales[i]=nullptr;request.weights[i]=floats((size_t)rows[i]*cols[i]);}
  int before=allocs;assert(call()==0&&!owner.terminal&&records==5&&owner.quant_cursor==0&&!owner.quant_ops&&allocs==before);
 }
 // Z-only packed Q4 leaves QKV on the original fused Q8 projection.
 setup(0);{
  capture=0;g_capture_open=false;g_graph_once_capture=nullptr;
  request.kind[1]=2;request.scales[1]=nullptr;request.weights[1]=region(128*144);request.quant_operation[1]=0;
  fak_graph_quant_op op{FAK_GRAPH_QUANT_Q4_K,128,256,request.weights[1],nullptr,request.input,request.z};
  assert(fak_graph_once_quant_prepare(&owner,&op,1,288)==0);
  capture=1;g_capture_open=true;g_graph_once_capture=&owner;
  int before=allocs;assert(call()==0&&!owner.terminal&&records==7&&owner.quant_cursor==1&&allocs==before);
 }
 setup(0);request.register_state=0;assert(call()==0&&records==5);assert(std::find(events.begin(),events.end(),"k_qwen35_gdn_recurrent_gated_norm")!=events.end());
 setup(2);request.quant_operation[1]=0;refusal(); // late projection mismatch, no first pair recorded
 setup(0);request.quant_operation[0]=0;refusal(); // Q8 cannot select Q8-activation scratch
 setup(2);request.kind[1]=3;refusal(); // Q6 remains excluded
 setup(0);request.kind[4]=2;refusal(); // no packed output projection
 setup(0);request.scales[1]=nullptr;refusal();
 setup(0);request.conv_state=(float *)request.weights[0];refusal();
 setup(0);request.q_norm=request.k_norm;refusal();
 setup(0);request.output=(float *)request.input;refusal();
 setup(0);g_live.erase(request.output);refusal();
 setup(0);request.hidden=INT_MAX-31;refusal(); // signed loop increment envelope
 setup(0);request.producer=2;refusal();
 setup(0);request.conv_kernel=1;refusal(); // zero-history contract not implemented
 setup(0);capture=0;refusal(); // actual stream disagrees with Go/native flag
 setup(0);last_error=700;assert(call()==3&&records==0&&g_cuda_cleanup_status==700);
 for(int failed=1;failed<=9;++failed){setup(2);fail_record=failed;assert(call()==3&&records==failed&&owner.terminal&&g_cuda_cleanup_status==700);}
 // Teardown is ordinary host test cleanup, not recovery after a CUDA fault.
 teardown();return 0;
}
`
