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

var kvFP8AttentionHost = flag.Bool("fak-kv-fp8-attention-owner-host", false, "run actual FP8 attention owner admission host controls, not CUDA attention")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVFP8AttentionOwnerSourceContract(t *testing.T) {
	src := readCUDASource(t, "cuda/kv_fp8_attention_owner.cuh")
	body := cudaFunctionBody(t, src, "fcuda_kv_fp8_attention_owned")
	prior := -1
	for _, part := range []string{"admit(FCUDA_KV_FP8_PHASE_ATTENTION)", "geometry(prefix,tokens",
		"input_live(q,expected)", "live(out,expected,FCUDA_KV_FP8_OWNER_WORKSPACE)",
		"live(k->plane[i],sizes[i],FCUDA_KV_FP8_OWNER_PLANE)", "fcuda_kv_fp8_attention_async("} {
		at := strings.Index(body, part)
		if at <= prior {
			t.Fatalf("missing/reordered %q", part)
		}
		prior = at
	}
	for _, part := range []string{"heads%kvheads", "SIZE_MAX/sizeof(float)", "out_bytes!=expected",
		"disjoint(q,expected,out,expected)", "disjoint(k->plane[i],sizes[i],status,sizeof(int))",
		"disjoint(k->plane[i],sizes[i],k->plane[j],sizes[j])"} {
		if !strings.Contains(body, part) {
			t.Fatalf("missing %q", part)
		}
	}
	core := readCUDASource(t, "kv_fp8_attention.go")
	prior = -1
	for _, part := range []string{"t.inputs[layer].root = root", "owner.allocate(&root.output",
		"t.consume(layer", "code := use(", "readErr = t.checked()", "owner.release(root.output)"} {
		at := strings.Index(core, part)
		if at <= prior {
			t.Fatalf("missing/reordered ownership %q", part)
		}
		prior = at
	}
	adapter := readCUDASource(t, "cuda_kv_fp8_attention_qualification.go")
	for _, part := range []string{"query.buf.(*cudaBuf)", "query.Dtype != F32", "query.Layout != RowMajor",
		"query.Quant != nil", "b.n != bytes", "b.graphOncePinned", "b.invalidForSubmit()", "root: query"} {
		if !strings.Contains(adapter, part) {
			t.Fatalf("missing metadata %q", part)
		}
	}
	for _, bad := range []string{"makeTensor(", "devTr(", "cudaMu.Lock(", "cudaBufForSubmit("} {
		if strings.Contains(adapter, bad) {
			t.Fatalf("unexpected public/allocation side effect %q", bad)
		}
	}
}

// fak-test:runtime medium est=1s lane=optin
func TestCUDAKVFP8AttentionOwnerHostControl(t *testing.T) {
	if !*kvFP8AttentionHost {
		t.Skip("explicit host admission witness not selected; no CUDA evidence")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("selected host witness requires C++")
	}
	header := readCUDASource(t, "cuda_kv_fp8_lifecycle.h") +
		strings.ReplaceAll(readCUDASource(t, "cuda_kv_fp8_append.h"), "#include \"cuda_kv_fp8_lifecycle.h\"", "") +
		strings.ReplaceAll(readCUDASource(t, "cuda_kv_fp8_attention_owner.h"), "#include \"cuda_kv_fp8_append.h\"", "")
	life := readCUDASource(t, "cuda/kv_fp8_lifecycle.cuh")
	life = strings.ReplaceAll(life, "#include \"../cuda_kv_fp8_lifecycle.h\"", "")
	life = strings.ReplaceAll(life, "#include \"kv_fp8.cuh\"", "")
	appendSource := readCUDASource(t, "cuda/kv_fp8_append.cuh")
	appendSource = strings.ReplaceAll(appendSource, "#include \"../cuda_kv_fp8_append.h\"", "")
	appendSource = strings.ReplaceAll(appendSource, "#include \"kv_fp8_lifecycle.cuh\"", "")
	attention := readCUDASource(t, "cuda/kv_fp8_attention_owner.cuh")
	attention = strings.ReplaceAll(attention, "#include \"../cuda_kv_fp8_attention_owner.h\"", "")
	attention = strings.ReplaceAll(attention, "#include \"kv_fp8_append.cuh\"", "")
	i, j := strings.Index(kvFP8LifecycleMain, "int main()"), strings.Index(kvFP8AppendNativeMain, "int main()")
	if i < 0 || j < 0 {
		t.Fatal("prerequisite harness layout drift")
	}
	program := header + kvFP8LifecycleHarness + kvFP8AppendNativeStubs + kvFP8AttentionNativeStubs +
		life + appendSource + attention + kvFP8LifecycleMain[:i] + kvFP8AppendNativeMain[:j] + kvFP8AttentionNativeMain
	dir := t.TempDir()
	cpp, bin := filepath.Join(dir, "attention-owner.cpp"), filepath.Join(dir, "attention-owner")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile actual admission: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("actual admission: %v\n%s", err, out)
	}
}

const kvFP8AttentionNativeStubs = `
#include <cmath>
using std::isfinite;
static int attention_calls,attention_failure;
static int fcuda_kv_fp8_attention_async(const float *,const uint8_t *,const float *,const uint8_t *,
 const float *,float *out,int tokens,int prefix,int heads,int kvheads,int hd,float scale,int *) {
 ++attention_calls;
 assert(tokens==1 && prefix==4 && heads==8 && kvheads==4 && hd==256 && scale==1.0f);
 if(attention_failure)return attention_failure;
 for(int i=0;i<tokens*heads*hd;i++)out[i]=1.0f;
 return 0;
}
`

const kvFP8AttentionNativeMain = `
static void prepare(fcuda_kv_fp8_append_destination *k,fcuda_kv_fp8_append_input *q,float **out,int **s) {
 restart();attention_calls=attention_failure=0;
 *k=destination(20,256);*q=input(8*256);
 *out=(float *)allocation(8*256*4,FCUDA_KV_FP8_OWNER_WORKSPACE);*s=status_allocation();**s=0;
}
int main() {
 fcuda_kv_fp8_append_destination k;fcuda_kv_fp8_append_input q;float *out;int *s;
 prepare(&k,&q,&out,&s);
 auto r=fcuda_kv_fp8_attention_owned(&k,q.raw,q.bytes,out,q.bytes,1,4,8,1.0f,s);
 assert(!r.status && !r.completed && attention_calls==1 && out[2047]==1.0f);
 for(int mode=0;mode<11;mode++) {
  prepare(&k,&q,&out,&s);size_t qbytes=q.bytes,obytes=q.bytes;int heads=8,prefix=4;float scale=1.0f;
  if(mode==0)g_managed_live[(void *)q.raw]=q.bytes;
  if(mode==1)g_panel_live[(void *)q.raw]=q.bytes;
  if(mode==2)g_alloc_tags[q.raw].cls=10;
  if(mode==3)qbytes-=4;
  if(mode==4)obytes-=4;
  if(mode==5)out=(float *)q.raw;
  if(mode==6)k.plane[2]=k.plane[1];
  if(mode==7)s=(int *)out;
  if(mode==8)heads=5;
  if(mode==9)prefix=INT_MAX;
  if(mode==10)scale=NAN;
  r=fcuda_kv_fp8_attention_owned(&k,q.raw,qbytes,out,obytes,1,prefix,heads,scale,s);
  assert(r.status && !attention_calls && !free_calls);
 }
 prepare(&k,&q,&out,&s);g_capture_open=true;
 r=fcuda_kv_fp8_attention_owned(&k,q.raw,q.bytes,out,q.bytes,1,4,8,1.0f,s);
 assert(r.status && !attention_calls);
 prepare(&k,&q,&out,&s);attention_failure=17;
 r=fcuda_kv_fp8_attention_owned(&k,q.raw,q.bytes,out,q.bytes,1,4,8,1.0f,s);
 assert(r.status==17 && r.terminal && g_cuda_cleanup_status==17 && attention_calls==1 && !free_calls);
 restart();return 0;
}
`
