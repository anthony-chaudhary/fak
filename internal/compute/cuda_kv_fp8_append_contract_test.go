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

var kvFP8AppendHost = flag.Bool("fak-kv-fp8-append-host", false, "run source-bound append ingress host controls, not CUDA kernels")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVFP8AppendSourceContract(t *testing.T) {
	src := readCUDASource(t, "cuda/kv_fp8_append.cuh")
	for _, want := range []string{"g_live.find", "it->second != bytes", "g_managed_live.find",
		"g_panel_live.find", "kv_snapshot_release_refused", "t.kind == FAK_ALLOC_DEVICE_LOCAL",
		"FAK_ALLOC_ACTIVATIONS_SCRATCH", "FAK_ALLOC_PREFILL_SCRATCH", "FAK_ALLOC_ATTENTION_SCRATCH",
		"tokens>INT_MAX-prefix", "all>INT_MAX/heads", "all*heads>INT_MAX/hd",
		"SIZE_MAX/sizeof(float)", "disjoint(status,sizeof(int),input->raw,input->bytes)",
		"FCUDA_KV_FP8_OWNER_PLANE", "fcuda_kv_fp8_validate_async"} {
		if !strings.Contains(src, want) {
			t.Fatalf("missing %q", want)
		}
	}
	body := cudaFunctionBody(t, src, "fcuda_kv_fp8_append_ingest_owned")
	previous := -1
	for _, want := range []string{"geometry(prefix", "inputs(input", "destination(dst", "cudaMemcpyAsync(",
		"fcuda_kv_fp8_quantize_async((uint8_t *)dst->plane[1]", "fcuda_kv_fp8_quantize_async((uint8_t *)dst->plane[2]"} {
		at := strings.Index(body, want)
		if at <= previous {
			t.Fatalf("wrong ordering %q", want)
		}
		previous = at
	}
	goSource := readCUDASource(t, "cuda_kv_fp8_append_qualification.go")
	for _, want := range []string{"t.buf.(*cudaBuf)", "t.Dtype != F32", "t.Layout != RowMajor", "t.Quant != nil",
		"b.n != sizes[kvFP8Raw]", "b.graphOncePinned", "b.invalidForSubmit()", "root: operands"} {
		if !strings.Contains(goSource, want) {
			t.Fatalf("missing input guard %q", want)
		}
	}
	for _, bad := range []string{"cudaBufForSubmit(", "makeTensor(", "devTr(", "cudaMu.Lock("} {
		if strings.Contains(goSource, bad) {
			t.Fatalf("unexpected preflight/activation effect %q", bad)
		}
	}
}

// Reuses the already source-bound lifecycle harness rather than inventing a
// second allocator model. Only CUDA/ledger/numerical calls are fake; actual
// admission/range/pointer arithmetic and wrapper ordering are compiled intact.
// fak-test:runtime medium est=1s lane=optin
func TestCUDAKVFP8AppendHostControl(t *testing.T) {
	if !*kvFP8AppendHost {
		t.Skip("explicit append host control not selected; no execution credit")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested host witness needs C++")
	}
	header := readCUDASource(t, "cuda_kv_fp8_lifecycle.h")
	appendHeader := strings.ReplaceAll(readCUDASource(t, "cuda_kv_fp8_append.h"), "#include \"cuda_kv_fp8_lifecycle.h\"", "")
	lifecycle := readCUDASource(t, "cuda/kv_fp8_lifecycle.cuh")
	lifecycle = strings.ReplaceAll(lifecycle, "#include \"../cuda_kv_fp8_lifecycle.h\"", "")
	lifecycle = strings.ReplaceAll(lifecycle, "#include \"kv_fp8.cuh\"", "")
	source := readCUDASource(t, "cuda/kv_fp8_append.cuh")
	source = strings.ReplaceAll(source, "#include \"../cuda_kv_fp8_append.h\"", "")
	source = strings.ReplaceAll(source, "#include \"kv_fp8_lifecycle.cuh\"", "")
	at := strings.Index(kvFP8LifecycleMain, "int main()")
	if at < 0 {
		t.Fatal("lifecycle helper extraction drift")
	}
	program := header + appendHeader + kvFP8LifecycleHarness + kvFP8AppendNativeStubs +
		lifecycle + source + kvFP8LifecycleMain[:at] + kvFP8AppendNativeMain
	dir := t.TempDir()
	cpp, bin := filepath.Join(dir, "append.cpp"), filepath.Join(dir, "append")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile source-bound ingress: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("source-bound ingress: %v\n%s", err, out)
	}
}

const kvFP8AppendNativeStubs = `
enum { FAK_ALLOC_PREFILL_SCRATCH=6,FAK_ALLOC_ATTENTION_SCRATCH=7 };
static int validate_calls,quantize_calls,validate_mask,quantize_failure;
static int fcuda_kv_fp8_validate_async(const float *,const float *,const float *,int,int *status) {
 ++validate_calls;*status|=validate_mask;return 0;
}
static int fcuda_kv_fp8_quantize_async(uint8_t *codes,float *scales,const float *,int rows,int hd,int *) {
 ++quantize_calls;if(quantize_failure==quantize_calls)return 17;
 memset(codes,0x38,(size_t)rows*hd);for(int i=0;i<rows;i++)scales[i]=1.0f;return 0;
}
`

const kvFP8AppendNativeMain = `
static fcuda_kv_fp8_append_input input(int elements) {
 fcuda_kv_fp8_append_input in={};in.bytes=(size_t)elements*4;
 float *p[3]={};
 for(int i=0;i<3;i++) {
  p[i]=(float *)allocation(in.bytes);g_alloc_tags[p[i]].cls=FAK_ALLOC_ACTIVATIONS_SCRATCH;
  for(int j=0;j<elements;j++)p[i][j]=(float)j;
 }
 in.raw=p[0];in.key=p[1];in.value=p[2];return in;
}
static fcuda_kv_fp8_append_destination destination(int rows,int hd) {
 fcuda_kv_fp8_append_destination d={};
 size_t n=(size_t)rows*hd;size_t sizes[5]={n*4,n,n,(size_t)rows*4,(size_t)rows*4};
 for(int i=0;i<5;i++){d.bytes[i]=sizes[i];d.plane[i]=allocation(sizes[i]);memset(d.plane[i],0xa5,sizes[i]);}
 return d;
}
static void restart(){reset();validate_calls=quantize_calls=validate_mask=quantize_failure=0;}
int main() {
 restart();auto in=input(12);auto dst=destination(6,3);int *s=status_allocation();*s=0;
 auto r=fcuda_kv_fp8_append_validate_owned(&in,12,s);assert(!r.status && !r.completed && validate_calls==1);
 r=fcuda_kv_fp8_finish_owned(s,1);assert(!r.status && r.completed && !r.numerical);
 r=fcuda_kv_fp8_append_ingest_owned(&dst,&in,1,2,2,3,s);
 assert(!r.status && !r.completed && quantize_calls==2);
 assert(!memcmp((float *)dst.plane[0]+6,in.raw,48));
 for(int i=0;i<24;i++)assert(((unsigned char *)dst.plane[0])[i]==0xa5);
 for(int p=1;p<=2;p++){for(int i=0;i<6;i++)assert(((uint8_t *)dst.plane[p])[i]==0xa5);
  for(int i=6;i<18;i++)assert(((uint8_t *)dst.plane[p])[i]==0x38);}
 for(int p=3;p<=4;p++)for(int i=2;i<6;i++)assert(((float *)dst.plane[p])[i]==1.0f);
 for(int mode=0;mode<9;mode++){
  restart();in=input(12);dst=destination(6,3);s=status_allocation();*s=0;
  if(mode==0)g_alloc_tags[in.key].cls=10;
  if(mode==1)g_managed_live[(void *)in.key]=in.bytes;
  if(mode==2)g_panel_live[(void *)in.key]=in.bytes;
  if(mode==3)quarantined=(void *)in.key;
  if(mode==4)in.bytes-=4;
  if(mode==5)dst.bytes[0]-=4;
  if(mode==6)dst.plane[2]=dst.plane[1];
  if(mode==7)dst.plane[0]=(void *)in.raw;
  if(mode==8)g_capture_open=true;
  r=fcuda_kv_fp8_append_ingest_owned(&dst,&in,1,2,2,3,s);
  assert(r.status && !copy_calls && !quantize_calls);
 }
 restart();in=input(1);s=status_allocation();*s=0;
 in.raw=(float *)s;in.key=(float *)s;in.value=(float *)s;
 r=fcuda_kv_fp8_append_validate_owned(&in,1,s);assert(r.status && !validate_calls);
 dst=destination(1,1);
 r=fcuda_kv_fp8_append_ingest_owned(&dst,&in,0,1,1,1,s);assert(r.status && !copy_calls);
 restart();in=input(12);s=status_allocation();*s=0;validate_mask=1;
 r=fcuda_kv_fp8_append_validate_owned(&in,12,s);assert(!r.status);
 r=fcuda_kv_fp8_finish_owned(s,1);assert(!r.status && r.completed && r.numerical==1 && !quantize_calls);
 for(int mode=0;mode<3;mode++){
  restart();in=input(12);dst=destination(6,3);s=status_allocation();*s=0;
  if(mode==0)fail_copy=7;else quantize_failure=mode;
  r=fcuda_kv_fp8_append_ingest_owned(&dst,&in,1,2,2,3,s);
  assert(r.status && r.terminal && g_cuda_cleanup_status && !free_calls);
  assert(quantize_calls==mode);
 }
 int rows,elements,old;
 assert(!fak_kv_fp8_append::geometry(INT_MAX,1,1,1,&rows,&elements,&old));
 assert(!fak_kv_fp8_append::geometry(0,INT_MAX,2,1,&rows,&elements,&old));
 assert(!fak_kv_fp8_append::geometry(0,1,INT_MAX,2,&rows,&elements,&old));
 assert(fak_kv_fp8_append::geometry(0,1,1,INT_MAX,&rows,&elements,&old)==(SIZE_MAX/4>=(size_t)INT_MAX));
 assert(!fak_kv_fp8_append::disjoint((void *)(UINTPTR_MAX-1),4,(void *)16,4));
 restart();return 0;
}
`
