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

var kvFP8LifecycleHost = flag.Bool("fak-kv-fp8-lifecycle-host", false, "run source-bound native lifetime host controls; not CUDA execution")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVFP8LifecycleSourceContract(t *testing.T) {
	src := readCUDASource(t, "cuda/kv_fp8_lifecycle.cuh")
	for _, fragment := range []string{
		"g_live.find((void *)p)", "g_managed_live.find((void *)p)", "g_panel_live.find((void *)p)",
		"kv_snapshot_release_refused((void *)p)", "t.kind == FAK_ALLOC_DEVICE_LOCAL",
		"strcmp(g_alloc_sites[t.site].name,site) == 0",
		"cudaPeekAtLastError()", "device != 0", "g_capture_open",
	} {
		if !strings.Contains(src, fragment) {
			t.Fatalf("missing preflight %q", fragment)
		}
	}
	finish := cudaFunctionBody(t, src, "fcuda_kv_fp8_finish_owned")
	prior := -1
	for _, fragment := range []string{
		"holder->next = readbacks; readbacks = holder",
		"cudaHostAlloc(&holder->host", "cudaMemcpyAsync(holder->host,status",
		"cudaStreamSynchronize(g_stream)", "r.completed = 1", "r.numerical =",
		"cudaFreeHost(holder->host)", "holder->host = nullptr", "free(holder)",
	} {
		at := strings.Index(finish, fragment)
		if at <= prior {
			t.Fatalf("missing/reordered readback lifetime %q", fragment)
		}
		prior = at
	}
	release := cudaFunctionBody(t, src, "fcuda_kv_fp8_release_owned")
	prior = -1
	for _, fragment := range []string{"live(expected,bytes", "cudaStreamSynchronize(g_stream)", "cudaFree(expected)", "r.retired = 1", "fak_alloc_untrack(expected)", "g_live.erase(expected)"} {
		at := strings.Index(release, fragment)
		if at <= prior {
			t.Fatalf("missing/reordered retirement %q", fragment)
		}
		prior = at
	}
	for _, forbidden := range []string{"fcuda_free(", "fcuda_pool_put(", "cudaMemcpyAsync(&", "cudaGetLastError("} {
		if strings.Contains(src, forbidden) {
			t.Fatalf("unsafe/incompatible lifetime path %q", forbidden)
		}
	}
}

// This compiles the ACTUAL lifecycle body with fake CUDA/ledger calls, not a
// rewritten control model. It has no GPU kernel and supplies no device proof.
// fak-test:runtime medium est=1s lane=optin
func TestCUDAKVFP8LifecycleHostControl(t *testing.T) {
	if !*kvFP8LifecycleHost {
		t.Skip("explicit source-bound host control not selected; no verification credit")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested native host witness needs a C++ compiler")
	}
	header := readCUDASource(t, "cuda_kv_fp8_lifecycle.h")
	source := readCUDASource(t, "cuda/kv_fp8_lifecycle.cuh")
	source = strings.ReplaceAll(source, "#include \"../cuda_kv_fp8_lifecycle.h\"", "")
	source = strings.ReplaceAll(source, "#include \"kv_fp8.cuh\"", "")
	program := header + kvFP8LifecycleHarness + source + kvFP8LifecycleMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "lifecycle.cpp"), filepath.Join(dir, "lifecycle")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile source-bound control: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("source-bound control: %v\n%s", err, out)
	}
}

const kvFP8LifecycleHarness = `
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <new>
#include <unordered_map>
#include <unordered_set>
using cudaError_t = int;
enum { cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorMemoryAllocation=2,
       cudaErrorNotReady=34,cudaErrorInvalidDevice=101,cudaErrorStreamCaptureUnsupported=900,
       cudaMemcpyDeviceToDevice=3,cudaMemcpyDeviceToHost=2,cudaHostAllocDefault=0 };
enum { FAK_ALLOC_KV_CACHE=3,FAK_ALLOC_ACTIVATIONS_SCRATCH=5,
       FAK_ALLOC_DEVICE_LOCAL=0,FAK_ALLOC_PINNED_HOST=2,FAK_ALLOC_SITE_OVERFLOW=0,FAK_ALLOC_SITE_CAP=16 };
static int g_stream,g_cuda_cleanup_status;
static bool g_capture_open,uncertain,throw_track,miss_untrack,allocation_with_error;
static int device,device_error,pending_error,fail_alloc,fail_host_alloc,fail_copy,fail_fence,fail_free,fail_freehost;
static int alloc_calls,hostalloc_calls,copy_calls,sync_calls,free_calls,hostfree_calls,transfer_calls;
static uint64_t g_host_bytes;
static void *quarantined;
struct fak_alloc_tag { int cls,kind,site; size_t bytes; };
struct Site { char name[64]; };
static Site g_alloc_sites[FAK_ALLOC_SITE_CAP];
static std::atomic<int> g_alloc_site_n{1};
static std::unordered_map<void *,size_t> g_live,g_managed_live,g_panel_live;
static std::unordered_map<const void *,fak_alloc_tag> g_alloc_tags;
static std::unordered_set<void *> device_blocks,host_blocks;
struct fak_alloc_tags_guard { fak_alloc_tags_guard() {} ~fak_alloc_tags_guard() {} };
static int fak_alloc_site_intern(const char *name) {
 for(int i=1;i<g_alloc_site_n.load();i++) if(!strcmp(g_alloc_sites[i].name,name)) return i;
 int n=g_alloc_site_n.load(); if(n>=FAK_ALLOC_SITE_CAP) return 0;
 strncpy(g_alloc_sites[n].name,name,63);g_alloc_site_n=n+1;return n;
}
#define FAK_ALLOC_SITE(name) fak_alloc_site_intern(name)
static void fak_alloc_track(const void *p,int cls,int kind,int site,size_t bytes) {
 if(throw_track) throw std::bad_alloc();
 g_alloc_tags[p]={cls,kind,site,bytes};
}
static bool fak_alloc_untrack(const void *p) { if(miss_untrack)return false;return g_alloc_tags.erase(p)==1; }
static bool kv_snapshot_release_refused(void *p) { return p==quarantined; }
static bool v41_candidates_stream_uncertain() { return uncertain || g_cuda_cleanup_status!=0; }
static cudaError_t cudaGetDevice(int *p) { *p=device;return device_error; }
static cudaError_t cudaPeekAtLastError() { return pending_error; }
static cudaError_t fcuda_alloc_status(cudaError_t e) {return e;}
static cudaError_t cudaMalloc(void **p,size_t n) {
 ++alloc_calls;
 if(fail_alloc && !allocation_with_error)return fail_alloc;
 *p=malloc(n);assert(*p);device_blocks.insert(*p);
 return fail_alloc;
}
static cudaError_t cudaHostAlloc(void **p,size_t n,unsigned) {
 ++hostalloc_calls;if(fail_host_alloc)return fail_host_alloc;
 *p=malloc(n);assert(*p);host_blocks.insert(*p);return 0;
}
static cudaError_t cudaMemcpyAsync(void *d,const void *s,size_t n,int,int) {
 ++copy_calls;if(fail_copy)return fail_copy;memcpy(d,s,n);return 0;
}
static cudaError_t cudaStreamSynchronize(int) {++sync_calls;return fail_fence;}
static cudaError_t cudaFree(void *p) {
 ++free_calls;if(fail_free)return fail_free;assert(device_blocks.erase(p)==1);free(p);return 0;
}
static cudaError_t cudaFreeHost(void *p) {
 ++hostfree_calls;if(fail_freehost)return fail_freehost;assert(host_blocks.erase(p)==1);free(p);return 0;
}
static void fcuda_ts_untimed_transfer() {++transfer_calls;}
static int fcuda_kv_fp8_begin_async(int *status) {*status=0;return 0;}
`

const kvFP8LifecycleMain = `
static void reset() {
 for(void *p:device_blocks) { free(p); } device_blocks.clear();
 for(void *p:host_blocks) { free(p); } host_blocks.clear();
 while(fak_kv_fp8_lifecycle::readbacks) {
  auto *p=fak_kv_fp8_lifecycle::readbacks;fak_kv_fp8_lifecycle::readbacks=p->next;free(p);
 }
 g_live.clear();g_managed_live.clear();g_panel_live.clear();g_alloc_tags.clear();
 memset(g_alloc_sites,0,sizeof(g_alloc_sites));g_alloc_site_n=1;
 g_capture_open=uncertain=throw_track=miss_untrack=allocation_with_error=false;
 g_cuda_cleanup_status=device=device_error=pending_error=fail_alloc=fail_host_alloc=fail_copy=fail_fence=fail_free=fail_freehost=0;
 alloc_calls=hostalloc_calls=copy_calls=sync_calls=free_calls=hostfree_calls=transfer_calls=0;
 g_host_bytes=0;quarantined=nullptr;
}
static void *allocation(size_t bytes,int role=FCUDA_KV_FP8_OWNER_PLANE) {
 void *p=nullptr;auto r=fcuda_kv_fp8_allocate_owned(&p,bytes,role);
 assert(r.status==0 && !r.terminal && !r.completed && p);return p;
}
static int *status_allocation() {return (int *)allocation(sizeof(int),FCUDA_KV_FP8_OWNER_WORKSPACE);}
int main() {
 reset();
 auto *s=(unsigned char *)allocation(64);auto *d=(unsigned char *)allocation(64);
 for(int i=0;i<64;i++)s[i]=(unsigned char)i;
 auto r=fcuda_kv_fp8_copy_owned(d,64,8,s,64,16,32);
 assert(!r.status && !r.completed && copy_calls==1 && !memcmp(d+8,s+16,32));
 r=fcuda_kv_fp8_finish_owned(nullptr,0);
 assert(!r.status && r.completed && r.numerical_checked && !r.numerical && sync_calls==1 && hostalloc_calls==0);
 r=fcuda_kv_fp8_release_owned(d,64);
 assert(!r.status && r.retired && free_calls==1 && sync_calls==2 && !g_live.count(d));

 reset();void *p=nullptr;g_capture_open=true;
 r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(r.status && !r.terminal && !p && !alloc_calls && !g_cuda_cleanup_status);
 reset();p=nullptr;r=fcuda_kv_fp8_allocate_owned(&p,64,42);
 assert(r.status && !r.terminal && !alloc_calls);
 reset();p=nullptr;device=1;r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(r.status && !r.terminal && !alloc_calls);
 reset();p=nullptr;device_error=17;r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(r.status==17 && r.terminal && !alloc_calls);
 reset();p=nullptr;pending_error=11;r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(r.status==11 && r.terminal && pending_error==11 && !alloc_calls);
 reset();p=nullptr;fail_alloc=cudaErrorMemoryAllocation;r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(r.status==cudaErrorMemoryAllocation && !r.terminal && !p && !g_cuda_cleanup_status);
 reset();p=nullptr;fail_alloc=cudaErrorMemoryAllocation;allocation_with_error=true;
 r=fcuda_kv_fp8_allocate_owned(&p,64,0);assert(p && r.terminal && !free_calls);
 reset();p=nullptr;throw_track=true;r=fcuda_kv_fp8_allocate_owned(&p,64,0);
 assert(p && r.terminal && r.status==FCUDA_KV_FP8_BOOKKEEPING_FAILED && !free_calls);

 for(int mode=0;mode<5;mode++) {
  reset();s=(unsigned char *)allocation(64);d=(unsigned char *)allocation(64);
  if(mode==0) r=fcuda_kv_fp8_copy_owned(d,63,0,s,64,0,4);
  if(mode==1) r=fcuda_kv_fp8_copy_owned(s,64,0,s,64,8,4);
  if(mode==2) r=fcuda_kv_fp8_copy_owned(d,64,63,s,64,0,2);
  if(mode==3) {g_alloc_tags[s].cls=0;r=fcuda_kv_fp8_copy_owned(d,64,0,s,64,0,4);}
  if(mode==4) {quarantined=s;r=fcuda_kv_fp8_copy_owned(d,64,0,s,64,0,4);}
  assert(r.status && r.terminal && !copy_calls);
 }
 reset();p=allocation(8,FCUDA_KV_FP8_OWNER_WORKSPACE);
 r=fcuda_kv_fp8_finish_owned((int *)p,1);
 assert(r.status && r.terminal && !hostalloc_calls && !copy_calls && !sync_calls);
 reset();int *st=status_allocation();r=fcuda_kv_fp8_begin_owned(st);
 assert(!r.status && !r.completed);*st=8;
 r=fcuda_kv_fp8_finish_owned(st,1);
 assert(!r.status && !r.terminal && r.completed && r.numerical_checked && r.numerical==8);
 assert(g_host_bytes==4 && transfer_calls==1 && hostfree_calls==1 && !fak_kv_fp8_lifecycle::readbacks);

 for(int phase=0;phase<4;phase++) {
  reset();st=status_allocation();*st=2;
  if(phase==0)fail_host_alloc=7;
  if(phase==1)fail_copy=7;
  if(phase==2)fail_fence=7;
  if(phase==3)fail_freehost=7;
  r=fcuda_kv_fp8_finish_owned(st,1);
  assert(r.status==7 && r.terminal && fak_kv_fp8_lifecycle::readbacks);
  if(phase<3) assert(!r.completed && !r.numerical_checked && !r.numerical && !hostfree_calls && !g_host_bytes);
  else assert(r.completed && r.numerical_checked && r.numerical==2 && hostfree_calls==1 && g_host_bytes==4);
 }
 for(int phase=0;phase<3;phase++) {
  reset();p=allocation(32);
  if(phase==0)fail_fence=7;
  if(phase==1)fail_free=7;
  if(phase==2)miss_untrack=true;
  r=fcuda_kv_fp8_release_owned(p,32);
  assert(r.status && r.terminal);
  if(phase<2) assert(!r.retired && g_live.count(p));
  else assert(r.retired && !g_live.count(p));
 }
 reset();p=allocation(4,FCUDA_KV_FP8_OWNER_WORKSPACE);
 r=fcuda_kv_fp8_release_owned(p,4);assert(!r.status && r.retired);
 reset();g_cuda_cleanup_status=9;
 r=fcuda_kv_fp8_lifecycle_failed(17,FCUDA_KV_FP8_PHASE_COPY);
 assert(r.status==17 && r.terminal && g_cuda_cleanup_status==9);
 reset();
 return 0;
}
`
