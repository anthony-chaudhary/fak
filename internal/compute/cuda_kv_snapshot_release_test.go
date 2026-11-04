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

var cudaKVSnapshotReleaseHost = flag.Bool("fak-cuda-kv-snapshot-release-host", false, "run exact-source snapshot release ownership witnesses with a host C++ shim")

// Keep the dedicated release seam bound to every existing generic refusal and
// to the actual allocation-site bridge. This is source evidence, not CUDA proof.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVSnapshotReleaseSourceContract(t *testing.T) {
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	native := readCUDASource(t, "cuda/kv_precision.cuh")
	generic := cudaFunctionBody(t, memory, "fcuda_free")
	cut := strings.Index(generic, "q4_coeff_cache_evict(d);")
	if cut < 0 {
		t.Fatal("generic free's first effect changed")
	}
	want := strings.TrimSpace(generic[1:cut])
	refusal := strings.TrimSpace(cudaFunctionBody(t, native, "kv_snapshot_release_refused"))
	if !strings.HasSuffix(refusal, "return false;\n}") {
		t.Fatal("snapshot refusal must have an explicit no-effect success")
	}
	refusal = strings.TrimSuffix(refusal[1:], "return false;\n}")
	refusal = strings.ReplaceAll(refusal, "return true;", "return;")
	if strings.TrimSpace(refusal) != want {
		t.Fatal("snapshot release lost, reordered or changed a generic refusal")
	}
	ordered := func(body string, tokens ...string) {
		t.Helper()
		at := 0
		for _, token := range tokens {
			next := strings.Index(body[at:], token)
			if next < 0 {
				t.Fatalf("missing/reordered release contract %q", token)
			}
			at += next + len(token)
		}
	}
	body := cudaFunctionBody(t, native, "fcuda_kv_snapshot_release_checked")
	ordered(body, "*disposition = FCUDA_KV_SNAPSHOT_RETAINED", "if (g_cuda_cleanup_status != 0)", "v41_candidates_stream_uncertain() || kv_snapshot_release_refused(expected)", "g_live.find(expected)", "g_managed_live.find(expected)", "g_panel_live.find(expected)", "matches != 1", "fak_alloc_tags_guard guard", "g_alloc_tags.find(expected)", "tag.site <= FAK_ALLOC_SITE_OVERFLOW", "strcmp(g_alloc_sites[tag.site].name, \"kv-snapshot-conversion\")", "const cudaError_t status = cudaFree(expected);", "if (status != cudaSuccess) return fcuda_kv_snapshot_release_failed((int)status);", "fak_alloc_untrack(expected);", "g_managed_live.erase(managed);", "g_panel_pool[bytes].push_back(expected)", "g_panel_live.erase(panel)", "g_live.erase(ordinary)", "fcuda_pool_put(expected, bytes)", "*disposition = FCUDA_KV_SNAPSHOT_RETIRED", "if (g_cuda_cleanup_status != 0) return g_cuda_cleanup_status;")
	for _, token := range []string{"tag.bytes != bytes", "tag.cls != cls", "tag.kind != kind", "tag.site >= sites", "tag.site >= FAK_ALLOC_SITE_CAP", "g_panel_live_bytes < bytes", "g_capture_open"} {
		if !strings.Contains(body, token) {
			t.Fatalf("release lost exact provenance/admission %q", token)
		}
	}
	for _, forbidden := range []string{"fcuda_free(", "q4_coeff_cache_evict(", "cudaStreamSynchronize(", "cudaDeviceSynchronize(", "cudaGetDevice("} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("snapshot cleanup gained unchecked/free/drain work: %s", forbidden)
		}
	}
	failed := strings.Join(strings.Fields(cudaFunctionBody(t, native, "fcuda_kv_snapshot_release_failed")), " ")
	if failed != "{ if (status == 0) status = (int)cudaErrorInvalidValue; (void)kv_snapshot_status(status); return g_cuda_cleanup_status; }" {
		t.Fatal("malformed receipt no longer closes admission with first-error precedence")
	}
	header := readCUDASource(t, "cuda_kv_precision.h")
	for _, token := range []string{"FCUDA_KV_SNAPSHOT_RETAINED = 0", "FCUDA_KV_SNAPSHOT_RETIRED = 1", "int fcuda_kv_snapshot_release_checked(void *expected, size_t bytes, int *disposition);", "int fcuda_kv_snapshot_release_failed(int status);"} {
		if !strings.Contains(header, token) {
			t.Fatalf("missing checked release ABI %q", token)
		}
	}
	goSource := readCUDASource(t, "cuda_kv.go")
	if !strings.Contains(goSource, `holder.conversion = k.be.dallocClass(src.len*4, MemoryScratchpad, "kv-snapshot-conversion")`) ||
		!strings.Contains(goSource, "C.size_t(holder.conversion.n), &disposition") {
		t.Fatal("release is not bound to the exact returned conversion owner")
	}
	bridge := readCUDASource(t, "cuda_alloc_ledger.go")
	if !strings.Contains(bridge, "C.int(DeviceAllocClassFor(class, site, promptPanel)), cudaAllocSite(site)") || !strings.Contains(bridge, "C.fcuda_alloc_site_intern(cs)") {
		t.Fatal("Go allocation class/site bridge changed")
	}
	ledger := readCUDASource(t, "cuda/alloc_ledger.h")
	if !strings.Contains(ledger, "if (n >= FAK_ALLOC_SITE_CAP) return FAK_ALLOC_SITE_OVERFLOW;") || !strings.Contains(ledger, "static const int FAK_ALLOC_SITE_OVERFLOW = 0;") {
		t.Fatal("native site overflow contract changed")
	}
	if got := DeviceAllocClassFor(MemoryScratchpad, "kv-snapshot-conversion", false); got != AllocActivationsScratch {
		t.Fatalf("ordinary snapshot class changed: %v", got)
	}
	if got := DeviceAllocClassFor(MemoryScratchpad, "kv-snapshot-conversion", true); got != AllocPrefillScratch {
		t.Fatalf("panel snapshot class changed: %v", got)
	}
}

// Actual shared predicate, peer refusal, ledger, pending inventory, pool and
// snapshot-release bodies. Only CUDA/runtime allocation boundaries are faked.
// These two native profiles are NOT a CUDA/GPU or cgo qualification.
// fak-test:runtime slow est=120s lane=optin
func TestCUDAKVSnapshotReleaseHostContract(t *testing.T) {
	if !*cudaKVSnapshotReleaseHost {
		t.Skip("set -fak-cuda-kv-snapshot-release-host for the host C++ ownership witness")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	native := readCUDASource(t, "cuda/kv_precision.cuh")
	root := readCUDASource(t, "cuda_kernels.cu")
	program := strings.TrimPrefix(readCUDASource(t, "cuda/alloc_ledger.h"), "#pragma once\n") + "\n" +
		readCUDASource(t, "cuda_kv_precision.h") + cudaKVSnapshotReleaseHarness +
		"\nstatic bool v41_candidates_stream_uncertain(void) " + cudaFunctionBody(t, root, "v41_candidates_stream_uncertain") +
		"\n" + readCUDASource(t, "cuda/v41_compressor_pending.cuh")
	for _, entry := range []struct{ source, signature, name string }{
		{memory, "static bool fcuda_pool_may_release(void) ", "fcuda_pool_may_release"},
		{memory, "static void fcuda_pool_evict_to_cap(void) ", "fcuda_pool_evict_to_cap"},
		{memory, "static void fcuda_pool_put(void *d, size_t bytes) ", "fcuda_pool_put"},
		{native, "static int kv_snapshot_status(int status) ", "kv_snapshot_status"},
		{native, "static bool kv_snapshot_release_refused(void *d) ", "kv_snapshot_release_refused"},
		{native, "extern \"C\" int fcuda_kv_snapshot_release_failed(int status) ", "fcuda_kv_snapshot_release_failed"},
		{native, "extern \"C\" int fcuda_kv_snapshot_release_checked(void *expected, size_t bytes, int *disposition) ", "fcuda_kv_snapshot_release_checked"},
	} {
		program += "\n" + entry.signature + cudaFunctionBody(t, entry.source, entry.name)
	}
	program += cudaKVSnapshotReleaseMain
	dir := t.TempDir()
	cpp := filepath.Join(dir, "snapshot_release.cpp")
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	for _, profile := range []string{"default", "selected"} {
		t.Run(profile, func(t *testing.T) {
			binary := filepath.Join(dir, profile)
			if runtime.GOOS == "windows" {
				binary += ".exe"
			}
			args := []string{"-std=c++17", "-O0", cpp, "-o", binary}
			if profile == "selected" {
				args = append([]string{"-DSNAPSHOT_TEST_SELECTED=1"}, args...)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 55*time.Second)
			defer cancel()
			build := exec.CommandContext(ctx, compiler, args...)
			build.WaitDelay = time.Second
			if output, err := build.CombinedOutput(); err != nil {
				t.Fatalf("compile source-bound release: %v\n%s", err, output)
			}
			run := exec.CommandContext(ctx, binary)
			run.WaitDelay = time.Second
			if output, err := run.CombinedOutput(); err != nil {
				t.Fatalf("source-bound release: %v\n%s", err, output)
			}
		})
	}
}

const cudaKVSnapshotReleaseHarness = `
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#ifdef SNAPSHOT_TEST_SELECTED
#define FAK_CUDA_V41_CHECKED_STREAM
#define FAK_CUDA_V41_ROUTED_SWIGLU
#define FAK_CUDA_V41_ENGRAM_GATE
#define FAK_CUDA_V41_COMPRESSOR_POOL
#define FAK_CUDA_V41_MHC
#define FAK_CUDA_V41_BF16_PUBLISH
#define FAK_CUDA_V41_TAIL_ROPE
#define FAK_CUDA_V41_TAIL_ROPE_TEST
#define FAK_CUDA_V41_INDEXER
#define FAK_CUDA_FLASHMLA_PREFILL
#define FAK_CUDA_DEEPGEMM_WKV
#define FAK_CUDA_V41_SHARED_ATTENTION
#define FAK_CUDA_V41_COMPRESSOR
#endif
#define CK(value) do { (void)(value); } while (0)
using cudaError_t = int;
static constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorNotReady=34;
static int g_cuda_cleanup_status=0, free_status=0, frees=0, syncs=0, g_stream=1;
static void *last_free=nullptr;
static bool g_capture_open=false, g_pool_graph_deep=false;
static bool g_v41_router_uncertain=false, g_v41_mmid_uncertain=false;
static bool g_v41_routed_swiglu_uncertain=false, g_v41_engram_uncertain=false, g_v41_pool_uncertain=false;
static bool g_v41_mhc_uncertain=false, g_v41_bf16_uncertain=false, g_v41_rope_uncertain=false, g_v41_indexer_uncertain=false;
static bool g_flashmla_uncertain=false, g_deepgemm_wkv_uncertain=false, g_v41_sa_uncertain=false, g_v41_compressor_uncertain=false;
static std::atomic<bool> g_v41_checked_stream_uncertain{false};
static const void *g_v41_routed_swiglu_quarantine[1]={}, *g_v41_engram_quarantine[1]={}, *g_v41_sa_quarantine[1]={};
static const void *g_v41_pool_quarantine[1]={}, *g_deepgemm_wkv_quarantine[1]={}, *g_flashmla_quarantine[1]={};
static const void *g_v41_mhc_quarantine[1]={}, *g_v41_indexer_quarantine[1]={}, *g_v41_bf16_quarantine[1]={}, *g_v41_rope_quarantine[1]={};
static unsigned long long g_v41_rope_test_free_attempts=0, g_v41_compressor_completed=0;
static std::map<void*,size_t> g_live,g_managed_live,g_panel_live;
static std::map<size_t,std::vector<void*>> g_pool,g_panel_pool;
static size_t g_pool_bytes=0,g_pool_peak_bytes=0,g_pool_cap_bytes=128;
static size_t g_panel_live_bytes=0,g_panel_cached_bytes=0;
static unsigned long long g_pool_clock=0,g_pool_evictions=0;
static std::map<size_t,unsigned long long> g_pool_touch;
static cudaError_t cudaFree(void *p) { ++frees; last_free=p; return free_status; }
static cudaError_t cudaStreamSynchronize(int) { ++syncs; return cudaSuccess; }
static void require(bool ok,const char *message) { if(!ok) { std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1); } }
static bool pooled(void *p) { for(auto &entry:g_pool) for(void *q:entry.second) if(q==p) return true;return false; }
`

const cudaKVSnapshotReleaseMain = `
static void reset() {
 g_live.clear();g_managed_live.clear();g_panel_live.clear();g_pool.clear();g_panel_pool.clear();g_pool_touch.clear();g_alloc_tags.clear();
 g_pool_bytes=g_pool_peak_bytes=g_panel_live_bytes=g_panel_cached_bytes=0;g_pool_cap_bytes=128;
 g_pool_clock=g_pool_evictions=0;g_cuda_cleanup_status=free_status=frees=syncs=0;last_free=nullptr;
 g_capture_open=g_pool_graph_deep=g_v41_router_uncertain=g_v41_mmid_uncertain=false;
 g_v41_routed_swiglu_uncertain=g_v41_engram_uncertain=g_v41_pool_uncertain=false;
 g_v41_mhc_uncertain=g_v41_bf16_uncertain=g_v41_rope_uncertain=g_v41_indexer_uncertain=false;
 g_flashmla_uncertain=g_deepgemm_wkv_uncertain=g_v41_sa_uncertain=g_v41_compressor_uncertain=false;
 g_v41_checked_stream_uncertain.store(false);
 g_v41_routed_swiglu_quarantine[0]=g_v41_engram_quarantine[0]=g_v41_sa_quarantine[0]=nullptr;
 g_v41_pool_quarantine[0]=g_deepgemm_wkv_quarantine[0]=g_flashmla_quarantine[0]=nullptr;
 g_v41_mhc_quarantine[0]=g_v41_indexer_quarantine[0]=g_v41_bf16_quarantine[0]=g_v41_rope_quarantine[0]=nullptr;
#ifdef SNAPSHOT_TEST_SELECTED
 g_v41c_pending_count=g_v41c_pending_bytes=0;g_v41c_pending_completed=false;
#endif
 // Keep interned names/IDs stable across cases, like the actual process.
 for(auto &site:g_alloc_sites){site.cls.store(FAK_ALLOC_OTHER);site.live.store(0);site.peak.store(0);site.live_count.store(0);site.allocs.store(0);}
 for(auto &cell:g_alloc_cells){cell.live.store(0);cell.peak.store(0);cell.live_count.store(0);cell.allocs.store(0);cell.frees.store(0);}
 g_alloc_untracked_frees.store(0);
}
static void own(void *p,size_t bytes,int type,const char *name="kv-snapshot-conversion") {
 if(type==0)g_live[p]=bytes;
 if(type==1)g_managed_live[p]=bytes;
 if(type==2){g_panel_live[p]=bytes;g_panel_live_bytes+=bytes;}
 fak_alloc_track(p,type==2?FAK_ALLOC_PREFILL_SCRATCH:FAK_ALLOC_ACTIVATIONS_SCRATCH,type==1?FAK_ALLOC_MANAGED:FAK_ALLOC_DEVICE_LOCAL,fak_alloc_site_intern(name),bytes);
}
static int release(void *p,size_t bytes,int &disposition){disposition=99;return fcuda_kv_snapshot_release_checked(p,bytes,&disposition);}
int main(){
 int storage[2]={};void *p=&storage[0],*older=&storage[1];int disposition=99,status=0;
 reset();own(p,16,1);status=release(p,16,disposition);
 require(status==0&&disposition==1&&g_managed_live.empty()&&g_alloc_tags.empty()&&frees==1&&syncs==0,"healthy managed receipt");
 reset();own(p,16,1);free_status=719;status=release(p,16,disposition);
 require(status==719&&disposition==0&&g_managed_live.count(p)==1&&g_alloc_tags.count(p)==1&&g_alloc_tags[p].kind==FAK_ALLOC_MANAGED&&frees==1&&g_cuda_cleanup_status==719,"failed managed release lost native owner or first status");
 require(fcuda_kv_snapshot_release_failed(888)==719,"managed first error overwritten");
 status=release(p,16,disposition);require(status==719&&disposition==0&&frees==1,"uncertain managed free retried");
 reset();own(p,16,2);status=release(p,16,disposition);
 require(status==0&&disposition==1&&g_panel_live.empty()&&g_panel_pool[16].size()==1&&g_panel_pool[16][0]==p&&g_panel_live_bytes==0&&g_panel_cached_bytes==16&&frees==0&&g_alloc_tags[p].cls==FAK_ALLOC_IDLE_POOL,"panel ownership/accounting transfer");
 reset();own(p,16,0);status=release(p,16,disposition);
 require(status==0&&disposition==1&&g_live.empty()&&pooled(p)&&frees==0&&g_pool_bytes==16&&g_alloc_tags[p].cls==FAK_ALLOC_IDLE_POOL,"ordinary pool ownership transfer");
 reset();own(p,64,0);g_pool_cap_bytes=0;status=release(p,64,disposition);
 require(status==0&&disposition==1&&g_live.empty()&&!pooled(p)&&g_alloc_tags.empty()&&frees==1,"oversize physical release");
 reset();own(p,64,0);g_pool_cap_bytes=0;free_status=719;status=release(p,64,disposition);
 require(status==719&&disposition==1&&g_live.empty()&&pooled(p)&&g_alloc_tags.count(p)==1&&frees==1&&last_free==p,"failed oversize free must transfer uncertain native pool ownership");
 reset();own(older,16,0);g_live.erase(older);fcuda_pool_put(older,16);own(p,32,0);g_pool_cap_bytes=32;free_status=719;status=release(p,32,disposition);
 require(status==719&&disposition==1&&g_live.empty()&&pooled(p)&&pooled(older)&&last_free==older&&g_alloc_tags.size()==2&&g_pool_bytes==48,"unrelated eviction error misclassified subject ownership");
 for(int mode=0;mode<10;++mode){
  reset();own(p,16,1);
  if(mode==0)g_cuda_cleanup_status=731;
  if(mode==1)g_v41_router_uncertain=true;
  if(mode==2)g_live[p]=16;
  if(mode==3)g_managed_live[p]=17;
  if(mode==4)g_alloc_tags[p].site=fak_alloc_site_intern("foreign-scratch");
  if(mode==5)g_alloc_tags[p].site=FAK_ALLOC_SITE_OVERFLOW;
  if(mode==6)g_alloc_tags[p].cls=FAK_ALLOC_WEIGHTS;
  if(mode==7)g_alloc_tags[p].kind=FAK_ALLOC_DEVICE_LOCAL;
  if(mode==8)g_alloc_tags.erase(p);
  if(mode==9)g_capture_open=true;
  status=release(p,16,disposition);
  require(status!=0&&disposition==0&&g_managed_live.count(p)==1&&frees==0&&syncs==0,"refused owner/provenance mutated or released");
  require(g_cuda_cleanup_status==status,"refusal not shared-terminal");
  if(mode==0)require(status==731,"preexisting first code replaced");
 }
 reset();own(p,16,1);require(fcuda_kv_snapshot_release_checked(p,16,nullptr)!=0&&g_managed_live.count(p)==1&&frees==0,"missing result pointer released owner");
 reset();own(p,16,1);status=release(older,16,disposition);require(status!=0&&disposition==0&&g_managed_live.count(p)==1&&frees==0,"unknown pointer fallback free");
 reset();
 const int savedSites=g_alloc_site_n.load(), snapshotSite=fak_alloc_site_intern("kv-snapshot-conversion");
 char savedName[FAK_ALLOC_SITE_NAME];std::memcpy(savedName,g_alloc_sites[snapshotSite].name,sizeof(savedName));
 std::strcpy(g_alloc_sites[snapshotSite].name,"hidden-prior-snapshot-site");
 g_alloc_site_n.store(FAK_ALLOC_SITE_CAP);own(p,16,1);
 require(g_alloc_tags[p].site==0,"real site overflow fixture");status=release(p,16,disposition);
 require(status!=0&&disposition==0&&g_managed_live.count(p)==1&&frees==0,"overflow site treated as private provenance");
 g_alloc_site_n.store(savedSites);std::memcpy(g_alloc_sites[snapshotSite].name,savedName,sizeof(savedName));
#ifdef SNAPSHOT_TEST_SELECTED
 for(const void **slot:{g_v41_routed_swiglu_quarantine,g_v41_engram_quarantine,g_v41_sa_quarantine,g_v41_pool_quarantine,g_deepgemm_wkv_quarantine,g_flashmla_quarantine,g_v41_mhc_quarantine,g_v41_indexer_quarantine,g_v41_bf16_quarantine,g_v41_rope_quarantine}){
  reset();own(p,16,0);*slot=p;status=release(p,16,disposition);
  require(status!=0&&disposition==0&&g_live.count(p)==1&&g_pool.empty()&&frees==0,"selected per-pointer quarantine bypass");
 }
 reset();own(p,16,0);require(fak_v41_compressor_pending_retain(p,16)==0,"actual pending retain refused fixture");status=release(p,16,disposition);
 require(status!=0&&disposition==0&&g_live.count(p)==1&&frees==0&&syncs==0,"pending producer bypass or extra drain");
 reset();own(p,16,0);g_v41_checked_stream_uncertain.store(true);status=release(p,16,disposition);
 require(status!=0&&disposition==0&&g_live.count(p)==1&&frees==0,"selected shared checked-stream bypass");
#endif
 std::puts("snapshot release ownership cases passed");return 0;
}
`
