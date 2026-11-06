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

var graphAllocationGenerationHost = flag.Bool("fak-cuda-graph-allocation-generation-host", false, "run exact-ledger reservation-generation host controls; no CUDA or graph activation")

// Bind the ticket to actual allocation metadata events rather than a Go
// completion counter. No graph caller is enabled by this source prerequisite.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphAllocationGenerationContract(t *testing.T) {
	source := readCUDASource(t, "cuda/alloc_ledger.h")
	ordered := func(body string, tokens ...string) {
		t.Helper()
		at := 0
		for _, token := range tokens {
			next := strings.Index(body[at:], token)
			if next < 0 {
				t.Fatalf("missing/reordered generation contract %q", token)
			}
			at += next + len(token)
		}
	}
	for _, name := range []string{"fak_alloc_track", "fak_alloc_retag"} {
		body := cudaFunctionBody(t, source, name)
		ordered(body, "fak_alloc_tags_guard guard", "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION", "fak_alloc_next_reservation_locked()", "#endif")
	}
	next := cudaFunctionBody(t, source, "fak_alloc_next_reservation_locked")
	ordered(next, "if (g_alloc_reservation_next == ~uint64_t(0)) return 0;", "return ++g_alloc_reservation_next;")
	query := cudaFunctionBody(t, source, "fcuda_alloc_reservation_id")
	ordered(query, "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION", "if (!p || !bytes) return 0;", "fak_alloc_tags_guard guard", "g_alloc_tags.find(p)", "tag.bytes != bytes", "tag.cls == FAK_ALLOC_IDLE_POOL", "tag.kind == FAK_ALLOC_PINNED_HOST", "return tag.reservation;", "#else", "(void)p; (void)bytes;", "return 0;", "#endif")
	for _, forbidden := range []string{"cudaMalloc", "cudaFree", "cudaStreamSynchronize", "cudaGetLastError", "fak_alloc_track(", "fak_alloc_retag("} {
		if strings.Contains(query, forbidden) {
			t.Fatalf("query gained an effect: %q", forbidden)
		}
	}
	if strings.Contains(cudaFunctionBody(t, source, "fcuda_alloc_ledger_reset_peak"), "reservation") {
		t.Fatal("accounting reset can recycle reservation identities")
	}
	if !strings.Contains(source, "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION\n  uint64_t reservation;\n#endif") {
		t.Fatal("ordinary tag layout gained generation state")
	}
	if !strings.Contains(readCUDASource(t, "cuda_backend.h"), "uint64_t fcuda_alloc_reservation_id(const void *p, size_t bytes);") {
		t.Fatal("missing exact ticket ABI")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	for _, name := range []string{"fcuda_malloc_tagged", "fcuda_malloc_panel_tagged"} {
		body := cudaFunctionBody(t, memory, name)
		if !strings.Contains(body, "fak_alloc_retag(d, cls, site)") || !strings.Contains(body, "fak_alloc_track(d, cls, FAK_ALLOC_DEVICE_LOCAL, site, bytes)") {
			t.Fatalf("pool/reallocation identity hook missing in %s", name)
		}
	}
}

// Compile the actual standalone native ledger in both profiles. No modeled
// counter substitutes for production track/retag/read; CUDA is not loaded.
// fak-test:runtime medium est=5s lane=optin
func TestCUDAGraphAllocationGenerationHost(t *testing.T) {
	if !*graphAllocationGenerationHost {
		t.Skip("explicit native ledger host witness not selected")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	program := strings.TrimPrefix(readCUDASource(t, "cuda/alloc_ledger.h"), "#pragma once\n") + graphAllocationGenerationMain
	dir := t.TempDir()
	cpp := filepath.Join(dir, "generation.cpp")
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	for _, qualified := range []bool{false, true} {
		name := "default"
		if qualified {
			name = "capture-once"
		}
		t.Run(name, func(t *testing.T) {
			binary := filepath.Join(dir, name)
			if runtime.GOOS == "windows" {
				binary += ".exe"
			}
			args := []string{"-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", "-pthread", cpp, "-o", binary}
			if qualified {
				args = append([]string{"-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION=1"}, args...)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, compiler, args...)
			cmd.WaitDelay = time.Second
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("compile actual ledger %s: %v\n%s", name, err, out)
			}
			cmd = exec.CommandContext(ctx, binary)
			cmd.WaitDelay = time.Second
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("actual ledger %s: %v\n%s", name, err, out)
			}
		})
	}
}

const graphAllocationGenerationMain = `
#include <algorithm>
#include <cassert>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
template<class T,class=void>struct has_reservation:std::false_type{};
template<class T>struct has_reservation<T,std::void_t<decltype(std::declval<T>().reservation)>>:std::true_type{};
#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION
static_assert(has_reservation<fak_alloc_tag>::value,"qualified tag requires reservation");
#else
static_assert(!has_reservation<fak_alloc_tag>::value,"default tag must have no reservation field");
#endif
int main(){
 int value=0,other=0,host=0;
 const int site=fak_alloc_site_intern("graph-generation-test");
 fak_alloc_track(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(value));
#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION
 uint64_t first=fcuda_alloc_reservation_id(&value,sizeof(value));assert(first!=0);
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==first);
 assert(fcuda_alloc_reservation_id(nullptr,sizeof(value))==0);
 assert(fcuda_alloc_reservation_id(&value,0)==0);
 assert(fcuda_alloc_reservation_id(&value,sizeof(value)-1)==0);
 assert(fcuda_alloc_reservation_id(&value,~size_t(0))==0);
 assert(fcuda_alloc_reservation_id((const char*)&value+1,sizeof(value)-1)==0);
 assert(fcuda_alloc_reservation_id(&other,sizeof(other))==0);
 fcuda_alloc_ledger_reset_peak();assert(fcuda_alloc_reservation_id(&value,sizeof(value))==first);
 // Re-track and same-class re-tag deliberately invalidate older reservations.
 fak_alloc_track(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(value));
 uint64_t second=fcuda_alloc_reservation_id(&value,sizeof(value));assert(second>first);
 fak_alloc_retag(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,site);
 uint64_t third=fcuda_alloc_reservation_id(&value,sizeof(value));assert(third>second);
 // Both ends of idle-pool ownership transfer are identity boundaries.
 fak_alloc_retag(&value,FAK_ALLOC_IDLE_POOL,site);assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0);
 fak_alloc_retag(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,site);
 uint64_t fourth=fcuda_alloc_reservation_id(&value,sizeof(value));assert(fourth>third);
 assert(fak_alloc_untrack(&value));assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0);
 fak_alloc_track(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(value));
 uint64_t reused=fcuda_alloc_reservation_id(&value,sizeof(value));assert(reused>fourth); // exact same-address/size ABA
 fak_alloc_track(&other,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_MANAGED,site,sizeof(other));
 uint64_t managed_id=fcuda_alloc_reservation_id(&other,sizeof(other));assert(managed_id>reused);
 fak_alloc_track(&host,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_PINNED_HOST,site,sizeof(host));
 assert(fcuda_alloc_reservation_id(&host,sizeof(host))==0);
 // Independent owners concurrently consume the one locked identity sequence.
 std::vector<std::vector<uint64_t>> ids(4);
 std::vector<std::thread> threads;
 for(int thread=0;thread<4;++thread)threads.emplace_back([&,thread](){
  int cell=0;for(int i=0;i<64;++i){
   fak_alloc_track(&cell,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(cell));
   uint64_t a=fcuda_alloc_reservation_id(&cell,sizeof(cell));assert(a!=0);ids[thread].push_back(a);
   fak_alloc_retag(&cell,FAK_ALLOC_ACTIVATIONS_SCRATCH,site);
   uint64_t b=fcuda_alloc_reservation_id(&cell,sizeof(cell));assert(b>a);ids[thread].push_back(b);
   assert(fak_alloc_untrack(&cell));
  }
 });
 for(auto &thread:threads)thread.join();
 std::vector<uint64_t> all;for(auto &list:ids)all.insert(all.end(),list.begin(),list.end());
 std::sort(all.begin(),all.end());assert(all.size()==512&&std::adjacent_find(all.begin(),all.end())==all.end());
 // Only this fake harness moves the counter to its boundary. No production reset exists.
 {fak_alloc_tags_guard guard;g_alloc_reservation_next=~uint64_t(0)-1;}
 fak_alloc_track(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(value));
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==~uint64_t(0));
 fcuda_alloc_ledger_reset_peak();assert(fcuda_alloc_reservation_id(&value,sizeof(value))==~uint64_t(0));
 assert(fak_alloc_untrack(&value));
 fak_alloc_track(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,site,sizeof(value));
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0&&g_alloc_reservation_next==~uint64_t(0));
 assert(fcuda_alloc_reservation_id(&other,sizeof(other))==managed_id); // older live identity remains valid
 fak_alloc_retag(&other,FAK_ALLOC_ACTIVATIONS_SCRATCH,site);
 assert(fcuda_alloc_reservation_id(&other,sizeof(other))==0&&g_alloc_reservation_next==~uint64_t(0));
 assert(fak_alloc_untrack(&other));assert(fak_alloc_untrack(&host));
#else
 (void)other;(void)host;
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0);
 fak_alloc_retag(&value,FAK_ALLOC_IDLE_POOL,site);
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0);
 fak_alloc_retag(&value,FAK_ALLOC_ACTIVATIONS_SCRATCH,site);
 assert(fcuda_alloc_reservation_id(&value,sizeof(value))==0);
#endif
 assert(fak_alloc_untrack(&value));
 assert(g_alloc_tags.empty());
 for(auto &cell:g_alloc_cells){assert(cell.live.load()==0);assert(cell.live_count.load()==0);}
 return 0;
}
`
