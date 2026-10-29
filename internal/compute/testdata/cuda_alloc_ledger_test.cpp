// Fake-allocator harness for cuda/alloc_ledger.h. Host malloc stands in for
// cudaMalloc; the Go test parses each "read" line and asserts the contract.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "alloc_ledger.h"

static void print_read(int label) {
  int64_t cells[FAK_ALLOC_CELLS * 5] = {};
  int64_t untracked = -1;
  assert(fcuda_alloc_ledger_read(cells, FAK_ALLOC_CELLS, &untracked) == FAK_ALLOC_CELLS);
  std::printf("read %d %lld", label, (long long)untracked);
  for (int64_t v : cells) std::printf(" %lld", (long long)v);
  std::printf("\n");
}

int main() {
  int w = fak_alloc_site_intern("weights-site");
  assert(w > 0 && fak_alloc_site_intern("weights-site") == w);
  int kv = fak_alloc_site_intern("kv-site");
  assert(kv > 0 && kv != w);

  void *a = std::malloc(1000), *b = std::malloc(500), *c = std::malloc(300);
  void *d = std::malloc(64), *e = std::malloc(32), *f = std::malloc(200);
  fak_alloc_track(a, FAK_ALLOC_WEIGHTS, FAK_ALLOC_DEVICE_LOCAL, w, 1000);
  fak_alloc_track(b, FAK_ALLOC_KV_CACHE, FAK_ALLOC_DEVICE_LOCAL, kv, 500);
  fak_alloc_track(c, FAK_ALLOC_WEIGHTS, FAK_ALLOC_MANAGED, w, 300);
  fak_alloc_track(d, -1, FAK_ALLOC_DEVICE_LOCAL, -7, 64); // untagged -> other / overflow site
  fak_alloc_track(e, 99, FAK_ALLOC_DEVICE_LOCAL, 9999, 32);
  print_read(1);

  fak_alloc_retag(b, FAK_ALLOC_IDLE_POOL, FAK_ALLOC_SITE("device-pool")); // pool return
  assert(fak_alloc_untrack(a));
  int stray = 0;
  assert(!fak_alloc_untrack(&stray));
  print_read(2);

  fak_alloc_track(f, FAK_ALLOC_WEIGHTS, FAK_ALLOC_DEVICE_LOCAL, w, 200);
  fak_alloc_track(f, FAK_ALLOC_WEIGHTS, FAK_ALLOC_DEVICE_LOCAL, w, 200); // re-track never double-counts
  print_read(3);

  // Concurrent alloc/free pairs conserve the sums.
  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([t]() {
      int site = fak_alloc_site_intern(t % 2 ? "thread-odd" : "thread-even");
      for (int i = 0; i < 2000; ++i) {
        void *p = std::malloc(16);
        fak_alloc_track(p, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL, site, 16);
        bool ok = fak_alloc_untrack(p);
        assert(ok);
        (void)ok;
        std::free(p);
      }
    });
  }
  for (auto &t : workers) t.join();
  print_read(4);

  fak_alloc_untrack(b);
  fak_alloc_untrack(c);
  fak_alloc_untrack(d);
  fak_alloc_untrack(e);
  fak_alloc_untrack(f);
  print_read(5);
  for (void *p : {a, b, c, d, e, f}) std::free(p);
  return 0;
}
