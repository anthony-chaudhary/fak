#pragma once

// Tagged device-allocation ledger. Every CUDA allocation path records
// {class, kind, bytes, site}; every free decrements the same cell. Counters are
// lock-free atomics, so a snapshot never blocks an allocation. The pointer tag
// table is serialized by the Go-side cudaMu like g_live; its spinlock only
// guards the rare unserialized device/handoff paths. Standalone C++17 so the
// fake-allocator harness compiles it without CUDA.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

// Class and kind ordinals are an ABI shared with internal/compute/cuda_alloc_ledger.go.
enum fak_alloc_class : int {
  FAK_ALLOC_WEIGHTS = 0,
  FAK_ALLOC_WEIGHTS_DEQUANT_SHADOW = 1,
  FAK_ALLOC_Q4_COEFF_CACHE = 2,
  FAK_ALLOC_KV_CACHE = 3,
  FAK_ALLOC_RECURRENT_STATE = 4,
  FAK_ALLOC_ACTIVATIONS_SCRATCH = 5,
  FAK_ALLOC_PREFILL_SCRATCH = 6,
  FAK_ALLOC_ATTENTION_SCRATCH = 7,
  FAK_ALLOC_SAMPLING = 8,
  FAK_ALLOC_WORKSPACE_CUBLAS = 9,
  FAK_ALLOC_GRAPH_POOL = 10,
  FAK_ALLOC_IDLE_POOL = 11,
  FAK_ALLOC_OTHER = 12,
  FAK_ALLOC_CLASS_COUNT = 13,
};

enum fak_alloc_kind : int {
  FAK_ALLOC_DEVICE_LOCAL = 0,
  FAK_ALLOC_MANAGED = 1,
  FAK_ALLOC_PINNED_HOST = 2,
  FAK_ALLOC_KIND_COUNT = 3,
};

static const int FAK_ALLOC_CELLS = FAK_ALLOC_CLASS_COUNT * FAK_ALLOC_KIND_COUNT;
static const int FAK_ALLOC_SITE_CAP = 512;
static const int FAK_ALLOC_SITE_NAME = 64;
static const int FAK_ALLOC_SITE_OVERFLOW = 0;

struct fak_alloc_cell {
  std::atomic<int64_t> live{0};
  std::atomic<int64_t> peak{0};
  std::atomic<int64_t> live_count{0};
  std::atomic<int64_t> allocs{0};
  std::atomic<int64_t> frees{0};
};

struct fak_alloc_site {
  char name[FAK_ALLOC_SITE_NAME];
  std::atomic<int> cls{FAK_ALLOC_OTHER};
  std::atomic<int64_t> live{0};
  std::atomic<int64_t> peak{0};
  std::atomic<int64_t> live_count{0};
  std::atomic<int64_t> allocs{0};
};

struct fak_alloc_tag {
  int cls;
  int kind;
  int site;
  size_t bytes;
};

static fak_alloc_cell g_alloc_cells[FAK_ALLOC_CELLS];
static fak_alloc_site g_alloc_sites[FAK_ALLOC_SITE_CAP];
static std::atomic<int> g_alloc_site_n{1}; // slot 0 is the overflow site
static std::atomic<int64_t> g_alloc_untracked_frees{0};
static std::unordered_map<const void *, fak_alloc_tag> g_alloc_tags;
static std::atomic_flag g_alloc_tags_lock = ATOMIC_FLAG_INIT;

struct fak_alloc_tags_guard {
  fak_alloc_tags_guard() {
    while (g_alloc_tags_lock.test_and_set(std::memory_order_acquire)) {
    }
  }
  ~fak_alloc_tags_guard() { g_alloc_tags_lock.clear(std::memory_order_release); }
  fak_alloc_tags_guard(const fak_alloc_tags_guard &) = delete;
  fak_alloc_tags_guard &operator=(const fak_alloc_tags_guard &) = delete;
};

static inline int fak_alloc_norm_class(int cls) {
  return (cls < 0 || cls >= FAK_ALLOC_CLASS_COUNT) ? FAK_ALLOC_OTHER : cls;
}
static inline int fak_alloc_norm_kind(int kind) {
  return (kind < 0 || kind >= FAK_ALLOC_KIND_COUNT) ? FAK_ALLOC_DEVICE_LOCAL : kind;
}
static inline int fak_alloc_norm_site(int site) {
  int n = g_alloc_site_n.load(std::memory_order_acquire);
  return (site <= 0 || site >= n) ? FAK_ALLOC_SITE_OVERFLOW : site;
}

static inline void fak_alloc_raise_peak(std::atomic<int64_t> &peak, int64_t now) {
  int64_t seen = peak.load(std::memory_order_relaxed);
  while (now > seen && !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
  }
}

static inline void fak_alloc_cell_add(int cls, int kind, int site, size_t bytes) {
  fak_alloc_cell &c = g_alloc_cells[cls * FAK_ALLOC_KIND_COUNT + kind];
  int64_t now = c.live.fetch_add((int64_t)bytes, std::memory_order_relaxed) + (int64_t)bytes;
  fak_alloc_raise_peak(c.peak, now);
  c.live_count.fetch_add(1, std::memory_order_relaxed);
  c.allocs.fetch_add(1, std::memory_order_relaxed);
  fak_alloc_site &s = g_alloc_sites[site];
  int64_t snow = s.live.fetch_add((int64_t)bytes, std::memory_order_relaxed) + (int64_t)bytes;
  fak_alloc_raise_peak(s.peak, snow);
  s.live_count.fetch_add(1, std::memory_order_relaxed);
  s.allocs.fetch_add(1, std::memory_order_relaxed);
}

static inline void fak_alloc_cell_sub(const fak_alloc_tag &t) {
  fak_alloc_cell &c = g_alloc_cells[t.cls * FAK_ALLOC_KIND_COUNT + t.kind];
  c.live.fetch_sub((int64_t)t.bytes, std::memory_order_relaxed);
  c.live_count.fetch_sub(1, std::memory_order_relaxed);
  c.frees.fetch_add(1, std::memory_order_relaxed);
  fak_alloc_site &s = g_alloc_sites[t.site];
  s.live.fetch_sub((int64_t)t.bytes, std::memory_order_relaxed);
  s.live_count.fetch_sub(1, std::memory_order_relaxed);
}

// fak_alloc_site_intern returns a stable id for name, or the overflow site when
// the table is full. Callers cache the id; a repeated name returns the same id.
static inline int fak_alloc_site_intern(const char *name) {
  if (!name || !*name) return FAK_ALLOC_SITE_OVERFLOW;
  fak_alloc_tags_guard guard;
  int n = g_alloc_site_n.load(std::memory_order_relaxed);
  for (int i = 1; i < n; ++i)
    if (strncmp(g_alloc_sites[i].name, name, FAK_ALLOC_SITE_NAME - 1) == 0) return i;
  if (n >= FAK_ALLOC_SITE_CAP) return FAK_ALLOC_SITE_OVERFLOW;
  strncpy(g_alloc_sites[n].name, name, FAK_ALLOC_SITE_NAME - 1);
  g_alloc_sites[n].name[FAK_ALLOC_SITE_NAME - 1] = 0;
  g_alloc_site_n.store(n + 1, std::memory_order_release);
  return n;
}

// fak_alloc_track records a fresh or reused allocation. A pointer already
// tracked is retagged, so a double track never double-counts its bytes.
static inline void fak_alloc_track(const void *p, int cls, int kind, int site, size_t bytes) {
  if (!p) return;
  fak_alloc_tag t{fak_alloc_norm_class(cls), fak_alloc_norm_kind(kind), fak_alloc_norm_site(site), bytes};
  fak_alloc_tags_guard guard;
  auto it = g_alloc_tags.find(p);
  if (it != g_alloc_tags.end()) {
    fak_alloc_cell_sub(it->second);
    it->second = t;
  } else {
    g_alloc_tags.emplace(p, t);
  }
  if (t.site == FAK_ALLOC_SITE_OVERFLOW && !g_alloc_sites[0].name[0])
    strncpy(g_alloc_sites[0].name, "overflow", FAK_ALLOC_SITE_NAME - 1);
  g_alloc_sites[t.site].cls.store(t.cls, std::memory_order_relaxed);
  fak_alloc_cell_add(t.cls, t.kind, t.site, bytes);
}

// fak_alloc_untrack decrements the pointer's cell. It returns false for an
// unknown pointer and counts it, so a missed track is visible, never silent.
static inline bool fak_alloc_untrack(const void *p) {
  if (!p) return false;
  fak_alloc_tags_guard guard;
  auto it = g_alloc_tags.find(p);
  if (it == g_alloc_tags.end()) {
    g_alloc_untracked_frees.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  fak_alloc_cell_sub(it->second);
  g_alloc_tags.erase(it);
  return true;
}

// fak_alloc_retag moves a tracked pointer to another class and site (for
// example a freed buffer entering the idle pool) without a free/alloc pair.
static inline void fak_alloc_retag(const void *p, int cls, int site) {
  if (!p) return;
  fak_alloc_tags_guard guard;
  auto it = g_alloc_tags.find(p);
  if (it == g_alloc_tags.end()) return;
  fak_alloc_cell_sub(it->second);
  it->second.cls = fak_alloc_norm_class(cls);
  it->second.site = fak_alloc_norm_site(site);
  g_alloc_sites[it->second.site].cls.store(it->second.cls, std::memory_order_relaxed);
  fak_alloc_cell_add(it->second.cls, it->second.kind, it->second.site, it->second.bytes);
}

// FAK_ALLOC_SITE caches one interned id per call site for C-owned literals.
#define FAK_ALLOC_SITE(literal) ([]() { static const int id = fak_alloc_site_intern(literal); return id; }())

extern "C" int fcuda_alloc_site_intern(const char *name) { return fak_alloc_site_intern(name); }

// fcuda_alloc_ledger_read copies FAK_ALLOC_CELLS cells, each {live, peak,
// live_count, allocs, frees}, into out (5*cells int64s), class-major. It
// returns the cells written and the count of frees of untracked pointers.
extern "C" int fcuda_alloc_ledger_read(int64_t *out, int cells, int64_t *untracked_frees) {
  if (!out || cells < FAK_ALLOC_CELLS) return 0;
  for (int i = 0; i < FAK_ALLOC_CELLS; ++i) {
    out[i * 5 + 0] = g_alloc_cells[i].live.load(std::memory_order_relaxed);
    out[i * 5 + 1] = g_alloc_cells[i].peak.load(std::memory_order_relaxed);
    out[i * 5 + 2] = g_alloc_cells[i].live_count.load(std::memory_order_relaxed);
    out[i * 5 + 3] = g_alloc_cells[i].allocs.load(std::memory_order_relaxed);
    out[i * 5 + 4] = g_alloc_cells[i].frees.load(std::memory_order_relaxed);
  }
  if (untracked_frees) *untracked_frees = g_alloc_untracked_frees.load(std::memory_order_relaxed);
  return FAK_ALLOC_CELLS;
}

// fcuda_alloc_site_read copies site i's name and {class, live, peak,
// live_count, allocs}. It returns the number of sites, or -1 for a bad index.
extern "C" int fcuda_alloc_site_read(int i, char *name, int namelen, int64_t *out5) {
  int n = g_alloc_site_n.load(std::memory_order_acquire);
  if (i < 0 || i >= n) return -1;
  if (name && namelen > 0) {
    strncpy(name, g_alloc_sites[i].name, (size_t)namelen - 1);
    name[namelen - 1] = 0;
  }
  if (out5) {
    out5[0] = g_alloc_sites[i].cls.load(std::memory_order_relaxed);
    out5[1] = g_alloc_sites[i].live.load(std::memory_order_relaxed);
    out5[2] = g_alloc_sites[i].peak.load(std::memory_order_relaxed);
    out5[3] = g_alloc_sites[i].live_count.load(std::memory_order_relaxed);
    out5[4] = g_alloc_sites[i].allocs.load(std::memory_order_relaxed);
  }
  return n;
}

// fcuda_alloc_ledger_reset_peak restarts every peak at its current live bytes.
extern "C" void fcuda_alloc_ledger_reset_peak(void) {
  for (int i = 0; i < FAK_ALLOC_CELLS; ++i)
    g_alloc_cells[i].peak.store(g_alloc_cells[i].live.load(std::memory_order_relaxed), std::memory_order_relaxed);
  int n = g_alloc_site_n.load(std::memory_order_acquire);
  for (int i = 0; i < n; ++i)
    g_alloc_sites[i].peak.store(g_alloc_sites[i].live.load(std::memory_order_relaxed), std::memory_order_relaxed);
}
