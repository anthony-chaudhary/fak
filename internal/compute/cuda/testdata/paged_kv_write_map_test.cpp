// Host qualification of the exact header used by the CUDA kernel, not GPU execution.
// Run: c++ -std=c++14 -Wall -Wextra -Werror paged_kv_write_map_test.cpp -o /tmp/paged-map-test
//      /tmp/paged-map-test
// fak-test:runtime fast est=10ms lane=optin
#include "../paged_kv_write_map.h"
#include <assert.h>
#include <stdio.h>
#include <vector>
#ifdef NDEBUG
#error "address regression requires active assertions"
#endif

int main() {
  using namespace fak_vllm_pa;
  PagedKVWriteMap pages = {};
  pages.first_offset = 14;
  const int ids[] = {7, 2, 11, 4};
  for (int p = 0; p < 4; ++p) pages.blocks[p] = ids[p];
  for (int hd : {32, 64, 120, 256}) {
    const int heads = 3, rows = 35, capacity = 12;
    assert(paged_kv_write_map_valid(ids, 4, capacity, 14, rows, heads, hd, 16));
    const int page_floats = heads * hd * 16;
    std::vector<int> keys(capacity * page_floats, -1), values(keys.size(), -1);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < heads * hd; ++i) {
        int64_t ki, vi;
        paged_kv_write_offsets(pages, row, i, heads, hd, 16, &ki, &vi);
        assert(ki >= 0 && size_t(ki) < keys.size());
        assert(vi >= 0 && size_t(vi) < values.size());
        assert(keys[ki] == -1 && values[vi] == -1);
        keys[ki] = values[vi] = row * heads * hd + i;
      }
    }
    // Independent layout walk: inspect every physical slot, including all holes.
    for (int page = 0; page < capacity; ++page) {
      for (int offset = 0; offset < 16; ++offset) {
        int row = -1;
        for (int p = 0; p < 4; ++p) {
          int r = p * 16 + offset - 14;
          if (ids[p] == page && r >= 0 && r < rows) row = r;
        }
        for (int h = 0; h < heads; ++h) {
          for (int d = 0; d < hd; ++d) {
            int want = row < 0 ? -1 : row * heads * hd + h * hd + d;
            int ki = (((page * heads + h) * (hd / 4) + d / 4) * 16 + offset) * 4 + d % 4;
            int vi = ((page * heads + h) * hd + d) * 16 + offset;
            assert(keys[ki] == want && values[vi] == want);
          }
        }
      }
    }
  }
  int full[65];
  for (int i = 0; i < 65; ++i) full[i] = 100 - i;
  assert(paged_kv_write_map_valid(full, 64, 101, 15, 1009, 2, 128, 16));
  assert(!paged_kv_write_map_valid(full, 65, 101, 0, 1040, 2, 128, 16));
  assert(!paged_kv_write_map_valid(full, 64, 101, 15, 1010, 2, 128, 16));
  assert(!paged_kv_write_map_valid(full, 64, 101, 0, 16, 2, 128, 16));
  assert(!paged_kv_write_map_valid(nullptr, 1, 1, 0, 1, 1, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 100, 0, 1, 1, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, -1, 1, 1, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, 16, 1, 1, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 0, 1, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 1, 0, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 1, 1, 30, 16));
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 1, 1, 32, 32));
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 1, INT_MAX, 32, 16));
  assert(!paged_kv_write_map_valid(full, 1, INT_MAX, 0, 1, 1, INT_MAX - 3, 16));
  full[0] = -1;
  assert(!paged_kv_write_map_valid(full, 1, 101, 0, 1, 1, 32, 16));
  full[0] = full[1];
  assert(!paged_kv_write_map_valid(full, 2, 101, 0, 17, 1, 32, 16));
  puts("PASS: actual paged writer admission and K/V address mapping (host only)");
}
