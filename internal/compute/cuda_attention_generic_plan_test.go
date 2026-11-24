package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestCUDAAttentionGenericPlan compiles the real host planner header for the generic
// tiled attention and checks the env off-switch, admission bounds, slot choice, shared
// memory sizing and the split-KV plan invariants the kernel's partial merge relies on.
// fak-test:runtime medium est=2s lane=default
func TestCUDAAttentionGenericPlan(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	header, err := filepath.Abs(filepath.Join("cuda", "attention_generic_plan.h"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(header); err != nil {
		t.Fatal(err)
	}
	program := "#include \"" + filepath.ToSlash(header) + "\"\n" + attnGenericPlanMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "plan.cpp"), filepath.Join(dir, "plan")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile planner: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("planner contract: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "ok") {
		t.Fatalf("planner contract printed %q, want ok", out)
	}
}

const attnGenericPlanMain = `
#include <cassert>
#include <cstdio>
static void checkChunks(int n) {
  int chunk = attn_generic_chunk(n), splits = attn_generic_splits(n);
  // Tile-aligned chunks keep decode chunk blocks and panel fold points on the same
  // tile grid; every chunk owns at least one key and the last ends exactly at n.
  assert(chunk >= ATTN_GENERIC_CHUNK && chunk % ATTN_GENERIC_TILE == 0);
  assert(splits >= 1 && splits <= ATTN_GENERIC_MAX_SPLITS);
  assert((long long)(splits - 1) * chunk < n && (long long)splits * chunk >= n);
  if (n <= ATTN_GENERIC_CHUNK * ATTN_GENERIC_MAX_SPLITS) assert(chunk == ATTN_GENERIC_CHUNK);
}
int main() {
  assert(attn_generic_mode_legacy("legacy"));
  assert(!attn_generic_mode_legacy(nullptr));
  assert(!attn_generic_mode_legacy(""));
  assert(!attn_generic_mode_legacy("tiled"));
  assert(!attn_generic_mode_legacy("LEGACY"));

  assert(attn_generic_admits(1, 0, 28, 4, 128));
  assert(attn_generic_admits(1, 4096, 32, 8, 64));
  assert(attn_generic_admits(919, 0, 8, 8, 256));
  assert(attn_generic_admits(3, 5, 28, 4, 80));
  assert(!attn_generic_admits(1, 0, 28, 4, 257));
  assert(!attn_generic_admits(1, 0, 28, 4, 0));
  assert(!attn_generic_admits(1, 0, 28, 3, 128));
  assert(!attn_generic_admits(0, 5, 28, 4, 128));
  assert(!attn_generic_admits(1, -1, 28, 4, 128));
  assert(!attn_generic_admits(0x7fffffff, 0, 2, 1, 64));
  assert(!attn_generic_admits(2, 0x7fffffff - 1, 2, 1, 64));

  assert(attn_generic_slots(1) == 1 && attn_generic_slots(4) == 1);
  assert(attn_generic_slots(5) == 2 && attn_generic_slots(7) == 2 && attn_generic_slots(8) == 2);
  assert(attn_generic_slots(9) == 4 && attn_generic_slots(1 << 20) == 4);

  // hd128 at the widest block fits the default 48 KiB; hd256 needs the opt-in.
  assert(attn_generic_smem_bytes(128, 16) <= ATTN_GENERIC_DEFAULT_SMEM);
  assert(attn_generic_smem_bytes(64, 16) <= ATTN_GENERIC_DEFAULT_SMEM);
  assert(attn_generic_smem_bytes(256, 16) > ATTN_GENERIC_DEFAULT_SMEM);
  assert(attn_generic_smem_bytes(256, 16) <= 99 * 1024);

  // Short windows run as one chunk (decode writes the output directly).
  assert(attn_generic_splits(1) == 1 && attn_generic_splits(64) == 1);
  assert(attn_generic_splits(65) == 2 && attn_generic_splits(919) == 15);
  assert(attn_generic_splits(4097) == 65);
  assert(attn_generic_splits(1 << 20) == ATTN_GENERIC_MAX_SPLITS);
  int ns[] = {1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 919, 1000, 4097, 8191, 8192, 8193, 8200, 9000, 32768, 131071, 1 << 24, 0x7fffffff};
  for (int n : ns) checkChunks(n);
  for (int n = 1; n < 20000; n += 7) checkChunks(n);
  puts("ok");
  return 0;
}
`
