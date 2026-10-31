package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var cudaTokenomosDeviceQualification = flag.Bool("tokenomos-staging-device-qualification", false,
	"require an actual CUDA gather with the recorded Tokenomos staging profile; never CPU fallback")

// This explicit required-mode witness never skips missing CUDA/profile/staging.
// The ordinary test run does not request device work. Its synthetic F32 table is
// a byte-copy witness, not whole-model quality, capture/fault injection or speed.
// fak-test:justify why=qualification when=changed:internal/compute/cuda/tokenomos_staging.h
// fak-test:runtime slow est=3s lane=cuda
func TestCUDATokenomosStagingDeviceQualification(t *testing.T) {
	if !*cudaTokenomosDeviceQualification {
		t.Skip("device qualification not selected; use -tokenomos-staging-device-qualification explicitly")
	}
	backend, ok := Lookup("cuda")
	if !ok {
		t.Fatal("required CUDA backend absent; no CPU fallback")
	}
	gather, ok := backend.(V41EmbeddingGatherBackend)
	if !ok || !gather.V41EmbeddingGatherAvailable() {
		t.Fatal("required native gather unavailable; compile the explicit V41 router Go profile on a supported device")
	}
	ledger, ok := backend.(DeviceAllocLedgerReader)
	if !ok {
		t.Fatal("required native allocation ledger unavailable")
	}
	const vocab, hidden = 97, 64
	table := make([]float32, vocab*hidden)
	bytes := make([]byte, len(table)*4)
	for i := range table {
		bits := uint32(i)*2654435761 | 1
		table[i] = math.Float32frombits(bits)
		binary.LittleEndian.PutUint32(bytes[i*4:], bits)
	}
	t.Logf("backend=%s tier=%s synthetic_table_sha256=%x vocab=%d hidden=%d", backend.Name(), backend.Tier(), sha256.Sum256(bytes), vocab, hidden)
	resident := backend.Upload(NewF32(backend, []int{vocab, hidden}, table), F32)
	defer backend.Free(resident)
	type pendingGather struct {
		output Tensor
		ids    []int
	}
	var pending []pendingGather
	// Three equal lengths reuse the existing device ID pool. The next pair
	// covers exact arena capacity and the pre-staging synchronous oversize path.
	for round, count := range []int{5, 5, 5, 16384, 16385, 5} {
		ids := make([]int, count)
		for i := range ids {
			ids[i] = (i*17 + round*11) % vocab
		}
		want := append([]int(nil), ids...)
		output, err := gather.V41EmbeddingGather(resident, ids, hidden)
		if err != nil {
			t.Fatalf("required gather round=%d rows=%d: %v", round, count, err)
		}
		defer backend.Free(output)
		pending = append(pending, pendingGather{output: output, ids: want})
		for i := range ids {
			ids[i] = -1 // borrowed caller storage cannot be an asynchronous source
		}
	}
	runtime.GC()
	for round, item := range pending {
		got := backend.Read(item.output)
		if len(got) != len(item.ids)*hidden {
			t.Fatalf("round=%d wrong output extent %d", round, len(got))
		}
		for row, id := range item.ids {
			for column := range hidden {
				if math.Float32bits(got[row*hidden+column]) != math.Float32bits(table[id*hidden+column]) {
					t.Fatalf("round=%d row=%d column=%d bit-exact gather mismatch", round, row, column)
				}
			}
		}
	}
	// A correctly functioning default synchronous library MUST fail this gate:
	// build tags or source marker presence alone cannot stand in for selection.
	found := false
	for _, site := range ledger.DeviceAllocLedger().Sites {
		if site.Site == "tokenomos-id-staging" {
			found = true
			if site.LiveBytes != 128*1024 || site.LiveCount != 2 || site.Allocs != 2 {
				t.Fatalf("process-owned staging bound changed: %+v", site)
			}
			t.Logf("selected_native_staging=%+v", site)
		}
	}
	if !found {
		t.Fatal("recorded Tokenomos staging native macro was not active; default synchronous library is not a qualification witness")
	}
}

// Compile the exact production owner and allocation ledger against a delayed
// fake CUDA driver. This is host lifetime/failure simulation, not CUDA execution,
// model quality, graph qualification or a performance result.
// fak-test:runtime medium est=3s lane=default
func TestCUDATokenomosStagingOwnership(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("Tokenomos staging ownership witness requires a host C++17 compiler:", err)
	}
	dir := t.TempDir()
	for _, name := range []string{"tokenomos_staging.h", "alloc_ledger.h"} {
		body, err := os.ReadFile(filepath.Join("cuda", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), body, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("production %s sha256=%x", name, sha256.Sum256(body))
	}
	program := cudaTokenomosStagingHarness + cudaTokenomosStagingCases
	source := filepath.Join(dir, "tokenomos-staging.cpp")
	if err := os.WriteFile(source, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "tokenomos-staging")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Os", "-Wall", "-Wextra", "-Werror", source, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile exact staging owner: %v\n%s", err, out)
	}
	artifact, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("fake driver program sha256=%x binary sha256=%x; CUDA/model/performance UNRUN", sha256.Sum256([]byte(program)), sha256.Sum256(artifact))
	for _, name := range []string{
		"rotation", "cancel", "batch-bounds", "capture", "oversize",
		"alloc-first", "alloc-second", "event-first", "event-second",
		"copy", "record", "wait", "launch", "drain", "destroy", "free",
		"device-query", "capture-query", "prior-error", "sync-copy",
	} {
		t.Run(name, func(t *testing.T) {
			// Each case is a fresh process, never a production poison-reset hook.
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if out, err := exec.CommandContext(ctx, binary, name).CombinedOutput(); err != nil {
				t.Fatalf("production owner scenario %s: %v\n%s", name, err, out)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATokenomosStagingCallerWiring(t *testing.T) {
	body, err := os.ReadFile("cuda/model_candidates.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(body)
	end := strings.Index(source, "extern \"C\" int fak_qwen35_pointer_is_device")
	if end < 0 {
		t.Fatal("native gather boundary missing")
	}
	source = source[:end]
	for _, marker := range []string{
		"#ifdef FAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION",
		"#include \"tokenomos_staging.h\"",
		"fak_tokenomos_staging_copy(dIDs, hIDs, idsBytes)",
		"if (staged != 0) return staged;",
		"cudaMemcpy(dIDs, hIDs, idsBytes, cudaMemcpyHostToDevice)",
		"fak_tokenomos_staging_error(launch)",
	} {
		if !strings.Contains(source, marker) {
			t.Fatalf("checked/default gather wiring lost %q", marker)
		}
	}
}

const cudaTokenomosStagingHarness = `
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using cudaError_t = int;
using cudaStream_t = void *;
struct fake_event { size_t marker = 0; };
using cudaEvent_t = fake_event *;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };
static constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorUnknown = 999;
static constexpr unsigned cudaHostAllocDefault = 0, cudaEventDisableTiming = 2;
static constexpr int cudaMemcpyHostToDevice = 1;
static void *g_blas = reinterpret_cast<void *>(uintptr_t(1));
static cudaStream_t g_stream = reinterpret_cast<void *>(uintptr_t(2));
static bool g_capture_open = false, peer_uncertain = false;
static int g_cuda_cleanup_status = 0, device_ordinal = 0;
static cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
static bool v41_candidates_stream_uncertain() { return peer_uncertain || g_cuda_cleanup_status != 0; }

enum api { Host, EventCreate, Copy, Record, Wait, Drain, Destroy, Free, Device, Capture, Peek, SyncCopy, APICount };
static int calls[APICount] = {};
static int primary_api = -1, primary_nth = 1, secondary_api = -1;
static constexpr int primary_error = 73, secondary_error = 91;
static cudaError_t inject(api which) {
  ++calls[which];
  if (which == primary_api && calls[which] == primary_nth) return primary_error;
  if (which == secondary_api) return secondary_error;
  return cudaSuccess;
}
struct operation { void *to; const void *from; size_t bytes; };
struct block { void *pointer; size_t bytes; bool freed; };
static std::vector<operation> queued;
static std::vector<block> blocks;
static size_t completed = 0;
static void flush(size_t marker) {
  assert(marker <= queued.size());
  while (completed < marker) {
    const operation &op = queued[completed++];
    std::memcpy(op.to, op.from, op.bytes);
  }
}
static size_t pinned_live() {
  size_t bytes = 0;
  for (const block &b : blocks) if (!b.freed) bytes += b.bytes;
  return bytes;
}
static cudaError_t cudaHostAlloc(void **out, size_t bytes, unsigned flags) {
  assert(flags == cudaHostAllocDefault);
  *out = nullptr;
  if (int error = inject(Host)) return error;
  *out = std::malloc(bytes); assert(*out);
  blocks.push_back({*out, bytes, false});
  return cudaSuccess;
}
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out, unsigned flags) {
  assert(flags == cudaEventDisableTiming);
  *out = nullptr;
  if (int error = inject(EventCreate)) return error;
  *out = new fake_event;
  return cudaSuccess;
}
static cudaError_t cudaMemcpyAsync(void *to, const void *from, size_t bytes, int kind, cudaStream_t stream) {
  assert(stream == g_stream && kind == cudaMemcpyHostToDevice);
  // Even an error may leave a queued operation: cleanup must prove its drain.
  queued.push_back({to, from, bytes});
  return inject(Copy);
}
static cudaError_t cudaMemcpy(void *to, const void *from, size_t bytes, int kind) {
  assert(kind == cudaMemcpyHostToDevice);
  if (int error = inject(SyncCopy)) return error;
  flush(queued.size());
  std::memcpy(to, from, bytes);
  return cudaSuccess;
}
static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  assert(event && stream == g_stream);
  if (int error = inject(Record)) return error;
  event->marker = queued.size();
  return cudaSuccess;
}
static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  assert(event);
  if (int error = inject(Wait)) return error;
  flush(event->marker);
  return cudaSuccess;
}
static cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(stream == g_stream);
  if (int error = inject(Drain)) return error;
  flush(queued.size());
  return cudaSuccess;
}
static cudaError_t cudaEventDestroy(cudaEvent_t event) {
  assert(event && completed >= event->marker);
  if (int error = inject(Destroy)) return error;
  delete event;
  return cudaSuccess;
}
static cudaError_t cudaFreeHost(void *pointer) {
  if (int error = inject(Free)) return error;
  for (block &b : blocks) if (b.pointer == pointer && !b.freed) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer), end = begin + b.bytes;
    for (size_t i = completed; i < queued.size(); ++i) {
      uintptr_t source = reinterpret_cast<uintptr_t>(queued[i].from);
      assert(source < begin || source >= end); // no outstanding source loan
    }
    b.freed = true;
    std::free(pointer);
    return cudaSuccess;
  }
  assert(false && "duplicate/unknown host release");
  return cudaErrorInvalidValue;
}
static cudaError_t cudaGetDevice(int *out) {
  if (int error = inject(Device)) return error;
  *out = device_ordinal;
  return cudaSuccess;
}
static cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *out) {
  assert(stream == g_stream);
  if (int error = inject(Capture)) return error;
  *out = capture_status;
  return cudaSuccess;
}
static cudaError_t cudaPeekAtLastError() { return inject(Peek); }

#include "tokenomos_staging.h"

static int64_t ledger_live() {
  return g_alloc_cells[FAK_ALLOC_ACTIVATIONS_SCRATCH * FAK_ALLOC_KIND_COUNT + FAK_ALLOC_PINNED_HOST].live.load();
}
static void consumer(int *device, int *output, size_t bytes) {
  queued.push_back({output, device, bytes}); // ordered existing gather consumer
}
static void assert_sticky(int *device, const int *source) {
  const int allocations = calls[Host], events = calls[EventCreate], drains = calls[Drain];
  const size_t size = queued.size();
  assert(fak_tokenomos_staging_copy(device, source, sizeof(int)) == 20000 + primary_error);
  assert(calls[Host] == allocations && calls[EventCreate] == events && calls[Drain] == drains);
  assert(queued.size() == size); // no new allocation, enqueue or cleanup retry
}
`

const cudaTokenomosStagingCases = `
int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  constexpr size_t capacity = fak_tokenomos_staging_pool::capacity;
  int ids[4] = {1, 4, 1, 7}, device[4] = {}, output[4] = {};
  if (mode == "rotation") {
    int second[4] = {9, 8, 9, 6}, third[4] = {3, 3, 2, 5};
    int out_second[4] = {}, out_third[4] = {};
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
    consumer(device, output, sizeof(ids));
    std::memset(ids, 0, sizeof(ids));
    assert(fak_tokenomos_staging_copy(device, second, sizeof(second)) == 0);
    consumer(device, out_second, sizeof(second));
    std::memset(second, 0, sizeof(second));
    assert(fak_tokenomos_staging_copy(device, third, sizeof(third)) == 0);
    consumer(device, out_third, sizeof(third));
    std::memset(third, 0, sizeof(third));
    assert(calls[Wait] == 1 && calls[Drain] == 0 && completed == 1);
    assert(output[0] == 0); // the source event need not fence its later consumer
    assert(cudaStreamSynchronize(g_stream) == 0);
    const int want[4] = {1, 4, 1, 7}, want_second[4] = {9, 8, 9, 6}, want_third[4] = {3, 3, 2, 5};
    assert(std::memcmp(output, want, sizeof(want)) == 0);
    assert(std::memcmp(out_second, want_second, sizeof(want_second)) == 0);
    assert(std::memcmp(out_third, want_third, sizeof(want_third)) == 0);
    assert(calls[Host] == 2 && calls[EventCreate] == 2 && calls[Free] == 0);
    assert(pinned_live() == 2 * capacity && ledger_live() == int64_t(2 * capacity));
  } else if (mode == "cancel") {
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
    consumer(device, output, sizeof(ids));
    std::memset(ids, 0, sizeof(ids));
    // Caller cancellation means no next begin; the sealed owner still lives.
    assert(g_tokenomos_staging.arenas[0].pending && !g_tokenomos_staging.open);
    assert(calls[Free] == 0 && cudaStreamSynchronize(g_stream) == 0);
    assert(output[0] == 1 && output[1] == 4 && output[2] == 1 && output[3] == 7);
    assert(pinned_live() == 2 * capacity && !g_tokenomos_staging.terminal);
  } else if (mode == "batch-bounds") {
    auto &pool = g_tokenomos_staging;
    assert(pool.begin(g_stream) == 0 && pool.begin(g_stream) == 0);
    assert(calls[Record] == 0); // empty batches do not fabricate fences
    unsigned char a[3] = {1, 2, 3}, b[65] = {}, out_a[3] = {}, out_b[65] = {};
    b[64] = 11;
    assert(pool.put(out_a, a, sizeof(a)) == 0 && pool.at == 64);
    assert(pool.put(out_b, b, sizeof(b)) == 0 && pool.at == 192);
    assert(static_cast<const unsigned char *>(queued[1].from) - static_cast<const unsigned char *>(queued[0].from) == 64);
    std::vector<unsigned char> tail(capacity - pool.at, 19), out_tail(tail.size());
    int copies = calls[Copy];
    assert(pool.put(out_a, a, capacity) == cudaErrorInvalidValue);
    assert(pool.at == 192 && calls[Copy] == copies);
    assert(pool.put(out_tail.data(), tail.data(), tail.size()) == 0 && pool.at == capacity);
    assert(pool.put(out_a, a, 1) == cudaErrorInvalidValue);
    assert(pool.put(out_a, a, 0) == cudaErrorInvalidValue);
    assert(pool.begin(g_stream) == 0 && calls[Record] == 1); // donor implicit seal
    assert(pool.begin(g_stream) == 0 && calls[Wait] == 1);
    assert(std::memcmp(out_a, a, sizeof(a)) == 0 && out_b[64] == 11 && out_tail == tail);
    assert(!pool.terminal && calls[Host] == 2);
  } else if (mode == "capture") {
    g_capture_open = true;
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == -3);
    g_capture_open = false; capture_status = cudaStreamCaptureStatusActive;
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == -3);
    capture_status = cudaStreamCaptureStatusNone; device_ordinal = 1;
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == -3);
    device_ordinal = 0; peer_uncertain = true;
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == -3);
    peer_uncertain = false;
    assert(calls[Host] == 0 && calls[Copy] == 0 && calls[SyncCopy] == 0 && calls[Drain] == 0);
    assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
  } else if (mode == "oversize") {
    std::vector<unsigned char> source(capacity + 1, 31), destination(source.size());
    assert(fak_tokenomos_staging_copy(destination.data(), source.data(), source.size()) == 0);
    assert(source == destination && calls[SyncCopy] == 1 && calls[Host] == 0 && calls[Copy] == 0);
    assert(fak_tokenomos_staging_copy(destination.data(), source.data(), capacity) == 0);
    assert(calls[Host] == 2 && calls[Copy] == 1);
    assert(cudaStreamSynchronize(g_stream) == 0);
  } else {
    if (mode == "alloc-first" || mode == "alloc-second") primary_api = Host;
    else if (mode == "event-first" || mode == "event-second") primary_api = EventCreate;
    else if (mode == "record") primary_api = Record;
    else if (mode == "device-query") primary_api = Device;
    else if (mode == "capture-query") primary_api = Capture;
    else if (mode == "prior-error") primary_api = Peek;
    else if (mode == "sync-copy") primary_api = SyncCopy;
    else if (mode == "wait" || mode == "launch") {} // arm after successful issue below
    else {
      assert(mode == "copy" || mode == "drain" || mode == "destroy" || mode == "free");
      primary_api = Copy;
      if (mode == "drain") secondary_api = Drain;
      if (mode == "destroy") secondary_api = Destroy;
      if (mode == "free") secondary_api = Free;
    }
    if (mode == "alloc-second" || mode == "event-second") primary_nth = 2;
    int result = 0;
    if (mode == "wait") {
      assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
      assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
      primary_api = Wait;
      result = fak_tokenomos_staging_copy(device, ids, sizeof(ids));
    } else if (mode == "launch") {
      assert(fak_tokenomos_staging_copy(device, ids, sizeof(ids)) == 0);
      result = fak_tokenomos_staging_error(primary_error);
    } else if (mode == "sync-copy") {
      std::vector<unsigned char> source(capacity + 1, 31), destination(source.size());
      result = fak_tokenomos_staging_copy(destination.data(), source.data(), source.size());
    } else {
      result = fak_tokenomos_staging_copy(device, ids, sizeof(ids));
    }
    assert(result == 20000 + primary_error && g_tokenomos_staging.terminal);
    assert(g_tokenomos_staging.first_error == primary_error);
    if (secondary_api >= 0) {
      assert(g_tokenomos_staging.retained && g_cuda_cleanup_status == secondary_error);
      assert(g_tokenomos_staging.cleanup_error == secondary_error);
      assert(g_tokenomos_staging.drained == (secondary_api != Drain));
      assert(pinned_live() == 2 * capacity && ledger_live() == int64_t(2 * capacity));
      if (secondary_api == Drain) assert(calls[Free] == 0 && calls[Destroy] == 0 && completed == 0);
    } else {
      assert(g_tokenomos_staging.drained && !g_tokenomos_staging.retained && g_cuda_cleanup_status == 0);
      assert(pinned_live() == 0 && ledger_live() == 0 && completed == queued.size());
    }
    assert_sticky(device, ids);
  }
  assert(g_alloc_untracked_frees.load() == 0);
  std::cout << "PASS " << mode << " (fake driver; CUDA/model/performance UNRUN)\n";
  return 0;
}
`
