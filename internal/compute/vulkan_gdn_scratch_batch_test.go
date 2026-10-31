//go:build linux

package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=2s lane=optin
// fak-test:justify Compiles one extracted native allocation helper with synthetic batch hooks.
// This is a CPU control-flow witness, not a Vulkan driver, full-shim build,
// numerical parity or physical submission-count witness.
func TestVulkanGDNScratchGrowthRestoresBatch(t *testing.T) {
	if os.Getenv("FAK_TEST_VULKAN_GDN_SCRATCH_BATCH") != "1" {
		t.Skip("set FAK_TEST_VULKAN_GDN_SCRATCH_BATCH=1 for the C++ control-flow witness")
	}
	compiler, err := exec.LookPath("g++")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	const signature = "Buffer* gdnConvOutScratch(size_t bytes) {\n"
	if len(source) > 1<<20 || strings.Count(source, signature) != 1 {
		t.Fatal("expected one bounded production GDN scratch helper")
	}
	start := strings.Index(source, signature)
	end := strings.Index(source[start:], "\n}\n")
	if end < 0 {
		t.Fatal("unterminated production GDN scratch helper")
	}
	body := source[start : start+end+2]
	fixture := gdnScratchBatchPreamble + body + gdnScratchBatchMain
	tmp := t.TempDir()
	cpp, binary := filepath.Join(tmp, "gdn_scratch.cpp"), filepath.Join(tmp, "gdn_scratch")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile extracted production helper: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production GDN scratch control flow: %v\n%s", err, out)
	}
}

const gdnScratchBatchPreamble = `
#include <cstdio>
#include <cstdlib>
#include <string>
struct Buffer { size_t bytes; };
Buffer* g_gdn_conv_out = nullptr;
bool g_batching = false, refuseBegin = false;
size_t g_maxBufferBytes = 0;
constexpr int VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1, STORAGE_USAGE = 2;
int failures = 0;
std::string events;
size_t scratchCapacity(size_t bytes) { return (bytes + 63) & ~size_t(63); }
void batchFlush() {
    if (!g_batching) std::exit(90);
    events += 'F';
    g_batching = false;
}
void batchBegin() {
    if (g_batching) std::exit(91);
    events += 'B';
    // The real batchBegin owns sticky submission/pending-failure refusal.
    if (!refuseBegin) g_batching = true;
}
void destroyBuffer(Buffer* buffer) {
    if (g_batching) std::exit(92); // replacing referenced scratch requires a fence
    events += 'D';
    delete buffer;
}
void clearDescriptorBindingCache() { events += 'C'; }
Buffer* allocBuffer(size_t bytes, int properties, int usage) {
    if (properties != VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT || usage != STORAGE_USAGE) std::exit(93);
    events += 'A';
    if (failures > 0) { --failures; return nullptr; }
    return new Buffer{bytes};
}
void reset(bool batching, size_t capacity, int fail = 0) {
    delete g_gdn_conv_out;
    g_gdn_conv_out = capacity ? new Buffer{capacity} : nullptr;
    g_batching = batching;
    refuseBegin = false;
    g_maxBufferBytes = 0;
    failures = fail;
    events.clear();
}
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "GDN scratch contract failed at line %d: %s; events=%s\n", \
        __LINE__, #condition, events.c_str()); return 1; } } while (0)
`

const gdnScratchBatchMain = `
int main() {
    reset(true, 16);
    Buffer* grown = gdnConvOutScratch(80);
    CHECK(grown && grown->bytes == 128 && g_batching);
    CHECK(events == "FDCAB"); // flush, destroy, invalidate, allocate, resume

    reset(true, 16, 1);
    grown = gdnConvOutScratch(80);
    CHECK(grown && grown->bytes == 80 && g_batching);
    CHECK(events == "FDCAAB"); // preserve the exact-size allocation fallback

    reset(true, 16, 2);
    CHECK(gdnConvOutScratch(80) == nullptr && g_gdn_conv_out == nullptr);
    CHECK(g_batching && events == "FDCAAB"); // preserve caller mode on allocation error

    reset(false, 16);
    CHECK(gdnConvOutScratch(80) != nullptr && !g_batching);
    CHECK(events == "DCA"); // do not begin a batch the caller did not own

    reset(true, 128);
    Buffer* cached = g_gdn_conv_out;
    CHECK(gdnConvOutScratch(80) == cached && g_batching && events.empty());

    reset(true, 0);
    CHECK(gdnConvOutScratch(80) != nullptr && g_batching && events == "A");

    reset(true, 16);
    g_maxBufferBytes = 100;
    grown = gdnConvOutScratch(80);
    CHECK(grown && grown->bytes == 80 && g_batching && events == "FDCAB");

    reset(true, 16);
    refuseBegin = true;
    CHECK(gdnConvOutScratch(80) != nullptr && !g_batching && events == "FDCAB");

    delete g_gdn_conv_out;
    g_gdn_conv_out = nullptr;
    return 0;
}
`
