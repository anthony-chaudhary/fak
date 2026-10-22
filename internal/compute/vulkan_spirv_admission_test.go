package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime medium est=2s lane=default
// fak-test:justify Compiles and links the standalone production admission API from two translation units without Vulkan headers or a loader.
func TestVulkanSPIRVAdmissionStandalone(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++ compiler unavailable")
	}
	dir := t.TempDir()
	include, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for name, source := range map[string]string{
		"peer.cpp": `#include "vulkan_spirv_admission.h"
bool peer(const std::vector<char>& bytes) {
    return fak::vulkan::spirv::swigluPushConstantABI(bytes);
}
`,
		"main.cpp": `#include "vulkan_spirv_admission.h"
#include "vulkan_spirv_admission.h"
#include <cassert>
bool peer(const std::vector<char>&);
int main() {
    using namespace fak::vulkan::spirv;
    // Minimal fixed push ABI: signed count at byte 0, float scale at byte 4.
    // This is synthetic host-parser evidence, not a valid executable shader.
    const uint32_t words[] = {
        0x07230203, 0x00010000, 0, 6, 0,
        (4u<<16)|21u, 1, 32, 1,
        (3u<<16)|22u, 2, 32,
        (4u<<16)|30u, 3, 1, 2,
        (4u<<16)|32u, 4, 9, 3,
        (4u<<16)|59u, 4, 5, 9,
        (3u<<16)|71u, 3, 2,
        (5u<<16)|72u, 3, 0, 35, 0,
        (5u<<16)|72u, 3, 1, 35, 4,
    };
    std::vector<char> bytes(sizeof(words));
    std::memcpy(bytes.data(), words, sizeof(words));
    assert(spirvModuleFraming(bytes));
    assert(swigluPushConstantABI(bytes) && peer(bytes));
    for (size_t size = 0; size < bytes.size(); ++size) {
        const std::vector<char> truncated(bytes.begin(), bytes.begin() + size);
        assert(!swigluPushConstantABI(truncated));
        assert(!v41TailRoPEABI(truncated));
        assert(!v41SharedAttentionABI(truncated));
        assert(!v41IndexerScoreABI(truncated));
        assert(!v41BF16BoundaryABI(truncated));
        assert(!v41BF16CheckedABI(truncated));
        assert(!v41F32FiniteABI(truncated));
        assert(!v41LogitScaleABI(truncated));
    }
    uint32_t wrongOffset = 8;
    std::memcpy(bytes.data() + bytes.size() - sizeof(wrongOffset), &wrongOffset, sizeof(wrongOffset));
    assert(!swigluPushConstantABI(bytes) && !peer(bytes));
    assert(v41TokenGridModuleABI(bytes, 0x51324731u) == 0);
    bytes[0] = 0;
    assert(!spirvModuleFraming(bytes));
    assert(v41TokenGridModuleABI(bytes, 0x51324731u) == -1);
}
`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	binary := filepath.Join(dir, "admission")
	cmd := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I"+include, filepath.Join(dir, "main.cpp"), filepath.Join(dir, "peer.cpp"), "-o", binary)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile standalone admission: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("standalone admission contract: %v\n%s", err, out)
	}
}
