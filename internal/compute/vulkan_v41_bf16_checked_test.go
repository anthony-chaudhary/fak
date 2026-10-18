//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// Exercise the actual native ABI parser against the reviewed compiler's real
// module and structural corruptions of those bytes. No device is needed, and
// no parser-only result establishes numerical or physical qualification.
// Fresh status initialization, barriers and completion require a separate native
// orchestration witness; accepting status bytes alone cannot prove freshness.
// fak-test:runtime fast est=20ms lane=optin
func TestVulkanV41BF16CheckedModuleABI(t *testing.T) {
	// Startup must leave FAK_VULKAN_SPIRV unset and keep this binary away
	// from a sibling spirv directory. An assertion after init cannot undo a
	// device initialization attempt; the execution receipt checks placement.
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("ABI-only witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("ABI-only process unexpectedly registered a Vulkan device")
	}
	module := os.Getenv("FAK_V41_BF16_CHECKED_WITNESS_MODULE")
	expected := os.Getenv("FAK_V41_BF16_CHECKED_WITNESS_MODULE_SHA256")
	if !filepath.IsAbs(module) || filepath.Base(module) != "v41_bf16_checked.spv" || len(expected) != hex.EncodedLen(sha256.Size) {
		t.Fatal("explicit compiled BF16 module path and reviewed SHA256 required")
	}
	code, err := os.ReadFile(module)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(code)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("ABI module bytes differ from execution receipt")
	}
	if !v41BF16CheckedModuleABI(code) {
		t.Fatal("actual compiled checked BF16 shader failed native ABI admission")
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	// Resolve the actual PushConstant pointer's struct rather than accidentally
	// mutating the offset decoration of a storage-buffer block member.
	pointerPointee := make(map[uint32]uint32)
	structTypes := make(map[uint32]bool)
	definitions := make(map[uint32][]uint32)
	storageVariables := make(map[uint32]uint32)
	bindings := make(map[uint32]uint32)
	var signedScalar uint32
	var pushPointer uint32
	for i := 5; i < len(words); {
		n := int(words[i] >> 16)
		if n <= 0 || n > len(words)-i {
			t.Fatal("invalid compiler module framing")
		}
		w := words[i : i+n]
		switch w[0] & 0xffff {
		case 21: // OpTypeInt
			if len(w) == 4 {
				definitions[w[1]] = w
				if w[2] == 32 && w[3] == 1 && signedScalar == 0 {
					signedScalar = w[1]
				}
			}
		case 29: // OpTypeRuntimeArray
			if len(w) == 3 {
				definitions[w[1]] = w
			}
		case 30: // OpTypeStruct
			if len(w) >= 2 {
				structTypes[w[1]] = true
				definitions[w[1]] = w
			}
		case 32: // OpTypePointer
			if len(w) == 4 {
				definitions[w[1]] = w
			}
			if len(w) == 4 && w[2] == 9 {
				pointerPointee[w[1]] = w[3]
			}
		case 59: // OpVariable
			if len(w) >= 4 && w[3] == 12 {
				storageVariables[w[2]] = w[1]
			}
			if len(w) >= 4 && w[3] == 9 {
				if pushPointer != 0 {
					t.Fatal("ambiguous push variable")
				}
				pushPointer = w[1]
			}
		case 71: // OpDecorate
			if len(w) == 4 && w[2] == 33 {
				bindings[w[1]] = w[3]
			}
		}
		i += n
	}
	pushStruct := pointerPointee[pushPointer]
	if pushPointer == 0 || !structTypes[pushStruct] {
		t.Fatal("compiled module has no push block variable")
	}
	type storageRole struct{ variable, structure, array, scalar uint32 }
	roles := make(map[uint32]storageRole)
	for id, pointerID := range storageVariables {
		binding, found := bindings[id]
		if !found || binding > 2 {
			t.Fatal("storage variable lacks its checked binding")
		}
		if _, duplicate := roles[binding]; duplicate {
			t.Fatal("ambiguous storage binding")
		}
		pointer := definitions[pointerID]
		if len(pointer) != 4 || pointer[0]&0xffff != 32 || pointer[2] != 12 {
			t.Fatal("storage variable has no storage pointer")
		}
		structure := definitions[pointer[3]]
		if len(structure) != 3 || structure[0]&0xffff != 30 {
			t.Fatal("storage block is not a single array member")
		}
		array := definitions[structure[2]]
		if len(array) != 3 || array[0]&0xffff != 29 {
			t.Fatal("storage member is not a runtime array")
		}
		scalar := definitions[array[2]]
		if len(scalar) != 4 || scalar[0]&0xffff != 21 || scalar[2] != 32 || scalar[3] != 0 {
			t.Fatal("storage array is not uint32")
		}
		roles[binding] = storageRole{id, pointer[3], structure[2], array[2]}
	}
	if len(roles) != 3 || signedScalar == 0 {
		t.Fatal("compiled module lacks three storage roles or signed index scalar")
	}
	for a := uint32(0); a < 3; a++ {
		for b := a + 1; b < 3; b++ {
			if roles[a].structure == roles[b].structure {
				t.Fatal("checked storage roles unexpectedly share a block")
			}
		}
	}
	encode := func(changed []uint32) []byte {
		raw := make([]byte, len(changed)*4)
		for i, word := range changed {
			binary.LittleEndian.PutUint32(raw[4*i:], word)
		}
		return raw
	}
	reject := func(t *testing.T, changed []uint32) {
		t.Helper()
		if v41BF16CheckedModuleABI(encode(changed)) {
			t.Fatal("incompatible module was admitted")
		}
	}
	mutate := func(t *testing.T, edit func([]uint32) bool) {
		t.Helper()
		changed := append([]uint32(nil), words...)
		found := false
		for i := 5; i < len(changed); {
			n := int(changed[i] >> 16)
			if n <= 0 || n > len(changed)-i {
				t.Fatal("invalid compiler module framing")
			}
			if edit(changed[i : i+n]) {
				found = true
				break
			}
			i += n
		}
		if !found {
			t.Fatal("mutation did not find its intended ABI field")
		}
		reject(t, changed)
	}
	// Replace only one role's array element type. A private array type keeps
	// other roles and the push uints unchanged even if their original array
	// declaration is shared. The native ABI parser is structural; these corrupt
	// modules are never submitted or treated as executable SPIR-V witnesses.
	for binding := uint32(0); binding < 3; binding++ {
		role := roles[binding]
		t.Run(fmt.Sprintf("signed storage binding %d", binding), func(t *testing.T) {
			if words[3] == 0 || words[3] == ^uint32(0) {
				t.Fatal("module ID bound cannot admit mutation type")
			}
			retype := func(scalar uint32) []uint32 {
				arrayID := words[3]
				changed := append([]uint32(nil), words[:5]...)
				changed[3]++
				decorated, replaced := false, false
				for i := 5; i < len(words); {
					n := int(words[i] >> 16)
					w := words[i : i+n]
					if w[0]&0xffff == 71 && len(w) == 4 && w[1] == role.array && w[2] == 6 && w[3] == 4 {
						if decorated {
							t.Fatal("duplicate role array stride")
						}
						changed = append(changed, 4<<16|71, arrayID, 6, 4)
						decorated = true
					}
					if w[0]&0xffff == 30 && w[1] == role.structure {
						changed = append(changed, 3<<16|29, arrayID, scalar)
						changed = append(changed, w[0], w[1], arrayID)
						replaced = true
					} else {
						changed = append(changed, w...)
					}
					i += n
				}
				if !decorated || !replaced {
					t.Fatal("signed role mutation missed array/block")
				}
				return changed
			}
			// The same graph rewrite with uint must still be admitted. This
			// rules out rejection solely because a fresh array/ID was introduced.
			if !v41BF16CheckedModuleABI(encode(retype(role.scalar))) {
				t.Fatal("equivalent unsigned role array was not admitted")
			}
			reject(t, retype(signedScalar))
		})
	}
	// Removing the required qualifier must fail without adding an opposite
	// qualifier which could independently mask a missing access check. Remove
	// both member and variable forms from this role, preserving all other roles.
	for _, tc := range []struct {
		name          string
		binding, kind uint32
	}{
		{"input readonly removed", 0, 24},
		{"output writeonly removed", 1, 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role := roles[tc.binding]
			changed := append([]uint32(nil), words[:5]...)
			removedMember := false
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				w := words[i : i+n]
				member := w[0]&0xffff == 72 && len(w) == 4 && w[1] == role.structure && w[2] == 0 && w[3] == tc.kind
				variable := w[0]&0xffff == 71 && len(w) == 3 && w[1] == role.variable && w[2] == tc.kind
				if member {
					removedMember = true
				}
				if !member && !variable {
					changed = append(changed, w...)
				}
				i += n
			}
			if !removedMember {
				t.Fatal("access mutation missed required role member decoration")
			}
			reject(t, changed)
		})
	}
	for _, tc := range []struct {
		name string
		kind uint32
	}{
		{"status readonly added", 24},
		{"status writeonly added", 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role := roles[2]
			changed := append([]uint32(nil), words[:5]...)
			inserted := false
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				w := words[i : i+n]
				if w[0]&0xffff == 72 && len(w) == 5 && w[1] == role.structure && w[2] == 0 && w[3] == 35 && w[4] == 0 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V OpMemberDecorate Offset has five words: opcode, target, member, decoration, offset.
					if inserted {
						t.Fatal("duplicate status member offset")
					}
					changed = append(changed, 4<<16|72, role.structure, 0, tc.kind)
					inserted = true
				}
				changed = append(changed, w...)
				i += n
			}
			if !inserted {
				t.Fatal("status mutation missed its member offset")
			}
			reject(t, changed)
		})
	}
	// Access qualifiers may be attached to a block member or its variable.
	// Preserve the effective requirement when moving it, then reject each
	// role's opposite variable restriction independently of its member flags.
	for _, tc := range []struct {
		name          string
		binding, kind uint32
	}{
		{"input readonly variable-only", 0, 24},
		{"output writeonly variable-only", 1, 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role := roles[tc.binding]
			hasVariable := false
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				w := words[i : i+n]
				if w[0]&0xffff == 71 && len(w) == 3 && w[1] == role.variable && w[2] == tc.kind {
					hasVariable = true
				}
				i += n
			}
			changed := append([]uint32(nil), words[:5]...)
			removed, inserted := false, false
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				w := words[i : i+n]
				if !hasVariable && w[0]&0xffff == 71 && len(w) == 4 && w[1] == role.variable && w[2] == 33 {
					if inserted {
						t.Fatal("duplicate variable binding")
					}
					changed = append(changed, 3<<16|71, role.variable, tc.kind)
					inserted = true
				}
				if w[0]&0xffff == 72 && len(w) == 4 && w[1] == role.structure && w[2] == 0 && w[3] == tc.kind {
					removed = true
				} else {
					changed = append(changed, w...)
				}
				i += n
			}
			if !removed || (!hasVariable && !inserted) {
				t.Fatal("qualifier relocation missed its role")
			}
			if !v41BF16CheckedModuleABI(encode(changed)) {
				t.Fatal("equivalent variable-only access was not admitted")
			}
		})
	}
	for _, tc := range []struct {
		name          string
		binding, kind uint32
	}{
		{"input variable unreadable", 0, 25},
		{"output variable unwritable", 1, 24},
		{"status variable readonly", 2, 24},
		{"status variable writeonly", 2, 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			role := roles[tc.binding]
			changed := append([]uint32(nil), words[:5]...)
			inserted := false
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				w := words[i : i+n]
				if w[0]&0xffff == 71 && len(w) == 3 && w[1] == role.variable && w[2] == tc.kind {
					t.Fatal("unmodified module already has forbidden variable access")
				}
				if w[0]&0xffff == 71 && len(w) == 4 && w[1] == role.variable && w[2] == 33 {
					if inserted {
						t.Fatal("duplicate variable binding")
					}
					changed = append(changed, 3<<16|71, role.variable, tc.kind)
					inserted = true
				}
				changed = append(changed, w...)
				i += n
			}
			if !inserted {
				t.Fatal("variable access mutation missed binding")
			}
			reject(t, changed)
		})
	}
	for name, edit := range map[string]func([]uint32) bool{
		"local width": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 6 && w[2] == 17 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V OpExecutionMode LocalSize has six words: opcode, entry point, mode, x, y, z.
				w[3] = 32
				return true
			}
			return false
		},
		"duplicate binding": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 33 && w[3] == 2 {
				w[3] = 1
				return true
			}
			return false
		},
		"wrong array stride": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 6 {
				w[3] = 8
				return true
			}
			return false
		},
		"push site offset": func(w []uint32) bool {
			if w[0]&0xffff == 72 && len(w) == 5 && w[1] == pushStruct && w[2] == 1 && w[3] == 35 && w[4] == 4 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V OpMemberDecorate Offset has five words: opcode, target, member, decoration, offset.
				w[4] = 8
				return true
			}
			return false
		},
		"wrong atomic reduction": func(w []uint32) bool {
			// SPIRV-Headers: OpAtomicUMin=237, OpAtomicUMax=239. The
			// same framed instruction must not pass the checked-kernel census.
			if w[0]&0xffff == 237 {
				w[0] = w[0]&0xffff0000 | 239
				return true
			}
			return false
		},
		"nonzero descriptor set": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 34 && w[3] == 0 {
				w[3] = 1
				return true
			}
			return false
		},
		"zero instruction size": func(w []uint32) bool { w[0] &= 0xffff; return true },
	} {
		t.Run(name, func(t *testing.T) { mutate(t, edit) })
	}
	for _, invalid := range [][]byte{nil, code[:19], code[:len(code)-1]} {
		if v41BF16CheckedModuleABI(invalid) {
			t.Fatal("truncated module admitted")
		}
	}
}

// This witness compiles the complete hash-bound production shim with all Vulkan
// entrypoints intercepted. It checks native orchestration and ownership, without
// a loader, physical device, or numerical shader execution.
// fak-test:runtime medium est=30s lane=optin
// fak-test:justify Compiles the complete native owner and checks its actual command/failure paths.
func TestVulkanV41BF16CheckedOrchestration(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_BF16_CHECKED_ORCHESTRATION") != "1" {
		t.Fatal("explicit checked orchestration opt-in required")
	}
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("orchestration witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("orchestration process unexpectedly registered a Vulkan device")
	}
	// Startup isolation is also required in the execution receipt; this test
	// runs after Go package initialization and cannot undo an earlier attempt.
	sourceDir := os.Getenv("FAK_V41_BF16_CHECKED_WITNESS_SOURCE_DIR")
	if !filepath.IsAbs(sourceDir) {
		t.Fatal("explicit absolute native source directory required")
	}
	tmp := t.TempDir()
	var sourceText strings.Builder
	for _, file := range []struct{ name, env string }{
		{"vulkan_shim.cpp", "FAK_V41_BF16_CHECKED_WITNESS_SHIM_SHA256"},
		{"vulkan_backend.h", "FAK_V41_BF16_CHECKED_WITNESS_HEADER_SHA256"},
		{"v41_index_scale_bf16_spirv.h", "FAK_V41_BF16_CHECKED_WITNESS_INDEX_SCALE_SHA256"},
	} {
		raw, err := os.ReadFile(filepath.Join(sourceDir, file.name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(raw)
		if len(os.Getenv(file.env)) != hex.EncodedLen(sha256.Size) || hex.EncodeToString(digest[:]) != os.Getenv(file.env) {
			t.Fatalf("%s differs from the reviewed native source receipt", file.name)
		}
		// An exact byte snapshot makes a concurrent checkout edit irrelevant to
		// this compilation. No production body is extracted or rewritten.
		if err := os.WriteFile(filepath.Join(tmp, file.name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
		t.Logf("included %s sha256=%x", file.name, digest)
		sourceText.Write(raw)
		sourceText.WriteByte('\n')
	}
	allowed := map[string]bool{}
	for _, name := range strings.Fields(`vkAllocateDescriptorSets vkUpdateDescriptorSets vkAllocateCommandBuffers vkResetCommandBuffer vkBeginCommandBuffer vkEndCommandBuffer vkFreeCommandBuffers vkResetFences vkQueueSubmit vkWaitForFences vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants vkCmdFillBuffer vkCmdPipelineBarrier vkCmdDispatch vkCmdCopyBuffer`) {
		allowed[name] = true
	}
	names := map[string]bool{}
	for _, name := range regexp.MustCompile(`\bvk[A-Z][A-Za-z0-9_]*\b`).FindAllString(sourceText.String(), -1) {
		names[name] = true
	}
	for name := range allowed {
		if !names[name] {
			t.Fatalf("reviewed interception boundary is missing %s", name)
		}
	}
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	var fixture strings.Builder
	fixture.WriteString(v41BF16CheckedOrchestrationPrefix)
	for _, name := range ordered {
		if allowed[name] {
			fmt.Fprintf(&fixture, "static_assert(std::is_same_v<decltype(&checked_%s), PFN_%s>);\n#define %s checked_%s\n", name, name, name, name)
		} else {
			// Official PFN types preserve the real argument/return ABI. Unknown
			// source tokens fail compilation instead of silently escaping.
			fmt.Fprintf(&fixture, "#define %s CheckedAbort<PFN_%s>::call\n", name, name)
		}
	}
	fixture.WriteString("\n#define FAK_V41_BF16_BOUNDARY_WITNESS 1\n#include \"vulkan_shim.cpp\"\n")
	fixture.WriteString(v41BF16CheckedOrchestrationCases)
	raw := []byte(fixture.String())
	path := filepath.Join(tmp, "checked-orchestration.cpp")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("orchestration fixture sha256=%x intercepted=%d allowed=%d", sha256.Sum256(raw), len(ordered), len(allowed))
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested witness requires a C++17 compiler: ", err)
	}
	binary := filepath.Join(tmp, "checked-orchestration")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	// No Vulkan loader or user-provided linker flags are permitted here.
	args := []string{"-std=c++17", "-O0", "-g0"}
	if include := os.Getenv("FAK_V41_BF16_CHECKED_WITNESS_VULKAN_INCLUDE"); include != "" {
		if !filepath.IsAbs(include) {
			t.Fatal("Vulkan header include root must be absolute")
		}
		if _, err := os.Stat(filepath.Join(include, "vulkan", "vulkan.h")); err != nil {
			t.Fatal(err)
		}
		args = append(args, "-I", include)
	}
	args = append(args, path, "-o", binary)
	cmd := exec.CommandContext(ctx, compiler, args...)
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile real-shim fixture: %v\n%s", err, output)
	}
	ctxRun, cancelRun := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancelRun()
	cmd = exec.CommandContext(ctxRun, binary)
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("real-shim orchestration: %v\n%s", err, output)
	}
	t.Logf("%s", output)
}

const v41BF16CheckedOrchestrationPrefix = `
#include <vulkan/vulkan.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace checked_fixture {
[[noreturn]] void die(const char* why) { std::fprintf(stderr, "checked orchestration: %s\n", why); std::abort(); }
void require(bool ok, const char* why) { if (!ok) die(why); }
template<class T> T handle(uintptr_t id) { return (T)id; }
struct Op { int kind; VkBuffer source{}, dest{}; uint32_t count{}, site{}; };
struct Command {
    bool begun = false, ended = false, filled = false, dependency = false, completed = false;
    VkPipeline pipeline{};
    std::vector<VkBuffer> bindings;
    std::vector<uint32_t> push;
    std::vector<Op> ops;
};
std::map<VkCommandBuffer, Command> commands;
std::map<VkCommandBuffer, std::pair<VkDevice,VkCommandPool>> commandOwners;
std::map<VkDescriptorSet, std::vector<VkBuffer>> descriptors;
std::map<VkBuffer, std::array<uint32_t, 4>> memory;
std::map<std::string, int> calls;
std::string failure;
int failureOrdinal = 1;
VkResult failureResult = VK_ERROR_OUT_OF_HOST_MEMORY;
uintptr_t nextHandle = 100;
VkCommandBuffer submitted{};
VkBuffer inputHandle{}, outputHandle{}, statusHandle{}, stageHandle{};
VkPipeline checkedPipeline{};
uint32_t expectedCount = 64, expectedSite = 1, expectedGroups = 1;
bool checkedDispatch = true;
bool finiteDispatch = false;
int fills = 0, dispatches = 0, copies = 0, barriers = 0;
std::array<uint32_t, 4> stage{};
VkResult call(const char* name) {
    const int n = ++calls[name];
    return failure == name && n == failureOrdinal ? failureResult : VK_SUCCESS;
}
Command& command(VkCommandBuffer id) {
    auto it = commands.find(id);
    require(it != commands.end(), "unknown command buffer");
    return it->second;
}
void execute(VkCommandBuffer id) {
    for (const auto& op : command(id).ops) {
        if (op.kind == 0) memory[op.dest].fill(UINT32_MAX);
        else if (op.kind == 1 || op.kind == 3) {
            require(memory[op.dest] == std::array<uint32_t,4>{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}, "status write did not follow its executed all-ones fill");
            memory[op.dest] = {op.kind == 3 ? 0x46333243u : 0x42463143u, op.site, UINT32_MAX, op.count};
        }
        else if (op.kind == 2) stage = memory.at(op.source);
        else die("unknown recorded operation");
    }
}
} // namespace checked_fixture

// Every unallowlisted Vulkan function has its official signature and aborts.
// No loader is linked, including for APIs in unused production helper bodies.
template<class> struct CheckedAbort;
template<class R, class... A> struct CheckedAbort<R(VKAPI_PTR*)(A...)> {
    static VKAPI_ATTR R VKAPI_CALL call(A...) { checked_fixture::die("unexpected Vulkan API reached"); }
};

static VKAPI_ATTR VkResult VKAPI_CALL checked_vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    using namespace checked_fixture;
    require(info && out && info->descriptorSetCount == 1, "descriptor allocation shape");
    const VkResult r = call("vkAllocateDescriptorSets");
    *out = r == VK_SUCCESS ? handle<VkDescriptorSet>(++nextHandle) : VK_NULL_HANDLE;
    if (r == VK_SUCCESS) descriptors[*out] = {};
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkUpdateDescriptorSets(VkDevice, uint32_t n, const VkWriteDescriptorSet* writes, uint32_t copyCount, const VkCopyDescriptorSet*) {
    using namespace checked_fixture;
    call("vkUpdateDescriptorSets");
    require((n == 2 || n == 3) && writes && copyCount == 0, "descriptor update shape");
    for (uint32_t i = 0; i < n; ++i) {
        const auto& w = writes[i];
        require(w.dstBinding == i && w.descriptorCount == 1 && w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && w.pBufferInfo, "descriptor role");
        require(w.pBufferInfo->offset == 0 && w.pBufferInfo->range == VK_WHOLE_SIZE, "descriptor range");
        auto& bindings = descriptors.at(w.dstSet);
        if (bindings.size() < n) bindings.resize(n);
        bindings[i] = w.pBufferInfo->buffer;
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* out) {
    using namespace checked_fixture;
    require(info && out && info->commandBufferCount == 1, "command allocation shape");
    const VkResult r = call("vkAllocateCommandBuffers");
    *out = r == VK_SUCCESS ? handle<VkCommandBuffer>(++nextHandle) : VK_NULL_HANDLE;
    if (r == VK_SUCCESS) { commands[*out] = {}; commandOwners[*out] = {device,info->commandPool}; }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkResetCommandBuffer(VkCommandBuffer id, VkCommandBufferResetFlags flags) {
    using namespace checked_fixture;
    require(flags == 0 && command(id).completed, "reset without successful fence");
    const VkResult r = call("vkResetCommandBuffer");
    if (r == VK_SUCCESS) command(id) = {};
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkBeginCommandBuffer(VkCommandBuffer id, const VkCommandBufferBeginInfo*) {
    using namespace checked_fixture;
    const VkResult r = call("vkBeginCommandBuffer");
    if (r == VK_SUCCESS) command(id).begun = true;
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkEndCommandBuffer(VkCommandBuffer id) {
    using namespace checked_fixture;
    require(command(id).begun, "end without begin");
    const VkResult r = call("vkEndCommandBuffer");
    if (r == VK_SUCCESS) command(id).ended = true;
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkFreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t n, const VkCommandBuffer* ids) {
    using namespace checked_fixture;
    call("vkFreeCommandBuffers");
    require(n == 1 && ids && commandOwners.at(ids[0]) == std::make_pair(device,pool) &&
            commands.erase(ids[0]) == 1 && commandOwners.erase(ids[0]) == 1, "command release identity/owner");
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkResetFences(VkDevice, uint32_t n, const VkFence*) {
    using namespace checked_fixture;
    require(n == 1, "fence reset shape");
    return call("vkResetFences");
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkQueueSubmit(VkQueue, uint32_t n, const VkSubmitInfo* infos, VkFence) {
    using namespace checked_fixture;
    require(n == 1 && infos && infos->commandBufferCount == 1, "submit shape");
    submitted = infos->pCommandBuffers[0];
    require(command(submitted).ended, "submit before command end");
    const VkResult r = call("vkQueueSubmit");
    // Even an uncertain submit may have changed memory. A successful-looking
    // envelope must never override the real native failure/pending result.
    execute(submitted);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkWaitForFences(VkDevice, uint32_t n, const VkFence*, VkBool32 all, uint64_t timeout) {
    using namespace checked_fixture;
    require(n == 1 && all == VK_TRUE && timeout == UINT64_MAX && submitted != VK_NULL_HANDLE, "wait shape");
    const VkResult r = call("vkWaitForFences");
    if (r == VK_SUCCESS) command(submitted).completed = true;
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdBindPipeline(VkCommandBuffer id, VkPipelineBindPoint point, VkPipeline pipeline) {
    using namespace checked_fixture;
    call("vkCmdBindPipeline");
    require(point == VK_PIPELINE_BIND_POINT_COMPUTE, "pipeline bind point");
    command(id).pipeline = pipeline;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdBindDescriptorSets(VkCommandBuffer id, VkPipelineBindPoint point, VkPipelineLayout, uint32_t first, uint32_t n, const VkDescriptorSet* sets, uint32_t dynamicCount, const uint32_t*) {
    using namespace checked_fixture;
    call("vkCmdBindDescriptorSets");
    require(point == VK_PIPELINE_BIND_POINT_COMPUTE && first == 0 && n == 1 && sets && dynamicCount == 0, "descriptor bind shape");
    command(id).bindings = descriptors.at(sets[0]);
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdPushConstants(VkCommandBuffer id, VkPipelineLayout, VkShaderStageFlags flags, uint32_t offset, uint32_t size, const void* data) {
    using namespace checked_fixture;
    call("vkCmdPushConstants");
    require(flags == VK_SHADER_STAGE_COMPUTE_BIT && offset == 0 && size == (checkedDispatch ? 8u : 4u), "push shape");
    auto& push = command(id).push;
    push.resize(size/4);
    std::memcpy(push.data(), data, size);
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdFillBuffer(VkCommandBuffer id, VkBuffer dest, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    using namespace checked_fixture;
    call("vkCmdFillBuffer"); ++fills;
    auto& c = command(id);
    require(checkedDispatch && c.begun && !c.filled && dest == statusHandle && offset == 0 && size == 16 && value == UINT32_MAX, "fresh exact status fill");
    c.filled = true; c.dependency = false;
    c.ops.push_back({0, {}, dest, 0, 0});
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdPipelineBarrier(VkCommandBuffer id, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags flags, uint32_t n, const VkMemoryBarrier* memoryBarriers, uint32_t bufferCount, const VkBufferMemoryBarrier*, uint32_t imageCount, const VkImageMemoryBarrier*) {
    using namespace checked_fixture;
    call("vkCmdPipelineBarrier"); ++barriers;
    const auto stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    require(srcStage == stages && dstStage == stages && flags == 0 && n == 1 && memoryBarriers && bufferCount == 0 && imageCount == 0, "conservative barrier stages/shape");
    require(memoryBarriers[0].srcAccessMask == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) && memoryBarriers[0].dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT), "input/status barrier access");
    command(id).dependency = true;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdDispatch(VkCommandBuffer id, uint32_t x, uint32_t y, uint32_t z) {
    using namespace checked_fixture;
    call("vkCmdDispatch"); ++dispatches;
    auto& c = command(id);
    require(c.push.size() == (checkedDispatch ? 2u : 1u) && x == expectedGroups && y == 1 && z == 1 && c.push[0] == expectedCount, "dispatch geometry/count");
    if (finiteDispatch) {
        require(checkedDispatch && c.bindings.size() == 2 && c.bindings[0] == inputHandle &&
                c.bindings[1] == statusHandle && c.pipeline == checkedPipeline &&
                c.filled && c.dependency && c.push[1] == expectedSite,
                "finite input/status roles and same-command initialization");
        c.ops.push_back({3, {}, statusHandle, expectedCount, expectedSite});
        return;
    }
    require(c.bindings.size() == (checkedDispatch ? 3u : 2u) && c.bindings[0] == inputHandle && c.bindings[1] == outputHandle, "dispatch input/output roles");
    if (checkedDispatch) {
        require(c.pipeline == checkedPipeline && c.filled && c.dependency && c.bindings[2] == statusHandle && c.push[1] == expectedSite, "same-command initialized status/push/site");
        c.ops.push_back({1, {}, statusHandle, expectedCount, expectedSite});
    } else {
        require(!c.filled, "old caller unexpectedly gained status fill");
    }
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdCopyBuffer(VkCommandBuffer id, VkBuffer source, VkBuffer dest, uint32_t n, const VkBufferCopy* regions) {
    using namespace checked_fixture;
    call("vkCmdCopyBuffer"); ++copies;
    require(command(id).dependency && source == statusHandle && dest == stageHandle && n == 1 && regions && regions[0].srcOffset == 0 && regions[0].dstOffset == 0 && regions[0].size == 16, "checked status readback range/dependency");
    command(id).ops.push_back({2, source, dest, 0, 0});
}
`

const v41BF16CheckedOrchestrationCases = `
namespace checked_fixture {
Buffer input{}, output{}, status{}, staging{};
void reset() {
    // Fixture-only teardown of synthetic handles AFTER each case's assertions.
    // This is not production recovery or evidence of real driver quiescence.
    commands.clear(); commandOwners.clear(); descriptors.clear(); memory.clear(); calls.clear();
    failure.clear(); failureOrdinal = 1; failureResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    submitted = VK_NULL_HANDLE; fills = dispatches = copies = barriers = 0;
    g_pool.clear(); g_poolCount = 0; g_batchFreed.clear(); g_batchSets.clear(); g_descSetPool.clear();
    g_batching = false; g_batchCmd = VK_NULL_HANDLE; g_batchOps = 0; g_batchHasV41 = false;
    g_stepBatch = false; g_batchIdleCmd = VK_NULL_HANDLE;
    g_batchIdleDevice = VK_NULL_HANDLE; g_batchIdlePool = VK_NULL_HANDLE;
    g_v41SubmissionPendingFailure = false; g_submissionStatus = VK_SUCCESS;
    g_restore = RestoreTransaction{}; g_debugD2HStagingFailureOnce = false;
    g_d2hCount = 0; g_d2hBytes = 0; g_transferCountersValid = true;
    require(!g_dp_on && !g_tsOn, "profiling must be disabled before process initialization");
    g_ready = true; g_dev = handle<VkDevice>(1); g_queue = handle<VkQueue>(2);
    g_cmdpool = handle<VkCommandPool>(3); g_submitFence = handle<VkFence>(4); g_descpool = handle<VkDescriptorPool>(5);
    g_maxComputeWorkGroupCountX = 2;
    g_maxStorageBufferRange = g_maxMemoryAllocationSize = g_maxBufferBytes = std::numeric_limits<VkDeviceSize>::max();
    input = {}; output = {}; status = {}; staging = {};
    Buffer* buffers[] = {&input, &output, &status, &staging};
    for (int i = 0; i < 4; ++i) {
        buffers[i]->buf = handle<VkBuffer>(10+i); buffers[i]->mem = handle<VkDeviceMemory>(20+i);
        buffers[i]->bytes = i < 2 ? 1024 : 16;
        buffers[i]->props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    inputHandle = input.buf; outputHandle = output.buf; statusHandle = status.buf; stageHandle = staging.buf;
    stage.fill(0); g_stage = &staging; g_stageMapped = stage.data(); g_stageCap = 16;
    auto& k = g_kern[K_V41_BF16_CHECKED];
    k = {}; k.pipe = handle<VkPipeline>(30); k.dsl = handle<VkDescriptorSetLayout>(31); k.layout = handle<VkPipelineLayout>(32); k.nbuf = 3; k.pcsize = 8;
    checkedPipeline = k.pipe; g_have_v41_bf16_checked = 1;
    auto& raw = g_kern[K_V41_BF16_BOUNDARY];
    raw = {}; raw.pipe = handle<VkPipeline>(40); raw.dsl = handle<VkDescriptorSetLayout>(41); raw.layout = handle<VkPipelineLayout>(42); raw.nbuf = 2; raw.pcsize = 4;
    g_have_v41_bf16_boundary = 1;
    expectedCount = 64; expectedSite = 1; expectedGroups = 1; checkedDispatch = true; finiteDispatch = false;
    g_kern[K_V41_F32_FINITE] = {};
    memory[statusHandle] = {0x42463143, expectedSite, UINT32_MAX, expectedCount};
}
int invoke() { return fvk_v41_bf16_checked_f32(&input, &output, &status, int(expectedCount), expectedSite); }
void releaseAndCheck(bool pending) {
    const auto before = calls;
    fvk_free(&output); fvk_free(&status);
    require(calls == before, "free unexpectedly reached a Vulkan API");
    if (pending) {
        require(g_batchFreed.size() == 2 && g_batchFreed[0] == &output && g_batchFreed[1] == &status && g_poolCount == 0, "uncertain owners escaped quarantine");
        require(fvk_malloc(output.bytes) == nullptr && fvk_malloc(status.bytes) == nullptr, "pending buffers were reusable");
    } else {
        require(g_batchFreed.empty() && g_poolCount == 2, "known owners were not pooled");
        require(fvk_malloc(output.bytes) == &output && fvk_malloc(status.bytes) == &status && g_poolCount == 0, "known pooled owners changed identity");
    }
}
void refused(const char* name, const void* a, void* b, void* c, int count, uint32_t site) {
    const int result = fvk_v41_bf16_checked_f32(a,b,c,count,site);
    require(result != 0 && calls.empty() && fills == 0 && dispatches == 0, name);
}
void geometryCases() {
    reset(); require(fvk_v41_bf16_checked_count_supported(1) != 0 && fvk_v41_bf16_checked_count_supported(0) == 0 && fvk_v41_bf16_checked_count_supported(-1) == 0, "count predicate boundaries");
    const bool maxFits = uint64_t(INT32_MAX)*4u <= SIZE_MAX;
    require((fvk_v41_bf16_checked_count_supported(INT32_MAX) != 0) == maxFits, "count byte-extent maximum");
    require(calls.empty(), "count predicate reached driver");
    g_maxBufferBytes = 15; require(fvk_v41_bf16_checked_count_supported(1) == 0, "status allocation bound missing");
    g_maxBufferBytes = 16; require(fvk_v41_bf16_checked_count_supported(4) != 0 && fvk_v41_bf16_checked_count_supported(5) == 0, "exact count byte budget");
    require(calls.empty(), "bounded count predicate reached driver");
    for (int count : {0, -1}) { reset(); refused("invalid count", &input,&output,&status,count,1); }
    for (uint32_t site : {0u,3u,UINT32_MAX}) { reset(); refused("invalid site",&input,&output,&status,64,site); }
    for (int role = 0; role < 3; ++role) {
        reset(); Buffer* b[] = {&input,&output,&status}; b[role]->bytes = role == 2 ? 15 : 255;
        refused("short extent",&input,&output,&status,64,1);
        reset(); Buffer* c[] = {&input,&output,&status}; c[role]->memoryOffset = std::numeric_limits<VkDeviceSize>::max();
        refused("allocation range overflow",&input,&output,&status,64,1);
    }
    for (int a = 0; a < 3; ++a) for (int b = a+1; b < 3; ++b) {
        reset(); Buffer* args[] = {&input,&output,&status}; args[b] = args[a];
        refused("identical owner alias",args[0],args[1],args[2],64,1);
        reset(); Buffer* buf[] = {&input,&output,&status}; buf[b]->buf = buf[a]->buf;
        refused("identical Vulkan buffer alias",&input,&output,&status,64,1);
        reset(); Buffer* mem[] = {&input,&output,&status}; mem[b]->mem = mem[a]->mem; mem[b]->memoryOffset = mem[a]->bytes-1;
        refused("overlapping memory alias",&input,&output,&status,64,1);
    }
    reset(); g_ready = false; refused("backend not ready",&input,&output,&status,64,1);
    reset(); g_batching = true; refused("active batch must refuse",&input,&output,&status,64,1);
    reset(); g_have_v41_bf16_checked = 0; refused("capability absent",&input,&output,&status,64,1);
    reset(); g_kern[K_V41_BF16_CHECKED].pipe = VK_NULL_HANDLE; refused("pipeline absent",&input,&output,&status,64,1);
    reset(); g_submissionStatus = VK_ERROR_DEVICE_LOST; refused("sticky failure",&input,&output,&status,64,1);
    reset(); g_v41SubmissionPendingFailure = true; refused("independent pending gate",&input,&output,&status,64,1);
    reset(); g_maxComputeWorkGroupCountX = 0; refused("zero group maximum",&input,&output,&status,64,1);
}
void successCases() {
    reset();
    for (uint32_t site : {1u,2u}) for (int reuse = 0; reuse < 2; ++reuse) {
        expectedSite = site; expectedCount = reuse ? 193 : 64; expectedGroups = reuse ? 2 : 1;
        // Always preload an identical stale success, including reused status.
        memory[statusHandle] = {0x42463143,site,UINT32_MAX,expectedCount};
        const int oldFills = fills;
        require(invoke() == 0 && fills == oldFills+1, "per-call fresh initialization");
        std::array<uint32_t,4> host{0xdeadc0de,0xdeadc0de,0xdeadc0de,0xdeadc0de};
        require(fvk_v41_d2h(host.data(),&status,16) == 0 && host == memory[statusHandle], "checked readback success");
        require(commands.empty() && g_descSetPool[g_kern[K_V41_BF16_CHECKED].dsl].size() == 1, "completed command/descriptor recycling");
    }
    require(fills == 4 && dispatches == 4 && copies == 4 && barriers == 8 && g_d2hCount == 4 && g_d2hBytes == 64 && g_transferCountersValid, "successful lifecycle counters");
    releaseAndCheck(false);
    reset(); checkedDispatch = false;
    require(fvk_v41_bf16_boundary_f32(&input,&output,64) == 0 && fills == 0 && dispatches == 1, "legacy null-init path changed");
    require(commands.empty(), "legacy completed command retained");
}
void failures() {
    const char* APIs[] = {"vkAllocateDescriptorSets","vkAllocateCommandBuffers","vkBeginCommandBuffer","vkEndCommandBuffer","vkResetFences","vkQueueSubmit","vkWaitForFences"};
    for (const char* api : APIs) {
        reset(); failure = api;
        const bool pending = failure == "vkQueueSubmit" || failure == "vkWaitForFences";
        failureResult = failure == "vkWaitForFences" ? VK_TIMEOUT : pending ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult expected = failureResult > VK_SUCCESS ? VK_ERROR_UNKNOWN : failureResult;
        require(invoke() == expected && g_submissionStatus == expected && g_v41SubmissionPendingFailure == pending && calls[api] == 1, "selected native failure identity");
        require(commands.size() == (pending ? 1u : 0u), "failed command ownership");
        const size_t expectedSets = pending || failure == "vkAllocateDescriptorSets" ? 0 : 1;
        require(g_descSetPool[g_kern[K_V41_BF16_CHECKED].dsl].size() == expectedSets, "failed descriptor ownership");
        if (pending) require(memory[statusHandle][0] == 0x42463143, "uncertain successful-looking envelope setup");
        releaseAndCheck(pending);
        std::printf("entrypoint failure %s retained\n",api);
    }
    reset(); failure = "vkAllocateDescriptorSets"; failureResult = VK_ERROR_DEVICE_LOST;
    require(invoke() == VK_ERROR_DEVICE_LOST && g_v41SubmissionPendingFailure && commands.empty(), "descriptor device-loss quarantine");
    releaseAndCheck(true);
    for (const char* api : APIs) {
        if (std::string(api) == "vkAllocateDescriptorSets") continue;
        reset(); require(invoke() == 0, "D2H prerequisite dispatch");
        failure = api; failureOrdinal = 2;
        const bool pending = failure == "vkQueueSubmit" || failure == "vkWaitForFences";
        failureResult = failure == "vkWaitForFences" ? VK_TIMEOUT : pending ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult expected = failureResult > VK_SUCCESS ? VK_ERROR_UNKNOWN : failureResult;
        stage = memory[statusHandle]; // stale valid staging is deliberately hostile
        std::array<uint32_t,4> host{0xdeadc0de,0xdeadc0de,0xdeadc0de,0xdeadc0de};
        const auto untouched = host;
        require(fvk_v41_d2h(host.data(),&status,16) == expected && host == untouched && calls[api] == 2, "failed D2H published stale status");
        require(g_submissionStatus == expected && g_v41SubmissionPendingFailure == pending && g_d2hCount == 0 && g_d2hBytes == 0, "D2H failure state/counters");
        require(commands.size() == (pending ? 1u : 0u), "D2H command ownership");
        releaseAndCheck(pending);
        std::printf("D2H failure %s withheld\n",api);
    }
}
// Exercise the real batch owner; mocked API counts are not GPU latency evidence.
void batchReuseCases() {
    reset(); g_stepBatch = true;
    for (int i = 0; i < 4; ++i) {
        batchBegin(); require(g_batching, "batch begin");
        g_batchOps = 1; batchFlush();
        require(!g_batching && g_batchCmd == VK_NULL_HANDLE && g_batchIdleCmd != VK_NULL_HANDLE,
                "completed batch did not transfer to idle ownership");
        require(commands.size() == 1 && calls["vkAllocateCommandBuffers"] == 1 &&
                calls["vkFreeCommandBuffers"] == 0 && calls["vkResetCommandBuffer"] == i,
                "steady-state command allocation/free traffic");
    }
    require(calls["vkQueueSubmit"] == 4 && calls["vkWaitForFences"] == 4,
            "reuse changed submission/fence count");
    batchBegin(); batchFlush(); // An empty recording has no new completion proof.
    require(commands.empty() && g_batchIdleCmd == VK_NULL_HANDLE, "empty batch retained");
    g_stepBatch = false;
    require(v41EndSubmitWaitChecked(v41BeginCmdChecked(), true) && commands.empty(),
            "one-shot ownership changed");
    for (const char* api : {"vkResetCommandBuffer", "vkBeginCommandBuffer"}) {
        for (VkResult result : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_DEVICE_LOST}) {
            reset(); g_stepBatch = true; batchBegin(); g_batchOps = 1; batchFlush();
            failure = api; failureOrdinal = failure == "vkBeginCommandBuffer" ? 2 : 1;
            failureResult = result;
            batchBegin();
            const bool lost = result == VK_ERROR_DEVICE_LOST;
            require(!g_batching && g_submissionStatus == result &&
                    g_v41SubmissionPendingFailure == lost && commands.size() == (lost ? 1u : 0u),
                    "reuse error status/ownership");
            const int attempts = calls[api]; batchBegin();
            require(calls[api] == attempts, "sticky failure silently retried");
        }
    }
    for (const char* api : {"vkEndCommandBuffer", "vkResetFences", "vkQueueSubmit", "vkWaitForFences"}) {
        reset(); g_stepBatch = true; batchBegin(); g_batchOps = 1;
        failure = api; failureResult = failure == "vkWaitForFences" ? VK_TIMEOUT : VK_ERROR_OUT_OF_HOST_MEMORY;
        batchFlush();
        const bool pending = failure == "vkQueueSubmit" || failure == "vkWaitForFences";
        require(g_batchIdleCmd == VK_NULL_HANDLE && g_v41SubmissionPendingFailure == pending &&
                commands.size() == (pending ? 1u : 0u), "failed batch recycled command");
        const int callsBefore = calls[api]; batchFlush(); batchBegin();
        require(calls[api] == callsBefore, "uncertain batch retried");
    }
    reset(); g_stepBatch = true; batchBegin(); g_batchOps = 1;
    failure = "vkResetCommandBuffer";
    require(fvk_batch_flush_status() == VK_ERROR_OUT_OF_HOST_MEMORY,
            "flush status hid resumed reset failure");
    reset(); g_stepBatch = true; batchBegin();
    memory[statusHandle] = {0x42463143,1,UINT32_MAX,64};
    std::array<uint32_t,4> host{}; failure = "vkResetCommandBuffer";
    require(stepRecordReadback(host.data(), &status, 16) == VK_ERROR_OUT_OF_HOST_MEMORY,
            "readback hid resumed reset failure");
    reset(); g_stepBatch = true; batchBegin(); g_batchOps = 1; batchFlush();
    g_dev = handle<VkDevice>(901); g_cmdpool = handle<VkCommandPool>(902);
    discardIdleBatchCommand();
    require(commands.empty() && g_batchIdleCmd == VK_NULL_HANDLE &&
            g_batchIdleDevice == VK_NULL_HANDLE && g_batchIdlePool == VK_NULL_HANDLE,
            "old device/pool idle ownership retained");
    reset(); // legacy batches retain only after submitWait returns successfully.
    batchBegin(); g_batchOps = 1; batchFlush();
    batchBegin(); g_batchOps = 1; batchFlush();
    require(calls["vkAllocateCommandBuffers"] == 1 && calls["vkResetCommandBuffer"] == 1 &&
            calls["vkWaitForFences"] == 2 && calls["vkFreeCommandBuffers"] == 0,
            "legacy batch reuse changed completion ownership");
    discardIdleBatchCommand();
    reset(); g_stepBatch = true; batchBegin(); g_batchOps = 1; batchFlush();
    g_stepBatch = false; failure = "vkResetCommandBuffer";
    require(fvk_v41_step_batch_begin() == VK_ERROR_OUT_OF_HOST_MEMORY,
            "step begin hid reused-command failure");
    reset(); // isolated fixture teardown only
}
// Optional finite-only entrypoint, using the same real checked owner above.
// The mocked shader writes only its envelope; arithmetic is a device witness.
void finiteReset() {
    reset(); finiteDispatch = true;
    auto& k = g_kern[K_V41_F32_FINITE];
    k = {}; k.pipe = handle<VkPipeline>(50); k.dsl = handle<VkDescriptorSetLayout>(51);
    k.layout = handle<VkPipelineLayout>(52); k.nbuf = 2; k.pcsize = 8;
    checkedPipeline = k.pipe;
    memory[statusHandle] = {0x46333243, expectedSite, UINT32_MAX, expectedCount};
}
int finiteInvoke() { return fvk_v41_f32_finite(&input,&status,int(expectedCount),expectedSite); }
void finiteRelease(bool pending) {
    const auto before = calls;
    fvk_free(&status);
    require(calls == before, "finite release unexpectedly reached driver");
    if (pending) {
        require(g_batchFreed.size() == 1 && g_batchFreed[0] == &status && g_poolCount == 0,
                "finite uncertain status escaped quarantine");
        require(fvk_malloc(status.bytes) == nullptr, "finite quarantine reused");
    } else {
        require(g_batchFreed.empty() && g_poolCount == 1 &&
                fvk_malloc(status.bytes) == &status && g_poolCount == 0,
                "finite completed status not reusable");
    }
}
void finiteCases() {
    finiteReset();
    require(fvk_have_v41_f32_finite() && fvk_v41_f32_finite_count_supported(1) &&
            !fvk_v41_f32_finite_count_supported(0), "finite capability/count");
    g_maxStorageBufferRange = 15;
    require(!fvk_v41_f32_finite_count_supported(1), "finite status range missing");
    g_maxStorageBufferRange = 16;
    require(fvk_v41_f32_finite_count_supported(4) && !fvk_v41_f32_finite_count_supported(5),
            "finite input descriptor range missing");
    for (int bad = 0; bad < 10; ++bad) {
        finiteReset();
        switch (bad) {
        case 0: expectedSite = 3; break;
        case 1: expectedCount = 0; break;
        case 2: status.bytes = 15; break;
        case 3: input.bytes = 255; break;
        case 4: status.buf = input.buf; break;
        case 5: status.mem = input.mem; status.memoryOffset = input.bytes-1; break;
        case 6: status.memoryOffset = std::numeric_limits<VkDeviceSize>::max(); break;
        case 7: g_batching = true; break;
        case 8: g_kern[K_V41_F32_FINITE].pipe = VK_NULL_HANDLE; break;
        case 9: g_restore.stage = &staging; break;
        }
        require(finiteInvoke() != 0 && calls.empty() && fills == 0 && dispatches == 0,
                "finite refusal reached recording");
    }
    for (uint32_t site : {1u,2u}) for (bool batched : {false,true}) {
        finiteReset(); expectedSite = site; expectedCount = 193; expectedGroups = 2;
        memory[statusHandle] = {0x46333243,site,UINT32_MAX,expectedCount}; // stale success
        if (batched) require(fvk_v41_step_batch_begin() == 0, "finite step batch begin");
        require(finiteInvoke() == 0 && fills == 1 && dispatches == 1 && copies == 0,
                "finite record dispatch or premature host read");
        if (batched) require(calls["vkQueueSubmit"] == 0, "finite batch flushed early");
        std::array<uint32_t,4> host{};
        require(fvk_v41_d2h(host.data(),&status,16) == 0 &&
                host == std::array<uint32_t,4>{0x46333243,site,UINT32_MAX,expectedCount},
                "finite completion/status freshness");
        if (batched) require(fvk_v41_step_batch_end() == 0, "finite step batch end");
        require(g_d2hCount == 1 && g_d2hBytes == 16 && copies == 1,
                "finite readback included an input row");
        finiteRelease(false);
    }
    const char* APIs[] = {"vkAllocateDescriptorSets","vkAllocateCommandBuffers","vkBeginCommandBuffer",
                         "vkEndCommandBuffer","vkResetFences","vkQueueSubmit","vkWaitForFences"};
    for (const char* api : APIs) {
        finiteReset(); failure = api;
        const bool pending = failure == "vkQueueSubmit" || failure == "vkWaitForFences";
        failureResult = failure == "vkWaitForFences" ? VK_TIMEOUT : pending ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult expected = failureResult > VK_SUCCESS ? VK_ERROR_UNKNOWN : failureResult;
        require(finiteInvoke() == expected && g_submissionStatus == expected &&
                g_v41SubmissionPendingFailure == pending && calls[api] == 1,
                "finite selected failure identity");
        const auto before = calls;
        require(finiteInvoke() == expected && calls == before, "finite selected error replayed");
        finiteRelease(pending);
    }
    // Interruption has no returned completion proof. Poison before retiring
    // known handles, and never overwrite an already observed primary fault.
    finiteReset(); fvk_v41_f32_finite_abandon();
    require(g_submissionStatus == VK_ERROR_UNKNOWN && g_v41SubmissionPendingFailure,
            "finite interruption failed to poison the shared execution domain");
    finiteRelease(true);
    require(finiteInvoke() == VK_ERROR_UNKNOWN && calls.empty(), "abandon allowed replay");
    finiteReset(); g_submissionStatus = VK_ERROR_DEVICE_LOST;
    fvk_v41_f32_finite_abandon();
    require(g_submissionStatus == VK_ERROR_DEVICE_LOST && g_v41SubmissionPendingFailure,
            "abandon replaced primary failure");
    finiteRelease(true);
    reset(); // synthetic fixture teardown only
}
} // namespace checked_fixture
int main() {
    checked_fixture::geometryCases();
    checked_fixture::successCases();
    checked_fixture::failures();
    checked_fixture::batchReuseCases();
    checked_fixture::finiteCases();
    std::puts("PASS real checked owner: initialization, dependency, refusal, transport and quarantine; no physical/numerical claim");
    return 0;
}
`
