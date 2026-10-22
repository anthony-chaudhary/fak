#ifndef FAK_VULKAN_SPIRV_ADMISSION_H
#define FAK_VULKAN_SPIRV_ADMISSION_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

// Pure byte admission: no Vulkan handles, device state, file I/O or initialization.
// These checks establish narrow ABI properties, never numerical GPU qualification.
namespace fak::vulkan::spirv {

using std::size_t;
using std::memcpy;

inline bool swigluPushConstantABI(const std::vector<char>& code) {
    if (code.size() < 5 * sizeof(uint32_t) || code.size() % sizeof(uint32_t) != 0) return false;
    std::vector<uint32_t> words(code.size() / sizeof(uint32_t));
    memcpy(words.data(), code.data(), code.size());
    if (words[0] != 0x07230203 || words[3] == 0 || words[4] != 0) return false;
    const uint32_t bound = words[3];
    const uint32_t opTypeInt = 21, opTypeFloat = 22, opTypeStruct = 30, opTypePointer = 32;
    const uint32_t opVariable = 59, opDecorate = 71, opMemberDecorate = 72;
    const uint32_t pushConstant = 9, blockDecoration = 2, offsetDecoration = 35;
    std::unordered_map<uint32_t, std::vector<uint32_t>> definitions;
    std::unordered_map<uint32_t, bool> blocks;
    std::unordered_map<uint64_t, uint32_t> offsets;
    uint32_t pushPointer = 0;
    size_t pushVariables = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    for (size_t pos = 5; pos < words.size();) {
        const uint32_t count = words[pos] >> 16;
        const uint32_t op = words[pos] & 0xffff;
        if (count == 0 || count > words.size() - pos) return false;
        const uint32_t* instruction = words.data() + pos;
        if (op == opTypeInt || op == opTypeFloat || op == opTypeStruct || op == opTypePointer || op == opVariable) {
            if ((op == opTypeInt && count != 4) || (op == opTypeFloat && count != 3) ||
                (op == opTypeStruct && count < 2) || (op == opTypePointer && count != 4) ||
                (op == opVariable && count != 4 && count != 5)) return false;
            const uint32_t id = instruction[op == opVariable ? 2 : 1];
            if (!validID(id) || !definitions.emplace(id, std::vector<uint32_t>(instruction, instruction + count)).second) return false;
            if (op == opTypeStruct) {
                for (uint32_t member = 2; member < count; ++member) if (!validID(instruction[member])) return false;
            } else if (op == opTypePointer) {
                if (!validID(instruction[3])) return false;
            } else if (op == opVariable) {
                if (!validID(instruction[1])) return false;
                if (instruction[3] == pushConstant) { pushPointer = instruction[1]; ++pushVariables; }
            }
        } else if (op == opDecorate) {
            if (count < 3 || !validID(instruction[1])) return false;
            if (instruction[2] == blockDecoration) {
                if (count != 3 || !blocks.emplace(instruction[1], true).second) return false;
            }
        } else if (op == opMemberDecorate) {
            if (count < 4 || !validID(instruction[1])) return false;
            if (instruction[3] == offsetDecoration) {
                const uint64_t key = (uint64_t(instruction[1]) << 32) | instruction[2];
                if (count != 5 || !offsets.emplace(key, instruction[4]).second) return false;
            }
        }
        pos += count;
    }
    if (pushVariables != 1) return false;
    auto pointer = definitions.find(pushPointer);
    if (pointer == definitions.end() || (pointer->second[0] & 0xffff) != opTypePointer || pointer->second[2] != pushConstant) return false;
    const uint32_t structID = pointer->second[3];
    auto structure = definitions.find(structID);
    if (structure == definitions.end() || (structure->second[0] & 0xffff) != opTypeStruct || structure->second.size() != 4 || blocks.count(structID) != 1) return false;
    auto integer = definitions.find(structure->second[2]);
    auto floating = definitions.find(structure->second[3]);
    if (integer == definitions.end() || (integer->second[0] & 0xffff) != opTypeInt || integer->second[2] != 32 || integer->second[3] != 1) return false;
    if (floating == definitions.end() || (floating->second[0] & 0xffff) != opTypeFloat || floating->second[2] != 32) return false;
    const uint64_t offsetKey = uint64_t(structID) << 32;
    auto first = offsets.find(offsetKey), second = offsets.find(offsetKey | 1);
    return first != offsets.end() && first->second == 0 && second != offsets.end() && second->second == 4;
}

inline bool v41TailRoPEABI(const std::vector<char>& code, bool strict = false) {
    if (strict && code.size() > 1024 * 1024) return false;
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16 && count >= 3 && a[2] == 17) { // LocalSize
            if (count != 6 || a[3] != 256 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 129 || op == 131 || op == 133) { // FAdd / FSub / FMul
            if (count != 5 || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
        } else if (op == 12) { // no extended arithmetic, especially Fma/trig
            return false;
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 5 || floatResults.size() < 6) return false;
    for (uint32_t id : floatResults) if (!decorated(id, noContraction, 0)) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 5);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 3; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    // Strict mode extends this existing ABI proof on the exact bytes passed
    // to vkCreateShaderModule. Ordinary callers retain their historical policy.
    // This is a bounded shape/dataflow check, not a full validator or provenance
    // attestation; independently reviewed source and actual compiler output stay
    // separate witness requirements.
    std::unordered_map<uint64_t, bool> variableAccess;
    if (strict) {
        uint32_t modes = 0, capabilities = 0, modeEntry = 0;
        size_t extensions = 0, floatTypes = 0;
        std::unordered_map<uint32_t, std::vector<uint32_t>> values;
        std::unordered_map<uint32_t, uint32_t> stores;
        std::vector<uint32_t> products;
        uint32_t subtract = 0, add = 0;
        for (size_t p = 5; p < w.size();) {
            const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
            const uint32_t* a = w.data() + p; // framing already checked above
            switch (op) {
            case 0: case 3: case 4: case 5: case 6: case 7: case 8: case 10: case 11:
            case 14: case 15: case 16: case 17:
            case 19: case 20: case 21: case 22: case 23: case 29: case 30: case 32: case 33:
            case 41: case 42: case 43: case 44: case 46:
            case 54: case 56: case 59: case 61: case 62: case 65: case 66:
            case 71: case 72: case 80: case 81: case 83: case 124:
            case 128: case 130: case 132: case 137: // integer index arithmetic
            case 129: case 131: case 133: // exactly four FMul, one FSub, one FAdd
            case 168: case 169: case 170: case 171: case 174: case 176: case 199:
            case 245: case 247: case 248: case 249: case 250: case 253: case 255:
            case 317: case 330:
                break;
            default:
                return false;
            }
            if (op == 16) {
                if (count < 3 || !validID(a[1])) return false;
                if (a[2] != 17) {
                    const uint32_t bit = a[2] == 4459 ? 1u : a[2] == 4461 ? 2u : a[2] == 4462 ? 4u : 0u;
                    if (!bit || count != 4 || a[3] != 32 || (modes & bit) ||
                        (modeEntry != 0 && modeEntry != a[1])) return false;
                    modes |= bit; modeEntry = a[1];
                }
            } else if (op == 17) {
                if (count != 2) return false;
                const uint32_t bit = a[1] == 1 ? 1u : a[1] == 4464 ? 2u : a[1] == 4466 ? 4u : a[1] == 4467 ? 8u : 0u;
                if (!bit || (capabilities & bit)) return false;
                capabilities |= bit;
            } else if (op == 10) {
                const char expected[] = "SPV_KHR_float_controls";
                if (count != 1 + (sizeof(expected) + 3) / 4 ||
                    memcmp(a + 1, expected, sizeof(expected)) != 0 || ++extensions != 1) return false;
            } else if (op == typeFloat) {
                if (a[2] != 32) return false;
                ++floatTypes;
            } else if (op == typeInt) {
                if (a[2] != 32 || a[3] > 1) return false;
            } else if (op == 23) {
                if (count != 4 || !definition(a[2], typeInt, 4)) return false;
            } else if (op == 71 || op == 72) {
                const uint32_t kind = a[op == 71 ? 2 : 3];
                if (kind == 0 || kind == 1 || kind == 39 || kind == 40) return false;
                if (op == 71 && (kind == 24 || kind == 25)) {
                    if (count != 3 || !variableAccess.emplace(key(a[1], kind), true).second) return false;
                }
            }
            if (op == 61 || op == 83 || op == 129 || op == 131 || op == 133) {
                if ((op == 61 && count != 4) || (op == 83 && count != 4) ||
                    ((op == 129 || op == 131 || op == 133) && count != 5) ||
                    !validID(a[1]) || !validID(a[2]) || !validID(a[3]) ||
                    !values.emplace(a[2], std::vector<uint32_t>(a, a + count)).second) return false;
            }
            if (op == 62) {
                if (count != 3 || !validID(a[1]) || !validID(a[2])) return false;
                // Ambiguous stores cannot be used to prove scalar dataflow.
                if (!stores.emplace(a[1], a[2]).second) stores[a[1]] = 0;
            }
            if (op == 129 || op == 131 || op == 133) {
                const auto* scalar = definition(a[1], typeFloat, 3);
                if (!scalar || (*scalar)[2] != 32 || !validID(a[4])) return false;
                if (op == 133) products.push_back(a[2]);
                else if (op == 131) { if (subtract) return false; subtract = a[2]; }
                else { if (add) return false; add = a[2]; }
            }
            p += count;
        }
        if (modes != 7 || capabilities != 15 || extensions != 1 || modeEntry != entry ||
            floatTypes != 1 || products.size() != 4 || !subtract || !add || floatResults.size() != 6) return false;
        // Follow single-assignment function locals and CopyObject aliases. This
        // binds the two sums to their four products while allowing unoptimized
        // GLSL output; table/input origin and complete program validity remain
        // independently reviewed compiler/source evidence.
        auto origin = [&](uint32_t id) {
            for (unsigned depth = 0; depth < 32; ++depth) {
                auto it = values.find(id);
                if (it == values.end()) return id;
                const auto& v = it->second;
                const uint32_t op = v[0] & 0xffff;
                if (op == 83) { id = v[3]; continue; }
                if (op != 61) return id;
                auto store = stores.find(v[3]);
                if (store == stores.end()) return id;
                if (!store->second) return uint32_t(0);
                id = store->second;
            }
            return uint32_t(0);
        };
        auto product = [&](uint32_t id) -> const std::vector<uint32_t>* {
            auto it = values.find(origin(id));
            return it != values.end() && (it->second[0] & 0xffff) == 133 ? &it->second : nullptr;
        };
        const auto& sub = values.at(subtract);
        const auto& sum = values.at(add);
        const auto* ac = product(sub[3]); const auto* bs = product(sub[4]);
        const auto* bc = product(sum[3]); const auto* as = product(sum[4]);
        if (!ac || !bs || !bc || !as || ac == bs || ac == bc || ac == as || bs == bc || bs == as || bc == as) return false;
        const uint32_t a = origin((*ac)[3]), c = origin((*ac)[4]);
        const uint32_t b = origin((*bs)[3]), sine = origin((*bs)[4]);
        if (!a || !b || !c || !sine || a == b || a == c || a == sine || b == c || b == sine || c == sine ||
            origin((*bc)[3]) != b || origin((*bc)[4]) != c ||
            origin((*as)[3]) != a || origin((*as)[4]) != sine) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 5 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], slot == 4 ? typeFloat : typeInt, slot == 4 ? 3 : 4);
        if (!scalar || (*scalar)[2] != 32 || (slot != 4 && (*scalar)[3] != 0)) return false;
        const uint32_t access = slot == 2 || slot == 3 ? 25 : 24;
        if (memberAccess.count(key(structID, access)) != 1) return false;
        if (strict) {
            const bool nonWritable = memberAccess.count(key(structID, 24)) || variableAccess.count(key(id, 24));
            const bool nonReadable = memberAccess.count(key(structID, 25)) || variableAccess.count(key(id, 25));
            if (nonWritable != (slot == 0 || slot == 1 || slot == 4) ||
                nonReadable != (slot == 2 || slot == 3)) return false;
        }
    }
    return seenBindings == 31;
}

// Narrow fixed attention ABI/NoContraction check, not a receipt verifier or
// full SPIR-V semantic validator. The source-pinned V4 bundle is a separate gate.
inline bool v41SharedAttentionABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults, expImports;
    uint32_t glslImport = 0;
    size_t glslImportCount = 0;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == 0 || kind == 40) return false; // RelaxedPrecision / FPFastMathMode
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16 && count >= 3 && a[2] == 17) { // LocalSize
            if (count != 6 || a[3] != 128 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 129 || op == 131 || op == 133 || op == 136) { // scalar FAdd/FSub/FMul/FDiv
            if (count != 5 || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
        } else if (op == 11) { // one explicit GLSL.std.450 import
            if (count != 6 || !validID(a[1]) || a[2] != 0x4c534c47 ||
                a[3] != 0x6474732e || a[4] != 0x3035342e || a[5] != 0) return false;
            glslImport = a[1]; ++glslImportCount;
        } else if (op == 12) { // only Exp; never extended Fma or reductions
            if (count != 6 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) ||
                a[4] != 27 || !validID(a[5])) return false;
            expImports.push_back(a[3]);
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 5 || floatResults.size() < 6 ||
        glslImportCount != 1 || expImports.empty()) return false;
    for (uint32_t id : expImports) if (id != glslImport) return false;
    for (uint32_t id : floatResults) if (!decorated(id, noContraction, 0)) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 8);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 5; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    const auto* scale = definition((*structure)[7], typeFloat, 3);
    if (!scale || (*scale)[2] != 32 || !atOffset(pushStruct, 5, 20)) return false;
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 5 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], slot == 4 ? typeInt : typeFloat, slot == 4 ? 4 : 3);
        if (!scalar || (*scalar)[2] != 32 || (slot == 4 && (*scalar)[3] != 0)) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        // Q/KV/sink are read-only; output is read/write for slot-ordered
        // accumulation; status is write-only and read back by the host.
        if (nonWritable != (slot < 3) || nonReadable != (slot == 4)) return false;
    }
    return seenBindings == 31;
}

// Narrow source-bound indexer ABI and binary32-mode admission. This is not a
// general SPIR-V validator or proof of algorithm semantics: the separately
// versioned V5 source/archive/module receipt and physical witness remain required.
// Numeric mode/capability values are from Khronos SPV_KHR_float_controls.
inline bool v41IndexerScoreABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults, floatTypes;
    std::vector<std::pair<uint32_t, uint32_t>> floatResultTypes;
    uint32_t modes = 0, capabilities = 0, modeEntry = 0;
    size_t floatControlExtensions = 0, floatAdds = 0, floatMuls = 0;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeFloat) {
                if (a[2] != 32) return false;
                floatTypes.push_back(id);
            }
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == 0 || kind == 1 || kind == 39 || kind == 40) return false; // no relaxed/rounding/fast-math overrides
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == 0 || a[3] == 39 || a[3] == 40) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) { // OpExecutionMode, fixed entry/modes/width
            if (count < 3 || !validID(a[1])) return false;
            if (a[2] == 17) {
                if (count != 6 || a[3] != 64 || a[4] != 1 || a[5] != 1) return false;
                localEntry = a[1]; ++localCount;
            } else {
                // SPV_KHR_float_controls: DenormPreserve, SignedZeroInfNanPreserve,
                // RoundingModeRTE. Defaults and conflicting/unknown modes fail closed.
                uint32_t bit = a[2] == 4459 ? 1u : a[2] == 4461 ? 2u : a[2] == 4462 ? 4u : 0u;
                if (bit == 0 || count != 4 || a[3] != 32 || (modes & bit) ||
                    (modeEntry != 0 && modeEntry != a[1])) return false;
                modes |= bit; modeEntry = a[1];
            }
        } else if (op == 17) { // only Shader and the three required float-control capabilities
            if (count != 2) return false;
            uint32_t bit = a[1] == 1 ? 1u : a[1] == 4464 ? 2u : a[1] == 4466 ? 4u : a[1] == 4467 ? 8u : 0u;
            if (bit == 0 || (capabilities & bit)) return false;
            capabilities |= bit;
        } else if (op == 10) { // explicit float-controls extension, no unknown module extensions
            const char expected[] = "SPV_KHR_float_controls";
            if (count != 1 + (sizeof(expected) + 3) / 4 ||
                memcmp(a + 1, expected, sizeof(expected)) != 0 || ++floatControlExtensions != 1) return false;
        } else if (op == 129 || op == 133) { // separately rounded scalar FAdd / FMul only
            if (count != 5 || !validID(a[1]) || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
            if (op == 129) ++floatAdds; else ++floatMuls;
            floatResultTypes.emplace_back(a[1], a[2]);
        } else if (op == 12 || (op >= 48 && op <= 52) || (op >= 109 && op <= 112) || op == 115 || op == 116 ||
                   op == 127 || op == 131 || (op >= 136 && op <= 148) ||
                   op == 24 || (op >= 73 && op <= 75) || op == 331 || op == 332 ||
                   op == 265 || op == 266 || op == 269 || op == 350 || op == 352 || op == 355 || op == 358) {
            // No extended Fma/reductions, floating conversions/quantization, matrices,
            // decoration groups or ID-based execution modes/decorations.
            return false;
        } else if (op == 23) { // only integer builtin ID vectors (workgroup, local invocation) are needed
            if (count != 4 || !validID(a[1]) || !validID(a[2])) return false;
            auto scalar = defs.find(a[2]);
            if (scalar == defs.end() || (scalar->second[0] & 0xffff) != typeInt) return false;
        }

        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 4 || floatResults.size() != 4 ||
        floatAdds != 2 || floatMuls != 2 || modes != 7 || capabilities != 15 || floatControlExtensions != 1 || modeEntry != entry || floatTypes.size() != 1) return false;
    for (const auto& result : floatResultTypes) {
        const auto* scalar = definition(result.first, typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32 || !decorated(result.second, noContraction, 0)) return false;
    }
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 5);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 3; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 4 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot < 3) || nonReadable != (slot == 3)) return false;
    }
    return seenBindings == 15;
}

// Reject only malformed SPIR-V framing, not invalid instructions or unsupported
// capabilities. SPIR-V physical layout requires five header words; its magic
// identifies either byte order. Leave version policy and code bytes unchanged.
// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_physical_layout_of_a_spir_v_module_and_instruction
// Fixed integer-only BF16 ABI admission, not a semantic/provenance proof.
// A separately reviewed source/module/archive witness is still mandatory.
inline bool v41BF16BoundaryABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0, shaderCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        // No float type or extended instruction can enter the raw bit path.
        if (op == typeFloat || op == 12) return false;
        if (op == typeInt || op == typeRuntimeArray || op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeRuntimeArray && count != 3) ||
                (op == typeStruct && count < 2) || (op == typePointer && count != 4) ||
                (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeInt && (a[2] != 32 || a[3] > 1)) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false;
            }
        } else if (op == 71) {
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == block) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) {
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) {
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) {
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) {
            if (count != 6 || !validID(a[1]) || a[2] != 17 || a[3] != 64 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 17) {
            if (count != 2 || a[1] != 1) return false; // only Shader
            ++shaderCount;
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || shaderCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 2) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 3);
    if (!structure || !decorated(pushStruct, block, 0) || !atOffset(pushStruct, 0, 0)) return false;
    const auto* countType = definition((*structure)[2], typeInt, 4);
    if (!countType || (*countType)[2] != 32 || (*countType)[3] != 0) return false;
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 2 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeInt, 4);
        if (!scalar || (*scalar)[2] != 32 || (*scalar)[3] != 0) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot == 0) || nonReadable != (slot == 1)) return false;
    }
    return seenBindings == 3;
}

// Optional finite BF16 publication: three disjoint uint buffers, count/site
// push, one unsigned invalid-index reduction. Structural admission is separate
// from actual SPIR-V validation, reviewed source/module provenance and hardware
// qualification. The original raw BF16 ABI and all-bit semantics stay unchanged.
inline bool v41BF16CheckedABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0, shaderCount = 0;
    std::vector<uint32_t> atomicResultTypes;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        // Integer-only publication: no floating or extended instruction.
        // Exactly one unsigned atomic minimum reduces the invalid index.
        if (op == typeFloat || op == 12) return false;
        if ((op >= 227 && op <= 242 && op != 237) || op == 318 || op == 319) return false;
        if (op == 237) { // OpAtomicUMin
            if (count != 7) return false;
            for (uint32_t i = 1; i < count; ++i) if (!validID(a[i])) return false;
            atomicResultTypes.push_back(a[1]);
        }
        if (op == typeInt || op == typeRuntimeArray || op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeRuntimeArray && count != 3) ||
                (op == typeStruct && count < 2) || (op == typePointer && count != 4) ||
                (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeInt && (a[2] != 32 || a[3] > 1)) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false;
            }
        } else if (op == 71) {
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == block || kind == 24 || kind == 25) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) {
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) {
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) {
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) {
            if (count != 6 || !validID(a[1]) || a[2] != 17 || a[3] != 64 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 17) {
            if (count != 2 || a[1] != 1) return false; // only Shader
            ++shaderCount;
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || shaderCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 3 || atomicResultTypes.size() != 1) return false;
    const auto* atomicType = definition(atomicResultTypes[0], typeInt, 4);
    if (!atomicType || (*atomicType)[2] != 32 || (*atomicType)[3] != 0) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 4);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t member = 0; member < 2; ++member) {
        const auto* scalar = definition((*structure)[member + 2], typeInt, 4);
        if (!scalar || (*scalar)[2] != 32 || (*scalar)[3] != 0 ||
            !atOffset(pushStruct, member, member * 4)) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 3 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeInt, 4);
        if (!scalar || (*scalar)[2] != 32 || (*scalar)[3] != 0) return false;
        // SPIR-V may put access restrictions on the variable, its block
        // member, or both. Enforce the effective role, including contradictions.
        const bool nonWritable = decorated(id, 24, 0) || memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = decorated(id, 25, 0) || memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot == 0) || nonReadable != (slot == 1)) return false;
    }
    return seenBindings == 7;
}

// Optional finite-only ABI: two disjoint uint buffers, no output publication.
// Structural admission is separate from registry/source identity and device qualification.
inline bool v41F32FiniteABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0, shaderCount = 0;
    std::vector<uint32_t> atomicResultTypes;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        // Integer-only raw finiteness check: no floating or extended instruction.
        // Exactly one unsigned atomic minimum reduces the invalid index.
        if (op == typeFloat || op == 12) return false;
        if ((op >= 227 && op <= 242 && op != 237) || op == 318 || op == 319) return false;
        if (op == 237) { // OpAtomicUMin
            if (count != 7) return false;
            for (uint32_t i = 1; i < count; ++i) if (!validID(a[i])) return false;
            atomicResultTypes.push_back(a[1]);
        }
        if (op == typeInt || op == typeRuntimeArray || op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeRuntimeArray && count != 3) ||
                (op == typeStruct && count < 2) || (op == typePointer && count != 4) ||
                (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeInt && (a[2] != 32 || a[3] > 1)) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false;
            }
        } else if (op == 71) {
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == block || kind == 24 || kind == 25) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) {
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) {
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) {
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) {
            if (count != 6 || !validID(a[1]) || a[2] != 17 || a[3] != 64 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 17) {
            if (count != 2 || a[1] != 1) return false; // only Shader
            ++shaderCount;
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || shaderCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 2 || atomicResultTypes.size() != 1) return false;
    const auto* atomicType = definition(atomicResultTypes[0], typeInt, 4);
    if (!atomicType || (*atomicType)[2] != 32 || (*atomicType)[3] != 0) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 4);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t member = 0; member < 2; ++member) {
        const auto* scalar = definition((*structure)[member + 2], typeInt, 4);
        if (!scalar || (*scalar)[2] != 32 || (*scalar)[3] != 0 ||
            !atOffset(pushStruct, member, member * 4)) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 2 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeInt, 4);
        if (!scalar || (*scalar)[2] != 32 || (*scalar)[3] != 0) return false;
        // SPIR-V may put access restrictions on the variable, its block
        // member, or both. Enforce the effective role, including contradictions.
        const bool nonWritable = decorated(id, 24, 0) || memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = decorated(id, 25, 0) || memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot == 0) || nonReadable) return false;
    }
    return seenBindings == 3;
}

// Witness-only scalar ABI and declared floating-mode admission. This parser
// checks shape, one NoContraction FMul and the required binary32 controls;
// it is not a general SPIR-V validator or an algorithm/provenance proof.
// Source/module/archive hashes and physical witnesses remain independently required.
// https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls.html
inline bool v41LogitScaleABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults, floatTypes;
    std::vector<std::pair<uint32_t, uint32_t>> floatResultTypes;
    uint32_t modes = 0, capabilities = 0, modeEntry = 0;
    size_t floatControlExtensions = 0;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        // Closed instruction family for this one-product source: no implicit
        // permission for newer floating opcodes, helpers, atomics or reductions.
        // ABI acceptance remains separate from validation of the actual module
        // and review of its algorithm and immutable source/module provenance.
        switch (op) {
        case 0: case 3: case 4: case 5: case 6: case 7: case 8: case 10: case 11:
        case 14: case 15: case 16: case 17: // debug/module declarations
        case 19: case 20: case 21: case 22: case 23: case 29: case 30: case 32: case 33:
        case 41: case 42: case 43: case 44: case 46: // types and ordinary constants
        case 54: case 56: case 59: case 61: case 62: case 65: case 66:
        case 71: case 72: case 80: case 81: case 83: // memory/shape, no conversion
        case 133: case 174: // scalar FMul and unsigned bounds comparison
        case 245: case 247: case 248: case 249: case 250: case 253: case 255:
        case 317: case 330: // control flow and debug annotations
            break;
        default:
            return false;
        }
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeInt && (a[2] != 32 || a[3] > 1)) return false;
            if (op == typeFloat) {
                if (a[2] != 32) return false;
                floatTypes.push_back(id);
            }
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == 0 || kind == 1 || kind == 39 || kind == 40) return false; // no relaxed/rounding/fast-math overrides
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == 0 || a[3] == 39 || a[3] == 40) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) { // OpExecutionMode, fixed entry/modes/width
            if (count < 3 || !validID(a[1])) return false;
            if (a[2] == 17) {
                if (count != 6 || a[3] != 64 || a[4] != 1 || a[5] != 1) return false;
                localEntry = a[1]; ++localCount;
            } else {
                // SPV_KHR_float_controls: DenormPreserve, SignedZeroInfNanPreserve,
                // RoundingModeRTE. Defaults and conflicting/unknown modes fail closed.
                uint32_t bit = a[2] == 4459 ? 1u : a[2] == 4461 ? 2u : a[2] == 4462 ? 4u : 0u;
                if (bit == 0 || count != 4 || a[3] != 32 || (modes & bit) ||
                    (modeEntry != 0 && modeEntry != a[1])) return false;
                modes |= bit; modeEntry = a[1];
            }
        } else if (op == 17) { // only Shader and the three required float-control capabilities
            if (count != 2) return false;
            uint32_t bit = a[1] == 1 ? 1u : a[1] == 4464 ? 2u : a[1] == 4466 ? 4u : a[1] == 4467 ? 8u : 0u;
            if (bit == 0 || (capabilities & bit)) return false;
            capabilities |= bit;
        } else if (op == 10) { // explicit float-controls extension, no unknown module extensions
            const char expected[] = "SPV_KHR_float_controls";
            if (count != 1 + (sizeof(expected) + 3) / 4 ||
                memcmp(a + 1, expected, sizeof(expected)) != 0 || ++floatControlExtensions != 1) return false;
        } else if (op == 133) { // exactly one separately rounded scalar FMul
            if (count != 5 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) || !validID(a[4])) return false;
            floatResults.push_back(a[2]);
            floatResultTypes.emplace_back(a[1], a[2]);
        } else if (op == 23) { // only the integer GlobalInvocationID vector is needed
            if (count != 4 || !validID(a[1]) || !validID(a[2])) return false;
            auto scalar = defs.find(a[2]);
            if (scalar == defs.end() || (scalar->second[0] & 0xffff) != typeInt) return false;
        }

        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 2 || floatResults.size() != 1 ||
        modes != 7 || capabilities != 15 || floatControlExtensions != 1 || modeEntry != entry || floatTypes.size() != 1) return false;
    for (const auto& result : floatResultTypes) {
        const auto* scalar = definition(result.first, typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32 || !decorated(result.second, noContraction, 0)) return false;
    }
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 4);
    if (!structure || !decorated(pushStruct, block, 0) ||
        !atOffset(pushStruct, 0, 0) || !atOffset(pushStruct, 1, 4)) return false;
    const auto* countType = definition((*structure)[2], typeInt, 4);
    const auto* scaleType = definition((*structure)[3], typeFloat, 3);
    if (!countType || (*countType)[2] != 32 || (*countType)[3] != 0 ||
        !scaleType || (*scaleType)[2] != 32) return false;
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 2 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot == 0) || nonReadable != (slot == 1)) return false;
    }
    return seenBindings == 3;
}

inline bool spirvModuleFraming(const std::vector<char>& code) {
    if (code.size() < 5 * sizeof(uint32_t) || code.size() % sizeof(uint32_t) != 0) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, code.data(), sizeof(magic));
    return magic == 0x07230203u || magic == 0x03022307u;
}

// Return 0 for an unmarked legacy module, 1 for the expected grid ABI, and -1
// for malformed or unknown marked modules. This identifies the exact loaded
// bytes, not a filename or source freshness assumption. No specialization
// override is supplied by buildKernel, so the encoded default is authoritative.
inline int v41TokenGridModuleABI(const std::vector<char>& code, uint32_t expectedMarker) {
    if (code.size() < 20 || code.size() % 4 != 0) return -1;
    const size_t words = code.size() / 4;
    const auto word = [&](size_t i) {
        uint32_t value;
        std::memcpy(&value, code.data() + i * 4, 4);
        return value;
    };
    if (word(0) != 0x07230203u || word(3) == 0) return -1;
    uint32_t marker = 0;
    for (size_t i = 5; i < words;) {
        const uint32_t n = word(i) >> 16, op = word(i) & 65535u;
        if (n == 0 || n > words - i) return -1;
        // OpDecorate <id> SpecId 31.
        if (op == 71u && n >= 3u && word(i + 2) == 1u) {
            if (n != 4u) return -1;
            if (word(i + 3) == 31u) {
                if (marker != 0 || word(i + 1) == 0 || word(i + 1) >= word(3)) return -1;
                marker = word(i + 1);
            }
        }
        i += n;
    }
    if (marker == 0) return 0;
    uint32_t type = 0;
    for (size_t i = 5; i < words; i += word(i) >> 16) {
        const uint32_t n = word(i) >> 16, op = word(i) & 65535u;
        // OpSpecConstant <uint32 type> <marker id> family-specific grid ABI.
        if (op == 50u && n >= 3u && word(i + 2) == marker) {
            if (type != 0 || n != 4u || word(i + 3) != expectedMarker ||
                word(i + 1) == 0 || word(i + 1) >= word(3)) return -1;
            type = word(i + 1);
        }
    }
    if (type == 0) return -1;
    bool uint32Type = false;
    for (size_t i = 5; i < words; i += word(i) >> 16) {
        const uint32_t n = word(i) >> 16, op = word(i) & 65535u;
        if (op == 21u && n >= 2u && word(i + 1) == type) {
            if (uint32Type || n != 4u || word(i + 2) != 32u || word(i + 3) != 0u) return -1;
            uint32Type = true;
        }
    }
    return uint32Type ? 1 : -1;
}

} // namespace fak::vulkan::spirv

#endif
