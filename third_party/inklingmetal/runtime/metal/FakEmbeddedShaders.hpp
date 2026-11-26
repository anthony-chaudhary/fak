// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "MetalBackend.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <mutex>
#include <string>
#include <string_view>

namespace fak::inkling::shaders {
struct Slice { const unsigned char *data; size_t size; };
struct Unit { const char *name; const Slice *parts; size_t count; };
struct Export { std::string_view name; size_t unit; };
#include "FakShaderIndex.inc"

// Metal headers are provided by the framework. Local headers are included as
// slices of the vendored files, once per translation unit in source order.
class Libraries final {
public:
    static bool has(std::string_view name) { return locate(name) != nullptr; }
    // The translation unit that exports `name`; SIZE_MAX for none.
    static size_t unitOf(std::string_view name) {
        const Export *entry = locate(name);
        return entry ? entry->unit : SIZE_MAX;
    }
    // FNV-1a over every unit's name and source bytes and the language
    // version they compile at: any edit to an embedded shader changes it.
    static uint64_t fingerprint() noexcept {
        uint64_t hash = 0xcbf29ce484222325ull;
        const auto mix = [&](const void *data, size_t size) {
            for (size_t k = 0; k < size; ++k)
                hash = (hash ^ static_cast<const unsigned char *>(data)[k]) * 0x100000001b3ull;
        };
        const auto version = static_cast<uint64_t>(MTLLanguageVersion4_0);
        mix(&version, sizeof version);
        for (const Unit &unit : units) {
            mix(unit.name, std::char_traits<char>::length(unit.name) + 1);
            for (size_t j = 0; j < unit.count; ++j) mix(unit.parts[j].data, unit.parts[j].size);
        }
        return hash;
    }
    id<MTLLibrary> forFunction(id<MTLDevice> device, std::string_view function) {
        const Export *entry = locate(function);
        if (!entry) throw splash::metal::MetalBackendError("missing embedded shader: " + std::string(function));
        const size_t i = entry->unit;
        std::call_once(compiled_[i], [&] {
            const auto &unit = units[i];
            std::string source;
            for (size_t j = 0; j < unit.count; ++j)
                source.append(reinterpret_cast<const char *>(unit.parts[j].data), unit.parts[j].size);
            NSString *text = [[NSString alloc] initWithBytes:source.data() length:source.size()
                                                   encoding:NSUTF8StringEncoding];
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            NSError *error = nil;
            libraries_[i] = [device newLibraryWithSource:text options:options error:&error];
            if (!libraries_[i])
                throw splash::metal::MetalBackendError(std::string(unit.name) + ": " +
                    (error.localizedDescription.UTF8String ?: "Metal source compilation failed"));
        });
        return libraries_[i];
    }
private:
    static const Export *locate(std::string_view name) {
        const auto *entry = std::lower_bound(std::begin(exports), std::end(exports), name,
                [](const Export &candidate, std::string_view key) { return candidate.name < key; });
        return entry != std::end(exports) && entry->name == name ? entry : nullptr;
    }
    std::array<id<MTLLibrary>, std::size(units)> libraries_{};
    std::array<std::once_flag, std::size(units)> compiled_{};
};
} // namespace fak::inkling::shaders
