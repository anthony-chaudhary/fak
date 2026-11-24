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
