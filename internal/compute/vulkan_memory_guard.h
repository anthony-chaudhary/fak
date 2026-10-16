#ifndef FAK_VULKAN_MEMORY_GUARD_H
#define FAK_VULKAN_MEMORY_GUARD_H

#include <cstdint>
#include <limits>

// Called under vulkanMu, including snapshots. Reservations count actual
// VkDeviceMemory bytes across all heaps, not buffer bindings or driver overhead.
struct VulkanMemoryGuard {
    uint64_t limit = 0, live = 0, peak = 0, refused = 0;
    bool initialized = false, configured = false, configurationValid = false;
    bool accountingValid = true;

    bool configure(const char* value) {
        if (initialized) return configurationValid;
        initialized = true;
        configured = value != nullptr;
        configurationValid = !configured;
        if (!configured) return true;
        if (!*value) return false;
        uint64_t parsed = 0;
        const uint64_t maximum = uint64_t(std::numeric_limits<int64_t>::max());
        for (const char* p = value; *p; ++p) {
            if (*p < '0' || *p > '9') return false;
            uint64_t digit = uint64_t(*p - '0');
            if (parsed > (maximum - digit) / 10) return false;
            parsed = parsed * 10 + digit;
        }
        if (!parsed) return false;
        limit = parsed;
        configurationValid = true;
        return true;
    }

    bool reserve(uint64_t bytes) {
        if (!initialized || !configurationValid || !accountingValid || !bytes ||
            bytes > std::numeric_limits<uint64_t>::max() - live ||
            (configured && (live > limit || bytes > limit - live))) {
            if (refused != std::numeric_limits<uint64_t>::max()) ++refused;
            return false;
        }
        live += bytes;
        return true;
    }

    void committed() {
        if (live > peak) peak = live;
    }

    void release(uint64_t bytes) {
        if (!accountingValid) return;
        if (!bytes || bytes > live) {
            accountingValid = false;
            return;
        }
        live -= bytes;
    }
};

#endif
