#ifndef FAK_VULKAN_BATCH_RANGE_HAZARDS_H
#define FAK_VULKAN_BATCH_RANGE_HAZARDS_H

#include <cstddef>
#include <cstdint>

namespace fak { namespace vulkan {
// Addresses are native allocation-relative half-open ranges, not Buffer* or Go
// tensor identities. Equal buffer handles conservatively alias even if metadata
// disagrees. A write flag includes read/modify/write outputs.
struct BatchRangeAccess {
    uint64_t buffer, memory, begin, end;
    bool writes;
};
inline bool batchRangesOverlap(const BatchRangeAccess& a, const BatchRangeAccess& b) {
    return a.buffer == b.buffer ||
        (a.memory == b.memory && a.begin < b.end && b.begin < a.end);
}

// Only the recorder may feed this bounded proof. It retains ALL accesses since
// the last full dependency, so a read/read middle op cannot hide an older write.
// No allocation, handle ownership, resource lifetime or queue/fence authority.
class BatchRangeHazards {
public:
    static constexpr size_t capacity = 64;
    void reset() { used_ = 0; command_ = 0; next_ = -1; complete_ = false; }

    // Called only once the command is guaranteed to record. Returns whether the
    // recorder must emit its existing full barrier. Unknown work, gaps, changed
    // commands and proof-budget exhaustion reset the pending proof conservatively.
    bool record(uint64_t command, int ordinal, const BatchRangeAccess* ranges,
                size_t count, bool known) {
        if (ordinal == 0) reset();
        bool valid = known && command != 0 && ordinal >= 0 && ranges && count > 0 && count <= capacity;
        for (size_t i = 0; valid && i < count; ++i) {
            valid = ranges[i].buffer != 0 && ranges[i].memory != 0 && ranges[i].begin < ranges[i].end;
            for (size_t j = 0; valid && j < i; ++j) {
                if ((ranges[i].writes || ranges[j].writes) && batchRangesOverlap(ranges[i], ranges[j])) valid = false;
            }
        }
        const bool contiguous = complete_ && command_ == command && next_ == ordinal;
        bool sync = ordinal != 0 && (!valid || !contiguous);
        if (sync || ordinal == 0) used_ = 0;
        if (valid && !sync) {
            for (size_t i = 0; i < count && !sync; ++i) {
                for (size_t j = 0; j < used_; ++j) {
                    if ((ranges[i].writes || pending_[j].writes) && batchRangesOverlap(ranges[i], pending_[j])) {
                        sync = true;
                        break;
                    }
                }
            }
            if (used_ + count > capacity) sync = true;
        }
        if (sync || !valid) used_ = 0;
        if (valid) {
            for (size_t i = 0; i < count; ++i) pending_[used_++] = ranges[i];
        }
        command_ = command;
        // Avoid signed overflow; an exhausted ordinal can never continue a proof.
        next_ = ordinal >= 0 && ordinal < INT32_MAX ? ordinal + 1 : -1;
        complete_ = valid && next_ >= 0;
        return sync;
    }
private:
    BatchRangeAccess pending_[capacity]{};
    size_t used_ = 0;
    uint64_t command_ = 0;
    int next_ = -1;
    bool complete_ = false;
};
}} // namespace fak::vulkan
#endif
