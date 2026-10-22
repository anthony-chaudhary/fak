// SOURCE-ONLY, UNWIRED. Not a completion API implementation or runtime admission.
// Include only after independent review and the hook integration in SEAM.md.
// All calls require vulkanMu. No concurrency, generic Tensor, or ownership casts.
#ifndef FVK_V41_BATCH_LIFETIME_PROTOTYPE_H
#define FVK_V41_BATCH_LIFETIME_PROTOTYPE_H
#include "proposed_api.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

struct Buffer;

class V41BatchLifetimePrototype final {
public:
    struct Allocation {
        Buffer* buffer;
        uint64_t generation, allocationID;
        size_t bytes;
    };
    struct Use {
        uint64_t id;
        Allocation source, destination;
        bool recorded;
    };
    struct Record {
        fvk_v41_batch_identity identity;
        uintptr_t command; // correlation only, NEVER the identity
        fvk_v41_batch_phase phase;
        int result;
        std::vector<Use> uses;
    };

    V41BatchLifetimePrototype() = default;
    V41BatchLifetimePrototype(const V41BatchLifetimePrototype&) = delete;
    V41BatchLifetimePrototype& operator=(const V41BatchLifetimePrototype&) = delete;

    // Hooks must supply actual native identities, never caller-selected values.
    // This object must survive context replacement. No reset operation exists.
    bool bindContext(uint64_t generation) {
        if (!generation || generation <= generation_ || !records_.empty()) return false;
        generation_ = generation;
        return true;
    }

    // Called only after checked step admission rejects inflight/restore/foreign
    // work, BEFORE command allocation/begin. Owner token identifies the step scope;
    // every auto-reopened window gets a fresh serial from this object.
    bool begin(uint64_t generation, uint64_t owner,
               fvk_v41_batch_identity& out) {
        out = {};
        if (!generation_ || generation != generation_ || !owner ||
            nextSerial_ == std::numeric_limits<uint64_t>::max()) return false;
        for (const auto& entry : records_)
            if (entry.second.phase == FVK_V41_BATCH_RESERVED ||
                entry.second.phase == FVK_V41_BATCH_RECORDING ||
                entry.second.phase == FVK_V41_BATCH_QUARANTINED) return false;
        const uint64_t serial = nextSerial_ + 1;
        const fvk_v41_batch_identity id{generation, owner, serial};
        try {
            if (!records_.emplace(serial, Record{id, 0,
                    FVK_V41_BATCH_RESERVED, 0, {}}).second) return false;
        } catch (...) { return false; }
        nextSerial_ = serial;
        out = id;
        return true;
    }

    // No allocation here: activate only after the actual command begin succeeds.
    bool activate(fvk_v41_batch_identity id, uintptr_t command) {
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_RESERVED || !command) return false;
        r->command = command;
        r->phase = FVK_V41_BATCH_RECORDING;
        return true;
    }

    // Only a command-never-begun reservation is cancellable by this method.
    // A begun command's cleanup/failure must use the checked lifecycle hooks.
    bool cancelReservation(fvk_v41_batch_identity id) {
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_RESERVED || r->command || !r->uses.empty()) return false;
        r->phase = FVK_V41_BATCH_CANCELLED;
        return true;
    }

    // Reserve/enroll BEFORE emitting any referencing Vulkan command. Source
    // may be absent for fill. Caller validates live Buffer and exact geometry
    // before this function, under the same lock. Refuse allocation aliasing.
    bool enroll(fvk_v41_batch_identity id, Allocation source,
                Allocation destination, uint64_t& useID) {
        useID = 0;
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_RECORDING ||
            !valid(destination, id.generation) ||
            (source.buffer && (!valid(source, id.generation) ||
                source.buffer == destination.buffer ||
                source.allocationID == destination.allocationID)) ||
            (!source.buffer && (source.generation || source.allocationID || source.bytes)) ||
            nextUse_ == std::numeric_limits<uint64_t>::max()) return false;
        // Conservative first seam: no unresolved use may overlap either buffer.
        if (pinnedAllocation(destination) || (source.buffer && pinnedAllocation(source))) return false;
        const uint64_t next = nextUse_ + 1;
        try { r->uses.push_back(Use{next, source, destination, false}); }
        catch (...) { return false; }
        nextUse_ = next;
        useID = next;
        return true;
    }

    // Called immediately after commands are emitted, before any possible flush.
    // Failure must quarantine the owning window; it cannot authorize rollback.
    bool recorded(fvk_v41_batch_identity id, uint64_t useID) {
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_RECORDING) return false;
        for (auto& use : r->uses) if (use.id == useID && !use.recorded) {
            use.recorded = true;
            return true;
        }
        return false;
    }

    // REQUIRED hook: INSIDE v41EndSubmitWaitChecked, immediately after wait
    // and BEFORE retainCompletedBatchCommand or any command free. This unwired
    // bookkeeping method does not itself mint or validate a completion proof.
    // Arguments must derive from that exact checked helper invocation, not
    // fvk_submission_status, host Read, queue idle guesses, or reopened windows.
    // checkedSuccess is only valid when end/reset/submit/wait all succeeded.
    bool retireChecked(fvk_v41_batch_identity id, uintptr_t command,
                       bool checkedSuccess, bool hadWork,
                       bool uncertain, int status) {
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_RECORDING || r->command != command) return false;
        bool allRecorded = true;
        for (const auto& use : r->uses) if (!use.recorded) allRecorded = false;
        if (!checkedSuccess || uncertain || status != 0 || !allRecorded ||
            (!hadWork && !r->uses.empty())) {
            r->phase = FVK_V41_BATCH_QUARANTINED;
            r->result = status != 0 ? status : -1;
        } else {
            r->phase = hadWork ? FVK_V41_BATCH_COMPLETE : FVK_V41_BATCH_CANCELLED;
            r->result = 0;
            r->command = 0; // no reusable command alias in a completion receipt
        }
        return true;
    }

    bool query(fvk_v41_batch_identity id, fvk_v41_batch_phase& phase, int& result) const {
        const Record* r = lookup(id);
        if (!r) return false;
        phase = r->phase;
        result = r->result;
        return true;
    }

    // Selected record-time failure or context failure has no rollback path.
    // Previously completed receipts remain true fence facts, but the caller's
    // sticky context gate still forbids access/release on an unhealthy device.
    void quarantineRecording(int status) {
        for (auto& entry : records_) if (entry.second.phase == FVK_V41_BATCH_RECORDING ||
                entry.second.phase == FVK_V41_BATCH_RESERVED) {
            entry.second.phase = FVK_V41_BATCH_QUARANTINED;
            entry.second.result = status != 0 ? status : -1;
        }
    }

    // Exact individual acknowledgement, AFTER committed Go metadata update.
    // Completed-but-unacknowledged source and destination stay pinned.
    bool acknowledge(fvk_v41_batch_identity id, uint64_t useID) {
        Record* r = lookup(id);
        if (!r || r->phase != FVK_V41_BATCH_COMPLETE) return false;
        for (auto it = r->uses.begin(); it != r->uses.end(); ++it)
            if (it->id == useID) { r->uses.erase(it); return true; }
        return false;
    }

    bool forget(fvk_v41_batch_identity id) {
        const Record* r = lookup(id);
        if (!r || !r->uses.empty() || (r->phase != FVK_V41_BATCH_COMPLETE &&
                r->phase != FVK_V41_BATCH_CANCELLED)) return false;
        records_.erase(id.serial);
        return true;
    }

    // Must guard actual native destruction/recycling, not just Go Free/Close.
    // Pointer match deliberately blocks reuse even if supplied IDs disagree.
    bool pinned(const Buffer* buffer) const {
        if (!buffer) return false;
        for (const auto& entry : records_) for (const auto& use : entry.second.uses)
            if (use.source.buffer == buffer || use.destination.buffer == buffer) return true;
        return false;
    }

    // Allocation identity catches distinct wrappers of the same allocation.
    // Real integration must additionally validate VkDeviceMemory aliasing and
    // reject imported/suballocated backing not covered by allocationID.
    bool pinnedAllocation(Allocation allocation) const {
        for (const auto& entry : records_) for (const auto& use : entry.second.uses)
            for (const Allocation candidate : {use.source, use.destination})
                if (candidate.buffer && (candidate.buffer == allocation.buffer ||
                    (candidate.generation == allocation.generation &&
                     candidate.allocationID == allocation.allocationID))) return true;
        return false;
    }

private:
    uint64_t generation_ = 0, nextSerial_ = 0, nextUse_ = 0;
    std::map<uint64_t, Record> records_;
    static bool valid(Allocation a, uint64_t generation) {
        return a.buffer && a.generation == generation && a.allocationID && a.bytes;
    }
    const Record* lookup(fvk_v41_batch_identity id) const {
        if (!id.serial || !id.owner || !id.generation || id.generation != generation_) return nullptr;
        const auto it = records_.find(id.serial);
        if (it == records_.end() || it->second.identity.owner != id.owner ||
            it->second.identity.generation != id.generation) return nullptr;
        return &it->second;
    }
    Record* lookup(fvk_v41_batch_identity id) {
        return const_cast<Record*>(static_cast<const V41BatchLifetimePrototype*>(this)->lookup(id));
    }
};
#endif
