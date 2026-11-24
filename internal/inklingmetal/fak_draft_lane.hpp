// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fak_draft_budget.hpp"

#include "metal/CommandGraph.hpp"
#include "model/ModelFactory.hpp"
#include "model/RuntimeArenas.hpp"

namespace fak::inkling {
struct DraftRow {
  uint32_t token;
  uint16_t parent;
  uint16_t depth;
};

// One request's draft caches. Its model, backend and arenas must outlive it.
// FAK chooses which verified target rows to append and when to submit/fence.
// append and propose can share a graph. Each propose reads a host-written
// anchor; finish or discard the graph before reusing the lane.
class DraftLane final {
public:
  DraftLane(splash::metal::MetalBackend &, const splash::ops::ExecutionPlans &,
            const splash::model::LoadedModel &,
            const splash::model::RuntimeGeometry &, splash::model::QwenTarget &,
            splash::model::PrefillArena &, splash::model::DecodeArena &);
  void reset();
  void append(splash::metal::CommandGraph &,
              splash::metal::MetalBuffer compactCaptured,
              uint32_t firstPosition, uint32_t rows);
  uint32_t propose(splash::metal::CommandGraph &, uint32_t anchor,
                   uint32_t cacheLength, uint32_t depth, uint32_t children = 1,
                   bool writeVerifyRows = false,
                   std::span<const uint32_t> committed = {});
  void commitPath(splash::metal::CommandGraph &, uint32_t position,
                  const splash::ops::StepRows &);
  void observe(std::span<const uint32_t> selections, uint32_t rows,
               double gpuSeconds);
  // Read after the proposal's successful completion. Includes anchor at row 0.
  [[nodiscard]] std::vector<DraftRow> readTree() const;

private:
  DraftBudget budget_;
  splash::model::QwenTarget &target_;
  splash::model::PrefillArena &prefill_;
  splash::model::DecodeArena &decode_;
  splash::model::DFlashDraft model_;
  std::shared_ptr<splash::model::StateAllocationTracker> tracker_;
  splash::model::DFlashDraftRing ring_;
  splash::model::DFlashContextWindow window_;
  uint32_t storageLanes_, mask_, vocabulary_, tapWidth_;
};
} // namespace fak::inkling
