// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fak_draft_lane.hpp"
#include "ops/PageStorage.hpp"

namespace fak::inkling {
splash::model::QwenTargetPrefillBuffers
targetBuffers(const splash::model::TargetArena &, uint32_t chunk);

// A target pass over a tree stages each row at its own scratch KV position.
// An explicit transaction commits a chosen leaf. A greedy cycle applies the
// caller's stop token and budget and returns the resulting ancestor mask.
class TreeVerify final {
public:
  TreeVerify(const splash::model::RuntimeGeometry &,
             splash::model::QwenTarget &, splash::model::PrefillArena &,
             splash::model::DecodeArena &, splash::kv::PageStorage &,
             splash::metal::MetalBuffer table);
  [[nodiscard]] splash::metal::MetalBuffer
  begin(splash::metal::CommandGraph &, uint32_t position, uint32_t pageCount,
        splash::metal::MetalBuffer current, std::span<const DraftRow> tree);
  void propose(splash::metal::CommandGraph &, DraftLane &, uint32_t anchor,
               uint32_t position, uint32_t pageCount,
               splash::metal::MetalBuffer current, uint32_t depth,
               uint32_t children, std::span<const uint32_t> committed = {});
  void setGPUSeconds(double seconds) { gpuSeconds_ = seconds; }
  [[nodiscard]] std::vector<DraftRow> readTree() const;
  [[nodiscard]] splash::metal::MetalBuffer
  argmax(splash::metal::CommandGraph &);
  void cycle(splash::metal::MetalBackend &, DraftLane &, uint32_t anchor,
             uint32_t position, uint32_t pages,
             splash::metal::MetalBuffer current,
             splash::metal::MetalBuffer next, uint32_t depth, uint32_t children,
             uint32_t maximum, int32_t eos,
             std::span<const uint32_t> committed);
  uint32_t retained() const;
  void learn(DraftLane &, uint32_t path);
  void commit(splash::metal::CommandGraph &, uint32_t leaf,
              splash::metal::MetalBuffer current,
              splash::metal::MetalBuffer next, DraftLane *draft);

private:
  void resolve(splash::metal::CommandGraph &, uint32_t limit, int32_t eos,
               splash::metal::MetalBuffer current,
               splash::metal::MetalBuffer next, DraftLane &);
  void encodeCommit(splash::metal::CommandGraph &, splash::metal::MetalBuffer,
                    splash::metal::MetalBuffer, DraftLane *, bool);
  splash::metal::MetalBuffer forward(splash::metal::CommandGraph &,
                                     uint32_t position, uint32_t pageCount,
                                     splash::metal::MetalBuffer current,
                                     uint32_t rows);
  const splash::model::RuntimeGeometry &geometry_;
  splash::model::QwenTarget &target_;
  splash::model::PrefillArena &prefill_;
  splash::model::DecodeArena &decode_;
  splash::kv::PageStorage &kv_;
  splash::metal::MetalBuffer table_;
  splash::ops::StepRows step_{};
  uint32_t position_{};
  bool adaptive_{};
  double gpuSeconds_{};
};
} // namespace fak::inkling
