//go:build cgo

// SPDX-License-Identifier: Apache-2.0
// The buffer bindings adapt Splash model/Runtime.cpp at
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, under Apache-2.0.
#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "fak_draft_lane.hpp"
#include "ops/RoPE.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fak::inkling {
using namespace splash;
using namespace splash::model;
using metal::CommandGraph;
using metal::MetalBuffer;

DraftLane::DraftLane(metal::MetalBackend &backend,
                     const ops::ExecutionPlans &operators,
                     const LoadedModel &weights,
                     const RuntimeGeometry &geometry, QwenTarget &target,
                     PrefillArena &prefill, DecodeArena &decode)
    : target_(target), prefill_(prefill), decode_(decode),
      model_(weights.draft, backend, operators, target.vocabularyProjection(),
             {}, geometry.target.logitFinish()),
      tracker_(std::make_shared<StateAllocationTracker>()),
      ring_(backend, tracker_, weights.draft.layout.stateLayout(),
            "fak-draft-ring"),
      window_(backend, tracker_, weights.draft.layout.stateLayout(),
              "fak-draft-window"),
      storageLanes_(draftStepLanes(geometry, operators, 1)),
      mask_(geometry.target.maskToken),
      vocabulary_(geometry.target.logitFinish().tokens),
      tapWidth_(weights.draft.layout.targetHiddenSize()) {
  if (mask_ >= vocabulary_ || !tapWidth_ ||
      tapWidth_ != geometry.target.capturedHiddenSize())
    throw std::invalid_argument("draft and target capture geometry differ");
}

void DraftLane::reset() {
  budget_.resetRequest();
  const auto clear = [](const MetalBuffer &b) {
    std::memset(b.contents(), 0, b.sizeBytes());
  };
  for (const auto &layer : ring_.layers()) {
    clear(layer.keys);
    clear(layer.values);
  }
  clear(window_.buffer());
}
void DraftLane::append(CommandGraph &graph, MetalBuffer captured,
                       uint32_t first, uint32_t rows) {
  if (!rows || rows > kPrefillRows ||
      first > std::numeric_limits<uint32_t>::max() - rows ||
      captured.sizeBytes() < uint64_t(rows) * tapWidth_ * sizeof(uint16_t))
    throw std::invalid_argument("invalid committed draft taps");
  const auto p = [&](PrefillTensor tensor) { return prefill_.get(tensor); };
  ops::LinearScratch scratch{.input = p(PrefillTensor::LinearTable),
                             .sums = p(PrefillTensor::LinearSums),
                             .partials = p(PrefillTensor::LinearPartials),
                             .counters = p(PrefillTensor::LinearCounters),
                             .rotated = p(PrefillTensor::LinearRotated)};
  DFlashPrefillBuffers buffers{std::move(captured),
                               scratch,
                               p(PrefillTensor::ContextProjected),
                               p(PrefillTensor::ContextHidden),
                               p(PrefillTensor::ContextKv),
                               p(PrefillTensor::DraftRopeCos),
                               p(PrefillTensor::DraftRopeSin)};
  ops::RoPE::addDraftRangeTables(
      graph, p(PrefillTensor::DraftInverseFrequencies), buffers.ropeCos,
      buffers.ropeSin, rows, first, kPrefillRows);
  DFlashPrefillSpan span{0, rows, first, ring_.layers(), window_.buffer()};
  model_.addContextPrefill(graph, std::move(buffers), rows, std::span(&span, 1),
                           {});
}

uint32_t DraftLane::propose(CommandGraph &graph, uint32_t anchor,
                            uint32_t length, uint32_t depth, uint32_t children,
                            bool writeVerifyRows,
                            std::span<const uint32_t> committed) {
  if (anchor >= vocabulary_ || !depth ||
      depth > (writeVerifyRows ? SPLASH_TREE_CHAIN_TOKENS
                               : SPLASH_DRAFT_PROPOSAL_TOKENS) ||
      !children || children > SPLASH_TREE_CHILDREN ||
      length > std::numeric_limits<uint32_t>::max() - SPLASH_DRAFT_QUERY_ROWS)
    throw std::invalid_argument("invalid draft proposal geometry");
  const auto d = [&](DecodeTensor tensor) {
    return decode_.batchSlice(tensor, storageLanes_);
  };
  const auto input = d(DecodeTensor::DraftInputTokens);
  auto *ids = static_cast<uint32_t *>(input.contents());
  ids[0] = anchor;
  std::fill_n(ids + 1, SPLASH_DRAFT_QUERY_ROWS - 1, mask_);
  DFlashDecodeBuffers buffers;
  buffers.linearScratch = decode_.linearScratch();
  buffers.hidden = {d(DecodeTensor::DraftHidden0),
                    d(DecodeTensor::DraftHidden1)};
  buffers.normalized = d(DecodeTensor::DraftNormalized);
  buffers.dynamic = d(DecodeTensor::DraftDynamic);
  buffers.convolved = d(DecodeTensor::DraftConvolved);
  buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
  buffers.attention = d(DecodeTensor::DraftAttention);
  buffers.projected = d(DecodeTensor::DraftProjected);
  buffers.residual = d(DecodeTensor::DraftResidual);
  buffers.intermediate = d(DecodeTensor::DraftIntermediate);
  buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
  buffers.logits = d(DecodeTensor::Logits);
  buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
  buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
  buffers.queryValues = d(DecodeTensor::DraftQueryValues);
  buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
  buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
  buffers.gateScratch = decode_.gateScratch();
  for (const auto &layer : ring_.layers()) {
    buffers.persistentKeys.emplace_back().fill(layer.keys);
    buffers.persistentValues.emplace_back().fill(layer.values);
  }
  ops::RoPE::addDraftRangeTables(
      graph, prefill_.get(PrefillTensor::DraftInverseFrequencies),
      buffers.ropeCos, buffers.ropeSin, SPLASH_DRAFT_QUERY_ROWS, length,
      SPLASH_DRAFT_QUERY_ROWS);
  target_.addDraftEmbedding(graph, input, buffers.hidden[0],
                            SPLASH_DRAFT_QUERY_ROWS);
  model_.addMaskRows(graph, buffers.hidden[0], 1);
  const uint32_t verifyRows = writeVerifyRows && !committed.empty()
                                  ? budget_.prepare(committed, anchor, depth)
                                  : depth + 1;
  ops::DraftTreeLane lane{
      .anchor = anchor, .depth = depth, .children = children};
  if (writeVerifyRows && !committed.empty()) {
    const auto &chain = budget_.chain();
    std::copy(chain.begin(), chain.end(),
              static_cast<uint32_t *>(d(DecodeTensor::ChainTokens).contents()));
    const auto &rates = budget_.rates();
    std::copy(rates.begin(), rates.end(),
              static_cast<float *>(d(DecodeTensor::ChainRates).contents()));
    lane.chain = chain.size();
    const auto &calibration = budget_.calibration();
    lane.calibration = {calibration.slope, calibration.intercepts};
  }
  DFlashDraftStep step{std::span(&length, 1), std::span(&lane, 1), {}};
  model_.addDecode(graph, std::move(buffers), step);
  const ops::DraftSelectorBuffers selection{d(DecodeTensor::Logits),
                                            d(DecodeTensor::TopPartialIds),
                                            d(DecodeTensor::TopPartialValues),
                                            d(DecodeTensor::Candidates),
                                            d(DecodeTensor::Unary),
                                            d(DecodeTensor::SelectorHidden),
                                            d(DecodeTensor::ChainTokens),
                                            d(DecodeTensor::ChainRates),
                                            d(DecodeTensor::DraftTreeTokens),
                                            d(DecodeTensor::DraftTrees),
                                            d(DecodeTensor::DraftTreeScores),
                                            d(DecodeTensor::DraftTreeSizes),
                                            d(DecodeTensor::DraftTreeDraws),
                                            d(DecodeTensor::DraftTreeNext),
                                            d(DecodeTensor::DraftTreeCaps),
                                            d(DecodeTensor::InputTokens),
                                            d(DecodeTensor::TreeRows),
                                            d(DecodeTensor::LaneLayout),
                                            d(DecodeTensor::VerifySources)};
  model_.addSelection(graph, selection, step);
  if (writeVerifyRows) {
    const uint32_t rows = verifyRows;
    ops::DraftSelector::addVerifyRows(
        graph, selection, {1, rows, ops::verifyLaneCap(rows, 1)}, 0.0f);
  }
  return verifyRows;
}

void DraftLane::commitPath(CommandGraph &graph, uint32_t position,
                           const ops::StepRows &step) {
  const auto d = [&](DecodeTensor t) {
    return decode_.batchSlice(t, storageLanes_);
  };
  DFlashContextBuffers b;
  b.windows[0] = window_.buffer();
  b.linearScratch = decode_.linearScratch();
  b.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
  b.projected = d(DecodeTensor::ContextProjected);
  b.hidden = d(DecodeTensor::ContextHidden);
  b.contextKv = d(DecodeTensor::ContextKv);
  b.ropeCos = d(DecodeTensor::DraftRopeCos);
  b.ropeSin = d(DecodeTensor::DraftRopeSin);
  b.retainedRows = d(DecodeTensor::RetainedRows);
  b.laneRows = d(DecodeTensor::LaneLayout);
  for (const auto &layer : ring_.layers()) {
    b.persistentKeys.emplace_back().fill(layer.keys);
    b.persistentValues.emplace_back().fill(layer.values);
  }
  ops::RoPE::addDraftRangeTables(
      graph, prefill_.get(PrefillTensor::DraftInverseFrequencies), b.ropeCos,
      b.ropeSin, SPLASH_TARGET_VERIFY_ROWS, position,
      SPLASH_TARGET_VERIFY_ROWS);
  model_.addContextCommit(graph, std::move(b), std::span(&position, 1), step);
}

void DraftLane::observe(std::span<const uint32_t> selections, uint32_t rows,
                        double gpuSeconds) {
  const auto d = [&](DecodeTensor t) { return decode_.get(0, t).contents(); };
  const auto size =
      *static_cast<const uint32_t *>(d(DecodeTensor::DraftTreeSizes));
  if (size == 0 || size > SPLASH_TARGET_VERIFY_ROWS)
    throw std::runtime_error("invalid draft observation");
  budget_.observe(
      {static_cast<const uint32_t *>(d(DecodeTensor::DraftTreeTokens)), size},
      {static_cast<const ops::TreeRow *>(d(DecodeTensor::DraftTrees)), size},
      {static_cast<const float *>(d(DecodeTensor::DraftTreeScores)), size * 2},
      {static_cast<const float *>(d(DecodeTensor::DraftTreeCaps)), size},
      {static_cast<const uint32_t *>(d(DecodeTensor::Candidates)),
       SPLASH_DRAFT_CANDIDATES * SPLASH_DRAFT_PROPOSAL_TOKENS},
      selections, rows, gpuSeconds);
}

std::vector<DraftRow> DraftLane::readTree() const {
  const auto view = [&](DecodeTensor tensor) {
    return decode_.get(0, tensor).contents();
  };
  const uint32_t count =
      *static_cast<const uint32_t *>(view(DecodeTensor::DraftTreeSizes));
  if (!count || count > SPLASH_DRAFT_QUERY_ROWS)
    throw std::runtime_error("invalid draft tree size");
  const auto *tokens =
      static_cast<const uint32_t *>(view(DecodeTensor::DraftTreeTokens));
  const auto *rows =
      static_cast<const SplashTreeRow *>(view(DecodeTensor::DraftTrees));
  std::vector<DraftRow> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const auto [ancestors, parent, depth] = rows[i];
    const bool validPath = i == 0
                               ? parent == SPLASH_TREE_ROOT && depth == 0
                               : parent < i && depth == rows[parent].depth + 1;
    if (!validPath || tokens[i] >= vocabulary_)
      throw std::runtime_error("invalid draft tree row");
    result.push_back({tokens[i], parent, depth});
  }
  return result;
}
} // namespace fak::inkling

#endif
