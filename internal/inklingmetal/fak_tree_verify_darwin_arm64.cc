//go:build cgo

// SPDX-License-Identifier: Apache-2.0
// Verify and commit bindings adapted from Splash model/Runtime.cpp at
// 2afc7337aacc044722fa95e4cb3da75c281a19a8. FAK supplies the accepted path.
#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "fak_tree_verify.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RoPE.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include <bit>
#include <stdexcept>

namespace fak::inkling {
using namespace splash;
using namespace splash::model;
using metal::CommandGraph;
using metal::MetalBuffer;

QwenTargetPrefillBuffers targetBuffers(const TargetArena &target,
                                       uint32_t chunk) {
  QwenTargetPrefillBuffers buffers;
  const auto t = [&](TargetTensor tensor) { return target.get(tensor, chunk); };
  buffers.hidden = {t(TargetTensor::Hidden0), t(TargetTensor::Hidden1)};
  buffers.normalized = t(TargetTensor::Normalized);
  buffers.gdnPacked = t(TargetTensor::GdnPacked);
  buffers.gdnQueries = t(TargetTensor::GdnQueries);
  buffers.gdnKeys = t(TargetTensor::GdnKeys);
  buffers.gdnValues = t(TargetTensor::GdnValues);
  buffers.gdnDecay = t(TargetTensor::GdnDecay);
  buffers.gdnBeta = t(TargetTensor::GdnBeta);
  buffers.recurrent = t(TargetTensor::Recurrent);
  buffers.gdnHidden = t(TargetTensor::GdnHidden);
  buffers.gdnOutput = t(TargetTensor::GdnOutput);
  buffers.denseIntermediate = t(TargetTensor::Intermediate);
  buffers.fullPacked = t(TargetTensor::FullPacked);
  buffers.fullQueries = t(TargetTensor::FullQueries);
  buffers.attentionPartials = t(TargetTensor::AttentionPartials);
  buffers.attentionStatistics = t(TargetTensor::AttentionStatistics);
  buffers.attentionHidden = t(TargetTensor::AttentionHidden);
  buffers.attentionOutput = t(TargetTensor::AttentionOutput);
  buffers.ropeCos = t(TargetTensor::RopeCos);
  buffers.ropeSin = t(TargetTensor::RopeSin);
  buffers.moe = target.moeScratch(chunk);
  return buffers;
}

TreeVerify::TreeVerify(const RuntimeGeometry &geometry, QwenTarget &target,
                       PrefillArena &prefill, DecodeArena &decode,
                       kv::PageStorage &kv, MetalBuffer table)
    : geometry_(geometry), target_(target), prefill_(prefill), decode_(decode),
      kv_(kv), table_(std::move(table)) {}

MetalBuffer TreeVerify::begin(CommandGraph &graph, uint32_t position,
                              uint32_t pageCount, MetalBuffer current,
                              std::span<const DraftRow> tree) {
  adaptive_ = false;
  if (tree.empty() || tree.size() > SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid Inkling verify tree size");
  const uint32_t n = static_cast<uint32_t>(tree.size());
  const auto d = [&](DecodeTensor t) { return decode_.batchSlice(t, 1); };
  const auto rowBuffer = d(DecodeTensor::TreeRows);
  const auto tokenBuffer = d(DecodeTensor::InputTokens);
  auto *rows = static_cast<SplashTreeRow *>(rowBuffer.contents());
  auto *tokens = static_cast<uint32_t *>(tokenBuffer.contents());
  for (uint32_t i = 0; i < n; ++i) {
    const auto [token, parent, depth] = tree[i];
    if (token >= geometry_.target.vocabularyTokens ||
        (i == 0 ? parent != SPLASH_TREE_ROOT || depth != 0
                : parent >= i || depth != tree[parent].depth + 1))
      throw std::invalid_argument(
          "invalid Inkling verify tree token or parent");
    const uint32_t path = (1u << i) | (i ? rows[parent].ancestors : 0u);
    rows[i] = {path, parent, depth};
    tokens[i] = token;
  }
  *static_cast<SplashLaneRows *>(d(DecodeTensor::LaneLayout).contents()) = {0,
                                                                            n};
  return forward(graph, position, pageCount, current, n);
}

MetalBuffer TreeVerify::forward(CommandGraph &graph, uint32_t position,
                                uint32_t pageCount, MetalBuffer current,
                                uint32_t n) {
  const auto d = [&](DecodeTensor t) { return decode_.batchSlice(t, 1); };
  const auto rowBuffer = d(DecodeTensor::TreeRows);
  const auto tokenBuffer = d(DecodeTensor::InputTokens);
  step_ = {1, n, ops::verifyLaneCap(n, 1)};
  position_ = position;
  auto b = targetBuffers(decode_.target(), 0);
  b.linearScratch = decode_.linearScratch();
  b.denseGateScratch = decode_.gateScratch();
  b.ropeInverseFrequencies =
      prefill_.get(PrefillTensor::TargetInverseFrequencies);
  b.laneRows = d(DecodeTensor::LaneLayout);
  target_.addVerifyInput(graph, {tokenBuffer, rowBuffer, b.laneRows, step_});
  if (geometry_.target.rotaryPairs)
    ops::RoPE::addTreeTables(graph, b.laneRows, rowBuffer,
                             std::span(&position, 1), b.ropeInverseFrequencies,
                             b.ropeCos, b.ropeSin, step_,
                             geometry_.target.ropeScaling.attentionFactor(),
                             geometry_.target.rotaryPairs);
  target_.addEmbedding(graph, tokenBuffer, b.hidden[0], n);
  QwenTargetDecodeBuffers verify;
  verify.step = step_;
  verify.tree = rowBuffer;
  verify.capturedTargetHidden = decode_.all(DecodeTensor::CapturedTargetHidden);
  verify.finalHidden = decode_.all(DecodeTensor::FinalHidden);
  verify.logits = decode_.all(DecodeTensor::Logits);
  QwenTargetPrefillSequence lane{};
  // The attention split uses the padded lane shape. At the final KV page,
  // bound its history+chunk to allocated slots, including tiny contexts.
  const uint64_t end = uint64_t{pageCount} * kv::kPageTokens;
  if (uint64_t{position} + n > end)
    throw std::invalid_argument("Inkling verify exceeds allocated KV pages");
  const uint32_t chunkRows = std::min<uint64_t>(step_.laneCap, end - position);
  lane.chunk = ops::PagedAttention::chunkParams(
      position, chunkRows, (chunkRows + kTileRows - 1) / kTileRows * kTileRows,
      pageCount);
  lane.pageTable = table_;
  lane.auxiliaryIn = current;
  QwenTargetPrefillChunk chunk{b, std::span(&lane, 1), n, &verify};
  SpilledKvReads reads;
  (void)target_.addPrefill(graph, std::span(&chunk, 1), kv_, reads,
                           {0, geometry_.target.layers}, nullptr);
  return verify.logits;
}

void TreeVerify::propose(CommandGraph &graph, DraftLane &draft, uint32_t anchor,
                         uint32_t position, uint32_t pageCount,
                         MetalBuffer current, uint32_t depth, uint32_t children,
                         std::span<const uint32_t> committed) {
  const uint32_t rows =
      draft.propose(graph, anchor, position, depth, children, true, committed);
  adaptive_ = !committed.empty();
  gpuSeconds_ = 0;
  (void)forward(graph, position, pageCount, current, rows);
  (void)argmax(graph);
}

std::vector<DraftRow> TreeVerify::readTree() const {
  const auto view = [&](DecodeTensor t) {
    return decode_.get(0, t).contents();
  };
  const auto count =
      static_cast<const SplashLaneRows *>(view(DecodeTensor::LaneLayout))->rows;
  if (!count || count > step_.rows)
    throw std::runtime_error("invalid laid-out Inkling tree length");
  const auto *tokens =
      static_cast<const uint32_t *>(view(DecodeTensor::InputTokens));
  const auto *rows =
      static_cast<const SplashTreeRow *>(view(DecodeTensor::TreeRows));
  std::vector<DraftRow> tree;
  tree.reserve(count);
  for (uint32_t i = 0; i < count; ++i)
    tree.push_back({tokens[i], rows[i].parent, rows[i].depth});
  return tree;
}

MetalBuffer TreeVerify::argmax(CommandGraph &graph) {
  const auto d = [&](DecodeTensor t) { return decode_.batchSlice(t, 1); };
  ops::SamplingBuffers buffers{d(DecodeTensor::Logits),
                               d(DecodeTensor::TargetPartialMasses),
                               d(DecodeTensor::TargetVocabularyRows),
                               d(DecodeTensor::SamplingUniforms),
                               d(DecodeTensor::ConstraintMasks),
                               d(DecodeTensor::OutputTokens),
                               d(DecodeTensor::ArgmaxValues),
                               d(DecodeTensor::ArgmaxIndices),
                               d(DecodeTensor::InputTokens),
                               d(DecodeTensor::TreeRows),
                               d(DecodeTensor::TargetVocabularyRanges),
                               d(DecodeTensor::TargetVocabularyArrivals),
                               d(DecodeTensor::LaneLayout),
                               d(DecodeTensor::VerifySources),
                               d(DecodeTensor::Candidates),
                               d(DecodeTensor::DraftTreeDraws),
                               d(DecodeTensor::DraftTreeNext)};
  ops::Sampling greedy(geometry_.target.vocabularySize);
  ops::SamplingPolicy policy{};
  greedy.addVerify(graph, std::span(&policy, 1), buffers, step_,
                   geometry_.target.stopTokens, {});
  return buffers.outputTokens;
}

void TreeVerify::encodeCommit(CommandGraph &graph, MetalBuffer current,
                              MetalBuffer next, DraftLane *draft, bool prefix) {
  auto mask = decode_.batchSlice(DecodeTensor::RetainedRows, 1);
  if (!prefix)
    target_.addKvPathCommit(graph, kv_.layout(), kv_.extentPages(),
                            std::span(&table_, 1), std::span(&position_, 1),
                            step_.laneCap, mask);
  QwenTargetCommitBuffers state;
  state.retainedRows = mask;
  state.laneRows = decode_.batchSlice(DecodeTensor::LaneLayout, 1);
  state.auxiliaryIn[0] = current;
  state.auxiliaryOut[0] = next;
  target_.addStateCommit(graph, std::move(state), step_);
  if (draft)
    draft->commitPath(graph, position_, step_);
}

void TreeVerify::learn(DraftLane &draft, uint32_t path) {
  if (!adaptive_)
    return;
  const auto *ids = static_cast<const uint32_t *>(
      decode_.get(0, DecodeTensor::OutputTokens).contents());
  std::array<uint32_t, SPLASH_TARGET_VERIFY_ROWS> selected{};
  uint32_t at = 0, live = path;
  while (live) {
    selected[at++] = ids[std::countr_zero(live)];
    live &= live - 1;
  }
  draft.observe(std::span(selected).first(at), step_.rows, gpuSeconds_);
  adaptive_ = false;
}

void TreeVerify::commit(CommandGraph &graph, uint32_t leaf, MetalBuffer current,
                        MetalBuffer next, DraftLane *draft) {
  const auto *actual = static_cast<const SplashLaneRows *>(
      decode_.get(0, DecodeTensor::LaneLayout).contents());
  if (leaf >= step_.rows || leaf >= actual->rows)
    throw std::invalid_argument("invalid accepted Inkling tree leaf");
  const auto *rows = static_cast<const SplashTreeRow *>(
      decode_.get(0, DecodeTensor::TreeRows).contents());
  const uint32_t path = rows[leaf].ancestors;
  *static_cast<uint32_t *>(
      decode_.get(0, DecodeTensor::RetainedRows).contents()) = path;
  if (draft)
    learn(*draft, path);
  encodeCommit(graph, current, next, draft,
               path == ((1u << std::popcount(path)) - 1));
}

void TreeVerify::resolve(CommandGraph &graph, uint32_t limit, int32_t eos,
                         MetalBuffer current, MetalBuffer next,
                         DraftLane &draft) {
  const auto d = [&](DecodeTensor t) { return decode_.batchSlice(t, 1); };
  // The acceptance operator compacts selections in place. Keep each row's
  // argmax available to the caller for an independent acceptance check.
  const uint32_t words = 2 * step_.rows;
  auto compact = d(DecodeTensor::ArgmaxIndices);
  ops::RowCopy::add(graph, d(DecodeTensor::OutputTokens), {0, words}, compact,
                    {0, words}, 1, words);
  const uint32_t maximum = std::min(limit, step_.rows);
  const ops::StopTokens stops =
      eos >= 0 ? ops::StopTokens{static_cast<uint32_t>(eos)}
               : ops::StopTokens{};
  ops::Sampling(geometry_.target.vocabularySize)
      .addAcceptance(
          graph,
          {d(DecodeTensor::InputTokens), d(DecodeTensor::TreeRows), compact,
           d(DecodeTensor::RetainedCount), d(DecodeTensor::AcceptedCount),
           d(DecodeTensor::RetainedRows), d(DecodeTensor::LaneLayout)},
          step_, std::span(&maximum, 1), stops);
  encodeCommit(graph, current, next, &draft, false);
}

void TreeVerify::cycle(metal::MetalBackend &backend, DraftLane &draft,
                       uint32_t anchor, uint32_t position, uint32_t pages,
                       MetalBuffer current, MetalBuffer next, uint32_t depth,
                       uint32_t children, uint32_t maximum, int32_t eos,
                       std::span<const uint32_t> committed) {
  if (!maximum || eos < -1 ||
      (eos >= 0 && uint32_t(eos) >= geometry_.target.vocabularyTokens))
    throw std::invalid_argument(
        "invalid Inkling greedy cycle output limit or stop token");
  CommandGraph first, second;
  const uint32_t rows =
      draft.propose(first, anchor, position, depth, children, true, committed);
  adaptive_ = !committed.empty();
  gpuSeconds_ = 0;
  const auto target = [&] {
    (void)forward(second, position, pages, current, rows);
    (void)argmax(second);
    resolve(second, maximum, eos, current, next, draft);
    return second.command();
  };
  gpuSeconds_ = backend.submitCommandPartsAsync(first.command(), target)
                    .wait()
                    .gpuSeconds;
}

uint32_t TreeVerify::retained() const {
  return *static_cast<const uint32_t *>(
      decode_.get(0, DecodeTensor::RetainedRows).contents());
}
} // namespace fak::inkling

#endif
