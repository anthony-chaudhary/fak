//go:build cgo

// SPDX-License-Identifier: Apache-2.0
// Arena buffer bindings adapted from incoai/splash at the revision in
// third_party/inklingmetal.
#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "bridge.h"
#include "fak_tree_verify.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/Inkling2LoopTarget.hpp"
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace splash;
using namespace splash::model;
using namespace splash::metal;

namespace {
constexpr uint32_t maxBlock = 32;
template <class F> char *guard(F &&fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const std::exception &e) {
    return strdup(e.what());
  } catch (...) {
    return strdup("unknown Inkling Metal failure");
  }
}
ops::LinearScratch prefillScratch(const PrefillArena &arena) {
  return {.input = arena.get(PrefillTensor::LinearTable),
          .sums = arena.get(PrefillTensor::LinearSums),
          .partials = arena.get(PrefillTensor::LinearPartials),
          .counters = arena.get(PrefillTensor::LinearCounters),
          .rotated = arena.get(PrefillTensor::LinearRotated)};
}
QwenTargetPrefillBuffers prefillBuffers(const PrefillArena &arena,
                                        uint32_t chunk) {
  QwenTargetPrefillBuffers buffers =
      fak::inkling::targetBuffers(arena.target(), chunk);
  buffers.linearScratch = prefillScratch(arena);
  buffers.captured = arena.get(PrefillTensor::Captured, chunk);
  buffers.denseGateScratch = arena.get(PrefillTensor::GateIntermediate);
  buffers.ropeInverseFrequencies =
      arena.get(PrefillTensor::TargetInverseFrequencies);
  buffers.laneRows = arena.get(PrefillTensor::LaneLayout, chunk);
  return buffers;
}

struct Weights {
  MetalBackend backend;
  ops::ExecutionPlans operators;
  LoadedModel weights;
  RuntimeGeometry geometry;
  std::mutex device;
  const bool hasDraft;

  Weights(const char *file, const char *draft, bool planOnly = false)
      : backend(""),
        operators(backend.capabilities(), ops::Int8Activations::On),
        hasDraft(*draft) {
    const Inkling2LoopLayout layout;
    constexpr std::array<uint32_t, 4> taps{5, 23, 41, 48};
    const auto draftShape = draftLayout(
        layout.hiddenSize, layout.vocabularySize, 5, 6912, 10000, taps);
    weights.images = std::make_shared<WeightImages>(
        backend, std::filesystem::canonical(file).string());
    if (planOnly)
      weights.images->release();
    GgufTargetLoader source(backend, *weights.images, {file},
                            ggufTargetPlan(layout));
    weights.target = loadInkling2LoopWeights(backend, layout, std::ref(source));
    weights.draft.layout = draftShape;
    if (hasDraft && !planOnly) {
      DraftCheckpointLoader loader(backend, *weights.images, draft, draftShape);
      weights.draft = loadDFlashDraftWeights(backend, loader, draftShape);
    }
    geometry.draft = draftShape;
    geometry.layout = layout;
    geometry.target =
        qwenTargetGeometry(std::get<Inkling2LoopWeights>(weights.target), taps);
    geometry.target.lanes = 1;
    geometry.target.kvLayout.format = kv::Format::Int8;
    geometry.target.stateLayout.format = ops::GdnStateFormat::BFloat16;
    if (!geometry.target.valid())
      throw std::invalid_argument("invalid Inkling Metal dimensions");
    geometry.targetScratch =
        qwenTargetScratchBytes(layout, geometry.target, 1, 0);
  }
};
} // namespace

struct FakInkModel {
  std::shared_ptr<Weights> owner;
};

struct FakInkSession {
  std::shared_ptr<Weights> owner;
  uint32_t limit, allocated = 0;
  std::unique_ptr<QwenTarget> target;
  PrefillArena arena;
  std::unique_ptr<DecodeArena> decode;
  std::unique_ptr<fak::inkling::DraftLane> draft;
  uint32_t extentPages;
  std::vector<uint32_t> pageIds;
  kv::PageStorage kvStore;
  MetalBuffer table, current, next, logits;
  std::vector<uint32_t> tokens;
  uint32_t position = 0;
  std::unique_ptr<fak::inkling::TreeVerify> verifier;
  std::vector<fak::inkling::DraftRow> stagedTree;
  MetalBuffer scoreBuffer;

  FakInkSession(std::shared_ptr<Weights> model, uint32_t cap)
      : owner(std::move(model)), limit(cap),
        target(makeQwenTarget(
            std::get<Inkling2LoopWeights>(owner->weights.target),
            owner->geometry.target, owner->backend, owner->operators, 1)),
        arena(owner->backend, owner->geometry, owner->operators),
        extentPages(owner->geometry.target.kvLayout.extentAlignmentPages()),
        pageIds(
            ((cap + kv::kPageTokens - 1) / kv::kPageTokens + extentPages - 1) /
            extentPages * extentPages),
        kvStore(owner->backend, admittedWithin, owner->geometry.target.kvLayout,
                pageIds.size(), extentPages),
        table(shared(pageIds.size() * sizeof(SplashKvPage),
                     "fak-inkling-page-table")),
        current(shared(owner->geometry.target.stateLayout.auxiliaryBytes,
                       "fak-inkling-convolution")),
        next(shared(owner->geometry.target.stateLayout.auxiliaryBytes,
                    "fak-inkling-staged-convolution")),
        logits(shared(uint64_t(maxBlock) *
                          owner->geometry.target.vocabularySize * sizeof(float),
                      "fak-inkling-logits")) {
    std::iota(pageIds.begin(), pageIds.end(), 0);
    scoreBuffer = logits;
    decode = std::make_unique<DecodeArena>(owner->backend, owner->geometry,
                                           owner->operators);
    verifier = std::make_unique<fak::inkling::TreeVerify>(
        owner->geometry, *target, arena, *decode, kvStore, table);
    memset(current.contents(), 0, current.sizeBytes());
    if (owner->hasDraft) {
      draft = std::make_unique<fak::inkling::DraftLane>(
          owner->backend, owner->operators, owner->weights, owner->geometry,
          *target, arena, *decode);
    }
  }
  MetalBuffer shared(uint64_t size, const char *label) {
    return owner->backend.allocateBuffer(size, BufferStorage::Shared, label);
  }
  void ensurePages(uint32_t end) {
    const uint32_t needed = (end + kv::kPageTokens - 1) / kv::kPageTokens;
    if (needed <= allocated)
      return;
    const uint32_t before = allocated;
    while (allocated < needed) {
      if (!kvStore.allocateExtent(allocated / extentPages))
        throw std::runtime_error("unable to allocate Inkling KV extent");
      allocated += extentPages;
    }
    kvStore.writeEntries(std::span(pageIds).first(allocated), before, table);
  }

  // Decode proposals write K/V only at and above position. The prefix length
  // gates visibility. current is immutable until commit so a partial accept may
  // reconstruct precisely its convolution history.
  void forward(std::span<const uint32_t> chunk, uint32_t scoreRows,
               bool appendDraft) {
    const uint32_t n = chunk.size();
    ensurePages(position + n);
    auto buffers = prefillBuffers(arena, 0);
    auto input = arena.get(PrefillTensor::InputTokens, 0);
    auto positions = static_cast<uint32_t *>(
        arena.get(PrefillTensor::TargetPositions, 0).contents());
    std::copy(chunk.begin(), chunk.end(),
              static_cast<uint32_t *>(input.contents()));
    for (uint32_t i = 0; i < n; ++i)
      std::fill_n(positions + i * 3, 3, position + i);
    *static_cast<SplashLaneRows *>(buffers.laneRows.contents()) = {0, n};
    CommandGraph graph;
    target->addEmbedding(graph, input, buffers.hidden[0], n);
    QwenTargetPrefillSequence seq{};
    seq.rows = n;
    seq.tokens = chunk;
    seq.request = {tokens, [](uint64_t i) {
                     return std::array<uint32_t, 3>{uint32_t(i), uint32_t(i),
                                                    uint32_t(i)};
                   }};
    seq.chunk = ops::PagedAttention::chunkParams(
        position, n, (n + kTileRows - 1) / kTileRows * kTileRows, allocated);
    seq.pageTable = table;
    seq.auxiliaryIn = current;
    seq.auxiliaryOut = next;
    if (draft) {
      seq.captures[0] = {0, 0, n};
      seq.captureCount = 1;
    }
    QwenTargetPrefillChunk group{buffers, std::span(&seq, 1), n, nullptr};
    SpilledKvReads reads;
    auto output = target
                      ->addPrefill(graph, std::span(&group, 1), kvStore, reads,
                                   {0, owner->geometry.target.layers}, nullptr)
                      .at(0);
    if (scoreRows) {
      const uint64_t stride =
          owner->geometry.target.hiddenSize * sizeof(uint16_t);
      // Tiled projections may bind padded rows past the last scored position.
      const uint64_t offset = (n - scoreRows) * stride;
      target->addHeadBatch(
          graph,
          owner->backend.view(output, offset, output.sizeBytes() - offset), {},
          logits, scoreRows, buffers.linearScratch);
    }
    if (appendDraft && draft)
      draft->append(graph, buffers.captured, position, n);
    (void)owner->backend.submitCommandAsync(graph.dispatches()).wait();
  }

  void reset(uint32_t capacity) {
    if (!capacity || capacity > owner->geometry.target.contextTokens() ||
        capacity > uint64_t(pageIds.size()) * kv::kPageTokens)
      throw std::invalid_argument("reused Inkling arena capacity exceeded");
    // Absolute position alone gates KV visibility. No command is outstanding
    // when a session returns to the model; committed taps and control masks
    // are reset independently of how the last request ended.
    limit = capacity;
    position = 0;
    tokens.clear();
    stagedTree.clear();
    scoreBuffer = logits;
    auto clear = [](const MetalBuffer &b) {
      if (b && b.sizeBytes())
        std::memset(b.contents(), 0, b.sizeBytes());
    };
    clear(current);
    clear(next);
    if (draft) {
      draft->reset();
      clear(arena.get(PrefillTensor::Captured));
      clear(decode->all(DecodeTensor::CapturedTargetHidden));
    }
    for (auto t : {DecodeTensor::LaneLayout, DecodeTensor::TreeRows,
                   DecodeTensor::RetainedRows, DecodeTensor::DraftTreeSizes,
                   DecodeTensor::ConstraintMasks})
      clear(decode->get(0, t));
    verifier = std::make_unique<fak::inkling::TreeVerify>(
        owner->geometry, *target, arena, *decode, kvStore, table);
  }

  void beginTree(std::vector<fak::inkling::DraftRow> rows, int32_t *argmax) {
    ensurePages(position + rows.size());
    CommandGraph graph;
    scoreBuffer = verifier->begin(graph, position, allocated, current, rows);
    MetalBuffer best;
    if (argmax)
      best = verifier->argmax(graph);
    (void)owner->backend.submitCommandAsync(graph.dispatches()).wait();
    if (argmax)
      std::copy_n(static_cast<const int32_t *>(best.contents()), rows.size(),
                  argmax);
    stagedTree = std::move(rows);
  }

  void commitTree(int32_t leaf) {
    if (stagedTree.empty() || leaf < -1 || leaf >= int32_t(stagedTree.size()))
      throw std::invalid_argument("invalid Inkling accepted tree leaf");
    std::vector<uint32_t> path;
    if (leaf >= 0) {
      for (uint32_t i = leaf;; i = stagedTree[i].parent) {
        path.push_back(stagedTree[i].token);
        if (i == 0)
          break;
      }
      CommandGraph graph;
      verifier->commit(graph, leaf, current, next, draft.get());
      (void)owner->backend.submitCommandAsync(graph.dispatches()).wait();
      std::swap(current, next);
    }
    tokens.resize(position);
    tokens.insert(tokens.end(), path.rbegin(), path.rend());
    position = tokens.size();
    stagedTree.clear();
  }

  void commit(uint32_t keep) {
    if (!stagedTree.empty()) {
      commitTree(int32_t(keep) - 1);
      return;
    }
    const uint32_t pending = tokens.size() - position;
    if (!pending || keep > pending)
      throw std::invalid_argument("invalid Inkling commit length");
    if (keep) {
      if (keep != pending) {
        forward(std::span(tokens).subspan(position, keep), 0, true);
      } else if (draft) {
        CommandGraph graph;
        draft->append(graph, arena.get(PrefillTensor::Captured, 0), position,
                      keep);
        (void)owner->backend.submitCommandAsync(graph.dispatches()).wait();
      }
      std::swap(current, next);
      position += keep;
    }
    tokens.resize(position);
  }
};

extern "C" char *fak_ink_plan(const char *file, const char *draft,
                              uint32_t context, FakInkMemory *plan) {
  return guard([&] {
    if (!fak_ink_available())
      throw std::runtime_error("Inkling Metal device unavailable");
    Weights owner(file, draft, true);
    const auto &g = owner.geometry;
    if (!context || context > g.target.contextTokens())
      throw std::invalid_argument("invalid planned Inkling context");
    for (const auto &image : owner.weights.targetFiles())
      plan->target_images += image.declaredBytes;
    if (*draft)
      plan->draft_images = draftImageBytes(draft, g.draft);
    plan->prefill_scratch = plannedPrefillBytes(g, owner.operators);
    plan->decode_scratch =
        plannedDecodeBytes(g, owner.operators) +
        uint64_t(maxBlock) * g.target.vocabularySize * sizeof(float);
    plan->state = 2 * g.target.stateLayout.auxiliaryBytes;
    if (*draft)
      plan->state += g.draft.stateLayout().ringBytes() +
                     g.draft.stateLayout().windowBytes();
    const auto &kvLayout = g.target.kvLayout;
    const auto align = kvLayout.extentAlignmentPages();
    const uint64_t pages =
        ((context + kv::kPageTokens - 1) / kv::kPageTokens + align - 1) /
        align * align;
    plan->kv = pages * (kvLayout.bytesPerModelPage() + sizeof(SplashKvPage));
  });
}
extern "C" char *fak_ink_model(const char *gguf, const char *draft,
                               FakInkModel **out) {
  return guard([&] {
    if (!fak_ink_available())
      throw std::runtime_error(
          "Inkling Metal requires an M3 or later on macOS 26.4+");
    auto m = std::make_unique<FakInkModel>();
    m->owner = std::make_shared<Weights>(gguf, draft);
    *out = m.release();
  });
}
extern "C" void fak_ink_free_model(FakInkModel *m) { delete m; }
extern "C" char *fak_ink_session(FakInkModel *m, uint32_t cap,
                                 FakInkSession **out) {
  return guard([&] {
    std::lock_guard lock(m->owner->device);
    if (!cap || cap > m->owner->geometry.target.contextTokens())
      throw std::invalid_argument(
          "Inkling session capacity exceeds model limits");
    *out = new FakInkSession(m->owner, cap);
  });
}
extern "C" char *fak_ink_reset(FakInkSession *s, uint32_t cap) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    s->reset(cap);
  });
}
extern "C" void fak_ink_free_session(FakInkSession *s) {
  if (!s)
    return;
  auto owner = s->owner;
  std::lock_guard lock(owner->device);
  delete s;
}
extern "C" char *fak_ink_step(FakInkSession *s, const int32_t *ids, uint32_t n,
                              int prefill, float *scores, int32_t *argmax) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    const auto &g = s->owner->geometry.target;
    if (s->position != s->tokens.size() || !n || n > s->limit - s->position ||
        (!prefill && n > maxBlock))
      throw std::invalid_argument("invalid Inkling block or context length");
    for (uint32_t i = 0; i < n; ++i)
      if (ids[i] < 0 || uint32_t(ids[i]) >= g.vocabularyTokens)
        throw std::invalid_argument("Inkling token outside vocabulary");
    const uint32_t scored = prefill ? 1 : n;
    s->scoreBuffer = s->logits;
    if (!prefill && n <= SPLASH_TARGET_VERIFY_ROWS) {
      std::vector<fak::inkling::DraftRow> rows;
      rows.reserve(n);
      for (uint32_t i = 0; i < n; ++i)
        rows.push_back({uint32_t(ids[i]),
                        uint16_t(i ? i - 1 : SPLASH_TREE_ROOT), uint16_t(i)});
      s->tokens.insert(s->tokens.end(), ids, ids + n);
      s->beginTree(std::move(rows), argmax);
    } else
      for (uint32_t begin = 0; begin < n;) {
        const uint32_t count = std::min(kPrefillRows, n - begin);
        s->tokens.insert(s->tokens.end(), ids + begin, ids + begin + count);
        begin += count;
        s->forward(std::span(s->tokens).subspan(s->position, count),
                   begin == n ? scored : 0, prefill);
        if (prefill) {
          s->position += count;
          std::swap(s->current, s->next);
        }
      }
    const float *head = static_cast<const float *>(s->scoreBuffer.contents());
    for (uint32_t row = 0; row < scored; ++row, head += g.vocabularySize) {
      if (argmax && s->stagedTree.empty())
        argmax[row] = std::max_element(head, head + g.vocabularyTokens) - head;
      if (scores)
        std::copy_n(head, g.vocabularyTokens,
                    scores + uint64_t(row) * g.vocabularyTokens);
    }
  });
}
extern "C" char *fak_ink_commit(FakInkSession *s, uint32_t keep) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    s->commit(keep);
  });
}
extern "C" char *fak_ink_tree(FakInkSession *s, const int32_t *rows, uint32_t n,
                              int32_t *out) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    if (s->tokens.size() != s->position || !n ||
        n > SPLASH_TARGET_VERIFY_ROWS || n > s->limit - s->position)
      throw std::invalid_argument(
          "invalid Inkling verification tree or capacity");
    std::vector<fak::inkling::DraftRow> nodes;
    nodes.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      const auto *r = rows + 3 * i;
      if (r[0] < 0 ||
          uint32_t(r[0]) >= s->owner->geometry.target.vocabularyTokens ||
          (i == 0 ? r[1] != -1 || r[2] != 0
                  : r[1] < 0 || r[1] >= int32_t(i) ||
                        r[2] != 1 + nodes[r[1]].depth))
        throw std::invalid_argument("invalid Inkling tree row");
      nodes.push_back({uint32_t(r[0]), uint16_t(r[1]), uint16_t(r[2])});
    }
    s->beginTree(std::move(nodes), out);
    for (const auto &r : s->stagedTree)
      s->tokens.push_back(r.token);
  });
}
namespace {
void proposeVerify(FakInkSession &s, int32_t anchor, uint32_t depth,
                   uint32_t width, uint32_t maximum, int32_t eos,
                   int32_t *tree, int32_t *validated, uint32_t *count,
                   uint32_t *retained) {
  *count = 0;
  if (retained)
    *retained = 0;
  if (!s.draft)
    return;
  if (s.position != s.tokens.size() || !s.stagedTree.empty() ||
      !depth || depth >= SPLASH_TARGET_VERIFY_ROWS ||
      depth + 1 > s.limit - s.position ||
      (retained && (!maximum || maximum > depth + 1)))
    throw std::invalid_argument("invalid Inkling fused draft context");
  s.ensurePages(s.position + depth + 1);
  const auto history = std::span(s.tokens).first(s.position);
  if (retained) {
    s.verifier->cycle(s.owner->backend, *s.draft, anchor, s.position,
                      s.allocated, s.current, s.next, depth, width, maximum,
                      eos, history);
  } else {
    CommandGraph graph;
    s.verifier->propose(graph, *s.draft, anchor, s.position, s.allocated,
                        s.current, depth, width, history);
    const auto timing = s.owner->backend.submitCommandAsync(graph.dispatches()).wait();
    s.verifier->setGPUSeconds(timing.gpuSeconds);
  }
  auto rows = s.verifier->readTree();
  if (rows.empty() || rows.size() > depth + 1)
    throw std::runtime_error("invalid proposed Inkling tree size");
  const auto *selected = static_cast<const int32_t *>(
      s.decode->get(0, DecodeTensor::OutputTokens).contents());
  uint32_t mask = (1u << rows.size()) - 1;
  if (retained) {
    const uint32_t path = s.verifier->retained();
    if (!(path & 1) || (path & ~mask) || uint32_t(std::popcount(path)) > maximum)
      throw std::runtime_error("invalid retained Inkling path");
    uint32_t previous = 0;
    for (uint32_t row = 1; row < rows.size(); ++row) {
      if (!(path & (1u << row)))
        continue;
      if (rows[row].parent != previous || selected[previous] != int32_t(rows[row].token) ||
          int32_t(rows[row].token) == eos)
        throw std::runtime_error("unverified token in retained Inkling path");
      previous = row;
    }
    s.verifier->learn(*s.draft, path);
    mask = path;
    *retained = path;
    std::swap(s.current, s.next);
  }
  for (uint32_t i = 0; i < rows.size(); ++i) {
    const auto &r = rows[i];
    tree[i * 3] = r.token;
    tree[i * 3 + 1] = r.parent == SPLASH_TREE_ROOT ? -1 : int32_t(r.parent);
    tree[i * 3 + 2] = r.depth;
    validated[i] = selected[i];
    if (mask & (1u << i))
      s.tokens.push_back(r.token);
  }
  *count = rows.size();
  if (retained)
    s.position = s.tokens.size();
  else
    s.stagedTree = std::move(rows);
}
} // namespace

extern "C" char *fak_ink_propose_verify(FakInkSession *s, int32_t anchor,
                                        uint32_t depth, uint32_t width,
                                        int32_t *tree, int32_t *validated,
                                        uint32_t *count) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    proposeVerify(*s, anchor, depth, width, 0, -1, tree, validated, count, nullptr);
  });
}

extern "C" char *fak_ink_cycle(FakInkSession *s, int32_t anchor,
                               uint32_t depth, uint32_t width,
                               uint32_t maximum, int32_t eos,
                               int32_t *tree, int32_t *validated,
                               uint32_t *count, uint32_t *retained) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    proposeVerify(*s, anchor, depth, width, maximum, eos, tree, validated,
                  count, retained);
  });
}
extern "C" char *fak_ink_commit_tree(FakInkSession *s, int32_t leaf) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    s->commitTree(leaf);
  });
}
extern "C" char *fak_ink_draft_tree(FakInkSession *s, int32_t anchor,
                                    uint32_t depth, uint32_t width,
                                    int32_t *rows, uint32_t *n) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    *n = 0;
    if (!s->draft)
      return;
    if (s->position != s->tokens.size())
      throw std::logic_error("commit Inkling block before drafting");
    CommandGraph graph;
    s->draft->propose(graph, anchor, s->position, depth, width);
    (void)s->owner->backend.submitCommandAsync(graph.dispatches()).wait();
    for (const auto &r : s->draft->readTree()) {
      rows[3 * (*n)] = r.token;
      rows[3 * (*n) + 1] = r.parent == SPLASH_TREE_ROOT ? -1 : int(r.parent);
      rows[3 * (*n) + 2] = r.depth;
      ++*n;
    }
  });
}
extern "C" char *fak_ink_propose(FakInkSession *s, int32_t anchor,
                                 uint32_t depth, int32_t *ids, uint32_t *n) {
  return guard([&] {
    std::lock_guard lock(s->owner->device);
    *n = 0;
    if (!s->draft)
      return;
    if (s->position != s->tokens.size())
      throw std::logic_error("commit Inkling block before drafting");
    CommandGraph graph;
    s->draft->propose(graph, anchor, s->position, depth);
    (void)s->owner->backend.submitCommandAsync(graph.dispatches()).wait();
    auto rows = s->draft->readTree();
    for (size_t i = 1; i < rows.size(); ++i) {
      if (rows[i].parent != i - 1 || i > depth)
        break;
      ids[(*n)++] = rows[i].token;
    }
  });
}

#endif
