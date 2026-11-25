// go:build cgo

// SPDX-License-Identifier: Apache-2.0
// Lookup scoring and per-tree observations adapt Splash model/Runtime.cpp
// at 2afc7337aacc044722fa95e4cb3da75c281a19a8.
#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "fak_draft_budget.hpp"
#include "metal/abi/Sampling.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string_view>

namespace fak::inkling {
using namespace splash::model;
namespace {
// GPU milliseconds of one fused cycle (draft, target verify, acceptance and
// commit in one submission) by verify rows: M3 Pro medians of cycles forced to
// each width, as TestDeterministicWidthCalibration prints them, with rows 12
// to 14 pooled at their mean so that no wider cycle is cheaper. Only the
// ratios steer the choice.
constexpr std::array<double, SPLASH_TARGET_VERIFY_ROWS + 1> kCycleMs{
    0,    44.9, 51.6, 60.0, 65.1, 70.9, 73.4, 76.3, 80.0,
    86.7, 89.7, 92.5, 95.8, 95.8, 95.8, 97.5, 100.0};

const char *setting(const char *name) {
  const char *value = std::getenv(name);
  return value && *value ? value : nullptr;
}
} // namespace

DraftBudget::Policy DraftBudget::policyFromEnvironment() {
  Policy policy{.cycleMs = kCycleMs};
  if (const char *name = setting("FAK_INKLING_METAL_WIDTH_SCHEDULE")) {
    const std::string_view schedule{name};
    if (schedule == "session")
      policy.schedule = WidthSchedule::Session;
    else if (schedule == "timed")
      policy.schedule = WidthSchedule::Timed;
    else if (schedule != "request")
      throw std::invalid_argument(
          "FAK_INKLING_METAL_WIDTH_SCHEDULE is request, session or timed");
  }
  if (const char *rows = setting("FAK_INKLING_METAL_VERIFY_ROWS")) {
    char *end = nullptr;
    const unsigned long forced = std::strtoul(rows, &end, 10);
    if (*end || !forced || forced > SPLASH_TARGET_VERIFY_ROWS)
      throw std::invalid_argument(
          "FAK_INKLING_METAL_VERIFY_ROWS is a row count from 1 to 16");
    policy.forcedRows = static_cast<uint32_t>(forced);
  }
  if (const char *costs = setting("FAK_INKLING_METAL_WIDTH_COSTS")) {
    char *end = nullptr;
    for (uint32_t rows = 1; rows <= SPLASH_TARGET_VERIFY_ROWS; ++rows) {
      const double ms = std::strtod(costs, &end);
      const bool last = rows == SPLASH_TARGET_VERIFY_ROWS;
      if (end == costs || !std::isfinite(ms) || !(ms > 0) ||
          *end != (last ? '\0' : ','))
        throw std::invalid_argument("FAK_INKLING_METAL_WIDTH_COSTS is 16 "
                                    "comma-separated positive milliseconds");
      policy.cycleMs[rows] = ms;
      costs = last ? end : end + 1;
    }
  }
  if (const char *trace = setting("FAK_INKLING_METAL_WIDTH_TRACE"))
    policy.trace = trace;
  return policy;
}

void DraftBudget::resetRequest() {
  policy_ = policyFromEnvironment();
  if (policy_.schedule == WidthSchedule::Request) {
    lookup_ = {};
    confidence_ = {};
  }
  index_ = LookupIndex{};
  match_ = 0;
  chain_.clear();
  rates_.clear();
  draftChances_.clear();
  joinedChances_.clear();
}

uint32_t DraftBudget::prepare(std::span<const uint32_t> committed,
                              uint32_t anchor, uint32_t depth) {
  // Only the previous anchor can have entered the index ahead of accepted
  // history. A discarded or replaced anchor restarts it from the caller's
  // prefix.
  auto held = index_.tokens();
  if (held.size() > committed.size() ||
      (!held.empty() && held.back() != committed[held.size() - 1])) {
    index_ = LookupIndex{};
    held = {};
  }
  index_.append(committed.subspan(held.size()));
  index_.append(std::span(&anchor, 1));
  auto next = index_.proposals(depth);
  chain_.assign(next.begin(), next.end());
  match_ = index_.match().length;
  std::vector<float> chainChances;
  rates_.clear();
  float chance = 1;
  for (uint32_t i = 0; i < chain_.size(); ++i) {
    for (uint32_t a = 0; a < SPLASH_TREE_AGREEMENTS; ++a)
      rates_.push_back(
          lookup_.rate(false, match_, i, static_cast<LookupAgreement>(a)));
    chance *= lookup_.rate(false, match_, i, LookupAgreement::Unknown);
    chainChances.push_back(chance);
  }
  if (policy_.forcedRows)
    return std::min(depth + 1, policy_.forcedRows);
  if (draftChances_.empty() && joinedChances_.empty())
    return std::min(depth + 1, std::max(8u, uint32_t(chain_.size() + 1)));
  const auto chances = VerifyBudget::laneChances(
      draftChances_, chain_.empty() ? joinedChances_ : chainChances, depth);
  auto fitted = costs_.line(1, false);
  if (fitted && !(fitted->first + fitted->second > 0))
    fitted.reset();
  const auto narrowPrice = [&](uint32_t rows) {
    return fitted ? fitted->first + fitted->second * rows
                  : CostLine::seed(1, rows);
  };
  const bool timed = policy_.schedule == WidthSchedule::Timed;
  const auto price = [&](uint32_t rows) {
    if (!timed)
      return policy_.cycleMs[rows];
    if (rows <= 8)
      return narrowPrice(rows);
    if (wideTimes_[rows] > 0 && fitted)
      return wideTimes_[rows];
    return narrowPrice(8) + (rows - 8) * (narrowPrice(8) - narrowPrice(4)) / 4;
  };
  double expected = 1.0, best = expected / price(1);
  uint32_t rows = 1;
  for (uint32_t i = 0; i < chances.size(); ++i) {
    expected += chances[i];
    const double rate = expected / price(i + 2);
    if (rate > best) {
      best = rate;
      rows = i + 2;
    }
  }
  return rows;
}

void DraftBudget::observe(std::span<const uint32_t> tokens,
                          std::span<const splash::ops::TreeRow> tree,
                          std::span<const float> scores,
                          std::span<const float> caps,
                          std::span<const uint32_t> candidates,
                          std::span<const uint32_t> selections, uint32_t rows,
                          double gpuSeconds) {
  const auto drafted = [&](size_t i) {
    return scores[2 * i] >= 0.0f && scores[2 * i] <= 1.0f;
  };
  bool holdsDraft = false;
  for (size_t i = 1; i < tree.size(); ++i)
    holdsDraft |= tree[i].parent == 0 && drafted(i);
  if (holdsDraft) {
    uint32_t chained = 0, parent = 0;
    for (auto token : chain_) {
      uint32_t i = parent + 1;
      while (i < tree.size() &&
             (tree[i].parent != parent || tokens[i] != token))
        ++i;
      if (i == tree.size())
        break;
      chained |= 1u << i;
      parent = i;
    }
    draftChances_.clear();
    joinedChances_.clear();
    for (size_t i = 1; i < tree.size(); ++i)
      if (drafted(i))
        ((chained >> i) & 1 ? joinedChances_ : draftChances_)
            .push_back(scores[2 * i + 1]);
    std::ranges::sort(draftChances_, std::greater<>{});
    std::ranges::sort(joinedChances_, std::greater<>{});
  }
  if (!chain_.empty()) {
    const auto outcome = lookupOutcome(chain_, tokens, tree,
                                       std::min<uint32_t>(rows, tree.size()),
                                       selections, candidates);
    if (!outcome.verified.empty())
      lookup_.observe(false, match_, outcome.verified, outcome.accepted);
  }
  const VerifyBudget::Lane lane{
      false, VerifyBudget::treeLabels(tokens, tree, scores, caps, selections)};
  confidence_.observe(std::span(&lane, 1));
  // Calibration reads each cycle's rows, kept tokens, history and GPU
  // milliseconds from this log.
  if (!policy_.trace.empty()) {
    if (std::FILE *trace = std::fopen(policy_.trace.c_str(), "a")) {
      std::fprintf(trace, "%u %zu %zu %.4f\n", rows, selections.size(),
                   index_.tokens().size(), gpuSeconds * 1000);
      std::fclose(trace);
    }
  }
  if (std::isfinite(gpuSeconds) && gpuSeconds > 0) {
    if (rows <= 8)
      costs_.observe(1, false, rows, selections.size(), gpuSeconds * 1000);
    else
      wideTimes_[rows] = gpuSeconds * 1000;
  }
}
} // namespace fak::inkling
#endif
