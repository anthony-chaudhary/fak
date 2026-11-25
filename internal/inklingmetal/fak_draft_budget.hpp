// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "model/LookupProposals.hpp"
#include "model/StepCosts.hpp"
#include "model/VerifyBudget.hpp"

#include <string>

namespace fak::inkling {
// Target logits differ by verify width, so greedy output follows the width
// schedule. Request and Session price a width from a constant table; Timed
// prices it from measured GPU time, whose schedule moves with machine load.
// Request also starts each request's acceptance learners from their priors,
// which makes the schedule a function of that request alone.
enum class WidthSchedule { Request, Session, Timed };

class DraftBudget final {
public:
  void resetRequest();
  uint32_t prepare(std::span<const uint32_t> committed, uint32_t anchor,
                   uint32_t depth);
  void observe(std::span<const uint32_t> treeTokens,
               std::span<const splash::ops::TreeRow> tree,
               std::span<const float> scores, std::span<const float> caps,
               std::span<const uint32_t> candidates,
               std::span<const uint32_t> selections, uint32_t rows,
               double gpuSeconds);
  const auto &chain() const { return chain_; }
  const auto &rates() const { return rates_; }
  const auto &calibration() const { return confidence_.acceptance(false); }

private:
  struct Policy {
    WidthSchedule schedule{WidthSchedule::Request};
    uint32_t forcedRows{};
    std::array<double, SPLASH_TARGET_VERIFY_ROWS + 1> cycleMs{};
    std::string trace;
  };
  static Policy policyFromEnvironment();
  Policy policy_{policyFromEnvironment()};
  splash::model::LookupIndex index_;
  splash::model::LookupAcceptance lookup_;
  splash::model::VerifyBudget confidence_;
  static constexpr std::array<uint32_t, 1> classes_ = {1};
  splash::model::CostLine costs_{classes_};
  std::vector<uint32_t> chain_;
  std::vector<float> rates_, draftChances_, joinedChances_;
  uint32_t match_{};
  std::array<double, SPLASH_TARGET_VERIFY_ROWS + 1> wideTimes_{};
};
} // namespace fak::inkling
