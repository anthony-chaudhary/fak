// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "model/LookupProposals.hpp"
#include "model/StepCosts.hpp"
#include "model/VerifyBudget.hpp"

namespace fak::inkling {
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
