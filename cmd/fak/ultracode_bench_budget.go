package main

import (
	"encoding/json"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

type ultracodeBenchBudgetReceipt struct {
	Budget orchestration.UltracodeEnvelopeReceipt `json:"budget"`
}

func decodeUltracodeBenchBudget(data []byte) (ultracodeBenchBudgetReceipt, error) {
	var receipt ultracodeBenchBudgetReceipt
	err := json.Unmarshal(data, &receipt)
	return receipt, err
}

func ultracodeBenchSelfcheckBudget(pair ultracodebench.Pair) ultracodeBenchBudgetReceipt {
	started := time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC)
	receipt, err := orchestration.NewUltracodeEnvelopeReceipt(
		pair.Fleet.Identity.TokenBudget,
		time.Duration(pair.Fleet.Identity.WallBudgetMS)*time.Millisecond,
		started,
		[]string{"worker-1", "worker-2", "worker-3"},
	)
	if err != nil {
		panic(err)
	}
	receipt, err = orchestration.FoldUltracodeEnvelopeReceipt(receipt, []orchestration.UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 1_134, Authority: orchestration.UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 1_133, Authority: orchestration.UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-3", ProviderTokens: 1_133, Authority: orchestration.UltracodeBudgetAuthorityProvider},
	}, started.Add(time.Duration(pair.Fleet.CriticalPathMS)*time.Millisecond))
	if err != nil {
		panic(err)
	}
	return ultracodeBenchBudgetReceipt{Budget: receipt}
}

func applyUltracodeBenchBudgetReceipt(report ultracodebench.Report, pair ultracodebench.Pair, budget ultracodeBenchBudgetReceipt) ultracodebench.Report {
	reason := ultracodeBenchBudgetReason(pair, budget.Budget)
	if reason == "" {
		return report
	}
	if report.Verdict != "ABSTAIN" {
		report.Verdict = "ABSTAIN"
		report.Reasons = []string{reason}
	} else if !slices.Contains(report.Reasons, reason) {
		report.Reasons = append(report.Reasons, reason)
	}
	return report
}

func ultracodeBenchBudgetReason(pair ultracodebench.Pair, receipt orchestration.UltracodeEnvelopeReceipt) string {
	if err := orchestration.ValidateUltracodeEnvelopeReceipt(receipt); err != nil {
		return orchestration.UltracodeBudgetReasonIncomplete
	}
	if receipt.DeclaredTokens != pair.Fleet.Identity.TokenBudget || receipt.WallBudgetMS != pair.Fleet.Identity.WallBudgetMS {
		return orchestration.UltracodeBudgetReasonIncomplete
	}
	if !receipt.Complete || receipt.TotalChildren == 0 || receipt.CoveredChildren != receipt.TotalChildren || receipt.Authority != orchestration.UltracodeBudgetAuthorityProvider {
		return orchestration.UltracodeBudgetReasonIncomplete
	}
	if receipt.Overrun || !receipt.Admitted {
		if receipt.Reason != "" {
			return receipt.Reason
		}
		return orchestration.UltracodeBudgetReasonIncomplete
	}
	return ""
}
