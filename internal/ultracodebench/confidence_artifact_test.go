package ultracodebench

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIssue8672ConfidenceArtifact(t *testing.T) {
	raw, err := os.ReadFile("testdata/issue8672-confidence-campaign.json")
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		RawReceipt       string `json:"raw_receipt"`
		RawReceiptSHA256 string `json:"raw_receipt_sha256"`
	}
	if err := json.Unmarshal(raw, &receipt); err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Clean(filepath.FromSlash(receipt.RawReceipt))
	if receipt.RawReceipt == "" || filepath.IsAbs(receiptPath) || receiptPath == ".." || strings.HasPrefix(receiptPath, ".."+string(filepath.Separator)) {
		t.Fatalf("raw_receipt must be a repository-relative path, got %q", receipt.RawReceipt)
	}
	receiptRaw, err := os.ReadFile(filepath.Join("..", "..", receiptPath))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(receiptRaw)); got != receipt.RawReceiptSHA256 {
		t.Fatalf("receipt digest = %s, want %s", got, receipt.RawReceiptSHA256)
	}
	var campaign FactorialCampaign
	if err := json.Unmarshal(raw, &campaign); err != nil {
		t.Fatal(err)
	}
	first, err := EvaluateConfidenceCampaign(campaign, []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	second, err := EvaluateConfidenceCampaign(campaign, []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	firstRaw := renderConfidenceReport(t, first)
	secondRaw := renderConfidenceReport(t, second)
	if !bytes.Equal(firstRaw, secondRaw) {
		t.Fatalf("identical campaign replays were not byte-stable:\nfirst:\n%s\nsecond:\n%s", firstRaw, secondRaw)
	}
	if len(first.Widths) != 4 {
		t.Fatalf("widths=%d", len(first.Widths))
	}
	for _, w := range first.Widths {
		if w.AcceptedRuns < 5 || w.OutcomeAbstentions != 0 {
			t.Fatalf("width %d runs=%d abstentions=%d", w.Width, w.AcceptedRuns, w.OutcomeAbstentions)
		}
		if w.Scope.Interval.Low == w.Scope.Interval.High || w.Prefix.Interval.Low == w.Prefix.Interval.High || w.Combined.Interval.Low == w.Combined.Interval.High || w.Interaction.Interval.Low == w.Interaction.Interval.High {
			t.Fatalf("width %d lacks nonzero interval: %+v", w.Width, w)
		}
		if w.Attribution.FixedClaimVerdict != "REJECT" {
			t.Fatalf("width %d fixed claim=%s interval=%+v", w.Width, w.Attribution.FixedClaimVerdict, w.Attribution.ScopedPercent.Interval)
		}
	}
	reportRaw, err := os.ReadFile("testdata/issue8672-confidence-report.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstRaw, reportRaw) {
		t.Fatalf("replayed report differs byte-for-byte from the checked artifact:\n%s", firstRaw)
	}
	if first.PromotionEvidence == "" || first.DemotionEvidence == "" || first.InvalidatingAssumption == "" || first.ReplayCommand == "" {
		t.Fatal("generation evidence incomplete")
	}
}

func renderConfidenceReport(t *testing.T, report ConfidenceReport) []byte {
	t.Helper()
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(raw, '\n')
}

func TestConfidenceCampaignAbstainsSeparatelyFromNoise(t *testing.T) {
	raw, _ := os.ReadFile("testdata/issue8672-confidence-campaign.json")
	var c FactorialCampaign
	_ = json.Unmarshal(raw, &c)
	c.Cells[0].Replicates[0].Accepted = false
	got, err := EvaluateConfidenceCampaign(c, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if got.Widths[0].OutcomeAbstentions != 1 || got.Widths[0].Verdict != "ABSTAIN" || got.Widths[0].WithinRunMeasurementNoise != 0 {
		t.Fatalf("result=%+v", got.Widths[0])
	}
}
