package ultracodebench

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
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
	receiptRaw, err := os.ReadFile("testdata/issue8672-cache-receipts.log")
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
	got, err := EvaluateConfidenceCampaign(campaign, []int{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Widths) != 4 {
		t.Fatalf("widths=%d", len(got.Widths))
	}
	for _, w := range got.Widths {
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
	var want ConfidenceReport
	if err := json.Unmarshal(reportRaw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		b, _ := json.MarshalIndent(got, "", "  ")
		t.Fatalf("replayed report differs:\n%s", b)
	}
	if got.PromotionEvidence == "" || got.DemotionEvidence == "" || got.InvalidatingAssumption == "" || got.ReplayCommand == "" {
		t.Fatal("generation evidence incomplete")
	}
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
