package ultracodebench

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestEvaluateNetWorkCampaignAccountsComponentsAndStopsAtFirstLoss(t *testing.T) {
	campaign := netWorkFixture()
	report, err := EvaluateNetWorkCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "LOSS" || report.HillClimb.ChosenWidth != 2 || report.HillClimb.StopWidth != 4 {
		t.Fatalf("report = %+v", report)
	}
	if report.Widths[2].CombinedTokenDelta >= 0 || report.Widths[2].CombinedNetWorkDeltaNS <= 0 {
		t.Fatalf("positive token savings must not hide net loss: %+v", report.Widths[2])
	}
	combined := report.Widths[0].Cells[3]
	if combined.WallLatencyMedianNS != 75 || combined.WallLatencyP95NS != 80 || combined.Components["orchestration"].Controller != "fak" {
		t.Fatalf("combined summary = %+v", combined)
	}
}

func TestEvaluateNetWorkCampaignAbstainsWithoutAuthoritativeTelemetry(t *testing.T) {
	campaign := netWorkFixture()
	delete(campaign.Cells[0].Repetitions[0].Components, "routing")
	report, err := EvaluateNetWorkCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "ABSTAIN" || report.Widths[0].Verdict != "ABSTAIN" || report.HillClimb.StopWidth != 1 {
		t.Fatalf("report = %+v", report)
	}
}

func TestEvaluateNetWorkCampaignAbstainsOnUnequalOutcome(t *testing.T) {
	campaign := netWorkFixture()
	campaign.Cells[3].Repetitions[1].OutcomeDigest = "different"
	report, err := EvaluateNetWorkCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "ABSTAIN" {
		t.Fatalf("report = %+v", report)
	}
}

func TestIssue8679NetWorkReceipt(t *testing.T) {
	const source = "testdata/issue8679-net-work-campaign.json"
	raw, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	var campaign NetWorkCampaign
	if err := json.Unmarshal(raw, &campaign); err != nil {
		t.Fatal(err)
	}
	got, err := EvaluateNetWorkCampaign(campaign)
	if err != nil {
		t.Fatal(err)
	}
	reportRaw, err := os.ReadFile("testdata/issue8679-net-work-report.json")
	if err != nil {
		t.Fatal(err)
	}
	var want NetWorkReport
	if err := json.Unmarshal(reportRaw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		encoded, _ := json.MarshalIndent(got, "", "  ")
		t.Fatalf("replayed report differs from captured net-true receipt:\n%s", encoded)
	}
	if got.SourceReceipt != campaign.SourceReceipt || got.ReplayCommand == "" || got.EvidenceKind != "synthetic_fixture" {
		t.Fatalf("receipt provenance = %+v", got)
	}
}

func TestEvaluateNetWorkCampaignRejectsInvalidEnvelope(t *testing.T) {
	campaign := netWorkFixture()
	campaign.Envelope.Model = ""
	if _, err := EvaluateNetWorkCampaign(campaign); err == nil {
		t.Fatal("missing production envelope must fail")
	}
}

func netWorkFixture() NetWorkCampaign {
	campaign := NetWorkCampaign{Schema: NetWorkCampaignSchema, EvidenceKind: "synthetic_fixture", SourceReceipt: "fixture.json", OutcomeDigest: "accepted", OrderPolicy: "rotating", MinimumRepetitions: 3, Envelope: NetWorkEnvelope{Model: "qwen", Runtime: "ollama/llama.cpp", Tokenizer: "qwen", TaskDigest: "task", CachePosture: "controlled", CampaignVersion: "test"}}
	for _, width := range []int{1, 2, 4, 8} {
		for _, treatment := range netWorkCells {
			base := int64(100)
			tokens := int64(1000)
			if treatment == "prefix-only" {
				base, tokens = 92, 800
			}
			if treatment == "scope-only" {
				base, tokens = 80, 600
			}
			if treatment == "combined" {
				base, tokens = 70, 400
				if width >= 4 {
					base = 105
				}
			}
			cell := NetWorkCell{Width: width, Treatment: treatment}
			for i, noise := range []int64{-5, 0, 5} {
				components := map[string]NetWorkComponent{}
				for _, name := range requiredWorkComponents {
					controller := "runtime"
					if name == "routing" || name == "orchestration" {
						controller = "fak"
					}
					components[name] = NetWorkComponent{Controller: controller, DurationNS: 0, Authoritative: true}
				}
				components["model_prefill"] = NetWorkComponent{Controller: "runtime", DurationNS: base + noise, Authoritative: true}
				cell.Repetitions = append(cell.Repetitions, NetWorkRepetition{Receipt: fmtReceipt(width, treatment, i), Accepted: true, OutcomeDigest: "accepted", InputTokens: tokens, CachedTokens: map[bool]int64{true: 200}[treatment == "prefix-only" || treatment == "combined"], WallLatencyNS: base + noise + 5, Components: components})
			}
			campaign.Cells = append(campaign.Cells, cell)
		}
	}
	return campaign
}

func fmtReceipt(width int, treatment string, repetition int) string {
	return treatment + string(rune('0'+width)) + string(rune('0'+repetition))
}
