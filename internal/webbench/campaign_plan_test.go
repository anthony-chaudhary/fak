package webbench

// Tests for the K3.1 bench-campaign arm-matrix plan and per-arm receipt fold
// (#14374). Fixtures are synthetic fak.serving-sweep.v1 JSON artifacts — the
// exact shape WriteServingSweepReport emits, plus the optional additive spec
// join — so the fold is witnessed on deterministic bytes, never wall-clock
// timing noise.

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// campaignSyntheticArmReport renders one per-arm fak.serving-sweep.v1 artifact
// as raw JSON and injects the additive spec join into the ours-track points
// named in acceptance (no join when the map is nil or empty).
func campaignSyntheticArmReport(t *testing.T, model, workloadDigest, engine string, points []syntheticSweepPoint, acceptance map[int]float64) []byte {
	t.Helper()
	const capacity = 128
	engineReceipt := "sha256:" + strings.Repeat("1", 64)
	contract := ServingSweepTrackContract{
		Track:               TrackOurs,
		Model:               model,
		Engine:              engine,
		EngineReceiptDigest: engineReceipt,
		BatchCapacity:       capacity,
		CapacitySource:      "fixture:declared",
	}
	report := &ServingSweepReport{
		Schema:    ServingSweepSchema,
		Model:     model,
		Workload:  ServingSweepWorkload{Digest: workloadDigest, Requests: 8},
		Contracts: []ServingSweepTrackContract{contract},
	}
	for _, point := range points {
		throughput, ttft, itl := point.throughput, point.ttftP99, point.itlP99
		report.Workload.Concurrencies = append(report.Workload.Concurrencies, point.concurrency)
		report.Points = append(report.Points, ServingSweepPoint{
			Concurrency:    point.concurrency,
			WorkloadDigest: workloadDigest,
			Tracks: []ServingSweepTrackPoint{{
				Track:               TrackOurs,
				Status:              "measured",
				MeasurementStatus:   "measured",
				Model:               contract.Model,
				Engine:              contract.Engine,
				EngineReceiptDigest: contract.EngineReceiptDigest,
				BatchCapacity:       contract.BatchCapacity,
				CapacitySource:      contract.CapacitySource,
				Stats: ServingStats{
					OK:                1,
					TokenCountBasis:   "usage.completion_tokens",
					ThroughputTokensS: ScalarMetric{Status: "measured", Unit: "usage.completion_tokens/s", Value: &throughput},
					TTFTMillis:        QuantileMetric{Status: "measured", P99: &ttft},
					ITLMillis:         QuantileMetric{Status: "measured", P99: &itl},
				},
			}},
		})
	}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if len(acceptance) == 0 {
		return data
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	rawPoints, ok := raw["points"].([]any)
	if !ok {
		t.Fatal("synthetic report JSON has no points array")
	}
	for _, rawPoint := range rawPoints {
		pointMap, ok := rawPoint.(map[string]any)
		if !ok {
			t.Fatal("synthetic report point is not a JSON object")
		}
		concurrency := int(pointMap["concurrency"].(float64))
		acceptanceLength, inject := acceptance[concurrency]
		if !inject {
			continue
		}
		tracks, ok := pointMap["tracks"].([]any)
		if !ok {
			t.Fatal("synthetic report point has no tracks array")
		}
		for _, rawTrack := range tracks {
			trackMap, ok := rawTrack.(map[string]any)
			if !ok {
				t.Fatal("synthetic report track is not a JSON object")
			}
			if trackMap["track"] == string(TrackOurs) {
				trackMap["spec"] = map[string]any{"acceptance_length": acceptanceLength}
			}
		}
	}
	injected, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	return injected
}

// fak-test:runtime fast est=1ms lane=default
func TestK31CampaignPlanShapeIsDeterministic(t *testing.T) {
	plan := K31CampaignPlan()
	if plan.Schema != CampaignPlanSchema || plan.ReceiptSchema != ServingSweepSchema || plan.CampaignIssue != CampaignIssue {
		t.Fatalf("plan identity = %#v", plan)
	}
	if want := []string{"#14368", "#14371", "#14370"}; !reflect.DeepEqual(plan.PromotionRequires, want) {
		t.Fatalf("promotion requires = %v, want %v", plan.PromotionRequires, want)
	}
	if len(plan.SpecArms) != 12 || len(plan.OffloadArms) != 3 {
		t.Fatalf("arm matrix = %d spec points / %d offload placeholders, want 12 / 3", len(plan.SpecArms), len(plan.OffloadArms))
	}
	wantArms := []struct {
		name, config string
		concurrency  int
	}{
		{CampaignArmSpecNone, CampaignSpecConfigNone, 1},
		{CampaignArmSpecNone, CampaignSpecConfigNone, 8},
		{CampaignArmSpecNone, CampaignSpecConfigNone, 32},
		{CampaignArmSpecNone, CampaignSpecConfigNone, 64},
		{CampaignArmSpecDSpark4, CampaignSpecConfigDSpark4, 1},
		{CampaignArmSpecDSpark4, CampaignSpecConfigDSpark4, 8},
		{CampaignArmSpecDSpark4, CampaignSpecConfigDSpark4, 32},
		{CampaignArmSpecDSpark4, CampaignSpecConfigDSpark4, 64},
		{CampaignArmSpecDSpark5, CampaignSpecConfigDSpark5, 1},
		{CampaignArmSpecDSpark5, CampaignSpecConfigDSpark5, 8},
		{CampaignArmSpecDSpark5, CampaignSpecConfigDSpark5, 32},
		{CampaignArmSpecDSpark5, CampaignSpecConfigDSpark5, 64},
	}
	for i, want := range wantArms {
		got := plan.SpecArms[i]
		if got.Name != want.name || got.SpeculativeConfig != want.config || got.Concurrency != want.concurrency {
			t.Fatalf("spec arm %d = %#v, want %s/%s@c%d", i, got, want.name, want.config, want.concurrency)
		}
	}
	wantOffload := []struct {
		name string
		hot  float64
	}{
		{"offload-h050", 0.5},
		{"offload-h075", 0.75},
		{"offload-h085", 0.85},
	}
	for i, want := range wantOffload {
		got := plan.OffloadArms[i]
		if got.Name != want.name || got.HotFraction != want.hot || got.Status != CampaignArmStatusGated || got.Gate != CampaignOffloadGate {
			t.Fatalf("offload arm %d = %#v, want %s h=%v gated on %s", i, got, want.name, want.hot, CampaignOffloadGate)
		}
	}
	if !reflect.DeepEqual(K31CampaignPlan(), plan) {
		t.Fatal("K31CampaignPlan is not deterministic across calls")
	}
	rendered, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip CampaignPlan
	if err := json.Unmarshal(rendered, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&roundTrip, plan) {
		t.Fatal("campaign plan JSON round trip changed the plan")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFoldCampaignIndexProjectsArmReceiptsOntoPlanMatrix(t *testing.T) {
	var (
		model        = "qwen3.8"
		workload     = "sha256:" + strings.Repeat("a", 64)
		noneEngine   = "fak-native"
		dsparkEngine = "fak-native-dspark4"
	)
	noneReport := campaignSyntheticArmReport(t, model, workload, noneEngine, []syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 8, throughput: 40, ttftP99: 25, itlP99: 6},
		{concurrency: 32, throughput: 90, ttftP99: 40, itlP99: 8},
		{concurrency: 64, throughput: 140, ttftP99: 60, itlP99: 10},
	}, nil)
	dspark4Report := campaignSyntheticArmReport(t, model, workload, dsparkEngine, []syntheticSweepPoint{
		{concurrency: 1, throughput: 8, ttftP99: 18, itlP99: 4},
		{concurrency: 8, throughput: 16, ttftP99: 30, itlP99: 6},
		{concurrency: 32, throughput: 30, ttftP99: 45, itlP99: 7},
		{concurrency: 64, throughput: 45, ttftP99: 70, itlP99: 9},
	}, map[int]float64{1: 2.5, 8: 3, 32: 3.2, 64: 3.1})
	index, err := FoldCampaignIndex(K31CampaignPlan(), []CampaignArmReport{
		{Arm: CampaignArmSpecNone, Report: noneReport},
		{Arm: CampaignArmSpecDSpark4, Report: dspark4Report},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index.Schema != CampaignIndexSchema || index.CampaignIssue != CampaignIssue || index.ReceiptSchema != ServingSweepSchema {
		t.Fatalf("index identity = %#v", index)
	}
	if index.Model != model || !strings.HasPrefix(index.WorkloadDigest, "sha256:") {
		t.Fatalf("index comparability identity = model %q workload %q", index.Model, index.WorkloadDigest)
	}
	if len(index.Rows) != 12 {
		t.Fatalf("index rows = %d, want 12 (offload placeholders excluded until #14372)", len(index.Rows))
	}
	for i, want := range []struct {
		arm string
		c   int
	}{
		{CampaignArmSpecNone, 1}, {CampaignArmSpecNone, 8}, {CampaignArmSpecNone, 32}, {CampaignArmSpecNone, 64},
		{CampaignArmSpecDSpark4, 1}, {CampaignArmSpecDSpark4, 8}, {CampaignArmSpecDSpark4, 32}, {CampaignArmSpecDSpark4, 64},
		{CampaignArmSpecDSpark5, 1}, {CampaignArmSpecDSpark5, 8}, {CampaignArmSpecDSpark5, 32}, {CampaignArmSpecDSpark5, 64},
	} {
		if row := index.Rows[i]; row.Arm != want.arm || row.Concurrency != want.c {
			t.Fatalf("row %d = %s@c%d, want %s@c%d (row order must follow the plan)", i, row.Arm, row.Concurrency, want.arm, want.c)
		}
	}
	for i := 0; i < 4; i++ {
		row := index.Rows[i]
		if row.Status != CampaignRowMeasured {
			t.Fatalf("none row c=%d status = %q (%s), want measured", row.Concurrency, row.Status, row.Reason)
		}
		if row.AcceptanceLength != nil {
			t.Fatalf("none row c=%d invented an acceptance_length %v", row.Concurrency, *row.AcceptanceLength)
		}
		if row.SingleStreamTokS == nil || *row.SingleStreamTokS != 10 {
			t.Fatalf("none row c=%d single-stream anchor = %v, want the arm c=1 point 10", row.Concurrency, row.SingleStreamTokS)
		}
	}
	if got := index.Rows[3]; *got.AggregateTokS != 140 || *got.TTFTP99Millis != 60 || *got.ITLP99Millis != 10 {
		t.Fatalf("none row c=64 metrics = %v/%v/%v, want 140/60/10", *got.AggregateTokS, *got.TTFTP99Millis, *got.ITLP99Millis)
	}
	for i := 4; i < 8; i++ {
		row := index.Rows[i]
		if row.Status != CampaignRowMeasured || row.AcceptanceLength == nil {
			t.Fatalf("dspark4 row c=%d = status %q acceptance %v, want measured with the spec join", row.Concurrency, row.Status, row.AcceptanceLength)
		}
		if row.SingleStreamTokS == nil || *row.SingleStreamTokS != 8 {
			t.Fatalf("dspark4 row c=%d single-stream anchor = %v, want the arm c=1 point 8", row.Concurrency, row.SingleStreamTokS)
		}
	}
	if got := index.Rows[5]; *got.AggregateTokS != 16 || *got.TTFTP99Millis != 30 || *got.ITLP99Millis != 6 || *got.AcceptanceLength != 3 {
		t.Fatalf("dspark4 row c=8 = %v/%v/%v/%v, want 16/30/6/3", *got.AggregateTokS, *got.TTFTP99Millis, *got.ITLP99Millis, *got.AcceptanceLength)
	}
	for i := 8; i < 12; i++ {
		row := index.Rows[i]
		if row.Status != CampaignRowMissing || row.Reason == "" {
			t.Fatalf("missing row c=%d = status %q reason %q, want missing with a reason", row.Concurrency, row.Status, row.Reason)
		}
		if row.SingleStreamTokS != nil || row.AggregateTokS != nil || row.TTFTP99Millis != nil || row.ITLP99Millis != nil || row.AcceptanceLength != nil {
			t.Fatalf("missing row c=%d carries numbers %+v", row.Concurrency, row)
		}
	}
	markdown := index.Markdown()
	for arm, want := range map[string]int{CampaignArmSpecNone: 4, CampaignArmSpecDSpark4: 4, CampaignArmSpecDSpark5: 4} {
		if got := strings.Count(markdown, "| "+arm+" "); got != want {
			t.Fatalf("markdown has %d %s rows, want %d", got, arm, want)
		}
	}
	if !strings.Contains(markdown, "| spec-dspark4 | dspark:4 | 8 | 8 | 16 | 30 | 6 | 3 | measured |") {
		t.Fatalf("markdown missing the measured dspark4 c=8 row:\n%s", markdown)
	}
	if !strings.Contains(markdown, "| spec-dspark5 | dspark:5 | 64 | - | - | - | - | - | missing |") {
		t.Fatalf("markdown missing the missing dspark5 c=64 row:\n%s", markdown)
	}
	blob, err := index.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var rendered map[string]any
	if err := json.Unmarshal(blob, &rendered); err != nil {
		t.Fatal(err)
	}
	if rendered["schema"] != CampaignIndexSchema {
		t.Fatalf("index JSON schema = %v, want %q", rendered["schema"], CampaignIndexSchema)
	}
	rows, ok := rendered["rows"].([]any)
	if !ok || len(rows) != 12 {
		t.Fatalf("index JSON rows = %#v, want 12", rendered["rows"])
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFoldCampaignIndexSpecArmWithoutJoinFoldsPresentWithoutSpecMetrics(t *testing.T) {
	var (
		model    = "qwen3.8"
		workload = "sha256:" + strings.Repeat("a", 64)
	)
	dspark5Report := campaignSyntheticArmReport(t, model, workload, "fak-native", []syntheticSweepPoint{
		{concurrency: 1, throughput: 7, ttftP99: 19, itlP99: 4},
		{concurrency: 8, throughput: 14, ttftP99: 28, itlP99: 6},
		{concurrency: 32, throughput: 26, ttftP99: 44, itlP99: 7},
		{concurrency: 64, throughput: 38, ttftP99: 68, itlP99: 9},
	}, nil)
	index, err := FoldCampaignIndex(K31CampaignPlan(), []CampaignArmReport{
		{Arm: CampaignArmSpecDSpark5, Report: dspark5Report},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index.Rows[0].Status != CampaignRowMissing {
		t.Fatalf("unsupplied none arm = %q, want missing", index.Rows[0].Status)
	}
	for i := 8; i < 12; i++ {
		row := index.Rows[i]
		if row.Status != CampaignRowPresentWithoutSpecMetrics {
			t.Fatalf("dspark5 row c=%d status = %q, want present-without-spec-metrics", row.Concurrency, row.Status)
		}
		if !strings.Contains(row.Reason, "acceptance_length") {
			t.Fatalf("dspark5 row c=%d reason = %q, want the missing-join explanation", row.Concurrency, row.Reason)
		}
		if row.AcceptanceLength != nil {
			t.Fatalf("dspark5 row c=%d invented an acceptance_length %v", row.Concurrency, *row.AcceptanceLength)
		}
		if row.SingleStreamTokS == nil || row.AggregateTokS == nil {
			t.Fatalf("dspark5 row c=%d dropped its real measurements: %+v", row.Concurrency, row)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFoldCampaignIndexRefusesForeignDuplicateAndIncomparableArms(t *testing.T) {
	var workload = "sha256:" + strings.Repeat("a", 64)
	plan := K31CampaignPlan()
	points := []syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 8, throughput: 40, ttftP99: 25, itlP99: 6},
	}
	noneReport := campaignSyntheticArmReport(t, "qwen3.8", workload, "fak-native", points, nil)
	otherModelReport := campaignSyntheticArmReport(t, "other-model", workload, "fak-native", points, nil)
	otherWorkloadReport := campaignSyntheticArmReport(t, "qwen3.8", "sha256:"+strings.Repeat("b", 64), "fak-native", points, nil)
	for _, tt := range []struct {
		name    string
		plan    *CampaignPlan
		reports []CampaignArmReport
		wantErr string
	}{
		{name: "nil plan", plan: nil, wantErr: "nil campaign plan"},
		{name: "arm outside the plan matrix", plan: plan, reports: []CampaignArmReport{{Arm: "spec-typo", Report: noneReport}}, wantErr: "not in the plan matrix"},
		{name: "duplicate arm report", plan: plan, reports: []CampaignArmReport{{Arm: CampaignArmSpecNone, Report: noneReport}, {Arm: CampaignArmSpecNone, Report: noneReport}}, wantErr: "duplicate campaign arm report"},
		{name: "empty arm name", plan: plan, reports: []CampaignArmReport{{Arm: "   ", Report: noneReport}}, wantErr: "empty arm name"},
		{name: "model identity differs across arms", plan: plan, reports: []CampaignArmReport{{Arm: CampaignArmSpecNone, Report: noneReport}, {Arm: CampaignArmSpecDSpark4, Report: otherModelReport}}, wantErr: "share one model identity"},
		{name: "workload digest differs across arms", plan: plan, reports: []CampaignArmReport{{Arm: CampaignArmSpecNone, Report: noneReport}, {Arm: CampaignArmSpecDSpark4, Report: otherWorkloadReport}}, wantErr: "share one immutable workload"},
		{name: "unparseable report JSON", plan: plan, reports: []CampaignArmReport{{Arm: CampaignArmSpecNone, Report: []byte("{not json")}}, wantErr: "parse fak.serving-sweep.v1 JSON"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := FoldCampaignIndex(tt.plan, tt.reports)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("FoldCampaignIndex error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFoldCampaignIndexFoldsInvalidReceiptAsMissingWithTypedReason(t *testing.T) {
	noneReport := campaignSyntheticArmReport(t, "qwen3.8", "sha256:"+strings.Repeat("a", 64), "fak-native", []syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 8, throughput: 40, ttftP99: 25, itlP99: 6},
		{concurrency: 32, throughput: 90, ttftP99: 40, itlP99: 8},
		{concurrency: 64, throughput: 140, ttftP99: 60, itlP99: 10},
	}, nil)
	var raw map[string]any
	if err := json.Unmarshal(noneReport, &raw); err != nil {
		t.Fatal(err)
	}
	contracts, ok := raw["contracts"].([]any)
	if !ok || len(contracts) == 0 {
		t.Fatal("synthetic report JSON has no contracts array")
	}
	contracts[0].(map[string]any)["engine_receipt_digest"] = "sha256:short"
	corrupted, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	index, err := FoldCampaignIndex(K31CampaignPlan(), []CampaignArmReport{
		{Arm: CampaignArmSpecNone, Report: corrupted},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		row := index.Rows[i]
		if row.Status != CampaignRowMissing {
			t.Fatalf("invalid-receipt row c=%d status = %q, want missing (never measured)", row.Concurrency, row.Status)
		}
		if !strings.Contains(row.Reason, "engine_identity_unknown") {
			t.Fatalf("invalid-receipt row c=%d reason = %q, want the evaluator's typed reason", row.Concurrency, row.Reason)
		}
		if row.SingleStreamTokS != nil || row.AggregateTokS != nil {
			t.Fatalf("invalid-receipt row c=%d carries numbers %+v", row.Concurrency, row)
		}
	}
}
