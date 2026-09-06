package cachevaluereport

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

var determinismNow = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

func determinismTrack1Fixture() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		// Week 1 (2026-W25, 2026-06-15): 8 rejected accesses
		{
			Date:                 "2026-06-15",
			SessionType:          "guard",
			Turns:                5,
			PromptTokens:         1000,
			ReusedTokens:         600,
			RejectedTierAccesses: 5,
		},
		{
			Date:                 "2026-06-17",
			SessionType:          "serve",
			Turns:                4,
			PromptTokens:         800,
			ReusedTokens:         400,
			RejectedTierAccesses: 3,
		},
		// Week 2 (2026-W26, 2026-06-22): 4 rejected accesses (total 12 >= 10)
		{
			Date:                 "2026-06-22",
			SessionType:          "guard",
			Turns:                6,
			PromptTokens:         1200,
			ReusedTokens:         900,
			RejectedTierAccesses: 4,
		},
	}
}

func determinismTrack2Fixture() []SavingsRow {
	return []SavingsRow{
		{
			Date:                "2026-06-15",
			SessionType:         "guard",
			Provider:            "anthropic",
			Mechanism:           "provider_prompt_cache",
			InputTokens:         2000,
			CacheCreationTokens: 5000,
			OutputTokens:        300,
			SavedTokenEquiv:     1000,
			NetSavedTokenEquiv:  1000,
			RebateUSD:           1.50,
			WritePremiumUSD:     0.50,
			SpendUSD:            0.25,
			CompactionSavedUSD:  0.10,
		},
		{
			Date:               "2026-06-22",
			SessionType:        "guard",
			Provider:           "anthropic",
			Mechanism:          "provider_prompt_cache",
			InputTokens:        1000,
			CacheReadTokens:    6000,
			OutputTokens:       300,
			SavedTokenEquiv:    5400,
			NetSavedTokenEquiv: 5400,
			RebateUSD:          4.00,
			WritePremiumUSD:    0.10,
			SpendUSD:           0.20,
			CompactionSavedUSD: 0.30,
		},
	}
}

func determinismShapesFixture() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		// Week 1 (2026-06-15)
		{Date: "2026-06-15", SessionType: "guard", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ReuseRatio: 0.0},
		{Date: "2026-06-15", SessionType: "guard", Turns: 5, PromptTokens: 1000, ReusedTokens: 200, ReuseRatio: 0.2},
		{Date: "2026-06-16", SessionType: "serve", Turns: 15, PromptTokens: 2000, ReusedTokens: 1600, ReuseRatio: 0.8},
		// Week 2 (2026-06-22)
		{Date: "2026-06-22", SessionType: "guard", Turns: 1, PromptTokens: 600, ReusedTokens: 0, ReuseRatio: 0.0},
		{Date: "2026-06-23", SessionType: "serve", Turns: 6, PromptTokens: 1200, ReusedTokens: 700, ReuseRatio: 0.58},
		{Date: "2026-06-24", SessionType: "guard", Turns: 18, PromptTokens: 3000, ReusedTokens: 2700, ReuseRatio: 0.9},
	}
}

func determinismCensusFixture() []WorkerRow {
	return []WorkerRow{
		{Worker: "worker-active-anthropic", Reached: true, Published: true, Active: true, Upgraded: 5, Wire: "anthropic"},
		{Worker: "worker-active-openai", Reached: true, Published: true, Active: true, Upgraded: 2, Wire: "openai"},
		{Worker: "worker-active-leverless", Reached: true, Published: true, Active: true, Upgraded: 0, Wire: ""},
		{Worker: "worker-passive", Reached: true, Published: true, Active: false, Upgraded: 0, Wire: "anthropic"},
		{Worker: "worker-unreached", Reached: false},
	}
}

// TestDeterminism runs the complete determinism test suite for internal/cachevaluereport.
func TestDeterminism(t *testing.T) {
	t.Run("FoldTwoTrack", testDeterminismFoldTwoTrack)
	t.Run("RenderTwoTrackMarkdown", testDeterminismMarkdownRendering)
	t.Run("Shapes", testDeterminismShapes)
	t.Run("Census", testDeterminismCensus)
	t.Run("ConcurrentRaceWitness", testDeterminismConcurrentRaceWitness)
}

// TestCacheValueReport_Determinism is an alias forwarding to TestDeterminism.
func TestCacheValueReport_Determinism(t *testing.T) {
	TestDeterminism(t)
}

func testDeterminismFoldTwoTrack(t *testing.T) {
	t1 := determinismTrack1Fixture()
	t2 := determinismTrack2Fixture()

	repA := FoldTwoTrack(t1, t2, determinismNow)
	repB := FoldTwoTrack(t1, t2, determinismNow)

	// Target operating envelope assertions:
	// 1. >= 2 report buckets
	if len(repA.Track1.Buckets) < 2 {
		t.Fatalf("operating envelope: want >= 2 Track 1 report buckets, got %d", len(repA.Track1.Buckets))
	}
	if len(repA.Track2) < 2 {
		t.Fatalf("operating envelope: want >= 2 Track 2 report buckets, got %d", len(repA.Track2))
	}

	// 2. >= 10 rejected accesses
	if repA.Track1.RejectedTierAccesses < 10 {
		t.Fatalf("operating envelope: want >= 10 rejected tier accesses, got %d", repA.Track1.RejectedTierAccesses)
	}

	// 3. >= 1 render mentions of rejected tier accesses in terminal output
	renderedA := RenderTwoTrack(repA)
	renderedB := RenderTwoTrack(repB)
	if renderedA != renderedB {
		t.Fatalf("RenderTwoTrack is not string-identical across identical runs:\n gotA:\n%s\n gotB:\n%s", renderedA, renderedB)
	}
	mentionCount := strings.Count(renderedA, "rejected tier accesses")
	if mentionCount < 1 {
		t.Fatalf("operating envelope: want >= 1 render mentions of rejected tier accesses, got %d in:\n%s", mentionCount, renderedA)
	}
	if !strings.Contains(renderedA, "rejected tier accesses: 12") {
		t.Fatalf("RenderTwoTrack missing exact rejected tier accesses line, got:\n%s", renderedA)
	}

	track1Render := Render(repA.Track1)
	if !strings.Contains(track1Render, "rejected tier accesses: 12") {
		t.Fatalf("Render(Track1) missing exact rejected tier accesses line, got:\n%s", track1Render)
	}

	// 4. reflect.DeepEqual assertion
	if !reflect.DeepEqual(repA, repB) {
		t.Fatalf("FoldTwoTrack is not deeply equal across runs:\n repA=%+v\n repB=%+v", repA, repB)
	}

	// 5. JSON byte-equality
	jsonA, errA := json.Marshal(repA)
	if errA != nil {
		t.Fatalf("json.Marshal(repA) error: %v", errA)
	}
	jsonB, errB := json.Marshal(repB)
	if errB != nil {
		t.Fatalf("json.Marshal(repB) error: %v", errB)
	}
	if !bytes.Equal(jsonA, jsonB) {
		t.Fatalf("FoldTwoTrack JSON is not byte-identical across runs:\n a=%s\n b=%s", jsonA, jsonB)
	}
	if !strings.Contains(string(jsonA), `"rejected_tier_accesses":12`) {
		t.Fatalf("FoldTwoTrack JSON omitted expected rejected_tier_accesses: %s", jsonA)
	}
}

func testDeterminismMarkdownRendering(t *testing.T) {
	t1 := determinismTrack1Fixture()
	t2 := determinismTrack2Fixture()

	repA := FoldTwoTrack(t1, t2, determinismNow)
	repB := FoldTwoTrack(t1, t2, determinismNow)

	mdA := RenderTwoTrackMarkdown(repA)
	mdB := RenderTwoTrackMarkdown(repB)

	if mdA != mdB {
		t.Fatalf("RenderTwoTrackMarkdown string mismatch across runs:\n mdA:\n%s\n mdB:\n%s", mdA, mdB)
	}
	if !bytes.Equal([]byte(mdA), []byte(mdB)) {
		t.Fatalf("RenderTwoTrackMarkdown byte mismatch across runs")
	}
	if len(mdA) == 0 {
		t.Fatal("RenderTwoTrackMarkdown returned empty output")
	}
	if !strings.Contains(mdA, "## cache-value P&L") {
		t.Fatalf("RenderTwoTrackMarkdown output missing header:\n%s", mdA)
	}
	if !strings.Contains(mdA, "### Track 1") || !strings.Contains(mdA, "### Track 2") {
		t.Fatalf("RenderTwoTrackMarkdown output missing track headers:\n%s", mdA)
	}
	if !strings.Contains(mdA, "### KPI") {
		t.Fatalf("RenderTwoTrackMarkdown output missing KPI table:\n%s", mdA)
	}
}

func testDeterminismShapes(t *testing.T) {
	rows := determinismShapesFixture()

	sA := FoldShapes(rows, determinismNow)
	sB := FoldShapes(rows, determinismNow)

	if !reflect.DeepEqual(sA, sB) {
		t.Fatalf("FoldShapes is not deeply equal across runs:\n sA=%+v\n sB=%+v", sA, sB)
	}

	sjA, errA := json.Marshal(sA)
	if errA != nil {
		t.Fatalf("json.Marshal(sA) error: %v", errA)
	}
	sjB, errB := json.Marshal(sB)
	if errB != nil {
		t.Fatalf("json.Marshal(sB) error: %v", errB)
	}
	if !bytes.Equal(sjA, sjB) {
		t.Fatalf("FoldShapes JSON is not byte-identical across runs:\n a=%s\n b=%s", sjA, sjB)
	}

	stA := FoldShapeTrend(rows, determinismNow)
	stB := FoldShapeTrend(rows, determinismNow)

	if !reflect.DeepEqual(stA, stB) {
		t.Fatalf("FoldShapeTrend is not deeply equal across runs:\n stA=%+v\n stB=%+v", stA, stB)
	}

	stjA, errA := json.Marshal(stA)
	if errA != nil {
		t.Fatalf("json.Marshal(stA) error: %v", errA)
	}
	stjB, errB := json.Marshal(stB)
	if errB != nil {
		t.Fatalf("json.Marshal(stB) error: %v", errB)
	}
	if !bytes.Equal(stjA, stjB) {
		t.Fatalf("FoldShapeTrend JSON is not byte-identical across runs:\n a=%s\n b=%s", stjA, stjB)
	}
}

func testDeterminismCensus(t *testing.T) {
	rows := determinismCensusFixture()

	cA := FoldCensus(rows, determinismNow)
	cB := FoldCensus(rows, determinismNow)

	if !reflect.DeepEqual(cA, cB) {
		t.Fatalf("FoldCensus is not deeply equal across runs:\n cA=%+v\n cB=%+v", cA, cB)
	}

	cjA, errA := json.Marshal(cA)
	if errA != nil {
		t.Fatalf("json.Marshal(cA) error: %v", errA)
	}
	cjB, errB := json.Marshal(cB)
	if errB != nil {
		t.Fatalf("json.Marshal(cB) error: %v", errB)
	}
	if !bytes.Equal(cjA, cjB) {
		t.Fatalf("FoldCensus JSON is not byte-identical across runs:\n a=%s\n b=%s", cjA, cjB)
	}

	rcA := RenderCensus(cA)
	rcB := RenderCensus(cB)

	if rcA != rcB {
		t.Fatalf("RenderCensus string mismatch across runs:\n rcA:\n%s\n rcB:\n%s", rcA, rcB)
	}
	if !bytes.Equal([]byte(rcA), []byte(rcB)) {
		t.Fatalf("RenderCensus byte mismatch across runs")
	}
}

func testDeterminismConcurrentRaceWitness(t *testing.T) {
	const workers = 16
	const iterations = 30

	t1Fixture := determinismTrack1Fixture()
	t2Fixture := determinismTrack2Fixture()
	shapesFixture := determinismShapesFixture()
	censusFixture := determinismCensusFixture()

	baseTwoTrack := FoldTwoTrack(t1Fixture, t2Fixture, determinismNow)
	baseTwoTrackJSON, err := json.Marshal(baseTwoTrack)
	if err != nil {
		t.Fatalf("marshal baseline two-track: %v", err)
	}
	baseTwoTrackRender := RenderTwoTrack(baseTwoTrack)
	baseMarkdownRender := RenderTwoTrackMarkdown(baseTwoTrack)

	baseShapes := FoldShapes(shapesFixture, determinismNow)
	baseShapesJSON, err := json.Marshal(baseShapes)
	if err != nil {
		t.Fatalf("marshal baseline shapes: %v", err)
	}

	baseShapeTrend := FoldShapeTrend(shapesFixture, determinismNow)
	baseShapeTrendJSON, err := json.Marshal(baseShapeTrend)
	if err != nil {
		t.Fatalf("marshal baseline shape trend: %v", err)
	}

	baseCensus := FoldCensus(censusFixture, determinismNow)
	baseCensusJSON, err := json.Marshal(baseCensus)
	if err != nil {
		t.Fatalf("marshal baseline census: %v", err)
	}
	baseCensusRender := RenderCensus(baseCensus)

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				var t1 []cachevalueledger.Row
				var t2 []SavingsRow
				var sf []cachevalueledger.Row
				var cf []WorkerRow

				if workerID%2 == 0 {
					t1 = append([]cachevalueledger.Row(nil), t1Fixture...)
					t2 = append([]SavingsRow(nil), t2Fixture...)
					sf = append([]cachevalueledger.Row(nil), shapesFixture...)
					cf = append([]WorkerRow(nil), censusFixture...)
				} else {
					t1 = t1Fixture
					t2 = t2Fixture
					sf = shapesFixture
					cf = censusFixture
				}

				// 1. FoldTwoTrack & renders
				rep := FoldTwoTrack(t1, t2, determinismNow)
				if rep.Track1.RejectedTierAccesses != 12 {
					t.Errorf("worker %d iter %d: rejected accesses = %d, want 12", workerID, i, rep.Track1.RejectedTierAccesses)
					return
				}
				if len(rep.Track1.Buckets) < 2 {
					t.Errorf("worker %d iter %d: buckets = %d, want >= 2", workerID, i, len(rep.Track1.Buckets))
					return
				}
				rawJSON, err := json.Marshal(rep)
				if err != nil {
					t.Errorf("worker %d iter %d: marshal two-track: %v", workerID, i, err)
					return
				}
				if !bytes.Equal(rawJSON, baseTwoTrackJSON) {
					t.Errorf("worker %d iter %d: two-track JSON mismatch", workerID, i)
					return
				}
				rendered := RenderTwoTrack(rep)
				if rendered != baseTwoTrackRender {
					t.Errorf("worker %d iter %d: two-track render mismatch", workerID, i)
					return
				}
				md := RenderTwoTrackMarkdown(rep)
				if md != baseMarkdownRender {
					t.Errorf("worker %d iter %d: markdown render mismatch", workerID, i)
					return
				}

				// 2. Shapes & ShapeTrend
				s := FoldShapes(sf, determinismNow)
				sJSON, err := json.Marshal(s)
				if err != nil {
					t.Errorf("worker %d iter %d: marshal shapes: %v", workerID, i, err)
					return
				}
				if !bytes.Equal(sJSON, baseShapesJSON) {
					t.Errorf("worker %d iter %d: shapes JSON mismatch", workerID, i)
					return
				}

				st := FoldShapeTrend(sf, determinismNow)
				stJSON, err := json.Marshal(st)
				if err != nil {
					t.Errorf("worker %d iter %d: marshal shape trend: %v", workerID, i, err)
					return
				}
				if !bytes.Equal(stJSON, baseShapeTrendJSON) {
					t.Errorf("worker %d iter %d: shape trend JSON mismatch", workerID, i)
					return
				}

				// 3. Census & RenderCensus
				c := FoldCensus(cf, determinismNow)
				cJSON, err := json.Marshal(c)
				if err != nil {
					t.Errorf("worker %d iter %d: marshal census: %v", workerID, i, err)
					return
				}
				if !bytes.Equal(cJSON, baseCensusJSON) {
					t.Errorf("worker %d iter %d: census JSON mismatch", workerID, i)
					return
				}
				cRender := RenderCensus(c)
				if cRender != baseCensusRender {
					t.Errorf("worker %d iter %d: census render mismatch", workerID, i)
					return
				}
			}
		}()
	}

	wg.Wait()
}
