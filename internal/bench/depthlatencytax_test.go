package bench

import (
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestDepthLatencyTax(t *testing.T) {
	file, err := os.Open("testdata/negation_depth_latency.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	rows, err := ReadDepthLatencyJSONL(file)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 6 {
		t.Fatalf("rows=%d, want six matched arm rows", len(rows))
	}
	deltas, err := ValidateDepthLatencyTax(rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, delta := range deltas {
		t.Logf("DEPTH_LATENCY_TAX pair=%s depth_saved=%d latency_saved_ms=%.6f accuracy_delta=%+.3f", delta.PairID, delta.DepthDelta, delta.LatencyDeltaMS, delta.AccuracyDelta)
	}
	if len(deltas) != 3 {
		t.Fatalf("deltas=%d, want three matched pairs", len(deltas))
	}
}

func TestDepthLatencyTaxCapture(t *testing.T) {
	out := os.Getenv("FAK_DEPTH_LATENCY_OUT")
	if out == "" {
		t.Skip("set FAK_DEPTH_LATENCY_OUT to capture an observed JSONL witness")
	}
	data, err := os.ReadFile("../model/testdata/negation_depth_cost.json")
	if err != nil {
		t.Fatal(err)
	}
	var source struct {
		Pairs []struct {
			PairID                 string      `json:"pair_id"`
			TargetToken            int         `json:"target_token"`
			AffirmativeLayerLogits [][]float32 `json:"affirmative_layer_logits"`
			NegatedLayerLogits     [][]float32 `json:"negated_layer_logits"`
		} `json:"pairs"`
	}
	if err := json.Unmarshal(data, &source); err != nil {
		t.Fatal(err)
	}
	host, _ := os.Hostname()
	file, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	const repetitions = 500000
	written := 0
	for _, pair := range source.Pairs {
		operatorDepth, operatorOK := model.CrystallizationDepth(pair.AffirmativeLayerLogits, pair.TargetToken)
		naiveDepth, naiveOK := model.CrystallizationDepth(pair.NegatedLayerLogits, pair.TargetToken)
		if !operatorOK || !naiveOK || naiveDepth <= operatorDepth {
			continue // cost-tax witness uses only pairs whose measured depth can be offloaded
		}
		arms := []struct {
			condition string
			depth     int
			logits    [][]float32
		}{
			{DepthLatencyNaiveArm, naiveDepth, pair.NegatedLayerLogits},
			{DepthLatencyOperatorArm, operatorDepth, pair.AffirmativeLayerLogits},
		}
		for _, arm := range arms {
			start := time.Now()
			checksum := replayLayerPrefix(arm.logits, arm.depth, repetitions)
			elapsedMS := float64(time.Since(start).Nanoseconds()) / 1e6 / repetitions
			if checksum == 0 {
				t.Fatal("dead replay checksum")
			}
			row := DepthLatencyRow{
				Schema: "fak-negation-depth-latency/1", PairID: pair.PairID, Condition: arm.condition,
				Depth: arm.depth, LatencyMS: elapsedMS, Accuracy: 1,
				Model: "fak-synthetic-r1", Host: host + "/" + runtime.GOOS + "/" + runtime.GOARCH,
				Surface: "bounded layer-logit prefix replay", Provenance: "OBSERVED deterministic replay", Repetitions: repetitions,
			}
			if err := encoder.Encode(row); err != nil {
				t.Fatal(err)
			}
			written++
		}
	}
	if written < 2 {
		t.Fatal("no positive depth-tax pairs captured")
	}
}

func replayLayerPrefix(logits [][]float32, depth, repetitions int) float64 {
	var checksum float64
	for repetition := 0; repetition < repetitions; repetition++ {
		for layer := 0; layer <= depth; layer++ {
			for _, logit := range logits[layer] {
				checksum += float64(logit)
			}
		}
	}
	return checksum
}

func TestDepthLatencyTaxRefusesBadEvidence(t *testing.T) {
	valid := func() []DepthLatencyRow {
		base := DepthLatencyRow{Schema: "fak-negation-depth-latency/1", PairID: "p", Depth: 3, LatencyMS: 2, Accuracy: 1, Model: "m", Host: "h", Surface: "s", Provenance: "OBSERVED", Repetitions: 10}
		operator := base
		base.Condition = DepthLatencyNaiveArm
		operator.Condition = DepthLatencyOperatorArm
		operator.Depth = 2
		return []DepthLatencyRow{base, operator}
	}
	tests := []struct {
		name string
		edit func([]DepthLatencyRow) []DepthLatencyRow
		want string
	}{
		{"missing arm", func(rows []DepthLatencyRow) []DepthLatencyRow { return rows[:1] }, "missing an arm"},
		{"accuracy regression", func(rows []DepthLatencyRow) []DepthLatencyRow { rows[1].Accuracy = 0; return rows }, "accuracy regressed"},
		{"no saving", func(rows []DepthLatencyRow) []DepthLatencyRow { rows[1].Depth = 3; rows[1].LatencyMS = 2; return rows }, "reduced neither"},
		{"workload mismatch", func(rows []DepthLatencyRow) []DepthLatencyRow { rows[1].Model = "other"; return rows }, "identical-workload"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ValidateDepthLatencyTax(tt.edit(valid()))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got err %v, want %q", err, tt.want)
			}
		})
	}
}
