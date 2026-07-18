package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestNegationOperatorCrossModel(t *testing.T) {
	data, err := os.ReadFile("testdata/negation_cross_model.json")
	if err != nil {
		t.Fatal(err)
	}
	var probes []NegationCheckpointProbe
	if err := json.Unmarshal(data, &probes); err != nil {
		t.Fatal(err)
	}
	if len(probes) == 0 {
		t.Fatal("negation transfer corpus is empty")
	}
	seenCheckpoints := map[string]bool{}
	for _, probe := range probes {
		if probe.Checkpoint == "" || probe.Family == "" || len(probe.Pairs) == 0 {
			t.Fatalf("incomplete checkpoint probe: %+v", probe)
		}
		if seenCheckpoints[probe.Checkpoint] {
			t.Fatalf("duplicate checkpoint probe %q", probe.Checkpoint)
		}
		seenCheckpoints[probe.Checkpoint] = true
	}

	anchor, err := FitNegationTransferDirection(probes[0].Pairs)
	if err != nil {
		t.Fatal(err)
	}
	if len(anchor) != 2 || anchor[0] != 1 || anchor[1] != 0 {
		t.Fatalf("pinned anchor artifact=%v, want [1 0]", anchor)
	}
	const threshold = .1
	rows, err := EvaluateNegationTransfer(anchor, probes, threshold)
	if err != nil {
		t.Fatal(err)
	}

	families := map[string]bool{}
	for _, row := range rows {
		families[row.Family] = true
		if row.Checkpoint == "glm" {
			if !row.RequiresRefit || math.Abs(row.TransferError-.8) > 1e-12 || row.RefitError > 1e-12 {
				t.Fatalf("honest GLM refit row=%+v", row)
			}
		} else if row.RequiresRefit || row.RefitGain > threshold {
			t.Fatalf("unexpected per-checkpoint refit row=%+v", row)
		}
		t.Logf("checkpoint=%-10s family=%-7s transfer_error=%.4f refit_error=%.4f refit_gain=%.4f requires_refit=%v", row.Checkpoint, row.Family, row.TransferError, row.RefitError, row.RefitGain, row.RequiresRefit)
	}
	for _, family := range []string{"llama", "gemma", "qwen", "mistral", "glm"} {
		if !families[family] {
			t.Fatalf("missing served family %s", family)
		}
	}
}
