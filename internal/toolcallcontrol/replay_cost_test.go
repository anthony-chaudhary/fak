package toolcallcontrol

import "testing"

func TestReplayChargesThinkingTurnAndCanBeNetNegative(t *testing.T) {
	rows := []ReplayRow{
		{ID: "seed", Turn: 1, Tool: "read_file", Args: []byte(`{"path":"a"}`), ReadOnly: true, StateEpoch: "s", PromptUnits: 100, Needed: boolp(true), ResultID: "r1", Succeeded: true, ControllerUnits: map[string]int64{"exact-reuse": 80}, CostBasis: "observed"},
		{ID: "repeat", Turn: 2, Tool: "read_file", Args: []byte(`{"path":"a"}`), ReadOnly: true, StateEpoch: "s", PromptUnits: 100, Needed: boolp(false), ResultID: "r2", Succeeded: true, ControllerUnits: map[string]int64{"exact-reuse": 80}, CostBasis: "observed"},
	}
	report := Replay(rows)
	exact, ok := report.Arm("exact-reuse")
	if !ok {
		t.Fatal("missing exact-reuse")
	}
	m := exact.Metrics
	if m.ReplayUnitsSaved != 100 || m.ControllerUnits != 160 || m.NetReplayValue != -60 || m.BreakEven {
		t.Fatalf("economics=%+v", m)
	}
	if m.CostBasis != "observed" {
		t.Fatalf("basis=%q", m.CostBasis)
	}
}

func TestReplayChargesFalseSuppressionRecoverySeparately(t *testing.T) {
	rows := []ReplayRow{{ID: "weak", Turn: 1, Tool: "read_file", Args: []byte(`{}`), ReadOnly: true, PromptUnits: 100, Needed: boolp(true), RecoveryUnits: 250, CostBasis: "scenario"}}
	arm, _ := Replay(rows).Arm("rationale-prefilter")
	if arm.Metrics.NeededSuppressed != 1 || arm.Metrics.FalseSuppressionRecoveryUnits != 250 || arm.Metrics.NetReplayValue != -150 {
		t.Fatalf("economics=%+v", arm.Metrics)
	}
}
