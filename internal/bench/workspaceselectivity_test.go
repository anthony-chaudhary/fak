package bench

import "testing"

func TestWorkspaceSelectivityAblation(t *testing.T) {
	report := WorkspaceSelectivityAblation("preserve contract tokens and complete the task", []int{1, 2, 4, 8, 16})
	if report.Provenance != WorkspaceSelectivityProvenance || len(report.Rows) != 10 {
		t.Fatalf("report=%+v", report)
	}
	var firstExternalCost int
	var firstModelCost, lastModelCost int
	var nonDegenerate bool
	for i := 0; i < len(report.Rows); i += 2 {
		ext, model := report.Rows[i], report.Rows[i+1]
		t.Logf("WORKSPACE_SELECTIVITY density=%d arm=%s workspace_tokens=%d fidelity=%.3f task_accuracy=%.3f preserved=%d/%d", ext.Density, ext.Arm, ext.WorkspaceTokens, ext.Fidelity, ext.TaskAccuracy, ext.PreservedTokens, ext.ContractTokens)
		t.Logf("WORKSPACE_SELECTIVITY density=%d arm=%s workspace_tokens=%d fidelity=%.3f task_accuracy=%.3f preserved=%d/%d", model.Density, model.Arm, model.WorkspaceTokens, model.Fidelity, model.TaskAccuracy, model.PreservedTokens, model.ContractTokens)
		if i == 0 {
			firstExternalCost = ext.WorkspaceTokens
			firstModelCost = model.WorkspaceTokens
		}
		lastModelCost = model.WorkspaceTokens
		if ext.WorkspaceTokens != firstExternalCost {
			t.Fatalf("external cost not flat: first=%d row=%+v", firstExternalCost, ext)
		}
		if ext.PreservedTokens != ext.ContractTokens {
			t.Fatalf("external token fail-safe failed: %+v", ext)
		}
		if ext.Fidelity < model.Fidelity || ext.TaskAccuracy < model.TaskAccuracy {
			t.Fatalf("external lost at density %d: ext=%+v model=%+v", ext.Density, ext, model)
		}
		if ext.TaskAccuracy != model.TaskAccuracy {
			nonDegenerate = true
		}
	}
	if lastModelCost <= firstModelCost*4 {
		t.Fatalf("model-side cost did not grow with density: first=%d last=%d", firstModelCost, lastModelCost)
	}
	if !nonDegenerate {
		t.Fatal("task-accuracy delta is degenerate")
	}
}
