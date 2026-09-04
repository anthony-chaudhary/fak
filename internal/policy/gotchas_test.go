package policy

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestDefaultDangerousGotchasRules(t *testing.T) {
	rules := DefaultDangerousGotchasRules()
	if len(rules) == 0 {
		t.Fatal("DefaultDangerousGotchasRules returned empty rules")
	}

	// Verify all rules compile cleanly
	preds, err := compileArgRules(rules)
	if err != nil {
		t.Fatalf("compileArgRules(DefaultDangerousGotchasRules) failed: %v", err)
	}
	if len(preds) != len(rules) {
		t.Errorf("compiled %d preds, want %d", len(preds), len(rules))
	}

	// Test tool-specific generator
	bashRules := DangerousGotchasRulesForTool("Bash", "command")
	if len(bashRules) == 0 {
		t.Fatal("DangerousGotchasRulesForTool(Bash, command) returned empty rules")
	}
	bashPreds, err := compileArgRules(bashRules)
	if err != nil {
		t.Fatalf("compileArgRules(bashRules) failed: %v", err)
	}
	if len(bashPreds) != len(bashRules) {
		t.Errorf("compiled %d bashPreds, want %d", len(bashPreds), len(bashRules))
	}

	// Verify gotchas catalog re-export
	catalog := DangerousGotchas()
	if len(catalog) != len(adjudicator.DangerousGotchas()) {
		t.Errorf("DangerousGotchas len = %d, want %d", len(catalog), len(adjudicator.DangerousGotchas()))
	}
}
