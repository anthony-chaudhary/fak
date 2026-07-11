package dispatchtick

import (
	"strings"
	"testing"
)

func TestObjectiveContractEmbedsWitnessedScorer(t *testing.T) {
	body := "## Working spine\nKeep dispatch throughput rising without collisions.\n\n## Witness\n`dos commit-audit` plus the dispatch timing ledger.\n\n## Out of scope\nworker narration"
	c := ParseObjectiveContract(body)
	if !c.Attached || c.Refusal != "" {
		t.Fatalf("contract=%+v, want attached", c)
	}
	prompt := RenderIssuePrompt(IssuePromptInput{Number: 2550, Lane: "dispatch", Body: body, ObjectiveContract: c})
	for _, want := range []string{"objective contract (kernel-authored", "Keep dispatch throughput rising", "attached scorers / witnessed progress", "dos commit-audit"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestObjectiveContractRefusesObjectiveWithoutScorer(t *testing.T) {
	c := ParseObjectiveContract("## Working spine\nIncrease autonomous throughput.\n\n## Done condition\nIt works.")
	if c.Attached || c.Refusal != RefuseObjectiveNoScorer {
		t.Fatalf("contract=%+v, want %s", c, RefuseObjectiveNoScorer)
	}
}

func TestObjectiveContractLegacyIssueAbstains(t *testing.T) {
	c := ParseObjectiveContract("fix the typo and add a regression test")
	if c.Attached || c.Refusal != "" {
		t.Fatalf("legacy contract=%+v, want abstain", c)
	}
}
