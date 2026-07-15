package bench

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

const WorkspaceSelectivityProvenance = "MODELED / HERMETIC REPLAY"

type WorkspaceSelectivityRow struct {
	Density         int     `json:"density"`
	Arm             string  `json:"arm"`
	WorkspaceTokens int     `json:"workspace_tokens"`
	Fidelity        float64 `json:"fidelity"`
	TaskAccuracy    float64 `json:"task_accuracy"`
	ContractTokens  int     `json:"contract_tokens"`
	PreservedTokens int     `json:"preserved_tokens"`
}

type WorkspaceSelectivityReport struct {
	Provenance string                    `json:"provenance"`
	Goal       string                    `json:"goal"`
	Rows       []WorkspaceSelectivityRow `json:"rows"`
}

// WorkspaceSelectivityAblation replays a fixed hermetic goal while sweeping
// transform density. EXTERNAL performs the deterministic fold before emit;
// MODEL_SIDE accounts for source + rule + rewritten result in the workspace and
// uses an explicit replay outcome rather than claiming live-model behavior.
func WorkspaceSelectivityAblation(goal string, densities []int) WorkspaceSelectivityReport {
	report := WorkspaceSelectivityReport{Provenance: WorkspaceSelectivityProvenance, Goal: goal}
	for _, density := range densities {
		if density < 1 {
			continue
		}
		raw, contracts := workspaceSelectivityPrompt(density)
		external := negframe.Reframe(raw)
		externalPreserved := countContractTokens(external, contracts)
		externalFidelity := transformFidelity(raw, external)
		report.Rows = append(report.Rows, WorkspaceSelectivityRow{Density: density, Arm: "EXTERNAL", WorkspaceTokens: 6, Fidelity: externalFidelity, TaskAccuracy: externalFidelity, ContractTokens: len(contracts), PreservedTokens: externalPreserved})

		// The model-side arm is a hermetic bounded replay, not a model claim: its
		// in-band transformer can rewrite three directives while the rest remain
		// unresolved in the shared workspace. Fidelity and task accuracy are
		// scored from the replayed bytes, never assigned from a cost formula.
		modelOutput := boundedModelSideTransform(raw, 3)
		modelPreserved := countContractTokens(modelOutput, contracts)
		modelFidelity := transformFidelity(raw, modelOutput)
		workspaceTokens := (len(raw) + len("Rewrite each directive positively and preserve every contract token.") + len(modelOutput) + 3) / 4
		report.Rows = append(report.Rows, WorkspaceSelectivityRow{Density: density, Arm: "MODEL_SIDE", WorkspaceTokens: workspaceTokens, Fidelity: modelFidelity, TaskAccuracy: modelFidelity, ContractTokens: len(contracts), PreservedTokens: modelPreserved})
	}
	return report
}

func boundedModelSideTransform(raw string, capacity int) string {
	lines := strings.Split(raw, "\n")
	for i := range lines {
		if i < capacity {
			lines[i] = negframe.Reframe(lines[i])
		}
	}
	return strings.Join(lines, "\n")
}

func transformFidelity(raw, transformed string) float64 {
	before := negationMarkerCount(raw)
	if before == 0 {
		return 1
	}
	after := negationMarkerCount(transformed)
	fidelity := 1 - float64(after)/float64(before)
	if fidelity < 0 {
		return 0
	}
	return fidelity
}

func workspaceSelectivityPrompt(density int) (string, []string) {
	var lines, contracts []string
	for i := 0; i < density; i++ {
		token := "REASON_" + itoaBench(i)
		contracts = append(contracts, token)
		lines = append(lines, "Do not forget to keep "+token+". Do not hesitate to carry "+token+".")
	}
	return strings.Join(lines, "\n"), contracts
}

func countContractTokens(text string, tokens []string) int {
	n := 0
	for _, token := range tokens {
		if strings.Contains(text, token) {
			n++
		}
	}
	return n
}
func itoaBench(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func negationMarkerCount(text string) int {
	lower := strings.ToLower(text)
	return strings.Count(lower, "do not forget") + strings.Count(lower, "do not hesitate")
}
