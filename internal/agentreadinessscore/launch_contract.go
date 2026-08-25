package agentreadinessscore

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

type launchEntryContract struct {
	command  string
	role     string
	pipeline []string
}

var codexLaunchContract = []launchEntryContract{
	{"codex", "canonical", []string{"managed-shim", "fak launch codex", "fak guard", "recorded-provider"}},
	{"fak m codex", "noncanonical", []string{"fak manage", "fak guard", "PATH codex"}},
	{"fak codex", "specialized", []string{"freshness-admission", "loop-gate", "fak guard", "PATH codex"}},
}

func launchEntryContractKPI(root string) KPI {
	source := safeRead(root, "cmd/fak/launch_doctor.go")
	doc := safeRead(root, "docs/zero-adoption-launch.md")
	defects := launchEntryContractDefects(source, doc)
	if defects == nil {
		defects = []string{}
	}
	detail := "Codex launch entry-point source and operator table agree"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d launch entry-point contract defect(s)", len(defects))
	}
	return KPI{"launch_entry_contract", "adopt", mathx.ClampScore(100 - 25*float64(len(defects))), detail, defects, []string{}}
}

func launchEntryContractDefects(source, doc string) []string {
	var defects []string
	if strings.TrimSpace(source) == "" {
		return []string{"cmd/fak/launch_doctor.go missing — no machine-readable launch contract"}
	}
	if strings.TrimSpace(doc) == "" {
		defects = append(defects, "docs/zero-adoption-launch.md missing — no operator launch contract")
	}
	canonical := strings.Count(source, `Provider: "codex", Role: "canonical"`)
	for _, want := range codexLaunchContract {
		sourceRow := fmt.Sprintf(`Command: %q, Provider: "codex", Role: %q`, want.command, want.role)
		if !strings.Contains(source, sourceRow) {
			defects = append(defects, fmt.Sprintf("source missing %s role for %q", want.role, want.command))
		}
		for _, step := range want.pipeline {
			if !strings.Contains(source, fmt.Sprintf("%q", step)) {
				defects = append(defects, fmt.Sprintf("source pipeline for %q missing %q", want.command, step))
			}
		}
		docRole := want.role
		if want.role == "canonical" {
			docRole = "**Canonical**"
		} else if want.role == "noncanonical" {
			docRole = "Noncanonical"
		} else {
			docRole = "Specialized"
		}
		if !strings.Contains(doc, "| `"+want.command+"` | "+docRole) {
			defects = append(defects, fmt.Sprintf("operator table missing %s row for %q", want.role, want.command))
		}
	}
	if canonical != 1 {
		defects = append(defects, fmt.Sprintf("source has %d recognized canonical Codex front doors; want exactly 1", canonical))
	}
	return dedup(defects)
}
