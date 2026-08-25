package agentreadinessscore

import (
	"strings"
	"testing"
)

func healthyLaunchSource() string {
	return `
{Command: "codex", Provider: "codex", Role: "canonical", Pipeline: []string{"managed-shim", "fak launch codex", "fak guard", "recorded-provider"}}
{Command: "fak m codex", Provider: "codex", Role: "noncanonical", Pipeline: []string{"fak manage", "fak guard", "PATH codex"}}
{Command: "fak codex", Provider: "codex", Role: "specialized", Pipeline: []string{"freshness-admission", "loop-gate", "fak guard", "PATH codex"}}
`
}
func healthyLaunchDoc() string {
	return "| `codex` | **Canonical** zero-adoption |\n| `fak m codex` | Noncanonical general |\n| `fak codex` | Specialized Codex loop |"
}
func TestLaunchEntryContractHealthy(t *testing.T) {
	if got := launchEntryContractDefects(healthyLaunchSource(), healthyLaunchDoc()); len(got) != 0 {
		t.Fatalf("defects=%v", got)
	}
}
func TestLaunchEntryContractDuplicateCanonical(t *testing.T) {
	got := launchEntryContractDefects(healthyLaunchSource()+`{Command: "other", Provider: "codex", Role: "canonical"}`, healthyLaunchDoc())
	if joined := strings.Join(got, "\n"); !strings.Contains(joined, "2 recognized canonical") {
		t.Fatalf("defects=%v", got)
	}
}

func TestLaunchEntryContractMissingAndDocsDrift(t *testing.T) {
	got := launchEntryContractDefects(strings.Replace(healthyLaunchSource(), `Role: "specialized"`, `Role: "canonical"`, 1), strings.Replace(healthyLaunchDoc(), "Noncanonical", "Canonical", 1))
	joined := strings.Join(got, "\n")
	for _, want := range []string{"source missing specialized role", "operator table missing noncanonical row"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %v", want, got)
		}
	}
}
func TestLaunchEntryContractMissingSource(t *testing.T) {
	got := launchEntryContractDefects("", healthyLaunchDoc())
	if len(got) != 1 || !strings.Contains(got[0], "machine-readable") {
		t.Fatalf("defects=%v", got)
	}
}
