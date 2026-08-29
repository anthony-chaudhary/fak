package architest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnesskitUpgradeContractFromCleanModule(t *testing.T) {
	if testing.Short() {
		t.Skip("external module upgrade witness")
	}
	root := repositoryRoot(t)
	program := `package main
import (
    "encoding/json"
    "fmt"
    kit "github.com/anthony-chaudhary/fak/pkg/harnesskit"
)
func main() {
    builder := kit.BuilderContract{ContractVersion: kit.ContractVersion, Requirements: []kit.CapabilityRequirement{
        {Name: "tools.invoke", MinRevision: 1, MaxRevision: 2, Status: kit.StatusStable},
        {Name: "events.trace", MinRevision: 1, MaxRevision: 1, Optional: true},
    }}
    current := kit.RuntimeContract{ContractVersion: kit.ContractVersion, Capabilities: []kit.CapabilityOffer{
        {Name: "tools.invoke", Revision: 1, Status: kit.StatusStable},
    }}
    supported := kit.RuntimeContract{ContractVersion: kit.ContractVersion, Capabilities: []kit.CapabilityOffer{
        {Name: "tools.invoke", Revision: 2, Status: kit.StatusStable},
    }}
    incompatible := kit.RuntimeContract{ContractVersion: kit.ContractVersion, Capabilities: []kit.CapabilityOffer{
        {Name: "tools.invoke", Revision: 3, Status: kit.StatusStable},
    }}
    acceptedPlan := kit.PlanUpgrade(builder, current, supported)
    refusedPlan := kit.PlanUpgrade(builder, supported, incompatible)
    witness := struct {
        Schema string ` + "`json:\"schema\"`" + `
        PlanSchema string ` + "`json:\"plan_schema\"`" + `
        NMinusOne kit.CompatibilityReport ` + "`json:\"supported_n_minus_one\"`" + `
        NMinusOneAllowed bool ` + "`json:\"supported_allowed\"`" + `
        Refused kit.CompatibilityReport ` + "`json:\"incompatible_refusal\"`" + `
        RefusedAllowed bool ` + "`json:\"incompatible_allowed\"`" + `
        RefusedSteps []kit.UpgradeStep ` + "`json:\"refusal_steps\"`" + `
        Readout string ` + "`json:\"readout\"`" + `
    }{
        Schema: "fak.harnesskit-upgrade-witness/v1",
        PlanSchema: acceptedPlan.SchemaVersion,
        NMinusOne: acceptedPlan.Target,
        NMinusOneAllowed: acceptedPlan.Allowed,
        Refused: refusedPlan.Target,
        RefusedAllowed: refusedPlan.Allowed,
        RefusedSteps: refusedPlan.Steps,
        Readout: refusedPlan.Target.Error(),
    }
    if !acceptedPlan.Allowed || refusedPlan.Allowed { panic("upgrade verdict invariant failed") }
    out, err := json.MarshalIndent(witness, "", "  ")
    if err != nil { panic(err) }
    fmt.Println(string(out))
}`
	dir := writeExternalModule(t, root, program)
	got := strings.TrimSpace(runGo(t, dir, true, "run", "."))
	witnessPath := filepath.Join(root, "docs", "_witnesses", "issue-6805-harnesskit-upgrade.json")
	want, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatal(err)
	}
	if got != strings.TrimSpace(string(want)) {
		t.Fatalf("clean-module upgrade witness drifted\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
	var decoded struct {
		Schema           string `json:"schema"`
		NMinusOneAllowed bool   `json:"supported_allowed"`
		RefusedAllowed   bool   `json:"incompatible_allowed"`
		Readout          string `json:"readout"`
	}
	if err := json.Unmarshal(want, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Schema != "fak.harnesskit-upgrade-witness/v1" || !decoded.NMinusOneAllowed || decoded.RefusedAllowed || !strings.Contains(decoded.Readout, "revision_above_maximum") {
		t.Fatalf("witness lost its upgrade verdicts: %+v", decoded)
	}
}
