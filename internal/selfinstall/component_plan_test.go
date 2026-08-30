package selfinstall

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestComponentPlanMixedStateActivatesOnlyStale(t *testing.T) {
	dir := t.TempDir()
	currentSource := writeTransactionFile(t, dir, "current-source", "current")
	currentTarget := writeTransactionFile(t, dir, "current-target", "current")
	staleSource := writeTransactionFile(t, dir, "stale-source", "new")
	staleTarget := writeTransactionFile(t, dir, "stale-target", "old")
	components := []Component{
		{Name: "fak", Source: currentSource, Target: currentTarget, CompatibilityGroup: "launcher", Acquisition: ComponentReuse},
		{Name: "fak-dev", Source: staleSource, Target: staleTarget, CompatibilityGroup: "launcher", Acquisition: ComponentTransferOrBuild},
	}
	plans, err := PlanComponents(components)
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != 2 {
		t.Fatalf("plans = %#v", plans)
	}
	var current, stale ComponentPlan
	for _, plan := range plans {
		switch plan.Name {
		case "fak":
			current = plan
		case "fak-dev":
			stale = plan
		}
	}
	if current.Activation != ComponentNoop || current.DesiredArtifactDigest == "" || current.InstalledArtifactDigest == "" {
		t.Fatalf("current plan = %#v", current)
	}
	if stale.Activation != ComponentActivate || stale.DesiredArtifactDigest == stale.InstalledArtifactDigest {
		t.Fatalf("stale plan = %#v", stale)
	}

	copies := CopiesForActivation(components, plans)
	if len(copies) != 1 || copies[0].Target != staleTarget {
		t.Fatalf("copies = %#v, want stale target only", copies)
	}
	activated := 0
	result := RunLaunchTransaction(copies, staleTarget, func(source, target string) error {
		activated++
		return OSSwap(source, target)
	})
	if got, ok := result.(Updated); !ok || got.Changed != 1 || activated != 1 {
		t.Fatalf("result=%#v activations=%d", result, activated)
	}
	assertTransactionContents(t, currentTarget, "current")
	assertTransactionContents(t, staleTarget, "new")
}

func TestComponentPlanCoupledFailureRestoresEveryMovedComponent(t *testing.T) {
	dir := t.TempDir()
	aSource := writeTransactionFile(t, dir, "a-source", "new-a")
	aTarget := writeTransactionFile(t, dir, "a-target", "old-a")
	bSource := writeTransactionFile(t, dir, "b-source", "new-b")
	bTarget := writeTransactionFile(t, dir, "b-target", "old-b")
	components := []Component{
		{Name: "fak", Source: aSource, Target: aTarget, CompatibilityGroup: "launcher"},
		{Name: "fak-dev", Source: bSource, Target: bTarget, CompatibilityGroup: "launcher"},
	}
	plans, err := PlanComponents(components)
	if err != nil {
		t.Fatal(err)
	}
	copies := CopiesForActivation(components, plans)
	attempt := 0
	result := RunLaunchTransaction(copies, aTarget, func(source, target string) error {
		attempt++
		if attempt == 2 {
			return errors.New("injected activation failure")
		}
		return OSSwap(source, target)
	})
	if got, ok := result.(RolledBack); !ok || got.Changed != 1 || got.Err == nil {
		t.Fatalf("result = %#v, want rollback after one move", result)
	}
	for path, want := range map[string]string{aTarget: "old-a", bTarget: "old-b"} {
		body, err := os.ReadFile(filepath.Clean(path))
		if err != nil || string(body) != want {
			t.Fatalf("%s = %q err=%v, want %q", path, body, err, want)
		}
	}
}
