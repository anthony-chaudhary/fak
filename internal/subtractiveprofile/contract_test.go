package subtractiveprofile

import (
	"strings"
	"testing"
)

func cap(id string, requires ...string) Capability {
	return Capability{ID: id, Aliases: []string{id + "-alias"}, Requires: requires, Help: true, Schema: true, Runtime: true, Artifact: true}
}
func TestResolutionIsDeterministicAndRemovalIsSticky(t *testing.T) {
	profiles := []Profile{{Include: []Capability{cap("tools"), cap("agent", "tools")}}, {Remove: map[string]Removal{"tools-alias": RemovalStatic}}, {Include: []Capability{cap("tools")}, Replace: map[string]Capability{"tools-alias": cap("tools")}}}
	_, err := Resolve(profiles, Report{})
	if err == nil || !strings.Contains(err.Error(), "agent requires tools, removed (static)") {
		t.Fatalf("error=%v", err)
	}
}
func TestMissingDependencyChainIsActionable(t *testing.T) {
	_, err := Resolve([]Profile{{Include: []Capability{cap("agent", "memory")}}}, Report{})
	if err == nil || err.Error() != "agent requires memory, which is missing" {
		t.Fatalf("error=%v", err)
	}
}
func TestRemovedCapabilityDisappearsFromEverySurface(t *testing.T) {
	report := Report{Minimal: Delta{BinaryBytes: 10, StartupMillis: 1}, Full: Delta{BinaryBytes: 100, StartupMillis: 5, IdleMemoryBytes: 20, ContextTokens: 30, SchemaBytes: 40}}
	got, err := Resolve([]Profile{{Include: []Capability{cap("ui"), cap("core")}}, {Remove: map[string]Removal{"ui": RemovalStatic}}}, report)
	if err != nil {
		t.Fatal(err)
	}
	for _, surface := range []string{"help", "schema", "runtime", "artifact"} {
		for _, id := range got.Surface(surface) {
			if id == "ui" {
				t.Fatalf("ui visible on %s", surface)
			}
		}
	}
	if err := got.ProbeAbsent("ui"); err != nil {
		t.Fatal(err)
	}
	if got.Report.Full.SchemaBytes != 40 || got.Report.Minimal.BinaryBytes != 10 {
		t.Fatalf("report=%+v", got.Report)
	}
}
func TestConfigureAndProvenance(t *testing.T) {
	got, err := Resolve([]Profile{{Include: []Capability{cap("core")}, Configure: map[string]map[string]string{"core": {"mode": "headless"}}}}, Report{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config["core"]["mode"] != "headless" || len(got.Provenance) != 2 {
		t.Fatalf("got=%+v", got)
	}
}
