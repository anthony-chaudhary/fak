package disambiguation

import (
	"errors"
	"fmt"
)

const RuntimeSourceSelfTestSchemaVersion = "fak-disambiguation-runtime-source-self-test/1"

// RuntimeSourceResolution is one scoped owner of the intentionally overloaded
// canonical token "runtime".
type RuntimeSourceResolution struct {
	Scope         Scope  `json:"scope"`
	Alias         string `json:"alias"`
	CanonicalTerm string `json:"canonical_term"`
	OwnerLeaf     string `json:"owner_leaf"`
	SourcePath    string `json:"source_path"`
}

// RuntimeSourceSelfTestReport proves unscoped ambiguity and exact scoped lookup.
type RuntimeSourceSelfTestReport struct {
	Schema            string                    `json:"schema"`
	IndexVersion      string                    `json:"index_version"`
	UnscopedAmbiguous bool                      `json:"unscoped_ambiguous"`
	Choices           []RuntimeSourceResolution `json:"choices"`
}

// RunRuntimeSourceSelfTest asks the ambiguous question first, then resolves all
// five public runtime surfaces with the required scope qualifier.
func RunRuntimeSourceSelfTest() (RuntimeSourceSelfTestReport, error) {
	if _, err := Query("runtime"); !errors.Is(err, ErrScopeRequired) {
		return RuntimeSourceSelfTestReport{}, fmt.Errorf("unscoped runtime error=%v, want ErrScopeRequired", err)
	}
	fixtures := []struct{ value, alias string }{
		{"agent-application", "agent application runtime"},
		{"gateway-serving", "gateway serving runtime"},
		{"guard-enforcement", "guard enforcement runtime"},
		{"model-serving", "model serving runtime"},
		{"worker-execution", "worker execution runtime"},
	}
	report := RuntimeSourceSelfTestReport{Schema: RuntimeSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion, UnscopedAmbiguous: true, Choices: make([]RuntimeSourceResolution, 0, len(fixtures))}
	for _, fixture := range fixtures {
		scope := Scope{Kind: "runtime", Value: fixture.value}
		canonical, err := QueryScoped("runtime", scope)
		if err != nil {
			return RuntimeSourceSelfTestReport{}, fmt.Errorf("query scope %s: %w", fixture.value, err)
		}
		alias, err := ResolveScoped(fixture.alias, scope)
		if err != nil {
			return RuntimeSourceSelfTestReport{}, fmt.Errorf("resolve alias %s: %w", fixture.alias, err)
		}
		if alias.Entry.Identity.CanonicalTerm != "runtime" || alias.MatchedAlias != fixture.alias {
			return RuntimeSourceSelfTestReport{}, fmt.Errorf("alias %q resolved incorrectly", fixture.alias)
		}
		entry := canonical.Entry
		if len(entry.Sources) == 0 {
			return RuntimeSourceSelfTestReport{}, fmt.Errorf("scope %s has no public source", fixture.value)
		}
		report.Choices = append(report.Choices, RuntimeSourceResolution{Scope: scope, Alias: fixture.alias, CanonicalTerm: entry.Identity.CanonicalTerm, OwnerLeaf: entry.Owner.Leaf, SourcePath: entry.Sources[0].Locator})
	}
	return report, nil
}
