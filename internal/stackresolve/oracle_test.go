package stackresolve

import (
	"context"
	"fmt"
	"sort"
	"testing"
)

// TestResolverMatchesIndependentSmallGraphOracle exhaustively checks the
// resolver's allow/refuse answer against a deliberately separate subset oracle.
// The oracle knows only roots, provided capabilities, requirements, and conflicts;
// it shares no search or closure code with Resolve.
func TestResolverMatchesIndependentSmallGraphOracle(t *testing.T) {
	evidence := Evidence{Authority: "oracle-fixture", Source: "generated-small-graph"}
	for mask := 0; mask < 1<<6; mask++ {
		components := []Component{
			{ID: "root@1", Kind: "root", Version: "1", Relations: []Relation{{Kind: Requires, Target: "cap.x", Evidence: evidence}}, Evidence: evidence},
		}
		if mask&1 != 0 {
			components = append(components, Component{ID: "a@1", Kind: "provider", Version: "1", Provides: []string{"cap.x"}, Evidence: evidence})
		}
		if mask&2 != 0 {
			a := Component{ID: "b@1", Kind: "provider", Version: "1", Provides: []string{"cap.x"}, Evidence: evidence}
			if mask&8 != 0 {
				a.Relations = append(a.Relations, Relation{Kind: Requires, Target: "cap.y", Evidence: evidence})
			}
			components = append(components, a)
		}
		if mask&4 != 0 {
			components = append(components, Component{ID: "y@1", Kind: "provider", Version: "1", Provides: []string{"cap.y"}, Evidence: evidence})
		}
		if mask&16 != 0 {
			components[0].Relations = append(components[0].Relations, Relation{Kind: Conflicts, Target: "bad", Evidence: evidence})
		}
		if mask&32 != 0 {
			components = append(components, Component{ID: "bad@1", Kind: "root", Version: "1", Provides: []string{"bad"}, Evidence: evidence})
		}
		roots := []string{"root@1"}
		if mask&32 != 0 {
			roots = append(roots, "bad@1")
		}
		want := oracleSatisfiable(components, roots)
		got, err := Resolve(context.Background(), "oracle@1", roots, ManifestProvider{Manifest: Manifest{Components: components}})
		if err != nil {
			t.Fatalf("mask %06b: %v", mask, err)
		}
		if (got.Status == "allow") != want {
			t.Fatalf("mask %06b: resolver status=%s oracle allow=%v conflict=%+v", mask, got.Status, want, got.Conflict)
		}
	}
}

func TestResolverBacktracksPastBrokenLexicalProvider(t *testing.T) {
	evidence := Evidence{Authority: "fixture", Source: "backtrack"}
	components := []Component{
		{ID: "root@1", Kind: "root", Version: "1", Relations: []Relation{{Kind: Requires, Target: "model.coder", Evidence: evidence}}, Evidence: evidence},
		{ID: "model:a-broken@1", Kind: "model", Version: "1", Provides: []string{"model.coder"}, Relations: []Relation{{Kind: Requires, Target: "kernel.missing", Evidence: evidence}}, Evidence: evidence},
		{ID: "model:z-working@1", Kind: "model", Version: "1", Provides: []string{"model.coder"}, Evidence: evidence},
	}
	receipt, err := Resolve(context.Background(), "coding@1", []string{"root@1"}, ManifestProvider{Manifest: Manifest{Components: components}})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "allow" || !containsAll(selectedIDs(receipt), "model:z-working@1") {
		t.Fatalf("resolver failed to backtrack: %+v", receipt)
	}
	if containsAll(selectedIDs(receipt), "model:a-broken@1") {
		t.Fatalf("failed branch leaked into receipt: %+v", receipt.Selected)
	}
}

// oracleSatisfiable enumerates every component subset. A subset is valid when it
// contains all roots, every selected requirement has a selected provider, and no
// selected conflict has a selected provider. Unselected providers are irrelevant.
func oracleSatisfiable(components []Component, roots []string) bool {
	byID := map[string]int{}
	for i, component := range components {
		byID[component.ID] = i
	}
	for subset := 0; subset < 1<<len(components); subset++ {
		valid := true
		for _, root := range roots {
			index, ok := byID[root]
			if !ok || subset&(1<<index) == 0 {
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		provides := map[string]bool{}
		for i, component := range components {
			if subset&(1<<i) == 0 {
				continue
			}
			provides[component.ID] = true
			for _, capability := range component.Provides {
				provides[capability] = true
			}
		}
		for i, component := range components {
			if subset&(1<<i) == 0 {
				continue
			}
			for _, relation := range component.Relations {
				switch relation.Kind {
				case Requires:
					if !provides[relation.Target] && !oracleSubstituteProvided(relation.Substitutes, provides) {
						valid = false
					}
				case Conflicts:
					if provides[relation.Target] {
						valid = false
					}
				case Recommends, Optional:
					// Advisory relations do not constrain satisfiability.
				default:
					valid = false
				}
			}
		}
		if valid {
			return true
		}
	}
	return false
}

func oracleSubstituteProvided(substitutes []string, provides map[string]bool) bool {
	sorted := append([]string(nil), substitutes...)
	sort.Strings(sorted)
	for _, substitute := range sorted {
		if provides[substitute] {
			return true
		}
	}
	return false
}

func BenchmarkResolveBranching(b *testing.B) {
	evidence := Evidence{Authority: "benchmark", Source: "branching"}
	components := []Component{{ID: "root@1", Kind: "root", Version: "1", Relations: []Relation{{Kind: Requires, Target: "cap.0", Evidence: evidence}}, Evidence: evidence}}
	for level := 0; level < 8; level++ {
		next := fmt.Sprintf("cap.%d", level+1)
		for candidate := 0; candidate < 3; candidate++ {
			component := Component{ID: fmt.Sprintf("provider:%d:%d", level, candidate), Kind: "provider", Version: "1", Provides: []string{fmt.Sprintf("cap.%d", level)}, Evidence: evidence}
			if level < 7 {
				component.Relations = []Relation{{Kind: Requires, Target: next, Evidence: evidence}}
			}
			components = append(components, component)
		}
	}
	provider := ManifestProvider{Manifest: Manifest{Components: components}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Resolve(context.Background(), "benchmark@1", []string{"root@1"}, provider); err != nil {
			b.Fatal(err)
		}
	}
}
