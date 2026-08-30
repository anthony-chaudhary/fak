package selfquery

import (
	"strings"
	"testing"
)

func TestCapabilitiesEmptyQueryListsStableToolbelt(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{})
	if err != nil {
		t.Fatal(err)
	}
	names := namesOf(resp.Cards)
	for _, want := range []string{
		"memory-driver:recall", "memory-driver:clean", "memory-driver:compact",
		"fak-dev index lane", "fak-dev index docs", "fak-dev index claims", "fak-dev index verbs", "fak-dev index work",
		"fak_changes", "dos_arbitrate",
	} {
		if !names[want] {
			t.Fatalf("capabilities empty query missing %s; got %v", want, sortedNames(resp.Cards))
		}
	}
	// Narrower than fak feature query: capabilities is the memory/index/kernel
	// toolbelt only, not the ask-policy or context-plan surfaces.
	for _, unwanted := range []string{"ask-policy:should-ask", "context-plan:assumptions"} {
		if names[unwanted] {
			t.Fatalf("capabilities query should stay narrower than fak feature query; unexpectedly found %s", unwanted)
		}
	}
}

func TestCapabilitiesRepoWorkIntentReturnsDevIndexCommand(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{Query: "pick ready issue backlog"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Cards) == 0 || resp.Cards[0].Name != "fak-dev index work" {
		t.Fatalf("repo-work intent top card = %v, want fak-dev index work", sortedNames(resp.Cards))
	}
	req := resp.Cards[0].Request
	if got := strings.Join(req.Command, " "); got != "fak-dev index work <query>" || req.Executed {
		t.Fatalf("repo-work request = %+v, want unexecuted fak-dev index work <query>", req)
	}
}

func TestCapabilitiesCompactIntentRanksHygieneFamilyTogether(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{Query: "compact my context"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Cards) == 0 {
		t.Fatal("compact intent returned no cards")
	}
	if top := resp.Cards[0].Name; top != "memory-driver:compact" {
		t.Fatalf("top ranked card = %s, want memory-driver:compact", top)
	}
	names := namesOf(resp.Cards)
	if !names["memory-driver:clean"] {
		t.Fatalf("compact intent should also surface memory-driver:clean via the hygiene-family synonym tags; got %v", sortedNames(resp.Cards))
	}
}

func TestCapabilitiesMemoryCardCarriesReadyMemoryRunCall(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{Query: "compact my context"})
	if err != nil {
		t.Fatal(err)
	}
	var found *FeatureCard
	for i := range resp.Cards {
		if resp.Cards[i].Name == "memory-driver:compact" {
			found = &resp.Cards[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("missing memory-driver:compact card: %v", sortedNames(resp.Cards))
	}
	req := found.Request
	if req.MCPTool != "fak_memory_run" || req.Executed {
		t.Fatalf("memory-driver:compact request = %+v, want a ready, unexecuted fak_memory_run call", req)
	}
	if apply, ok := req.Arguments["apply"].(bool); !ok || apply {
		t.Fatalf("memory-driver:compact request arguments = %+v, want apply=false", req.Arguments)
	}
	if req.Arguments["driver"] != "compact" {
		t.Fatalf("memory-driver:compact request arguments = %+v, want driver=compact", req.Arguments)
	}
}

func TestCapabilitiesLimitCapsResults(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Cards) != 2 {
		t.Fatalf("capabilities limit=2 returned %d cards, want 2", len(resp.Cards))
	}
}

func TestCapabilitiesNegativeLimitFailsClosed(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cat.Capabilities(CapabilitiesRequest{Limit: -1}); err == nil {
		t.Fatal("negative limit should fail closed")
	}
}

func TestCapabilitiesKernelVerbCardsAreReadOnly(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{DevLoader: testDevLoader, Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Capabilities(CapabilitiesRequest{Query: "arbitrate lane concurrency"})
	if err != nil {
		t.Fatal(err)
	}
	var found *FeatureCard
	for i := range resp.Cards {
		if resp.Cards[i].Name == "dos_arbitrate" {
			found = &resp.Cards[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("dos_arbitrate intent missing dos_arbitrate card: %v", sortedNames(resp.Cards))
	}
	if found.Effect != EffectRead || found.Request.Executed {
		t.Fatalf("dos_arbitrate card = %+v, want read-only unexecuted request", found)
	}
}

// TestCapabilitiesDiscoversRuntimeEfficiencyOutcomes prevents the repository's
// machine-readable capability answer from regressing to dev tooling only. The
// performance-first product focus must be discoverable in operator language,
// including the high-value turn-control surface.
func TestCapabilitiesDiscoversRuntimeEfficiencyOutcomes(t *testing.T) {
	catalog := &Catalog{}
	tests := []struct {
		query   string
		wantIDs []string
	}{
		{"token savings", []string{"docs/CAPABILITIES.md#context-reuse", "docs/CAPABILITIES.md#model-routing", "docs/CAPABILITIES.md#savings-observability"}},
		{"turn control", []string{"docs/CAPABILITIES.md#turn-savings", "docs/CAPABILITIES.md#session-control"}},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			got, err := catalog.Capabilities(CapabilitiesRequest{Query: tc.query, Limit: 8})
			if err != nil {
				t.Fatalf("Capabilities(%q): %v", tc.query, err)
			}
			seen := map[string]bool{}
			for _, card := range got.Cards {
				seen[card.DetailRef] = true
			}
			for _, want := range tc.wantIDs {
				if !seen[want] {
					t.Errorf("Capabilities(%q) missing %q; got IDs %v", tc.query, want, cardIDs(got.Cards))
				}
			}
			if len(got.Cards) == 0 || got.Cards[0].Kind != "runtime-capability" {
				t.Errorf("Capabilities(%q) first result = %#v, want runtime capability", tc.query, got.Cards)
			}
		})
	}
}

func TestCapabilitiesDiscoversNativePerformanceStagesFromSharedCatalog(t *testing.T) {
	catalog := &Catalog{}
	tests := []struct {
		query   string
		detail  string
		command string
	}{
		{"serve native model", "docs/model-engine-env.md", "fak serve --gguf <model.gguf> --metal"},
		{"benchmark native inference", "docs/model-engine-env.md", "fak benchmarks describe modelbench"},
		{"evaluate model quality", "docs/quality/output-quality-regression-runbook.md", "fak quality run --json"},
		{"profile native bottleneck", "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md", "fak native-performance --profile-next profile.json"},
		{"performance receipt", "docs/benchmarks/NATIVE-PERFORMANCE-REGRESSION-GATE.md", "fak native-performance --gate gate-request.json"},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			got, err := catalog.Capabilities(CapabilitiesRequest{Query: tc.query, Limit: 1})
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Cards) != 1 || got.Cards[0].DetailRef != tc.detail {
				t.Fatalf("Capabilities(%q) = %#v, want detail %q first", tc.query, got.Cards, tc.detail)
			}
			if command := strings.Join(got.Cards[0].Request.Command, " "); command != tc.command {
				t.Fatalf("Capabilities(%q) command = %q, want %q", tc.query, command, tc.command)
			}
		})
	}
}

func cardIDs(cards []FeatureCard) []string {
	ids := make([]string, 0, len(cards))
	for _, card := range cards {
		ids = append(ids, card.DetailRef)
	}
	return ids
}
