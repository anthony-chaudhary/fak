package selfquery

import "testing"

func TestCapabilitiesEmptyQueryListsStableToolbelt(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
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
		"fak index lane", "fak index docs", "fak index claims", "fak index verbs",
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

func TestCapabilitiesCompactIntentRanksHygieneFamilyTogether(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
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
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
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
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
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
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cat.Capabilities(CapabilitiesRequest{Limit: -1}); err == nil {
		t.Fatal("negative limit should fail closed")
	}
}

func TestCapabilitiesKernelVerbCardsAreReadOnly(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
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
