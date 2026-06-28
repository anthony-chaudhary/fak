package contextq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// mockResolver is a test Resolver implementation.
type mockResolver struct {
	cards     []CapCard
	capMap    map[CapRef]Capability
}

func (m *mockResolver) Index() []CapCard {
	return m.cards
}

func (m *mockResolver) Fault(ref CapRef) Capability {
	return m.capMap[ref]
}

// newMockResolver creates a resolver with a few test capabilities.
func newMockResolver() *mockResolver {
	return &mockResolver{
		cards: []CapCard{
			{
				Name:          "skill-1",
				Kind:          CapKindSkill,
				Version:       "v1",
				Trigger:       "file operations read write",
				Tags:          []string{"file", "io"},
				EstimateBytes: 100,
			},
			{
				Name:          "skill-2",
				Kind:          CapKindSkill,
				Version:       "v1",
				Trigger:       "network request fetch download",
				Tags:          []string{"network", "http"},
				EstimateBytes: 150,
			},
			{
				Name:          "skill-3",
				Kind:          CapKindSkill,
				Version:       "v1",
				Trigger:       "code analysis refactor",
				Tags:          []string{"code", "analysis"},
				EstimateBytes: 200,
			},
		},
		capMap: map[CapRef]Capability{
			{
				Name:   "skill-1",
				Source: string(CapKindSkill),
			}: {
				Ref: CapRef{
					Name:   "skill-1",
					Source: string(CapKindSkill),
				},
				Digest: "abc123",
				Card: CapCard{
					Name:          "skill-1",
					Kind:          CapKindSkill,
					Version:       "v1",
					Trigger:       "file operations read write",
					Tags:          []string{"file", "io"},
					EstimateBytes: 100,
				},
				Resolve: func() []byte { return []byte("skill-1 body") },
				Scope:   abi.ScopeAgent,
			},
			{
				Name:   "skill-2",
				Source: string(CapKindSkill),
			}: {
				Ref: CapRef{
					Name:   "skill-2",
					Source: string(CapKindSkill),
				},
				Digest: "def456",
				Card: CapCard{
					Name:          "skill-2",
					Kind:          CapKindSkill,
					Version:       "v1",
					Trigger:       "network request fetch download",
					Tags:          []string{"network", "http"},
					EstimateBytes: 150,
				},
				Resolve: func() []byte { return []byte("skill-2 body") },
				Scope:   abi.ScopeAgent,
			},
			{
				Name:   "skill-3",
				Source: string(CapKindSkill),
			}: {
				Ref: CapRef{
					Name:   "skill-3",
					Source: string(CapKindSkill),
				},
				Digest: "ghi789",
				Card: CapCard{
					Name:          "skill-3",
					Kind:          CapKindSkill,
					Version:       "v1",
					Trigger:       "code analysis refactor",
					Tags:          []string{"code", "analysis"},
					EstimateBytes: 200,
				},
				Resolve: func() []byte { return []byte("skill-3 body") },
				Scope:   abi.ScopeAgent,
			},
		},
	}
}

// TestQueryCapabilities_RanksByIntent tests that results are ranked by intent relevance.
func TestQueryCapabilities_RanksByIntent(t *testing.T) {
	resolver := newMockResolver()
	ledger := NewCapabilityLedger()

	req := CapQueryRequest{
		Intent:      "file read write operations",
		BudgetBytes: 1000,
	}

	result := QueryCapabilities([]Resolver{resolver}, req, ledger)

	if len(result.Winners) != 3 {
		t.Fatalf("expected 3 winners, got %d", len(result.Winners))
	}

	if result.Winners[0].Card.Name != "skill-1" {
		t.Fatalf("expected skill-1 first (highest relevance), got %q", result.Winners[0].Card.Name)
	}

	if len(result.Omitted) != 0 {
		t.Fatalf("expected no omissions with generous budget, got %d", len(result.Omitted))
	}
}

// TestQueryCapabilities_RespectsBudget tests budget exhaustion behavior.
func TestQueryCapabilities_RespectsBudget(t *testing.T) {
	resolver := newMockResolver()
	ledger := NewCapabilityLedger()

	req := CapQueryRequest{
		Intent:      "file network code",
		BudgetBytes: 250, // fits skill-1 (100) + skill-2 (150) exactly
	}

	result := QueryCapabilities([]Resolver{resolver}, req, ledger)

	usedBytes := int64(0)
	for _, w := range result.Winners {
		usedBytes += int64(w.Card.EstimateBytes)
	}

	if usedBytes > 250 {
		t.Fatalf("budget violated: used %d > 250", usedBytes)
	}

	if result.BudgetHit != true {
		t.Fatalf("expected budget_hit=true, got %v", result.BudgetHit)
	}

	if len(result.Omitted) != 1 {
		t.Fatalf("expected 1 omission (skill-3), got %d", len(result.Omitted))
	}
}

// TestQueryCapabilities_RespectsK tests the max-results limit.
func TestQueryCapabilities_RespectsK(t *testing.T) {
	resolver := newMockResolver()
	ledger := NewCapabilityLedger()

	req := CapQueryRequest{
		Intent:      "file network code",
		BudgetBytes: 10000,
		K:           2,
	}

	result := QueryCapabilities([]Resolver{resolver}, req, ledger)

	if len(result.Winners) != 2 {
		t.Fatalf("expected K=2 winners, got %d", len(result.Winners))
	}

	if len(result.Omitted) != 1 {
		t.Fatalf("expected 1 omission, got %d", len(result.Omitted))
	}
}

// TestQueryCapabilities_RecordsFaults tests that the ledger records faults for winners.
func TestQueryCapabilities_RecordsFaults(t *testing.T) {
	resolver := newMockResolver()
	ledger := NewCapabilityLedger()

	req := CapQueryRequest{
		Intent:      "operations", // skill-1 has "operations" in trigger, others don't
		BudgetBytes: 1000,
	}

	result := QueryCapabilities([]Resolver{resolver}, req, ledger)

	// All 3 are winners because budget allows all, but only skill-1 has positive score
	if len(result.Winners) != 3 {
		t.Fatalf("expected 3 winners (budget allows all), got %d", len(result.Winners))
	}

	// skill-1 should be first due to positive score
	if result.Winners[0].Card.Name != "skill-1" {
		t.Fatalf("expected skill-1 first (positive score), got %q", result.Winners[0].Card.Name)
	}

	// All winners should be recorded in the ledger
	snap := ledger.Query()
	if len(snap.Spans) != 3 {
		t.Fatalf("expected 3 recorded faults, got %d", len(snap.Spans))
	}

	// skill-1 should have been recorded
	foundSkill1 := false
	for _, span := range snap.Spans {
		if span.CapRef.Name == "skill-1" {
			foundSkill1 = true
			if span.Faults != 1 {
				t.Fatalf("expected skill-1 fault count 1, got %d", span.Faults)
			}
		}
	}
	if !foundSkill1 {
		t.Fatalf("skill-1 not recorded in ledger")
	}
}

// TestQueryCapabilities_EmptyResolvers handles empty resolver list.
func TestQueryCapabilities_EmptyResolvers(t *testing.T) {
	ledger := NewCapabilityLedger()

	req := CapQueryRequest{
		Intent:      "file",
		BudgetBytes: 1000,
	}

	result := QueryCapabilities([]Resolver{}, req, ledger)

	if len(result.Winners) != 0 {
		t.Fatalf("expected no winners with empty resolvers, got %d", len(result.Winners))
	}
}

// TestRankByIntent_TestsRanking tests the ranking logic directly.
func TestRankByIntent_TestsRanking(t *testing.T) {
	cards := []CapCard{
		{
			Name:    "skill-a",
			Trigger: "network requests",
			Tags:    []string{"network", "http"},
		},
		{
			Name:    "skill-b",
			Trigger: "file operations",
			Tags:    []string{"file", "io"},
		},
		{
			Name:    "skill-c",
			Trigger: "file network operations",
			Tags:    []string{"mixed"},
		},
	}

	ranked := rankByIntent(cards, "file operations")

	// skill-b: trigger "file operations" = 2 overlap + tag "file" = 1 = 3
	// skill-c: trigger "file network operations" = 2 overlap + no tag matches = 2
	// skill-a: no trigger match = 0
	if ranked[0].Name != "skill-b" {
		t.Fatalf("expected skill-b (highest overlap 3) first, got %q", ranked[0].Name)
	}

	if ranked[1].Name != "skill-c" {
		t.Fatalf("expected skill-c second (overlap 2), got %q", ranked[1].Name)
	}

	if ranked[2].Name != "skill-a" {
		t.Fatalf("expected skill-a last (no overlap), got %q", ranked[2].Name)
	}
}