package gateway

import (
	"errors"
	"strings"
	"testing"
)

// synthetic catalog mirroring the shape of real MCP catalog names, so the precise
// matching rules can be exercised without coupling to the live registry.
func testCatalog() []catalogCapability {
	names := []string{
		"fak_index_lane",
		"fak_index_docs",
		"fak_revoke",
		"fak_tools_search",
		"fak_memory_run",
	}
	caps := make([]catalogCapability, 0, len(names))
	for _, n := range names {
		caps = append(caps, catalogCapability{name: n, tokens: significantTokens(n)})
	}
	return caps
}

func TestAdmitCoreTool_RefusesCatalogCoveredCapability(t *testing.T) {
	cases := []struct {
		name         string
		proposed     ProposedCoreTool
		wantSidestep string // a catalog tool the refusal must name
	}{
		{
			name:         "exact capability duplicate",
			proposed:     ProposedCoreTool{Name: "fak_revoke"},
			wantSidestep: "fak_revoke",
		},
		{
			name:         "proposal is a more general umbrella of a catalog tool",
			proposed:     ProposedCoreTool{Name: "fak_index"}, // {index} ⊊ {index,lane} and {index,docs}
			wantSidestep: "fak_index_lane",
		},
		{
			name:         "capability phrase supplies the matching token",
			proposed:     ProposedCoreTool{Name: "fak_unlease", Capability: "revoke a witness"},
			wantSidestep: "fak_revoke",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := admitCoreToolAgainst(tc.proposed, testCatalog())
			if err == nil {
				t.Fatalf("expected refusal for %+v, got admit", tc.proposed)
			}
			var adm *CoreToolAdmissionError
			if !errors.As(err, &adm) {
				t.Fatalf("expected *CoreToolAdmissionError, got %T: %v", err, err)
			}
			if adm.Reason != ReasonMCPCatalogCovers {
				t.Errorf("Reason = %q, want %q", adm.Reason, ReasonMCPCatalogCovers)
			}
			if !containsStr(adm.Sidesteps, tc.wantSidestep) {
				t.Errorf("Sidesteps = %v, want to contain %q", adm.Sidesteps, tc.wantSidestep)
			}
			// The refusal message must name the sidestep so a reviewer sees the escape hatch.
			if !strings.Contains(adm.Error(), tc.wantSidestep) {
				t.Errorf("Error() = %q, want it to name sidestep %q", adm.Error(), tc.wantSidestep)
			}
		})
	}
}

func TestAdmitCoreTool_AdmitsNovelCapability(t *testing.T) {
	cases := []struct {
		name     string
		proposed ProposedCoreTool
	}{
		{
			name:     "distinctive token no catalog tool carries",
			proposed: ProposedCoreTool{Name: "fak_translate", Capability: "translate a phrase between languages"},
		},
		{
			name: "proper-subset match on a generic verb only does not refuse",
			// {search} ⊊ {tools,search} but the only shared token is the generic verb
			// "search" — too weak to claim the catalog already covers a novel search.
			proposed: ProposedCoreTool{Name: "fak_lattice_search"},
		},
		{
			name:     "empty signature cannot be claimed covered",
			proposed: ProposedCoreTool{Name: "fak", Capability: "the a of"},
		},
		{
			name: "distinctive extra token keeps a partial overlap novel",
			// {index,graph} is NOT a subset of any catalog tool: "graph" is distinctive.
			proposed: ProposedCoreTool{Name: "fak_index_graph"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := admitCoreToolAgainst(tc.proposed, testCatalog()); err != nil {
				t.Fatalf("expected admit for %+v, got refusal: %v", tc.proposed, err)
			}
		})
	}
}

// TestAdmitCoreTool_WiresToLiveCatalog proves the exported entry point reads the real
// registered MCP tool set (toolDescriptors), not just a synthetic one: proposing a tool
// that duplicates a live MCP capability is refused, while a genuinely novel one admits.
func TestAdmitCoreTool_WiresToLiveCatalog(t *testing.T) {
	if err := AdmitCoreTool(ProposedCoreTool{Name: "fak_revoke"}); err == nil {
		t.Errorf("proposing fak_revoke (a live MCP tool) should be refused by the live catalog")
	} else {
		var adm *CoreToolAdmissionError
		if errors.As(err, &adm) && !containsStr(adm.Sidesteps, "fak_revoke") {
			t.Errorf("live-catalog refusal Sidesteps = %v, want to contain fak_revoke", adm.Sidesteps)
		}
	}
	if err := AdmitCoreTool(ProposedCoreTool{
		Name:       "fak_quantum_teleport",
		Capability: "teleport a qubit across the lattice",
	}); err != nil {
		t.Errorf("a genuinely novel capability should admit against the live catalog, got: %v", err)
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
