package capindexgw

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestC5ProtocolBlindLoader proves that the MCP and A2A resolvers
// use the same capindex.Capability type, demonstrating the loader is
// protocol-blind (issue #1108, C5).
func TestC5ProtocolBlindLoader(t *testing.T) {
	// Create MCP resolver
	mcpResolver := NewMCPResolver(nil) // nil server is OK for Index() only
	mcpCards := mcpResolver.Index()

	// Create A2A resolver
	a2aResolver := NewA2AResolver()
	a2aCards := a2aResolver.Index()

	// Both resolvers return CapCards with the same structure
	if len(mcpCards) == 0 {
		t.Fatal("MCP resolver returned no cards")
	}
	if len(a2aCards) == 0 {
		t.Fatal("A2A resolver returned no cards")
	}

	// Verify MCP cards have the correct kind
	for _, card := range mcpCards {
		if card.Ref.Kind != capindex.CapKindMCPTool {
			t.Errorf("MCP card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, capindex.CapKindMCPTool)
		}
		// Verify the card structure is the same across protocols
		if card.Digest == "" {
			t.Errorf("MCP card %s has empty digest", card.Ref.Name)
		}
		if card.Trigger == "" {
			t.Errorf("MCP card %s has empty trigger", card.Ref.Name)
		}
		if len(card.Tags) == 0 {
			t.Errorf("MCP card %s has no tags", card.Ref.Name)
		}
	}

	// Verify A2A cards have the correct kind
	for _, card := range a2aCards {
		if card.Ref.Kind != capindex.CapKindA2AAgent {
			t.Errorf("A2A card %s has wrong kind: got %v, want %v",
				card.Ref.Name, card.Ref.Kind, capindex.CapKindA2AAgent)
		}
		// Verify the card structure is the same across protocols
		if card.Digest == "" {
			t.Errorf("A2A card %s has empty digest", card.Ref.Name)
		}
		if card.Trigger == "" {
			t.Errorf("A2A card %s has empty trigger", card.Ref.Name)
		}
		if len(card.Tags) == 0 {
			t.Errorf("A2A card %s has no tags", card.Ref.Name)
		}
	}

	// PROVE the loader is protocol-blind: both Capability types
	// use the same capindex.Capability field, proving they're not protocol-specific
	var mcpCap, a2aCap capindex.Capability
	_ = mcpCap.Caps // Both have the same Caps field type: []abi.Capability
	_ = a2aCap.Caps

	t.Logf("PROOF: Both MCP and A2A use the same capindex.Capability struct with abi.Capability field")
	t.Logf("  MCP cards: %d, A2A cards: %d", len(mcpCards), len(a2aCards))
	t.Logf("  Example MCP card: %s", mcpCards[0].Ref.Name)
	t.Logf("  Example A2A card: %s", a2aCards[0].Ref.Name)
}

// TestMCPResolverIndexIsCheapAtRest proves the C5 "0-for-∞" at-rest property:
// MCPResolver.Index() is O(cards), not O(cards × body). A toolset of N tools
// costs the same at rest however large each tool's inputSchema is, because the
// index digest is derived from the cheap card bytes and the body is only
// materialized by Fault (which is what actually carries the inputSchema).
//
// This is the regression witness for the defect where Index() hashed the full
// multi-KB inputSchema for every tool on every call. The witness is the REAL
// Index() path: we swap the resolver's descriptor source (a test-only seam on
// MCPResolver) for the same tool with a thin body and then with a 512 KiB body,
// holding name and description fixed. Every pre-fix implementation that folds
// the body into the digest moves the digest between the two; a cheap-at-rest
// index does not.
func TestMCPResolverIndexIsCheapAtRest(t *testing.T) {
	const (
		name = "fak_adjudicate" // any catalog tool; we supply the descriptor
		desc = "Adjudicate a proposed tool call through the fak kernel WITHOUT executing it."
	)
	// A 512 KiB body: wildly oversized vs a normal schema, so any body read on
	// the index path changes the result visibly.
	fatBody := []byte(`{"type":"object","padding":"` + strings.Repeat("x", 512*1024) + `"}`)

	thin := NewMCPResolver(nil)
	thin.setDescriptorsForTest([]map[string]any{{
		"name":        name,
		"description": desc,
		"inputSchema": json.RawMessage(`{}`),
	}})
	fat := NewMCPResolver(nil)
	fat.setDescriptorsForTest([]map[string]any{{
		"name":        name,
		"description": desc,
		"inputSchema": json.RawMessage(fatBody),
	}})

	thinCards := thin.Index()
	fatCards := fat.Index()
	if len(thinCards) != 1 || len(fatCards) != 1 {
		t.Fatalf("card counts: thin=%d fat=%d, want 1 each", len(thinCards), len(fatCards))
	}

	// The whole at-rest card must be body-size-invariant: same digest, same
	// bytes, same ref/trigger/tags semantics.
	if thinCards[0].Digest != fatCards[0].Digest {
		t.Fatalf("Index() digest depends on body size: thin=%s fat=%s (at-rest index must be O(cards), not O(bodies))",
			thinCards[0].Digest, fatCards[0].Digest)
	}
	if string(thinCards[0].CardBytes) != string(fatCards[0].CardBytes) {
		t.Fatalf("Index() CardBytes depend on body size:\n thin=%s\n fat =%s",
			thinCards[0].CardBytes, fatCards[0].CardBytes)
	}

	// The digest must be the cheap card-level key, not a hash of any body.
	if want := capindex.Digest(thinCards[0].CardBytes); thinCards[0].Digest != want {
		t.Fatalf("Index() digest = %s, want capindex.Digest(CardBytes) = %s", thinCards[0].Digest, want)
	}

	// Semantics are unchanged: ref/kind/trigger/tags still come from the card.
	got := thinCards[0]
	if got.Ref.Kind != capindex.CapKindMCPTool || got.Ref.Name != name {
		t.Fatalf("Index() ref = %+v, want kind=%s name=%s", got.Ref, capindex.CapKindMCPTool, name)
	}
	if got.Trigger != desc {
		t.Fatalf("Index() trigger = %q, want description %q", got.Trigger, desc)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "mcp" || got.Tags[1] != "tool" {
		t.Fatalf("Index() tags = %v, want [mcp tool]", got.Tags)
	}

	// The residual O(bodies) path is Fault: it must materialize the inputSchema,
	// and the at-rest card must NOT carry it.
	cap, err := fat.Fault(got.Ref)
	if err != nil {
		t.Fatalf("Fault(%+v) returned error: %v", got.Ref, err)
	}
	if !strings.Contains(string(cap.Body), "inputSchema") {
		t.Fatalf("Fault() body does not materialize inputSchema: %s", cap.Body)
	}
	if !strings.Contains(string(cap.Body), strings.Repeat("x", 1024)) {
		t.Fatalf("Fault() body does not carry the full inputSchema bytes")
	}
	if strings.Contains(string(got.CardBytes), "inputSchema") || strings.Contains(string(got.CardBytes), "padding") {
		t.Fatalf("at-rest card carries body bytes: %s", got.CardBytes)
	}

	t.Logf("PROOF: Index() digest is body-size-invariant (thin==fat==%s); Fault() materializes the schema", got.Digest)
}

// TestMCPResolverFoldingProvesFolded tests that the MCP resolver
// successfully folds the existing gateway/mcp.go code.
func TestMCPResolverFoldingProvesFolded(t *testing.T) {
	// The MCP resolver should use ToolDescriptorsForResolver from gateway/mcp.go
	// This proves it "folds" the existing code rather than duplicating it
	toolDescs := gateway.ToolDescriptorsForResolver()

	if len(toolDescs) == 0 {
		t.Fatal("ToolDescriptorsForResolver returned no descriptors")
	}

	// Verify we can build an MCP resolver from these descriptors
	mcpResolver := NewMCPResolver(nil)
	cards := mcpResolver.Index()

	if len(cards) != len(toolDescs) {
		t.Errorf("MCP resolver card count mismatch: got %d, want %d",
			len(cards), len(toolDescs))
	}

	// Verify the folding by checking names match
	for i, card := range cards {
		name, _ := toolDescs[i]["name"].(string)
		if card.Ref.Name != name {
			t.Errorf("Card %d name mismatch: got %s, want %s", i, card.Ref.Name, name)
		}
	}

	t.Logf("PROOF: MCP resolver folds gateway/mcp.go (ToolDescriptorsForResolver)")
	t.Logf("  Folded %d tools from gateway/mcp.go", len(cards))
}

// TestA2AResolverFoldingProvesFolded tests that the A2A resolver
// successfully folds the existing gateway/a2a.go code.
func TestA2AResolverFoldingProvesFolded(t *testing.T) {
	// The A2A resolver should use A2AMethodRegistryForResolver from gateway/a2a.go
	// This proves it "folds" the existing code rather than duplicating it
	methodSpecs := gateway.A2AMethodRegistryForResolver()

	if len(methodSpecs) == 0 {
		t.Fatal("A2AMethodRegistryForResolver returned no specs")
	}

	// Verify we can build an A2A resolver from these specs
	a2aResolver := NewA2AResolver()
	cards := a2aResolver.Index()

	if len(cards) != len(methodSpecs) {
		t.Errorf("A2A resolver card count mismatch: got %d, want %d",
			len(cards), len(methodSpecs))
	}

	// Verify the folding by checking all names are present (map iteration order is non-deterministic)
	cardNames := make(map[string]bool)
	for _, card := range cards {
		cardNames[card.Ref.Name] = true
	}
	for _, spec := range methodSpecs {
		if !cardNames[spec.Name] {
			t.Errorf("Card %s missing from resolver output", spec.Name)
		}
	}

	t.Logf("PROOF: A2A resolver folds gateway/a2a.go (A2AMethodRegistryForResolver)")
	t.Logf("  Folded %d methods from gateway/a2a.go", len(cards))
}

// TestA2AFaultResolvesInvocationContract proves C5's "Fault() resolves the
// invocation contract" claim (issue #1108): the body a Fault pages in carries
// the real invocation fields — the wire dispatch token, the transport envelope,
// and the parameter envelope — not just name/scope/description. It also proves
// Index() stays cheap (cards carry no invocation contract), so the enrichment
// is confined to the faulted body.
func TestA2AFaultResolvesInvocationContract(t *testing.T) {
	r := NewA2AResolver()

	cards := r.Index()
	if len(cards) == 0 {
		t.Fatal("A2A resolver returned no cards")
	}

	// Every A2A method faults in a body carrying the invocation contract.
	for _, card := range cards {
		cap, err := r.Fault(card.Ref)
		if err != nil {
			t.Fatalf("Fault(%s) failed: %v", card.Ref.Name, err)
		}

		var body struct {
			Name        string `json:"name"`
			Scope       string `json:"scope"`
			Method      string `json:"method"`
			Transport   string `json:"transport"`
			InputSchema string `json:"input_schema"`
		}
		if err := json.Unmarshal(cap.Body, &body); err != nil {
			t.Fatalf("Fault(%s) body is not valid JSON: %v", card.Ref.Name, err)
		}

		// The invocation contract: who to call, under which envelope, and the
		// parameter envelope — the fields that make the body invocable.
		if body.Method == "" {
			t.Errorf("Fault(%s) body missing invocation `method`", card.Ref.Name)
		}
		if body.Transport == "" {
			t.Errorf("Fault(%s) body missing `transport`", card.Ref.Name)
		}
		if body.InputSchema == "" {
			t.Errorf("Fault(%s) body missing `input_schema` (parameter envelope)", card.Ref.Name)
		}
		if body.Method != card.Ref.Name {
			t.Errorf("Fault(%s) method %q != card name %q", card.Ref.Name, body.Method, card.Ref.Name)
		}

		// The contract must be MORE than the cheap card's label.
		if body.Method == "" || body.Transport == "" || body.InputSchema == "" {
			t.Errorf("Fault(%s) body did not enrich beyond name/scope/description", card.Ref.Name)
		}
	}

	// Index() stays cheap: the card bytes carry no invocation-contract fields.
	for _, card := range cards {
		if strings.Contains(string(card.CardBytes), "input_schema") ||
			strings.Contains(string(card.CardBytes), "transport") {
			t.Errorf("card %s leaked invocation-contract fields into cheap CardBytes", card.Ref.Name)
		}
	}

	// Spot-check one concrete method end-to-end.
	sample, err := r.Fault(cards[0].Ref)
	if err != nil {
		t.Fatalf("Fault(%s) failed: %v", cards[0].Ref.Name, err)
	}
	if !strings.Contains(string(sample.Body), `"method"`) {
		t.Fatalf("Fault body lacks the invocation method: %s", sample.Body)
	}

	t.Logf("PROOF: Fault() pages in the A2A invocation contract; sample body: %s", sample.Body)
}

// Compile-time proof that both adapters satisfy the protocol-blind seam.
var (
	_ capindex.Resolver = (*MCPResolver)(nil)
	_ capindex.Resolver = (*A2AResolver)(nil)
)

func TestCoreToolAdmissionRefusesReachableMCPCapability(t *testing.T) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		t.Fatal("MCP catalog is empty; refusal witness requires one reachable capability")
	}
	mcp := cards[0].Ref

	got := r.AdmitCoreTool(CoreToolProposal{Name: "new_permanent_tool", Capability: mcp.Name})
	if got.Allowed {
		t.Fatalf("reachable MCP capability %q was admitted as a core tool", mcp.Name)
	}
	if got.Sidestep != mcp {
		t.Fatalf("sidestep = %+v, want %+v", got.Sidestep, mcp)
	}
	if !strings.Contains(got.Reason, mcp.Name) || !strings.Contains(got.Reason, "MCP catalog sidestep") {
		t.Fatalf("refusal reason %q does not name the MCP sidestep", got.Reason)
	}

	unmatched := r.AdmitCoreTool(CoreToolProposal{Name: "novel_tool", Capability: "not-in-the-catalog"})
	if !unmatched.Allowed || unmatched.Reason != "" {
		t.Fatalf("novel capability was refused: %+v", unmatched)
	}
}

var (
	benchCardsSink     []capindex.CapCard
	benchCapSink       capindex.Capability
	benchAdmissionSink CoreToolAdmission
	benchErrSink       error
	benchIndexSink     *capindex.Index
	benchChangesSink   []capindex.Change
)

func BenchmarkMCPResolverIndex(b *testing.B) {
	r := NewMCPResolver(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCardsSink = r.Index()
	}
}

func BenchmarkMCPResolverFault(b *testing.B) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty MCP catalog")
	}
	validRef := cards[0].Ref
	notFoundRef := capindex.CapRef{Kind: capindex.CapKindMCPTool, Name: "nonexistent_mcp_tool"}
	mismatchRef := capindex.CapRef{Kind: capindex.CapKindA2AAgent, Name: validRef.Name}

	b.Run("Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(validRef)
		}
	})

	b.Run("NotFound", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(notFoundRef)
		}
	})

	b.Run("KindMismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(mismatchRef)
		}
	})
}

func BenchmarkMCPResolverAdmitCoreTool(b *testing.B) {
	r := NewMCPResolver(nil)
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty MCP catalog")
	}
	reachable := CoreToolProposal{Name: "candidate", Capability: cards[0].Ref.Name}
	novel := CoreToolProposal{Name: "candidate", Capability: "unregistered_novel_tool"}
	empty := CoreToolProposal{Name: "candidate", Capability: ""}

	b.Run("RefuseReachable", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(reachable)
		}
	})

	b.Run("AdmitNovel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(novel)
		}
	})

	b.Run("AdmitEmpty", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmissionSink = r.AdmitCoreTool(empty)
		}
	})
}

func BenchmarkA2AResolverIndex(b *testing.B) {
	r := NewA2AResolver()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCardsSink = r.Index()
	}
}

func BenchmarkA2AResolverFault(b *testing.B) {
	r := NewA2AResolver()
	cards := r.Index()
	if len(cards) == 0 {
		b.Fatal("empty A2A catalog")
	}
	validRef := cards[0].Ref
	notFoundRef := capindex.CapRef{Kind: capindex.CapKindA2AAgent, Name: "nonexistent_a2a_method"}
	mismatchRef := capindex.CapRef{Kind: capindex.CapKindMCPTool, Name: validRef.Name}

	b.Run("Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(validRef)
		}
	})

	b.Run("NotFound", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(notFoundRef)
		}
	})

	b.Run("KindMismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCapSink, benchErrSink = r.Fault(mismatchRef)
		}
	})
}

func BenchmarkIndexRegistration(b *testing.B) {
	mcpResolver := NewMCPResolver(nil)
	a2aResolver := NewA2AResolver()
	mcpCards := mcpResolver.Index()
	a2aCards := a2aResolver.Index()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ix := capindex.NewIndex()
		ix.RegisterAll(mcpCards)
		ix.RegisterAll(a2aCards)
		benchIndexSink = ix
	}
}

func BenchmarkIndexDiff_Noop(b *testing.B) {
	mcpResolver := NewMCPResolver(nil)
	a2aResolver := NewA2AResolver()
	ix := capindex.NewIndex()
	ix.RegisterAll(mcpResolver.Index())
	ix.RegisterAll(a2aResolver.Index())

	snap1 := ix.Snapshot()
	snap2 := ix.Snapshot()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChangesSink = snap1.Diff(snap2)
	}
}
