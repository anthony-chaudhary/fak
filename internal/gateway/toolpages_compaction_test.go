package gateway

// toolpages_compaction_test.go — the gateway half of #2440's acceptance gate: the
// tool-schema catalog lives in the ctxmmu, not the transcript, so a compaction event
// can only EVICT a schema re-faultably, never LOSE it. Registration is wired at the
// maybeCompactInboundTools seam (registerToolSchemaPages); this witness drives that
// seam, forces a real evict/re-fault paging event, and proves the schema comes back
// byte-identical AND that the /metrics rows the issue names are exposed. The
// resident-set-determinism and prefix-hash-stability halves live in
// internal/ctxmmu/toolpages_test.go; together the three named tests are the gate.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // CAS backend backing the page-out path
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// toolReq builds an Anthropic passthrough request advertising the given (name, schema)
// tool definitions — the shape maybeCompactInboundTools registers into the catalog.
func toolReq(defs ...[2]string) *agent.AnthropicMessagesRequest {
	req := &agent.AnthropicMessagesRequest{}
	for _, d := range defs {
		req.Tools = append(req.Tools, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        d[0],
				Description: d[0] + " tool",
				Parameters:  []byte(d[1]),
			},
		})
	}
	return req
}

// TestToolPages_ByteIdenticalAfterCompaction proves the ctxmmu is the tool catalog's
// home: schemas registered at the maybeCompactInboundTools seam survive a compaction
// event (a real evict through the capability-body pager) byte-identical after re-fault,
// re-advertising an identical schema each turn dedupes by CONTENT HASH (never re-inflates
// the catalog), and the two /metrics rows the issue names are exposed.
func TestToolPages_ByteIdenticalAfterCompaction(t *testing.T) {
	ctx := context.Background()
	s := &Server{toolPages: ctxmmu.NewToolPageTable(ctxmmu.New())}

	readSchema := `{"type":"object","properties":{"path":{"type":"string"}}}`
	grepSchema := `{"type":"object","properties":{"pattern":{"type":"string"}}}`
	req := toolReq([2]string{"fak_read", readSchema}, [2]string{"grep_search", grepSchema})

	// Turn 1: the seam admits both schemas as content-hashed pages (first-seen, no dedup).
	s.registerToolSchemaPages(req)
	if got := s.toolPages.Len(); got != 2 {
		t.Fatalf("catalog size after turn 1 = %d, want 2", got)
	}
	if hits := s.toolPages.DedupHits(); hits != 0 {
		t.Fatalf("DedupHits after first turn = %d, want 0", hits)
	}

	// The re-injection churn the inspiring harness suffered: the SAME schemas re-advertised
	// every turn. Because the catalog is keyed by content hash, turn 2 dedupes both — the
	// catalog never grows and the dedup witness climbs.
	s.registerToolSchemaPages(req)
	if got := s.toolPages.Len(); got != 2 {
		t.Fatalf("re-advertising identical schemas grew the catalog to %d, want 2", got)
	}
	if hits := s.toolPages.DedupHits(); hits != 2 {
		t.Fatalf("DedupHits after re-advertising 2 tools = %d, want 2", hits)
	}

	// The compaction event: page fak_read's schema OUT through the capability-body pager
	// (the same seam compaction uses). The page leaves residency but stays in the catalog.
	hash := ctxmmu.ToolSchemaHash(canonicalToolSchemaBytes(req.Tools[0]))
	if !s.toolPages.Evict(ctx, hash) {
		t.Fatal("evict of the fak_read schema page must succeed with the blob CAS registered")
	}
	if s.toolPages.Resident(hash) {
		t.Fatal("an evicted schema page must not be resident")
	}
	if s.toolPages.Len() != 2 {
		t.Fatalf("eviction changed catalog membership (size %d, want 2): compaction must EVICT, never LOSE", s.toolPages.Len())
	}

	// Re-fault: the schema comes back byte-identical to what the seam registered — never
	// a lost schema (type errors), never mutated read-only content.
	got, err := s.toolPages.Fault(ctx, hash)
	if err != nil {
		t.Fatalf("re-fault after compaction: %v", err)
	}
	want := canonicalToolSchemaBytes(req.Tools[0])
	if !strings.Contains(string(got), readSchema) || len(got) != len(want) {
		t.Fatalf("re-faulted schema not byte-identical:\n got %q\nwant %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("re-faulted schema differs at byte %d: %q vs %q", i, got, want)
		}
	}

	// The /metrics witness: both rows the issue names must be exposed, and the dedup
	// counter must carry the value the catalog actually accrued.
	var b strings.Builder
	s.writeToolPageMetrics(&b)
	out := b.String()
	if !strings.Contains(out, "tool_schema_resident_bytes") {
		t.Errorf("/metrics missing tool_schema_resident_bytes row:\n%s", out)
	}
	if !strings.Contains(out, "fak_gateway_tool_page_dedup_hits_total 2") {
		t.Errorf("/metrics tool_page_dedup_hits_total row wrong (want 2):\n%s", out)
	}
}

// TestToolPages_CanonicalSchemaBytesStable proves the content-hash input the seam feeds
// the catalog is stable for an unchanged tool and distinct for any changed field — so
// dedup keys by real content and two versions of one tool name never collide.
func TestToolPages_CanonicalSchemaBytesStable(t *testing.T) {
	base := agent.ToolDef{Function: agent.ToolDefFunction{Name: "fak_read", Description: "read", Parameters: []byte(`{"a":1}`)}}
	same := agent.ToolDef{Function: agent.ToolDefFunction{Name: "fak_read", Description: "read", Parameters: []byte(`{"a":1}`)}}
	if ctxmmu.ToolSchemaHash(canonicalToolSchemaBytes(base)) != ctxmmu.ToolSchemaHash(canonicalToolSchemaBytes(same)) {
		t.Fatal("identical tool defs must hash identically")
	}
	// Same NAME, different schema bytes → a distinct page (dedup by content, not name).
	v2 := agent.ToolDef{Function: agent.ToolDefFunction{Name: "fak_read", Description: "read", Parameters: []byte(`{"a":2}`)}}
	if ctxmmu.ToolSchemaHash(canonicalToolSchemaBytes(base)) == ctxmmu.ToolSchemaHash(canonicalToolSchemaBytes(v2)) {
		t.Fatal("two versions of one tool name must be two distinct pages")
	}
	// An empty def contributes nothing to page.
	if canonicalToolSchemaBytes(agent.ToolDef{}) != nil {
		t.Fatal("an empty tool def must yield no schema bytes")
	}
}
