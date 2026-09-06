package selfquery

import (
	"reflect"
	"strings"
	"testing"
)

func TestFindBundles_ManageDurableMemory(t *testing.T) {
	bundles := FindBundles("manage durable memory")
	if len(bundles) == 0 {
		t.Fatal("expected at least one bundle for query 'manage durable memory', got none")
	}

	top := bundles[0]
	if top.ID != "memory_drivers" {
		t.Fatalf("expected top bundle ID 'memory_drivers', got %q", top.ID)
	}
	if top.Domain != "memory" {
		t.Fatalf("expected top bundle Domain 'memory', got %q", top.Domain)
	}

	wantTools := []string{"fak_read", "fak_recall", "fak_save_memory", "fak_delete_memory"}
	if !reflect.DeepEqual(top.Tools, wantTools) {
		t.Fatalf("expected tools %v, got %v", wantTools, top.Tools)
	}

	// Verify other memory-related queries also discover memory_drivers
	for _, q := range []string{"durable memory", "store memory", "recall"} {
		res := FindBundles(q)
		if len(res) == 0 || res[0].ID != "memory_drivers" {
			t.Fatalf("expected 'memory_drivers' for query %q, got %+v", q, res)
		}
	}
}

func TestFindBundles_ContextCompaction(t *testing.T) {
	bundles := FindBundles("context compaction")
	if len(bundles) == 0 {
		t.Fatal("expected at least one bundle for query 'context compaction', got none")
	}

	top := bundles[0]
	if top.ID != "context_mmu" {
		t.Fatalf("expected top bundle ID 'context_mmu', got %q", top.ID)
	}
	if top.Domain != "context" {
		t.Fatalf("expected top bundle Domain 'context', got %q", top.Domain)
	}

	wantTools := []string{"ctx_page_in", "ctx_page_out", "ctx_snapshot", "ctx_evict"}
	if !reflect.DeepEqual(top.Tools, wantTools) {
		t.Fatalf("expected tools %v, got %v", wantTools, top.Tools)
	}

	// Verify context keywords also match context_mmu
	for _, q := range []string{"mmu", "paging", "kv cache"} {
		res := FindBundles(q)
		if len(res) == 0 || res[0].ID != "context_mmu" {
			t.Fatalf("expected 'context_mmu' for query %q, got %+v", q, res)
		}
	}
}

func TestFindBundles_KernelAdjudication(t *testing.T) {
	for _, q := range []string{"adjudicate", "admission", "kernel primitive", "syscall", "adjudication"} {
		res := FindBundles(q)
		if len(res) == 0 {
			t.Fatalf("expected bundle match for %q, got none", q)
		}
		if res[0].ID != "kernel_adjudication" {
			t.Fatalf("expected top bundle 'kernel_adjudication' for query %q, got %q", q, res[0].ID)
		}
		if res[0].Domain != "kernel" {
			t.Fatalf("expected domain 'kernel' for query %q, got %q", q, res[0].Domain)
		}
	}
}

func TestFaultBundle_MemoryDrivers_Atomic(t *testing.T) {
	tools, err := FaultBundle("memory_drivers")
	if err != nil {
		t.Fatalf("unexpected error faulting memory_drivers: %v", err)
	}

	want := []string{"fak_read", "fak_recall", "fak_save_memory", "fak_delete_memory"}
	if !reflect.DeepEqual(tools, want) {
		t.Fatalf("expected atomically faulted tools %v, got %v", want, tools)
	}

	// Verify slice mutation does not affect internal registry
	tools[0] = "mutated_tool"
	toolsAgain, err := FaultBundle("memory_drivers")
	if err != nil {
		t.Fatalf("unexpected error on second fault: %v", err)
	}
	if toolsAgain[0] != "fak_read" {
		t.Fatalf("internal registry was corrupted by caller slice mutation: got %q, want 'fak_read'", toolsAgain[0])
	}
}

func TestFaultBundle_ContextMMU_Atomic(t *testing.T) {
	tools, err := FaultBundle("context_mmu")
	if err != nil {
		t.Fatalf("unexpected error faulting context_mmu: %v", err)
	}

	want := []string{"ctx_page_in", "ctx_page_out", "ctx_snapshot", "ctx_evict"}
	if !reflect.DeepEqual(tools, want) {
		t.Fatalf("expected atomically faulted tools %v, got %v", want, tools)
	}
}

func TestFaultBundle_KernelAdjudication_Atomic(t *testing.T) {
	for _, id := range []string{"kernel_adjudication", "adjudication"} {
		tools, err := FaultBundle(id)
		if err != nil {
			t.Fatalf("unexpected error faulting %q: %v", id, err)
		}
		want := []string{"fak_adjudicate", "fak_syscall"}
		if !reflect.DeepEqual(tools, want) {
			t.Fatalf("expected %v for %q, got %v", want, id, tools)
		}
	}
}

func TestFaultBundle_Unknown(t *testing.T) {
	badIDs := []string{
		"unknown_bundle",
		"non_existent",
		"",
		"   ",
	}

	for _, id := range badIDs {
		tools, err := FaultBundle(id)
		if err == nil {
			t.Fatalf("expected error for unknown bundle ID %q, got tools %v", id, tools)
		}
		if tools != nil {
			t.Fatalf("expected nil tools for unknown bundle ID %q, got %v", id, tools)
		}
	}
}

func TestFindBundles_EmptyAndUnrelated(t *testing.T) {
	emptyQueries := []string{"", "   ", "\t\n"}
	for _, q := range emptyQueries {
		res := FindBundles(q)
		if len(res) != 0 {
			t.Fatalf("expected 0 bundles for empty query %q, got %d", q, len(res))
		}
	}

	unrelatedQueries := []string{
		"weather forecast for seattle tomorrow",
		"quantum physics entangled particle simulation",
		"banana split ice cream sundae recipe",
		"completely unrelated text that matches no keywords",
	}
	for _, q := range unrelatedQueries {
		res := FindBundles(q)
		if len(res) != 0 {
			t.Fatalf("expected 0 bundles for unrelated query %q, got %d (top: %q)", q, len(res), res[0].ID)
		}
	}
}

func TestSearchWithBundles_Helper(t *testing.T) {
	// Query matching memory_drivers
	bundles, tools := SearchWithBundles("manage durable memory")
	if len(bundles) == 0 || bundles[0].ID != "memory_drivers" {
		t.Fatalf("expected memory_drivers bundle, got %+v", bundles)
	}
	wantMemTools := []string{"fak_read", "fak_recall", "fak_save_memory", "fak_delete_memory"}
	if !reflect.DeepEqual(tools, wantMemTools) {
		t.Fatalf("expected tools %v, got %v", wantMemTools, tools)
	}

	// Query matching context_mmu
	bundles, tools = SearchWithBundles("context compaction")
	if len(bundles) == 0 || bundles[0].ID != "context_mmu" {
		t.Fatalf("expected context_mmu bundle, got %+v", bundles)
	}
	wantCtxTools := []string{"ctx_page_in", "ctx_page_out", "ctx_snapshot", "ctx_evict"}
	if !reflect.DeepEqual(tools, wantCtxTools) {
		t.Fatalf("expected tools %v, got %v", wantCtxTools, tools)
	}

	// Empty query
	bundles, tools = SearchWithBundles("")
	if bundles != nil || tools != nil {
		t.Fatalf("expected nil results for empty query, got bundles=%v tools=%v", bundles, tools)
	}

	// Unrelated query
	bundles, tools = SearchWithBundles("unrelated query 12345")
	if bundles != nil || tools != nil {
		t.Fatalf("expected nil results for unrelated query, got bundles=%v tools=%v", bundles, tools)
	}
}

func TestCatalog_SearchWithBundles(t *testing.T) {
	cat, err := Load("", Options{
		Tools: []ToolDescriptor{
			{Name: "fak_read", Description: "Read file with receipt"},
			{Name: "fak_extra_tool", Description: "Extra memory utility tool"},
			{Name: "unrelated_tool", Description: "Something else entirely"},
		},
	})
	if err != nil {
		t.Fatalf("failed to load catalog: %v", err)
	}

	// Catalog SearchWithBundles should return bundle tools first (atomic faulting)
	// followed by any extra matching tools from the catalog.
	bundles, tools := cat.SearchWithBundles("manage durable memory")
	if len(bundles) == 0 || bundles[0].ID != "memory_drivers" {
		t.Fatalf("expected memory_drivers bundle, got %+v", bundles)
	}

	// Verify all 4 bundle tools are present at the front of the list
	for i, want := range []string{"fak_read", "fak_recall", "fak_save_memory", "fak_delete_memory"} {
		if i >= len(tools) || tools[i] != want {
			t.Fatalf("expected tool[%d] = %q, got %v", i, want, tools)
		}
	}

	// Also verify Catalog wrapper methods
	catalogBundles := cat.FindBundles("context compaction")
	if len(catalogBundles) == 0 || catalogBundles[0].ID != "context_mmu" {
		t.Fatalf("cat.FindBundles failed: %+v", catalogBundles)
	}

	ctxTools, err := cat.FaultBundle("context_mmu")
	if err != nil || len(ctxTools) != 4 {
		t.Fatalf("cat.FaultBundle failed: tools=%v, err=%v", ctxTools, err)
	}
}

func TestToolBundle_MethodsAndRegistration(t *testing.T) {
	defer ResetBundles()

	custom := ToolBundle{
		ID:          "custom_analytics",
		Name:        "Custom Analytics",
		Description: "Telemetry and analysis tools",
		Domain:      "telemetry",
		Tools:       []string{"tool_metric", "tool_trace"},
		Keywords:    []string{"telemetry", "custom analytics", "metrics"},
	}

	if err := RegisterBundle(custom); err != nil {
		t.Fatalf("failed to register custom bundle: %v", err)
	}

	// Invalid registrations
	if err := RegisterBundle(ToolBundle{ID: ""}); err == nil {
		t.Fatal("expected error registering bundle with empty ID")
	}
	if err := RegisterBundle(ToolBundle{ID: "no_tools", Tools: nil}); err == nil {
		t.Fatal("expected error registering bundle with nil Tools")
	}

	// Check ContainsTool
	if !custom.ContainsTool("tool_metric") {
		t.Fatal("ContainsTool('tool_metric') should be true")
	}
	if custom.ContainsTool("fak_read") {
		t.Fatal("ContainsTool('fak_read') should be false")
	}

	// Find custom bundle
	res := FindBundles("custom analytics")
	if len(res) == 0 || res[0].ID != "custom_analytics" {
		t.Fatalf("expected custom_analytics bundle, got %+v", res)
	}

	// Fault custom bundle
	faulted, err := FaultBundle("custom_analytics")
	if err != nil {
		t.Fatalf("failed to fault custom bundle: %v", err)
	}
	if !reflect.DeepEqual(faulted, custom.Tools) {
		t.Fatalf("expected %v, got %v", custom.Tools, faulted)
	}

	// ResetBundles and verify custom bundle is gone
	ResetBundles()
	_, err = FaultBundle("custom_analytics")
	if err == nil {
		t.Fatal("expected error faulting custom_analytics after ResetBundles")
	}

	// Canonical bundles should still exist
	canonical := RegisteredBundles()
	if len(canonical) != 3 {
		t.Fatalf("expected 3 canonical bundles after reset, got %d", len(canonical))
	}
}

func TestRegisteredBundles_Sorted(t *testing.T) {
	bundles := RegisteredBundles()
	for i := 1; i < len(bundles); i++ {
		if strings.Compare(bundles[i-1].ID, bundles[i].ID) >= 0 {
			t.Fatalf("RegisteredBundles not sorted: %q >= %q", bundles[i-1].ID, bundles[i].ID)
		}
	}
}
