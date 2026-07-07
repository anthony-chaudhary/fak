package ctxmmu_test

// toolpages_test.go — witness suite for #2440's ctxmmu half: the tool-schema
// page table's resident-set determinism and prefix-hash stability. The
// compaction-survival byte-identity witness (the gateway seam) lives in
// internal/gateway/toolpages_compaction_test.go; together the three named tests
// are the issue's acceptance gate.

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob" // CAS backend backing the page-out path
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// toolSchema renders a plausible JSON schema body for a named tool, distinct per
// (name, version) pair so distinct versions are distinct bytes.
func toolSchema(name, version string) []byte {
	return []byte(`{"name":"` + name + `","description":"` + name + ` v` + version +
		`","input_schema":{"type":"object","properties":{"arg":{"type":"string"}}}}`)
}

// TestToolPages_ResidentSetDeterministic proves the per-turn resident set is a
// pure function of (catalog, intent, max): identical across repeated calls,
// across tables built in permuted registration order, and after dedup — and that
// dedup is keyed by CONTENT HASH, never by tool name, so two versions of one
// tool name are two pages that never collide.
func TestToolPages_ResidentSetDeterministic(t *testing.T) {
	type reg struct {
		name string
		body []byte
	}
	regs := []reg{
		{"fak_read", toolSchema("fak_read", "1")},
		{"fak_read", toolSchema("fak_read", "2")}, // same NAME, different bytes — must NOT collide
		{"fak_changes", toolSchema("fak_changes", "1")},
		{"grep_search", toolSchema("grep_search", "1")},
		{"web_fetch", toolSchema("web_fetch", "1")},
	}

	build := func(order []int) *ctxmmu.ToolPageTable {
		table := ctxmmu.NewToolPageTable(ctxmmu.New())
		for _, i := range order {
			if _, dedup := table.Register(regs[i].name, regs[i].body); dedup {
				t.Fatalf("first registration of regs[%d] must not dedup", i)
			}
		}
		return table
	}
	a := build([]int{0, 1, 2, 3, 4})
	b := build([]int{4, 2, 0, 3, 1}) // permuted registration order

	// Two versions of the same tool name are two distinct pages (dedup by hash).
	if a.Len() != len(regs) {
		t.Fatalf("catalog size = %d, want %d (two fak_read versions must be two pages)", a.Len(), len(regs))
	}

	// Re-registering identical bytes (any name) dedupes to the existing page.
	if _, dedup := a.Register("fak_read", regs[0].body); !dedup {
		t.Fatal("re-registering identical schema bytes must dedup")
	}
	if _, dedup := a.Register("renamed_tool", regs[0].body); !dedup {
		t.Fatal("identical bytes under another name are the SAME page (dedup by hash, not name)")
	}
	if got := a.DedupHits(); got != 2 {
		t.Fatalf("DedupHits = %d, want 2", got)
	}
	if a.Len() != len(regs) {
		t.Fatalf("dedup grew the catalog to %d, want %d", a.Len(), len(regs))
	}

	for _, tc := range []struct {
		intent string
		max    int
	}{
		{"read the changed files", 2},
		{"search the web", 0},
		{"", 3},                        // empty intent → canonical-catalog fallback
		{"zz-nothing-matches-this", 4}, // zero-score → canonical-catalog fallback
	} {
		first := a.ResidentSet(tc.intent, tc.max)
		if again := a.ResidentSet(tc.intent, tc.max); !reflect.DeepEqual(first, again) {
			t.Fatalf("intent %q: repeated call differs:\n%v\n%v", tc.intent, first, again)
		}
		if other := b.ResidentSet(tc.intent, tc.max); !reflect.DeepEqual(first, other) {
			t.Fatalf("intent %q: permuted registration order changed the resident set:\n%v\n%v", tc.intent, first, other)
		}
		if tc.max > 0 && len(first) > tc.max {
			t.Fatalf("intent %q: resident set %d exceeds max %d", tc.intent, len(first), tc.max)
		}
		if len(first) == 0 {
			t.Fatalf("intent %q: resident set must never be empty on a non-empty catalog", tc.intent)
		}
	}

	// The ranked order itself is deterministic and intent-led: "read" must rank a
	// fak_read page first (name-part hit beats the canonical fallback order).
	ranked := a.ResidentSet("read", 0)
	if ranked[0].Name != "fak_read" {
		t.Fatalf("intent \"read\": top page = %q, want fak_read", ranked[0].Name)
	}
}

// TestToolPages_PrefixHashStable proves the splice commitment cannot move: the
// prefix hash is independent of input order (canonical splice order), unchanged
// across a REAL evict/re-fault paging event, and moves only when the resident
// set genuinely changes.
func TestToolPages_PrefixHashStable(t *testing.T) {
	ctx := context.Background()
	table := ctxmmu.NewToolPageTable(ctxmmu.New())
	for _, name := range []string{"fak_read", "fak_changes", "web_fetch"} {
		if _, dedup := table.Register(name, toolSchema(name, "1")); dedup {
			t.Fatalf("unexpected dedup registering %s", name)
		}
	}

	resident := table.ResidentSet("", 0)
	h1 := ctxmmu.PrefixHash(resident)
	if h1 == "" {
		t.Fatal("prefix hash must be non-empty for a non-empty resident set")
	}

	// Input order must not matter: the splice order is canonical, not caller order.
	permuted := []ctxmmu.ToolPage{resident[2], resident[0], resident[1]}
	if got := ctxmmu.PrefixHash(permuted); got != h1 {
		t.Fatalf("prefix hash depends on input order: %s vs %s", got, h1)
	}

	// A REAL paging event: evict one page through the capability-body pager, then
	// re-fault it. The recomputed hash over the same selection must not move.
	victim := resident[1]
	if !table.Evict(ctx, victim.Hash) {
		t.Fatalf("evict of %s must succeed with the blob CAS backend registered", victim.Name)
	}
	if table.Resident(victim.Hash) {
		t.Fatal("an evicted page must not be resident")
	}
	if got := ctxmmu.PrefixHash(table.ResidentSet("", 0)); got != h1 {
		t.Fatalf("prefix hash moved across an EVICTION (paging must never tax the prompt prefix): %s vs %s", got, h1)
	}
	refaulted, err := table.Fault(ctx, victim.Hash)
	if err != nil {
		t.Fatalf("re-fault of %s: %v", victim.Name, err)
	}
	if !bytes.Equal(refaulted, toolSchema(victim.Name, "1")) {
		t.Fatal("re-faulted page bytes differ from the registered schema")
	}
	if got := ctxmmu.PrefixHash(table.ResidentSet("", 0)); got != h1 {
		t.Fatalf("prefix hash moved across a RE-FAULT: %s vs %s", got, h1)
	}

	// Sanity: the hash is a real commitment — a different resident set moves it.
	if got := ctxmmu.PrefixHash(resident[:2]); got == h1 {
		t.Fatal("a smaller resident set must yield a different prefix hash")
	}
}
