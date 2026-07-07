package enginecache_test

// capability_inventory_test.go is the focused witness for DEFAULT-ENABLEMENT-NEXT-50
// item 31 / issue #1549: the external-engine cache capability inventory. It loads the
// machine-read row-set at docs/cache-frontier/external-engine-cache-capability-inventory.jsonl
// and asserts (a) a row exists for each of the five named engines, (b) every verdict is a
// member of the closed capability vocabulary (no free-text), and (c) the SGLang/vLLM rows
// do not claim exact-span eviction — tied to the live enginecache.SupportsExactSpan fact so
// the doc inventory cannot silently drift from the code it summarizes.

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

const inventoryPath = "../../docs/cache-frontier/external-engine-cache-capability-inventory.jsonl"

// closedVocabulary is the exact set of allowed verdicts from issue #1549's default/evidence
// target. It must stay identical to the #1550 engine.CacheCapability contract terms.
var closedVocabulary = map[string]bool{
	"passive observe": true,
	"active warm":     true,
	"exact evict":     true,
	"prefix clone":    true,
	"paged KV":        true,
	"unknown":         true,
}

// requiredEngines are the five external engines the item names, keyed by the JSONL "engine"
// id. The value maps to the enginecache.Engine identity when the tree governs that engine's
// cache (empty when fak has no cache-control identity for it yet).
var requiredEngines = map[string]enginecache.Engine{
	"sglang":    enginecache.EngineSGLang,
	"vllm":      enginecache.EngineVLLM,
	"llama.cpp": "",
	"ollama":    "",
	"lm-studio": "",
}

type inventoryRow struct {
	Engine   string `json:"engine"`
	Verdict  string `json:"verdict"`
	Evidence string `json:"evidence"`
}

func loadInventory(t *testing.T) []inventoryRow {
	t.Helper()
	f, err := os.Open(filepath.FromSlash(inventoryPath))
	if err != nil {
		t.Fatalf("open inventory %s: %v", inventoryPath, err)
	}
	defer f.Close()

	var rows []inventoryRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r inventoryRow
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("parse inventory row %q: %v", line, err)
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan inventory: %v", err)
	}
	return rows
}

func TestExternalEngineCacheCapabilityInventory(t *testing.T) {
	rows := loadInventory(t)

	byEngine := make(map[string]inventoryRow, len(rows))
	for _, r := range rows {
		if _, dup := byEngine[r.Engine]; dup {
			t.Errorf("duplicate inventory row for engine %q", r.Engine)
		}
		byEngine[r.Engine] = r
	}

	// (a) every named engine has a row.
	for id := range requiredEngines {
		if _, ok := byEngine[id]; !ok {
			t.Errorf("inventory is missing a row for required engine %q", id)
		}
	}

	// (b) every verdict is a member of the closed vocabulary (no free-text), and
	// (c) the row carries a non-empty evidence anchor.
	for _, r := range rows {
		if !closedVocabulary[r.Verdict] {
			t.Errorf("engine %q verdict %q is not in the closed capability vocabulary", r.Engine, r.Verdict)
		}
		if strings.TrimSpace(r.Evidence) == "" {
			t.Errorf("engine %q has no evidence anchor", r.Engine)
		}
	}
}

// TestInventoryExactEvictMatchesEngineCache ties the SGLang/vLLM rows to the live
// enginecache fact: their public control plane resets the whole prefix/radix cache, so
// SupportsExactSpan is false and neither row may claim "exact evict". This is the anti-drift
// guard between the doc inventory and the code it summarizes.
func TestInventoryExactEvictMatchesEngineCache(t *testing.T) {
	rows := loadInventory(t)
	byEngine := make(map[string]inventoryRow, len(rows))
	for _, r := range rows {
		byEngine[r.Engine] = r
	}

	for id, eng := range requiredEngines {
		if eng == "" {
			continue // fak has no cache-control identity for this engine yet.
		}
		if enginecache.SupportsExactSpan(eng) {
			continue // if the code ever gains exact-span, the inventory may claim it.
		}
		if r, ok := byEngine[id]; ok && r.Verdict == "exact evict" {
			t.Errorf("engine %q claims %q but enginecache.SupportsExactSpan(%q)=false", id, r.Verdict, eng)
		}
	}
}
