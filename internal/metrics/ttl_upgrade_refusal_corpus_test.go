package metrics

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTTLUpgradeRefusalCorpusWitness(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "nightrun", "gateway-usage.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	type counters struct {
		CachedPromptTokens       uint64            `json:"cached_prompt_tokens"`
		CachedTurns              uint64            `json:"cached_turns"`
		CacheTTLUpgradesUpgraded uint64            `json:"cache_ttl_upgrades_upgraded"`
		CacheTTLUpgradeReasons   map[string]uint64 `json:"cache_ttl_upgrade_reasons"`
	}
	type row struct {
		Counters counters `json:"counters"`
	}

	var rows, witnesses int
	var got counters
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		rows++
		var r row
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			t.Fatalf("row %d: %v", rows, err)
		}
		if len(r.Counters.CacheTTLUpgradeReasons) > 0 {
			witnesses++
			got = r.Counters
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}

	if rows != 3353 || witnesses != 1 {
		t.Fatalf("corpus rows=%d reason-bearing rows=%d, want 3353 and 1", rows, witnesses)
	}
	if got.CacheTTLUpgradeReasons["volatile_head"] != 45 || got.CacheTTLUpgradeReasons["no_stable_breakpoint"] != 1 {
		t.Fatalf("refusal histogram = %#v", got.CacheTTLUpgradeReasons)
	}
	if got.CacheTTLUpgradesUpgraded != 19 || got.CachedPromptTokens != 3519466 || got.CachedTurns != 51 {
		t.Fatalf("witness counters = %#v", got)
	}
}
