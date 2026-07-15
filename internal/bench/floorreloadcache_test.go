package bench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestModelFloorReloadCache(t *testing.T) {
	got := ModelFloorReloadCache(FloorReloadCacheScenario{Turns: 8, ReloadTurn: 4, StablePrefixTokens: 1000, FreshTokensPerTurn: 100})
	if got.ForfeitedCachedTokens != 1000 || got.BaselineCostTokens != 2500 || got.ReloadCostTokens != 3400 || got.TurnsToAmortize != 9 {
		t.Fatalf("result=%+v", got)
	}
}

func TestFloorReloadCacheGolden(t *testing.T) {
	got, err := json.MarshalIndent(DefaultFloorReloadCacheReport(), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	path := filepath.Join("testdata", "floor_reload_cache_report.json")
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(path, got, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("golden mismatch; run UPDATE_GOLDEN=1 go test ./internal/bench -run TestFloorReloadCacheGolden")
	}
}
