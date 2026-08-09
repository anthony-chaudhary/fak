package enginecache_test

import (
	"strings"
	"testing"
)

func TestInventoryIncludesLMCacheAsUnknownExternalTier(t *testing.T) {
	rows := loadInventory(t)
	for _, row := range rows {
		if row.Engine != "lmcache" {
			continue
		}
		if row.Verdict != "unknown" {
			t.Fatalf("verdict = %q, want unknown until fak has an observation/control adapter", row.Verdict)
		}
		for _, want := range []string{"LMCache/LMCache", "4521c3f9f1b8", "no LMCache adapter"} {
			if !strings.Contains(row.Evidence, want) {
				t.Fatalf("evidence missing %q: %q", want, row.Evidence)
			}
		}
		return
	}
	t.Fatal("LMCache inventory row missing")
}
