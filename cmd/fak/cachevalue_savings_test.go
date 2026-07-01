package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestAppendObservedCacheSavingsReportsRowsAndErrors(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens:          20,
		CachedPromptTokens:   60,
		CacheCreationTokens:  20,
		OutputTokens:         3,
		CompactionShedTokens: 9,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}
	if res.RowsPlanned != 2 || res.RowsWritten != 2 {
		t.Fatalf("rows planned/written = %d/%d, want 2/2", res.RowsPlanned, res.RowsWritten)
	}
	if res.ProviderTokenEquiv != 49 || res.FakCompactionTokenEquiv != 9 || res.TotalTokenEquiv != 58 {
		t.Fatalf("token-equiv split = provider %.1f fak %.1f total %.1f, want 49/9/58",
			res.ProviderTokenEquiv, res.FakCompactionTokenEquiv, res.TotalTokenEquiv)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %d, want 2", len(rows))
	}
	if rows[0].Mechanism != "provider_prompt_cache" || rows[1].Mechanism != "compaction_shed" {
		t.Fatalf("mechanisms = %q/%q, want provider_prompt_cache/compaction_shed", rows[0].Mechanism, rows[1].Mechanism)
	}

	badPath := filepath.Join(t.TempDir(), "missing", "cache-savings.jsonl")
	failed := appendObservedCacheSavingsTo(badPath, "guard", "anthropic", "claude", sum, now)
	if failed.Err == nil {
		t.Fatal("append to a missing parent must report an error")
	}
	if failed.RowsPlanned != 2 || failed.RowsWritten != 0 {
		t.Fatalf("failed rows planned/written = %d/%d, want 2/0", failed.RowsPlanned, failed.RowsWritten)
	}
}

func TestFormatCacheValuePersistenceSummaryNamesEvidenceAndNextCommand(t *testing.T) {
	rep := cacheValuePersistenceReport{
		Since: "2026-06-30",
		Track1: cacheValueTrack1Result{
			Path:         "docs/nightrun/cache-value.jsonl",
			RowsPlanned:  1,
			RowsWritten:  1,
			Turns:        12,
			PromptTokens: 2000,
			ReusedTokens: 1500,
		},
		Track2: cacheValueTrack2Result{
			Path:                    "docs/nightrun/cache-savings.jsonl",
			RowsPlanned:             2,
			RowsWritten:             2,
			ProviderTokenEquiv:      1200,
			FakCompactionTokenEquiv: 300,
			TotalTokenEquiv:         1500,
			CacheReadTokens:         1400,
			CompactionShedTokens:    300,
		},
	}
	out := formatCacheValuePersistenceSummary("fak guard", rep)
	for _, want := range []string{
		"fak guard: cache-value evidence",
		"Track 1 WITNESSED kernel row",
		"Track 2 OBSERVED-$ rows",
		"provider +1,200 tok-eq",
		"fak compaction +300 tok-eq",
		"total +1,500 tok-eq",
		"docs/nightrun/cache-value.jsonl",
		"docs/nightrun/cache-savings.jsonl",
		"fak cachevalue report --since 2026-06-30",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("summary missing %q:\n%s", want, out)
		}
	}
}

func TestFormatCacheValuePersistenceSummaryMakesNoEvidenceExplicit(t *testing.T) {
	out := formatCacheValuePersistenceSummary("fak guard", cacheValuePersistenceReport{
		Since: "2026-06-30",
		Track1: cacheValueTrack1Result{
			Path: "docs/nightrun/cache-value.jsonl",
		},
		Track2: cacheValueTrack2Result{
			Path: "docs/nightrun/cache-savings.jsonl",
		},
	})
	for _, want := range []string{
		"cache-value evidence",
		"Track 1 WITNESSED kernel row not written",
		"no KV-prefix reuse turns",
		"Track 2 OBSERVED-$ row not written",
		"no provider-cache or compaction tokens",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("no-evidence summary missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "fak cachevalue report --since") {
		t.Fatalf("no-evidence summary should not point at an empty report command:\n%s", out)
	}
}
