package gateway

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestTokenSavingsDebugDefaultReceipt(t *testing.T) {
	s := newDeferServer(t, false)
	s.deferColdTools = true
	s.elideStaleReads = true
	s.compactHistoryBudget = 2048
	got := s.debugVars(time.Now()).TokenSavings
	if got.NativeMCPFilter.State != "active" || got.NativeMCPFilter.SavedBytes == 0 {
		t.Fatalf("native receipt=%+v", got.NativeMCPFilter)
	}
	for name, lever := range map[string]debugTokenSavingLever{
		"cold": got.ColdToolDefer, "stale": got.StaleReadElide, "compact": got.HistoryCompact,
	} {
		if lever.State != "ready" || lever.Reason != "not_observed" || lever.Rollback == "" {
			t.Errorf("%s receipt=%+v", name, lever)
		}
	}
}

func TestTokenSavingsDebugBailoutsAndPrivacy(t *testing.T) {
	t.Setenv("FAK_ABLATE_MCP_TOOL_FILTER", "1")
	t.Setenv("FAK_ABLATE_DEFER_TOOLS", "1")
	s := newDeferServer(t, false)
	s.deferColdTools = true
	s.elideStaleReads = true
	s.compactHistoryBudget = 2048
	s.metrics.observeStaleElide("no_stale_reads", 0, 0)
	s.metrics.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget}, false)
	got := s.debugVars(time.Now()).TokenSavings
	if got.NativeMCPFilter.Reason != "ablation" || got.ColdToolDefer.Reason != "ablation" {
		t.Fatalf("ablation receipts native=%+v cold=%+v", got.NativeMCPFilter, got.ColdToolDefer)
	}
	if got.StaleReadElide.State != "bypassed" || got.StaleReadElide.Reason != "no_stale_reads" {
		t.Fatalf("stale receipt=%+v", got.StaleReadElide)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"trace_id", "tool_args", "content", "path"} {
		if json.Valid(b) && containsJSONKey(b, forbidden) {
			t.Fatalf("privacy leak key %q in %s", forbidden, b)
		}
	}
}

func containsJSONKey(b []byte, key string) bool {
	var v map[string]any
	if json.Unmarshal(b, &v) != nil {
		return false
	}
	var walk func(any) bool
	walk = func(x any) bool {
		switch y := x.(type) {
		case map[string]any:
			for k, v := range y {
				if k == key || walk(v) {
					return true
				}
			}
		case []any:
			for _, v := range y {
				if walk(v) {
					return true
				}
			}
		}
		return false
	}
	return walk(v)
}
