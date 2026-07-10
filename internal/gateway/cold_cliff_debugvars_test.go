package gateway

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// #3623: the live frozen-trajectory cache-cliff finding rides the AdjudicationSummary,
// which /debug/vars embeds verbatim under "adjudication" (see debug.go). This witnesses the
// wire contract deterministically — without depending on the process-global cacheobs.Default
// state, which sibling tests share: a fired verdict surfaces PREFIX_COLD_CLIFF under
// kv_prefix_cold_cliff; a healthy session omits the key entirely (omitempty), so its mere
// presence is the alarm.
func TestColdCliffFindingWireShapeOnAdjudicationSummary(t *testing.T) {
	// Healthy session: nil verdict -> key absent.
	healthy, err := json.Marshal(AdjudicationSummary{})
	if err != nil {
		t.Fatalf("marshal healthy summary: %v", err)
	}
	if strings.Contains(string(healthy), "kv_prefix_cold_cliff") {
		t.Fatalf("healthy summary must omit kv_prefix_cold_cliff, got:\n%s", healthy)
	}

	// Cold-cliffed session: the detector's verdict surfaces the finding + its evidence.
	cliff := cacheobs.Stats{Turns: 4, ColdTurns: 3, ReuseRatio: 0.2}.ColdCliff()
	if !cliff.Fired {
		t.Fatalf("expected a fired verdict from a cold-dominated snapshot, got %+v", cliff)
	}
	fired, err := json.Marshal(AdjudicationSummary{KVPrefixColdCliff: &cliff})
	if err != nil {
		t.Fatalf("marshal cliffed summary: %v", err)
	}
	body := string(fired)
	for _, want := range []string{
		`"kv_prefix_cold_cliff"`,
		`"finding":"` + cacheobs.ColdCliffFinding + `"`,
		`"reason":"cold_fraction"`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("cliffed summary missing %q, got:\n%s", want, body)
		}
	}
	// Fired is presence-encoded, never a redundant always-true wire field.
	if strings.Contains(body, `"fired"`) {
		t.Fatalf("Fired must not be serialized (presence is the alarm), got:\n%s", body)
	}
}
