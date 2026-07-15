package gateway

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestPositiveResidueCompactionRestoreAndMetrics(t *testing.T) {
	body := positiveResidueGatewayFixture(t)
	compacted, outcome := agent.CompactAnthropicHistoryWithOptions(body, agent.CompactOptions{
		Budget: 700, Anchor: agent.CompactAnchorHead, ColdCache: true, PositiveResidue: true,
	})
	if outcome.PositiveResidue != "route=China" || outcome.PositiveAssertionsKept != 1 {
		t.Fatalf("residue=%q kept=%d", outcome.PositiveResidue, outcome.PositiveAssertionsKept)
	}
	if strings.Contains(string(compacted), "route=France") || strings.Contains(string(compacted), "we tried X") {
		t.Fatalf("negated/superseded residue survived: %s", compacted)
	}
	if outcome.ResidueBytesDropped <= 0 || outcome.ResidueRestoreID == "" {
		t.Fatalf("metrics/restore missing: %+v", outcome)
	}

	srv := newTestServer(t)
	const trace = "positive-residue"
	srv.stashRestore(trace, outcome.ResidueRestoreID, outcome.PositiveResidue, outcome.ResidueRestoreBytes)
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: outcome.ResidueRestoreID, TraceID: trace})
	if err != nil {
		t.Fatalf("restore dropped residue: %v", err)
	}
	if got.Bytes != string(outcome.ResidueRestoreBytes) || !strings.Contains(got.Bytes, "route=France") {
		t.Fatalf("raw bytes did not round-trip: %q", got.Bytes)
	}
	if got.Provenance != "WITNESSED" {
		t.Fatalf("provenance=%q", got.Provenance)
	}
}

func positiveResidueGatewayFixture(t *testing.T) []byte {
	t.Helper()
	type block map[string]any
	states := []string{
		"complete the originating task",
		"fact: route=France",
		"we tried X and it failed",
		"superseded: route",
		"fact: route=China",
	}
	messages := make([]map[string]any, 0, 18)
	for i := 0; i < 18; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		text := strings.Repeat("droppable filler ", 20)
		if i < len(states) {
			text = states[i]
		}
		messages = append(messages, map[string]any{"role": role, "content": []block{{"type": "text", "text": text}}})
	}
	body, err := json.Marshal(map[string]any{
		"model": "test", "max_tokens": 512,
		"system":   []block{{"type": "text", "text": "shared policy", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages": messages,
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}
