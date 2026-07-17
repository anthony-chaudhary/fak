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
func TestPositiveResidualSubstitutionConfigGate(t *testing.T) {
	off := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model"})
	if off.positiveResidualSubstitution {
		t.Fatal("positive residual substitution must default off")
	}
	on := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", PositiveResidualSubstitution: true})
	if !on.positiveResidualSubstitution {
		t.Fatal("explicit positive residual substitution config did not reach server")
	}
}

func TestPositiveResidualSubstitutionRestore(t *testing.T) {
	body := positiveResidualGatewayFixture(t)
	decode := func() *agent.AnthropicMessagesRequest {
		req, err := agent.DecodeAnthropicMessagesRequest(body)
		if err != nil {
			t.Fatalf("decode fixture: %v", err)
		}
		return req
	}

	// The production gateway gate is explicit and default-off: even when compaction fires,
	// disabled mode neither substitutes the negated span nor creates a residual restore entry.
	off := anthropicPassthroughServer(700)
	off.compactAnchorHead = true
	reqOff := decode()
	if fired, reason := off.compactAnthropicRawWithReason(reqOff, 100, "positive-residual-off"); !fired || reason != agent.CompactReasonNone {
		t.Fatalf("default-off compaction fired=%v reason=%q", fired, reason)
	}
	if strings.Contains(string(reqOff.Raw), "positive residual state") || strings.Contains(string(reqOff.Raw), "remember to push after the commit.") {
		t.Fatalf("default-off path must not substitute residual: %s", reqOff.Raw)
	}
	off.ctxRestoreMu.Lock()
	offEntries := len(off.ctxRestore["positive-residual-off"].entries)
	off.ctxRestoreMu.Unlock()
	if offEntries != 1 { // the ordinary originating-task tombstone only
		t.Fatalf("default-off path stashed %d entries, want only the originating task", offEntries)
	}

	on := anthropicPassthroughServer(700)
	on.compactAnchorHead = true
	on.positiveResidualSubstitution = true
	const trace = "positive-residual-substitution"
	reqOn := decode()
	if fired, reason := on.compactAnthropicRawWithReason(reqOn, 100, trace); !fired || reason != agent.CompactReasonNone {
		t.Fatalf("enabled compaction fired=%v reason=%q", fired, reason)
	}
	if strings.Contains(string(reqOn.Raw), "Do not forget") || !strings.Contains(string(reqOn.Raw), "remember to push after the commit.") {
		t.Fatalf("negated span survived production compaction: %s", reqOn.Raw)
	}

	on.ctxRestoreMu.Lock()
	stash := on.ctxRestore[trace]
	var residualID string
	if stash != nil {
		for _, entry := range stash.entries {
			if entry.excerpt == "remember to push after the commit." {
				residualID = entry.id
				break
			}
		}
	}
	on.ctxRestoreMu.Unlock()
	if residualID == "" {
		t.Fatal("production gateway did not stash the positive-residual source bytes")
	}
	got, err := on.restoreContext("", ContextRestoreRequest{ID: residualID, TraceID: trace})
	if err != nil {
		t.Fatalf("restore original span: %v", err)
	}
	if !strings.Contains(got.Bytes, "Do not forget to push after the commit.") {
		t.Fatalf("ctxrestore did not recover original bytes: %q", got.Bytes)
	}
}

func positiveResidualGatewayFixture(t *testing.T) []byte {
	t.Helper()
	type block map[string]any
	states := []string{"complete the originating task", "Do not forget to push after the commit."}
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
