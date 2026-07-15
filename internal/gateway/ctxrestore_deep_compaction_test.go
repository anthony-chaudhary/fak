package gateway

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

func TestCtxRestoreOriginatingTaskAfterRepeatedCompaction(t *testing.T) {
	const task = "rotate the auth tokens and record the witnessed rollout"
	body := repeatedCompactionBody(t, task)
	var restoreID string
	var restoreBytes []byte

	for round := 0; round < 6; round++ {
		compacted, outcome := agent.CompactAnthropicHistoryWithOptions(body, agent.CompactOptions{
			Budget:    700 - round*40,
			Anchor:    agent.CompactAnchorHead,
			ColdCache: true,
		})
		if restoreID == "" && outcome.RestoreID != "" && strings.Contains(string(outcome.RestoreBytes), task) {
			restoreID = outcome.RestoreID
			restoreBytes = append(restoreBytes[:0], outcome.RestoreBytes...)
		}
		if restoreID != "" && !strings.Contains(string(compacted), restoreID) {
			t.Fatalf("round %d: compacted body lost originating-task handle %q", round, restoreID)
		}
		body = compacted
	}
	if restoreID == "" || len(restoreBytes) == 0 {
		t.Fatal("repeated compaction did not expose originating-task restore material")
	}
	if got := ctxplan.Digest(restoreBytes); got != restoreID {
		t.Fatalf("restore id=%q digest=%q", restoreID, got)
	}

	srv := newTestServer(t)
	const trace = "deep-compaction"
	srv.stashRestore(trace, restoreID, task, restoreBytes)
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: restoreID, TraceID: trace})
	if err != nil {
		t.Fatalf("restore after repeated compaction: %v", err)
	}
	if got.Bytes != string(restoreBytes) || !strings.Contains(got.Bytes, task) {
		t.Fatalf("recovered bytes=%q want seed containing %q", got.Bytes, task)
	}
	if got.Provenance != "WITNESSED" {
		t.Fatalf("provenance=%q want WITNESSED", got.Provenance)
	}
}

func repeatedCompactionBody(t *testing.T, task string) []byte {
	t.Helper()
	type block map[string]any
	messages := []map[string]any{{"role": "user", "content": []block{{"type": "text", "text": task}}}}
	for i := 1; i < 18; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages = append(messages, map[string]any{"role": role, "content": []block{{"type": "text", "text": strings.Repeat("round residue ", 20)}}})
	}
	body, err := json.Marshal(map[string]any{
		"model":      "test",
		"max_tokens": 512,
		"system":     []block{{"type": "text", "text": "shared policy", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages":   messages,
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}
