package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCompactOriginatingTaskSurvivesDeepCompaction(t *testing.T) {
	seedTask := "rotate the auth tokens and record the witnessed rollout"
	body := deepCompactionFixture(seedTask, false)
	var stableID string
	var restoreBytes []byte
	for round := 0; round < 6; round++ {
		budget := 700 - round*40
		compacted, outcome := CompactAnthropicHistoryWithOptions(body, CompactOptions{Budget: budget, Anchor: CompactAnchorHead, ColdCache: true})
		assertOriginRecoverable(t, round, compacted, outcome, seedTask, &stableID, &restoreBytes)
		body = appendDeepCompactionResidue(t, compacted, round)
	}
	if stableID == "" {
		t.Fatal("deep compaction never produced the expected originating-task restore handle")
	}
}

func TestCompactGoalMarkedOriginatingTaskSurvivesDeepCompaction(t *testing.T) {
	seedTask := "ship the signed release witness"
	body := deepCompactionFixture(compactGoalMarker+" "+seedTask, true)
	for round := 0; round < 6; round++ {
		budget := 700 - round*40
		compacted, outcome := CompactAnthropicHistoryWithOptions(body, CompactOptions{Budget: budget, Anchor: CompactAnchorHead, ColdCache: true})
		if !bytes.Contains(compacted, []byte(seedTask)) || !bytes.Contains(compacted, []byte(compactGoalMarker)) {
			t.Fatalf("round %d: goal-pinned originating task left body: %s", round, compacted)
		}
		if outcome.RestoreID != "" || len(outcome.RestoreBytes) != 0 {
			t.Fatalf("round %d: goal-pinned task was tombstoned: id=%q bytes=%d", round, outcome.RestoreID, len(outcome.RestoreBytes))
		}
		body = appendDeepCompactionResidue(t, compacted, round)
	}
}

func deepCompactionFixture(task string, goal bool) []byte {
	type block map[string]any
	content := task
	if goal && !strings.Contains(content, compactGoalMarker) {
		content = compactGoalMarker + " " + content
	}
	messages := []map[string]any{{"role": "user", "content": []block{{"type": "text", "text": content}}}}
	for i := 1; i < 18; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages = append(messages, map[string]any{"role": role, "content": []block{{"type": "text", "text": strings.Repeat("round residue ", 20)}}})
	}
	body, _ := json.Marshal(map[string]any{
		"model":      "test",
		"max_tokens": 512,
		"system":     []block{{"type": "text", "text": "shared policy", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages":   messages,
	})
	return body
}

func assertOriginRecoverable(t *testing.T, round int, body []byte, outcome CompactOutcome, task string, stableID *string, restoreBytes *[]byte) {
	t.Helper()
	bodyID := restoreIDFromCompactedBody(body)
	if bodyID == "" && bytes.Contains(body, []byte(task)) {
		return
	}
	if bodyID == "" {
		t.Fatalf("round %d: originating task absent and no restore handle: outcome=%+v body=%s", round, outcome, body)
	}
	if outcome.RestoreID != "" && outcome.RestoreID != bodyID {
		t.Fatalf("round %d: outcome restore id=%q body id=%q", round, outcome.RestoreID, bodyID)
	}
	if len(outcome.RestoreBytes) > 0 {
		if got := originatingTaskDigestID(outcome.RestoreBytes); bodyID != got {
			t.Fatalf("round %d: restore id=%q digest=%q", round, bodyID, got)
		}
		if text, ok := elementTextContent(outcome.RestoreBytes); !ok || !strings.Contains(text, task) {
			t.Fatalf("round %d: restore bytes do not recover seed: ok=%v text=%q", round, ok, text)
		}
		*restoreBytes = append((*restoreBytes)[:0], outcome.RestoreBytes...)
	}
	if len(*restoreBytes) == 0 || originatingTaskDigestID(*restoreBytes) != bodyID {
		t.Fatalf("round %d: handle %q has no stable seed bytes", round, bodyID)
	}
	if *stableID == "" {
		*stableID = bodyID
	} else if bodyID != *stableID {
		t.Fatalf("round %d: restore id changed: got=%q want=%q", round, bodyID, *stableID)
	}
}

func restoreIDFromCompactedBody(body []byte) string {
	const digestLen = 64
	marker := []byte(compactRestoreIDField)
	at := bytes.Index(body, marker)
	if at < 0 {
		return ""
	}
	start := at + len(marker)
	if start+digestLen > len(body) {
		return ""
	}
	id := string(body[start : start+digestLen])
	for _, c := range id {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return ""
		}
	}
	return id
}

func appendDeepCompactionResidue(t *testing.T, body []byte, round int) []byte {
	t.Helper()
	var req struct {
		Model    string            `json:"model"`
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("round %d: decode compacted body: %v", round, err)
	}
	req.Messages = append(req.Messages,
		json.RawMessage(`{"role":"assistant","content":"continue from durable state"}`),
		json.RawMessage(`{"role":"user","content":"record the next witnessed effect"}`),
	)
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return out
}
