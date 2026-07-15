package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestPositiveResidueFiltersSupersededAndKeepsRestoreBytes(t *testing.T) {
	raw := positiveResidueFixture(t, []string{
		"fact: route=France",
		"state: owner=planner",
		"superseded: route",
		"fact: route=China",
		"abandoned: owner",
	})
	out, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{
		Budget: 700, Anchor: CompactAnchorHead, ColdCache: true, PositiveResidue: true,
	})
	if outcome.PositiveResidue != "route=China" || outcome.PositiveAssertionsKept != 1 {
		t.Fatalf("residue=%q kept=%d", outcome.PositiveResidue, outcome.PositiveAssertionsKept)
	}
	if bytes.Contains(out, []byte("France")) || bytes.Contains(out, []byte("owner=planner")) {
		t.Fatalf("superseded state survived compacted output: %s", out)
	}
	if !bytes.Contains(out, []byte("positive residual state")) || !bytes.Contains(out, []byte("route=China")) {
		t.Fatalf("positive residue missing from output: %s", out)
	}
	if outcome.ResidueRestoreID == "" || len(outcome.ResidueRestoreBytes) == 0 {
		t.Fatalf("restore material missing: %+v", outcome)
	}
	if got := originatingTaskDigestID(outcome.ResidueRestoreBytes); got != outcome.ResidueRestoreID {
		t.Fatalf("restore digest=%q id=%q", got, outcome.ResidueRestoreID)
	}
	if !bytes.Contains(outcome.ResidueRestoreBytes, []byte("France")) {
		t.Fatalf("raw dropped bytes not retrievable: %s", outcome.ResidueRestoreBytes)
	}
	if outcome.ResidueBytesDropped != len(outcome.ResidueRestoreBytes) {
		t.Fatalf("dropped metric=%d bytes=%d", outcome.ResidueBytesDropped, len(outcome.ResidueRestoreBytes))
	}
}

func TestPositiveResidueEmptyWhenNoPositiveState(t *testing.T) {
	raw := positiveResidueFixture(t, []string{"fact: route=France", "fixed: route", "we tried X and it failed"})
	out, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{
		Budget: 700, Anchor: CompactAnchorHead, ColdCache: true, PositiveResidue: true,
	})
	if outcome.PositiveResidue != "" || outcome.PositiveAssertionsKept != 0 {
		t.Fatalf("empty residue=%q kept=%d", outcome.PositiveResidue, outcome.PositiveAssertionsKept)
	}
	if bytes.Contains(out, []byte("we tried X")) || bytes.Contains(out, []byte("route=France")) {
		t.Fatalf("negated prose survived: %s", out)
	}
}

func TestPositiveResidueOffByDefault(t *testing.T) {
	raw := positiveResidueFixture(t, []string{"fact: route=China"})
	out, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: 700, Anchor: CompactAnchorHead, ColdCache: true})
	if outcome.PositiveResidue != "" || outcome.ResidueRestoreID != "" || bytes.Contains(out, []byte("positive residual state")) {
		t.Fatalf("opt-in feature active by default: outcome=%+v out=%s", outcome, out)
	}
}

func positiveResidueFixture(t *testing.T, lines []string) []byte {
	t.Helper()
	type block map[string]any
	messages := make([]map[string]any, 0, 18)
	for i := 0; i < 18; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		text := strings.Repeat("droppable filler ", 20)
		if i == 0 {
			text = "complete the originating task"
		} else if i-1 < len(lines) {
			text = lines[i-1]
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
