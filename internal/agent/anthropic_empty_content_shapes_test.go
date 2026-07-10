package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestRepairEmptyToolResultContent_Shapes is the #4156 regression: before the fix the OUTBOUND
// empty-content gate repaired ONLY content:[] and its fast-path guard skipped the whole body unless
// a literal `[]` was present, so the sibling shapes the detector (toolResultContentIsEmpty) already
// recognized — content:"" and an array of all-empty-text blocks — escaped untouched and 400'd
// upstream as malformed. Each shape here must now be repaired to a non-empty one-element text array,
// and the repair counter (outcome.Repaired, which drives fak_gateway_empty_tool_result_repaired_total)
// must increment.
func TestRepairEmptyToolResultContent_Shapes(t *testing.T) {
	cases := map[string]string{
		// content:"" — an empty STRING content. No literal `[]` anywhere in the body, so the old
		// containsEmptyJSONArray fast-path guard skipped the entire repair and this escaped.
		"empty_string": `{"model":"claude-opus-4-8","messages":[` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":""}]}` +
			`]}`,
		// content:[{"type":"text","text":""}] — a structurally NON-empty array whose only block is
		// empty-text. isEmptyJSONArray called this "not empty"; the detector calls it empty.
		"all_empty_text": `{"model":"claude-opus-4-8","messages":[` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":""}]}]}` +
			`]}`,
		// A multi-block array where EVERY text block is empty is still empty content.
		"all_empty_text_multi": `{"model":"m","messages":[` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":""},{"type":"text","text":""}]}]}` +
			`]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			// The fast-path guard must NOT skip this body (checkbox 2: the guard is a true superset).
			if !containsRepairableEmptyContent([]byte(body)) {
				t.Fatalf("widened fast-path guard skipped a repairable body:\n%s", body)
			}
			out, outcome := RepairEmptyToolResultContent([]byte(body))
			t.Logf("reason=%q repaired=%d out=%s", outcome.Reason, outcome.Repaired, out)
			if outcome.Reason != EmptyContentReasonNone || outcome.Repaired != 1 {
				t.Fatalf("empty content not repaired: reason=%q repaired=%d", outcome.Reason, outcome.Repaired)
			}
			if !strings.Contains(string(out), emptyToolResultText) {
				t.Fatalf("placeholder text missing from repaired body:\n%s", out)
			}
			// The repaired body must be valid JSON and the tool_result content must now be a
			// non-empty one-element text array — i.e. no longer a shape the API 400s as malformed.
			var probe struct {
				Messages []struct {
					Content []struct {
						Type    string          `json:"type"`
						Content json.RawMessage `json:"content"`
					} `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(out, &probe); err != nil {
				t.Fatalf("repaired body is not valid JSON: %v\n%s", err, out)
			}
			inner := probe.Messages[0].Content[0].Content
			if toolResultContentIsEmpty(inner) {
				t.Fatalf("repaired content is still detector-empty (would still 400): %s", inner)
			}
			var blocks []map[string]any
			if err := json.Unmarshal(inner, &blocks); err != nil || len(blocks) != 1 || blocks[0]["type"] != "text" {
				t.Fatalf("repaired content is not a single text block: %s", inner)
			}
			if blocks[0]["text"] == "" {
				t.Fatalf("repaired text block is empty: %s", inner)
			}
		})
	}
}

// TestContainsRepairableEmptyContent_Superset proves the widened fast-path guard is a true SUPERSET
// of toolResultContentIsEmpty for the value-present shapes: any body whose tool_result content is
// detector-empty must make the guard fire (a false there would silently skip the repair, the #4156
// bug). Non-empty bodies may still fire (false positives are the fail-safe direction) so only the
// must-fire direction is asserted.
func TestContainsRepairableEmptyContent_Superset(t *testing.T) {
	mustFire := []string{
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[]}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":""}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":""}]}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[ ]}]}]}`, // whitespace-only array
	}
	for _, body := range mustFire {
		if !containsRepairableEmptyContent([]byte(body)) {
			t.Errorf("guard skipped a detector-empty body (would drop the repair):\n%s", body)
		}
	}
	// A body with no empty array and no empty string must NOT fire — the fast path is preserved for
	// the common case so the every-wire gate stays off the decode path.
	noFire := `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]}]}`
	if containsRepairableEmptyContent([]byte(noFire)) {
		t.Errorf("guard fired on a body with no empty content (fast path lost):\n%s", noFire)
	}
}
