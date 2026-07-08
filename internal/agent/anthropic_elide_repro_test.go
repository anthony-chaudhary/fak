package agent

// anthropic_elide_repro_test.go — REPRO HARNESS (investigation only, not product code).
//
// Goal: determine whether outbound tool_result elision (ElideAnthropicResultsWithOutcome)
// can emit a body that is valid-JSON and valid-to-fak's-own-decoder yet SEMANTICALLY invalid
// to the real Anthropic /v1/messages API — the class of body that would come back as
// "400 upstream rejected the request as malformed".
//
// fak's post-splice proof (verifySplicedBody, anthropic_compact.go:537) only calls
// DecodeAnthropicMessagesRequest — a permissive json.Unmarshal that accepts empty text,
// empty content arrays, and unpaired tool_use/tool_result. This harness applies the
// STRICTER Anthropic semantic rules to elision output across many thresholds and content
// shapes and reports any shape that yields a semantically-invalid body.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// ---- Anthropic semantic validator (the rules fak's decoder does NOT enforce) ----

// anthropicSemErrors returns the list of Anthropic Messages semantic violations in a body.
// Empty list == a body Anthropic would accept (for the rules we can check statically).
// Rules enforced (the ones a 400 "malformed" would flag):
//   - every message.content array must be non-empty
//   - every "text" block must have a non-empty text value
//   - every tool_result.content (when an array) must be non-empty, and each nested text
//     block non-empty; when a string, it must be non-empty
//   - every tool_result.tool_use_id must reference an assistant tool_use.id earlier in the
//     transcript, and every assistant tool_use.id must be answered by a later tool_result
//     (tool_use / tool_result pairing)
func anthropicSemErrors(raw []byte) []string {
	var errs []string
	var body struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return []string{"body is not JSON: " + err.Error()}
	}

	toolUseIDs := map[string]bool{}    // assistant tool_use ids seen so far
	toolResultIDs := map[string]bool{} // tool_result tool_use_ids seen

	for mi, m := range body.Messages {
		if len(m.Content) == 0 {
			continue // bare-string content on non-tool messages is legal; skip structural checks
		}
		if m.Content[0] == '"' {
			var s string
			if json.Unmarshal(m.Content, &s) == nil && strings.TrimSpace(s) == "" {
				errs = append(errs, fmt.Sprintf("msg[%d] role=%s: bare-string content is empty", mi, m.Role))
			}
			continue
		}
		if m.Content[0] != '[' {
			continue
		}
		var blocks []json.RawMessage
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		if len(blocks) == 0 {
			errs = append(errs, fmt.Sprintf("msg[%d] role=%s: content array is EMPTY", mi, m.Role))
			continue
		}
		for bi, blk := range blocks {
			var b struct {
				Type      string          `json:"type"`
				Text      string          `json:"text"`
				ID        string          `json:"id"`
				ToolUseID string          `json:"tool_use_id"`
				Content   json.RawMessage `json:"content"`
			}
			if json.Unmarshal(blk, &b) != nil {
				continue
			}
			switch b.Type {
			case "text":
				if b.Text == "" {
					errs = append(errs, fmt.Sprintf("msg[%d].block[%d] role=%s: text block has EMPTY text", mi, bi, m.Role))
				}
			case "tool_use":
				if b.ID != "" {
					toolUseIDs[b.ID] = true
				}
			case "tool_result":
				if b.ToolUseID != "" {
					toolResultIDs[b.ToolUseID] = true
					if !toolUseIDs[b.ToolUseID] {
						errs = append(errs, fmt.Sprintf("msg[%d].block[%d]: tool_result tool_use_id=%q has no matching earlier tool_use", mi, bi, b.ToolUseID))
					}
				}
				// content of a tool_result: string or array
				if len(b.Content) == 0 {
					errs = append(errs, fmt.Sprintf("msg[%d].block[%d]: tool_result has NO content", mi, bi))
					break
				}
				switch b.Content[0] {
				case '"':
					var s string
					if json.Unmarshal(b.Content, &s) == nil && s == "" {
						errs = append(errs, fmt.Sprintf("msg[%d].block[%d]: tool_result string content is EMPTY", mi, bi))
					}
				case '[':
					var inner []json.RawMessage
					if json.Unmarshal(b.Content, &inner) == nil {
						if len(inner) == 0 {
							errs = append(errs, fmt.Sprintf("msg[%d].block[%d]: tool_result content array is EMPTY", mi, bi))
						}
						for ii, ib := range inner {
							var tb struct {
								Type string `json:"type"`
								Text string `json:"text"`
							}
							if json.Unmarshal(ib, &tb) == nil && tb.Type == "text" && tb.Text == "" {
								errs = append(errs, fmt.Sprintf("msg[%d].block[%d].inner[%d]: nested text block has EMPTY text", mi, bi, ii))
							}
						}
					}
				}
			}
		}
	}
	// pairing: every assistant tool_use answered by a tool_result
	for id := range toolUseIDs {
		if !toolResultIDs[id] {
			errs = append(errs, fmt.Sprintf("tool_use id=%q is never answered by a tool_result (orphaned tool_use)", id))
		}
	}
	return errs
}

// sanity: the validator flags a known-bad body and passes a known-good one.
func TestReproValidatorSanity(t *testing.T) {
	bad := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":""}]}]}`)
	if errs := anthropicSemErrors(bad); len(errs) == 0 {
		t.Fatal("validator failed to flag an empty text block")
	}
	empty := []byte(`{"messages":[{"role":"user","content":[]}]}`)
	if errs := anthropicSemErrors(empty); len(errs) == 0 {
		t.Fatal("validator failed to flag an empty content array")
	}
	good := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	if errs := anthropicSemErrors(good); len(errs) != 0 {
		t.Fatalf("validator false-flagged a good body: %v", errs)
	}
}

// reproWireBody builds a 10-message body with a paired tool_use (assistant, msg[1]) and
// its tool_result (msg[2], eligible band). shape controls how the eligible tool_result's
// content is encoded. head breakpoint on msg[0]; system carries cc too.
func reproWireBody(t *testing.T, shape, eligiblePayload string) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}

	var eligibleContent any
	switch shape {
	case "string":
		eligibleContent = eligiblePayload
	case "array":
		eligibleContent = []obj{{"type": "text", "text": eligiblePayload}}
	case "array-multi":
		// two text blocks: one tiny, one oversized — does elision of the big one leave a valid array?
		eligibleContent = []obj{
			{"type": "text", "text": "short preamble"},
			{"type": "text", "text": eligiblePayload},
		}
	default:
		t.Fatalf("unknown shape %q", shape)
	}

	msgs := []obj{
		{"role": "user", "content": []obj{{"type": "text", "text": "cached head context", "cache_control": cc}}}, // 0 breakpoint
		{"role": "assistant", "content": []obj{ // 1 — tool_use that msg[2] answers (pairing)
			{"type": "text", "text": "let me look"},
			{"type": "tool_use", "id": "call_t2", "name": "read", "input": obj{"path": "/x"}},
		}},
		{"role": "user", "content": []obj{ // 2 — ELIGIBLE tool_result (past prefix, before recent window)
			{"type": "tool_result", "tool_use_id": "call_t2", "content": eligibleContent},
		}},
		{"role": "assistant", "content": []obj{{"type": "text", "text": "a3"}}}, // 3
		{"role": "user", "content": []obj{{"type": "text", "text": "u4"}}},      // 4
		{"role": "assistant", "content": []obj{{"type": "text", "text": "a5"}}}, // 5
		{"role": "user", "content": []obj{{"type": "text", "text": "u6"}}},      // 6 — recent window starts (len-4)
		{"role": "assistant", "content": []obj{{"type": "text", "text": "a7"}}}, // 7
		{"role": "user", "content": []obj{{"type": "text", "text": "u8"}}},      // 8
		{"role": "assistant", "content": []obj{{"type": "text", "text": "a9"}}}, // 9
	}
	raw, err := json.Marshal(obj{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system":     []obj{{"type": "text", "text": "policy header", "cache_control": cc}},
		"messages":   msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestReproElisionSemanticValidity is the primary repro: sweep shapes x thresholds x payloads
// and, for every FIRED elision, run the strict Anthropic semantic validator on the output.
// A single semantic violation on a FIRED body reproduces the bug.
func TestReproElisionSemanticValidity(t *testing.T) {
	shapes := []string{"string", "array", "array-multi"}
	thresholds := []int{16, 64, 100, 256, 1024, 4096}

	// payload generators: (name, builder(threshold) => payload). The interesting ones are
	// near-boundary and multi-byte, where head/tail slicing on runes could misbehave.
	payloads := []struct {
		name  string
		build func(threshold int) string
	}{
		{"plain-4k", func(int) string { return strings.Repeat("A", 4000) }},
		{"threshold+1", func(th int) string { return strings.Repeat("A", th+1) }},
		{"threshold+2", func(th int) string { return strings.Repeat("A", th+2) }},
		{"emoji", func(th int) string { return strings.Repeat("😀", th+50) }},         // multi-byte runes, count > threshold
		{"mixed-emoji", func(th int) string { return strings.Repeat("a😀b", th+50) }}, // interleaved multi-byte
		{"newlines", func(th int) string { return strings.Repeat("line of output\n", th) }},
		{"whitespace-heavy", func(th int) string { return strings.Repeat("   \t  \n", th) }},
	}

	var reproduced, fired, notFired int
	notFiredReasons := map[string]int{}
	for _, shape := range shapes {
		for _, th := range thresholds {
			for _, p := range payloads {
				payload := p.build(th)
				raw := reproWireBody(t, shape, payload)
				out, oc := ElideAnthropicResultsWithOutcome(append([]byte(nil), raw...), th)
				if oc.Reason != ElideReasonNone {
					// Did not fire for this combination — not a repro candidate.
					notFired++
					notFiredReasons[fmt.Sprintf("%s/%s:%s", shape, p.name, oc.Reason)]++
					continue
				}
				fired++
				// It FIRED. fak already proved it re-decodes with its own decoder. Now check
				// the strict Anthropic semantics.
				errs := anthropicSemErrors(out)
				if len(errs) != 0 {
					reproduced++
					t.Errorf("REPRO shape=%s threshold=%d payload=%s: FIRED elision produced a semantically-invalid Anthropic body:\n  violations: %v\n  input len=%d output len=%d",
						shape, th, p.name, errs, len(raw), len(out))
					// Dump the offending eligible block so the exact malformed shape is on record.
					dumpEligibleBlock(t, out)
				}
			}
		}
	}
	t.Logf("SWEEP: %d combinations FIRED elision, %d did not fire.", fired, notFired)
	for k, n := range notFiredReasons {
		t.Logf("  not-fired: %s x%d", k, n)
	}
	if reproduced == 0 {
		t.Logf("EXONERATION: of the %d FIRED elisions, NONE produced a semantically-invalid Anthropic body.", fired)
	}
}

// dumpEligibleBlock prints msg[2]'s tool_result block from a body for evidence.
func dumpEligibleBlock(t *testing.T, raw []byte) {
	t.Helper()
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(raw, &body) != nil || len(body.Messages) < 3 {
		return
	}
	t.Logf("  msg[2] as shipped: %s", string(body.Messages[2]))
}

// TestReproElisionExonerationSummary is a non-failing companion that prints, for each shape,
// whether the FIRED output stayed semantically valid — so the report has a clean matrix even
// when nothing reproduces.
func TestReproElisionExonerationSummary(t *testing.T) {
	shapes := []string{"string", "array", "array-multi"}
	th := 100
	payload := strings.Repeat("A", 4000)
	for _, shape := range shapes {
		raw := reproWireBody(t, shape, payload)
		out, oc := ElideAnthropicResultsWithOutcome(append([]byte(nil), raw...), th)
		if oc.Reason != ElideReasonNone {
			t.Logf("shape=%s: did NOT fire (reason=%s)", shape, oc.Reason)
			continue
		}
		errs := anthropicSemErrors(out)
		if len(errs) == 0 {
			t.Logf("shape=%s: FIRED (elided=%d shed=%d) — output SEMANTICALLY VALID", shape, oc.Elided, oc.ShedBytes)
		} else {
			t.Logf("shape=%s: FIRED but INVALID: %v", shape, errs)
		}
	}
}
