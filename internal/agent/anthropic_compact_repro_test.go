package agent

// anthropic_compact_repro_test.go — REPRO HARNESS (investigation only, not product code).
//
// Companion to anthropic_elide_repro_test.go. Elision was exonerated for the empty-block and
// tool-pairing failure modes; this file applies the SAME strict-Anthropic-semantics validator to
// the history-compaction rewrite (CompactAnthropicHistoryWithOptions), which drops whole middle
// turns and could orphan a tool_use / tool_result pair.
//
// Two orphan directions exist, and the existing assertNoOrphanToolResult only checks ONE:
//
//	(A) tool_result kept, tool_use dropped  — a tool_result in the kept window whose matching
//	    tool_use was in the dropped middle. Guarded by chooseKeptWindow's snap + the
//	    messageHasToolResult re-assert (anthropic_compact.go:410).
//
//	(B) tool_use kept, tool_result dropped  — an assistant tool_use in the PROTECTED PREFIX whose
//	    matching tool_result is in the dropped middle. This leaves a tool_use with no answer. The
//	    kept-window guards never look at the prefix/drop boundary, so this direction is the
//	    interesting probe. (Anthropic requires every tool_use be followed by its tool_result.)
//
// The validator here (reused conceptually from the elision harness, extended to the compaction
// output shape) flags empty text/content and BOTH orphan directions.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// compactSemErrors runs the strict Anthropic pairing + non-empty checks over a compaction output.
// It returns every violation found (empty list == a body Anthropic would accept for these rules).
func compactSemErrors(raw []byte) []string {
	var errs []string
	var body struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return []string{"not JSON: " + err.Error()}
	}
	type ref struct{ mi int }
	toolUse := map[string]ref{}     // id -> where the assistant tool_use is
	toolResult := map[string]bool{} // ids answered by a tool_result
	prevRole := ""
	for mi, m := range body.Messages {
		if mi == 0 && m.Role != "user" {
			errs = append(errs, fmt.Sprintf("msg[0] role=%s: leading turn must be user", m.Role))
		}
		if m.Role == prevRole {
			errs = append(errs, fmt.Sprintf("msg[%d]: two consecutive %s turns (alternation broken)", mi, m.Role))
		}
		prevRole = m.Role
		if len(m.Content) == 0 || m.Content[0] != '[' {
			continue
		}
		var blocks []json.RawMessage
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		if len(blocks) == 0 {
			errs = append(errs, fmt.Sprintf("msg[%d]: EMPTY content array", mi))
			continue
		}
		for bi, blk := range blocks {
			var b struct {
				Type      string `json:"type"`
				Text      string `json:"text"`
				ID        string `json:"id"`
				ToolUseID string `json:"tool_use_id"`
			}
			if json.Unmarshal(blk, &b) != nil {
				continue
			}
			switch b.Type {
			case "text":
				if b.Text == "" {
					errs = append(errs, fmt.Sprintf("msg[%d].block[%d]: EMPTY text", mi, bi))
				}
			case "tool_use":
				if b.ID != "" {
					toolUse[b.ID] = ref{mi}
				}
			case "tool_result":
				if b.ToolUseID != "" {
					toolResult[b.ToolUseID] = true
					if _, ok := toolUse[b.ToolUseID]; !ok {
						// direction (A): tool_result whose tool_use was dropped/not-yet-seen
						errs = append(errs, fmt.Sprintf("ORPHAN-A msg[%d].block[%d]: tool_result id=%q has no earlier tool_use", mi, bi, b.ToolUseID))
					}
				}
			}
		}
	}
	// direction (B): tool_use that survived but whose tool_result was dropped.
	for id := range toolUse {
		if !toolResult[id] {
			errs = append(errs, fmt.Sprintf("ORPHAN-B tool_use id=%q survived but its tool_result was dropped", id))
		}
	}
	return errs
}

// compactPairBody builds a body where a tool_use / tool_result PAIR straddles the protected
// prefix and the compactible middle, so a middle-drop can strand the tool_use.
//
// Layout (n messages), pairStyle controls where the pair sits:
//   - "prefix-use/mid-result": assistant tool_use at msg[1] (inside the protected prefix, which
//     ends at the first breakpoint msg[0]... but a breakpoint at msg[2] extends the prefix); its
//     tool_result at a later message that falls in the compactible middle. This is direction (B).
//   - "mid-pair": both tool_use and tool_result live in the compactible middle (both dropped
//     together — the safe baseline).
//   - "mid-use/kept-result": tool_use in the middle, tool_result in the kept recent window
//     (direction A — the guarded case).
func compactPairBody(t *testing.T, n int, pairStyle string, bigPad string) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	txt := func(role, s string) obj { return obj{"role": role, "content": []obj{{"type": "text", "text": s}}} }
	assistantUse := func(id string) obj {
		return obj{"role": "assistant", "content": []obj{
			{"type": "text", "text": "calling a tool"},
			{"type": "tool_use", "id": id, "name": "read", "input": obj{"path": "/x"}},
		}}
	}
	userResult := func(id, text string) obj {
		return obj{"role": "user", "content": []obj{
			{"type": "tool_result", "tool_use_id": id, "content": []obj{{"type": "text", "text": text}}},
		}}
	}

	msgs := make([]obj, n)
	// msg[0] carries the head breakpoint (protected prefix ends here by default).
	msgs[0] = obj{"role": "user", "content": []obj{{"type": "text", "text": "the head task " + bigPad, "cache_control": cc}}}
	for i := 1; i < n; i++ {
		if i%2 == 1 {
			msgs[i] = txt("assistant", fmt.Sprintf("assistant note %d %s", i, bigPad))
		} else {
			msgs[i] = txt("user", fmt.Sprintf("user note %d %s", i, bigPad))
		}
	}

	switch pairStyle {
	case "prefix-use/mid-result":
		// Put a SECOND breakpoint at msg[2] so the protected prefix extends through msg[2],
		// and place the assistant tool_use at msg[1] (inside that prefix). Its tool_result lands
		// at msg[4] (compactible middle for a tight budget). msg[1] must be assistant (odd index ✓).
		msgs[1] = assistantUse("pairid")
		msgs[2] = obj{"role": "user", "content": []obj{{"type": "text", "text": "second breakpoint turn " + bigPad, "cache_control": cc}}}
		msgs[4] = userResult("pairid", "the tool output "+bigPad)
	case "mid-pair":
		msgs[3] = assistantUse("pairid") // assistant, middle
		msgs[4] = userResult("pairid", "the tool output "+bigPad)
	case "mid-use/kept-result":
		// tool_use in the middle, tool_result near the end (kept window). n-2 is a user index if even.
		msgs[3] = assistantUse("pairid")
		ri := n - 2
		if ri%2 == 1 {
			ri = n - 3
		}
		msgs[ri] = userResult("pairid", "the tool output "+bigPad)
	default:
		t.Fatalf("unknown pairStyle %q", pairStyle)
	}

	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"system":   []obj{{"type": "text", "text": "You are a coding agent.", "cache_control": cc}},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestReproCompactionOrphan sweeps compaction over both anchors, all pair styles, and a range of
// budgets/sizes, running the strict validator on every FIRE. Any orphan (A or B) or empty block on
// a fired body reproduces a compaction defect.
func TestReproCompactionOrphan(t *testing.T) {
	pad := strings.Repeat("x", 200) // inflate each turn so tight budgets force real drops
	styles := []string{"prefix-use/mid-result", "mid-pair", "mid-use/kept-result"}
	var fired, reproduced int
	for _, style := range styles {
		for _, n := range []int{10, 14, 20, 30} {
			for _, budget := range []int{40, 80, 150, 300, 600} {
				raw := compactPairBody(t, n, style, pad)
				// default (first-breakpoint) anchor
				runCompactCase(t, "firstbp", style, n, budget, raw,
					func() ([]byte, CompactOutcome) { return CompactAnthropicHistoryWithOutcome(raw, budget) },
					&fired, &reproduced)
				// head anchor with a cold cache (fires more aggressively — whole array compactible)
				runCompactCase(t, "head-cold", style, n, budget, raw,
					func() ([]byte, CompactOutcome) {
						return CompactAnthropicHistoryWithOptions(raw, CompactOptions{
							Budget: budget, Anchor: CompactAnchorHead, ColdCache: true,
							TotalTurns: 1000, CurrentTurn: 1,
						})
					},
					&fired, &reproduced)
			}
		}
	}
	t.Logf("SWEEP: %d compaction FIRES checked.", fired)
	if reproduced == 0 {
		t.Logf("EXONERATION: of the %d FIRED compactions, NONE produced an orphaned pair or empty block.", fired)
	}
}

func runCompactCase(t *testing.T, anchor, style string, n, budget int, raw []byte,
	run func() ([]byte, CompactOutcome), fired, reproduced *int) {
	t.Helper()
	out, oc := run()
	if oc.Reason != CompactReasonNone {
		return
	}
	*fired++
	// fak's own decoder accepts it (that's the whole point). Now the strict semantics:
	errs := compactSemErrors(out)
	if len(errs) != 0 {
		*reproduced++
		t.Errorf("REPRO anchor=%s style=%s n=%d budget=%d: FIRED compaction produced a semantically-invalid body:\n  %v\n  dropped=%d shed=%d",
			anchor, style, n, budget, errs, oc.Dropped, oc.ShedTokens)
		// One-time evidence dump: input roles/pair positions vs output roles.
		if *reproduced == 1 {
			dumpCompactCase(t, raw, out)
		}
	}
}

// dumpCompactCase prints, for the FIRST repro, the input and output messages[] role/type shape so
// the orphan is unambiguous evidence rather than an index-math artifact.
func dumpCompactCase(t *testing.T, in, out []byte) {
	t.Helper()
	shape := func(label string, raw []byte) {
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if json.Unmarshal(raw, &body) != nil {
			return
		}
		for i, m := range body.Messages {
			var mm struct {
				Role    string `json:"role"`
				Content []struct {
					Type      string `json:"type"`
					ID        string `json:"id"`
					ToolUseID string `json:"tool_use_id"`
				} `json:"content"`
			}
			_ = json.Unmarshal(m, &mm)
			var tags []string
			for _, c := range mm.Content {
				switch c.Type {
				case "tool_use":
					tags = append(tags, "tool_use:"+c.ID)
				case "tool_result":
					tags = append(tags, "tool_result->"+c.ToolUseID)
				default:
					tags = append(tags, c.Type)
				}
			}
			t.Logf("  %s[%d] %-9s %s", label, i, mm.Role, strings.Join(tags, ","))
		}
	}
	shape("IN ", in)
	shape("OUT", out)
	// Raw output messages so the stub content and the orphaned block are on record verbatim.
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(out, &body) == nil {
		for i, m := range body.Messages {
			t.Logf("  OUT-RAW[%d] %s", i, string(m))
		}
	}
}
