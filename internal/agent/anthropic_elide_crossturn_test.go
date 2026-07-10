package agent

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// fileBody is a realistic multi-line tool-output payload a coding agent re-displays across turns
// (cat → sed → git diff). 40 lines, ~1.2 KB — comfortably over both cross-turn floors.
func fileBody() string {
	var b strings.Builder
	for i := 0; i < 40; i++ {
		fmt.Fprintf(&b, "func handler%02d(w http.ResponseWriter, r *http.Request) { render(w, tmpl%02d) }\n", i, i)
	}
	return b.String()
}

// TestCrossTurnDedup is the #3340 witness. It proves, at the pure-core level, the prefix-
// monotonicity invariant dedup(blocks[:k]) == dedup(full)[:k] for every k, and at the request
// level that a file body repeated across two later tool_results folds to pointers, the earliest
// occurrence stays verbatim, and the protected cache prefix's sha256 is unchanged.
func TestCrossTurnDedup(t *testing.T) {
	t.Run("PrefixMonotonic", func(t *testing.T) {
		body := splitLinesKeepNL(fileBody())
		// Ordered blocks: earliest carries the body; two later blocks repeat it; interleaved unique
		// blocks ensure folding is driven by content, not position.
		blocks := [][]string{
			body,                                  // 0 — earliest occurrence (never folded)
			splitLinesKeepNL("unique tail one\n"), // 1
			body,                                  // 2 — repeat → folds to a pointer at turn 0
			splitLinesKeepNL("unique tail two\n"), // 3
			body,                                  // 4 — repeat → folds to a pointer at turn 0
		}
		turns := []int{0, 1, 2, 3, 4}

		full, changed := dedupBlockLines(blocks, turns)

		// Done-condition (core): earliest verbatim, both repeats folded to a turn-0 pointer.
		if changed[0] {
			t.Error("earliest block was folded — earliest occurrence must stay verbatim")
		}
		if full[0] != fileBody() {
			t.Error("earliest block bytes changed")
		}
		for _, k := range []int{2, 4} {
			if !changed[k] {
				t.Fatalf("repeat block %d was not folded", k)
			}
			if !strings.Contains(full[k], "identical to output shown earlier, turn 0, lines 1-40") {
				t.Errorf("block %d pointer missing/wrong: %q", k, full[k])
			}
			if strings.Contains(full[k], "handler39") {
				t.Errorf("block %d still carries the verbatim body after folding", k)
			}
		}

		// Prefix-monotonicity: truncating the block list at any k reproduces the first k outputs
		// byte-for-byte — appending a turn never mutates an earlier turn's folded bytes.
		for k := 1; k <= len(blocks); k++ {
			pfx, _ := dedupBlockLines(blocks[:k], turns[:k])
			for i := 0; i < k; i++ {
				if pfx[i] != full[i] {
					t.Fatalf("prefix-monotonicity broken at k=%d, block %d:\n truncated=%q\n full=%q", k, i, pfx[i], full[i])
				}
			}
		}
	})

	t.Run("FloorRespected", func(t *testing.T) {
		// A 4-line shared run is under crossTurnMinDupLines (8): it must NOT fold.
		short := splitLinesKeepNL("aa\nbb\ncc\ndd\n")
		blocks := [][]string{short, short}
		out, changed := dedupBlockLines(blocks, []int{0, 1})
		if changed[1] || out[1] != "aa\nbb\ncc\ndd\n" {
			t.Errorf("sub-floor run was folded: changed=%v out=%q", changed[1], out[1])
		}
	})

	t.Run("RequestFoldsAndKeepsPrefix", func(t *testing.T) {
		raw := crossTurnWireBody(t)
		orig := append([]byte(nil), raw...)

		// pfxEnd is msg 0 (the cache_control breakpoint); the first edit lands in msg 2, so the whole
		// prefix through msg 1 must be byte-identical. Hash it before and after.
		var o map[string]json.RawMessage
		if err := json.Unmarshal(orig, &o); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		_, spans, ok := decodeArrayElements(orig, o["messages"])
		if !ok {
			t.Fatal("decodeArrayElements failed")
		}
		prefixEnd := spans[2].start // start of the first eligible (foldable) message
		beforeHash := sha256.Sum256(orig[:prefixEnd])

		// threshold huge → head+tail elision cannot fire; any shrink is cross-turn dedup alone.
		out, outcome := ElideAnthropicResultsWithOutcome(raw, 1<<20)

		if outcome.Reason != ElideReasonNone {
			t.Fatalf("expected FIRED, got reason %q", outcome.Reason)
		}
		if outcome.Elided < 2 {
			t.Fatalf("expected >=2 folded results, got %d", outcome.Elided)
		}
		if outcome.ShedBytes <= 0 || len(out) >= len(orig) {
			t.Fatalf("expected a net-shorter body: shed=%d len %d->%d", outcome.ShedBytes, len(orig), len(out))
		}
		// Earliest occurrence (msg 0, the protected body) stays verbatim; the two later repeats fold.
		if !bytes.Contains(out, []byte("handler39")) {
			t.Error("earliest verbatim body missing from output")
		}
		if n := bytes.Count(out, []byte("handler39(w http.ResponseWriter")); n != 1 {
			t.Errorf("expected the body to survive exactly once (earliest), found %d copies", n)
		}
		if !bytes.Contains(out, []byte("identical to output shown earlier, turn 0")) {
			t.Error("cross-turn pointer missing from output")
		}
		// Cache prefix sha256 unchanged.
		if prefixEnd > len(out) || sha256.Sum256(out[:prefixEnd]) != beforeHash {
			t.Error("protected cache prefix sha256 changed — cache hit would be lost")
		}
		// Still a valid Anthropic request, and the input slice was not mutated.
		if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
			t.Errorf("rewritten body failed to re-decode: %v", err)
		}
		if !bytes.Equal(raw, orig) {
			t.Error("input raw was mutated in place")
		}
	})
}

// crossTurnWireBody builds a /v1/messages body where the same file body appears at msg 0 (the
// protected cache_control HEAD — the earliest occurrence), then again at msgs 2 and 4 (eligible,
// un-cached, before the recent window). len=9 so lastEligible = 9-elideRecentKeepMsgs = 5 and both
// repeats sit inside the eligible band (0,5).
func crossTurnWireBody(t *testing.T) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	body := fileBody()
	toolResult := func(id, text string, cached bool) obj {
		blk := obj{"type": "tool_result", "tool_use_id": id, "content": []obj{{"type": "text", "text": text}}}
		if cached {
			blk["cache_control"] = cc
		}
		return obj{"role": "user", "content": []obj{blk}}
	}
	text := func(role, s string) obj {
		return obj{"role": role, "content": []obj{{"type": "text", "text": s}}}
	}
	msgs := []obj{
		toolResult("t0", body, true),  // 0 — breakpoint + earliest body (protected, verbatim source)
		text("assistant", "a1"),       // 1
		toolResult("t2", body, false), // 2 — repeat → folds
		text("assistant", "a3"),       // 3
		toolResult("t4", body, false), // 4 — repeat → folds
		text("assistant", "a5"),       // 5 — recent window (len-4 .. len) below
		text("user", "u6"),            // 6
		text("assistant", "a7"),       // 7
		text("user", "u8"),            // 8
	}
	raw, err := json.Marshal(obj{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages":   msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}
