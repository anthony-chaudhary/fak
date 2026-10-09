package agent

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

// fak-test:runtime fast est=10ms lane=default
func TestElideOmittedFragmentLines(t *testing.T) {
	t.Parallel()
	t.Run("splitlines of removed fragment", func(t *testing.T) {
		for _, tc := range []struct {
			text string
			want int
		}{
			{"", 0}, {"partial", 1}, {"\n", 1}, {"a\n", 1}, {"a\nb", 2},
			{"\n\n", 2}, {"\r\n", 1}, {"a\r\nb", 2}, {"\r\n\r\n", 2},
			{"\r", 1}, {"a\r", 1}, {"\v\f\x1c\x1d\x1e\u0085\u2028\u2029", 8},
			{"a\vB\fC\x1cD\x1dE\x1eF\u0085G\u2028H\u2029尾", 9},
		} {
			if got := elidedFragmentLines([]rune(tc.text)); got != tc.want {
				t.Errorf("fragment %q: got %d lines, want %d", tc.text, got, tc.want)
			}
		}
	})
	t.Run("exact cut and original restore target", func(t *testing.T) {
		const head, tail = "HEADER", "尾巴"
		const restoreText = "original\nresult\nbefore\nfolding\n"
		const trace = "trace\"with quote"
		id := originatingTaskDigestID([]byte(restoreText))
		for _, tc := range []struct {
			middle string
			lines  int
		}{
			{"partial", 1}, {"\r\n", 1}, {"\npartial\r", 2}, {"α\u2028β\n", 2},
		} {
			got := elideHeadTailWithRestore(head+tc.middle+tail, 8, restoreText, trace)
			if !strings.HasPrefix(got, head) || !strings.HasSuffix(got, tail) || !utf8.ValidString(got) {
				t.Fatalf("head/tail or UTF-8 changed: %q", got)
			}
			if !strings.Contains(got, fmt.Sprintf("; %d fragment lines;", tc.lines)) ||
				!strings.Contains(got, compactRestoreIDField+id) ||
				!strings.Contains(got, "trace_id="+strconv.Quote(trace)) {
				t.Errorf("fragment count or original restore handle missing: %q", got)
			}
			legacy := head + elideMarkerf(len([]rune(tc.middle)), id, trace) + tail
			if len(got) > len(legacy) {
				t.Errorf("marker grew: %d bytes, legacy %d", len(got), len(legacy))
			}
		}
		// Cuts inside CRLF count only the removed fragment, not the kept half.
		if got := elideHeadTail("HEADR\r\nbody尾巴", 8); !strings.Contains(got, "; 2 fragment lines;") {
			t.Errorf("leading LF fragment was not counted independently: %q", got)
		}
	})

	const threshold = 128
	big := strings.Repeat("A\n", 2000)
	recent := strings.Repeat("C", 4000)
	id := originatingTaskDigestID([]byte(big))
	const marker = "; 1936 fragment lines;"
	checkRestore := func(t *testing.T, oc ElideOutcome) {
		t.Helper()
		if oc.Reason != ElideReasonNone || oc.Elided != 1 || oc.ShedBytes <= 0 {
			t.Fatalf("expected one shrinking elision, got %+v", oc)
		}
		if len(oc.Restores) != 1 || oc.Restores[0].ID != id || string(oc.Restores[0].Bytes) != big {
			t.Fatalf("original restore target changed: %+v", oc.Restores)
		}
	}
	t.Run("Anthropic byte splice", func(t *testing.T) {
		cached := strings.Repeat("B", 4000)
		raw := elideWireBody(t, big, cached, recent)
		original := bytes.Clone(raw)
		prefixEnd := protectedPrefixEnd(t, raw)
		out, oc := ElideAnthropicResultsWithOutcome(raw, threshold)
		checkRestore(t, oc)
		if !bytes.Contains(out, []byte(marker)) || !bytes.Contains(out, []byte(compactRestoreIDField+id)) {
			t.Error("exact removed-fragment count or restore ID missing from live byte route")
		}
		if len(out) >= len(raw) || len(out) < prefixEnd || !bytes.Equal(out[:prefixEnd], raw[:prefixEnd]) || !bytes.Equal(raw, original) {
			t.Error("byte budget, protected prefix, or input immutability changed")
		}
		if !bytes.Contains(out, []byte(cached)) || !bytes.Contains(out, []byte(recent)) {
			t.Error("cached or recent tool result changed")
		}
		if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
			t.Fatalf("rewritten wire no longer decodes: %v", err)
		}
	})
	t.Run("decoded tool messages", func(t *testing.T) {
		in := []Message{
			{Role: "system", Content: big}, {Role: "user", Content: "request"},
			{Role: "tool", ToolCallID: "t2", Content: big}, {Role: "assistant", Content: "analyzing"},
			{Role: "user", Content: "next"}, {Role: "assistant", Content: "reading"},
			{Role: "tool", ToolCallID: "t6", Content: recent}, {Role: "assistant", Content: "done"},
		}
		out, oc := ElideMessages(in, threshold)
		checkRestore(t, oc)
		if !strings.Contains(out[2].Content, marker) || !strings.Contains(out[2].Content, compactRestoreIDField+id) ||
			len(out[2].Content) >= len(big) || out[2].Role != "tool" || out[2].ToolCallID != "t2" {
			t.Error("decoded route lost count, budget, role, or restore identity")
		}
		for _, i := range []int{0, 1, 3, 4, 5, 6, 7} {
			if out[i].Content != in[i].Content {
				t.Errorf("protected/non-tool message %d changed", i)
			}
		}
		if in[2].Content != big {
			t.Error("decoded input mutated")
		}
	})
}
