package gateway

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// feedGuard streams content through a liftGuard in chunkSize-byte fragments and
// returns the bytes the guard let through live. content must be ASCII when chunkSize
// can split it; a real upstream only ever feeds whole-rune delta.content fragments.
func feedGuard(t *testing.T, content string, chunkSize int) string {
	t.Helper()
	var live strings.Builder
	g := newLiftGuard(func(s string) error { live.WriteString(s); return nil })
	for i := 0; i < len(content); i += chunkSize {
		end := i + chunkSize
		if end > len(content) {
			end = len(content)
		}
		if err := g.write(content[i:end]); err != nil {
			t.Fatalf("guard.write: %v", err)
		}
	}
	if g.streamed() != live.String() {
		t.Fatalf("streamed() = %q but emitted %q", g.streamed(), live.String())
	}
	return live.String()
}

// TestLiftGuardStreamsPrefixOfLiftedContentAndNeverLeaksDialect is the load-bearing
// proof: for EVERY fragmentation of a turn's content, the bytes the guard streams live
// are a prefix of the content the buffered LiftTextToolCalls leaves behind (so the
// finalize remainder reconstructs it exactly), and no text-form tool-call dialect ever
// reaches the wire (so a buried — possibly denied — call's raw bytes never leak).
func TestLiftGuardStreamsPrefixOfLiftedContentAndNeverLeaksDialect(t *testing.T) {
	cases := []struct {
		name    string
		content string
	}{
		{"plain prose", "Let me check the weather for you right now."},
		{"prose ending in lt", "compare a < b and continue"},
		{"hermes call only", `<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>`},
		{"prose then hermes call", `Sure, checking now. <tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>`},
		{"function_call tag", `<function_call>{"name":"do_x","arguments":{}}</function_call>`},
		{"llama python tag", `<|python_tag|>{"name":"do_x","arguments":{"a":1}}<|eom_id|>`},
		{"mistral tool calls", `[TOOL_CALLS][{"name":"do_x","arguments":{}}]`},
		{"fenced json call", "```json\n{\"name\":\"do_x\",\"arguments\":{}}\n```"},
		{"fenced bare call no hint", "```\n{\"name\":\"do_x\",\"arguments\":{}}\n```"},
		{"bare json call", `{"name":"do_x","arguments":{"a":1}}`},
		{"bare json call leading ws", `   {"name":"do_x","arguments":{"a":1}}`},
		{"prose then fenced call", "Working on it.\n```json\n{\"name\":\"do_x\",\"arguments\":{}}\n```"},
		// Non-calls the guard must NOT suppress beyond a final tail (it streams them).
		{"code fence not a call", "Here is code:\n```python\nprint('hello world')\n```\nAll done."},
		{"prose inline backticks", "Use the `ls` command to list files in the directory tree."},
		{"json answer not a call", `{"answer":42,"note":"the computed result here"}`},
		{"js fence not a call", "```js\nconst x = {a: 1}\n```"},
	}
	for _, tc := range cases {
		lifted := agent.LiftTextToolCalls(agent.Message{Content: tc.content})
		cleaned := lifted.Content
		for _, chunk := range []int{1, 2, 3, 4, 7, 13, 1000} {
			t.Run(fmt.Sprintf("%s/chunk=%d", tc.name, chunk), func(t *testing.T) {
				streamed := feedGuard(t, tc.content, chunk)
				full := streamed + liftRemainder(streamed, cleaned)

				// Parity: the client ultimately receives exactly the buffered post-lift
				// content — no prose dropped or invented. Edge whitespace aside: lift
				// TrimSpace's the whole turn, so a trailing space before a stripped call
				// may stream live where the buffered path drops it (cosmetic, not a leak).
				if strings.TrimSpace(full) != strings.TrimSpace(cleaned) {
					t.Fatalf("client content = %q, want %q (streamed=%q)", full, cleaned, streamed)
				}

				// Leak: every non-whitespace byte streamed live is a prefix of the
				// post-lift content, so no span lift STRIPS — a buried tool call — ever
				// reached the wire.
				if ts := strings.TrimSpace(streamed); ts != "" && !strings.HasPrefix(cleaned, ts) {
					t.Fatalf("live bytes %q are not a prefix of lifted content %q — a stripped span leaked", streamed, cleaned)
				}

				// When a call WAS buried, the live bytes must carry neither a dialect
				// marker nor the call's JSON payload keys.
				if len(lifted.ToolCalls) > 0 {
					for _, marker := range append(append([]string{}, delimitedToolCallTags...), "```", `"arguments"`, `"name"`) {
						if strings.Contains(streamed, marker) {
							t.Fatalf("buried-call marker %q leaked into live stream: %q", marker, streamed)
						}
					}
				}
			})
		}
	}
}

// TestLiftGuardStreamsCodeBlockLiveNotJustAtFinalize proves the fence peek-release:
// an ordinary code block (no JSON tool-call body) is NOT held — its bytes stream live
// — so enabling the guard does not regress time-to-first-token for the common
// "write me code" turn.
func TestLiftGuardStreamsCodeBlockLiveNotJustAtFinalize(t *testing.T) {
	content := "Here is the code:\n```python\nfor i in range(10):\n    print(i)\n```\n"
	streamed := feedGuard(t, content, 4)
	// The fence opener and the code body must have streamed live (a code fence is not a
	// tool call). Only a short trailing tail may be withheld for finalize.
	if !strings.Contains(streamed, "```python") {
		t.Fatalf("code fence opener was held, not streamed live: %q", streamed)
	}
	if !strings.Contains(streamed, "print(i)") {
		t.Fatalf("code body was held, not streamed live: %q", streamed)
	}
	if len(streamed) < len(content)-len("```")-1 {
		t.Fatalf("withheld more than a trailing partial-fence tail: streamed %d of %d bytes", len(streamed), len(content))
	}
}

// TestLiftGuardHoldsJSONFenceUntilFinalize proves the other half of the peek: a fenced
// block whose body IS a JSON object (a tool-call dialect) is held entirely, so its raw
// text never streams live.
func TestLiftGuardHoldsJSONFenceUntilFinalize(t *testing.T) {
	content := "```json\n{\"name\":\"do_x\",\"arguments\":{\"a\":1}}\n```"
	streamed := feedGuard(t, content, 3)
	if streamed != "" {
		t.Fatalf("JSON-bodied fence leaked into the live stream: %q", streamed)
	}
}

// TestLiftGuardWithholdsSplitDialectOpener proves a dialect opener split across two
// fragments is never partially streamed: the half-opener is withheld until the next
// fragment completes (and holds) it.
func TestLiftGuardWithholdsSplitDialectOpener(t *testing.T) {
	var live strings.Builder
	g := newLiftGuard(func(s string) error { live.WriteString(s); return nil })
	// Fragment 1 ends mid-opener ("<tool_c"); nothing of the opener may stream.
	if err := g.write("ok <tool_c"); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(live.String(), "<tool_c") {
		t.Fatalf("partial opener streamed before completion: %q", live.String())
	}
	if live.String() != "ok " {
		t.Fatalf("prose before the split opener = %q, want %q", live.String(), "ok ")
	}
	// Fragment 2 completes the opener; the whole call is now held.
	if err := g.write(`all>{"name":"x","arguments":{}}</tool_call>`); err != nil {
		t.Fatal(err)
	}
	if live.String() != "ok " {
		t.Fatalf("dialect leaked after completion: %q", live.String())
	}
}

func TestFenceDecisionClassifiesBodies(t *testing.T) {
	cases := []struct {
		in   string
		want fenceVerdict
	}{
		{"```json\n{\"a\":1}", fenceHold},
		{"```\n{\"a\":1}", fenceHold},
		{"```[1,2,3]", fenceHold},
		{"```python\nprint(1)", fenceRelease},
		{"```js\nconst x = 1", fenceRelease},
		{"```jsonx\nx", fenceRelease},
		{"```", fencePending},
		{"```j", fencePending},
		{"```js", fencePending},
		{"```jso", fencePending},
		{"```json", fencePending},
		{"```json\n", fencePending},
	}
	for _, c := range cases {
		if got := fenceDecision(c.in); got != c.want {
			t.Errorf("fenceDecision(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestLiftRemainderReconciliation(t *testing.T) {
	cases := []struct {
		name              string
		streamed, cleaned string
		want              string
	}{
		{"streamed is full prefix", "hello wor", "hello world", "ld"},
		{"streamed all of cleaned", "hello world", "hello world", ""},
		{"nothing streamed", "", "hello world", "hello world"},
		{"leading whitespace trimmed by lift", "  hello", "hello world", " world"},
		{"divergence emits past common prefix", "abXYZ", "abcdef", "cdef"},
	}
	for _, c := range cases {
		if got := liftRemainder(c.streamed, c.cleaned); got != c.want {
			t.Errorf("%s: liftRemainder(%q,%q) = %q, want %q", c.name, c.streamed, c.cleaned, got, c.want)
		}
	}
}
