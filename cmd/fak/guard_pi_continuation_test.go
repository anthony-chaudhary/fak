package main

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

// The generated extension is plain JavaScript despite its .ts filename. Exercise
// its real event handler with Node so the continuation contract is checked at the
// Pi boundary, rather than copying its decision into a Go test helper.
func TestGuardPiContinuationTurnEnd(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("Node is required to execute Pi's generated extension")
	}

	for _, tc := range []struct {
		name   string
		source string
	}{
		{name: "anthropic", source: guardPiExtensionSource("http://127.0.0.1:4567")},
		{name: "fak", source: func() string {
			s, err := guardPiFakExtensionSource("http://127.0.0.1:4567/v1", "fixture")
			if err != nil {
				t.Fatal(err)
			}
			return s
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A refusal note is the actual in-band Anthropic response shape from
			// internal/gateway/messages.go, with no surviving tool call/result.
			const script = `
const vm = require("node:vm");
const source = process.argv[1];
const handlers = {};
const pi = {
  registerProvider() {},
  on(name, handler) { handlers[name] = handler; },
};
const sandbox = { module: { exports: {} } };
vm.runInNewContext(source.replace("export default function", "module.exports = function"), sandbox);
sandbox.module.exports(pi);
const turnEnd = handlers.turn_end;
if (typeof turnEnd !== "function") throw Error("generated extension has no turn_end handler");
const refusal = (tool) => ({
  type: "turn_end",
  message: { role: "assistant", content: [{ type: "text", text: "[fak] Allowed next step for 1 refused tool call(s): Choose an allowed alternative. Constraint: " + tool + " (DEFAULT_DENY/TERMINAL)." }] },
  toolResults: [],
  continue: false,
  entries: [],
  outcome: "completed",
  context: { canContinue: false, pendingMessages: [] },
});
const clean = {
  ...refusal("Write"),
  message: { role: "assistant", content: [{ type: "text", text: "Task completed." }] },
};
const decisions = [await turnEnd(clean, {})];
for (let i = 0; i < 6; i++) decisions.push(await turnEnd(refusal("Write"), {}));
const withToolCall = refusal("Write");
withToolCall.message.content.push({ type: "toolCall", id: "call-1", name: "Read", arguments: {} });
decisions.push(await turnEnd(withToolCall, {}));
const withToolResult = refusal("Write");
withToolResult.toolResults.push({ role: "toolResult", toolCallId: "call-1", content: [{ type: "text", text: "ok" }] });
decisions.push(await turnEnd(withToolResult, {}));
for (let i = 0; i < 25; i++) decisions.push(await turnEnd(refusal("DistinctTool" + i), {}));
// OpenAI Completions uses denySummary, whose prefix differs from the Anthropic
// adjudicationNote. Reset the bound before each six-turn shape check.
decisions.push(await turnEnd(clean, {}));
const openAIRefusal = () => {
  const event = refusal("Write");
  event.message.content[0].text = "Allowed next step for each refused tool call: Use a permitted tool. Constraint: Write: DENY (DEFAULT_DENY/TERMINAL)";
  return event;
};
for (let i = 0; i < 6; i++) decisions.push(await turnEnd(openAIRefusal(), {}));
decisions.push(await turnEnd(clean, {}));
// Streaming Anthropic can emit model prose first and the gateway note as a
// later text block when it flushes withheld tool_use blocks.
const laterNote = () => {
  const event = refusal("Write");
  event.message.content.unshift({ type: "text", text: "I will update the file now." });
  return event;
};
for (let i = 0; i < 6; i++) decisions.push(await turnEnd(laterNote(), {}));
decisions.push(await turnEnd(clean, {}));
const openAIProseAndNote = () => {
  const event = openAIRefusal();
  event.message.content[0].text = "I will update the file now.\n" + event.message.content[0].text;
  return event;
};
for (let i = 0; i < 6; i++) decisions.push(await turnEnd(openAIProseAndNote(), {}));
decisions.push(await turnEnd(clean, {}));
const retryableRefusal = () => {
  const event = refusal("Write");
  event.message.content[0].text = "[fak] Allowed next step for 1 refused tool call(s): Correct the call shape. Constraint: Write (MALFORMED/RETRYABLE).";
  return event;
};
for (let i = 0; i < 25; i++) decisions.push(await turnEnd(retryableRefusal(), {}));
process.stdout.write(JSON.stringify(decisions));
`
			cmd := exec.Command("node", "-e", "(async () => {"+script+"})().catch(e => { console.error(e); process.exitCode = 1 })", tc.source)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("execute generated Pi extension: %v\n%s", err, out)
			}
			var decisions []struct {
				Continue bool `json:"continue"`
				Entries  []struct {
					Type       string `json:"type"`
					CustomType string `json:"customType"`
					Content    string `json:"content"`
					Display    bool   `json:"display"`
				} `json:"entries"`
			}
			if err := json.Unmarshal(out, &decisions); err != nil {
				t.Fatalf("decode Pi decisions: %v: %s", err, out)
			}
			if len(decisions) != 81 {
				t.Fatalf("got %d turn decisions, want 81", len(decisions))
			}
			for _, i := range []int{1, 2, 3, 4, 5} {
				d := decisions[i]
				if !d.Continue || len(d.Entries) != 1 || d.Entries[0].Type != "custom_message" || d.Entries[0].CustomType != "fak_guard_tool_feedback" || d.Entries[0].Display || !strings.Contains(d.Entries[0].Content, "allowed alternative") {
					t.Errorf("refusal %d did not queue exactly one model continuation: %+v", i, d)
				}
			}
			if decisions[0].Continue || len(decisions[0].Entries) != 0 {
				t.Errorf("ordinary completion must settle: %+v", decisions[0])
			}
			if decisions[6].Continue || len(decisions[6].Entries) != 0 {
				t.Errorf("sixth identical refusal must settle at bound: %+v", decisions[6])
			}
			for _, i := range []int{7, 8} {
				if decisions[i].Continue || len(decisions[i].Entries) != 0 {
					t.Errorf("refusal with surviving tool call/result must settle (case %d): %+v", i, decisions[i])
				}
			}
			for i := 9; i < 33; i++ {
				if !decisions[i].Continue || len(decisions[i].Entries) != 1 {
					t.Errorf("distinct refusal %d must continue before total bound: %+v", i-8, decisions[i])
				}
			}
			if decisions[33].Continue || len(decisions[33].Entries) != 0 {
				t.Errorf("25th distinct refusal must settle at total bound: %+v", decisions[33])
			}
			for _, shape := range []struct {
				name  string
				reset int
				first int
				bound int
			}{
				{name: "OpenAI denySummary", reset: 34, first: 35, bound: 40},
				{name: "Anthropic streamed later note", reset: 41, first: 42, bound: 47},
				{name: "OpenAI prose plus denial note", reset: 48, first: 49, bound: 54},
			} {
				if decisions[shape.reset].Continue || len(decisions[shape.reset].Entries) != 0 {
					t.Errorf("%s reset completion must settle: %+v", shape.name, decisions[shape.reset])
				}
				for i := shape.first; i < shape.bound; i++ {
					if !decisions[i].Continue || len(decisions[i].Entries) != 1 {
						t.Errorf("%s refusal %d must continue: %+v", shape.name, i-shape.first+1, decisions[i])
					}
				}
				if decisions[shape.bound].Continue || len(decisions[shape.bound].Entries) != 0 {
					t.Errorf("%s sixth identical refusal must settle: %+v", shape.name, decisions[shape.bound])
				}
			}
			if decisions[55].Continue || len(decisions[55].Entries) != 0 {
				t.Errorf("retryable reset completion must settle: %+v", decisions[55])
			}
			for i := 56; i < 80; i++ {
				if !decisions[i].Continue || len(decisions[i].Entries) != 1 {
					t.Errorf("retryable refusal %d must continue through sixth identical turn until total bound: %+v", i-55, decisions[i])
				}
			}
			if decisions[80].Continue || len(decisions[80].Entries) != 0 {
				t.Errorf("25th identical retryable refusal must settle at total bound: %+v", decisions[80])
			}
		})
	}
}
