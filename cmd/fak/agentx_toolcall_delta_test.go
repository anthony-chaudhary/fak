package main

import (
	"encoding/json"
	"testing"
)

// TestToolCallDeltaMerge verifies the streaming tool-call delta merge ported
// from ai-dynamo/aiperf openai_chat.py (ChatEndpoint._merge_tool_call_delta,
// Apache-2.0, pin ab610ad): argument deltas accumulate across stream chunks
// into a complete tool call, with index fallback, name replace, and argument
// append semantics.
func TestToolCallDeltaMerge(t *testing.T) {
	// mergeJSON unmarshals a raw JSON tool_calls delta array and folds each
	// delta into the accumulator, mirroring how stream chunks arrive on the
	// wire.
	mergeJSON := func(acc *toolCallDeltaAccumulator, raw string) {
		t.Helper()
		var deltas []toolCallDelta
		if err := json.Unmarshal([]byte(raw), &deltas); err != nil {
			t.Fatalf("unmarshal %q: %v", raw, err)
		}
		for _, d := range deltas {
			acc.mergeToolCallDelta(d)
		}
	}

	t.Run("parallel tool calls with explicit index", func(t *testing.T) {
		acc := newToolCallDeltaAccumulator()
		// Chunk 1: two parallel tool calls; id/type/name set, arguments empty.
		mergeJSON(acc, `[
			{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}},
			{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}
		]`)
		// Chunk 2: argument fragments appended.
		mergeJSON(acc, `[
			{"index":0,"function":{"arguments":"city="}},
			{"index":1,"function":{"arguments":"zone="}}
		]`)
		// Chunk 3: final argument fragments appended.
		mergeJSON(acc, `[
			{"index":0,"function":{"arguments":"Paris"}},
			{"index":1,"function":{"arguments":"UTC"}}
		]`)

		got := acc.merged()
		if len(got) != 2 {
			t.Fatalf("merged() = %d slots, want 2", len(got))
		}
		// Slot 0.
		if got[0].id != "call_1" || got[0].typ != "function" {
			t.Errorf("slot 0 id/type = %q/%q, want call_1/function", got[0].id, got[0].typ)
		}
		if got[0].function == nil {
			t.Fatalf("slot 0 function is nil")
		}
		if got[0].function.name != "get_weather" {
			t.Errorf("slot 0 name = %q, want get_weather", got[0].function.name)
		}
		if want := "city=Paris"; got[0].function.arguments != want {
			t.Errorf("slot 0 arguments = %q, want %q", got[0].function.arguments, want)
		}
		// Slot 1.
		if got[1].id != "call_2" || got[1].typ != "function" {
			t.Errorf("slot 1 id/type = %q/%q, want call_2/function", got[1].id, got[1].typ)
		}
		if got[1].function == nil {
			t.Fatalf("slot 1 function is nil")
		}
		if got[1].function.name != "get_time" {
			t.Errorf("slot 1 name = %q, want get_time", got[1].function.name)
		}
		if want := "zone=UTC"; got[1].function.arguments != want {
			t.Errorf("slot 1 arguments = %q, want %q", got[1].function.arguments, want)
		}
	})

	t.Run("index fallback to slot count", func(t *testing.T) {
		acc := newToolCallDeltaAccumulator()
		// No index in any delta: each lands in a fresh slot (0, then 1), so
		// parallel calls do not collapse into slot 0 (the source's reason for
		// the len(slots) fallback rather than a 0 default).
		mergeJSON(acc, `[{"id":"call_a","type":"function","function":{"name":"f1","arguments":"A"}}]`)
		mergeJSON(acc, `[{"id":"call_b","type":"function","function":{"name":"f2","arguments":"B"}}]`)

		got := acc.merged()
		if len(got) != 2 {
			t.Fatalf("merged() = %d slots, want 2", len(got))
		}
		if got[0].id != "call_a" || got[0].function.name != "f1" || got[0].function.arguments != "A" {
			t.Errorf("slot 0 = id %q name %q args %q, want call_a/f1/A",
				got[0].id, got[0].function.name, got[0].function.arguments)
		}
		if got[1].id != "call_b" || got[1].function.name != "f2" || got[1].function.arguments != "B" {
			t.Errorf("slot 1 = id %q name %q args %q, want call_b/f2/B",
				got[1].id, got[1].function.name, got[1].function.arguments)
		}
	})

	t.Run("name replaced when present", func(t *testing.T) {
		acc := newToolCallDeltaAccumulator()
		// Name appears in two chunks; the later one REPLACES the earlier
		// (unlike arguments, which append).
		mergeJSON(acc, `[{"index":0,"function":{"name":"old_name","arguments":""}}]`)
		mergeJSON(acc, `[{"index":0,"function":{"name":"new_name","arguments":"x"}}]`)

		got := acc.merged()
		if len(got) != 1 {
			t.Fatalf("merged() = %d slots, want 1", len(got))
		}
		if got[0].function.name != "new_name" {
			t.Errorf("name = %q, want new_name (replaced, not appended)", got[0].function.name)
		}
		if got[0].function.arguments != "x" {
			t.Errorf("arguments = %q, want x", got[0].function.arguments)
		}
	})

	t.Run("legacy function_call delta appends name", func(t *testing.T) {
		acc := newToolCallDeltaAccumulator()
		// Legacy streaming function_call deltas carry no index and land in
		// slot 0; the name is APPENDED across chunks (unlike the modern path,
		// which replaces).
		fn1 := toolCallDeltaFunction{Name: "get_", Arguments: strPtrTC("")}
		fn2 := toolCallDeltaFunction{Name: "weather", Arguments: strPtrTC("city=Paris")}
		acc.mergeLegacyFunctionCallDelta(&fn1)
		acc.mergeLegacyFunctionCallDelta(&fn2)

		got := acc.merged()
		if len(got) != 1 {
			t.Fatalf("merged() = %d slots, want 1", len(got))
		}
		if got[0].typ != "function" {
			t.Errorf("type = %q, want function", got[0].typ)
		}
		if got[0].function.name != "get_weather" {
			t.Errorf("name = %q, want get_weather (appended)", got[0].function.name)
		}
		if want := "city=Paris"; got[0].function.arguments != want {
			t.Errorf("arguments = %q, want %q", got[0].function.arguments, want)
		}
	})
}

// strPtrTC returns a pointer to s, for constructing toolCallDeltaFunction
// arguments in tests. Distinct from the package's existing strPtr helper.
func strPtrTC(s string) *string { return &s }
