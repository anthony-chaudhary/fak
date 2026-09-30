package trajectory

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestNativeReceiptAdapter(t *testing.T) {
	const source = "fak-agent-native-receipt-json"
	versioned := []byte(`{
		"schema":"fak.agent.native.v1",
		"task":"repair the adapter",
		"model":"fixture",
		"status":"completed",
		"metrics":{"arm":"fak","turns":2},
		"calls":{"schema":"fak.agent.native.calls.v1","entries":[
			{"arm":"fak","turn":1,"tool":"read_file","verdict":"ALLOW","by":"policy-floor","args":"{\"path\":\"README.md\"}"},
			{"arm":"fak","turn":2,"tool":"write_file","verdict":"DENY","reason":"POLICY_BLOCK","by":"workspace-floor","disposition":"TERMINAL","args":"{\"path\":\"outside.txt\"}"}
		]}
	}`)

	t.Run("versioned ordered calls preserve deny evidence without invented fidelity", func(t *testing.T) {
		registry := DefaultAdapterRegistry()
		first, firstReceipt, err := registry.Ingest(source, versioned)
		if err != nil {
			t.Fatal(err)
		}
		second, secondReceipt, err := registry.Ingest(source, versioned)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstReceipt, secondReceipt) {
			t.Fatal("native receipt reingest is not deterministic")
		}
		if len(first) != 2 || firstReceipt.InputRecords != 1 || firstReceipt.EmittedEvents != 2 {
			t.Fatalf("events=%d receipt=%+v", len(first), firstReceipt)
		}
		if firstReceipt.SourceType != source || firstReceipt.SourceDigest != digestBytes(versioned) {
			t.Fatalf("source identity lost: %+v", firstReceipt)
		}
		for i, event := range first {
			if event.Kind != EventTool || event.Sequence != uint64(i+1) || event.Source.OrderingKey != string(rune('1'+i)) {
				t.Fatalf("event[%d] order/kind=%+v", i, event)
			}
			var call struct {
				Turn        int    `json:"turn"`
				Tool        string `json:"tool"`
				Verdict     string `json:"verdict"`
				Reason      string `json:"reason"`
				Disposition string `json:"disposition"`
			}
			if err := json.Unmarshal(event.Payload, &call); err != nil {
				t.Fatal(err)
			}
			if call.Turn != i+1 {
				t.Fatalf("event[%d] turn=%d, want %d", i, call.Turn, i+1)
			}
			if i == 1 && (call.Tool != "write_file" || call.Verdict != "DENY" || call.Reason != "POLICY_BLOCK" || call.Disposition != "TERMINAL") {
				t.Fatalf("deny evidence lost: %+v", call)
			}
		}
		warnings := strings.ToLower(strings.Join(firstReceipt.Warnings, " "))
		for _, limitation := range []string{"timestamp", "tool result", "usage"} {
			if !strings.Contains(warnings, limitation) {
				t.Fatalf("receipt does not disclose missing %s fidelity: %+v", limitation, firstReceipt)
			}
		}
	})

	t.Run("legacy aggregate receipt is identifiable but not fully audited", func(t *testing.T) {
		legacy := []byte(`{"schema":"fak.agent.native.v1","task":"legacy","model":"fixture","status":"completed","metrics":{"arm":"fak","turns":2}}`)
		events, receipt, err := DefaultAdapterRegistry().Ingest(source, legacy)
		if err != nil {
			t.Fatal(err)
		}
		for _, event := range events {
			if event.Kind == EventTool {
				t.Fatalf("aggregate-only receipt invented audited tool event: %+v", event)
			}
		}
		warnings := strings.ToLower(strings.Join(receipt.Warnings, " "))
		if receipt.SourceDigest != digestBytes(legacy) || !strings.Contains(warnings, "aggregate") || !strings.Contains(warnings, "audit") {
			t.Fatalf("legacy fidelity limitation missing: events=%+v receipt=%+v", events, receipt)
		}
	})

	t.Run("unsupported call envelope is refused with source identity", func(t *testing.T) {
		unsupported := []byte(`{"schema":"fak.agent.native.v1","task":"future","model":"fixture","calls":{"schema":"fak.agent.native.calls.v2","entries":[]}}`)
		_, receipt, err := DefaultAdapterRegistry().Ingest(source, unsupported)
		if err == nil || !strings.Contains(strings.ToLower(err.Error()), "unsupported") {
			t.Fatalf("unsupported native calls version accepted: err=%v receipt=%+v", err, receipt)
		}
		if receipt.SourceType != source || receipt.SourceDigest != digestBytes(unsupported) {
			t.Fatalf("refusal lost source identity: %+v", receipt)
		}
	})
}

func TestCodexAdapterProducesDeterministicFidelityReceipt(t *testing.T) {
	input := []byte(strings.Join([]string{
		`{"timestamp":"2026-08-17T16:00:00Z","type":"session_meta","payload":{"id":"codex-session"}}`,
		`{"timestamp":"2026-08-17T16:00:01Z","type":"event_msg","payload":{"type":"user_message","id":"m1","message":"inspect it"}}`,
		`{"timestamp":"2026-08-17T16:00:02Z","type":"response_item","payload":{"type":"function_call","call_id":"call-1","name":"shell","arguments":"{}"}}`,
		`{"timestamp":"2026-08-17T16:00:03Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-1","output":"ok"}}`,
		`{"timestamp":"2026-08-17T16:00:04Z","type":"future_record","payload":{"type":"new_shape","secret":"retained only by source digest"}}`,
	}, "\n") + "\n")

	registry := DefaultAdapterRegistry()
	firstEvents, firstReceipt, err := registry.Ingest("codex-jsonl", input)
	if err != nil {
		t.Fatal(err)
	}
	secondEvents, secondReceipt, err := registry.Ingest("codex-jsonl", input)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := EncodeEvents(firstEvents)
	secondJSON, _ := EncodeEvents(secondEvents)
	if !bytes.Equal(firstJSON, secondJSON) || firstReceipt.EventDigest != secondReceipt.EventDigest {
		t.Fatal("codex ingestion is not deterministic")
	}
	if firstReceipt.InputRecords != 5 || firstReceipt.EmittedEvents != 4 || firstReceipt.UnknownKinds["future_record/new_shape"] != 1 {
		t.Fatalf("receipt=%+v", firstReceipt)
	}
	if firstReceipt.SourceDigest == "" || firstReceipt.EventDigest == "" || len(firstReceipt.Warnings) == 0 {
		t.Fatalf("incomplete receipt=%+v", firstReceipt)
	}
	if firstEvents[0].ConversationID != "codex-session" || firstEvents[2].Kind != EventTool || firstEvents[2].Action != "proposed" {
		t.Fatalf("events=%+v", firstEvents)
	}
}

func TestAGUIAdapterPreservesStreamingAndStateSemantics(t *testing.T) {
	input := []byte(strings.Join([]string{
		`{"type":"RUN_STARTED","threadId":"thread-1","runId":"run-1","eventId":"run-start","timestamp":"2026-08-17T16:00:00Z"}`,
		`{"type":"TEXT_MESSAGE_START","threadId":"thread-1","messageId":"message-1","role":"assistant","timestamp":"2026-08-17T16:00:01Z"}`,
		`{"type":"TEXT_MESSAGE_CONTENT","threadId":"thread-1","messageId":"message-1","delta":"hello","timestamp":"2026-08-17T16:00:02Z"}`,
		`{"type":"STATE_DELTA","threadId":"thread-1","eventId":"state-1","delta":[{"op":"add","path":"/status","value":"working"}],"timestamp":"2026-08-17T16:00:03Z"}`,
		`{"type":"TOOL_CALL_START","threadId":"thread-1","toolCallId":"tool-1","toolCallName":"search","timestamp":"2026-08-17T16:00:04Z"}`,
	}, "\n") + "\n")

	events, receipt, err := DefaultAdapterRegistry().Ingest("ag-ui-jsonl", input)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.InputRecords != 5 || receipt.EmittedEvents != 5 || len(receipt.UnknownKinds) != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
	want := []struct {
		kind   EventKind
		action string
	}{{EventRunLifecycle, "started"}, {EventMessage, "started"}, {EventMessage, "delta"}, {EventState, "delta"}, {EventTool, "started"}}
	for i := range want {
		if events[i].Kind != want[i].kind || events[i].Action != want[i].action || events[i].Source.RawDigest == "" {
			t.Fatalf("event %d=%+v", i, events[i])
		}
	}
}

func TestAdapterRegistryRequiresExplicitSource(t *testing.T) {
	registry := DefaultAdapterRegistry()
	if got := strings.Join(registry.Sources(), ","); got != "ag-ui-jsonl,claude-code-jsonl,codex-jsonl,fak-agent-native-receipt-json,openai-chat-export-jsonl,pi-jsonl" {
		t.Fatalf("sources=%q", got)
	}
	if _, _, err := registry.Ingest("guess", []byte(`{}`)); err == nil || !strings.Contains(err.Error(), "no trajectory adapter") {
		t.Fatalf("err=%v", err)
	}
}

func TestAdaptersRejectMalformedRecordsWithPartialReceipt(t *testing.T) {
	input := []byte("{\"type\":\"RUN_STARTED\",\"threadId\":\"t\"}\nnot-json\n")
	_, receipt, err := DefaultAdapterRegistry().Ingest("ag-ui-jsonl", input)
	if err == nil || !strings.Contains(err.Error(), "record 2") {
		t.Fatalf("err=%v", err)
	}
	if receipt.InputRecords != 2 || receipt.MalformedRecord != 1 || receipt.SourceDigest == "" {
		t.Fatalf("partial receipt=%+v", receipt)
	}
}

func TestMissingTimestampsAreVisibleAndDeterministic(t *testing.T) {
	input := []byte(`{"type":"CUSTOM","threadId":"thread-1","eventId":"e1","name":"phase"}` + "\n")
	first, firstReceipt, err := DefaultAdapterRegistry().Ingest("ag-ui-jsonl", input)
	if err != nil {
		t.Fatal(err)
	}
	second, secondReceipt, err := DefaultAdapterRegistry().Ingest("ag-ui-jsonl", input)
	if err != nil {
		t.Fatal(err)
	}
	if firstReceipt.SyntheticTimes != 1 || len(firstReceipt.Warnings) != 1 || !first[0].Timestamp.Equal(second[0].Timestamp) || firstReceipt.EventDigest != secondReceipt.EventDigest {
		t.Fatalf("first=%+v receipt=%+v second=%+v", first, firstReceipt, second)
	}
}

func TestClaudeAndOpenAIExportAdaptersPreserveSemanticsAndFidelity(t *testing.T) {
	cases := []struct {
		source    string
		input     string
		wantKinds []EventKind
	}{
		{"claude-code-jsonl", strings.Join([]string{
			`{"type":"user","session_id":"claude-session","id":"u1","timestamp":"2026-08-17T16:00:00Z","message":{"role":"user","content":"hello"}}`,
			`{"type":"tool_use","session_id":"claude-session","id":"t1","timestamp":"2026-08-17T16:00:01Z","payload":{"name":"search"}}`,
			`{"type":"future_kind","session_id":"claude-session","id":"x1","timestamp":"2026-08-17T16:00:02Z","payload":{"opaque":true}}`,
		}, "\n") + "\n", []EventKind{EventMessage, EventTool, EventObservation}},
		{"openai-chat-export-jsonl", strings.Join([]string{
			`{"type":"message","conversation_id":"openai-chat","id":"m1","timestamp":"2026-08-17T16:00:00Z","payload":{"role":"assistant","text":"hi"}}`,
			`{"type":"function_call_output","conversation_id":"openai-chat","id":"o1","timestamp":"2026-08-17T16:00:01Z","payload":{"call_id":"c1","output":"ok"}}`,
		}, "\n") + "\n", []EventKind{EventMessage, EventTool}},
	}
	registry := DefaultAdapterRegistry()
	if got := strings.Join(registry.Sources(), ","); got != "ag-ui-jsonl,claude-code-jsonl,codex-jsonl,fak-agent-native-receipt-json,openai-chat-export-jsonl,pi-jsonl" {
		t.Fatalf("sources=%q", got)
	}
	for _, tc := range cases {
		t.Run(tc.source, func(t *testing.T) {
			first, firstReceipt, err := registry.Ingest(tc.source, []byte(tc.input))
			if err != nil {
				t.Fatal(err)
			}
			second, secondReceipt, err := registry.Ingest(tc.source, []byte(tc.input))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstReceipt, secondReceipt) {
				t.Fatal("export ingestion is not deterministic")
			}
			for i, want := range tc.wantKinds {
				if first[i].Kind != want {
					t.Fatalf("event %d kind=%q, want %q", i, first[i].Kind, want)
				}
			}
			for _, event := range first {
				if event.Source.RawDigest == "" || event.Source.EventID == "" {
					t.Fatalf("event lacks source provenance: %#v", event)
				}
			}
			if tc.source == "claude-code-jsonl" {
				if firstReceipt.UnknownKinds["future_kind"] != 1 || first[2].Loss == nil {
					t.Fatalf("unknown kind not visible: %#v %#v", firstReceipt, first[2])
				}
			}
		})
	}
}

func TestExportAdaptersReturnPartialReceiptOnMalformedRecord(t *testing.T) {
	input := []byte("{\"type\":\"message\",\"conversation_id\":\"chat\",\"payload\":{}}\n{broken\n")
	events, receipt, err := DefaultAdapterRegistry().Ingest("openai-chat-export-jsonl", input)
	if err == nil {
		t.Fatal("malformed export accepted")
	}
	if len(events) != 1 || receipt.MalformedRecord != 1 || receipt.InputRecords != 2 || receipt.EventDigest == "" {
		t.Fatalf("partial receipt lost: events=%d receipt=%#v", len(events), receipt)
	}
}

// TestPiAdapter pins #13368: the Pi session JSONL adapter registers with the
// existing Adapter/FidelityReceipt API and preserves session/entry/parent IDs,
// active branch order, raw digest, tool-call/result linkage, usage provenance and
// unknown kinds. The fixture is an inline sanitized copy of the installed Pi
// session schema (version 3); no private transcript is used.
const piFixture = `{"type":"session","version":3,"id":"pi-session-1","timestamp":"2026-09-30T00:00:00.000Z","cwd":"C:/work/fak"}
{"type":"model_change","id":"mc1","parentId":null,"timestamp":"2026-09-30T00:00:01.000Z","provider":"hive-ai","modelId":"deepseek-ai/DeepSeek-V4.1-Flash"}
{"type":"message","id":"u1","parentId":"mc1","timestamp":"2026-09-30T00:00:02.000Z","message":{"role":"user","content":[{"type":"text","text":"run the tests"}]}}
{"type":"message","id":"a1","parentId":"u1","timestamp":"2026-09-30T00:00:03.000Z","message":{"role":"assistant","content":[{"type":"text","text":"running"},{"type":"toolCall","id":"call_1","name":"bash","arguments":{"command":"go test ./..."}}],"usage":{"input":100,"output":20,"totalTokens":120},"provider":"hive-ai","model":"deepseek-ai/DeepSeek-V4.1-Flash"}}
{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-09-30T00:00:04.000Z","message":{"role":"toolResult","toolCallId":"call_1","toolName":"bash","content":[{"type":"text","text":"ok"}]}}
{"type":"future_kind","id":"x1","parentId":"r1","timestamp":"2026-09-30T00:00:05.000Z","opaque":true}
`

func TestPiAdapter(t *testing.T) {
	registry := DefaultAdapterRegistry()
	if got := strings.Join(registry.Sources(), ","); !strings.Contains(got, "pi-jsonl") {
		t.Fatalf("pi-jsonl not registered: %q", got)
	}

	first, firstReceipt, err := registry.Ingest("pi-jsonl", []byte(piFixture))
	if err != nil {
		t.Fatal(err)
	}
	second, secondReceipt, err := registry.Ingest("pi-jsonl", []byte(piFixture))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstReceipt, secondReceipt) {
		t.Fatal("pi ingestion is not deterministic")
	}
	if firstReceipt.SourceType != "pi-jsonl" || firstReceipt.Adapter != "pi-jsonl" || firstReceipt.AdapterVersion == "" {
		t.Fatalf("receipt identity wrong: %#v", firstReceipt)
	}
	if firstReceipt.InputRecords != 6 || firstReceipt.MalformedRecord != 0 {
		t.Fatalf("record accounting wrong: %#v", firstReceipt)
	}
	if firstReceipt.UnknownKinds["future_kind"] != 1 {
		t.Fatalf("unknown kind not counted: %#v", firstReceipt.UnknownKinds)
	}

	// Session/entry/parent IDs and raw digests are preserved on every event.
	for _, event := range first {
		if event.ConversationID != "pi-session-1" {
			t.Fatalf("event %s conversation=%q, want pi-session-1", event.ID, event.ConversationID)
		}
		if event.Source.EventID == "" || event.Source.RawDigest == "" || event.Source.SessionID != "pi-session-1" {
			t.Fatalf("event lacks preserved source identity: %#v", event.Source)
		}
	}

	// Kinds in order: session, model_change, user msg, assistant msg, tool
	// (proposed), tool (completed), unknown observation.
	wantKinds := []EventKind{EventRunLifecycle, EventObservation, EventMessage, EventMessage, EventTool, EventTool, EventObservation}
	if len(first) != len(wantKinds) {
		t.Fatalf("event count = %d, want %d: %#v", len(first), len(wantKinds), first)
	}
	for i, want := range wantKinds {
		if first[i].Kind != want {
			t.Fatalf("event %d kind=%q, want %q", i, first[i].Kind, want)
		}
	}

	// Tool-call/result linkage: the completed tool names the proposed call.
	var proposed, completed *Event
	for i := range first {
		if first[i].Kind == EventTool && first[i].Action == "proposed" {
			proposed = &first[i]
		}
		if first[i].Kind == EventTool && first[i].Action == "completed" {
			completed = &first[i]
		}
	}
	if proposed == nil || completed == nil {
		t.Fatalf("missing tool proposed/completed pairing: %#v", first)
	}
	if !contains(completed.ParentIDs, proposed.Source.EventID) {
		t.Fatalf("completed tool does not link to its call: parent=%v call=%q", completed.ParentIDs, proposed.Source.EventID)
	}
	if proposed.Payload == nil || !strings.Contains(string(proposed.Payload), "go test") {
		t.Fatalf("tool call arguments not preserved: %s", proposed.Payload)
	}

	// Usage provenance survives on the assistant message payload.
	var assistantPayload string
	for _, event := range first {
		if event.Kind == EventMessage && strings.Contains(string(event.Payload), `"assistant"`) {
			assistantPayload = string(event.Payload)
		}
	}
	if !strings.Contains(assistantPayload, "totalTokens") {
		t.Fatalf("usage provenance not preserved: %s", assistantPayload)
	}
}

// TestPiAdapterBranchedSessionIsNotFlattened pins that a parent-linked branch is
// preserved as distinct entries with their parent IDs, rather than collapsed into
// one fictitious executed trajectory.
func TestPiAdapterBranchedSessionIsNotFlattened(t *testing.T) {
	const branched = `{"type":"session","version":3,"id":"pi-branch","timestamp":"2026-09-30T00:00:00.000Z"}
{"type":"message","id":"root","parentId":"","timestamp":"2026-09-30T00:00:01.000Z","message":{"role":"user","content":[{"type":"text","text":"go"}]}}
{"type":"message","id":"left","parentId":"root","timestamp":"2026-09-30T00:00:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"left"}]}}
{"type":"message","id":"right","parentId":"root","timestamp":"2026-09-30T00:00:03.000Z","message":{"role":"assistant","content":[{"type":"text","text":"right"}]}}
`
	events, _, err := DefaultAdapterRegistry().Ingest("pi-jsonl", []byte(branched))
	if err != nil {
		t.Fatal(err)
	}
	var leftEv, rightEv *Event
	for i := range events {
		switch events[i].Source.EventID {
		case "left":
			leftEv = &events[i]
		case "right":
			rightEv = &events[i]
		}
	}
	if leftEv == nil || rightEv == nil {
		t.Fatalf("branch entries dropped: %#v", events)
	}
	if !contains(leftEv.ParentIDs, "root") || !contains(rightEv.ParentIDs, "root") {
		t.Fatalf("branch parent IDs not preserved: left=%v right=%v", leftEv.ParentIDs, rightEv.ParentIDs)
	}
}

// TestPiAdapterDuplicateToolResultDoesNotDuplicateEffect ensures a repeated
// terminal record for the same tool call does not create a second tool effect.
func TestPiAdapterDuplicateToolResultDoesNotDuplicateEffect(t *testing.T) {
	const dup = `{"type":"session","version":3,"id":"pi-dup","timestamp":"2026-09-30T00:00:00.000Z"}
{"type":"message","id":"a1","timestamp":"2026-09-30T00:00:01.000Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_9","name":"bash","arguments":{"command":"ls"}}]}}
{"type":"message","id":"r1","timestamp":"2026-09-30T00:00:02.000Z","message":{"role":"toolResult","toolCallId":"call_9","toolName":"bash","content":[]}}
{"type":"message","id":"r1dup","timestamp":"2026-09-30T00:00:03.000Z","message":{"role":"toolResult","toolCallId":"call_9","toolName":"bash","content":[]}}
`
	events, _, err := DefaultAdapterRegistry().Ingest("pi-jsonl", []byte(dup))
	if err != nil {
		t.Fatal(err)
	}
	completed := 0
	for _, event := range events {
		if event.Kind == EventTool && event.Action == "completed" {
			completed++
		}
	}
	if completed != 1 {
		t.Fatalf("duplicate terminal record created %d tool effects, want 1", completed)
	}
}

// TestPiAdapterRejectsUnsupportedVersion: a session that declares a schema
// version this adapter does not understand cannot qualify a run.
func TestPiAdapterRejectsUnsupportedVersion(t *testing.T) {
	const future = `{"type":"session","version":99,"id":"pi-future","timestamp":"2026-09-30T00:00:00.000Z"}` + "\n"
	if _, _, err := DefaultAdapterRegistry().Ingest("pi-jsonl", []byte(future)); err == nil {
		t.Fatal("unsupported Pi session version must be refused, not qualified")
	}
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
