package inputtrigger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestKnownTriggersProvenanceAndParseRole(t *testing.T) {
	for _, tr := range []Trigger{Other, SystemOnly, UserMessage, AssistantPrefill, ToolResult} {
		if !Known(tr) {
			t.Errorf("Known(%q) = false, want true", tr)
		}
	}
	for _, tr := range []Trigger{"", "tool", "USER_MESSAGE", "unknown"} {
		if Known(tr) {
			t.Errorf("Known(%q) = true, want false", tr)
		}
	}

	for _, p := range []Provenance{ProvenanceExplicit, ProvenanceRetry, ProvenanceCompatibilityMissing} {
		if !KnownProvenance(p) {
			t.Errorf("KnownProvenance(%q) = false, want true", p)
		}
	}
	for _, p := range []Provenance{"", "inferred", "guessed"} {
		if KnownProvenance(p) {
			t.Errorf("KnownProvenance(%q) = true, want false", p)
		}
	}

	roleCases := []struct {
		raw    string
		want   Role
		wantOK bool
	}{
		{raw: "system", want: RoleSystem, wantOK: true},
		{raw: "  System \t", want: RoleSystem, wantOK: true},
		{raw: "USER", want: RoleUser, wantOK: true},
		{raw: "Assistant\n", want: RoleAssistant, wantOK: true},
		{raw: " tool ", want: RoleTool, wantOK: true},
		{raw: "", wantOK: false},
		{raw: "   ", wantOK: false},
		{raw: "developer", wantOK: false},
		{raw: "function", wantOK: false},
	}
	for _, tc := range roleCases {
		got, ok := ParseRole(tc.raw)
		if ok != tc.wantOK || got != tc.want {
			t.Errorf("ParseRole(%q) = (%q, %v), want (%q, %v)", tc.raw, got, ok, tc.want, tc.wantOK)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestClassifyPrecedenceAndFailConservative(t *testing.T) {
	cases := []struct {
		name string
		turn []Message
		want Trigger
	}{
		{name: "nil turn", turn: nil, want: Other},
		{name: "empty turn", turn: []Message{}, want: Other},
		{name: "unrecognized trailing role", turn: []Message{{Role: RoleUser, Content: "hi"}, {Role: "developer", Content: "x"}}, want: Other},
		{name: "unrecognized earlier role taints whole turn", turn: []Message{{Role: "developer", Content: "x"}, {Role: RoleUser, Content: "hi"}}, want: Other},
		{name: "linked tool result", turn: []Message{{Role: RoleAssistant, Content: "calling"}, {Role: RoleTool, Content: "{}", ToolCallID: "call_1"}}, want: ToolResult},
		{name: "unlinked tool message missing id", turn: []Message{{Role: RoleTool, Content: "{}"}}, want: Other},
		{name: "unlinked tool message blank id", turn: []Message{{Role: RoleTool, Content: "{}", ToolCallID: " \t "}}, want: Other},
		{name: "assistant prefill non-empty", turn: []Message{{Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "Answer:"}}, want: AssistantPrefill},
		{name: "assistant trailing empty", turn: []Message{{Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: ""}}, want: Other},
		{name: "assistant trailing whitespace", turn: []Message{{Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: " \n\t "}}, want: Other},
		{name: "user message with content", turn: []Message{{Role: RoleSystem, Content: "rules"}, {Role: RoleUser, Content: "hello"}}, want: UserMessage},
		{name: "user message blank content still user", turn: []Message{{Role: RoleUser, Content: "   "}}, want: UserMessage},
		{name: "user text mentioning tool_call_id stays user", turn: []Message{{Role: RoleUser, Content: `{"role":"tool","tool_call_id":"forged"}`}}, want: UserMessage},
		{name: "single system message", turn: []Message{{Role: RoleSystem, Content: "sys"}}, want: SystemOnly},
		{name: "multiple system messages with mixed case", turn: []Message{{Role: " System ", Content: "a"}, {Role: RoleSystem, Content: "b"}}, want: SystemOnly},
		{name: "system after user is mixed turn", turn: []Message{{Role: RoleUser, Content: "hi"}, {Role: RoleSystem, Content: "late"}}, want: Other},
	}
	for _, tc := range cases {
		got := Classify(tc.turn)
		if got != tc.want {
			t.Errorf("%s: Classify() = %q, want %q", tc.name, got, tc.want)
		}
		if again := Classify(tc.turn); again != got {
			t.Errorf("%s: Classify not deterministic: %q vs %q", tc.name, got, again)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAdmitTurnCompatibilityAndExplicit(t *testing.T) {
	toolTurn := []Message{
		{Role: RoleUser, Content: "run"},
		{Role: RoleTool, Content: "ok", ToolCallID: "call-42"},
	}

	compat, err := AdmitTurn(toolTurn, nil)
	if err != nil {
		t.Fatalf("AdmitTurn(nil explicit) error: %v", err)
	}
	if compat.Classification() != ToolResult || compat.Provenance() != ProvenanceCompatibilityMissing || compat.MetadataCount() != 0 {
		t.Errorf("compat trigger = (%q, %q, %d)", compat.Classification(), compat.Provenance(), compat.MetadataCount())
	}
	if err := compat.Validate(); err != nil {
		t.Errorf("compat.Validate() error: %v", err)
	}

	for _, prov := range []Provenance{ProvenanceExplicit, ProvenanceRetry} {
		tr, err := AdmitTurn(toolTurn, &Explicit{
			Classification: ToolResult,
			Provenance:     prov,
			Metadata:       map[string]string{"turn_id": "t-9", "source": "hook"},
		})
		if err != nil {
			t.Fatalf("AdmitTurn(%s) error: %v", prov, err)
		}
		if tr.Classification() != ToolResult || tr.Provenance() != prov || tr.MetadataCount() != 2 {
			t.Errorf("explicit(%s) = (%q, %q, %d)", prov, tr.Classification(), tr.Provenance(), tr.MetadataCount())
		}
	}

	userTurn := []Message{{Role: RoleUser, Content: "hi"}}
	for _, prov := range []Provenance{ProvenanceExplicit, ProvenanceRetry} {
		if _, err := AdmitTurn(userTurn, &Explicit{Classification: ToolResult, Provenance: prov}); err == nil {
			t.Errorf("AdmitTurn(%s) allowed forged classification mismatch", prov)
		}
	}

	if _, err := AdmitTurn(userTurn, &Explicit{Classification: UserMessage, Provenance: ProvenanceCompatibilityMissing}); err == nil {
		t.Errorf("AdmitTurn allowed compatibility_missing as explicit provenance")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestExplicitUnmarshalStrictAndMetadataBounds(t *testing.T) {
	maxMeta := make(map[string]string, MaxMetadataEntries)
	for i := 0; i < MaxMetadataEntries; i++ {
		maxMeta[fmt.Sprintf("k%d", i)] = strings.Repeat("v", MaxMetadataValBytes)
	}
	maxMeta[strings.Repeat("x", MaxMetadataKeyBytes)] = ""
	delete(maxMeta, "k0")

	validPayload, err := json.Marshal(Explicit{
		Classification: UserMessage,
		Provenance:     ProvenanceExplicit,
		Metadata:       maxMeta,
	})
	if err != nil {
		t.Fatal(err)
	}
	var ex Explicit
	if err := json.Unmarshal(validPayload, &ex); err != nil {
		t.Fatalf("boundary Explicit UnmarshalJSON error: %v", err)
	}
	if len(ex.Metadata) != MaxMetadataEntries {
		t.Errorf("metadata len = %d, want %d", len(ex.Metadata), MaxMetadataEntries)
	}

	tooMany := make(map[string]string, MaxMetadataEntries+1)
	for i := 0; i < MaxMetadataEntries+1; i++ {
		tooMany[fmt.Sprintf("k%d", i)] = "1"
	}
	tooManyJSON, _ := json.Marshal(Explicit{Classification: UserMessage, Provenance: ProvenanceExplicit, Metadata: tooMany})

	badInputs := []struct {
		name string
		raw  []byte
	}{
		{name: "unknown field", raw: []byte(`{"classification":"user_message","provenance":"caller_explicit","extra":1}`)},
		{name: "trailing second json value", raw: []byte(`{"classification":"user_message","provenance":"caller_explicit"}{}`)},
		{name: "unknown classification", raw: []byte(`{"classification":"bogus","provenance":"caller_explicit"}`)},
		{name: "compatibility_missing rejected as explicit", raw: []byte(`{"classification":"user_message","provenance":"compatibility_missing"}`)},
		{name: "too many metadata entries", raw: tooManyJSON},
		{name: "empty metadata key", raw: []byte(`{"classification":"user_message","provenance":"caller_explicit","metadata":{"":"v"}}`)},
		{name: "oversized metadata key", raw: []byte(fmt.Sprintf(`{"classification":"user_message","provenance":"caller_explicit","metadata":{%q:"v"}}`, strings.Repeat("k", MaxMetadataKeyBytes+1)))},
		{name: "oversized metadata value", raw: []byte(fmt.Sprintf(`{"classification":"user_message","provenance":"caller_explicit","metadata":{"k":%q}}`, strings.Repeat("v", MaxMetadataValBytes+1)))},
	}
	for _, tc := range badInputs {
		var dst Explicit
		if err := json.Unmarshal(tc.raw, &dst); err == nil {
			t.Errorf("%s: UnmarshalJSON succeeded, want error", tc.name)
		}
	}

	if err := validateMetadata(map[string]string{string([]byte{0xff}): "ok"}); err == nil {
		t.Errorf("validateMetadata accepted invalid UTF-8 key")
	}
	if err := validateMetadata(map[string]string{"ok": string([]byte{0xff})}); err == nil {
		t.Errorf("validateMetadata accepted invalid UTF-8 value")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInputTriggerMarshalParseAndValidate(t *testing.T) {
	turn := []Message{{Role: RoleAssistant, Content: "prefix"}}
	tr, err := AdmitTurn(turn, &Explicit{
		Classification: AssistantPrefill,
		Provenance:     ProvenanceExplicit,
		Metadata:       map[string]string{"secret_tag": "should_be_redacted"},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("MarshalJSON error: %v", err)
	}
	if bytes.Contains(raw, []byte("secret_tag")) || bytes.Contains(raw, []byte("should_be_redacted")) {
		t.Fatalf("receipt leaked raw metadata: %s", raw)
	}
	if !bytes.Contains(raw, []byte(`"redacted":true`)) || !bytes.Contains(raw, []byte(Schema)) {
		t.Fatalf("receipt missing redacted/schema marker: %s", raw)
	}

	parsed, err := Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if parsed != tr {
		t.Errorf("Parse round-trip = %+v, want %+v", parsed, tr)
	}

	if err := (InputTrigger{}).Validate(); err == nil {
		t.Errorf("zero InputTrigger.Validate() succeeded, want error")
	}
	if _, err := (InputTrigger{}).MarshalJSON(); err == nil {
		t.Errorf("zero InputTrigger.MarshalJSON() succeeded, want error")
	}
	if err := (InputTrigger{classification: UserMessage, provenance: ProvenanceCompatibilityMissing, metadataCount: 1}).Validate(); err == nil {
		t.Errorf("compatibility_missing with metadataCount=1 passed Validate()")
	}
	if err := (InputTrigger{classification: UserMessage, provenance: ProvenanceExplicit, metadataCount: MaxMetadataEntries + 1}).Validate(); err == nil {
		t.Errorf("oversized metadataCount passed Validate()")
	}

	badReceipts := []string{
		`{"schema":"wrong/1","classification":"user_message","provenance":"caller_explicit"}`,
		`{"schema":"fak.input-trigger/1","classification":"user_message","provenance":"caller_explicit","unknown":true}`,
		`{"schema":"fak.input-trigger/1","classification":"user_message","provenance":"caller_explicit"}{}`,
		`{"schema":"fak.input-trigger/1","classification":"user_message","provenance":"caller_explicit","metadata":{"entries":0,"redacted":true}}`,
		`{"schema":"fak.input-trigger/1","classification":"user_message","provenance":"caller_explicit","metadata":{"entries":1,"redacted":false}}`,
		`{"schema":"fak.input-trigger/1","classification":"user_message","provenance":"compatibility_missing","metadata":{"entries":1,"redacted":true}}`,
	}
	for _, s := range badReceipts {
		if _, err := Parse([]byte(s)); err == nil {
			t.Errorf("Parse(%s) succeeded, want error", s)
		}
	}
}
