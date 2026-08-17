package trajectory

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestAudienceViewsDifferAndRedactBeforeRendering(t *testing.T) {
	events := viewFixture()
	operator, operatorReceipt, err := CompileView(events, OperatorView())
	if err != nil {
		t.Fatal(err)
	}
	endUser, endUserReceipt, err := CompileView(events, EndUserView())
	if err != nil {
		t.Fatal(err)
	}
	if len(operator) <= len(endUser) {
		t.Fatalf("operator=%d end-user=%d, want materially different views", len(operator), len(endUser))
	}
	operatorJSON, _ := json.Marshal(operator)
	endUserJSON, _ := json.Marshal(endUser)
	for name, rendered := range map[string][]byte{"operator": operatorJSON, "end-user": endUserJSON} {
		if bytes.Contains(rendered, []byte("super-secret")) || bytes.Contains(rendered, []byte("token-123")) {
			t.Fatalf("%s renderer received secret material: %s", name, rendered)
		}
	}
	if !bytes.Contains(endUserJSON, []byte("[REDACTED]")) || endUserReceipt.RedactedFields < 1 {
		t.Fatalf("end-user receipt=%+v projection=%s", endUserReceipt, endUserJSON)
	}
	if operatorReceipt.Omitted["not-included"] == 0 || endUserReceipt.Omitted["not-included"] == 0 {
		t.Fatalf("omission accounting missing: operator=%+v end-user=%+v", operatorReceipt, endUserReceipt)
	}
}

func TestViewCompilationIsDeterministicAndDoesNotMutateEvidence(t *testing.T) {
	events := viewFixture()
	before, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	first, firstReceipt, err := CompileView(events, OperatorView())
	if err != nil {
		t.Fatal(err)
	}
	second, secondReceipt, err := CompileView(events, OperatorView())
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := json.Marshal(first)
	secondJSON, _ := json.Marshal(second)
	firstReceiptJSON, _ := json.Marshal(firstReceipt)
	secondReceiptJSON, _ := json.Marshal(secondReceipt)
	if !bytes.Equal(firstJSON, secondJSON) || !bytes.Equal(firstReceiptJSON, secondReceiptJSON) {
		t.Fatalf("non-deterministic projection\n%s\n%s\n%s\n%s", firstJSON, secondJSON, firstReceiptJSON, secondReceiptJSON)
	}
	after, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || !bytes.Contains(after, []byte("super-secret")) {
		t.Fatal("projection mutated canonical evidence")
	}
	if firstReceipt.TrajectoryDigest == "" || firstReceipt.ViewSpecDigest == "" || firstReceipt.ProjectionDigest == "" {
		t.Fatalf("receipt=%+v", firstReceipt)
	}
}

func TestViewRejectsInvalidSpecsAndPayloads(t *testing.T) {
	if _, _, err := CompileView(viewFixture(), ViewSpec{Schema: ViewSpecSchema, Audience: "operator", RedactionProfile: "p"}); err == nil || !strings.Contains(err.Error(), "include") {
		t.Fatalf("err=%v", err)
	}
	bad := viewFixture()[0]
	bad.Payload = json.RawMessage(`{"secret"`)
	if _, _, err := CompileView([]Event{bad}, OperatorView()); err == nil {
		t.Fatal("invalid canonical payload accepted")
	}
}

func TestNestedRedactionRecordsPaths(t *testing.T) {
	payload := json.RawMessage(`{"request":{"headers":{"token":"token-123","safe":"yes"}},"items":[{"api_key":"super-secret"}]}`)
	redacted, fields, err := redactPayload(payload, []string{"token", "api_key"})
	if err != nil {
		t.Fatal(err)
	}
	if string(redacted) != `{"items":[{"api_key":"[REDACTED]"}],"request":{"headers":{"safe":"yes","token":"[REDACTED]"}}}` {
		t.Fatalf("redacted=%s", redacted)
	}
	if strings.Join(fields, ",") != "items[0].api_key,request.headers.token" {
		t.Fatalf("fields=%v", fields)
	}
}

func viewFixture() []Event {
	ts := time.Date(2026, 8, 17, 16, 0, 0, 0, time.UTC)
	return []Event{
		viewEvent("run", EventRunLifecycle, "started", 1, ts, `{}`),
		viewEvent("user", EventMessage, "completed", 2, ts, `{"role":"user","text":"deploy it","secret":"super-secret"}`),
		viewEvent("delta", EventMessage, "delta", 3, ts, `{"role":"assistant","delta":"work"}`),
		viewEvent("tool", EventTool, "completed", 4, ts, `{"name":"shell","token":"token-123","result":"ok"}`),
		viewEvent("approval", EventApproval, "requested", 5, ts, `{"summary":"push?"}`),
		viewEvent("artifact", EventArtifact, "created", 6, ts, `{"path":"report.md"}`),
		viewEvent("steer", EventIntervention, "steer", 7, ts, `{"text":"also test"}`),
	}
}

func viewEvent(id string, kind EventKind, action string, sequence uint64, ts time.Time, payload string) Event {
	return Event{Schema: EventSchema, ID: id, ConversationID: "conv", Kind: kind, Action: action, Timestamp: ts, Sequence: sequence, Visibility: VisibilityDeveloper, Source: EventSource{Type: "fixture", Adapter: "fixture", AdapterVersion: "1"}, Payload: json.RawMessage(payload)}
}
