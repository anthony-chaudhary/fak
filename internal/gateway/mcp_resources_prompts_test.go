package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// rpcRoundTrip drives one JSON-RPC frame through the real dispatch path (the same
// path ServeStdio and POST /mcp use) and decodes the wire response a client sees.
func rpcRoundTrip(t *testing.T, s *Server, method string, params string) rpcResponse {
	t.Helper()
	frame := `{"jsonrpc":"2.0","id":1,"method":"` + method + `"`
	if params != "" {
		frame += `,"params":` + params
	}
	frame += `}`
	resp := s.dispatchRPC(context.Background(), []byte(frame))
	if resp == nil {
		t.Fatalf("%s: dispatch returned nil (a request with an id must get a response)", method)
	}
	return *resp
}

// resultMap re-marshals the response Result back to the wire and decodes it into a
// generic map — the shape a real MCP client parses — so assertions read the same
// bytes a client would, not the in-process `any`.
func resultMap(t *testing.T, resp rpcResponse) map[string]any {
	t.Helper()
	if resp.Error != nil {
		t.Fatalf("unexpected JSON-RPC error: %+v", resp.Error)
	}
	b, err := json.Marshal(resp.Result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	return m
}

// TestInitializeAdvertisesAllThreePrimitives pins that initialize now advertises
// resources and prompts alongside tools — the gate a spec-compliant client uses to
// decide whether to call resources/* and prompts/* at all (#213).
func TestInitializeAdvertisesAllThreePrimitives(t *testing.T) {
	srv := newTestServer(t)
	caps, ok := srv.initializeResult(nil)["capabilities"].(map[string]any)
	if !ok {
		t.Fatalf("capabilities missing or wrong type")
	}
	for _, primitive := range []string{"tools", "resources", "prompts"} {
		if _, present := caps[primitive]; !present {
			t.Errorf("initialize capabilities must advertise %q, got %v", primitive, caps)
		}
	}
}

// TestResourcesListAndRead exercises the MCP resource primitive end to end: list
// surfaces the capabilities resource, and read returns content DERIVED from live
// server state (the running version and the real tool catalog).
func TestResourcesListAndRead(t *testing.T) {
	srv := newTestServer(t)

	list := resultMap(t, rpcRoundTrip(t, srv, "resources/list", ""))
	resources, ok := list["resources"].([]any)
	if !ok || len(resources) == 0 {
		t.Fatalf("resources/list returned no resources: %v", list)
	}
	first := resources[0].(map[string]any)
	uri, _ := first["uri"].(string)
	if uri != "fak://server/capabilities" {
		t.Fatalf("first resource uri = %q, want fak://server/capabilities", uri)
	}
	for _, field := range []string{"name", "description", "mimeType"} {
		if _, present := first[field]; !present {
			t.Errorf("resource descriptor missing %q: %v", field, first)
		}
	}

	read := resultMap(t, rpcRoundTrip(t, srv, "resources/read", `{"uri":"fak://server/capabilities"}`))
	contents, ok := read["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("resources/read contents malformed: %v", read)
	}
	entry := contents[0].(map[string]any)
	text, _ := entry["text"].(string)

	// The body is derived: it must carry the running version and the real tools,
	// not a hand-stamped snapshot.
	var doc struct {
		Name             string `json:"name"`
		Version          string `json:"version"`
		ProtocolVersions []string
		SelfFeatureQuery struct {
			Tool    string   `json:"tool"`
			Ready   bool     `json:"ready"`
			Digest  string   `json:"digest"`
			Sources []string `json:"sources"`
		} `json:"selfFeatureQuery"`
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte(text), &doc); err != nil {
		t.Fatalf("capabilities resource is not valid JSON: %v\n%s", err, text)
	}
	if doc.Name != "fak-gateway" || doc.Version != srv.version {
		t.Errorf("capabilities resource name/version = %q/%q, want fak-gateway/%q", doc.Name, doc.Version, srv.version)
	}
	var names []string
	for _, tl := range doc.Tools {
		names = append(names, tl.Name)
	}
	// It must reflect the actual catalog — fak_adjudicate is a load-bearing tool.
	if !contains(names, "fak_adjudicate") {
		t.Errorf("capabilities resource omitted fak_adjudicate; tools=%v", names)
	}
	if doc.SelfFeatureQuery.Tool != "fak_feature_query" || !doc.SelfFeatureQuery.Ready || doc.SelfFeatureQuery.Digest == "" {
		t.Errorf("capabilities resource omitted self-feature query summary: %+v", doc.SelfFeatureQuery)
	}
	if len(doc.Tools) != len(toolDescriptors()) {
		t.Errorf("capabilities resource lists %d tools, registry has %d", len(doc.Tools), len(toolDescriptors()))
	}
}

// TestResourceReadUnknownURI confirms an unknown URI is a structured parameter
// fault, not a panic or a silent empty body.
func TestResourceReadUnknownURI(t *testing.T) {
	srv := newTestServer(t)
	resp := rpcRoundTrip(t, srv, "resources/read", `{"uri":"fak://server/nope"}`)
	if resp.Error == nil || resp.Error.Code != rpcInvalidParams {
		t.Fatalf("unknown resource must return InvalidParams, got %+v / err %+v", resp.Result, resp.Error)
	}
}

func TestMCPMissingContextResourceReadReturnsClarification(t *testing.T) {
	srv := newTestServer(t)
	read := resultMap(t, rpcRoundTrip(t, srv, "resources/read", `{"uri":"fak://context/missing/deploy-target"}`))
	contents, ok := read["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("resources/read contents malformed: %v", read)
	}
	text, _ := contents[0].(map[string]any)["text"].(string)
	var doc struct {
		Schema  string `json:"schema"`
		Request struct {
			Method  string `json:"method"`
			URI     string `json:"uri"`
			Key     string `json:"key"`
			Reason  string `json:"reason"`
			Audited bool   `json:"audited"`
		} `json:"request"`
		Clarifications struct {
			Bounded   bool `json:"bounded"`
			Questions []struct {
				Key           string `json:"key"`
				Reason        string `json:"reason"`
				DefaultChoice string `json:"default_choice"`
				BudgetTokens  int    `json:"budget_tokens"`
			} `json:"questions"`
		} `json:"clarifications"`
	}
	if err := json.Unmarshal([]byte(text), &doc); err != nil {
		t.Fatalf("missing-context resource is not valid JSON: %v\n%s", err, text)
	}
	if doc.Schema != "fak-mcp-missing-context-resource/1" {
		t.Fatalf("schema = %q", doc.Schema)
	}
	if doc.Request.Method != "resources/read" || doc.Request.Key != "deploy-target" || doc.Request.Reason != "missing_context" || !doc.Request.Audited {
		t.Fatalf("request = %+v, want audited missing-context resource read", doc.Request)
	}
	if !doc.Clarifications.Bounded || len(doc.Clarifications.Questions) != 1 {
		t.Fatalf("clarifications = %+v, want one bounded question", doc.Clarifications)
	}
	q := doc.Clarifications.Questions[0]
	if q.Key != "deploy-target" || q.Reason != "missing_context" || q.DefaultChoice == "" || q.BudgetTokens <= 0 {
		t.Fatalf("question = %+v, want actionable missing-context clarification", q)
	}
}

func TestQueryAuditRecordsMissingContextDefaultAndAssumptionLink(t *testing.T) {
	srv := newTestServer(t)
	read := resultMap(t, rpcRoundTrip(t, srv, "resources/read", `{"uri":"fak://context/missing/deploy-target"}`))
	contents, ok := read["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("resources/read contents malformed: %v", read)
	}
	text, _ := contents[0].(map[string]any)["text"].(string)
	var doc struct {
		Audit []ContextQueryAuditRecord `json:"audit"`
	}
	if err := json.Unmarshal([]byte(text), &doc); err != nil {
		t.Fatalf("missing-context resource is not valid JSON: %v\n%s", err, text)
	}
	if len(doc.Audit) != 1 {
		t.Fatalf("audit rows = %+v, want one context-query row", doc.Audit)
	}
	row := doc.Audit[0]
	if row.Seq != 1 || row.ID == "" || row.Event != "context_question" {
		t.Fatalf("audit identity = %+v, want stable context_question row", row)
	}
	if row.Method != "resources/read" || row.URI != "fak://context/missing/deploy-target" {
		t.Fatalf("audit request provenance = %+v", row)
	}
	if row.Key != "deploy-target" || row.Reason != "missing_context" || row.Question == "" {
		t.Fatalf("audit question = %+v, want missing-context question", row)
	}
	if row.AnswerSource != "default" || row.DefaultChoice == "" || row.Answer != row.DefaultChoice {
		t.Fatalf("audit answer = %+v, want explicit default answer", row)
	}
	if row.AssumptionKey != "deploy-target" || row.AssumptionSource != "default" || row.AssumptionSourceRef != row.ID {
		t.Fatalf("audit assumption link = %+v, want answer/default source ref", row)
	}

	snap := srv.contextQueryAuditSnapshot()
	if len(snap) != 1 || snap[0].ID != row.ID {
		t.Fatalf("server query-audit snapshot = %+v, want row %q", snap, row.ID)
	}
	vars := srv.debugVars(time.Unix(1, 0))
	if len(vars.ContextQueries) != 1 || vars.ContextQueries[0].AssumptionSourceRef != row.ID {
		t.Fatalf("debug vars context query audit = %+v, want replay link %q", vars.ContextQueries, row.ID)
	}
}

// TestPromptsListAndGet exercises the MCP prompt primitive: list surfaces the
// guarded-call template, and get expands it with the caller's argument.
func TestPromptsListAndGet(t *testing.T) {
	srv := newTestServer(t)

	list := resultMap(t, rpcRoundTrip(t, srv, "prompts/list", ""))
	prompts, ok := list["prompts"].([]any)
	if !ok || len(prompts) == 0 {
		t.Fatalf("prompts/list returned no prompts: %v", list)
	}
	if name, _ := prompts[0].(map[string]any)["name"].(string); name != "fak_guarded_call" {
		t.Fatalf("first prompt name = %q, want fak_guarded_call", name)
	}

	got := resultMap(t, rpcRoundTrip(t, srv, "prompts/get",
		`{"name":"fak_guarded_call","arguments":{"tool":"Bash","task":"delete logs"}}`))
	messages, ok := got["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("prompts/get messages malformed: %v", got)
	}
	content := messages[0].(map[string]any)["content"].(map[string]any)
	text, _ := content["text"].(string)
	// The argument must be interpolated and the guidance must name the real verb.
	if !strings.Contains(text, "Bash") {
		t.Errorf("expanded prompt did not interpolate the tool argument: %q", text)
	}
	if !strings.Contains(text, "fak_adjudicate") {
		t.Errorf("expanded prompt did not reference fak_adjudicate: %q", text)
	}
	if !strings.Contains(text, "delete logs") {
		t.Errorf("expanded prompt did not carry the task argument: %q", text)
	}
}

// TestPromptGetUnknownName confirms an unknown prompt is a structured parameter fault.
func TestPromptGetUnknownName(t *testing.T) {
	srv := newTestServer(t)
	resp := rpcRoundTrip(t, srv, "prompts/get", `{"name":"does_not_exist"}`)
	if resp.Error == nil || resp.Error.Code != rpcInvalidParams {
		t.Fatalf("unknown prompt must return InvalidParams, got %+v / err %+v", resp.Result, resp.Error)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
