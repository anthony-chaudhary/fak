package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// mcp_trajquery_test.go is the contract test for the fak_trajquery MCP tool (#3550):
// it is discoverable in tools/list, its inputSchema advertises the scoped View schema
// (columns + enum literals), and it routes the SAME validate -> rewrite -> execute scope
// enforcement the CLI runs — a scoped query executes, and every escape (base target,
// hidden column, out-of-enum WHERE value) is refused as data, never leaked rows.

// scopedTrajView is the operator-published view the round-trip tests query through.
func scopedTrajView() map[string]any {
	return map[string]any{
		"name":    "myturns",
		"base":    "turns",
		"columns": []string{"id", "role", "status"},
		"enums": map[string][]string{
			"role":   {"agent", "user", "system"},
			"status": {"ok", "error"},
		},
		// role != 'system' is ANDed into every query and cannot be removed by the caller.
		"scope": []map[string]any{{"field": "role", "op": "!=", "value": "system"}},
	}
}

// trajCorpus is the trajectory corpus. `secret` is a hidden column (not allowlisted); the
// system row must be excluded by the non-removable scope.
func trajCorpus() []map[string]any {
	return []map[string]any{
		{"id": "1", "role": "agent", "status": "ok", "secret": "x"},
		{"id": "2", "role": "user", "status": "ok", "secret": "y"},
		{"id": "3", "role": "system", "status": "error", "secret": "z"},
	}
}

func TestMCPTrajQueryToolAdvertisesScopedViewSchema(t *testing.T) {
	// The descriptor is present in the live tool list...
	found := false
	for _, td := range toolDescriptors() {
		if td["name"] == "fak_trajquery" {
			found = true
		}
	}
	if !found {
		t.Fatalf("fak_trajquery missing from toolDescriptors()")
	}

	// ...its description states the scoped-view contract...
	desc, _ := trajqueryToolDescriptor()["description"].(string)
	for _, want := range []string{"columns", "enums", "scope escape", "scoped view"} {
		if !strings.Contains(strings.ToLower(desc), want) {
			t.Fatalf("fak_trajquery description missing %q; got: %s", want, desc)
		}
	}

	// ...and its inputSchema advertises the View's columns + enum literals so a client can
	// build a legal query without probing.
	schema, _ := trajqueryToolDescriptor()["inputSchema"].(json.RawMessage)
	var parsed struct {
		Properties struct {
			View struct {
				Properties map[string]json.RawMessage `json:"properties"`
			} `json:"view"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(schema, &parsed); err != nil {
		t.Fatalf("inputSchema is not valid JSON: %v", err)
	}
	for _, want := range []string{"name", "base", "columns", "enums", "scope"} {
		if _, ok := parsed.Properties.View.Properties[want]; !ok {
			t.Fatalf("inputSchema view schema missing %q property; got keys %v", want, trajSchemaKeys(parsed.Properties.View.Properties))
		}
	}
}

func TestMCPTrajQueryDispatchRegistered(t *testing.T) {
	srv := newTestServer(t)
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_trajquery","arguments":{"view":{"name":"myturns","base":"turns","columns":["id"],"scope":[{"column":"role","equals":"agent"}]},"sql":"SELECT id FROM myturns","corpus":[]}}`))
	if rpcErr != nil {
		t.Fatalf("fak_trajquery dispatch missing: %v", rpcErr)
	}
	if result == nil {
		t.Fatal("fak_trajquery dispatch returned nil result")
	}
}

func TestMCPTrajQueryScopedQueryExecutes(t *testing.T) {
	srv := newTestServer(t)
	resp := callMCPTool[TrajQueryResponse](t, srv, "fak_trajquery", map[string]any{
		"view":   scopedTrajView(),
		"sql":    "SELECT id, status FROM myturns WHERE status = 'ok'",
		"corpus": trajCorpus(),
	})
	if !resp.Valid {
		t.Fatalf("scoped query refused: %v", resp.Violations)
	}
	// Two ok rows survive; the system row is dropped by the non-removable scope.
	if len(resp.Rows) != 2 {
		t.Fatalf("got %d rows, want 2: %+v", len(resp.Rows), resp.Rows)
	}
	for _, r := range resp.Rows {
		if _, leaked := r["secret"]; leaked {
			t.Fatalf("hidden column leaked into projection: %+v", r)
		}
		if _, ok := r["id"]; !ok {
			t.Fatalf("projected row missing id: %+v", r)
		}
	}
	// The scope-enforced rewritten query is echoed for audit and carries the scope.
	if !strings.Contains(resp.Rewritten, "role != 'system'") {
		t.Fatalf("rewritten query dropped the scope: %q", resp.Rewritten)
	}
	// The schema is echoed back so the caller always sees what bounded the query.
	if got := resp.Schema.Columns; strings.Join(got, ",") != "id,role,status" {
		t.Fatalf("echoed schema columns = %v, want [id role status]", got)
	}
	if got := resp.Schema.Enums["status"]; strings.Join(got, ",") != "ok,error" {
		t.Fatalf("echoed schema status enum = %v, want [ok error]", got)
	}
}

func TestMCPTrajQueryRefusesScopeEscapes(t *testing.T) {
	srv := newTestServer(t)
	cases := []struct {
		name string
		sql  string
	}{
		{"query targets the base relation", "SELECT id FROM turns"},
		{"projects a hidden column", "SELECT secret FROM myturns"},
		{"WHERE on a hidden column", "SELECT id FROM myturns WHERE secret = 'x'"},
		{"WHERE value outside the enum literals", "SELECT id FROM myturns WHERE status = 'pending'"},
		{"LIKE-probe of an enum column", "SELECT id FROM myturns WHERE status LIKE 'o'"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := callMCPTool[TrajQueryResponse](t, srv, "fak_trajquery", map[string]any{
				"view":   scopedTrajView(),
				"sql":    tc.sql,
				"corpus": trajCorpus(),
			})
			if resp.Valid {
				t.Fatalf("scope escape %q was ADMITTED", tc.sql)
			}
			if len(resp.Violations) == 0 {
				t.Fatalf("refusal carried no violations for %q", tc.sql)
			}
			if len(resp.Rows) != 0 {
				t.Fatalf("refused query leaked %d rows: %+v", len(resp.Rows), resp.Rows)
			}
		})
	}
}

func TestMCPTrajQueryEnumLiteralUnderLikeIsAllowed(t *testing.T) {
	srv := newTestServer(t)
	// A LIKE whose value is a FULL declared literal is legal (documents-the-value use),
	// while a partial substring probe (the previous test) is refused.
	resp := callMCPTool[TrajQueryResponse](t, srv, "fak_trajquery", map[string]any{
		"view":   scopedTrajView(),
		"sql":    "SELECT id FROM myturns WHERE status LIKE 'ok'",
		"corpus": trajCorpus(),
	})
	if !resp.Valid {
		t.Fatalf("enum-literal LIKE refused: %v", resp.Violations)
	}
	if len(resp.Rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(resp.Rows))
	}
}

func TestMCPTrajQueryValidateOnlySkipsExecution(t *testing.T) {
	srv := newTestServer(t)
	resp := callMCPTool[TrajQueryResponse](t, srv, "fak_trajquery", map[string]any{
		"view":          scopedTrajView(),
		"sql":           "SELECT * FROM myturns",
		"corpus":        trajCorpus(),
		"validate_only": true,
	})
	if !resp.Valid {
		t.Fatalf("validate_only refused a sound query: %v", resp.Violations)
	}
	if len(resp.Rows) != 0 {
		t.Fatalf("validate_only returned %d rows, want 0", len(resp.Rows))
	}
	if resp.Rewritten == "" {
		t.Fatalf("validate_only should still echo the rewritten query")
	}
}

func trajSchemaKeys(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
