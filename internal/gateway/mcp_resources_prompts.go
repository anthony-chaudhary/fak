package gateway

import "encoding/json"

// MCP ecosystem surface beyond the tool registry (#213). The gateway already
// serves the "tools" primitive (tools/list + tools/call = tool auto-discovery);
// this file adds the other two standard MCP primitives so the kernel is a
// fuller MCP server, not just an adjudication tool endpoint:
//
//	resources/list + resources/read  — readable, URI-addressed content
//	prompts/list    + prompts/get    — reusable server-provided prompt templates
//
// Both are advertised in the initialize capabilities (see initializeResult), so
// a spec-compliant client knows to call them. The content is DERIVED from live
// server state (the tool catalog, the negotiated protocol revisions, the running
// version) — never hand-stamped — so a resource read can never drift from what
// the server actually offers.

// mcpResource is one server-provided resource. build is evaluated at read time
// against live server state, so the bytes a client reads always reflect the
// running server (its version, its current tool catalog), not a frozen snapshot.
type mcpResource struct {
	uri   string
	name  string
	desc  string
	mime  string
	build func(s *Server) string
}

// resources is the resource registry. Adding a resource is one literal here; the
// list/read handlers and the advertised capability need no further edit. Today it
// holds the kernel's machine-readable self-description — the one document a
// discovering MCP client reads to learn the server name/version, which MCP
// revisions it speaks, and the full tool catalog in one fetch.
func (s *Server) resources() []mcpResource {
	return []mcpResource{
		{
			uri:  "fak://server/capabilities",
			name: "fak gateway capabilities",
			desc: "machine-readable self-description: server name/version, the MCP protocol revisions this server speaks, and the full tool catalog with descriptions",
			mime: "application/json",
			build: func(s *Server) string {
				doc := map[string]any{
					"name":             "fak-gateway",
					"version":          s.version,
					"protocolVersions": mcpProtocolVersions,
					"tools":            toolCatalogSummary(),
				}
				b, _ := json.Marshal(doc)
				return string(b)
			},
		},
	}
}

// resourceDescriptors is the resources/list payload: {uri, name, description,
// mimeType} per resource (no content — the client fetches that via resources/read).
func (s *Server) resourceDescriptors() []map[string]any {
	rs := s.resources()
	out := make([]map[string]any, 0, len(rs))
	for _, r := range rs {
		out = append(out, map[string]any{
			"uri":         r.uri,
			"name":        r.name,
			"description": r.desc,
			"mimeType":    r.mime,
		})
	}
	return out
}

// readResource handles resources/read. params is {uri}; the response is the MCP
// {contents:[{uri, mimeType, text}]} shape. An unknown URI is a parameter fault
// (InvalidParams), the same convention this file's tools/call uses for an unknown
// tool — not a JSON-RPC internal error.
func (s *Server) readResource(params json.RawMessage) (any, *rpcError) {
	var p struct {
		URI string `json:"uri"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid resources/read params: " + err.Error()}
	}
	for _, r := range s.resources() {
		if r.uri == p.URI {
			return map[string]any{
				"contents": []map[string]any{{
					"uri":      r.uri,
					"mimeType": r.mime,
					"text":     r.build(s),
				}},
			}, nil
		}
	}
	return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown resource: " + p.URI}
}

// toolCatalogSummary projects the tools/list descriptors down to {name,
// description} — the same source of truth the tool registry serves, so the
// capabilities resource can never list a tool the server does not actually offer.
func toolCatalogSummary() []map[string]string {
	tds := toolDescriptors()
	out := make([]map[string]string, 0, len(tds))
	for _, td := range tds {
		name, _ := td["name"].(string)
		desc, _ := td["description"].(string)
		out = append(out, map[string]string{"name": name, "description": desc})
	}
	return out
}

// promptDescriptors is the prompts/list payload. fak_guarded_call is the canonical
// adjudication workflow as a reusable template: a client (or a user via a slash
// command) can instantiate it to make the assistant route a risky call through
// fak_adjudicate before executing.
func promptDescriptors() []map[string]any {
	return []map[string]any{
		{
			"name":        "fak_guarded_call",
			"description": "Wrap a proposed tool call in the fak adjudication workflow: call fak_adjudicate first and obey the verdict before executing.",
			"arguments": []map[string]any{
				{"name": "tool", "description": "the tool the assistant intends to call", "required": true},
				{"name": "task", "description": "optional task context to carry into the guarded turn", "required": false},
			},
		},
	}
}

// getPrompt handles prompts/get. params is {name, arguments}; the response is the
// MCP {description, messages:[{role, content:{type,text}}]} shape. The template is
// expanded against the live tool vocabulary so the guidance names the real verdict
// kinds the kernel returns.
func (s *Server) getPrompt(params json.RawMessage) (any, *rpcError) {
	var p struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid prompts/get params: " + err.Error()}
	}
	switch p.Name {
	case "fak_guarded_call":
		tool := p.Arguments["tool"]
		if tool == "" {
			tool = "the tool"
		}
		text := "Before executing " + tool + ", call the fak_adjudicate tool with the proposed " +
			"{tool, arguments} and obey the returned verdict: ALLOW — run it; DENY — do not run it and " +
			"follow the disposition (RETRYABLE/WAIT/ESCALATE/TERMINAL); TRANSFORM — run the repaired " +
			"arguments fak returns; REQUIRE_WITNESS — supply the witness fak asks for, then retry."
		if task := p.Arguments["task"]; task != "" {
			text += "\n\nTask: " + task
		}
		return map[string]any{
			"description": "fak-guarded tool-call workflow",
			"messages": []map[string]any{{
				"role":    "user",
				"content": map[string]any{"type": "text", "text": text},
			}},
		}, nil
	default:
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown prompt: " + p.Name}
	}
}
