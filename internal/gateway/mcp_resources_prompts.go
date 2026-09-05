package gateway

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

const (
	mcpCacheSemanticsURI    = "fak://server/cache-semantics"
	mcpCacheSemanticsSchema = "fak-mcp-cache-semantics/1"
	mcpCatalogTTLMillis     = 600000
	mcpResourceTTLMillis    = 60000
	mcpNoCacheTTLMillis     = 0
	mcpCacheScopePublic     = "public"
	mcpCacheScopePrivate    = "private"
)

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
		jsonMCPResource(
			"fak://server/capabilities",
			"fak gateway capabilities",
			"machine-readable self-description: server name/version, the MCP protocol revisions this server speaks, and the full tool catalog with descriptions",
			func(s *Server) any {
				return map[string]any{
					"name":             "fak-gateway",
					"version":          s.version,
					"protocolVersions": mcpProtocolVersions,
					"tools":            s.toolCatalogSummary(),
					"selfFeatureQuery": s.selfFeatureSummary(),
					"cacheSemantics": map[string]string{
						"resource": mcpCacheSemanticsURI,
						"schema":   mcpCacheSemanticsSchema,
					},
				}
			},
		),
		jsonMCPResource(
			mcpCacheSemanticsURI,
			"fak MCP cache semantics",
			"machine-readable cache contract for MCP clients: descriptor/resource reuse, tool-result hits, provider-prefix constraints, and invalidation verbs",
			func(s *Server) any { return mcpCacheSemanticsDoc(s.version) },
		),
		{
			uri:  contextq.MCPMissingContextURIPrefix + "{key}",
			name: "missing context demand-page template",
			desc: "read fak://context/missing/<key> to turn missing context into a bounded clarification payload before acting",
			mime: "application/json",
			build: func(s *Server) string {
				doc := map[string]any{
					"schema": contextq.MCPMissingContextSchema,
					"template": map[string]any{
						"uri":    contextq.MCPMissingContextURIPrefix + "{key}",
						"method": "resources/read",
						"reason": "missing_context",
					},
				}
				b, _ := json.Marshal(doc)
				return string(b)
			},
		},
	}
}

func jsonMCPResource(uri, name, desc string, document func(*Server) any) mcpResource {
	return mcpResource{
		uri: uri, name: name, desc: desc, mime: "application/json",
		build: func(s *Server) string {
			b, _ := json.Marshal(document(s))
			return string(b)
		},
	}
}

func mcpCacheSemanticsDoc(serverVersion string) map[string]any {
	return map[string]any{
		"schema":        mcpCacheSemanticsSchema,
		"serverVersion": serverVersion,
		"standardHints": map[string]any{
			"fields": []string{
				"ttlMs",
				"cacheScope",
			},
			"ttlMs":      "milliseconds a client may treat a result as fresh before re-fetching",
			"cacheScope": "public means shared caches may reuse the result; private means only the requesting client may reuse it",
		},
		"descriptorCache": map[string]any{
			"surfaces": []string{
				"initialize",
				"tools/list",
				"resources/list",
				"prompts/list",
				"fak://server/capabilities",
			},
			"reuseUntil": []string{
				"server_version_changes",
				"protocol_revision_changes",
				"tool_catalog_changes",
				"policy_or_feature_catalog_changes",
			},
			"economics": "tool schemas are prompt-prefix material; prefer fak_tools_search or resource reads for progressive disclosure instead of injecting every schema into every model turn",
		},
		"resourceCache": map[string]any{
			"surface":  "resources/read",
			"identity": "uri plus returned schema/digest when present",
			"sessionBoundTemplates": []string{
				contextq.MCPMissingContextURIPrefix + "<key>",
			},
			"rule": "cache durable server descriptions by URI, but treat demand pages that create audit/default-answer rows as session-bound reads",
		},
		"toolResultCache": map[string]any{
			"hitSurfaces": []string{
				"fak_syscall(read_only=true)",
				"fak_read",
			},
			"keyAxes": []string{
				"tool",
				"arguments_digest",
				"witness_or_write_epoch",
				"principal_scope",
				"policy_version",
			},
			"invalidation": []string{
				"fak_changes",
				"fak_revoke",
			},
			"admission": "tool bytes enter model-visible context only after fak_syscall or fak_admit; quarantine is a successful adjudication value, not a reusable hit",
		},
		"providerPrefixCache": map[string]any{
			"scope": "model-provider prompt cache or in-kernel KV prefix, not an MCP protocol cache",
			"stablePrefixInputs": []string{
				"system_prompt",
				"tool_descriptors",
				"prompt_templates",
				"unchanged_prefix_bytes",
			},
			"rule":      "MCP schema churn can bust provider prefix reuse; keep descriptor bytes stable and load uncommon schemas lazily",
			"guarantee": "fak can preserve byte-identical prefixes it sends, but provider cache reuse remains observed telemetry unless fak owns the KV cache",
		},
		"changeFeed": map[string]any{
			"poll":   "fak_changes",
			"refute": "fak_revoke",
			"rule":   "clients with their own MCP-side caches should consume changes before trusting a prior read-only result",
		},
	}
}

func mcpCacheHint(result map[string]any, ttlMs int, scope string) map[string]any {
	result["ttlMs"] = ttlMs
	result["cacheScope"] = scope
	return result
}

func (s *Server) selfFeatureSummary() map[string]any {
	cat, err := selfquery.Load("", selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(s.exposedToolDescriptors()),
	})
	if err != nil {
		return map[string]any{
			"tool":  "fak_feature_query",
			"ready": false,
			"error": err.Error(),
		}
	}
	return map[string]any{
		"tool":    "fak_feature_query",
		"ready":   true,
		"digest":  cat.SummaryDigest(),
		"sources": cat.Sources(),
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

// resourceTemplateDescriptors is the resources/templates/list payload:
// {uriTemplate, name, description, mimeType} per template resource (#10014).
func (s *Server) resourceTemplateDescriptors() []map[string]any {
	rs := s.resources()
	out := make([]map[string]any, 0, len(rs))
	for _, r := range rs {
		if strings.Contains(r.uri, "{") {
			out = append(out, map[string]any{
				"uriTemplate": r.uri,
				"name":        r.name,
				"description": r.desc,
				"mimeType":    r.mime,
			})
		}
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
	if e := mcpUnmarshalParams(params, &p, "resources/read"); e != nil {
		return nil, e
	}
	cleanURI := strings.TrimRight(p.URI, "/")
	if cleanURI == "fak://capabilities" || strings.HasPrefix(p.URI, "fak://capabilities?") || strings.HasPrefix(p.URI, "fak://server/capabilities?") {
		p.URI = "fak://server/capabilities"
	}
	if req, ok := contextq.MCPMissingContextResourceRequest(p.URI, 0); ok {
		plan := selfquery.MissingContextClarifications([]string{req.Key})
		audit := s.recordMissingContextQueryAudit(req, plan)
		doc := map[string]any{
			"schema":         contextq.MCPMissingContextSchema,
			"request":        req,
			"clarifications": plan,
			"audit":          audit,
		}
		b, _ := json.Marshal(doc)
		return mcpCacheHint(map[string]any{
			"contents": []map[string]any{{
				"uri":      req.URI,
				"mimeType": "application/json",
				"text":     string(b),
			}},
		}, mcpNoCacheTTLMillis, mcpCacheScopePrivate), nil
	}
	for _, r := range s.resources() {
		if r.uri == p.URI {
			return mcpCacheHint(map[string]any{
				"contents": []map[string]any{{
					"uri":      r.uri,
					"mimeType": r.mime,
					"text":     r.build(s),
				}},
			}, mcpResourceTTLMillis, mcpCacheScopePublic), nil
		}
	}
	return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown resource: " + p.URI}
}

// toolCatalogSummary projects the tools/list descriptors down to {name,
// description} — the same source of truth the tool registry serves, so the
// capabilities resource can never list a tool the server does not actually offer.
func (s *Server) toolCatalogSummary() []map[string]string {
	tds := s.exposedToolDescriptors()
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
	if e := mcpUnmarshalParams(params, &p, "prompts/get"); e != nil {
		return nil, e
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
