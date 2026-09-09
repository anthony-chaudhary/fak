package gateway

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/hil"
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
			"fak://server/tools",
			"fak gateway tool catalog",
			"machine-readable tool catalog: exposed tools with names, descriptions, and input schemas",
			func(s *Server) any {
				return map[string]any{
					"tools": s.exposedToolDescriptors(),
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
		jsonMCPResource(
			"fak://hardware/inventory",
			"fak hardware inventory",
			"machine-readable dynamic hardware inventory: local silicon topology and LAN compute node readiness",
			func(s *Server) any {
				return s.getHardwareInventoryReport()
			},
		),
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
	if cleanURI == "fak://tools" || strings.HasPrefix(p.URI, "fak://tools?") || strings.HasPrefix(p.URI, "fak://server/tools?") {
		p.URI = "fak://server/tools"
	}
	if cleanURI == "fak://hardware/inventory" || strings.HasPrefix(p.URI, "fak://hardware/inventory?") {
		p.URI = "fak://hardware/inventory"
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

// Dynamic hardware inventory & MCP resource subscription management (#12476).

type mcpHardwareState struct {
	subMu sync.Mutex
	subs  map[string]map[string]bool                 // uri -> set of peerIDs
	sinks map[string]func(method string, params any) // peerID -> sink

	invMu     sync.RWMutex
	storedHw  hil.InventoryReport
	hasReport bool
}

var (
	mcpHwStateMu sync.Mutex
	mcpHwStates  = make(map[*Server]*mcpHardwareState)
)

func (s *Server) getMCPHardwareState() *mcpHardwareState {
	mcpHwStateMu.Lock()
	defer mcpHwStateMu.Unlock()
	st := mcpHwStates[s]
	if st == nil {
		st = &mcpHardwareState{
			subs:  make(map[string]map[string]bool),
			sinks: make(map[string]func(method string, params any)),
		}
		mcpHwStates[s] = st
	}
	return st
}

type mcpPeerKey struct{}

func withMCPPeer(ctx context.Context, peerID string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, mcpPeerKey{}, peerID)
}

func mcpPeerFrom(ctx context.Context) string {
	if ctx == nil {
		return "default"
	}
	if sid, ok := ctx.Value(mcpPeerKey{}).(string); ok && sid != "" {
		return sid
	}
	return "default"
}

type mcpResourceSubscribeParams struct {
	URI string `json:"uri"`
}

func (s *Server) handleResourceSubscribe(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	var p mcpResourceSubscribeParams
	if e := mcpUnmarshalParams(params, &p, "resources/subscribe"); e != nil {
		return nil, e
	}
	if p.URI == "" {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "missing uri"}
	}
	cleanURI := p.URI
	if idx := strings.Index(cleanURI, "?"); idx != -1 {
		cleanURI = cleanURI[:idx]
	}
	cleanURI = strings.TrimRight(cleanURI, "/")
	if cleanURI != "fak://hardware/inventory" && cleanURI != "fak://server/capabilities" && cleanURI != "fak://server/tools" && cleanURI != mcpCacheSemanticsURI {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown resource: " + p.URI}
	}
	s.subscribeResource(mcpPeerFrom(ctx), cleanURI)
	return map[string]any{}, nil
}

func (s *Server) handleResourceUnsubscribe(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	var p mcpResourceSubscribeParams
	if e := mcpUnmarshalParams(params, &p, "resources/unsubscribe"); e != nil {
		return nil, e
	}
	if p.URI == "" {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "missing uri"}
	}
	cleanURI := p.URI
	if idx := strings.Index(cleanURI, "?"); idx != -1 {
		cleanURI = cleanURI[:idx]
	}
	cleanURI = strings.TrimRight(cleanURI, "/")
	if cleanURI != "fak://hardware/inventory" && cleanURI != "fak://server/capabilities" && cleanURI != "fak://server/tools" && cleanURI != mcpCacheSemanticsURI {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown resource: " + p.URI}
	}
	s.unsubscribeResource(mcpPeerFrom(ctx), cleanURI)
	return map[string]any{}, nil
}

func (s *Server) subscribeResource(peerID, uri string) {
	st := s.getMCPHardwareState()
	st.subMu.Lock()
	defer st.subMu.Unlock()
	if st.subs[uri] == nil {
		st.subs[uri] = make(map[string]bool)
	}
	st.subs[uri][peerID] = true
}

func (s *Server) unsubscribeResource(peerID, uri string) {
	st := s.getMCPHardwareState()
	st.subMu.Lock()
	defer st.subMu.Unlock()
	if st.subs[uri] != nil {
		delete(st.subs[uri], peerID)
	}
}

// RegisterMCPNotificationSink registers a sink callback for the given peer ID.
// Returns an unregister function.
func (s *Server) RegisterMCPNotificationSink(peerID string, sink func(method string, params any)) func() {
	st := s.getMCPHardwareState()
	st.subMu.Lock()
	defer st.subMu.Unlock()
	st.sinks[peerID] = sink
	return func() {
		st.subMu.Lock()
		defer st.subMu.Unlock()
		delete(st.sinks, peerID)
		for uri := range st.subs {
			delete(st.subs[uri], peerID)
		}
	}
}

func (s *Server) notifySubscribedClients(uri string) {
	st := s.getMCPHardwareState()
	st.subMu.Lock()
	defer st.subMu.Unlock()
	peers := st.subs[uri]
	if len(peers) == 0 {
		return
	}
	for pid := range peers {
		if sink, ok := st.sinks[pid]; ok && sink != nil {
			fn := sink
			go fn("notifications/resources/updated", map[string]any{
				"uri": uri,
			})
		}
	}
}

func (s *Server) getHardwareInventoryReport() hil.InventoryReport {
	st := s.getMCPHardwareState()
	st.invMu.Lock()
	defer st.invMu.Unlock()
	if st.hasReport {
		return st.storedHw
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	lanHost := os.Getenv("FAK_STRIX_HOST")
	if lanHost == "" {
		lanHost = os.Getenv("FAK_LAN_HOST")
	}
	report := hil.ProbeInventory(ctx, lanHost)
	report = scrubInventoryReport(report)
	st.storedHw = report
	st.hasReport = true
	return st.storedHw
}

// SetHardwareInventory overrides the cached hardware inventory report without emitting notifications.
func (s *Server) SetHardwareInventory(report hil.InventoryReport) {
	scrubbed := scrubInventoryReport(report)
	st := s.getMCPHardwareState()
	st.invMu.Lock()
	st.storedHw = scrubbed
	st.hasReport = true
	st.invMu.Unlock()
}

// UpdateHardwareInventory updates the cached hardware inventory report.
// If a hardware status transition is detected compared to the previous cached report,
// it emits notifications/resources/updated to subscribed clients. Returns true if transition occurred.
func (s *Server) UpdateHardwareInventory(report hil.InventoryReport) bool {
	scrubbed := scrubInventoryReport(report)
	st := s.getMCPHardwareState()

	st.invMu.Lock()
	hadReport := st.hasReport
	prev := st.storedHw
	st.storedHw = scrubbed
	st.hasReport = true
	st.invMu.Unlock()

	transition := false
	if hadReport {
		transition = detectHardwareTransition(prev, scrubbed)
	} else {
		transition = true
	}

	if transition {
		s.notifySubscribedClients("fak://hardware/inventory")
	}
	return transition
}

// ProbeAndRefreshHardwareInventory probes live hardware and updates the inventory cache,
// emitting notifications if a status transition is detected.
func (s *Server) ProbeAndRefreshHardwareInventory(ctx context.Context, lanHost string) (hil.InventoryReport, bool) {
	rep := hil.ProbeInventory(ctx, lanHost)
	transitioned := s.UpdateHardwareInventory(rep)
	return s.getHardwareInventoryReport(), transitioned
}

func detectHardwareTransition(prev, curr hil.InventoryReport) bool {
	if prev.LAN.Status != curr.LAN.Status {
		return true
	}
	if prev.LAN.Reachable != curr.LAN.Reachable {
		return true
	}
	if prev.LAN.Host != curr.LAN.Host {
		return true
	}
	if prev.LAN.Transport != curr.LAN.Transport {
		return true
	}
	if prev.LAN.Endpoint != curr.LAN.Endpoint {
		return true
	}
	if prev.LAN.Appliance != curr.LAN.Appliance {
		return true
	}
	if prev.LAN.GPU != curr.LAN.GPU {
		return true
	}
	if prev.Local.PhysicalAvailable != curr.Local.PhysicalAvailable {
		return true
	}
	if prev.Local.Kind != curr.Local.Kind {
		return true
	}
	if prev.Local.DeviceName != curr.Local.DeviceName {
		return true
	}
	if prev.Local.MemoryTotalBytes != curr.Local.MemoryTotalBytes {
		return true
	}
	if prev.Local.MemoryUnified != curr.Local.MemoryUnified {
		return true
	}
	return false
}

var (
	lanIPv4Re       = regexp.MustCompile(`\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b`)
	lanIPv6Re       = regexp.MustCompile(`\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|::1`)
	sensitiveHostRe = regexp.MustCompile(`(?i)\b[a-zA-Z0-9_\-]*(?:strix-halo|strix-agent|dgx|gpu-server|lab-)[a-zA-Z0-9_\-\.]*\b|[a-zA-Z0-9_\-]+\.(?:local|internal|lan|corp|lab|home)\b`)
)

func isInternalOrSensitiveHost(h string) bool {
	h = strings.TrimSpace(h)
	if h == "" {
		return false
	}
	if h == "<LAN_IP>" || strings.Contains(h, "<LAN_IP>") {
		return true
	}
	hostPart := h
	if sh, _, err := net.SplitHostPort(h); err == nil {
		hostPart = sh
	}
	if ip := net.ParseIP(hostPart); ip != nil {
		return isInternalIP(ip)
	}
	if lanIPv4Re.MatchString(hostPart) || lanIPv6Re.MatchString(hostPart) {
		return true
	}
	lower := strings.ToLower(hostPart)
	if lower == "localhost" {
		return true
	}
	for _, suffix := range []string{".local", ".internal", ".lan", ".corp", ".lab", ".home"} {
		if strings.HasSuffix(lower, suffix) {
			return true
		}
	}
	for _, prefix := range []string{"strix-halo", "strix-agent", "dgx", "gpu-server", "lab-"} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

func isInternalIP(ip net.IP) bool {
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return true
	}
	if v4 := ip.To4(); len(v4) == 4 {
		if v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127 {
			return true
		}
	}
	return false
}

func sanitizeEndpoint(rawEndpoint, alias string) string {
	if rawEndpoint == "" {
		return ""
	}
	scheme := ""
	rest := rawEndpoint
	if idx := strings.Index(rawEndpoint, "://"); idx != -1 {
		scheme = rawEndpoint[:idx+3]
		rest = rawEndpoint[idx+3:]
	}
	hostPart := rest
	portPart := ""
	if idx := strings.LastIndex(rest, ":"); idx != -1 && !strings.Contains(rest[idx:], "/") {
		hostPart = rest[:idx]
		portPart = rest[idx:]
	}
	if isInternalOrSensitiveHost(hostPart) || hostPart == alias {
		return scheme + alias + portPart
	}
	if lanIPv4Re.MatchString(hostPart) || lanIPv6Re.MatchString(hostPart) || strings.Contains(hostPart, "<LAN_IP>") || sensitiveHostRe.MatchString(hostPart) {
		return scheme + alias + portPart
	}
	return rawEndpoint
}

func sanitizeText(text, alias string) string {
	if text == "" {
		return ""
	}
	text = strings.ReplaceAll(text, "<LAN_IP>", alias)
	text = lanIPv4Re.ReplaceAllString(text, alias)
	text = lanIPv6Re.ReplaceAllString(text, alias)
	text = sensitiveHostRe.ReplaceAllString(text, alias)
	return text
}

func scrubInventoryReport(rep hil.InventoryReport) hil.InventoryReport {
	res := rep
	res.Sanitized = true

	// 1. Scrub Local
	if isInternalOrSensitiveHost(res.Local.DeviceName) || res.Local.DeviceName == "" {
		res.Local.DeviceName = "local-silicon"
	} else if lanIPv4Re.MatchString(res.Local.DeviceName) || lanIPv6Re.MatchString(res.Local.DeviceName) || sensitiveHostRe.MatchString(res.Local.DeviceName) {
		res.Local.DeviceName = "local-silicon"
	}
	if res.Local.Details != nil {
		newDetails := make(map[string]string, len(res.Local.Details))
		for k, v := range res.Local.Details {
			if isInternalOrSensitiveHost(v) {
				newDetails[k] = "local-silicon"
			} else {
				newDetails[k] = sanitizeText(v, "local-silicon")
			}
		}
		res.Local.Details = newDetails
	}

	// 2. Scrub LAN
	if res.LAN.Host != "" {
		if isInternalOrSensitiveHost(res.LAN.Host) || res.LAN.Host == "<LAN_IP>" {
			res.LAN.Host = "strix1"
		}
	}
	if res.LAN.Endpoint != "" {
		res.LAN.Endpoint = sanitizeEndpoint(res.LAN.Endpoint, "strix1")
	}
	if res.LAN.Error != "" {
		res.LAN.Error = sanitizeText(res.LAN.Error, "strix1")
	}

	// 3. Update Local.LANNode to match LAN
	if res.Local.LANNode != nil {
		lanCopy := res.LAN
		res.Local.LANNode = &lanCopy
	}

	// 4. Scrub NextAction
	if res.NextAction != "" {
		res.NextAction = sanitizeText(res.NextAction, "strix1")
	}

	return res
}
