package gateway

// mcp_defer.go — schema-light tools/list for fak's OWN MCP server (#3231, epic
// #3229). fak's hand-rolled MCP server advertises all ~24 fak_* schemas on every
// tools/list — the ~9.9k MCP slice of the always-sent floor. When deferral is
// enabled, tools/list returns only a small BOOTSTRAP set (the hot core + the
// search tool); the cold schemas are absent from the resident floor and faulted
// in on demand by fak_tools_search, which ranks the FULL registry through the
// selfquery catalog (the #3235 hybrid ranker — recall is the failure mode).
//
// Two explicit views, per the issue:
//   - toolsListDescriptors: the RESIDENT bootstrap view (tools/list).
//   - exposedToolDescriptors: the FULL retrieval view (the search tool + the
//     other discovery surfaces). tools/call routes EVERY registered tool,
//     deferred or not — deferral hides the schema, never the route or the guard.
//
// Deferral composes UNDER --expose: it filters the already-exposed set, so a
// tool hidden by the allowlist never reappears in the bootstrap or the search.
//
// Default OFF (Config.DeferMCPTools / FAK_DEFER_MCP_TOOLS): the floor reduction
// depends on the client re-finding a searched tool and on the pin/quarantine
// guard (#3200), so flipping the default on is gated on that validation. The
// mechanism is fully built and witnessed here so the flip is a one-line change.

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// bootstrapToolNames is the hot resident core kept eager on tools/list when
// deferral is on: the syscall/read/adjudicate spine plus the tool_search_tool
// entry (fak_tools_search) the model uses to fault every other schema back in.
// Deliberately tiny — the whole point is that the long tail is cold.
var bootstrapToolNames = map[string]bool{
	"fak_syscall":      true,
	"fak_read":         true,
	"fak_adjudicate":   true,
	"fak_tools_search": true,
}

// toolsListDescriptors is what tools/list returns. With deferral off (default)
// it is the full exposed registry — byte-for-byte the pre-#3231 surface. With
// deferral on it is only the bootstrap view.
func (s *Server) toolsListDescriptors() []map[string]any {
	if s != nil && s.deferMCPTools {
		return s.bootstrapToolDescriptors()
	}
	return s.exposedToolDescriptors()
}

// bootstrapToolDescriptors filters the exposed registry (so --expose is still
// honored) down to the bootstrap set. If the search tool itself was hidden by
// an allowlist, it is simply absent — deferral never surfaces a hidden tool.
func (s *Server) bootstrapToolDescriptors() []map[string]any {
	full := s.exposedToolDescriptors()
	out := make([]map[string]any, 0, len(bootstrapToolNames))
	for _, td := range full {
		if n, _ := td["name"].(string); bootstrapToolNames[n] {
			out = append(out, td)
		}
	}
	return out
}

// rankToolNamesByIntent orders the FULL exposed registry by relevance to query
// through the selfquery hybrid ranker (#3235) — the ranked recall path the
// substring matcher lacked. It returns the tool NAMES in ranked order (best
// first); callers map names back to descriptors at the requested detail level.
//
// It fails OPEN: on any catalog error it falls back to a substring filter, so a
// transient ranker problem degrades recall but never drops the search entirely
// (a schema the model cannot re-find is an invisible capability). An empty query
// returns every exposed tool name in the registry's own order.
func (s *Server) rankToolNamesByIntent(query string) []string {
	full := s.exposedToolDescriptors()
	names := make([]string, 0, len(full))
	known := make(map[string]bool, len(full))
	for _, td := range full {
		if n, _ := td["name"].(string); n != "" {
			names = append(names, n)
			known[n] = true
		}
	}
	if strings.TrimSpace(query) == "" {
		return names
	}

	cat, err := selfquery.Load("", selfquery.Options{Tools: selfquery.ToolDescriptorsFromMaps(full)})
	if err != nil {
		return substringToolFilter(full, query)
	}
	resp, err := cat.Query(selfquery.Request{Query: query, Plane: selfquery.PlaneLive})
	if err != nil {
		return substringToolFilter(full, query)
	}
	ranked := make([]string, 0, len(resp.Cards))
	seen := make(map[string]bool, len(resp.Cards))
	for _, c := range resp.Cards {
		if known[c.Name] && !seen[c.Name] {
			ranked = append(ranked, c.Name)
			seen[c.Name] = true
		}
	}
	return ranked
}

// substringToolFilter is the fail-open baseline: the pre-#3231 case-insensitive
// name+description substring match, returned in a deterministic (name-sorted)
// order so a ranker outage still yields a stable, non-empty result.
func substringToolFilter(full []map[string]any, query string) []string {
	q := strings.ToLower(strings.TrimSpace(query))
	var out []string
	for _, td := range full {
		name, _ := td["name"].(string)
		desc, _ := td["description"].(string)
		if q == "" || strings.Contains(strings.ToLower(name), q) || strings.Contains(strings.ToLower(desc), q) {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}
