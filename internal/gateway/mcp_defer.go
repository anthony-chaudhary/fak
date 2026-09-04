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
// Default ON: the recall, first-call, and quarantine gates are now witnessed.
// FAK_ABLATE_MCP_TOOL_FILTER=1 restores the full list immediately. Filtering
// also bails out fail-open when fak_tools_search is not exposed, and every
// tools/list response reports the applied/bypass reason plus measured bytes.

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// DefaultMCPToolAdvertisementCeiling is the default ceiling on advertised MCP tools
// when deferral is disabled, clamping the schema footprint to a curated active set.
const DefaultMCPToolAdvertisementCeiling = 10

// MaxMCPToolAdvertisementCeiling is the maximum ceiling enforced on tools/list
// to prevent clients from being flooded with 40+ full tool schemas unless explicitly requested.
const MaxMCPToolAdvertisementCeiling = 40

// curatedPriorityToolNames defines the ordered precedence for selecting the active
// top-K tool set when an advertisement ceiling is in effect.
var curatedPriorityToolNames = []string{
	"fak_adjudicate",
	"fak_syscall",
	"fak_read",
	"fak_tools_search",
	"fak_context_change",
	"fak_context_restore",
	"fak_feature_query",
	"fak_trajquery",
	"fak_memory_drivers",
	"fak_memory_explain",
}

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

// MCPToolFilterStatus is the machine-readable proof attached to tools/list.
// DescriptorBytesBefore/After price the exact JSON descriptor arrays, not a
// tokenizer estimate; SavedBytes is therefore provider-neutral and auditable.
type MCPToolFilterStatus struct {
	Mode                  string `json:"mode"`
	Reason                string `json:"reason"`
	ToolsBefore           int    `json:"tools_before"`
	ToolsAfter            int    `json:"tools_after"`
	DescriptorBytesBefore int    `json:"descriptor_bytes_before"`
	DescriptorBytesAfter  int    `json:"descriptor_bytes_after"`
	SavedBytes            int    `json:"saved_bytes"`
}

// toolsListView returns resident descriptors and an operator-visible receipt.
// Native filtering is default-on. It fails OPEN to the full registry when the
// recovery tool is hidden, no cold tail exists, or emergency ablation is set:
// a bad optimization must cost tokens, never capabilities.
func (s *Server) toolsListView() ([]map[string]any, MCPToolFilterStatus) {
	return s.toolsListViewWithAblation(s.disableMCPDefer || envEnabled("FAK_ABLATE_MCP_TOOL_FILTER"))
}

func (s *Server) toolsListViewWithAblation(ablate bool) ([]map[string]any, MCPToolFilterStatus) {
	full := s.exposedToolDescriptors()
	resident := full
	status := MCPToolFilterStatus{Mode: "bypass", Reason: "ablation"}

	if !ablate {
		hasRecovery := false
		for _, td := range full {
			if td["name"] == "fak_tools_search" {
				hasRecovery = true
				break
			}
		}
		if !hasRecovery {
			status.Reason = "recovery_tool_hidden"
		} else if bootstrap := s.bootstrapToolDescriptors(); len(bootstrap) >= len(full) {
			status.Reason = "no_cold_tools"
		} else {
			resident = bootstrap
			status.Mode = "active"
			status.Reason = "default_on"
		}
	} else if s != nil && s.mcpToolCeiling > 0 {
		if len(full) > s.mcpToolCeiling {
			resident = s.curatedCeilingToolDescriptors(full, s.mcpToolCeiling)
		}
		status.Mode = "ceiling"
		status.Reason = "advertisement_ceiling"
	} else if s != nil && s.mcpToolCeiling == 0 && len(full) > MaxMCPToolAdvertisementCeiling {
		resident = s.curatedCeilingToolDescriptors(full, MaxMCPToolAdvertisementCeiling)
		status.Mode = "ceiling"
		status.Reason = "advertisement_ceiling_capped"
	}

	before, _ := json.Marshal(full)
	after, _ := json.Marshal(resident)
	status.ToolsBefore = len(full)
	status.ToolsAfter = len(resident)
	status.DescriptorBytesBefore = len(before)
	status.DescriptorBytesAfter = len(after)
	status.SavedBytes = len(before) - len(after)
	return resident, status
}

// MCPToolListSnapshot returns the public tools/list view and its aggregate
// receipt for an explicitly selected A/B arm. It does not mutate process
// environment, and the descriptors are the same public data served over MCP.
func (s *Server) MCPToolListSnapshot(fullListControl bool) ([]map[string]any, MCPToolFilterStatus) {
	return s.toolsListViewWithAblation(fullListControl)
}

// MCPToolFilterStatusSnapshot reports the same privacy-safe receipt emitted on
// tools/list, for the unified /debug/vars operator surface.
func (s *Server) MCPToolFilterStatusSnapshot() MCPToolFilterStatus {
	_, status := s.toolsListView()
	return status
}

// MCPToolSearchSnapshot exposes the same full-registry ranked discovery path as
// fak_tools_search for offline benchmarks and operator diagnostics. It returns
// descriptors only; calls still cross the normal MCP guard.
func (s *Server) MCPToolSearchSnapshot(query, detail string) (ToolsSearchResponse, error) {
	return s.toolsSearch(ToolsSearchRequest{Query: query, DetailLevel: detail})
}

// NativeMCPFilterProof evaluates the default arm against the full-list control
// over a held intent corpus. It is deterministic and offline: task success is
// defined as discovering the required tool, then confirming the same tool route
// exists; no model narration is accepted as evidence.
type NativeMCPFilterProof struct {
	Schema                  string                     `json:"schema"`
	Verdict                 string                     `json:"verdict"`
	Active                  MCPToolFilterStatus        `json:"active"`
	Control                 MCPToolFilterStatus        `json:"control"`
	Tasks                   []NativeMCPFilterProofTask `json:"tasks"`
	TaskSuccessRate         float64                    `json:"task_success_rate"`
	SearchRecall            float64                    `json:"search_recall"`
	FirstCallRouteSuccess   float64                    `json:"first_call_route_success"`
	SecurityParityConfirmed bool                       `json:"security_parity_confirmed"`
}

type NativeMCPFilterProofTask struct {
	ID           string `json:"id"`
	Query        string `json:"query"`
	RequiredTool string `json:"required_tool"`
	SearchHit    bool   `json:"search_hit"`
	RouteExists  bool   `json:"route_exists"`
	Success      bool   `json:"success"`
}

func (s *Server) NativeMCPFilterProof() NativeMCPFilterProof {
	_, active := s.toolsListView()
	full := s.exposedToolDescriptors()
	control := active
	control.Mode, control.Reason = "bypass", "full_list_control"
	control.ToolsAfter = len(full)
	control.DescriptorBytesAfter = control.DescriptorBytesBefore
	control.SavedBytes = 0
	cases := []struct{ id, query, tool string }{
		{"memory", "memory drivers", "fak_memory_drivers"},
		{"context", "change context budget", "fak_context_change"},
		{"features", "query available features", "fak_feature_query"},
		{"restore", "restore dropped context", "fak_context_restore"},
		{"trajectory", "trajectory SQL scoped view", "fak_trajquery"},
	}
	routes := make(map[string]bool, len(full))
	for _, td := range full {
		if n, _ := td["name"].(string); n != "" {
			routes[n] = true
		}
	}
	p := NativeMCPFilterProof{Schema: "fak-native-mcp-filter-proof/1", Active: active, Control: control}
	for _, tc := range cases {
		names := s.rankToolNamesByIntent(tc.query)
		hit := len(names) > 0 && names[0] == tc.tool
		route := routes[tc.tool]
		p.Tasks = append(p.Tasks, NativeMCPFilterProofTask{ID: tc.id, Query: tc.query, RequiredTool: tc.tool, SearchHit: hit, RouteExists: route, Success: hit && route})
		if hit {
			p.SearchRecall++
		}
		if route {
			p.FirstCallRouteSuccess++
		}
		if hit && route {
			p.TaskSuccessRate++
		}
	}
	n := float64(len(p.Tasks))
	p.SearchRecall /= n
	p.FirstCallRouteSuccess /= n
	p.TaskSuccessRate /= n
	oldExpose := s.exposeAllow
	s.exposeAllow = func(name string) bool { return name != "fak_memory_run" }
	hiddenNames := s.rankToolNamesByIntent("run memory compaction")
	hiddenRoute := false
	for _, td := range s.exposedToolDescriptors() {
		if td["name"] == "fak_memory_run" {
			hiddenRoute = true
		}
	}
	s.exposeAllow = oldExpose
	p.SecurityParityConfirmed = !hiddenRoute
	for _, name := range hiddenNames {
		if name == "fak_memory_run" {
			p.SecurityParityConfirmed = false
		}
	}
	if active.Mode == "active" && active.SavedBytes > 0 && p.TaskSuccessRate == 1 && p.SecurityParityConfirmed {
		p.Verdict = "PASS"
	} else {
		p.Verdict = "NOT_YET"
	}
	return p
}

func (s *Server) toolsListDescriptors() []map[string]any {
	tools, _ := s.toolsListView()
	return tools
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

// curatedCeilingToolDescriptors selects a curated top-K active set of tools up to ceiling.
// It prioritizes the core tools (bootstrapToolNames), then common tools
// (fak_context_change, fak_context_restore, fak_feature_query, fak_trajquery,
// fak_memory_drivers, fak_memory_explain, etc.), followed by any remaining exposed tools.
func (s *Server) curatedCeilingToolDescriptors(full []map[string]any, ceiling int) []map[string]any {
	if ceiling <= 0 || len(full) <= ceiling {
		return full
	}
	out := make([]map[string]any, 0, ceiling)
	byName := make(map[string]map[string]any, len(full))
	for _, td := range full {
		if n, _ := td["name"].(string); n != "" {
			byName[n] = td
		}
	}
	seen := make(map[string]bool, ceiling)
	for _, name := range curatedPriorityToolNames {
		if td, ok := byName[name]; ok && !seen[name] {
			out = append(out, td)
			seen[name] = true
			if len(out) == ceiling {
				return out
			}
		}
	}
	for _, td := range full {
		n, _ := td["name"].(string)
		if bootstrapToolNames[n] && !seen[n] {
			out = append(out, td)
			seen[n] = true
			if len(out) == ceiling {
				return out
			}
		}
	}
	for _, td := range full {
		n, _ := td["name"].(string)
		if !seen[n] {
			out = append(out, td)
			seen[n] = true
			if len(out) == ceiling {
				return out
			}
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
	resp, err := cat.Query(selfquery.Request{Query: query, Plane: selfquery.PlaneLive, All: true})
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
