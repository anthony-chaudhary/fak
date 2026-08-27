package trajectory

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const AuditDistributionUnit = "utf8_bytes"

// AuditDistributionRow is deterministic transcript-payload attribution. It is not
// provider billing: providers expose exact tokens only for whole requests, not each
// message or tool block.
type AuditDistributionRow struct {
	Name  string  `json:"name"`
	Bytes int64   `json:"bytes"`
	Share float64 `json:"share"`
	Calls int     `json:"calls,omitempty"`
}

type auditDistribution struct {
	categories map[string]int64
	tools      map[string]*AuditDistributionRow
	callTools  map[string]string
}

func newAuditDistribution() auditDistribution {
	return auditDistribution{map[string]int64{}, map[string]*AuditDistributionRow{}, map[string]string{}}
}

func (d *auditDistribution) observe(source string, line []byte) {
	var root map[string]json.RawMessage
	if json.Unmarshal(line, &root) != nil {
		return
	}
	typ := rawString(root["type"])
	payload := root["payload"]
	if source == "claude" {
		payload = root["message"]
	}
	category, tool, callID := classifyDistribution(source, typ, payload, root)
	if category == "" {
		return
	}
	if category == "tool_result" && tool == "" && callID != "" {
		tool = d.callTools[callID]
	}
	weight := int64(len(line))
	d.categories[category] += weight
	if tool != "" {
		row := d.tools[tool]
		if row == nil {
			row = &AuditDistributionRow{Name: tool}
			d.tools[tool] = row
		}
		row.Bytes += weight
		if category == "tool_call" {
			row.Calls++
		}
	}
	if callID != "" && tool != "" {
		d.callTools[callID] = tool
	}
}

func classifyDistribution(source, typ string, payload json.RawMessage, root map[string]json.RawMessage) (string, string, string) {
	if source == "codex" {
		switch typ {
		case "response_item":
			var p map[string]json.RawMessage
			if json.Unmarshal(payload, &p) != nil {
				return "other", "", ""
			}
			pt := rawString(p["type"])
			switch pt {
			case "function_call", "custom_tool_call", "web_search_call":
				return "tool_call", firstNonempty(rawString(p["name"]), pt), firstNonempty(rawString(p["call_id"]), rawString(p["id"]))
			case "function_call_output", "custom_tool_call_output":
				id := firstNonempty(rawString(p["call_id"]), rawString(p["id"]))
				return "tool_result", "", id
			case "reasoning":
				return "reasoning", "", ""
			case "message":
				if rawString(p["role"]) == "user" {
					return "user_message", "", ""
				}
				return "assistant_message", "", ""
			}
		case "event_msg":
			var p map[string]json.RawMessage
			if json.Unmarshal(payload, &p) == nil {
				if rawString(p["type"]) == "user_message" {
					return "user_message", "", ""
				}
				if rawString(p["type"]) == "agent_message" {
					return "assistant_message", "", ""
				}
			}
		}
		return "other", "", ""
	}
	if typ == "user" {
		return classifyClaudeContent("user_message", payload)
	}
	if typ == "assistant" {
		return classifyClaudeContent("assistant_message", payload)
	}
	return "other", "", ""
}

func classifyClaudeContent(base string, payload json.RawMessage) (string, string, string) {
	var msg map[string]json.RawMessage
	if json.Unmarshal(payload, &msg) != nil {
		return base, "", ""
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(msg["content"], &blocks) != nil {
		return base, "", ""
	}
	for _, b := range blocks {
		switch rawString(b["type"]) {
		case "tool_use":
			return "tool_call", rawString(b["name"]), rawString(b["id"])
		case "tool_result":
			return "tool_result", "", rawString(b["tool_use_id"])
		case "thinking", "reasoning":
			return "reasoning", "", ""
		}
	}
	return base, "", ""
}

func rawString(v json.RawMessage) string { var s string; _ = json.Unmarshal(v, &s); return s }
func firstNonempty(v ...string) string {
	for _, s := range v {
		if s != "" {
			return s
		}
	}
	return ""
}

func distributionRows(m map[string]int64) []AuditDistributionRow {
	var total int64
	for _, v := range m {
		total += v
	}
	out := make([]AuditDistributionRow, 0, len(m))
	for n, v := range m {
		r := AuditDistributionRow{Name: n, Bytes: v}
		if total > 0 {
			r.Share = float64(v) / float64(total)
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Name < out[j].Name
	})
	return out
}
func toolDistributionRows(m map[string]*AuditDistributionRow) []AuditDistributionRow {
	var total int64
	for _, r := range m {
		total += r.Bytes
	}
	out := make([]AuditDistributionRow, 0, len(m))
	for _, p := range m {
		r := *p
		if total > 0 {
			r.Share = float64(r.Bytes) / float64(total)
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// RenderAuditDistributionCompact renders a stable one-line status/TUI view.
func RenderAuditDistributionCompact(categories, tools []AuditDistributionRow, width int) string {
	parts := []string{"tokens→"}
	for i, r := range categories {
		if i == 3 {
			break
		}
		parts = append(parts, fmt.Sprintf("%s %.0f%%", r.Name, r.Share*100))
	}
	if len(tools) > 0 {
		parts = append(parts, fmt.Sprintf("top-tool %s %.0f%%", tools[0].Name, tools[0].Share*100))
	}
	s := strings.Join(parts, " · ")
	if width > 0 && len(s) > width {
		if width <= 1 {
			return "…"
		}
		return s[:width-1] + "…"
	}
	return s
}
