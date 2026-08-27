package trajectory

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const AuditDistributionUnit = "utf8_content_bytes"
const AuditStorageUnit = "utf8_serialized_bytes"

type AuditDistributionRow struct {
	Name  string  `json:"name"`
	Bytes int64   `json:"bytes"`
	Share float64 `json:"share"`
	Calls int     `json:"calls,omitempty"`
}
type AuditStorageRow struct {
	Source  string  `json:"source"`
	Subtype string  `json:"subtype"`
	Bytes   int64   `json:"bytes"`
	Share   float64 `json:"share"`
	Records int     `json:"records"`
}
type auditDistribution struct {
	categories map[string]int64
	tools      map[string]*AuditDistributionRow
	callTools  map[string]string
	pending    map[string]int64
	storage    map[string]*AuditStorageRow
}

func newAuditDistribution() auditDistribution {
	return auditDistribution{map[string]int64{}, map[string]*AuditDistributionRow{}, map[string]string{}, map[string]int64{}, map[string]*AuditStorageRow{}}
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
	cat, tool, id, content, visible, sub := classifyDistribution(source, typ, payload, root)
	if !visible {
		k := source + "\x00" + sub
		r := d.storage[k]
		if r == nil {
			r = &AuditStorageRow{Source: source, Subtype: sub}
			d.storage[k] = r
		}
		r.Bytes += int64(len(line))
		r.Records++
		return
	}
	weight := int64(len(content))
	if cat == "" {
		cat = "visible_unknown"
	}
	d.categories[cat] += weight
	if cat == "tool_result" && tool == "" && id != "" {
		tool = d.callTools[id]
		if tool == "" {
			d.pending[id] += weight
		}
	}
	if tool != "" {
		r := d.tools[tool]
		if r == nil {
			r = &AuditDistributionRow{Name: tool}
			d.tools[tool] = r
		}
		r.Bytes += weight
		if cat == "tool_call" {
			r.Calls++
		}
		if id != "" {
			d.callTools[id] = tool
			if p := d.pending[id]; p > 0 {
				r.Bytes += p
				delete(d.pending, id)
			}
		}
	}
}

func classifyDistribution(source, typ string, payload json.RawMessage, root map[string]json.RawMessage) (cat, tool, id string, content []byte, visible bool, sub string) {
	if source == "codex" {
		if typ == "response_item" {
			var p map[string]json.RawMessage
			if json.Unmarshal(payload, &p) != nil {
				return "visible_unknown", "", "", nil, true, "response_item/malformed"
			}
			pt := rawString(p["type"])
			sub = "response_item/" + pt
			switch pt {
			case "function_call", "custom_tool_call", "web_search_call":
				return "tool_call", firstNonempty(rawString(p["name"]), pt), firstNonempty(rawString(p["call_id"]), rawString(p["id"])), joinRaw(p["arguments"], p["input"]), true, sub
			case "function_call_output", "custom_tool_call_output":
				return "tool_result", "", firstNonempty(rawString(p["call_id"]), rawString(p["id"])), joinRaw(p["output"]), true, sub
			case "reasoning":
				return "reasoning", "", "", joinRaw(p["summary"], p["content"]), true, sub
			case "message":
				role := rawString(p["role"])
				c := "assistant_message"
				if role == "user" {
					c = "user_message"
				}
				return c, "", "", joinRaw(p["content"]), true, sub
			}
			return "visible_unknown", "", "", joinRaw(payload), true, sub
		}
		if typ == "event_msg" {
			var p map[string]json.RawMessage
			_ = json.Unmarshal(payload, &p)
			pt := rawString(p["type"])
			if pt == "user_message" {
				return "user_message", "", "", joinRaw(p["message"]), true, "event_msg/user_message"
			}
			if pt == "agent_message" {
				return "assistant_message", "", "", joinRaw(p["message"]), true, "event_msg/agent_message"
			}
			if pt == "item_completed" || pt == "item_started" {
				var item map[string]json.RawMessage
				_ = json.Unmarshal(p["item"], &item)
				return "", "", "", nil, false, "event_msg/" + pt + "/" + firstNonempty(rawString(item["type"]), "unknown_item")
			}
			return "", "", "", nil, false, "event_msg/" + firstNonempty(pt, "unknown")
		}
		return "", "", "", nil, false, firstNonempty(typ, "unknown")
	}
	if typ == "user" || typ == "assistant" {
		base := "assistant_message"
		if typ == "user" {
			base = "user_message"
		}
		c, t, i, b := classifyClaudeContent(base, payload)
		return c, t, i, b, true, typ
	}
	if typ == "attachment" {
		var a map[string]json.RawMessage
		_ = json.Unmarshal(root["attachment"], &a)
		return "", "", "", nil, false, "attachment/" + firstNonempty(rawString(a["type"]), "unknown")
	}
	return "", "", "", nil, false, firstNonempty(typ, "unknown")
}
func classifyClaudeContent(base string, payload json.RawMessage) (string, string, string, []byte) {
	var msg map[string]json.RawMessage
	if json.Unmarshal(payload, &msg) != nil {
		return "visible_unknown", "", "", nil
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(msg["content"], &blocks) != nil {
		return base, "", "", joinRaw(msg["content"])
	}
	cat, tool, id := base, "", ""
	var b []byte
	for _, x := range blocks {
		switch rawString(x["type"]) {
		case "tool_use":
			cat, tool, id = "tool_call", rawString(x["name"]), rawString(x["id"])
			b = append(b, joinRaw(x["input"])...)
		case "tool_result":
			cat, id = "tool_result", rawString(x["tool_use_id"])
			b = append(b, joinRaw(x["content"])...)
		case "thinking", "reasoning":
			cat = "reasoning"
			b = append(b, joinRaw(x["thinking"], x["text"], x["content"])...)
		default:
			b = append(b, joinRaw(x["text"], x["content"])...)
		}
	}
	return cat, tool, id, b
}
func joinRaw(v ...json.RawMessage) []byte {
	var out []byte
	for _, r := range v {
		if len(r) == 0 {
			continue
		}
		var s string
		if json.Unmarshal(r, &s) == nil {
			out = append(out, []byte(s)...)
			continue
		}
		var anyv any
		if json.Unmarshal(r, &anyv) == nil {
			b, _ := json.Marshal(anyv)
			out = append(out, b...)
		}
	}
	return out
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
	o := make([]AuditDistributionRow, 0, len(m))
	for n, v := range m {
		r := AuditDistributionRow{Name: n, Bytes: v}
		if total > 0 {
			r.Share = float64(v) / float64(total)
		}
		o = append(o, r)
	}
	sort.Slice(o, func(i, j int) bool {
		if o[i].Bytes != o[j].Bytes {
			return o[i].Bytes > o[j].Bytes
		}
		return o[i].Name < o[j].Name
	})
	return o
}
func toolDistributionRows(m map[string]*AuditDistributionRow) []AuditDistributionRow {
	var total int64
	for _, r := range m {
		total += r.Bytes
	}
	o := make([]AuditDistributionRow, 0, len(m))
	for _, p := range m {
		r := *p
		if total > 0 {
			r.Share = float64(r.Bytes) / float64(total)
		}
		o = append(o, r)
	}
	sort.Slice(o, func(i, j int) bool {
		if o[i].Bytes != o[j].Bytes {
			return o[i].Bytes > o[j].Bytes
		}
		return o[i].Name < o[j].Name
	})
	return o
}
func storageDistributionRows(m map[string]*AuditStorageRow) []AuditStorageRow {
	tot := map[string]int64{}
	for _, r := range m {
		tot[r.Source] += r.Bytes
	}
	o := make([]AuditStorageRow, 0, len(m))
	for _, p := range m {
		r := *p
		if tot[r.Source] > 0 {
			r.Share = float64(r.Bytes) / float64(tot[r.Source])
		}
		o = append(o, r)
	}
	sort.Slice(o, func(i, j int) bool {
		if o[i].Bytes != o[j].Bytes {
			return o[i].Bytes > o[j].Bytes
		}
		if o[i].Source != o[j].Source {
			return o[i].Source < o[j].Source
		}
		return o[i].Subtype < o[j].Subtype
	})
	return o
}
func CompactAuditDistributionLine(c, t []AuditDistributionRow, width int) string {
	p := []string{"tokens→"}
	for i, r := range c {
		if i == 3 {
			break
		}
		p = append(p, fmt.Sprintf("%s %.0f%%", r.Name, r.Share*100))
	}
	if len(t) > 0 {
		p = append(p, fmt.Sprintf("top-tool %s %.0f%%", t[0].Name, t[0].Share*100))
	}
	s := strings.Join(p, " · ")
	if width > 0 && len(s) > width {
		if width <= 1 {
			return "…"
		}
		return s[:width-1] + "…"
	}
	return s
}
