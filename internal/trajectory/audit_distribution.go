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
type AuditToolResultRow struct {
	Name           string `json:"name"`
	Subtype        string `json:"subtype"`
	Bytes          int64  `json:"bytes"`
	Results        int    `json:"results"`
	Success        int    `json:"success,omitempty"`
	Errors         int    `json:"errors,omitempty"`
	Timeouts       int    `json:"timeouts,omitempty"`
	Truncated      int    `json:"truncated,omitempty"`
	Unknown        int    `json:"unknown,omitempty"`
	Unmatched      int    `json:"unmatched,omitempty"`
	ExitKnown      int    `json:"exit_known,omitempty"`
	ExitZero       int    `json:"exit_zero,omitempty"`
	ExitNonzero    int    `json:"exit_nonzero,omitempty"`
	DurationKnown  int    `json:"duration_known,omitempty"`
	DurationMS     int64  `json:"duration_ms,omitempty"`
	Stdout         int    `json:"stdout,omitempty"`
	Stderr         int    `json:"stderr,omitempty"`
	CombinedOutput int    `json:"combined_output,omitempty"`
	ChannelUnknown int    `json:"channel_unknown,omitempty"`
}
type AuditStorageRow struct {
	Source  string  `json:"source"`
	Subtype string  `json:"subtype"`
	Bytes   int64   `json:"bytes"`
	Share   float64 `json:"share"`
	Records int     `json:"records"`
}
type auditDistribution struct {
	categories     map[string]int64
	tools          map[string]*AuditDistributionRow
	results        map[string]*AuditToolResultRow
	callTools      map[string]string
	pending        map[string]int64
	resultCalls    map[string]auditResultCall
	pendingResults map[string][]auditToolResult
	storage        map[string]*AuditStorageRow
}

type auditResultCall struct {
	name    string
	subtype string
}

type auditToolResult struct {
	bytes          int64
	status         string
	truncated      bool
	exitKnown      bool
	exitCode       int64
	durationKnown  bool
	durationMS     int64
	stdout         bool
	stderr         bool
	combinedOutput bool
}

func newAuditDistribution() auditDistribution {
	return auditDistribution{
		categories: map[string]int64{}, tools: map[string]*AuditDistributionRow{}, results: map[string]*AuditToolResultRow{},
		callTools: map[string]string{}, pending: map[string]int64{}, resultCalls: map[string]auditResultCall{},
		pendingResults: map[string][]auditToolResult{}, storage: map[string]*AuditStorageRow{},
	}
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
	d.observeToolResult(source, typ, payload)
	if source == AuditSourceClaude && d.observeClaudeDistribution(typ, payload) {
		return
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
	d.observeVisible(cat, tool, id, int64(len(content)))
}

func (d *auditDistribution) observeVisible(cat, tool, id string, weight int64) {
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

func (d *auditDistribution) observeClaudeDistribution(typ string, payload json.RawMessage) bool {
	if typ != "assistant" && typ != "user" {
		return false
	}
	var message map[string]json.RawMessage
	if json.Unmarshal(payload, &message) != nil {
		return false
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(message["content"], &blocks) != nil {
		return false
	}
	base := "assistant_message"
	if typ == "user" {
		base = "user_message"
	}
	for _, block := range blocks {
		cat, tool, id := base, "", ""
		var content []byte
		switch rawString(block["type"]) {
		case "tool_use":
			cat, tool, id = "tool_call", rawString(block["name"]), rawString(block["id"])
			content = joinRaw(block["input"])
		case "tool_result":
			cat, id = "tool_result", rawString(block["tool_use_id"])
			content = joinRaw(block["content"])
		case "thinking", "reasoning":
			cat = "reasoning"
			content = joinRaw(block["thinking"], block["text"], block["content"])
		default:
			content = joinRaw(block["text"], block["content"])
		}
		d.observeVisible(cat, tool, id, int64(len(content)))
	}
	return true
}

// observeToolResult deliberately uses only the harness-native call/result IDs:
// Codex payload.call_id and Claude tool_use.id/tool_result.tool_use_id. Generic
// response item IDs and mirrored event records are not correlation evidence.
func (d *auditDistribution) observeToolResult(source, typ string, payload json.RawMessage) {
	if source == AuditSourceCodex {
		if typ != "response_item" {
			return
		}
		var item map[string]json.RawMessage
		if json.Unmarshal(payload, &item) != nil {
			return
		}
		switch rawString(item["type"]) {
		case "function_call", "custom_tool_call", "web_search_call":
			kind := rawString(item["type"])
			name := firstNonempty(rawString(item["name"]), kind)
			d.observeResultCall(rawString(item["call_id"]), name, auditToolSubtype(kind, name))
		case "function_call_output", "custom_tool_call_output":
			content, present := item["output"]
			d.observeResult(rawString(item["call_id"]), auditToolResultProjection(content, present, false, false))
		}
		return
	}
	if source != AuditSourceClaude || (typ != "assistant" && typ != "user") {
		return
	}
	var message map[string]json.RawMessage
	if json.Unmarshal(payload, &message) != nil {
		return
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(message["content"], &blocks) != nil {
		return
	}
	for _, block := range blocks {
		switch rawString(block["type"]) {
		case "tool_use":
			name := rawString(block["name"])
			d.observeResultCall(rawString(block["id"]), name, auditToolSubtype("tool_use", name))
		case "tool_result":
			content, present := block["content"]
			_, errorPresent := block["is_error"]
			explicitError := rawBool(block["is_error"])
			d.observeResult(rawString(block["tool_use_id"]), auditToolResultProjection(content, present, explicitError, errorPresent && !explicitError))
		}
	}
}

func (d *auditDistribution) observeResultCall(id, tool, subtype string) {
	if id == "" {
		return
	}
	if tool == "" {
		tool = "unknown"
	}
	call := auditResultCall{name: tool, subtype: subtype}
	d.resultCalls[id] = call
	for _, result := range d.pendingResults[id] {
		d.addResult(call, result)
	}
	delete(d.pendingResults, id)
}

func (d *auditDistribution) observeResult(id string, result auditToolResult) {
	if id == "" {
		result.status = "unmatched"
		d.addResult(auditResultCall{name: "unmatched", subtype: "unknown"}, result)
		return
	}
	if call, ok := d.resultCalls[id]; ok {
		d.addResult(call, result)
		return
	}
	d.pendingResults[id] = append(d.pendingResults[id], result)
}

func (d *auditDistribution) addResult(call auditResultCall, result auditToolResult) {
	key := auditToolResultKey(call.name, call.subtype)
	row := d.results[key]
	if row == nil {
		row = &AuditToolResultRow{Name: call.name, Subtype: call.subtype}
		d.results[key] = row
	}
	applyAuditToolResult(row, result)
}

func applyAuditToolResult(row *AuditToolResultRow, result auditToolResult) {
	row.Bytes += result.bytes
	row.Results++
	switch result.status {
	case "success":
		row.Success++
	case "error":
		row.Errors++
	case "timeout":
		row.Timeouts++
	case "unmatched":
		row.Unmatched++
	default:
		row.Unknown++
	}
	if result.truncated {
		row.Truncated++
	}
	if result.exitKnown {
		row.ExitKnown++
		if result.exitCode == 0 {
			row.ExitZero++
		} else {
			row.ExitNonzero++
		}
	}
	if result.durationKnown {
		row.DurationKnown++
		row.DurationMS += result.durationMS
	}
	if result.stdout {
		row.Stdout++
	}
	if result.stderr {
		row.Stderr++
	}
	if result.combinedOutput {
		row.CombinedOutput++
	}
	if !result.stdout && !result.stderr && !result.combinedOutput {
		row.ChannelUnknown++
	}
}

func auditToolResultKey(name, subtype string) string { return name + "\x00" + subtype }

func auditToolResultProjection(content json.RawMessage, present, explicitError, explicitSuccess bool) auditToolResult {
	result := auditToolResult{bytes: int64(len(joinRaw(content))), status: "unknown"}
	if !present {
		return result
	}
	var value any
	if json.Unmarshal(content, &value) != nil {
		return result
	}
	result.truncated = auditOutputIsTruncated(value)
	auditToolResultMetadata(value, &result)
	switch {
	case auditOutputIsTimeout(value):
		result.status = "timeout"
	case explicitError || auditOutputIsError(value):
		result.status = "error"
	case explicitSuccess || auditOutputIsSuccess(value):
		result.status = "success"
	}
	return result
}

func auditOutputIsSuccess(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"is_error", "isError"} {
			if raw, ok := typed[key]; ok {
				if flag, boolean := raw.(bool); boolean && !flag {
					return true
				}
			}
		}
		for _, key := range []string{"exit_code", "exitCode"} {
			if raw, ok := typed[key]; ok {
				if code, err := auditAnyInt(raw); err == nil && code == 0 {
					return true
				}
			}
		}
		if status, _ := typed["status"].(string); auditSuccessStatus(status) {
			return true
		}
		for _, key := range []string{"output", "content", "result"} {
			if child, ok := typed[key]; ok && auditOutputIsSuccess(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if auditOutputIsSuccess(child) {
				return true
			}
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		var decoded any
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && json.Unmarshal([]byte(trimmed), &decoded) == nil {
			return auditOutputIsSuccess(decoded)
		}
	}
	return false
}

func auditSuccessStatus(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "success", "succeeded", "completed", "complete", "ok":
		return true
	default:
		return false
	}
}

func auditToolResultMetadata(value any, result *auditToolResult) {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"exit_code", "exitCode"} {
			if raw, ok := typed[key]; ok && !result.exitKnown {
				if code, err := auditAnyInt(raw); err == nil {
					result.exitKnown, result.exitCode = true, code
				}
			}
		}
		if !result.durationKnown {
			for _, field := range []struct {
				key        string
				multiplier float64
			}{{"duration_ms", 1}, {"durationMs", 1}, {"wall_time_ms", 1}, {"wall_time_seconds", 1000}} {
				if raw, ok := typed[field.key]; ok {
					if number, valid := auditResultNumber(raw); valid && number >= 0 {
						result.durationKnown = true
						result.durationMS = int64(number * field.multiplier)
						break
					}
				}
			}
		}
		_, hasStdout := typed["stdout"]
		_, hasStderr := typed["stderr"]
		_, hasOutput := typed["output"]
		result.stdout = result.stdout || hasStdout
		result.stderr = result.stderr || hasStderr
		result.combinedOutput = result.combinedOutput || hasOutput
		for _, child := range typed {
			auditToolResultMetadata(child, result)
		}
	case []any:
		for _, child := range typed {
			auditToolResultMetadata(child, result)
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		var decoded any
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && json.Unmarshal([]byte(trimmed), &decoded) == nil {
			auditToolResultMetadata(decoded, result)
		}
	}
}

func auditResultNumber(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		value, err := typed.Float64()
		return value, err == nil
	default:
		return 0, false
	}
}

func auditToolSubtype(kind, name string) string {
	lowerName := strings.ToLower(strings.TrimSpace(name))
	if strings.HasPrefix(lowerName, "mcp__") || strings.Contains(lowerName, "__mcp__") {
		return "mcp"
	}
	for _, command := range []string{"exec_command", "write_stdin"} {
		if lowerName == command || strings.HasSuffix(lowerName, "."+command) || strings.HasSuffix(lowerName, "__"+command) {
			return "command"
		}
	}
	switch kind {
	case "custom_tool_call":
		return "custom"
	case "web_search_call":
		return "web_search"
	case "function_call", "tool_use":
		return "function"
	default:
		return "unknown"
	}
}

func auditOutputIsTruncated(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"truncated", "is_truncated", "isTruncated"} {
			if flag, _ := typed[key].(bool); flag {
				return true
			}
		}
		if status, _ := typed["status"].(string); strings.EqualFold(strings.TrimSpace(status), "truncated") {
			return true
		}
		for _, key := range []string{"output", "content", "result", "text"} {
			if child, ok := typed[key]; ok && auditOutputIsTruncated(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if auditOutputIsTruncated(child) {
				return true
			}
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return false
		}
		var decoded any
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && json.Unmarshal([]byte(trimmed), &decoded) == nil {
			return auditOutputIsTruncated(decoded)
		}
		normalized := strings.ToLower(strings.Join(strings.Fields(trimmed), " "))
		return strings.Contains(normalized, "output truncated") || strings.Contains(normalized, "truncated output")
	}
	return false
}

func rawBool(v json.RawMessage) bool { var b bool; _ = json.Unmarshal(v, &b); return b }

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
func (d *auditDistribution) toolResultRows() []AuditToolResultRow {
	totals := make(map[string]*AuditToolResultRow, len(d.results)+1)
	for key, row := range d.results {
		copy := *row
		totals[key] = &copy
	}
	for _, pending := range d.pendingResults {
		for _, result := range pending {
			key := auditToolResultKey("unmatched", "unknown")
			row := totals[key]
			if row == nil {
				row = &AuditToolResultRow{Name: "unmatched", Subtype: "unknown"}
				totals[key] = row
			}
			result.status = "unmatched"
			applyAuditToolResult(row, result)
		}
	}
	return auditToolResultRows(totals)
}
func mergeAuditToolResultRows(groups ...[]AuditToolResultRow) []AuditToolResultRow {
	totals := map[string]*AuditToolResultRow{}
	for _, rows := range groups {
		for _, row := range rows {
			key := auditToolResultKey(row.Name, row.Subtype)
			total := totals[key]
			if total == nil {
				total = &AuditToolResultRow{Name: row.Name, Subtype: row.Subtype}
				totals[key] = total
			}
			total.Bytes += row.Bytes
			total.Results += row.Results
			total.Success += row.Success
			total.Errors += row.Errors
			total.Timeouts += row.Timeouts
			total.Truncated += row.Truncated
			total.Unknown += row.Unknown
			total.Unmatched += row.Unmatched
			total.ExitKnown += row.ExitKnown
			total.ExitZero += row.ExitZero
			total.ExitNonzero += row.ExitNonzero
			total.DurationKnown += row.DurationKnown
			total.DurationMS += row.DurationMS
			total.Stdout += row.Stdout
			total.Stderr += row.Stderr
			total.CombinedOutput += row.CombinedOutput
			total.ChannelUnknown += row.ChannelUnknown
		}
	}
	return auditToolResultRows(totals)
}
func auditToolResultRows(m map[string]*AuditToolResultRow) []AuditToolResultRow {
	o := make([]AuditToolResultRow, 0, len(m))
	for _, row := range m {
		o = append(o, *row)
	}
	sort.Slice(o, func(i, j int) bool {
		if o[i].Bytes != o[j].Bytes {
			return o[i].Bytes > o[j].Bytes
		}
		if o[i].Name != o[j].Name {
			return o[i].Name < o[j].Name
		}
		return o[i].Subtype < o[j].Subtype
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
