package trajectory

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestAuditDistributionCompactCapturedRender(t *testing.T) {
	c := []AuditDistributionRow{{Name: "tool_result", Bytes: 60, Share: .6}, {Name: "reasoning", Bytes: 40, Share: .4}}
	tools := []AuditDistributionRow{{Name: "exec_command", Bytes: 60, Share: 1, Calls: 2}}
	got := CompactAuditDistributionLine(c, tools, 120)
	want := "tokens→ · tool_result 60% · reasoning 40% · top-tool exec_command 100%"
	if got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
}

func TestDistributionRowsConserveAndSort(t *testing.T) {
	rows := distributionRows(map[string]int64{"z": 10, "a": 10, "big": 80})
	if len(rows) != 3 || rows[0].Name != "big" || rows[1].Name != "a" {
		t.Fatalf("rows=%+v", rows)
	}
	var share float64
	var bytes int64
	for _, r := range rows {
		share += r.Share
		bytes += r.Bytes
	}
	if bytes != 100 || share < .999999 || share > 1.000001 {
		t.Fatalf("bytes=%d share=%f", bytes, share)
	}
}

func TestClassifyDistributionClaudeAndCodex(t *testing.T) {
	cat, tool, id, _, _, _ := classifyDistribution("codex", "response_item", []byte(`{"type":"function_call","name":"exec_command","call_id":"c1"}`), nil)
	if cat != "tool_call" || tool != "exec_command" || id != "c1" {
		t.Fatalf("codex=%q %q %q", cat, tool, id)
	}
	cat, tool, id, _, _, _ = classifyDistribution("claude", "assistant", []byte(`{"content":[{"type":"tool_use","name":"Read","id":"t1"}]}`), nil)
	if cat != "tool_call" || tool != "Read" || id != "t1" {
		t.Fatalf("claude=%q %q %q", cat, tool, id)
	}
}

func TestAuditDistributionAssociatesToolResult(t *testing.T) {
	d := newAuditDistribution()
	d.observe("codex", []byte(`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"c1"}}`))
	d.observe("codex", []byte(`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"large"}}`))
	rows := toolDistributionRows(d.tools)
	if len(rows) != 1 || rows[0].Name != "exec_command" || rows[0].Calls != 1 {
		t.Fatalf("rows=%+v", rows)
	}
	if rows[0].Bytes <= 0 {
		t.Fatalf("result bytes not attributed: %+v", rows[0])
	}
}

func TestDistributionSeparatesStorageMirrorsAndVisibleContent(t *testing.T) {
	d := newAuditDistribution()
	d.observe("codex", []byte(`{"type":"event_msg","payload":{"type":"item_completed","item":{"type":"CommandExecution","output":"duplicate"}}}`))
	d.observe("codex", []byte(`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"abc"}}`))
	if len(d.storage) != 1 {
		t.Fatalf("storage=%+v", d.storage)
	}
	rows := distributionRows(d.categories)
	if len(rows) != 1 || rows[0].Name != "tool_result" || rows[0].Bytes != 3 {
		t.Fatalf("visible=%+v", rows)
	}
}
func TestDistributionReconcilesResultBeforeCall(t *testing.T) {
	d := newAuditDistribution()
	d.observe("codex", []byte(`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"abc"}}`))
	d.observe("codex", []byte(`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"c1","arguments":"xy"}}`))
	rows := toolDistributionRows(d.tools)
	if len(rows) != 1 || rows[0].Name != "exec_command" || rows[0].Bytes != 5 {
		t.Fatalf("rows=%+v pending=%v", rows, d.pending)
	}
}
func TestClaudeAttachmentIsTypedStorage(t *testing.T) {
	d := newAuditDistribution()
	d.observe("claude", []byte(`{"type":"attachment","attachment":{"type":"deferred_tools_delta","addedNames":["x"]}}`))
	rows := storageDistributionRows(d.storage)
	if len(rows) != 1 || rows[0].Subtype != "attachment/deferred_tools_delta" || rows[0].Records != 1 {
		t.Fatalf("rows=%+v", rows)
	}
	if got := d.exemplars.snapshot(); got.Retained != 0 {
		t.Fatalf("known attachment subtypes entered unknown reservoir: %+v", got)
	}
}

func TestClaudeValidatedHookAttachmentIsTypedStorage(t *testing.T) {
	d := newAuditDistribution()
	d.observe("claude", []byte(`{"type":"attachment","attachment":{"type":"hook_success","durationMs":12}}`))
	rows := storageDistributionRows(d.storage)
	if len(rows) != 1 || rows[0].Subtype != "attachment/hook_success" || rows[0].Records != 1 {
		t.Fatalf("rows=%+v", rows)
	}
	if got := d.exemplars.snapshot(); got.Retained != 0 {
		t.Fatalf("known hook subtype entered unknown reservoir: %+v", got)
	}
}

func TestAuditToolResultsCodexExecOutcomesAndDirectIDs(t *testing.T) {
	d := newAuditDistribution()
	for _, line := range []string{
		`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"ok"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"ok","output":{"exit_code":0,"wall_time_seconds":1.25,"output":"done"}}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"err"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"err","output":{"exit_code":7,"duration_ms":22,"stderr":"failed","truncated":true}}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"wait"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"wait","output":{"status":"timed_out"}}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"cut"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"cut","output":{"status":"success","stdout":"partial","truncated":true}}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"mystery"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"mystery","output":"plain content has no outcome evidence"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"missing","output":"orphan"}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"mirror_only","id":"mirror"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"mirror","output":"not correlated"}}`,
	} {
		d.observe(AuditSourceCodex, []byte(line))
	}
	rows := d.toolResultRows()
	got := auditToolResultByName(rows, "exec_command")
	if got.Subtype != "command" || got.Results != 5 || got.Success != 2 || got.Errors != 1 || got.Timeouts != 1 || got.Truncated != 2 || got.Unknown != 1 || got.Unmatched != 0 {
		t.Fatalf("exec outcomes = %+v", got)
	}
	if got.ExitKnown != 2 || got.ExitZero != 1 || got.ExitNonzero != 1 || got.DurationKnown != 2 || got.DurationMS != 1272 {
		t.Fatalf("exec status/duration = %+v", got)
	}
	if got.Stdout != 1 || got.Stderr != 1 || got.CombinedOutput != 1 || got.ChannelUnknown != 2 {
		t.Fatalf("exec channels = %+v", got)
	}
	unmatched := auditToolResultByName(rows, "unmatched")
	if unmatched.Results != 2 || unmatched.Unmatched != 2 {
		t.Fatalf("unmatched outcomes = %+v; rows=%+v", unmatched, rows)
	}
}

func TestAuditToolResultsCodexMCPAndInputTextSuccess(t *testing.T) {
	d := newAuditDistribution()
	lines := []string{
		`{"type":"response_item","payload":{"type":"function_call","name":"fak_read","call_id":"c1"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"Wall time: 0.02s\nOutput:"},{"type":"input_text","text":"{\"results\":[{\"file_path\":\"a.go\",\"result\":{\"status\":\"OK\"}}]}"}]}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"fak_adjudicate","call_id":"c2"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"{\"outcome\":\"allowed\",\"verdict\":\"ALLOW\"}"}]}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"dos_arbitrate","call_id":"c3"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c3","output":[{"type":"input_text","text":"{\"outcome\":\"acquire\",\"reason\":\"ok\"}"}]}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"fak_tools_search","call_id":"c4"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"c4","output":[{"type":"input_text","text":"Wall time: 0.02s\nOutput:"},{"type":"input_text","text":"{\"tools\":[{\"name\":\"fak_capabilities\"}]}"}]}}`,
	}
	for _, line := range lines {
		d.observe(AuditSourceCodex, []byte(line))
	}
	rows := d.toolResultRows()
	read := auditToolResultByName(rows, "fak_read")
	if read.Results != 1 || read.Success != 1 || read.Unknown != 0 {
		t.Fatalf("fak_read outcomes = %+v, want 1 success, 0 unknown", read)
	}
	adjudicate := auditToolResultByName(rows, "fak_adjudicate")
	if adjudicate.Results != 1 || adjudicate.Success != 1 || adjudicate.Unknown != 0 {
		t.Fatalf("fak_adjudicate outcomes = %+v, want 1 success, 0 unknown", adjudicate)
	}
	arbitrate := auditToolResultByName(rows, "dos_arbitrate")
	if arbitrate.Results != 1 || arbitrate.Success != 1 || arbitrate.Unknown != 0 {
		t.Fatalf("dos_arbitrate outcomes = %+v, want 1 success, 0 unknown", arbitrate)
	}
	toolsSearch := auditToolResultByName(rows, "fak_tools_search")
	if toolsSearch.Results != 1 || toolsSearch.Success != 1 || toolsSearch.Unknown != 0 {
		t.Fatalf("fak_tools_search outcomes = %+v, want 1 success, 0 unknown", toolsSearch)
	}
}

func TestAuditToolResultMetadataNestedMapPrecedenceDeterministic(t *testing.T) {
	payload := make(map[string]any, 2)
	payload["z_later"] = map[string]any{"exit_code": 9, "duration_ms": 90}
	payload["a_first"] = map[string]any{"exit_code": 0, "duration_ms": 10}

	for attempt := 0; attempt < 256; attempt++ {
		var got auditToolResult
		auditToolResultMetadata(payload, &got)
		if !got.exitKnown || got.exitCode != 0 || !got.durationKnown || got.durationMS != 10 {
			t.Fatalf("attempt %d metadata = %+v, want lexically first nested envelope", attempt, got)
		}
	}
}

func TestAuditToolResultsClaudeMCPAndResultBeforeCall(t *testing.T) {
	d := newAuditDistribution()
	d.observe(AuditSourceClaude, []byte(`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"mcp-1","is_error":false,"content":[{"type":"text","text":"ok"}]}]}}`))
	d.observe(AuditSourceClaude, []byte(`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"mcp-1","name":"mcp__github__get_issue","input":{"owner":"private"}},{"type":"tool_use","id":"mcp-2","name":"mcp__github__get_issue","input":{}},{"type":"tool_use","id":"mcp-3","name":"mcp__github__get_issue","input":{}}]}}`))
	d.observe(AuditSourceClaude, []byte(`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"mcp-2","is_error":true,"content":"permission denied"}]}}`))
	d.observe(AuditSourceClaude, []byte(`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"mcp-3","is_error":true,"content":{"status":"timed_out"}}]}}`))
	got := auditToolResultByName(d.toolResultRows(), "mcp__github__get_issue")
	if got.Subtype != "mcp" || got.Results != 3 || got.Success != 1 || got.Errors != 1 || got.Timeouts != 1 || got.ChannelUnknown != 3 {
		t.Fatalf("Claude MCP outcomes = %+v", got)
	}
}

func TestClassifyClaudeMixedContentBlocksPreservesOrderIDsAndBytes(t *testing.T) {
	payload := json.RawMessage(`{"content":[{"type":"text","text":"TEXT"},{"type":"thinking","thinking":"THINK"},{"type":"tool_use","id":"call-7","name":"mcp__x__read","input":{"path":"a.txt"}},{"type":"tool_result","tool_use_id":"call-7","content":"RESULT"}]}`)
	events := classifyClaudeContentEvents("assistant_message", "assistant", payload)
	if len(events) != 4 {
		t.Fatalf("events = %d, want one per content block", len(events))
	}
	want := []struct {
		category string
		tool     string
		id       string
		content  string
	}{
		{category: "assistant_message", content: "TEXT"},
		{category: "reasoning", content: "THINK"},
		{category: "tool_call", tool: "mcp__x__read", id: "call-7", content: `{"path":"a.txt"}`},
		{category: "tool_result", id: "call-7", content: "RESULT"},
	}
	var gotBytes, wantBytes int
	for i, expected := range want {
		got := events[i]
		if got.category != expected.category || got.tool != expected.tool || got.id != expected.id || string(got.content) != expected.content {
			t.Fatalf("event %d = {category:%q tool:%q id:%q content:%q}, want %+v", i, got.category, got.tool, got.id, got.content, expected)
		}
		gotBytes += len(got.content)
		wantBytes += len(expected.content)
	}
	if gotBytes != wantBytes {
		t.Fatalf("content bytes = %d, want %d", gotBytes, wantBytes)
	}
}
func TestAuditClaudeMixedBlocksConserveResultProjectionBothOrders(t *testing.T) {
	orders := []string{
		`{"type":"text","text":"TEXT"},{"type":"thinking","thinking":"THINK"},{"type":"tool_result","tool_use_id":"c1","is_error":false,"content":"RESULT"}`,
		`{"type":"tool_result","tool_use_id":"c1","is_error":false,"content":"RESULT"},{"type":"thinking","thinking":"THINK"},{"type":"text","text":"TEXT"}`,
	}
	for _, blocks := range orders {
		d := newAuditDistribution()
		d.observe(AuditSourceClaude, []byte(`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c1","name":"mcp__x__read","input":{}}]}}`))
		d.observe(AuditSourceClaude, []byte(`{"type":"user","message":{"content":[`+blocks+`]}}`))
		categories := map[string]int64{}
		for _, row := range distributionRows(d.categories) {
			categories[row.Name] = row.Bytes
		}
		result := auditToolResultByName(d.toolResultRows(), "mcp__x__read")
		if categories["tool_result"] != result.Bytes || result.Bytes != int64(len("RESULT")) {
			t.Fatalf("result conservation failed: categories=%v result=%+v", categories, result)
		}
		if categories["user_message"] != int64(len("TEXT")) || categories["reasoning"] != int64(len("THINK")) {
			t.Fatalf("mixed block categories collapsed: %v", categories)
		}
	}
}

func TestAuditToolResultsConserveBytesAndDoNotRetainBodies(t *testing.T) {
	const sentinel = "PRIVATE_SENTINEL_9555"
	d := newAuditDistribution()
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"secret-id","arguments":"`+sentinel+`"}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"function_call_output","call_id":"secret-id","output":"`+sentinel+`"}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"function_call_output","call_id":"orphan-id","output":"abc"}}`))
	rows := d.toolResultRows()
	var resultBytes int64
	for _, row := range rows {
		resultBytes += row.Bytes
	}
	var categoryBytes int64
	for _, row := range distributionRows(d.categories) {
		if row.Name == "tool_result" {
			categoryBytes = row.Bytes
		}
	}
	if resultBytes != categoryBytes {
		t.Fatalf("result bytes = %d, category bytes = %d; rows=%+v", resultBytes, categoryBytes, rows)
	}
	result := AuditResult{Summary: AuditSummaryRow{Schema: AuditSchema, Kind: "summary", ToolResults: rows}, Transcripts: []AuditTranscriptRow{{Schema: AuditSchema, Kind: "session", ToolResults: rows}}}
	var jsonOut bytes.Buffer
	err := WriteAuditJSONL(&jsonOut, result)
	if err != nil {
		t.Fatal(err)
	}
	var markdown strings.Builder
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(rows)
	for label, report := range map[string][]byte{"rows": encoded, "jsonl": jsonOut.Bytes(), "markdown": []byte(markdown.String())} {
		if bytes.Contains(report, []byte(sentinel)) || bytes.Contains(report, []byte("secret-id")) || bytes.Contains(report, []byte("orphan-id")) {
			t.Fatalf("%s leaked raw identifiers or bodies: %s", label, report)
		}
	}
}

func TestAuditToolResultsSummaryAggregationPreservesSubtypesAndEvidence(t *testing.T) {
	transcripts := []AuditTranscriptRow{
		{ToolResults: []AuditToolResultRow{{Name: "runner", Subtype: "command", Bytes: 7, Results: 1, Errors: 1, Truncated: 1, ExitKnown: 1, ExitNonzero: 1, DurationKnown: 1, DurationMS: 12, Stderr: 1}}},
		{ToolResults: []AuditToolResultRow{{Name: "runner", Subtype: "command", Bytes: 5, Results: 1, Success: 1, ExitKnown: 1, ExitZero: 1, DurationKnown: 1, DurationMS: 8, Stdout: 1}, {Name: "runner", Subtype: "custom", Bytes: 3, Results: 1, Unknown: 1, ChannelUnknown: 1}}},
	}
	summary := summarizeAudit(nil, transcripts, nil)
	command := auditToolResultByNameSubtype(summary.ToolResults, "runner", "command")
	if command.Bytes != 12 || command.Results != 2 || command.Success != 1 || command.Errors != 1 || command.Truncated != 1 || command.ExitZero != 1 || command.ExitNonzero != 1 || command.DurationMS != 20 || command.Stdout != 1 || command.Stderr != 1 {
		t.Fatalf("command summary = %+v", command)
	}
	custom := auditToolResultByNameSubtype(summary.ToolResults, "runner", "custom")
	if custom.Results != 1 || custom.Unknown != 1 || custom.Bytes != 3 {
		t.Fatalf("custom summary = %+v; all=%+v", custom, summary.ToolResults)
	}
}

func TestAuditToolResultsCapturedMarkdownDrilldown(t *testing.T) {
	result := AuditResult{Summary: AuditSummaryRow{ToolResults: []AuditToolResultRow{
		{Name: "exec_command", Subtype: "command", Bytes: 42, Results: 4, Success: 1, Errors: 1, Timeouts: 1, Truncated: 1, Unknown: 1, ExitKnown: 2, ExitZero: 1, ExitNonzero: 1, DurationKnown: 1, DurationMS: 12, Stdout: 1, Stderr: 1, CombinedOutput: 1, ChannelUnknown: 1},
		{Name: "unmatched", Subtype: "unknown", Bytes: 3, Results: 1, Unmatched: 1, ChannelUnknown: 1},
	}}}
	var out strings.Builder
	if err := WriteAuditMarkdown(&out, result); err != nil {
		t.Fatal(err)
	}
	want := "### Tool result outcomes\n\n| Tool | Subtype | Results | Success | Errors | Timeouts | Truncated | Unknown | Unmatched | Exit 0/nonzero/unknown | Duration known/total ms | Output channel stdout/stderr/combined/unknown | UTF-8 bytes |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| `exec_command` | `command` | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 1/1/2 | 1/12 | 1/1/1/1 | 42 |\n| `unmatched` | `unknown` | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0/0/1 | 0/0 | 0/0/0/1 | 3 |"
	if !strings.Contains(out.String(), want) {
		t.Fatalf("captured drilldown missing:\n%s", out.String())
	}
}

func auditToolResultByName(rows []AuditToolResultRow, name string) AuditToolResultRow {
	for _, row := range rows {
		if row.Name == name {
			return row
		}
	}
	return AuditToolResultRow{}
}

func auditToolResultByNameSubtype(rows []AuditToolResultRow, name, subtype string) AuditToolResultRow {
	for _, row := range rows {
		if row.Name == name && row.Subtype == subtype {
			return row
		}
	}
	return AuditToolResultRow{}
}
