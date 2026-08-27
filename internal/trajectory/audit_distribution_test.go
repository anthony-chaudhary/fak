package trajectory

import "testing"

func TestAuditDistributionCompactCapturedRender(t *testing.T) {
	c := []AuditDistributionRow{{Name: "tool_result", Bytes: 60, Share: .6}, {Name: "reasoning", Bytes: 40, Share: .4}}
	tools := []AuditDistributionRow{{Name: "exec_command", Bytes: 60, Share: 1, Calls: 2}}
	got := RenderAuditDistributionCompact(c, tools, 120)
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
	cat, tool, id := classifyDistribution("codex", "response_item", []byte(`{"type":"function_call","name":"exec_command","call_id":"c1"}`), nil)
	if cat != "tool_call" || tool != "exec_command" || id != "c1" {
		t.Fatalf("codex=%q %q %q", cat, tool, id)
	}
	cat, tool, id = classifyDistribution("claude", "assistant", []byte(`{"content":[{"type":"tool_use","name":"Read","id":"t1"}]}`), nil)
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
