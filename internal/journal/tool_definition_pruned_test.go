package journal

import "testing"

func TestAppendToolDefinitionPruned(t *testing.T) {
	j := OpenMemory()
	row := j.AppendToolDefinitionPruned("trace-1", "customer_lookup")
	if row.Kind != KindToolDefinitionPruned || row.Tool != "customer_lookup" || row.TraceID != "trace-1" || row.Verdict != "ADVISORY" || row.By != "tool-definition-pruner" {
		t.Fatalf("row=%+v", row)
	}
	if row.ArgsDigest != "" || row.ResultDigest != "" {
		t.Fatalf("pruned-definition row leaked payload digests: %+v", row)
	}
	if got := j.AppendToolDefinitionPruned("", "missing-trace"); got.Seq != 0 {
		t.Fatalf("invalid row appended: %+v", got)
	}
}
