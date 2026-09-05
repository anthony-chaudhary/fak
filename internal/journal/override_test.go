package journal

import (
	"testing"
)

func TestSecurityOverrideChainsAndVerifies(t *testing.T) {
	j := OpenMemory()

	row := j.AppendSecurityOverride("IFC_SINK", "send_email", "sess-123", "TAINT_EGRESS", "User confirmed email destination is approved", "ifc-sink(override)")
	if row.Kind != KindSecurityOverride {
		t.Fatalf("row.Kind = %q, want %q", row.Kind, KindSecurityOverride)
	}
	if row.Tool != "send_email" || row.TraceID != "sess-123" {
		t.Fatalf("unexpected row targets: %+v", row)
	}
	if row.Witness != "User confirmed email destination is approved" {
		t.Fatalf("row.Witness = %q, want justification", row.Witness)
	}

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}

	// Verify chain integrity
	if _, err := VerifyRows(rows); err != nil {
		t.Fatalf("VerifyRows failed: %v", err)
	}
}
