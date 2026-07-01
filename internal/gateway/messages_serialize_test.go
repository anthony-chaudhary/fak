package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

func TestPromptSerializeAuditReportsUnstableGatewayBody(t *testing.T) {
	raw := []byte(`{"tools":[{"name":"z","description":"last","input_schema":{"type":"object","properties":{"z":{},"a":{}}}}],"model":"claude","max_tokens":64,"messages":[]}`)
	audit := auditPromptSerialization(raw)
	if audit.Status != promptmmu.SerializationChanged {
		t.Fatalf("status = %q, want %q for a body Go would re-key on marshal", audit.Status, promptmmu.SerializationChanged)
	}
	if audit.Range.IsZero() || audit.Range.OriginalEnd <= audit.Range.Start || audit.Range.CandidateEnd <= audit.Range.Start {
		t.Fatalf("changed byte range must be reported, got %+v", audit.Range)
	}
}
