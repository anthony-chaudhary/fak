package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archcheck"
)

func TestMCPArchCheckTool(t *testing.T) {
	srv := newTestServer(t)

	res := callMCPTool[archcheck.CheckResult](t, srv, "fak_arch_check", map[string]any{
		"package": "internal/agentquery",
	})
	if !res.OK {
		t.Fatalf("expected internal/agentquery to be clean over MCP, got violations: %+v", res.Violations)
	}
	if res.CheckedPackages != 1 {
		t.Fatalf("CheckedPackages = %d, want 1", res.CheckedPackages)
	}
}
