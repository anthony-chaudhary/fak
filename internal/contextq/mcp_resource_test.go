package contextq

import "testing"

func TestMCPMissingContextResourceRequest(t *testing.T) {
	req, ok := MCPMissingContextResourceRequest("fak://context/missing/deploy-target", 512)
	if !ok {
		t.Fatal("missing-context URI was not recognized")
	}
	if req.Method != "resources/read" || req.Key != "deploy-target" || req.Reason != "missing_context" || !req.Audited {
		t.Fatalf("request = %+v, want typed audited missing-context resource read", req)
	}
	if req.BudgetBytes != 512 {
		t.Fatalf("budget bytes = %d, want 512", req.BudgetBytes)
	}
}

func TestMCPMissingContextResourceRequestRejectsOtherURIs(t *testing.T) {
	for _, uri := range []string{"", "fak://server/capabilities", "fak://context/missing/"} {
		if req, ok := MCPMissingContextResourceRequest(uri, 0); ok {
			t.Fatalf("uri %q unexpectedly parsed as %+v", uri, req)
		}
	}
}
