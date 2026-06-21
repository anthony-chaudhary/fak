package gateway

import "testing"

// TestMCPProtocolListIsSingleSourceOfTruth pins the invariant the centralization
// refactor establishes: defaultProtocol and supportedProtocols are BOTH derived
// from mcpProtocolVersions, so the list is the only thing an editor touches.
func TestMCPProtocolListIsSingleSourceOfTruth(t *testing.T) {
	if len(mcpProtocolVersions) == 0 {
		t.Fatal("mcpProtocolVersions must declare at least one revision")
	}
	// The default is the first declared revision.
	if defaultProtocol != mcpProtocolVersions[0] {
		t.Errorf("defaultProtocol = %q, want first list entry %q", defaultProtocol, mcpProtocolVersions[0])
	}
	// supportedProtocols is exactly the set of declared revisions — no more, no less.
	if len(supportedProtocols) != len(mcpProtocolVersions) {
		t.Errorf("supportedProtocols has %d entries, list declares %d", len(supportedProtocols), len(mcpProtocolVersions))
	}
	for _, v := range mcpProtocolVersions {
		if !supportedProtocols[v] {
			t.Errorf("declared revision %q missing from supportedProtocols", v)
		}
	}
	// The default must itself be supported (we never answer with a revision we'd reject).
	if !supportedProtocols[defaultProtocol] {
		t.Errorf("defaultProtocol %q is not in supportedProtocols", defaultProtocol)
	}
}

// TestMCPNegotiatorUsesDerivedList confirms the negotiator answers from the
// derived set: every declared revision is echoed back, an undeclared one falls
// back to the default.
func TestMCPNegotiatorUsesDerivedList(t *testing.T) {
	srv := newTestServer(t)
	for _, v := range mcpProtocolVersions {
		got := srv.initializeResult(jsonProto(v))["protocolVersion"]
		if got != v {
			t.Errorf("declared revision %q must be echoed, got %v", v, got)
		}
	}
	if got := srv.initializeResult(jsonProto("0000-00-00"))["protocolVersion"]; got != defaultProtocol {
		t.Errorf("undeclared revision must fall back to %q, got %v", defaultProtocol, got)
	}
}

func jsonProto(v string) []byte {
	return []byte(`{"protocolVersion":"` + v + `"}`)
}
