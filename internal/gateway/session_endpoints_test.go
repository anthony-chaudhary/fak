package gateway

import (
	"testing"
	"time"
)

// TestSessionEndpointsRoundTripsThroughDebugVars is the witness for the live accounts+nodes
// status area: a provider the host sets comes back on /debug/vars, an unset provider omits
// the block, and a provider with nothing to report also omits it (never an empty object).
func TestSessionEndpointsRoundTripsThroughDebugVars(t *testing.T) {
	srv := newTestServer(t)
	if got := srv.debugVars(time.Now()).Endpoints; got != nil {
		t.Fatalf("endpoints before SetSessionEndpointsProvider = %+v, want nil (omitted)", got)
	}

	// A provider that reports nothing must still omit the block.
	srv.SetSessionEndpointsProvider(func() SessionEndpoints { return SessionEndpoints{} })
	if got := srv.debugVars(time.Now()).Endpoints; got != nil {
		t.Fatalf("endpoints for an empty provider = %+v, want nil (omitted)", got)
	}

	want := SessionEndpoints{
		Accounts: []SessionAccount{
			{Name: "july2", Email: "july2@x", Active: true, CanServe: true, LoginStatus: "ready"},
			{Name: "work", Walled: true, LoginStatus: "ready"},
		},
		Nodes: []SessionNode{
			{Role: "kernel", ID: "win-box", Kind: "host"},
			{Role: "serving", ID: "api.anthropic.com", Kind: "proxy", Detail: "anthropic"},
		},
	}
	srv.SetSessionEndpointsProvider(func() SessionEndpoints { return want })
	got := srv.debugVars(time.Now()).Endpoints
	if got == nil {
		t.Fatal("endpoints after SetSessionEndpointsProvider = nil, want the reported block")
	}
	if len(got.Accounts) != 2 || !got.Accounts[0].Active || !got.Accounts[1].Walled {
		t.Fatalf("endpoints.Accounts = %+v, want july2 active + work walled", got.Accounts)
	}
	if len(got.Nodes) != 2 || got.Nodes[0].Role != "kernel" || got.Nodes[1].Role != "serving" {
		t.Fatalf("endpoints.Nodes = %+v, want a kernel + a serving node", got.Nodes)
	}

	// Detaching restores the omitted state.
	srv.SetSessionEndpointsProvider(nil)
	if got := srv.debugVars(time.Now()).Endpoints; got != nil {
		t.Fatalf("endpoints after detach = %+v, want nil", got)
	}
}

// TestSessionHarnessRoundTripsThroughDebugVars pins the harness block's provider seam and
// its unsampled-omit gate (Samples <= 0 → block absent, never an all-zero object).
func TestSessionHarnessRoundTripsThroughDebugVars(t *testing.T) {
	srv := newTestServer(t)
	if got := srv.debugVars(time.Now()).Harness; got != nil {
		t.Fatalf("harness before provider = %+v, want nil", got)
	}
	srv.SetSessionHarnessProvider(func() SessionHarness { return SessionHarness{Samples: 0} })
	if got := srv.debugVars(time.Now()).Harness; got != nil {
		t.Fatalf("harness for an unsampled provider = %+v, want nil (omitted)", got)
	}
	srv.SetSessionHarnessProvider(func() SessionHarness {
		return SessionHarness{Samples: 3, KernelCPUPercent: 12.5, KernelRSSBytes: 4096}
	})
	got := srv.debugVars(time.Now()).Harness
	if got == nil || got.Samples != 3 || got.KernelRSSBytes != 4096 {
		t.Fatalf("harness = %+v, want the sampled block", got)
	}
}

// TestEndpointsAndHarnessSafeOnNilServer pins the shared nil-Server contract.
func TestEndpointsAndHarnessSafeOnNilServer(t *testing.T) {
	var srv *Server
	srv.SetSessionEndpointsProvider(func() SessionEndpoints { return SessionEndpoints{} })
	srv.SetSessionHarnessProvider(func() SessionHarness { return SessionHarness{Samples: 1} })
	if _, ok := srv.sessionEndpoints(); ok {
		t.Fatal("nil Server sessionEndpoints ok=true, want false")
	}
	if _, ok := srv.sessionHarness(); ok {
		t.Fatal("nil Server sessionHarness ok=true, want false")
	}
}
