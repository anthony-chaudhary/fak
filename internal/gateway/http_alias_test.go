package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestAliasHTTPEndpointAndLiveRedirect is the load-bearing witness for #11091: an
// alias set through the management endpoint at RUNTIME redirects a subsequent chat
// completion to the alias's NEW target — no restart, no changed client request.
func TestAliasHTTPEndpointAndLiveRedirect(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	boot := newChatRouteUpstream(t, "boot")
	targetA := newChatRouteUpstream(t, "a")
	targetB := newChatRouteUpstream(t, "b")
	t.Setenv("FAK_ALIAS_A_KEY", "target-a-secret")
	t.Setenv("FAK_ALIAS_B_KEY", "target-b-secret")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "account-a", Kind: modelroute.KindOpenAI, BaseURL: targetA.server.URL, CredEnv: "FAK_ALIAS_A_KEY"},
			{ID: "account-b", Kind: modelroute.KindOpenAI, BaseURL: targetB.server.URL, CredEnv: "FAK_ALIAS_B_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "model-a", Account: "account-a", UpstreamModel: "upstream-alias-a"},
			{Model: "model-b", Account: "account-b", UpstreamModel: "upstream-alias-b"},
		},
		Default: "account-a",
	}
	aliases, err := modelroute.NewAliasStore(modelroute.AliasRegistry{Aliases: []modelroute.Alias{
		{Name: "prod", Target: "model-a"},
	}})
	if err != nil {
		t.Fatalf("NewAliasStore: %v", err)
	}
	srv, err := New(Config{EngineID: "test", Model: "boot-model", BaseURL: boot.server.URL, Provider: "openai-compatible", APIKey: "boot-secret", RouteAccounts: roster, RouteAliases: aliases})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// A request naming the alias routes to model-a's account (target A).
	beforeA, beforeB := len(targetA.snapshot()), len(targetB.snapshot())
	status, response := postChatRoute(t, ts.URL, "/v1/chat/completions", "prod", "", false, nil)
	if status != http.StatusOK || !bytes.Contains(response, []byte("reply-a")) {
		t.Fatalf("initial alias route: status=%d response=%s", status, response)
	}
	if len(targetA.snapshot()) != beforeA+1 || len(targetB.snapshot()) != beforeB {
		t.Fatalf("initial alias route hit wrong target: a=%d b=%d", len(targetA.snapshot()), len(targetB.snapshot()))
	}

	// Reassign the alias through the management endpoint.
	body, _ := json.Marshal(aliasMutationRequest{Name: "prod", Target: "model-b"})
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/fak/route/aliases", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST aliases: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST aliases status = %d, want 200", resp.StatusCode)
	}

	// GET lists the new table.
	getResp, err := http.Get(ts.URL + "/v1/fak/route/aliases")
	if err != nil {
		t.Fatalf("GET aliases: %v", err)
	}
	defer getResp.Body.Close()
	if getResp.StatusCode != http.StatusOK {
		t.Fatalf("GET aliases status = %d, want 200", getResp.StatusCode)
	}
	var listed aliasListResponse
	if err := json.NewDecoder(getResp.Body).Decode(&listed); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	found := false
	for _, a := range listed.Aliases {
		if a.Name == "prod" && a.Target == "model-b" {
			found = true
		}
	}
	if !found {
		t.Fatalf("GET did not list prod->model-b: %+v", listed.Aliases)
	}

	// The load-bearing witness: the NEXT completion for the SAME alias now reaches
	// target B, with no restart and no client change.
	beforeA, beforeB = len(targetA.snapshot()), len(targetB.snapshot())
	status, response = postChatRoute(t, ts.URL, "/v1/chat/completions", "prod", "", false, nil)
	if status != http.StatusOK || !bytes.Contains(response, []byte("reply-b")) {
		t.Fatalf("redirected alias route: status=%d response=%s", status, response)
	}
	if len(targetA.snapshot()) != beforeA || len(targetB.snapshot()) != beforeB+1 {
		t.Fatalf("reassigned alias did not redirect: a=%d b=%d", len(targetA.snapshot()), len(targetB.snapshot()))
	}
}

// TestAliasEndpointDisabledWithoutStore proves an unconfigured server 404s (inert
// by default) and that nil-guarding keeps the surface off.
func TestAliasEndpointDisabledWithoutStore(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/v1/fak/route/aliases")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("unconfigured alias endpoint status = %d, want 404", resp.StatusCode)
	}
}