package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestNewConfiguredHTTPPlannerCarriesLlamaSlotAffinity pins the opt-in seam: the probe is
// off on a zero Config (so gateway fake upstreams never see a /props request) and
// Config.LlamaSlotAffinity reaches the proxy planner `fak serve` builds.
//
// fak-test:runtime fast est=20ms
func TestNewConfiguredHTTPPlannerCarriesLlamaSlotAffinity(t *testing.T) {
	for _, on := range []bool{false, true} {
		lone, err := newProxyPlanner(Config{Provider: "openai", APIKey: "k", LlamaSlotAffinity: on}, "m", []string{"https://a.example"})
		if err != nil {
			t.Fatalf("newProxyPlanner: %v", err)
		}
		hp, ok := lone.(*agent.HTTPPlanner)
		if !ok {
			t.Fatalf("planner = %T, want *agent.HTTPPlanner", lone)
		}
		if hp.LlamaSlotAffinity != on {
			t.Fatalf("Config.LlamaSlotAffinity=%v reached the planner as %v", on, hp.LlamaSlotAffinity)
		}
	}
}

// TestChatRouteSelfHostedPlannerCarriesLlamaSlotAffinity: a routed chat planner to a
// self-hosted account inherits Config.LlamaSlotAffinity; a vendor account never probes.
//
// fak-test:runtime fast est=50ms
func TestChatRouteSelfHostedPlannerCarriesLlamaSlotAffinity(t *testing.T) {
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:1/v1"},
			{ID: "vendor", Kind: modelroute.KindOpenAI, BaseURL: "https://vendor.invalid/v1", CredEnv: "FAK_LLAMA_SLOT_TEST_VENDOR_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "local-model", Account: "local"},
			{Model: "vendor-model", Account: "vendor"},
		},
		Default: "local",
	}
	t.Setenv("FAK_LLAMA_SLOT_TEST_VENDOR_KEY", "k")
	for _, on := range []bool{false, true} {
		srv, err := New(Config{EngineID: "mock", Model: "boot-model", RouteAccounts: roster, LlamaSlotAffinity: on})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(srv.Close)
		for model, want := range map[string]bool{"local-model": on, "vendor-model": false} {
			b, err := srv.bindChatRoute(context.Background(), model)
			if err != nil || b == nil {
				t.Fatalf("bindChatRoute(%s): %v, %v", model, b, err)
			}
			hp, ok := b.Planner.(*agent.HTTPPlanner)
			if !ok {
				t.Fatalf("%s planner = %T, want *agent.HTTPPlanner", model, b.Planner)
			}
			if hp.LlamaSlotAffinity != want {
				t.Fatalf("Config.LlamaSlotAffinity=%v: %s planner LlamaSlotAffinity=%v, want %v", on, model, hp.LlamaSlotAffinity, want)
			}
		}
	}
}
