package agent

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

// TestRouterLiveProbe is the LIVE end-to-end witness for the router.com vendor
// registration (issue #13477). It drives the REAL https://api.router.com/v1
// Responses endpoint through the provider="router" planner when ROUTER_API_KEY is
// set, and SKIPS otherwise so the default CI run stays offline and deterministic.
//
// What it proves: the router planner reaches the vendor and speaks its wire well
// enough to get either a parsed completion OR the vendor's own TYPED status error.
// It FAILS on a 401/404 (auth or wire/path shape refused) and on any non-upstream
// error (transport/DNS/serialization failure) — so a silent mis-marshal or a wrong
// endpoint cannot pass. An unfunded key legitimately yields a 402 payment_required,
// which is a PASS: it is the vendor telling us it recognized the request.
func TestRouterLiveProbe(t *testing.T) {
	key := strings.TrimSpace(os.Getenv("ROUTER_API_KEY"))
	if key == "" {
		t.Skip("set ROUTER_API_KEY to run the live router.com probe")
	}
	planner, err := NewProviderHTTPPlanner("router", "https://api.router.com/v1", RouterDefaultModelForTest, key)
	if err != nil {
		t.Fatalf("build router planner: %v", err)
	}
	comp, err := planner.Complete(context.Background(), adapterTestMessages("Hello from Router"), nil)
	if err != nil {
		var use *UpstreamStatusError
		if !errors.As(err, &use) {
			t.Fatalf("live router probe: non-upstream error (wire/transport failure?): %v", err)
		}
		if use.Status == 401 || use.Status == 404 {
			t.Fatalf("live router probe: status %d means the vendor refused our auth or wire shape: %v", use.Status, err)
		}
		t.Logf("router.com reached; typed upstream status %d: %v", use.Status, err)
		return
	}
	if comp == nil || strings.TrimSpace(comp.Message.Content) == "" {
		t.Fatalf("live router probe: completion carried no text: %+v", comp)
	}
	t.Logf("router.com completion ok: %q", comp.Message.Content)
}

// RouterDefaultModelForTest mirrors the modelroute default the roster binds the
// "router" route to, kept here so the live probe exercises a real catalog id
// without importing the (stdlib-only) modelroute package into this test.
const RouterDefaultModelForTest = "deepseek-v4.1-flash"
