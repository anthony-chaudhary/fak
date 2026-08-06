package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// TestNativeLoopRouteParityLabelMatchedManifest is the cross-path witness for #5706.
// The rule can only fire when BOTH call-shape and caller labels reach Subject.Labels:
// routedTool is write-shaped and the authenticated caller is tenant acme. Before #5706,
// the proxy selected guard-a while the owned loop silently fell through to default.
func TestNativeLoopRouteParityLabelMatchedManifest(t *testing.T) {
	vdso.Default.BumpWorld()
	manifest := &modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: defaultRoute, Role: "primary"}}},
		Rules: []modelroute.Rule{{
			Name: "acme-read-to-guard",
			Match: modelroute.Match{
				Aspect: modelroute.AspectToolCall,
				Tool:   routedTool,
				Labels: map[string]string{"read_only": "false", "tenant": "acme"},
			},
			Plan: modelroute.Plan{Members: []modelroute.Member{{Model: "guard-a", Role: "primary"}}},
		}},
	}
	s, rec := nativeParityServer(t, manifest, nil, "guard-a", defaultRoute)
	ctx := WithPrincipal(context.Background(), "acme")

	proxy, err := proxyRoute(s, ctx, routedTool)
	if err != nil {
		t.Fatalf("proxy route: %v", err)
	}
	if proxy != "guard-a" {
		t.Fatalf("proxy route = %q, want guard-a: label-matched rule did not fire", proxy)
	}

	driveNativeTurn(t, s, ctx)
	native, ok := rec.route(routedTool)
	if !ok {
		t.Fatalf("native loop did not dispatch %q", routedTool)
	}
	if native != proxy {
		t.Fatalf("label-matched route diverged: native=%q proxy=%q; native must lower read_only and tenant labels", native, proxy)
	}
}
