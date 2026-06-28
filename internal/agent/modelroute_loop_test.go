package agent

// modelroute_loop_test.go — witnesses the per-tool-call routing seam wired into the
// in-process agent loop (#598, epic #595). The acceptance: a configured manifest
// routes a NAMED tool to a DIFFERENT engine while leaving unmatched tools on the
// kernel default, and NO Engine is set when there is no manifest. We assert at two
// rungs so the test is non-vacuous:
//
//   - the pure routing decision (runConfig.routeToolEngine) mirrors the gateway
//     child: matched tool -> the manifest's Plan.Primary(), unmatched -> "", and a
//     nil manifest -> "" for every tool; and
//   - the WIRE: execViaKernel binds that route to abi.ToolCall.Engine BEFORE
//     k.Syscall, so a routed call actually DISPATCHES to the chosen engine (a
//     recording engine captures the c.Engine it received) while an unmatched call
//     lands on the loop's "localtools" default.

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// recordingEngine is a test engine that records the Engine route on each call it
// receives and returns a trivial OK result, so the test can witness WHICH engine a
// routed tool call actually dispatched to.
type recordingEngine struct {
	id   string
	mu   sync.Mutex
	seen []string // the c.Engine value of every call dispatched here
}

func (*recordingEngine) Caps() []abi.Capability { return nil }

func (e *recordingEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.mu.Lock()
	e.seen = append(e.seen, c.Engine)
	e.mu.Unlock()
	return engineResult(ctx, c, nil, []byte(`{"ok":true}`), false, e.id), nil
}

func (e *recordingEngine) calls() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]string, len(e.seen))
	copy(out, e.seen)
	return out
}

// guardManifest routes ONLY toolBook to the "guard-engine" model, leaving every other
// tool to fall through to the fail-closed default (model id "default"). Single-model
// PICKs both, so Plan.Primary() is the bound engine id.
func guardManifest() *modelroute.Manifest {
	return &modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "default", Role: "primary"}}},
		Rules: []modelroute.Rule{
			{
				Name:  "book-to-guard",
				Match: modelroute.Match{Aspect: modelroute.AspectToolCall, Tool: toolBook},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "guard-engine"}}},
			},
		},
	}
}

func TestRouteToolEngine_ManifestRoutesNamedToolOnly(t *testing.T) {
	cfg := resolveRunConfig([]RunOption{WithRouteManifest(guardManifest())})

	// The named tool routes to the DIFFERENT engine the manifest selected.
	if got := cfg.routeToolEngine(toolBook); got != "guard-engine" {
		t.Fatalf("routeToolEngine(%q) = %q, want %q (manifest must route the named tool)", toolBook, got, "guard-engine")
	}
	// An unmatched tool gets the manifest DEFAULT plan's primary ("default"), not the
	// guard engine — a named rule must not leak onto unmatched tools.
	if got := cfg.routeToolEngine(toolSearch); got != "default" {
		t.Fatalf("routeToolEngine(%q) = %q, want %q (unmatched tool must take the default plan)", toolSearch, got, "default")
	}
}

func TestRouteToolEngine_NoManifestLeavesEngineUnset(t *testing.T) {
	cfg := resolveRunConfig(nil) // no WithRouteManifest => nil manifest

	for _, tool := range []string{toolBook, toolSearch, toolGetUser} {
		if got := cfg.routeToolEngine(tool); got != "" {
			t.Fatalf("routeToolEngine(%q) = %q with no manifest, want \"\" (the no-manifest path MUST leave Engine unset)", tool, got)
		}
	}
}

// TestExecViaKernel_RoutesNamedToolToDifferentEngine is the WIRE witness: with a
// manifest the loop binds the route to abi.ToolCall.Engine pre-Syscall, so the routed
// tool DISPATCHES to the chosen engine while an unmatched tool takes the manifest
// default route. Without the manifest the call leaves Engine unset and dispatches to
// the loop's "localtools" kernel default — the historical path.
func TestExecViaKernel_RoutesNamedToolToDifferentEngine(t *testing.T) {
	Configure() // registers "localtools" as the default engine + the agent policy
	guard := &recordingEngine{id: "guard-engine"}
	deflt := &recordingEngine{id: "default"}
	localtools := &recordingEngine{id: "localtools"}
	// Register a recorder under each route id the test exercises so we can witness which
	// engine a call actually dispatched to. RegisterEngine replaces by id; overriding
	// "localtools" lets us read the route an UNSET-Engine call falls back to.
	abi.RegisterEngine("localtools", localtools)
	abi.RegisterEngine("guard-engine", guard)
	abi.RegisterEngine("default", deflt)

	k := kernel.New("localtools")
	ctx := context.Background()
	cfg := resolveRunConfig([]RunOption{WithRouteManifest(guardManifest())})

	// Routed tool: Engine must be "guard-engine" and the call must reach the guard engine.
	_, _ = execViaKernel(ctx, k, toolBook, `{"user_id":"u","flight_id":"f"}`, cfg.routeToolEngine(toolBook), traceEvent{})
	if g := guard.calls(); len(g) != 1 || g[0] != "guard-engine" {
		t.Fatalf("routed tool %q: guard engine saw %v, want exactly one call carrying Engine=%q", toolBook, g, "guard-engine")
	}

	// Unmatched tool: routeToolEngine returns the manifest DEFAULT plan's primary
	// ("default"), bound to Engine pre-Syscall, so the call dispatches to the "default"
	// engine — NOT the guard engine, and NOT the loop's localtools fallback.
	_, _ = execViaKernel(ctx, k, toolSearch, `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`, cfg.routeToolEngine(toolSearch), traceEvent{})
	if d := deflt.calls(); len(d) != 1 || d[0] != "default" {
		t.Fatalf("unmatched tool %q: default engine saw %v, want exactly one call carrying Engine=%q", toolSearch, d, "default")
	}
	if g := guard.calls(); len(g) != 1 {
		t.Fatalf("unmatched tool %q leaked onto the guard engine: guard now saw %v, want still exactly one (the book call)", toolSearch, g)
	}
	if l := localtools.calls(); len(l) != 0 {
		t.Fatalf("with a manifest, a routed call must NOT fall back to the kernel default; localtools saw %v, want none", l)
	}

	// No manifest: an unmatched tool now leaves Engine unset, so it dispatches to the
	// loop's "localtools" kernel default — the no-manifest path is the historical loop.
	// Distinct args avoid any (tool,args) result dedup with the routed calls above, so
	// this is a real dispatch we can witness.
	noRoute := resolveRunConfig(nil)
	if got := noRoute.routeToolEngine(toolSearch); got != "" {
		t.Fatalf("no-manifest routeToolEngine(%q) = %q, want \"\" (the no-manifest path MUST leave Engine unset)", toolSearch, got)
	}
	_, _ = execViaKernel(ctx, k, toolSearch, `{"origin":"LAX","destination":"BOS","date":"2026-08-02"}`, noRoute.routeToolEngine(toolSearch), traceEvent{})
	if l := localtools.calls(); len(l) != 1 || l[0] != "" {
		t.Fatalf("no-manifest unmatched tool %q: localtools default saw %v, want exactly one call with an UNSET Engine (\"\")", toolSearch, l)
	}
}
