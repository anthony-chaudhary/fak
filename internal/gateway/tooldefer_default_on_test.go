package gateway

// tooldefer_default_on_test.go — the #3537 witness: the cold-tool deferral lever
// (#3232, epic #3229) is DEFAULT-ON at the front doors via DefaultDeferColdTools, and
// the flip is SAFE — deferral moves a cold def out of the provider's eagerly-loaded
// set, it never removes it. Three properties pinned here, on a server armed exactly the
// way `fak guard` / `fak serve` arm it from the parsed flag default:
//
//  1. DEFAULT-ON DEFERS: the cold custom tail is deferred out of the initial eager set
//     and one tool_search_tool is injected (the discovery channel).
//  2. FAIL-SAFE RESOLUTION: every deferred def stays BYTE-COMPLETE in tools[] — name,
//     description, input_schema all untouched, only the defer_loading key added — so a
//     first real use still resolves via the search tool's fault-in; nothing goes
//     silently missing. (The floor still adjudicates a deferred name: tooldefer_no_bypass_test.go.)
//  3. HOT UNAFFECTED: an always-on tool (Read/Bash) keeps its element verbatim, never
//     deferred.
//
// Plus the preserved rollback levers: FAK_ABLATE_DEFER_TOOLS=1 and the parsed
// --defer-cold-tools=false opt-out both stand the transform down to byte-identity.

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// armedDefaultDeferServer builds a passthrough server whose deferColdTools mirrors
// exactly what both front doors wire from the parsed flag default (#3537 flip):
// gateway.Config{DeferColdTools: DefaultDeferColdTools} → New's cfg.DeferColdTools copy.
func armedDefaultDeferServer() *Server {
	return &Server{
		planner:        &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		deferColdTools: DefaultDeferColdTools,
		metrics:        &gatewayMetrics{},
		logf:           func(string, ...any) {},
	}
}

func TestDeferColdToolsDefaultOnDefersColdTailKeepsDefsResolvable(t *testing.T) {
	if !DefaultDeferColdTools {
		t.Fatalf("DefaultDeferColdTools must be true (the #3537 flip)")
	}
	srv := armedDefaultDeferServer()
	req := &agent.AnthropicMessagesRequest{Model: "claude-x", Raw: []byte(deferBody)}
	n := srv.maybeDeferColdTools(req, "trace-default-on")
	if n != 2 {
		t.Fatalf("default-on server deferred %d tools; want 2 (the cold mcp__ tail)", n)
	}

	inputByName := map[string]map[string]json.RawMessage{}
	for _, m := range toolsOf(t, []byte(deferBody)) {
		inputByName[rawStringField(m, "name")] = m
	}
	outByName := map[string]map[string]json.RawMessage{}
	searchTools := 0
	for _, m := range toolsOf(t, req.Raw) {
		outByName[rawStringField(m, "name")] = m
		if rawStringField(m, "name") == toolSearchToolName {
			searchTools++
		}
	}

	// (1) Cold custom tools are deferred out of the initial eager set…
	for _, cold := range []string{"mcp__fak__fak_syscall", "mcp__dos__dos_verify"} {
		out, ok := outByName[cold]
		if !ok {
			t.Fatalf("deferred tool %q VANISHED from tools[] — deferral must never remove a def", cold)
		}
		if !hasDeferLoading(out) {
			t.Errorf("cold tool %q not deferred under the default-on posture", cold)
		}
		// (2) …but each stays byte-complete, so a first real use still resolves.
		for _, field := range []string{"description", "input_schema"} {
			if !bytes.Equal(out[field], inputByName[cold][field]) {
				t.Errorf("deferred tool %q field %q changed: got %s, want %s (fault-in needs the untouched def)",
					cold, field, out[field], inputByName[cold][field])
			}
		}
	}
	// …and the discovery channel for the fault-in is present exactly once.
	if searchTools != 1 {
		t.Fatalf("injected tool_search_tool count=%d, want exactly 1", searchTools)
	}

	// (3) A hot/always-on tool is unaffected: never deferred, def untouched.
	for _, hot := range []string{"Read", "Bash"} {
		out, ok := outByName[hot]
		if !ok {
			t.Fatalf("hot tool %q missing from tools[]", hot)
		}
		if hasDeferLoading(out) {
			t.Errorf("hot tool %q was deferred; it must stay eager", hot)
		}
		for _, field := range []string{"description", "input_schema"} {
			if !bytes.Equal(out[field], inputByName[hot][field]) {
				t.Errorf("hot tool %q field %q changed under deferral", hot, field)
			}
		}
	}
}

func TestDeferColdToolsDefaultOnRollbackLeversStandDown(t *testing.T) {
	// FAK_ABLATE_DEFER_TOOLS=1: the A/B ablation arm stands the armed default down.
	t.Setenv("FAK_ABLATE_DEFER_TOOLS", "1")
	srv := armedDefaultDeferServer()
	req := &agent.AnthropicMessagesRequest{Model: "claude-x", Raw: []byte(deferBody)}
	if n := srv.maybeDeferColdTools(req, "trace-ablated"); n != 0 {
		t.Fatalf("FAK_ABLATE_DEFER_TOOLS=1 server deferred %d tools; want 0", n)
	}
	if !bytes.Equal(req.Raw, []byte(deferBody)) {
		t.Fatalf("ablated server mutated req.Raw")
	}
}

func TestDeferColdToolsExplicitOptOutStaysOff(t *testing.T) {
	// The parsed --defer-cold-tools=false opt-out: deferColdTools false is byte-identity.
	srv := armedDefaultDeferServer()
	srv.deferColdTools = false
	req := &agent.AnthropicMessagesRequest{Model: "claude-x", Raw: []byte(deferBody)}
	if n := srv.maybeDeferColdTools(req, "trace-opt-out"); n != 0 {
		t.Fatalf("opted-out server deferred %d tools; want 0", n)
	}
	if !bytes.Equal(req.Raw, []byte(deferBody)) {
		t.Fatalf("opted-out server mutated req.Raw")
	}
}
