package hooks

import (
	"sort"
	"strings"
)

// demo_registry.go — the Go twin of tools/demo_registry.py, the shared browser-demo registry the
// demo audit tools key on (demo identity, base paths, default ports, hosted metadata) plus the
// lifecycle decisions. It backs the DEMO_COMMAND coverage gate (demoBrowserNames, derived here) and
// the BROWSER_CONTRACT gate (gate_browsercontract.go), exactly as demo_registry.py backs both
// demo_command_audit.py and demo_browser_contract.py.
//
// Parity anchor: tools/demo_registry.py DEMOS / DEMO_LIFECYCLE / VALID_LIFECYCLE_STATES /
// lifecycle_defects(). gate_browsercontract_test.go pins this table against the live Python
// demo_registry so a registry add/remove/edit that skips this file reds `go test`.

// demoReg is one browser-demo registry entry — the Go twin of demo_registry.Demo. Only the fields
// the ported audits read are carried; the smoke-test-only extra_args are omitted.
type demoReg struct {
	name          string
	basePath      string
	apiPath       string
	pageMarker    string
	defaultPort   int
	hostedPath    string
	hostedPort    int
	hostedAPIKeys []string
}

// demoRegistry mirrors demo_registry.DEMOS verbatim, in the same order (the order coverage and
// contract defects are emitted in).
var demoRegistry = []demoReg{
	{name: "guarddemo", basePath: "/guarddemo", apiPath: "api/scenarios", pageMarker: "safety floor", defaultPort: 8151},
	{name: "turntaxdemo", basePath: "/turntax", apiPath: "api/suites", pageMarker: "turn-tax demo", defaultPort: 8150,
		hostedPath: "/", hostedPort: 8150, hostedAPIKeys: []string{"suites"}},
	{name: "ctxdemo", basePath: "/ctxdemo", apiPath: "api/scenarios", pageMarker: "multi-agent context demo", defaultPort: 8153,
		hostedPath: "/", hostedPort: 8153, hostedAPIKeys: []string{"models", "scenarios"}},
	{name: "demorace", basePath: "/demorace", apiPath: "api/ladder", pageMarker: "reuse demo", defaultPort: 8147,
		hostedPath: "/demorace/", hostedAPIKeys: []string{"models", "prefill_tok_ratio_a_over_c"}},
	{name: "dropindemo", basePath: "/dropin", apiPath: "api/gallery", pageMarker: "fak guard drop-in gallery", defaultPort: 8154},
	{name: "unseedemo", basePath: "/unsee", apiPath: "api/events", pageMarker: "Un-See It", defaultPort: 8156},
	{name: "timewolfdemo", basePath: "/timewolf", apiPath: "api/scenarios", pageMarker: "what time is it, Mr. Wolf?", defaultPort: 8155},
	{name: "trychatdemo", basePath: "/trychat", apiPath: "api/suggestions", pageMarker: "try-it agentic chat", defaultPort: 8157},
}

// lifecycleDecision is the Go twin of demo_registry.LifecycleDecision.
type lifecycleDecision struct {
	state  string
	issue  int
	reason string
}

// validLifecycleStates mirrors demo_registry.VALID_LIFECYCLE_STATES.
var validLifecycleStates = []string{"hosted-keep", "promote-next", "local-keep", "archive", "tombstone"}

// demoLifecycle mirrors demo_registry.DEMO_LIFECYCLE.
var demoLifecycle = map[string]lifecycleDecision{
	"guarddemo":    {"local-keep", 1738, "healthy security-floor proof; keep in local/deep catalog unless promoted by a later rubric pass"},
	"turntaxdemo":  {"hosted-keep", 1167, "hosted front-door efficiency proof; live link currently witnessed"},
	"ctxdemo":      {"hosted-keep", 1739, "hosted research/performance slot; must re-earn with current model and net-value witness"},
	"demorace":     {"hosted-keep", 1739, "hosted live-model research slot; must re-earn with current model and net-value witness"},
	"dropindemo":   {"local-keep", 1738, "healthy adoption proof; keep in local/deep catalog unless it earns a front-door slot"},
	"unseedemo":    {"local-keep", 1738, "healthy KV-removal witness; keep in local/deep catalog unless promoted by the rubric"},
	"timewolfdemo": {"promote-next", 1736, "healthy LCD agentic loop; next candidate for hosted agentic card"},
	"trychatdemo":  {"promote-next", 1736, "healthy LCD try-it chat; next candidate for hosted agentic card"},
}

// demoRegNames returns the registry demo names in registry (DEMOS) order.
func demoRegNames() []string {
	names := make([]string, len(demoRegistry))
	for i, d := range demoRegistry {
		names[i] = d.name
	}
	return names
}

// isValidLifecycleState reports whether state is one of validLifecycleStates.
func isValidLifecycleState(state string) bool {
	for _, s := range validLifecycleStates {
		if s == state {
			return true
		}
	}
	return false
}

// demoLifecycleDefects ports demo_registry.lifecycle_defects(): every registered demo needs a
// lifecycle decision (and vice versa), each decision needs a valid state, a positive issue, and a
// non-blank reason, and hosted_path <-> hosted-keep must agree in both directions. The emission
// order mirrors the Python: missing-decision names, then unknown-demo names, then the per-demo
// checks over the intersection — each block sorted by name.
func demoLifecycleDefects() []string {
	registered := map[string]demoReg{}
	for _, d := range demoRegistry {
		registered[d.name] = d
	}

	var missingDecision, unknownDemo, both []string
	for name := range registered {
		if _, ok := demoLifecycle[name]; !ok {
			missingDecision = append(missingDecision, name)
		} else {
			both = append(both, name)
		}
	}
	for name := range demoLifecycle {
		if _, ok := registered[name]; !ok {
			unknownDemo = append(unknownDemo, name)
		}
	}
	sort.Strings(missingDecision)
	sort.Strings(unknownDemo)
	sort.Strings(both)

	var defects []string
	for _, name := range missingDecision {
		defects = append(defects, "browser demo lacks lifecycle decision: cmd/"+name)
	}
	for _, name := range unknownDemo {
		defects = append(defects, "lifecycle decision names unknown browser demo: cmd/"+name)
	}
	for _, name := range both {
		decision := demoLifecycle[name]
		demo := registered[name]
		if !isValidLifecycleState(decision.state) {
			defects = append(defects, name+": invalid lifecycle state '"+decision.state+"'; want one of "+
				joinComma(validLifecycleStates))
		}
		if decision.issue <= 0 {
			defects = append(defects, name+": lifecycle decision missing GitHub issue number")
		}
		if strings.TrimSpace(decision.reason) == "" {
			defects = append(defects, name+": lifecycle decision missing reason")
		}
		if demo.hostedPath != "" && decision.state != "hosted-keep" {
			defects = append(defects, name+": hosted demo must have lifecycle hosted-keep, got "+decision.state)
		}
		if demo.hostedPath == "" && decision.state == "hosted-keep" {
			defects = append(defects, name+": lifecycle hosted-keep requires hosted_path metadata")
		}
	}
	return defects
}

// joinComma renders a slice as ", "-joined text (the Python `', '.join(...)`).
func joinComma(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ", "
		}
		out += x
	}
	return out
}
