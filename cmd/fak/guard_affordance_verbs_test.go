package main

import (
	"bytes"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// affordanceVerbRe extracts fak_-prefixed verb tokens from an affordance's prose. The
// mcp__fak__ tool prefix is stripped to a space BEFORE matching (see extractFakVerbs) so
// a "mcp__fak__fak_capabilities" mention yields the bare verb "fak_capabilities", not a
// mangled "fak__fak_capabilities". A verb name is [a-z_]+ after the fak_ stem.
var affordanceVerbRe = regexp.MustCompile(`fak_[a-z_]+`)

// affordanceNonVerbTokens are fak_-prefixed tokens that legitimately appear in affordance
// prose but are NOT MCP verbs, so they must not be checked against the verb registry.
// Currently only the gateway metric the #3093 advisory cites by name. A NEW fak_ token
// that lands here uninvited is far more likely a typo'd verb than a new metric — which is
// exactly the rot this test exists to surface, so keep this set deliberately tiny.
var affordanceNonVerbTokens = map[string]bool{
	"fak_mcp_verb_calls_total": true,
}

// extractFakVerbs returns the distinct fak verb names an affordance's text names, with the
// mcp__fak__ prefix disarmed and known non-verb tokens (e.g. the metric) removed.
func extractFakVerbs(text string) []string {
	cleaned := strings.ReplaceAll(text, "mcp__fak__", " ")
	seen := map[string]bool{}
	var out []string
	for _, v := range affordanceVerbRe.FindAllString(cleaned, -1) {
		if affordanceNonVerbTokens[v] || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

// TestGuardAffordancesNameOnlyRealVerbs is the rot-proofing witness for the guard
// affordances that exist to make fak actually get used: the #3092 SessionStart hint and
// the #3093 unused-substrate Stop advisory. BOTH name fak entry verbs as free-text prose
// bound to nothing — a later gateway verb rename would leave them pointing at a DEAD verb
// an autonomous agent cannot call, silently defeating the fix while every string-contains
// test stays green. This binds every verb the affordances name to the gateway's advertised
// registry (toolDescriptors, the single source of truth behind tools/list, fak_tools_search
// and the --expose filter), so a rename that skips the prose fails loudly here instead.
func TestGuardAffordancesNameOnlyRealVerbs(t *testing.T) {
	advertised := map[string]bool{}
	for _, d := range gateway.ToolDescriptorsForResolver() {
		if n, _ := d["name"].(string); n != "" {
			advertised[n] = true
		}
	}
	if len(advertised) == 0 {
		t.Fatal("gateway advertises no verbs — the registry accessor is broken; cannot witness the affordances")
	}

	// Capture the ACTUAL bytes each affordance emits, not a restatement of them.
	t.Setenv(guardStopHookUnusedEnvMode, guardPreCompactModeShadow) // force the advisory on
	var stopAdvisory bytes.Buffer
	emitUnusedSubstrateAdvisory(&stopAdvisory, guardStopHookSignals{FakVerbCallsSeen: true, FakVerbCalls: 0})
	if stopAdvisory.Len() == 0 {
		t.Fatal("unused-substrate advisory emitted nothing for a clean 0-verb stop; cannot witness its named verbs")
	}

	sources := map[string]string{
		"SessionStart affordance (#3092)":        guardSessionStartHint,
		"unused-substrate Stop advisory (#3093)": stopAdvisory.String(),
	}
	for label, text := range sources {
		verbs := extractFakVerbs(text)
		if len(verbs) == 0 {
			t.Errorf("%s names no fak verbs — an affordance that points at nothing cannot make fak get used", label)
			continue
		}
		for _, v := range verbs {
			if !advertised[v] {
				t.Errorf("%s names %q, which is NOT an advertised gateway verb — dead-verb rot; update the prose or the registry", label, v)
			}
		}
	}
}
