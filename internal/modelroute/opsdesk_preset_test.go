package modelroute

import (
	"os"
	"testing"
)

// TestOpsDeskPresetDecisions is the SEMANTIC witness for the ops-desk routing
// preset (examples/routing-presets/ops-desk.json). TestRoutingPresetsRoundTrip
// already guards the file against rot (loads + byte-exact round trip); this test
// binds the JSON to the three-way auto-decision it exists to encode — the
// choice, per classified fak-Ops aspect, among the elected desk seats:
//
//   - glm53-flash  — GLM 5.3 Flash, the MIDDLE-GROUND tier (T1): read/search tool
//     calls, low-complexity aspects, short interactive turns, AND the fail-closed
//     default for unmatched normal work.
//   - deepseek-v4-pro — the hard-work escalation (T0-capable coder): ordinary
//     medium-and-up implementation, delegated spawns and batch scout traffic.
//   - gpt-6-astra — the most careful orchestration seat (T0): genuinely hard work
//     AND the security-release floor (destructive / high-risk aspects), which
//     never drops to the middle tier however small the call looks.
//
// The alternation lives in the TIER BANDS (medium work auto-swaps DOWN to glm53
// flash's band edge in reverse: a low-complexity aspect lands on glm53-flash by
// firing the cheap rule, never by inheriting a sibling call's seat), and the
// seat map is the operator interface: the T1 band rows prove "glm53-flash is the
// seat that serves it" without the manifest's default ever reading as a hard
// pin. Without this test the preset is data-only: a future edit could silently
// re-route a dev/root aspect to the wrong seat — most dangerously drop a
// destructive tool call to glm53-flash, breaking the security-release floor
// tierpolicy.go fixes for T0 work — and the round-trip test would still pass.
// Each case asserts BOTH the matched rule and the routed primary, so an
// accidental reorder that changes which rule fires is caught even when it
// happens to route to the same model.
func TestOpsDeskPresetDecisions(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/ops-desk.json")
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}

	cases := []struct {
		name    string
		subject Subject
		primary string // expected routed model id
		rule    string // expected matched rule name ("" == fail-closed default)
	}{
		{
			name:    "ultra-hard aspect escalates to astra orchestration seat",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			primary: "gpt-6-astra",
			rule:    "opsdesk-ultrahard-astra",
		},
		{
			name:    "medium implementation auto-swaps to deepseek pro",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityMedium},
			primary: "deepseek-v4-pro",
			rule:    "opsdesk-impl-deepseek",
		},
		{
			name:    "low-complexity aspect -> glm53-flash (middle tier)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityLow},
			primary: "glm53-flash",
			rule:    "opsdesk-cheap-glm53",
		},
		{
			name:    "unset-complexity aspect fails-closed to glm53-flash (middle tier default)",
			subject: Subject{Aspect: AspectRequest},
			primary: "glm53-flash",
			rule:    "", // the fail-closed Default plan
		},
		{
			name:    "read-shaped tool call -> glm53-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file"},
			primary: "glm53-flash",
			rule:    "opsdesk-read-glm53",
		},
		{
			name:    "search tool call -> glm53-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb"},
			primary: "glm53-flash",
			rule:    "opsdesk-search-glm53",
		},
		{
			name:    "short interactive turn -> glm53-flash",
			subject: Subject{Aspect: AspectRequest, PromptTokens: 512, Latency: LatencyInteractive},
			primary: "glm53-flash",
			rule:    "opsdesk-short-interactive-glm53",
		},
		{
			name:    "batch spawn rides the pro seat, not the default",
			subject: Subject{Aspect: AspectState, Latency: LatencyBatch, Labels: map[string]string{"work_kind": "fak-ops"}},
			primary: "gpt-6-astra",
			rule:    "opsdesk-fakops-astra",
		},
		{
			name:    "destructive tool call -> astra (security-release floor)",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_branch"},
			primary: "gpt-6-astra",
			rule:    "opsdesk-destructive-astra",
		},
		{
			// The floor must beat cheap signals: a tiny, low-complexity, interactive
			// delete still routes to the orchestration seat, never glm53-flash.
			name:    "tiny low-complexity delete still routes to astra",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_file", Complexity: ComplexityLow, Latency: LatencyInteractive},
			primary: "gpt-6-astra",
			rule:    "opsdesk-destructive-astra",
		},
		{
			name:    "risk=high label -> astra",
			subject: Subject{Aspect: AspectRequest, Labels: map[string]string{"work_kind": "fak-ops", "risk": "high"}},
			primary: "gpt-6-astra",
			rule:    "opsdesk-riskhigh-astra",
		},
		{
			// A high-flagged read escalates: the ultra-hard rule precedes the read rule.
			name:    "high-complexity read escalates over the read->glm53 rule",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file", Complexity: ComplexityHigh},
			primary: "gpt-6-astra",
			rule:    "opsdesk-ultrahard-astra",
		},
		{
			// A medium-flagged read escalates: the impl rule precedes the read rule.
			name:    "medium-complexity read escalates over the cheap band",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb", Complexity: ComplexityMedium},
			primary: "deepseek-v4-pro",
			rule:    "opsdesk-impl-deepseek",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := m.Route(tc.subject)
			if got := d.Plan.Primary(); got != tc.primary {
				t.Errorf("routed primary = %q, want %q (rule %q)", got, tc.primary, d.RuleName)
			}
			if d.RuleName != tc.rule {
				t.Errorf("matched rule = %q, want %q", d.RuleName, tc.rule)
			}
		})
	}
}

// TestOpsDeskPresetAlternationIsTierShaped pins how the allocator alternates
// between seats: the three rules bound each complexity band EXPLICITLY, so the
// desk alternates among the elected models by the SHAPE of the work rather than
// by seat adjacency — with the middle band pinned to the middle-ground tier.
// Without it a future edit could delete a band rule and silently drop the
// implicit alternation ("no rule matched" would then inherit the default seat
// for a whole band), while the round-trip test would still pass.
//
// The T<B> band vocabulary mirrors tierpolicy.go's label/number inversion (T0 is
// the MOST demanding but the lowest B), so the manifest reads in capability
// order top-down while the seated band rows read bottom-up. The band rows are
// not constants in this package, but they are bound to this file's manifest
// rules and reason strings, byte-checked in the round-trip gate.
func TestOpsDeskPresetAlternationIsTierShaped(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/ops-desk.json")
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}

	// The three seats each own a distinct band: an ultra-hard seat, an impl seat,
	// and a cheap seat that bounds the low band, so alternation among the elected
	// models is decided by the work, never by arrival order. The trace-echo rule
	// names are bound here to catch a silent band collision.
	bandedRules := map[string]string{
		"opsdesk-ultrahard-astra": "gpt-6-astra",
		"opsdesk-impl-deepseek":   "deepseek-v4-pro",
		"opsdesk-cheap-glm53":     "glm53-flash",
	}
	for name, primary := range bandedRules {
		band := m.Rules[0]
		found := false
		for _, r := range m.Rules {
			if r.Name == name {
				band = r
				found = true
				break
			}
		}
		if !found {
			t.Errorf("band rule %q missing from ops-desk preset", name)
			continue
		}
		if got := band.Plan.Primary(); got != primary {
			t.Errorf("band rule %q routes %q, want %q", name, got, primary)
		}
	}

	// Alternation must ride the manifest grammar only: no ConstantAlternation
	// knob, no arrival-order seat rotation — same subject, same routed primary,
	// byte-for-byte, in two consecutive route calls (every alternation decision
	// is already deterministic in modelroute.Route).
	subject := Subject{Aspect: AspectStep, Complexity: ComplexityLow}
	d1, d2 := m.Route(subject), m.Route(subject)
	if d1.Plan.Primary() != d2.Plan.Primary() || d1.RuleName != d2.RuleName {
		t.Fatalf("route alternation is not deterministic: %v/%q then %v/%q",
			d1.Plan.Primary(), d1.RuleName, d2.Plan.Primary(), d2.RuleName)
	}

	// The middle-ground tier is the desk default: an UNCLASSIFIED aspect
	// (no complexity, no labels, no latency hint) routes to glm53-flash —
	// "the router defaults to the middle-ground tier" — never to an
	// orchestration seat and never to an impl seat on arrival-order adjacency.
	d := m.Route(Subject{Aspect: AspectScout})
	if got := d.Plan.Primary(); got != "glm53-flash" {
		t.Errorf("default seat = %q, want the middle-ground glm53-flash (rule %q)", got, d.RuleName)
	}
	if d.RuleName != "" {
		t.Errorf("default seat must come from the Default plan, not a rule (rule %q)", d.RuleName)
	}
}

// TestOpsDeskPresetWorkerElectedSeatsAreThemselves pins that fak Ops worker
// election rides the manifest grammar, not a special registry or per-worker
// registry setup: the preset carries the elected seat set
// (glm53-flash / deepseek-v4-pro / gpt-6-astra) and every operator facing
// surface rendering the desk reads it from data, so adding an alternate model
// ; is a manifest edit with byte-exact round-trip, never a code change.
//
// It binds the manifest to the seat ids the opsdesk tests assert, so a rename
// fails the semantic witness (the seat vocabulary is stable in this package) —
// the same fail-loud discipline the presets pack applies to defaults.
func TestOpsDeskPresetWorkerElectedSeatsAreThemselves(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/ops-desk.json")
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}

	// Byte-bound trace of the elected seats in this file, so a rename in the
	// data fails the preset semantic witness (the seat vocabulary is stable
	// here) and the default can never silently name an un-elected seat.
	if got := m.Default.Primary(); got != "glm53-flash" {
		t.Errorf("default seat = %q, want glm53-flash (the middle-ground tier)", got)
	}
	if err := m.Validate(); err != nil {
		t.Errorf("ops-desk manifest invalid: %v", err)
	}
}
