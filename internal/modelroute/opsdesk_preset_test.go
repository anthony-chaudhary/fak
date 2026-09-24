package modelroute

import (
	"os"
	"path/filepath"
	"testing"
)

// TestOpsDeskPresetDecisions is the SEMANTIC witness for the ops-desk routing
// preset (examples/routing-presets/ops-desk.json), rebuilt on the hosted hive
// setup (issue #13502 rev 2): TestRoutingPresetsRoundTrip already guards the
// file against rot (loads + byte-exact round trip); this test binds the JSON to
// the two-way auto-decision it encodes - the choice, per classified fak-Ops
// aspect, among the elected desk seats:
//
//   - glm-5.3-flash: GLM 5.3 Flash (zai-org/glm-5.3-flash on hive-ai, 131k
//     context), the fast MIDDLE seat of the hosted-flash rotation pool:
//     read/search tool calls, low-complexity aspects, short interactive turns,
//     AND the fail-closed default for unmatched normal work.
//   - deepseek-v4.1-flash: DeepSeek V4.1 Flash (deepseek-ai/DeepSeek-V4.1-Flash
//     on hive-ai, 1M context), the hard/long-context seat: ordinary medium-and-up
//     implementation, delegated spawns and batch facade traffic, hard/long-horizon
//     work, AND the security-release floor (destructive / high-risk aspects),
//     which never drops to the fast middle seat however small the call looks.
//
// gpt-6-astra and deepseek-v4-pro are NOT available on the hosted setup and are
// deleted from the desk; their duties re-seat onto the strongest available
// same-class peer (deepseek-v4.1-flash) rather than inventing an unavailable
// tier. The alternation lives in the TIER BANDS (a low-complexity aspect lands
// on glm-5.3-flash by firing the cheap rule, never by inheriting a sibling
// call's seat). Each case asserts BOTH the matched rule and the routed primary,
// so an accidental reorder that changes which rule fires is caught even when it
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
			name:    "hard/long-horizon aspect escalates to the deepseek seat",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-hard-deepseek",
		},
		{
			name:    "medium implementation auto-swaps to the deepseek impl seat",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityMedium},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-impl-deepseek",
		},
		{
			name:    "low-complexity aspect -> glm-5.3-flash (fast middle seat)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityLow},
			primary: "glm-5.3-flash",
			rule:    "opsdesk-cheap-glm53",
		},
		{
			name:    "unset-complexity aspect fails-closed to glm-5.3-flash (middle-ground default)",
			subject: Subject{Aspect: AspectRequest},
			primary: "glm-5.3-flash",
			rule:    "", // the fail-closed Default plan
		},
		{
			name:    "read-shaped tool call -> glm-5.3-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file"},
			primary: "glm-5.3-flash",
			rule:    "opsdesk-read-glm53",
		},
		{
			name:    "search tool call -> glm-5.3-flash",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb"},
			primary: "glm-5.3-flash",
			rule:    "opsdesk-search-glm53",
		},
		{
			name:    "short interactive turn -> glm-5.3-flash",
			subject: Subject{Aspect: AspectRequest, PromptTokens: 512, Latency: LatencyInteractive},
			primary: "glm-5.3-flash",
			rule:    "opsdesk-short-interactive-glm53",
		},
		{
			name:    "batch facade traffic rides the deepseek seat, not the default",
			subject: Subject{Aspect: AspectState, Latency: LatencyBatch, Labels: map[string]string{"work_kind": "fak-ops"}},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-fakops-deepseek",
		},
		{
			name:    "destructive tool call -> deepseek (security-release floor)",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_branch"},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-destructive-deepseek",
		},
		{
			// The floor must beat cheap signals: a tiny, low-complexity, interactive
			// delete still routes to the hard seat, never glm-5.3-flash.
			name:    "tiny low-complexity delete still routes to the hard seat",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_file", Complexity: ComplexityLow, Latency: LatencyInteractive},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-destructive-deepseek",
		},
		{
			name:    "drop-shaped tool call -> deepseek (same floor as delete)",
			subject: Subject{Aspect: AspectToolCall, Tool: "drop_table"},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-drop-deepseek",
		},
		{
			name:    "risk=high label -> deepseek",
			subject: Subject{Aspect: AspectRequest, Labels: map[string]string{"work_kind": "fak-ops", "risk": "high"}},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-riskhigh-deepseek",
		},
		{
			// A high-flagged read escalates: the hard rule precedes the read rule.
			name:    "high-complexity read escalates over the read->glm53 rule",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file", Complexity: ComplexityHigh},
			primary: "deepseek-v4.1-flash",
			rule:    "opsdesk-hard-deepseek",
		},
		{
			// A medium-flagged read escalates: the impl rule precedes the read rule.
			name:    "medium-complexity read escalates over the cheap band",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb", Complexity: ComplexityMedium},
			primary: "deepseek-v4.1-flash",
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
// between seats: the band rules bound each complexity band EXPLICITLY, so the
// desk alternates between the two elected models by the SHAPE of the work -
// fast middle seat for genuinely-low work, hard/long-context seat from medium
// up - never by seat adjacency or arrival order. Without it a future edit could
// delete a band rule and silently drop a whole band onto the default seat,
// while the round-trip test would still pass.
func TestOpsDeskPresetAlternationIsTierShaped(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/ops-desk.json")
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}

	// The two seats each own a distinct band: a hard/impl seat that bounds the
	// medium band and a cheap seat that bounds the low band, so alternation is
	// decided by the work. The trace-rule names are bound here to catch a
	// silent band collision.
	bandedRules := map[string]string{
		"opsdesk-hard-deepseek":        "deepseek-v4.1-flash",
		"opsdesk-impl-deepseek":        "deepseek-v4.1-flash",
		"opsdesk-cheap-glm53":          "glm-5.3-flash",
		"opsdesk-read-glm53":           "glm-5.3-flash",
		"opsdesk-search-glm53":         "glm-5.3-flash",
		"opsdesk-destructive-deepseek": "deepseek-v4.1-flash",
		"opsdesk-riskhigh-deepseek":    "deepseek-v4.1-flash",
	}
	for name, primary := range bandedRules {
		found := false
		for _, r := range m.Rules {
			if r.Name != name {
				continue
			}
			found = true
			if got := r.Plan.Primary(); got != primary {
				t.Errorf("band rule %q routes %q, want %q", name, got, primary)
			}
			break
		}
		if !found {
			t.Errorf("band rule %q missing from ops-desk preset", name)
		}
	}

	// The deleted seats must be gone everywhere: no member, no rule reads
	// gpt-6-astra or deepseek-v4-pro (both unavailable on the hosted setup).
	for _, gone := range []string{"gpt-6-astra", "deepseek-v4-pro"} {
		if got := m.Default.Primary(); got == gone {
			t.Errorf("default seat is %q: the unavailable seat must not return", gone)
		}
		for _, r := range m.Rules {
			for _, mem := range r.Plan.Members {
				if mem.Model == gone {
					t.Errorf("rule %q names deleted seat %q", r.Name, gone)
				}
			}
		}
	}

	// Alternation must ride the manifest grammar only: no arrival-order seat
	// rotation - same subject, same routed primary, byte-for-byte, in two
	// consecutive route calls (rotation by CLASS is a separate, explicit
	// mechanism; see TestOpsDeskRotationPoolNextSeat).
	subject := Subject{Aspect: AspectStep, Complexity: ComplexityLow}
	d1, d2 := m.Route(subject), m.Route(subject)
	if d1.Plan.Primary() != d2.Plan.Primary() || d1.RuleName != d2.RuleName {
		t.Fatalf("route alternation is not deterministic: %v/%q then %v/%q",
			d1.Plan.Primary(), d1.RuleName, d2.Plan.Primary(), d2.RuleName)
	}

	// The middle-ground tier is the desk default: an UNCLASSIFIED aspect
	// (no complexity, no labels, no latency hint) routes to glm-5.3-flash -
	// "the router defaults to the middle-ground tier" - never to the hard seat.
	d := m.Route(Subject{Aspect: AspectScout})
	if got := d.Plan.Primary(); got != "glm-5.3-flash" {
		t.Errorf("default seat = %q, want the middle-ground glm-5.3-flash (rule %q)", got, d.RuleName)
	}
	if d.RuleName != "" {
		t.Errorf("default seat must come from the Default plan, not a rule (rule %q)", d.RuleName)
	}
}

// TestOpsDeskRotationPoolNextSeat witnesses the GENERAL same-class rotation
// primitive (rotation.go) that the desk is built on: the hosted-flash pool
// declares the two seats the provider carries together, RotationPeers returns
// the same-class alternates, and NextSeat is a pure cyclic step across the
// pool - the seam dispatch wiring applies per call to spread load or step off
// an exhausted/degraded seat. A seat NOT declared in any pool (e.g. the
// deleted/unavailable gpt-6-astra) must not rotate anywhere.
func TestOpsDeskRotationPoolNextSeat(t *testing.T) {
	if err := ValidateRotationGroups(DeskRotationGroups()); err != nil {
		t.Fatalf("declared rotation pools invalid: %v", err)
	}

	class, ok := RotationClass("glm-5.3-flash")
	if !ok || class != "hosted-flash" {
		t.Fatalf("RotationClass(glm-5.3-flash) = %q, %v; want hosted-flash", class, ok)
	}
	peers, ok := RotationPeers("glm-5.3-flash")
	if !ok || len(peers) != 1 || peers[0] != "deepseek-v4.1-flash" {
		t.Fatalf("RotationPeers(glm-5.3-flash) = %v, %v; want [deepseek-v4.1-flash]", peers, ok)
	}

	// The pure cyclic step: glm53 -> deepseek -> glm53 (pool wraps). No state,
	// so two calls from the same seat agree - rotation never disturbs the
	// deterministic Route spine.
	for _, cycle := range []struct{ from, want string }{
		{"glm-5.3-flash", "deepseek-v4.1-flash"},
		{"deepseek-v4.1-flash", "glm-5.3-flash"},
	} {
		next, class, ok := NextSeat(cycle.from)
		if !ok || next != cycle.want || class != "hosted-flash" {
			t.Errorf("NextSeat(%q) = %q, %q, %v; want %q in hosted-flash", cycle.from, next, class, ok, cycle.want)
		}
	}

	// Peers is a copy: mutating it must not leak into the declared pools.
	peers[0] = "mutated"
	fresh, _ := RotationPeers("glm-5.3-flash")
	if fresh[0] != "deepseek-v4.1-flash" {
		t.Fatalf("RotationPeers leaked mutable state: %v", fresh)
	}

	// An undeclared seat is not a rotation candidate: the unavailable astra
	// seat must fail closed rather than rotate onto anything.
	if _, _, ok := NextSeat("gpt-6-astra"); ok {
		t.Error("NextSeat(gpt-6-astra) returned ok; an undeclared seat must not rotate")
	}
	if _, ok := RotationPeers("gpt-6-astra"); ok {
		t.Error("RotationPeers(gpt-6-astra) returned ok; an undeclared seat has no peers")
	}
}

// TestOpsDeskRotationGroupsValidateFailClosed proves the rotation validator is
// a fail-loud boundary: duplicate classes, single-seat pools, ambiguous
// membership, and non-token seat ids each refuse rather than silently
// mis-rotate.
func TestOpsDeskRotationGroupsValidateFailClosed(t *testing.T) {
	cases := []struct {
		name   string
		groups []RotationGroup
	}{
		{
			name:   "duplicate class",
			groups: []RotationGroup{{Class: "x", Seats: []string{"a", "b"}}, {Class: "x", Seats: []string{"c", "d"}}},
		},
		{
			name:   "single-seat pool",
			groups: []RotationGroup{{Class: "solo", Seats: []string{"a"}}},
		},
		{
			name:   "seat in two classes",
			groups: []RotationGroup{{Class: "x", Seats: []string{"a", "b"}}, {Class: "y", Seats: []string{"b", "c"}}},
		},
		{
			name:   "seat with route delimiter",
			groups: []RotationGroup{{Class: "x", Seats: []string{"openai:gpt", "b"}}},
		},
		{
			name:   "empty class name",
			groups: []RotationGroup{{Class: "", Seats: []string{"a", "b"}}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateRotationGroups(tc.groups); err == nil {
				t.Fatalf("ValidateRotationGroups(%v) = nil; want a fail-loud refusal", tc.groups)
			}
		})
	}
}

// TestOpsDeskHostedHiveRosterResolvesBothSeats is the "the provider supports
// both" witness: the shipped sidecar roster
// (examples/model-accounts.hosted-hive.json) must parse, validate, resolve the
// desk's Default plan AND its hard/destructive rules, and bind BOTH same-class
// seats to the SAME hive-ai account - one provider carrying both hosts of the
// hosted-flash pool, which is the precondition that makes GLM<->DeepSeek
// rotation legal. Upstream wire ids are asserted verbatim against the live
// operator config (hive-ai's catalog), plus the Nebius/Modal account-level
// rungs that carry the same deepseek upstream under separate credentials.
func TestOpsDeskHostedHiveRosterResolvesBothSeats(t *testing.T) {
	rosterPath := filepath.Join("..", "..", "examples", "model-accounts.hosted-hive.json")
	raw, err := os.ReadFile(rosterPath)
	if err != nil {
		t.Fatalf("read hosted-hive roster: %v", err)
	}
	r, err := ParseRoster(raw)
	if err != nil {
		t.Fatalf("shipped hosted-hive roster must parse and validate: %v", err)
	}

	// Every desk seat (default + every rule member) must resolve through the
	// roster - a preset id with no binding is an unresolved route waiting to
	// fail a turn.
	manifestPath := filepath.Join("..", "..", "examples", "routing-presets", "ops-desk.json")
	mraw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(mraw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}
	ids := map[string]bool{m.Default.Primary(): true}
	for _, rl := range m.Rules {
		for _, mem := range rl.Plan.Members {
			ids[mem.Model] = true
		}
	}
	for id := range ids {
		tgt, err := r.ResolvePlan(Plan{Members: []Member{{Model: id}}})
		if err != nil {
			t.Fatalf("desk seat %q unresolved in hosted-hive roster: %v", id, err)
		}
		if tgt.Members[0].Kind != KindOpenAI {
			t.Errorf("desk seat %q resolved kind %q, want openai-compatible", id, tgt.Members[0].Kind)
		}
		if tgt.Members[0].Account == "" {
			t.Errorf("desk seat %q resolved to an empty account", id)
		}
	}

	// The rotation precondition: BOTH same-class seats bind to the SAME
	// provider account (hive-ai carries both), each behind its verbatim
	// upstream wire id from the live hive catalog.
	want := map[string]struct {
		acct, upstream string
	}{
		"glm-5.3-flash":       {"hive-ai", "zai-org/glm-5.3-flash"},
		"deepseek-v4.1-flash": {"hive-ai", "deepseek-ai/DeepSeek-V4.1-Flash"},
	}
	for id, w := range want {
		tgt, err := r.Resolve(id)
		if err != nil {
			t.Fatalf("resolve %q: %v", id, err)
		}
		if tgt.Account != w.acct || tgt.UpstreamModel != w.upstream {
			t.Errorf("%q resolved to %q/%q, want %q/%q", id, tgt.Account, tgt.UpstreamModel, w.acct, w.upstream)
		}
		if tgt.BaseURL != "https://api.thehive.ai/api/v3" {
			t.Errorf("%q base URL %q, want the hive wire", id, tgt.BaseURL)
		}
		if tgt.CredEnv != "HIVE_API_KEY" {
			t.Errorf("%q credential env %q, want HIVE_API_KEY (a NAME; the secret stays out of the tree)", id, tgt.CredEnv)
		}
	}

	// The account-level failover rungs: the same deepseek upstream under
	// separate credentials/hosts - level 3 rotation (roster rungs), distinct
	// from the class-level pool. Each rung resolves ONLY through its own
	// dedicated binding added on top of the shipped file, never by stealing
	// the seat's binding.
	for _, rung := range []struct{ model, acct, wantUpstream string }{
		{"deepseek-v4.1-flash-nebius", "nebius", "deepseek-ai/DeepSeek-V4.1-Flash"},
		{"deepseek-v4.1-flash-modal", "modal", "deepseek-ai/DeepSeek-V4.1-Flash"},
	} {
		r2 := r
		r2.Bindings = append(append([]Binding(nil), r.Bindings...), Binding{Model: rung.model, Account: rung.acct, UpstreamModel: rung.wantUpstream})
		if err := r2.Validate(); err != nil {
			t.Fatalf("rung binding %q onto %q must validate: %v", rung.model, rung.acct, err)
		}
		tgt, err := r2.Resolve(rung.model)
		if err != nil {
			t.Fatalf("resolve rung %q: %v", rung.model, err)
		}
		if tgt.Account != rung.acct || tgt.UpstreamModel != rung.wantUpstream {
			t.Errorf("rung %q resolved to %q/%q, want %q/%q", rung.model, tgt.Account, tgt.UpstreamModel, rung.acct, rung.wantUpstream)
		}
	}
}

// TestOpsDeskPresetWorkerElectedSeatsAreThemselves pins that fak Ops worker
// election rides the manifest grammar, not a special registry: the preset
// carries the elected seat set (glm-5.3-flash / deepseek-v4.1-flash, declared
// together in the hosted-flash rotation pool) and every operator-facing surface
// rendering the desk reads it from data, so adding an alternate same-class
// model is a manifest + pool edit with byte-exact round trip, never a code
// change. It also re-checks manifest validity so the Captain/Mate semantic
// witness can never pass against an unvalidated file.
func TestOpsDeskPresetWorkerElectedSeatsAreThemselves(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/ops-desk.json")
	if err != nil {
		t.Fatalf("read ops-desk preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse ops-desk preset: %v", err)
	}

	// Both elected seats are declared rotation pool members (the hosted-flash
	// class) so the desk can rotate through them - and NO unavailable seat
	// (astra, v4-pro) is a pool member anywhere.
	pool := DeskRotationGroups()
	declared := map[string]bool{}
	for _, g := range pool {
		for _, s := range g.Seats {
			declared[s] = true
		}
	}
	for _, seat := range []string{"glm-5.3-flash", "deepseek-v4.1-flash"} {
		if !declared[seat] {
			t.Errorf("elected seat %q is not a declared rotation pool member", seat)
		}
	}
	for _, gone := range []string{"gpt-6-astra", "deepseek-v4-pro"} {
		if declared[gone] {
			t.Errorf("unavailable seat %q is declared in a rotation pool; it must not return", gone)
		}
		if got := m.Default.Primary(); got == gone {
			t.Errorf("default seat = %q; the unavailable seat must not return", gone)
		}
		if err := m.Validate(); err != nil {
			t.Errorf("ops-desk manifest invalid: %v", err)
		}
	}
}
