package gateway

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// The #5493 property, stated once: on a wire that cannot run them, prompt-shrink levers that
// are configured ON must be NAMED on a live surface. The failure this guards against is not a
// wrong number, it is the absence of any number at all — an operator whose self-hosted model
// gets an unshrunk prompt seeing a surface identical to one where the levers were never on.
func TestShrinkLeverVarsNamesInertLeversOnAForeignWire(t *testing.T) {
	got := shrinkLeverVars(false /*passthrough*/, false /*dualLocal*/, "openai",
		48000 /*compact*/, true /*elideStale*/, true /*deferCold*/)
	if got == nil {
		t.Fatal("a foreign wire with all three levers ON must emit a block; nil IS the silence this closes")
	}
	if got.WireRunsLevers {
		t.Error("WireRunsLevers = true on a non-passthrough wire")
	}
	if len(got.LiveOnWire) != 0 {
		t.Errorf("LiveOnWire = %v on a wire that runs none of them, want empty", got.LiveOnWire)
	}
	want := []string{
		guardvars.ShrinkLeverCompactHistoryBudget,
		guardvars.ShrinkLeverElideStaleReads,
		guardvars.ShrinkLeverDeferColdTools,
	}
	if strings.Join(got.InertOnWire, ",") != strings.Join(want, ",") {
		t.Errorf("InertOnWire = %v, want all three in admission order %v", got.InertOnWire, want)
	}
	if got.Finding != guardvars.FindingShrinkLeverInertOnWire {
		t.Errorf("Finding = %q, want %q — the field's presence is the alarm", got.Finding, guardvars.FindingShrinkLeverInertOnWire)
	}
	if got.Wire != "openai" {
		t.Errorf("Wire = %q, want the resolved provider %q", got.Wire, "openai")
	}
}

// The token is a contract with two readers that never meet: the `fak serve` / `fak guard`
// startup admission writes it to stderr and this block raises it on /debug/vars. A scraper
// matches one string. Pin the literal so a rename has to be a deliberate, visible act.
func TestShrinkLeverTokensAreTheScrapeableContract(t *testing.T) {
	for _, tc := range []struct{ got, want string }{
		{guardvars.FindingShrinkLeverInertOnWire, "SHRINK_LEVER_INERT_ON_WIRE"},
		{guardvars.ShrinkLeverCompactHistoryBudget, "compact_history_budget"},
		{guardvars.ShrinkLeverElideStaleReads, "elide_stale_reads"},
		{guardvars.ShrinkLeverDeferColdTools, "defer_cold_tools"},
	} {
		if tc.got != tc.want {
			t.Errorf("token = %q, want %q (cmd/fak/shrink_lever_wire.go prints the same constant)", tc.got, tc.want)
		}
	}
}

// A healthy passthrough must render too. If the block appeared only when something was wrong,
// its mere presence would become the finding and an operator would learn to skim past it; the
// question "which levers are live" is only answerable if the surface answers it both ways.
func TestShrinkLeverVarsReportsLiveLeversOnThePassthrough(t *testing.T) {
	got := shrinkLeverVars(true, false, "anthropic", 48000, true, true)
	if got == nil {
		t.Fatal("a passthrough with levers ON must still report which ones are live")
	}
	if !got.WireRunsLevers {
		t.Error("WireRunsLevers = false on the Anthropic passthrough")
	}
	if len(got.LiveOnWire) != 3 || len(got.InertOnWire) != 0 {
		t.Errorf("live=%v inert=%v, want all three live and none inert", got.LiveOnWire, got.InertOnWire)
	}
	if got.Finding != "" {
		t.Errorf("Finding = %q on a healthy wire, want empty", got.Finding)
	}
}

// A lever that is OFF is not inert — it is off. Booking it as inert would manufacture a
// finding out of an operator honoring the refusal the startup admission asked for.
func TestShrinkLeverVarsSplitsPartialConfiguration(t *testing.T) {
	got := shrinkLeverVars(false, false, "openai", 0 /*compaction OFF*/, false /*stale OFF*/, true)
	if got == nil {
		t.Fatal("one lever ON on a foreign wire must still emit a block")
	}
	if len(got.InertOnWire) != 1 || got.InertOnWire[0] != guardvars.ShrinkLeverDeferColdTools {
		t.Errorf("InertOnWire = %v, want only %q", got.InertOnWire, guardvars.ShrinkLeverDeferColdTools)
	}
	if got.Finding != guardvars.FindingShrinkLeverInertOnWire {
		t.Errorf("Finding = %q, want the inert token", got.Finding)
	}
}

// All three off: nothing configured, nothing to say. A session that opted out of every lever
// stays quiet rather than emitting an all-empty object on every poll.
func TestShrinkLeverVarsQuietWhenNothingConfigured(t *testing.T) {
	if got := shrinkLeverVars(false, false, "openai", 0, false, false); got != nil {
		t.Fatalf("no lever configured must emit no block, got %+v", got)
	}
	if got := shrinkLeverVars(true, false, "anthropic", 0, false, false); got != nil {
		t.Fatalf("no lever configured must emit no block on the passthrough either, got %+v", got)
	}
}

// Dual mode is the one shape where the process-wide answer is not the per-turn answer: the
// Anthropic proxy side IS a passthrough, but a request naming the local model is not, so those
// turns get none of the three. Report that the ambiguity exists rather than claiming a single
// answer — this is also the deployment most likely to be A/B'd local-vs-remote.
func TestShrinkLeverVarsFlagsDualLocalRouting(t *testing.T) {
	proxy := &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}
	d, err := NewDualPlanner(proxy, &dualSide{id: "tiny-local"}, "tiny-local")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{planner: d, logf: func(string, ...any) {}}
	if !s.dualRoutesLocalModels() {
		t.Fatal("a DualPlanner must report per-turn local routing")
	}
	// Sanity-bind the caveat to the real predicate: the process reads as a passthrough while
	// the local-model request does not.
	if !s.anthropicPassthrough() || s.anthropicPassthroughFor("tiny-local") {
		t.Fatal("fixture no longer reproduces the dual split the caveat exists for")
	}
	got := shrinkLeverVars(s.anthropicPassthrough(), s.dualRoutesLocalModels(), "anthropic", 48000, true, true)
	if got == nil || !got.DualLocalRouting {
		t.Fatalf("dual mode must carry the per-turn caveat, got %+v", got)
	}

	single := &Server{planner: proxy, logf: func(string, ...any) {}}
	if single.dualRoutesLocalModels() {
		t.Error("a single HTTP planner must not report dual local routing")
	}
}

// End-to-end on the real surface: a server whose upstream is a self-hosted OpenAI-wire model,
// with all three levers on, must carry the finding on /debug/vars. This is the assertion that
// the pure builder is actually WIRED — a builder nobody calls closes no silence.
func TestDebugVarsCarriesShrinkLeverFindingOnSelfHostedWire(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = agent.NewHTTPPlanner("http://127.0.0.1:8000/v1", "qwen3-coder-30b", "")
	srv.provider = "openai"
	srv.compactHistoryBudget = 48000
	srv.elideStaleReads = true
	srv.deferColdTools = true

	if srv.anthropicPassthrough() {
		t.Fatal("fixture must be a non-Anthropic upstream")
	}
	vars := srv.debugVars(time.Now())
	if vars.ShrinkLevers == nil {
		t.Fatal("/debug/vars omitted shrink_levers on the wire the levers cannot run on")
	}
	if vars.ShrinkLevers.Finding != guardvars.FindingShrinkLeverInertOnWire {
		t.Fatalf("shrink_levers.finding = %q, want %q", vars.ShrinkLevers.Finding, guardvars.FindingShrinkLeverInertOnWire)
	}
	raw, err := json.Marshal(vars)
	if err != nil {
		t.Fatalf("marshal debug vars: %v", err)
	}
	text := string(raw)
	for _, want := range []string{
		`"shrink_levers"`,
		guardvars.FindingShrinkLeverInertOnWire,
		guardvars.ShrinkLeverCompactHistoryBudget,
		guardvars.ShrinkLeverElideStaleReads,
		guardvars.ShrinkLeverDeferColdTools,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("/debug/vars JSON is missing %q", want)
		}
	}
	// PRIVACY FENCE, same as the startup admission: name the provider, never the upstream
	// URL — an operator-supplied base URL can carry an embedded credential and a host that
	// has no business on a debug surface.
	if b, err := json.Marshal(vars.ShrinkLevers); err != nil {
		t.Fatalf("marshal shrink levers: %v", err)
	} else if strings.Contains(string(b), "127.0.0.1") {
		t.Errorf("shrink_levers leaked the upstream base URL: %s", b)
	}
}

// In-kernel serving wire supports typed prompt-shrink levers; /debug/vars must report
// WireRunsLevers = true, LiveOnWire populated, and Finding empty.
func TestDebugVarsReportsLiveLeversOnInKernelWire(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = agent.NewInKernelPlanner(nil, nil, "synthetic-local", false, nil, false)
	srv.provider = "in-kernel"
	srv.compactHistoryBudget = 48000
	srv.elideStaleReads = true
	srv.deferColdTools = true

	if !srv.wireRunsShrinkLevers() {
		t.Fatal("in-kernel planner must report wireRunsShrinkLevers = true")
	}
	vars := srv.debugVars(time.Now())
	if vars.ShrinkLevers == nil {
		t.Fatal("/debug/vars omitted shrink_levers for in-kernel wire")
	}
	if !vars.ShrinkLevers.WireRunsLevers {
		t.Error("WireRunsLevers = false on in-kernel wire, want true")
	}
	if vars.ShrinkLevers.Finding != "" {
		t.Errorf("Finding = %q, want empty", vars.ShrinkLevers.Finding)
	}
	if len(vars.ShrinkLevers.LiveOnWire) != 3 {
		t.Errorf("LiveOnWire = %v, want 3 active levers", vars.ShrinkLevers.LiveOnWire)
	}
	if len(vars.ShrinkLevers.InertOnWire) != 0 {
		t.Errorf("InertOnWire = %v, want empty", vars.ShrinkLevers.InertOnWire)
	}
}
