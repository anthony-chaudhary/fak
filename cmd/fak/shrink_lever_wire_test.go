package main

// shrink_lever_wire_test.go — the #5493 (`fak serve`) and #5538 (`fak guard`) witness: the
// three biggest prompt-shrink levers are gated on the Anthropic passthrough inside the
// gateway, so on any other upstream wire they are identity by construction. These cases pin
// that a non-passthrough launch of EITHER command with a lever enabled produces a NAMED
// refusal or a NAMED notice — never the silent identity the ticket calls a false null.
//
// The serve cases drive the REAL parsed `fak serve` flag surface (newServeFlagSet + Parse +
// admitServeShrinkLevers), so a lever that is declared but never read, or a check that is
// written but never wired, fails here. Guard registers its flags inline in cmdGuard (which
// spawns a child), so the guard cases drive the resolved-posture entry point directly and
// TestGuardBootWiresTheShrinkLeverAdmission carries the wiring half. Nothing binds a socket
// or dials an upstream.

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestShrinkLeverWireClassification pins the flag→wire classification against gateway.New's own
// planner switch: upstream URLs beat loaded weights (that pair is the dual planner, whose
// proxy side is still the passthrough), 2+ upstreams route through the replica router and
// so are NOT a passthrough even on the anthropic wire, and an unparseable provider stands
// down to unknown so this rule never pre-empts gateway.New's precise refusal.
func TestShrinkLeverWireClassification(t *testing.T) {
	cases := []struct {
		name     string
		provider string
		baseURL  string
		replicas []string
		gguf     string
		want     string
	}{
		{"single anthropic upstream is the passthrough", "anthropic", "https://api.anthropic.example", nil, "", shrinkWireAnthropicPassthrough},
		{"the claude alias parses to anthropic too", "claude", "https://api.anthropic.example", nil, "", shrinkWireAnthropicPassthrough},
		{"a self-hosted openai-wire model is not", "openai", "http://127.0.0.1:8000/v1", nil, "", shrinkWireForeignProxy},
		{"an empty provider defaults to the openai wire", "", "http://127.0.0.1:8000/v1", nil, "", shrinkWireForeignProxy},
		{"gemini is not the passthrough", "gemini", "https://gemini.example", nil, "", shrinkWireForeignProxy},
		{"two anthropic upstreams route through the replica router", "anthropic", "https://a.example", []string{"https://b.example"}, "", shrinkWireReplicaFleet},
		{"replicas alone with no base URL still count", "anthropic", "", []string{"https://a.example", "https://b.example"}, "", shrinkWireReplicaFleet},
		{"one replica and no base URL is a lone upstream", "anthropic", "", []string{"https://a.example"}, "", shrinkWireAnthropicPassthrough},
		{"blank replica entries do not inflate the count", "anthropic", "https://a.example", []string{"   "}, "", shrinkWireAnthropicPassthrough},
		{"weights with an upstream are the dual planner, still a passthrough", "anthropic", "https://api.anthropic.example", nil, "/models/m.gguf", shrinkWireAnthropicPassthrough},
		{"weights with no upstream decode in kernel", "openai", "", nil, "/models/m.gguf", shrinkWireInKernel},
		{"no upstream and no weights is the mock", "openai", "", nil, "", shrinkWireMock},
		{"an unparseable provider is unclassified", "not-a-provider", "http://127.0.0.1:8000/v1", nil, "", shrinkWireUnknown},
	}
	for _, c := range cases {
		if got := classifyShrinkWire(c.provider, c.baseURL, c.replicas, c.gguf); got != c.want {
			t.Errorf("%s: classifyShrinkWire(%q, %q, %v, %q) = %q, want %q",
				c.name, c.provider, c.baseURL, c.replicas, c.gguf, got, c.want)
		}
	}
}

// TestShrinkLeverWireRunsLeversOnlyOnPassthroughOrUnknown pins the two admitting classes. An
// unknown wire admits on purpose: an unproven classification must never refuse a boot.
func TestShrinkLeverWireRunsLeversOnlyOnPassthroughOrUnknown(t *testing.T) {
	runs := map[string]bool{
		shrinkWireAnthropicPassthrough: true,
		shrinkWireInKernel:             true,
		shrinkWireUnknown:              true,
		shrinkWireForeignProxy:         false,
		shrinkWireReplicaFleet:         false,
		shrinkWireMock:                 false,
	}
	for wire, want := range runs {
		if got := shrinkWireRunsLevers(wire); got != want {
			t.Errorf("shrinkWireRunsLevers(%q) = %v, want %v", wire, got, want)
		}
		// The advisory is the strictly narrower set: everything that does not run the
		// levers warrants a notice EXCEPT the scripted mock planner.
		wantNotice := !want && wire != shrinkWireMock
		if got := shrinkWireNoticeWarranted(wire); got != wantNotice {
			t.Errorf("shrinkWireNoticeWarranted(%q) = %v, want %v", wire, got, wantNotice)
		}
	}
}

// TestInertShrinkLeversSelectsOnlyEnabledLeversOffTheWire pins the pure selection: on a
// passthrough nothing is inert regardless of what is on, and off it exactly the ON levers
// are named (never an off one, which would refuse a boot for a lever nobody asked for).
func TestInertShrinkLeversSelectsOnlyEnabledLeversOffTheWire(t *testing.T) {
	all := []shrinkLever{
		{Flag: "--compact-history-budget", Token: "compact_history_budget", On: true},
		{Flag: "--elide-stale-reads", Token: "elide_stale_reads", On: false},
		{Flag: "--defer-cold-tools", Token: "defer_cold_tools", On: true},
	}
	if inert := inertShrinkLevers(shrinkWireAnthropicPassthrough, all); len(inert) != 0 {
		t.Errorf("levers on the passthrough must never be inert, got %d: %v", len(inert), inert)
	}
	if inert := inertShrinkLevers(shrinkWireInKernel, all); len(inert) != 0 {
		t.Errorf("levers on the in-kernel wire must never be inert, got %d: %v", len(inert), inert)
	}
	if inert := inertShrinkLevers(shrinkWireUnknown, all); len(inert) != 0 {
		t.Errorf("an unclassified wire must admit everything, got %d: %v", len(inert), inert)
	}
	inert := inertShrinkLevers(shrinkWireForeignProxy, all)
	if len(inert) != 2 {
		t.Fatalf("off the passthrough exactly the 2 enabled levers are inert, got %d: %v", len(inert), inert)
	}
	if inert[0].Token != "compact_history_budget" || inert[1].Token != "defer_cold_tools" {
		t.Errorf("inert set must keep the caller's order and skip the disabled lever, got %v", inert)
	}
}

// TestShrinkLeverRefusalCarriesTokenLeverWireAndRemedy pins the refusal CONTENT the ticket's
// definition of done requires: a typed token, the lever by name, the wire it is inert on,
// and the exact argument that resolves it. An empty explicit set renders nothing.
func TestShrinkLeverRefusalCarriesTokenLeverWireAndRemedy(t *testing.T) {
	if got := shrinkLeverRefusal(shrinkLeverServeCommand, shrinkWireForeignProxy, "openai", nil); got != "" {
		t.Errorf("no explicitly enabled lever must render no refusal, got %q", got)
	}
	explicit := []shrinkLever{{
		Flag: "--defer-cold-tools", Token: "defer_cold_tools",
		Gate: "gateway.maybeDeferColdTools", Off: "--defer-cold-tools=false",
		On: true, Explicit: true,
	}}
	got := shrinkLeverRefusal(shrinkLeverServeCommand, shrinkWireForeignProxy, "openai", explicit)
	for _, want := range []string{
		"fak serve:",                  // the command that is refusing
		shrinkLeverInertToken,         // the typed, scrapeable reason
		"--defer-cold-tools",          // the lever, by its operator-facing name
		"defer_cold_tools",            // the lever, by its stable token
		"gateway.maybeDeferColdTools", // the seam that stands it down
		"non-Anthropic proxy",         // the WIRE it is inert on
		"--provider openai",           // the provider the operator actually configured
		"--defer-cold-tools=false",    // the remedy, spelled out
		"--provider anthropic",        // the other remedy
	} {
		if !strings.Contains(got, want) {
			t.Errorf("refusal must name %q; got:\n%s", want, got)
		}
	}
	// The refusal must never echo an upstream URL: a base URL can carry an embedded
	// credential and can name a host that has no business in a log line.
	if strings.Contains(got, "http") {
		t.Errorf("refusal must not echo an upstream URL; got:\n%s", got)
	}
}

// TestShrinkLeverNoticeNamesTheFalseNull pins that the default-on advisory says the thing
// that prevents the ticket's second-order harm: a ~0-saving A/B on this wire is a verdict
// on an unwired lever, not on the kernel.
func TestShrinkLeverNoticeNamesTheFalseNull(t *testing.T) {
	if got := shrinkLeverNotice(shrinkLeverServeCommand, shrinkWireForeignProxy, "openai", nil); got != "" {
		t.Errorf("no default-on inert lever must render no notice, got %q", got)
	}
	got := shrinkLeverNotice(shrinkLeverServeCommand, shrinkWireForeignProxy, "openai", []shrinkLever{{
		Flag: "--compact-history-budget", Token: "compact_history_budget",
		Gate: "gateway.compactAnthropicRawWithReason", Off: "--compact-history-budget 0", On: true,
	}})
	for _, want := range []string{
		shrinkLeverInertToken, "--compact-history-budget", "compact_history_budget",
		"non-Anthropic proxy", "A/B", "unwired lever",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("notice must name %q; got:\n%s", want, got)
		}
	}
}

// TestAdmitServeShrinkLeversOnRealFlagSurface is the end-to-end witness. It parses the REAL
// `fak serve` flag surface, so a lever the check forgot to read — or a check nobody wired —
// shows up here. It binds nothing: the rule reads flag values only.
//
// The severity split is the load-bearing assertion. A lever the operator TYPED is a stated
// intent this wire cannot honor, so the boot is refused; a lever that is merely shipped
// default-on gets a named notice and serves, because refusing there would take down every
// ordinary non-Anthropic `fak serve`.
func TestAdmitServeShrinkLeversOnRealFlagSurface(t *testing.T) {
	const local = "http://127.0.0.1:8000/v1" // a stand-in self-hosted OpenAI-wire upstream
	cases := []struct {
		name      string
		argv      []string
		wantAdmit bool
		wantAll   []string // substrings stderr must carry (empty slice = stderr must stay empty)
		wantNone  []string // substrings stderr must NOT carry
	}{
		{
			name:      "the anthropic passthrough runs every lever and says nothing",
			argv:      []string{"-provider", "anthropic", "-base-url", "https://api.anthropic.example"},
			wantAdmit: true,
		},
		{
			name:      "an explicitly deferred cold tail on a self-hosted openai wire is refused by name",
			argv:      []string{"-provider", "openai", "-base-url", local, "-defer-cold-tools"},
			wantAdmit: false,
			wantAll:   []string{shrinkLeverInertToken, "refusing to start", "--defer-cold-tools", "defer_cold_tools", "non-Anthropic proxy"},
		},
		{
			name:      "an explicit compaction budget on a self-hosted openai wire is refused by name",
			argv:      []string{"-provider", "openai", "-base-url", local, "-compact-history-budget", "96000"},
			wantAdmit: false,
			wantAll:   []string{shrinkLeverInertToken, "--compact-history-budget", "compact_history_budget", "--compact-history-budget 0"},
		},
		{
			name:      "all three explicitly enabled are all three named in one refusal",
			argv:      []string{"-provider", "openai", "-base-url", local, "-compact-history-budget", "96000", "-elide-stale-reads", "-defer-cold-tools"},
			wantAdmit: false,
			wantAll:   []string{"3 prompt-shrink lever(s)", "--compact-history-budget", "--elide-stale-reads", "--defer-cold-tools"},
		},
		{
			name:      "explicitly turning every lever off on the same wire admits silently",
			argv:      []string{"-provider", "openai", "-base-url", local, "-compact-history-budget", "0", "-elide-stale-reads=false", "-defer-cold-tools=false"},
			wantAdmit: true,
		},
		{
			name:      "the shipped default-on levers on a self-hosted openai wire serve, loudly named",
			argv:      []string{"-provider", "openai", "-base-url", local},
			wantAdmit: true,
			wantAll:   []string{shrinkLeverInertToken, "NOTICE", "--compact-history-budget", "--elide-stale-reads", "--defer-cold-tools", "unwired lever"},
			wantNone:  []string{"refusing to start"},
		},
		{
			name:      "an in-kernel serve supports prompt-shrink levers and admits silently on default levers",
			argv:      []string{"-gguf", "/models/m.gguf"},
			wantAdmit: true,
			wantNone:  []string{shrinkLeverInertToken, "refusing to start", "NOTICE"},
		},
		{
			name:      "explicit prompt-shrink levers are active and accepted on in-kernel serve wire",
			argv:      []string{"-gguf", "/models/m.gguf", "-compact-history-budget", "32000", "-elide-stale-reads", "-defer-cold-tools"},
			wantAdmit: true,
			wantNone:  []string{shrinkLeverInertToken, "refusing to start", "NOTICE"},
		},
		{
			name:      "a replica fleet is not a passthrough even on the anthropic wire",
			argv:      []string{"-provider", "anthropic", "-base-url", "https://a.example", "-replica-base-url", "https://b.example"},
			wantAdmit: true,
			wantAll:   []string{shrinkLeverInertToken, "replica fleet"},
		},
		{
			name:      "an explicit lever is refused on the replica fleet too",
			argv:      []string{"-provider", "anthropic", "-base-url", "https://a.example", "-replica-base-url", "https://b.example", "-defer-cold-tools"},
			wantAdmit: false,
			wantAll:   []string{shrinkLeverInertToken, "refusing to start", "replica fleet"},
		},
		{
			name:      "the scripted mock planner is not advised (it can produce no measurement to falsify)",
			argv:      nil,
			wantAdmit: true,
		},
		{
			name:      "an explicit lever is still refused on the mock",
			argv:      []string{"-defer-cold-tools"},
			wantAdmit: false,
			wantAll:   []string{shrinkLeverInertToken, "mock planner"},
		},
		{
			name:      "an unparseable provider defers to gateway.New's own refusal",
			argv:      []string{"-provider", "not-a-provider", "-base-url", local, "-defer-cold-tools"},
			wantAdmit: true,
		},
	}
	for _, c := range cases {
		fs, sf := newServeFlagSet()
		if err := fs.Parse(c.argv); err != nil {
			t.Fatalf("%s: parse %v: %v", c.name, c.argv, err)
		}
		var stderr bytes.Buffer
		got := admitServeShrinkLevers(fs, sf, &stderr)
		if got != c.wantAdmit {
			t.Errorf("%s: admitServeShrinkLevers(%v) = %v, want %v (stderr: %s)", c.name, c.argv, got, c.wantAdmit, stderr.String())
		}
		if len(c.wantAll) == 0 && stderr.Len() != 0 {
			t.Errorf("%s: admitServeShrinkLevers(%v) must stay quiet, wrote: %s", c.name, c.argv, stderr.String())
		}
		for _, want := range c.wantAll {
			if !strings.Contains(stderr.String(), want) {
				t.Errorf("%s: stderr must contain %q, got: %s", c.name, want, stderr.String())
			}
		}
		for _, none := range c.wantNone {
			if strings.Contains(stderr.String(), none) {
				t.Errorf("%s: stderr must NOT contain %q, got: %s", c.name, none, stderr.String())
			}
		}
	}
}

// TestServeShrinkLeversReadsExplicitnessFromTheParsedFlags pins the distinction the whole
// severity split rests on: a flag the operator TYPED is Explicit even when its value equals
// the shipped default, and an untyped flag is not — so `--defer-cold-tools` (default true)
// refuses while an untouched default only warns.
func TestServeShrinkLeversReadsExplicitnessFromTheParsedFlags(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"-defer-cold-tools=true", "-compact-history-budget", "0"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	byToken := map[string]shrinkLever{}
	for _, l := range serveShrinkLevers(fs, sf) {
		byToken[l.Token] = l
	}
	if len(byToken) != 3 {
		t.Fatalf("all three levers must be read, got %d: %v", len(byToken), byToken)
	}
	if l := byToken["defer_cold_tools"]; !l.Explicit || !l.On {
		t.Errorf("a typed --defer-cold-tools=true must be Explicit and On, got %+v", l)
	}
	if l := byToken["compact_history_budget"]; !l.Explicit || l.On {
		t.Errorf("a typed --compact-history-budget 0 must be Explicit and OFF, got %+v", l)
	}
	if l := byToken["elide_stale_reads"]; l.Explicit {
		t.Errorf("an untyped --elide-stale-reads must not read as Explicit, got %+v", l)
	}
	// Every lever must carry its own remedy and name its own gateway seam, or the
	// refusal text degrades into prose the operator cannot act on.
	for token, l := range byToken {
		if strings.TrimSpace(l.Off) == "" || strings.TrimSpace(l.Gate) == "" || strings.TrimSpace(l.Flag) == "" {
			t.Errorf("lever %q must carry Flag, Gate and Off, got %+v", token, l)
		}
	}
}

// ---------------------------------------------------------------------------------------
// #5538 — the `fak guard` sibling. Guard sets the same three gateway.Config fields
// (guard.go: CompactHistoryBudget, ElideStaleReads, DeferColdTools) and reaches a
// non-passthrough wire at least as easily as serve does (--local forces provider=openai,
// --remote-serve forces the OpenAI-compatible wire, a bare --gguf makes the in-kernel
// planner the upstream), so the same admission has to run there.
// ---------------------------------------------------------------------------------------

// guardDefaultShrinkLeverBoot returns the lever half of a `fak guard` launch on which the
// operator typed NONE of the three flags — i.e. the shipped defaults, taken from the gateway
// constants guard registers them from, with the compaction budget passed through guard's own
// resolveGuardCompactBudget so the number under test is the number gateway.Config carries.
func guardDefaultShrinkLeverBoot() guardShrinkLeverInputs {
	return guardShrinkLeverInputs{
		SetFlags:             map[string]bool{},
		CompactHistoryBudget: resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, false),
		ElideStaleReads:      gateway.DefaultElideStaleReads,
		DeferColdTools:       gateway.DefaultDeferColdTools,
	}
}

// TestGuardShipsAllThreeLeversOnByDefault pins the premise the whole guard admission rests
// on: on a launch where the operator typed nothing, all three levers are ON. That is what
// makes "silently inert" the DEFAULT experience for a self-hosted guard rather than an edge
// case — if any of these three ever flipped default-off, the urgency of this rule changes and
// this test is the place that says so.
func TestGuardShipsAllThreeLeversOnByDefault(t *testing.T) {
	in := guardDefaultShrinkLeverBoot()
	levers := shrinkLevers(in.SetFlags, in.CompactHistoryBudget, in.ElideStaleReads, in.DeferColdTools)
	if len(levers) != 3 {
		t.Fatalf("three levers expected, got %d: %v", len(levers), levers)
	}
	for _, l := range levers {
		if !l.On {
			t.Errorf("%s ships default-ON for `fak guard`; got On=false (%+v)", l.Flag, l)
		}
		if l.Explicit {
			t.Errorf("%s was not typed by the operator, so it must not read as Explicit (%+v)", l.Flag, l)
		}
	}
}

// TestShrinkLeverCommandRemediesAreCommandCorrect pins the one thing that legitimately
// differs between the two admissions. A remedy is the operator's only exit from a refusal, so
// naming serve's flag pair inside a guard message (or vice versa) would send them to a flag
// combination that does not reach the passthrough on the command they actually ran.
func TestShrinkLeverCommandRemediesAreCommandCorrect(t *testing.T) {
	for _, c := range []struct {
		cmd     shrinkLeverCommand
		wantAll []string
	}{
		{shrinkLeverServeCommand, []string{"fak serve", "--provider anthropic", "--base-url"}},
		// Guard reaches the passthrough by DEFAULT, so its remedy has to name the default
		// launch AND the three flags that steer off it — not serve's --provider/--base-url pair.
		{shrinkLeverGuardCommand, []string{"fak guard", "fak guard -- claude", "--provider anthropic", "--local", "--remote-serve", "--gguf"}},
	} {
		if strings.TrimSpace(c.cmd.Name) == "" || strings.TrimSpace(c.cmd.ToWire) == "" {
			t.Fatalf("%+v: a command must carry both a Name and a ToWire remedy", c.cmd)
		}
		joined := c.cmd.Name + " " + c.cmd.ToWire
		for _, want := range c.wantAll {
			if !strings.Contains(joined, want) {
				t.Errorf("%s remedy must name %q; got %q", c.cmd.Name, want, c.cmd.ToWire)
			}
		}
		// PRIVACY FENCE: a remedy is rendered into every refusal and notice, so it must be
		// flag vocabulary only — never a host or URL.
		if strings.Contains(c.cmd.ToWire, "http") {
			t.Errorf("%s remedy must not carry a URL; got %q", c.cmd.Name, c.cmd.ToWire)
		}
	}
	if shrinkLeverServeCommand.ToWire == shrinkLeverGuardCommand.ToWire {
		t.Error("the two commands reach the passthrough differently; identical remedies mean one of them is wrong")
	}
}

// TestGuardRawFlagsWouldMisclassifyTheFlagshipWire is the crux of why the guard entry point
// takes RESOLVED values rather than the parsed FlagSet serve uses. Read off guard's raw flags,
// the flagship `fak guard -- claude` — no --provider, no --base-url, no --gguf — looks exactly
// like serve's mock planner, and an operator who typed `--defer-cold-tools` on it would be
// refused on the ONE wire that runs all three levers. Read off the resolved posture (guard's
// own resolveGuardProvider + guardDefaultBaseURL), it is the Anthropic passthrough and admits
// in silence. If someone later "simplifies" the guard call site back to raw flags, this fails.
func TestGuardRawFlagsWouldMisclassifyTheFlagshipWire(t *testing.T) {
	// As TYPED: every wire-deciding flag empty.
	if raw := classifyShrinkWire("", "", nil, ""); raw != shrinkWireMock {
		t.Fatalf("guard's raw flags for `fak guard -- claude` classify as %q; this test's premise assumed %q", raw, shrinkWireMock)
	}
	// As RESOLVED by guard itself.
	provider, autodetected := resolveGuardProvider("", "claude")
	if provider != "anthropic" || !autodetected {
		t.Fatalf("resolveGuardProvider(\"\", \"claude\") = (%q, %v), want (\"anthropic\", true)", provider, autodetected)
	}
	base := guardDefaultBaseURL(provider)
	if strings.TrimSpace(base) == "" {
		t.Fatalf("guardDefaultBaseURL(%q) is empty; the flagship guard upstream must resolve to the public API", provider)
	}
	if got := classifyShrinkWire(provider, base, nil, ""); got != shrinkWireAnthropicPassthrough {
		t.Fatalf("resolved `fak guard -- claude` = %q, want %q", got, shrinkWireAnthropicPassthrough)
	}

	// The false refusal the resolved reading avoids, demonstrated end to end.
	typed := guardDefaultShrinkLeverBoot()
	typed.SetFlags = map[string]bool{"defer-cold-tools": true}
	rawRead := typed
	var rawErr bytes.Buffer
	if admitGuardShrinkLevers(rawRead, &rawErr) {
		t.Fatalf("premise check: reading guard's RAW empty flags must refuse a typed --defer-cold-tools (that is the bug this test fences), stderr: %s", rawErr.String())
	}
	resolved := typed
	resolved.Provider, resolved.BaseURL = provider, base
	var resolvedErr bytes.Buffer
	if !admitGuardShrinkLevers(resolved, &resolvedErr) {
		t.Errorf("the flagship `fak guard -- claude` with a typed --defer-cold-tools must ADMIT; got a refusal: %s", resolvedErr.String())
	}
	if resolvedErr.Len() != 0 {
		t.Errorf("the passthrough wire runs every lever, so it must stay silent; got: %s", resolvedErr.String())
	}
}

// TestAdmitGuardShrinkLeversOnResolvedGuardBoot is the guard-side end-to-end witness the
// ticket asks for: a non-passthrough guard wire with an EXPLICITLY typed lever refuses, and a
// default-only lever produces a named notice rather than silence.
//
// Provider/BaseURL are the RESOLVED pair guard hands to gateway.Config — see the case comments
// for which guard flag combination produces each. Nothing here dials, binds, or loads weights.
func TestAdmitGuardShrinkLeversOnResolvedGuardBoot(t *testing.T) {
	const anthropicBase = "https://api.anthropic.example" // stand-in for the resolved public API
	const localBase = "http://127.0.0.1:11434/v1"         // stand-in for a detected local server
	cases := []struct {
		name      string
		provider  string
		baseURL   string
		gguf      string
		typed     []string // flag names the operator named on the command line
		off       []string // flag names the operator explicitly turned OFF
		wantAdmit bool
		wantAll   []string
		wantNone  []string
	}{
		{
			name:     "`fak guard -- claude` runs every lever and says nothing",
			provider: "anthropic", baseURL: anthropicBase,
			wantAdmit: true,
		},
		{
			name:     "`fak guard --local -- codex` keeps default-on inert levers quiet",
			provider: "openai", baseURL: localBase,
			wantAdmit: true,
		},
		{
			name:     "a typed --defer-cold-tools on a local server is refused by name",
			provider: "openai", baseURL: localBase, typed: []string{"defer-cold-tools"},
			wantAdmit: false,
			wantAll: []string{
				"fak guard:", shrinkLeverInertToken, "refusing to start",
				"--defer-cold-tools", "defer_cold_tools", "gateway.maybeDeferColdTools",
				"--defer-cold-tools=false", "fak guard -- claude",
			},
		},
		{
			name:     "a typed --compact-history-budget on the --remote-serve OpenAI wire is refused by name",
			provider: "openai", baseURL: "http://lab-box.invalid:8080/v1", typed: []string{"compact-history-budget"},
			wantAdmit: false,
			wantAll: []string{
				shrinkLeverInertToken, "refusing to start",
				"--compact-history-budget", "compact_history_budget", "--compact-history-budget 0",
			},
		},
		{
			name:     "`fak guard -- codex` keeps default-on Responses-wire levers quiet",
			provider: "openai-responses", baseURL: "https://api.openai.example/v1",
			wantAdmit: true,
		},
		{
			name:     "a bare --gguf keeps default-on in-kernel levers quiet",
			provider: "anthropic", baseURL: "", gguf: "/models/m.gguf",
			wantAdmit: true,
		},
		{
			name:      "a typed --elide-stale-reads on the in-kernel upstream is admitted and active",
			provider:  "anthropic", baseURL: "", gguf: "/models/m.gguf", typed: []string{"elide-stale-reads"},
			wantAdmit: true,
			wantNone:  []string{shrinkLeverInertToken, "refusing to start"},
		},
		{
			name:      "a typed --compact-history-budget on the in-kernel upstream is admitted and active",
			provider:  "anthropic", baseURL: "", gguf: "/models/m.gguf", typed: []string{"compact-history-budget"},
			wantAdmit: true,
			wantNone:  []string{shrinkLeverInertToken, "refusing to start"},
		},
		{
			name:      "a typed --defer-cold-tools on the in-kernel upstream is admitted and active",
			provider:  "anthropic", baseURL: "", gguf: "/models/m.gguf", typed: []string{"defer-cold-tools"},
			wantAdmit: true,
			wantNone:  []string{shrinkLeverInertToken, "refusing to start"},
		},
		{
			name:     "--gguf ALONGSIDE an anthropic upstream keeps the passthrough on the proxy side",
			provider: "anthropic", baseURL: anthropicBase, gguf: "/models/m.gguf",
			typed:     []string{"defer-cold-tools"},
			wantAdmit: true,
		},
		{
			name:     "stating every lever off on a local server admits in silence",
			provider: "openai", baseURL: localBase,
			typed:     []string{"compact-history-budget", "elide-stale-reads", "defer-cold-tools"},
			off:       []string{"compact-history-budget", "elide-stale-reads", "defer-cold-tools"},
			wantAdmit: true,
		},
		{
			name:     "an unparseable --provider defers to gateway.New's own refusal",
			provider: "not-a-provider", baseURL: localBase, typed: []string{"defer-cold-tools"},
			wantAdmit: true,
		},
	}
	for _, c := range cases {
		in := guardDefaultShrinkLeverBoot()
		in.Provider, in.BaseURL, in.GGUFPath = c.provider, c.baseURL, c.gguf
		for _, name := range c.typed {
			in.SetFlags[name] = true
		}
		for _, name := range c.off {
			switch name {
			case "compact-history-budget":
				in.CompactHistoryBudget = 0
			case "elide-stale-reads":
				in.ElideStaleReads = false
			case "defer-cold-tools":
				in.DeferColdTools = false
			}
		}
		var stderr bytes.Buffer
		got := admitGuardShrinkLevers(in, &stderr)
		if got != c.wantAdmit {
			t.Errorf("%s: admitGuardShrinkLevers = %v, want %v (stderr: %s)", c.name, got, c.wantAdmit, stderr.String())
		}
		if len(c.wantAll) == 0 && stderr.Len() != 0 {
			t.Errorf("%s: admitGuardShrinkLevers must stay quiet, wrote: %s", c.name, stderr.String())
		}
		for _, want := range c.wantAll {
			if !strings.Contains(stderr.String(), want) {
				t.Errorf("%s: stderr must contain %q, got: %s", c.name, want, stderr.String())
			}
		}
		for _, none := range c.wantNone {
			if strings.Contains(stderr.String(), none) {
				t.Errorf("%s: stderr must NOT contain %q, got: %s", c.name, none, stderr.String())
			}
		}
		// PRIVACY FENCE, guard side: an upstream base URL can carry an embedded credential
		// and can name a host that has no business in a log line, so no message may echo it.
		if c.baseURL != "" && strings.Contains(stderr.String(), c.baseURL) {
			t.Errorf("%s: message must not echo the upstream base URL, got: %s", c.name, stderr.String())
		}
	}
}

// TestGuardBootWiresTheShrinkLeverAdmission is the WIRING witness. The rule above is pure, so
// nothing in it can tell whether cmdGuard ever calls it — a check that is written but never
// wired would leave every case green and the defect shipped. Serve gets this for free (its
// test drives newServeFlagSet + admitServeShrinkLevers); guard registers its flags inline in
// cmdGuard, which spawns a child process, so the call site is asserted at the source instead.
//
// It also pins the ORDERING that makes the admission cheap: it must run before the in-kernel
// weight load, so an operator with a typed-but-inert lever is refused in milliseconds rather
// than after a multi-GB pull.
func TestGuardBootWiresTheShrinkLeverAdmission(t *testing.T) {
	src, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatalf("read guard.go: %v", err)
	}
	text := string(src)
	call := strings.Index(text, "admitGuardShrinkLevers(guardShrinkLeverInputs{")
	if call < 0 {
		t.Fatal("cmdGuard must call admitGuardShrinkLevers — the three levers are set in gateway.Config here (#5538)")
	}
	// The three fields the admission exists to police must still be the ones guard sets.
	for _, field := range []string{"CompactHistoryBudget: *compactHistoryBudget", "ElideStaleReads:", "DeferColdTools: *deferColdTools"} {
		if !strings.Contains(text, field) {
			t.Errorf("guard.go no longer sets %q; the shrink-lever admission may be policing the wrong fields", field)
		}
	}
	load := strings.Index(text, "loadServeInKernelModel(")
	if load < 0 {
		t.Fatal("guard.go must still load the in-kernel model; the ordering assertion below has lost its anchor")
	}
	if call > load {
		t.Errorf("the shrink-lever admission (offset %d) must run BEFORE the in-kernel weight load (offset %d) so a refusal costs milliseconds, not a multi-GB pull", call, load)
	}
}

func TestGuardCompactStartupSuppressesDefaultShrinkLeverEssay(t *testing.T) {
	in := guardDefaultShrinkLeverBoot()
	in.Provider = "openai-responses"
	in.BaseURL = "https://api.openai.com/v1"
	var stderr strings.Builder
	if !admitGuardShrinkLevers(in, &stderr) {
		t.Fatal("default-on inert levers must not refuse startup")
	}
	if got := stderr.String(); got != "" {
		t.Fatalf("compact startup leaked inert-lever essay: %q", got)
	}

	// Silence applies only to defaults. An operator-explicit inert lever remains a
	// refusal because continuing would misrepresent the requested configuration.
	in.SetFlags = map[string]bool{"elide-stale-reads": true}
	stderr.Reset()
	if admitGuardShrinkLevers(in, &stderr) {
		t.Fatal("explicit inert lever unexpectedly admitted")
	}
	if got := stderr.String(); !strings.Contains(got, "SHRINK_LEVER_INERT_ON_WIRE") {
		t.Fatalf("explicit refusal missing from compact startup: %q", got)
	}
}

// TestShrinkLeverWireInKernel verifies that --compact-history-budget,
// --elide-stale-reads, and --defer-cold-tools are accepted and active on the in-kernel wire.
func TestShrinkLeverWireInKernel(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"compact history budget", []string{"-gguf", "/models/m.gguf", "-compact-history-budget", "48000"}},
		{"elide stale reads", []string{"-gguf", "/models/m.gguf", "-elide-stale-reads=true"}},
		{"defer cold tools", []string{"-gguf", "/models/m.gguf", "-defer-cold-tools=true"}},
		{"all levers combined", []string{"-gguf", "/models/m.gguf", "-compact-history-budget", "32000", "-elide-stale-reads", "-defer-cold-tools"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, sf := newServeFlagSet()
			if err := fs.Parse(tc.args); err != nil {
				t.Fatalf("parse %v: %v", tc.args, err)
			}
			var stderr bytes.Buffer
			if !admitServeShrinkLevers(fs, sf, &stderr) {
				t.Fatalf("admitServeShrinkLevers(%v) refused: %s", tc.args, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("admitServeShrinkLevers(%v) wrote unexpected output: %s", tc.args, stderr.String())
			}
		})
	}
}
