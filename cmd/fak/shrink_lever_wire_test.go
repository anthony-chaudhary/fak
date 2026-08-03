package main

// shrink_lever_wire_test.go — the #5493 witness: the three biggest prompt-shrink levers are
// gated on the Anthropic passthrough inside the gateway, so on any other upstream wire they
// are identity by construction. These cases pin that a non-passthrough `fak serve` with a
// lever enabled produces either a NAMED refusal or a NAMED notice — never the silent
// identity the ticket calls a false null.
//
// Every admission case drives the REAL parsed `fak serve` flag surface (newServeFlagSet +
// Parse + admitServeShrinkLevers), so a lever that is declared but never read, or a check
// that is written but never wired, fails here. Nothing binds a socket or dials an upstream.

import (
	"bytes"
	"strings"
	"testing"
)

// TestClassifyShrinkWireMatrix pins the flag→wire classification against gateway.New's own
// planner switch: upstream URLs beat loaded weights (that pair is the dual planner, whose
// proxy side is still the passthrough), 2+ upstreams route through the replica router and
// so are NOT a passthrough even on the anthropic wire, and an unparseable provider stands
// down to unknown so this rule never pre-empts gateway.New's precise refusal.
func TestClassifyShrinkWireMatrix(t *testing.T) {
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

// TestShrinkWireRunsLeversOnlyOnPassthroughOrUnknown pins the two admitting classes. An
// unknown wire admits on purpose: an unproven classification must never refuse a boot.
func TestShrinkWireRunsLeversOnlyOnPassthroughOrUnknown(t *testing.T) {
	runs := map[string]bool{
		shrinkWireAnthropicPassthrough: true,
		shrinkWireUnknown:              true,
		shrinkWireForeignProxy:         false,
		shrinkWireReplicaFleet:         false,
		shrinkWireInKernel:             false,
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
	if got := shrinkLeverRefusal(shrinkWireForeignProxy, "openai", nil); got != "" {
		t.Errorf("no explicitly enabled lever must render no refusal, got %q", got)
	}
	explicit := []shrinkLever{{
		Flag: "--defer-cold-tools", Token: "defer_cold_tools",
		Gate: "gateway.maybeDeferColdTools", Off: "--defer-cold-tools=false",
		On: true, Explicit: true,
	}}
	got := shrinkLeverRefusal(shrinkWireForeignProxy, "openai", explicit)
	for _, want := range []string{
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
	if got := shrinkLeverNotice(shrinkWireInKernel, "openai", nil); got != "" {
		t.Errorf("no default-on inert lever must render no notice, got %q", got)
	}
	got := shrinkLeverNotice(shrinkWireInKernel, "openai", []shrinkLever{{
		Flag: "--compact-history-budget", Token: "compact_history_budget",
		Gate: "gateway.compactAnthropicRawWithReason", Off: "--compact-history-budget 0", On: true,
	}})
	for _, want := range []string{
		shrinkLeverInertToken, "--compact-history-budget", "compact_history_budget",
		"in-kernel planner", "A/B", "unwired lever",
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
			name:      "an in-kernel serve is the same self-hosted case and is named too",
			argv:      []string{"-gguf", "/models/m.gguf"},
			wantAdmit: true,
			wantAll:   []string{shrinkLeverInertToken, "in-kernel planner"},
			wantNone:  []string{"refusing to start"},
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
