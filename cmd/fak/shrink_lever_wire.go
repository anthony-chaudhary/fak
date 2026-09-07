// shrink_lever_wire.go — the prompt-shrink-lever WIRE admission rule for the two commands
// that bind a gateway: `fak serve` (#5493) and `fak guard` (#5538).
//
// The three largest prompt-shrink levers this gateway ships are supported on the
// Anthropic passthrough wire (Server.anthropicPassthroughFor) and the native in-kernel
// wire (agent.InKernelPlanner via ApplyTypedPromptShrinkLevers):
//
//   - --compact-history-budget → gateway.compactAnthropicRawWithReason / agent.CompactMessagesWithOptions
//   - --elide-stale-reads      → gateway.maybeElideStaleReads / agent.ElideStaleReadMessages
//   - --defer-cold-tools       → gateway.maybeDeferColdTools / agent.DeferColdToolDefs
//
// Each of those three returns identity the moment the request is not on a supported wire, so an
// operator whose upstream is a self-hosted OpenAI-wire model gets NONE of them — and, until
// this rule, got no signal at all that the levers they configured were inert. That silence
// is the expensive half of the defect: an A/B run on such a wire measures ~0 saving and the
// number reads as a verdict on the kernel, when it is only a verdict on an unwired lever.
//
// This file closes the silence at the ONE boundary that can be checked before anything
// binds or loads: parsed flags. It is deliberately shaped like serve_bind_safety.go —
// pure, total, no I/O, no socket — so the whole wire × lever matrix is table-testable.
//
// The severity split is the point, and it is not arbitrary:
//
//   - A lever the operator EXPLICITLY named on the command line is a stated intent that
//     this wire cannot honor, so it is a startup REFUSAL by name. There is no override
//     flag because the honest escape is already spelled by the lever itself — turn it off
//     (--defer-cold-tools=false, --compact-history-budget 0) or move to the wire that runs
//     it. A run that is refused here never gets to be quoted as a lever result.
//   - A lever that is merely ON BY DEFAULT (two of the three are, and the third flipped
//     default-on in #3537) was not enabled by this operator, so refusing would break every
//     ordinary non-Anthropic serve. Those get a loud, named NOTICE instead.
//
// PRIVACY FENCE: the messages name the PROVIDER and the wire class only. They never echo
// --base-url, because an operator-supplied upstream URL can carry an embedded credential
// and can name a host that has no business in a log line.
//
// GUARD IS NOT SERVE, and the difference is load-bearing (#5538). Serve's --provider and
// --base-url flags ARE its wire, so serve can classify straight off the parsed FlagSet.
// Guard's are not: an empty --base-url there means "the provider's public API" rather than
// serve's mock, --provider is auto-detected from the wrapped agent's name when unset
// (claude→anthropic, codex→openai-responses), and --local / --remote-serve REWRITE both
// behind the operator's back. Classifying guard off its raw flags would read the flagship
// `fak guard -- claude` as the mock planner and refuse it — a false refusal of the one wire
// that runs all three levers. So the guard entry point takes the RESOLVED provider and base
// URL (guard_upstream_posture.go) and feeds them to the SAME classifier; only the operator-
// facing command name and the "move to the wire that runs these levers" remedy differ, and
// those are carried by shrinkLeverCommand rather than a forked second rule.
package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// shrinkLeverInertToken is the stable, structured reason `fak serve` names when a
// prompt-shrink lever cannot run on the configured wire, so an operator (or a log scrape)
// can match it by token rather than by prose that may be reworded.
//
// It is the SAME token the live /debug/vars shrink_levers block raises as its Finding
// (internal/gateway/shrink_lever_live.go), taken from the one definition in internal/guardvars
// rather than re-spelled here: a scraper that watches boot stderr and a scraper that polls
// /debug/vars must match the identical string, and two literals would let that quietly rot.
const shrinkLeverInertToken = guardvars.FindingShrinkLeverInertOnWire

// shrinkLeverCommand carries the ONLY two things that differ between the `fak serve` and
// `fak guard` admissions: how the process names itself in the message, and what "move to the
// wire that runs these levers" concretely means there. Everything else — the classifier, the
// severity split, the lever table, the privacy fence — is shared, so a fix to the rule lands
// on both commands at once instead of drifting between two copies.
type shrinkLeverCommand struct {
	// Name is the operator-facing command, e.g. "fak guard".
	Name string
	// ToWire is the imperative that reaches the passthrough on THIS command. It is a verb
	// phrase, spliced after "Either " in the refusal and "To exercise them, " in the notice.
	ToWire string
}

var (
	// shrinkLeverServeCommand — `fak serve` IS the wire, so its remedy is the flag pair
	// that selects it.
	shrinkLeverServeCommand = shrinkLeverCommand{
		Name:   "fak serve",
		ToWire: "serve the Anthropic Messages wire (--provider anthropic with a single --base-url) or the in-kernel engine (--gguf)",
	}
	// shrinkLeverGuardCommand — `fak guard` reaches the passthrough by DEFAULT when it wraps
	// an Anthropic-wire agent, so its remedy names the default launch and the three flags
	// that steer off it (--local forces provider=openai, --remote-serve forces the
	// OpenAI-compatible wire, and a bare --gguf makes the in-kernel planner the upstream).
	shrinkLeverGuardCommand = shrinkLeverCommand{
		Name:   "fak guard",
		ToWire: "wrap an Anthropic-wire agent (the default `fak guard -- claude`, or --provider anthropic) or the in-kernel engine (--gguf) with no --local and no --remote-serve",
	}
)

// The wire classes this rule distinguishes. They mirror, from flags alone, the planner
// switch gateway.New makes and the passthrough predicate that reads its result: only a
// SINGLE upstream on the anthropic provider wire ends up as an *agent.HTTPPlanner whose
// Provider is ProviderAnthropic, which is the one shape unwrapHTTPPlanner can unwrap and
// therefore the one shape anthropicPassthrough() answers true for.
const (
	// shrinkWireAnthropicPassthrough — one upstream, anthropic provider: the flagship
	// passthrough. All three levers run here; nothing is inert.
	shrinkWireAnthropicPassthrough = "anthropic-passthrough"
	// shrinkWireForeignProxy — one upstream on a non-anthropic provider wire (the
	// self-hosted OpenAI-compatible local model, vLLM/SGLang, gemini, xai). The generic
	// planner serves it and every req.Raw transform stands down.
	shrinkWireForeignProxy = "non-anthropic-proxy"
	// shrinkWireReplicaFleet — two or more upstreams. gateway.newProxyPlanner builds a
	// *ReplicaRouter, which is not an *agent.HTTPPlanner, so unwrapHTTPPlanner returns nil
	// and the passthrough is off EVEN ON the anthropic provider wire. This case is easy to
	// configure by accident and impossible to notice without being told.
	shrinkWireReplicaFleet = "replica-fleet"
	// shrinkWireInKernel — no upstream, weights loaded (--gguf): every turn decodes on this
	// box through the in-kernel planner with typed prompt-shrink levers.
	shrinkWireInKernel = "in-kernel"
	// shrinkWireMock — no upstream and no weights: the deterministic offline mock planner.
	shrinkWireMock = "mock"
	// shrinkWireUnknown — the --provider name does not parse. gateway.New refuses that with
	// a precise error of its own, so this rule stands down rather than pre-empting it with a
	// vaguer one. Empty on purpose: an unclassified wire admits everything.
	shrinkWireUnknown = ""
)

// classifyShrinkWire answers which wire class `fak serve` will run, reading only the four
// flags that decide it. Pure and total: no I/O, no dial, no lookup.
//
// The precedence follows gateway.New's own planner switch: upstream URLs win over loaded
// weights (that combination is the DUAL planner, whose proxy side is still the passthrough
// for every model id not routed local), and only then does the provider name matter.
func classifyShrinkWire(provider, baseURL string, replicaBaseURLs []string, ggufPath string) string {
	urls := 0
	if strings.TrimSpace(baseURL) != "" {
		urls++
	}
	for _, r := range replicaBaseURLs {
		if strings.TrimSpace(r) != "" {
			urls++
		}
	}
	switch {
	case urls == 0 && strings.TrimSpace(ggufPath) != "":
		return shrinkWireInKernel
	case urls == 0:
		return shrinkWireMock
	case urls > 1:
		return shrinkWireReplicaFleet
	}
	p, ok := agent.ParseProvider(provider)
	if !ok {
		return shrinkWireUnknown
	}
	if p == agent.ProviderAnthropic {
		return shrinkWireAnthropicPassthrough
	}
	return shrinkWireForeignProxy
}

// shrinkWireRunsLevers reports whether the classified wire is the one the three levers
// actually run on. An unknown wire counts as running them: an unproven classification must
// never be the thing that refuses a boot.
func shrinkWireRunsLevers(wire string) bool {
	return wire == shrinkWireAnthropicPassthrough || wire == shrinkWireInKernel || wire == shrinkWireUnknown
}

// shrinkWireNoticeWarranted reports whether a wire deserves the default-on ADVISORY. The
// refusal is not gated on this — an explicitly enabled lever that cannot run is a stated
// intent this process cannot honor no matter which wire it is on.
//
// The mock wire is the one exclusion, and it is not noise-avoidance for its own sake: the
// deterministic offline planner returns SCRIPTED text, so no prompt-shrink measurement is
// possible against it at all, and gateway.New already emits a loud WARNING saying exactly
// that. Repeating a shrink-lever advisory there would attach the #5493 finding to the one
// deployment that can never produce the false null it exists to prevent — including every
// `fak serve --stdio` MCP server, which does not serve /v1/messages in the first place.
func shrinkWireNoticeWarranted(wire string) bool {
	return !shrinkWireRunsLevers(wire) && wire != shrinkWireMock
}

// shrinkWireDescription renders an operator-facing description of a wire class. It names
// the provider (a flag value the operator typed, and a closed vocabulary) but never the
// base URL — see the privacy fence in the file header.
func shrinkWireDescription(wire, provider string) string {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		provider = "openai" // the `fak serve --provider` default, as registered in serve.go
	}
	switch wire {
	case shrinkWireForeignProxy:
		return fmt.Sprintf("a non-Anthropic proxy upstream (--provider %s), served by the generic planner", provider)
	case shrinkWireReplicaFleet:
		return fmt.Sprintf("a multi-upstream replica fleet (--provider %s with 2+ base URLs), which routes through the replica router rather than a single passthrough planner", provider)
	case shrinkWireInKernel:
		return "the in-kernel planner (--gguf weights, no upstream), which decodes on this box"
	case shrinkWireMock:
		return "the deterministic offline mock planner (no --base-url and no --gguf)"
	case shrinkWireAnthropicPassthrough:
		return "the Anthropic Messages passthrough"
	default:
		return "an unclassified upstream wire"
	}
}

// shrinkLever is one prompt-shrink lever as this rule sees it: what the operator would
// type, the stable token a log scrape matches, the gateway seam that stands it down, how
// to turn it off, and whether it is on and whether the operator said so.
type shrinkLever struct {
	Flag  string // operator-facing flag, e.g. "--defer-cold-tools"
	Token string // stable machine token, e.g. "defer_cold_tools"
	// Gate names the gateway function that returns identity off the passthrough. A
	// function name rather than a file:line, so the message cannot rot on an edit
	// elsewhere in the file.
	Gate string
	// Off is the exact argument that disables this lever, which is the honest escape
	// from the refusal (the other being a move to the passthrough wire).
	Off string
	// On is the lever's effective value for this run; Explicit records whether the
	// operator named the flag on the command line rather than inheriting the default.
	On       bool
	Explicit bool
}

// shrinkLevers builds the three-lever table from EFFECTIVE values plus the set of flag names
// the operator actually typed. `named` must come from flag.FlagSet.Visit, which walks only
// the flags Parse SET — exactly the distinction between "the operator asked for this" and
// "this is the shipped default". Both commands register the same three flag names, so one
// table serves both.
//
// The order is fixed (compaction, stale-read elision, cold-tool defer) so every rendered
// message is deterministic without a sort.
func shrinkLevers(named map[string]bool, compactHistoryBudget int, elideStaleReads, deferColdTools bool) []shrinkLever {
	return []shrinkLever{
		{
			Flag: "--compact-history-budget", Token: guardvars.ShrinkLeverCompactHistoryBudget,
			Gate: "gateway.compactAnthropicRawWithReason", Off: "--compact-history-budget 0",
			On: compactHistoryBudget > 0, Explicit: named["compact-history-budget"],
		},
		{
			Flag: "--elide-stale-reads", Token: guardvars.ShrinkLeverElideStaleReads,
			Gate: "gateway.maybeElideStaleReads", Off: "--elide-stale-reads=false",
			On: elideStaleReads, Explicit: named["elide-stale-reads"],
		},
		{
			Flag: "--defer-cold-tools", Token: guardvars.ShrinkLeverDeferColdTools,
			Gate: "gateway.maybeDeferColdTools", Off: "--defer-cold-tools=false",
			On: deferColdTools, Explicit: named["defer-cold-tools"],
		},
	}
}

// serveShrinkLevers reads the three levers out of the parsed serve flags.
func serveShrinkLevers(fs *flag.FlagSet, sf *serveFlags) []shrinkLever {
	named := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { named[f.Name] = true })
	return shrinkLevers(named, *sf.compactHistoryBudget, *sf.elideStaleReads, *sf.deferColdTools)
}

// inertShrinkLevers returns the levers that are ON yet cannot fire on this wire, in the
// caller's order. On a wire that runs them — and on an unclassified one — the result is
// empty, so the caller says nothing.
func inertShrinkLevers(wire string, levers []shrinkLever) []shrinkLever {
	if shrinkWireRunsLevers(wire) {
		return nil
	}
	var inert []shrinkLever
	for _, l := range levers {
		if l.On {
			inert = append(inert, l)
		}
	}
	return inert
}

// explicitShrinkLevers / defaultShrinkLevers split an inert set by who turned it on. The
// first set refuses the boot; the second only warns.
func explicitShrinkLevers(inert []shrinkLever) []shrinkLever {
	var out []shrinkLever
	for _, l := range inert {
		if l.Explicit {
			out = append(out, l)
		}
	}
	return out
}

func defaultShrinkLevers(inert []shrinkLever) []shrinkLever {
	var out []shrinkLever
	for _, l := range inert {
		if !l.Explicit {
			out = append(out, l)
		}
	}
	return out
}

// shrinkLeverLines renders one indented line per lever: the flag, its stable token, and the
// gateway seam that stands it down.
func shrinkLeverLines(levers []shrinkLever) string {
	var sb strings.Builder
	for _, l := range levers {
		fmt.Fprintf(&sb, "\n  %s (%s) — stands down to identity in %s", l.Flag, l.Token, l.Gate)
	}
	return sb.String()
}

// shrinkLeverOffForms renders the exact arguments that disable the named levers, so the
// refusal carries its own remedy rather than sending the operator to the help text.
func shrinkLeverOffForms(levers []shrinkLever) string {
	forms := make([]string, 0, len(levers))
	for _, l := range levers {
		forms = append(forms, l.Off)
	}
	return strings.Join(forms, ", ")
}

// shrinkLeverRefusal is the operator-facing refusal for levers that were EXPLICITLY enabled
// on a wire that cannot run them, or "" when there are none. Pure: it renders text and
// decides nothing else.
func shrinkLeverRefusal(cmd shrinkLeverCommand, wire, provider string, explicit []shrinkLever) string {
	if len(explicit) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"%s: %s — refusing to start: %d prompt-shrink lever(s) were explicitly enabled but cannot run on this wire, which is %s.%s\n"+
			"Every one of them is gated on the Anthropic passthrough or in-kernel planner inside the gateway, so on this wire the prompt is forwarded UNSHRUNK and the lever is a no-op. "+
			"Either %s, or state the lever is off (%s) so this run cannot be quoted as a lever result it never produced.",
		cmd.Name, shrinkLeverInertToken, len(explicit), shrinkWireDescription(wire, provider),
		shrinkLeverLines(explicit), cmd.ToWire, shrinkLeverOffForms(explicit))
}

// shrinkLeverNotice is the loud, named advisory for levers that are inert only because they
// ship default-on, or "" when there are none. It does not block the boot: those levers were
// not this operator's stated intent, and refusing would take down every ordinary
// non-Anthropic serve.
func shrinkLeverNotice(cmd shrinkLeverCommand, wire, provider string, defaults []shrinkLever) string {
	if len(defaults) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"%s: %s — NOTICE: %d prompt-shrink lever(s) are on by default and inert on this wire, which is %s.%s\n"+
			"Serving continues; nothing is broken. But an A/B on this wire measures ~0 saving from these levers, and that number is a verdict on an unwired lever, not on the kernel. "+
			"To exercise them, %s.",
		cmd.Name, shrinkLeverInertToken, len(defaults), shrinkWireDescription(wire, provider),
		shrinkLeverLines(defaults), cmd.ToWire)
}

// admitServeShrinkLevers applies the wire rule to the parsed serve flags, writing the
// refusal or the notice to stderr. It returns false when the caller must NOT continue to
// boot. It reads flags only — no dial, no listener, no weight load — so it is cheap enough
// to run before every expensive startup stage.
func admitServeShrinkLevers(fs *flag.FlagSet, sf *serveFlags, stderr io.Writer) bool {
	wire := classifyShrinkWire(*sf.provider, *sf.baseURL, sf.replicaBaseURLs.Values(), *sf.ggufPath)
	return admitShrinkLevers(shrinkLeverServeCommand, wire, *sf.provider, serveShrinkLevers(fs, sf), stderr)
}

// guardShrinkLeverInputs is the RESOLVED guard boot state this rule reads (#5538). Every
// field is what the gateway will actually be built with, NOT the raw flag — see the "guard
// is not serve" note in the file header for why the raw flags cannot be used here:
//
//   - Provider / BaseURL are guardUpstreamPosture.up / .resolvedBase, i.e. AFTER the
//     agent-name auto-detect, the --local rewrite to provider=openai, the --remote-serve
//     force to the OpenAI-compatible wire, and the public-API base-URL default. They are
//     exactly the values handed to gateway.Config.Provider / .BaseURL, which is what the
//     passthrough predicate ends up reading.
//   - GGUFPath non-empty means weights are loaded; with an empty BaseURL that is the pure
//     in-kernel upstream, and with a base URL it is the dual planner (whose proxy side is
//     still the passthrough), which classifyShrinkWire already orders correctly.
//   - CompactHistoryBudget is the value AFTER resolveGuardCompactBudget, so the number the
//     rule judges is the number gateway.Config carries.
//   - SetFlags is guard's own fs.Visit map, so "the operator typed it" means the same thing
//     on both commands.
type guardShrinkLeverInputs struct {
	SetFlags             map[string]bool
	Provider             string
	BaseURL              string
	GGUFPath             string
	CompactHistoryBudget int
	ElideStaleReads      bool
	DeferColdTools       bool
	// SuppressDefaultNotice keeps non-actionable default-on lever diagnostics out of
	// compact attended startup. Explicit inert levers still refuse with full detail.
	ShowDefaultNotice bool
}

// admitGuardShrinkLevers applies the same wire rule to a resolved `fak guard` launch,
// writing the refusal or the notice to stderr and returning false when the caller must NOT
// continue to boot. Guard has no replica-fleet flag, so the replica dimension is empty here;
// the class stays reachable from serve and is left in the shared classifier rather than
// special-cased out.
func admitGuardShrinkLevers(in guardShrinkLeverInputs, stderr io.Writer) bool {
	wire := classifyShrinkWire(in.Provider, in.BaseURL, nil, in.GGUFPath)
	levers := shrinkLevers(in.SetFlags, in.CompactHistoryBudget, in.ElideStaleReads, in.DeferColdTools)
	if !in.ShowDefaultNotice {
		inert := inertShrinkLevers(wire, levers)
		if refusal := shrinkLeverRefusal(shrinkLeverGuardCommand, wire, in.Provider, explicitShrinkLevers(inert)); refusal != "" {
			fmt.Fprintln(stderr, refusal)
			return false
		}
		return true
	}
	return admitShrinkLevers(shrinkLeverGuardCommand, wire, in.Provider, levers, stderr)
}

// admitShrinkLevers is the shared severity split both commands run: an EXPLICITLY enabled
// lever that cannot fire on this wire refuses the boot by name, and a merely default-on one
// gets the named notice. Pure apart from the write to stderr — no dial, no listener, no
// weight load — so it is cheap enough to run before every expensive startup stage.
func admitShrinkLevers(cmd shrinkLeverCommand, wire, provider string, levers []shrinkLever, stderr io.Writer) bool {
	inert := inertShrinkLevers(wire, levers)
	if refusal := shrinkLeverRefusal(cmd, wire, provider, explicitShrinkLevers(inert)); refusal != "" {
		fmt.Fprintln(stderr, refusal)
		return false
	}
	if shrinkWireNoticeWarranted(wire) {
		if notice := shrinkLeverNotice(cmd, wire, provider, defaultShrinkLevers(inert)); notice != "" {
			fmt.Fprintln(stderr, notice)
		}
	}
	return true
}
