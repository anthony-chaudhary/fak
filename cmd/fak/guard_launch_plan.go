package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

type guardLaunchPlan struct {
	semantic          []string
	executable        []string
	baseName          string
	profile           harnessprofile.HarnessProfile
	recognizedProfile bool
	initialized       bool
}

func newGuardLaunchPlan(command []string) guardLaunchPlan {
	plan := guardLaunchPlan{
		semantic:    append([]string(nil), command...),
		executable:  append([]string(nil), command...),
		initialized: true,
	}
	if len(command) == 0 {
		return plan
	}
	plan.baseName = guardAgentBaseName(command[0])
	profile, ok := harnessprofile.Lookup(command[0])
	if !ok {
		return plan
	}
	plan.profile = cloneGuardHarnessProfile(profile)
	plan.recognizedProfile = true
	return plan
}

func cloneGuardHarnessProfile(profile harnessprofile.HarnessProfile) harnessprofile.HarnessProfile {
	profile.Names = append([]string(nil), profile.Names...)
	profile.Repoint = append([]harnessprofile.RepointMechanism(nil), profile.Repoint...)
	return profile
}

func (p guardLaunchPlan) semanticCommand() []string {
	return append([]string(nil), p.semantic...)
}

func (p guardLaunchPlan) executableCommand() []string {
	return append([]string(nil), p.executable...)
}

func (p guardLaunchPlan) harnessProfile() harnessprofile.HarnessProfile {
	return cloneGuardHarnessProfile(p.profile)
}

func (p guardLaunchPlan) recognized() bool { return p.recognizedProfile }

func (p guardLaunchPlan) withExecutableCommand(command []string) guardLaunchPlan {
	p.executable = append([]string(nil), command...)
	return p
}

func (p guardLaunchPlan) resolveProvider(explicit string) (string, bool) {
	if provider := strings.ToLower(strings.TrimSpace(explicit)); provider != "" {
		// `fak m --provider openai -- codex ...` is the documented OpenAI spelling,
		// but Codex speaks the Responses wire. Keep the operator's explicit billing
		// choice while selecting the wire required by the harness; otherwise the
		// subscription resolver is skipped and an empty OPENAI_API_KEY fails launch.
		if provider == "openai" && p.recognizedProfile && p.profile.Wire == harnessprofile.WireOpenAIResponses {
			return string(p.profile.Wire), false
		}
		return provider, false
	}
	if p.recognizedProfile {
		return string(p.profile.Wire), true
	}
	return "anthropic", false
}

// openCodeModelPrefixWire maps the provider prefix of an OpenCode `--model
// provider/id` onto the upstream wire guard proxies to.
var openCodeModelPrefixWire = map[string]string{
	"google":    string(harnessprofile.WireGemini),
	"gemini":    string(harnessprofile.WireGemini),
	"anthropic": string(harnessprofile.WireAnthropic),
	"openai":    string(harnessprofile.WireOpenAI),
	"xai":       "xai",
}

// openCodeModelRoute derives the upstream wire and the bare upstream model id
// from an OpenCode child's --model/-m. A missing or unknown prefix is no route.
func (p guardLaunchPlan) openCodeModelRoute() (provider, model string, ok bool) {
	if p.baseName != "opencode" || len(p.semantic) < 2 {
		return "", "", false
	}
	prefix, id, found := strings.Cut(strings.TrimSpace(extractModelFromCommand(p.semantic[1:])), "/")
	id = strings.TrimSpace(id)
	if !found || id == "" {
		return "", "", false
	}
	provider, ok = openCodeModelPrefixWire[strings.ToLower(strings.TrimSpace(prefix))]
	if !ok {
		return "", "", false
	}
	return provider, id, true
}

// applyOpenCodeModelRoute is consulted only when the operator named no upstream
// endpoint, so the provider's public API is the upstream: the model prefix then
// picks that API, and the prefix is stripped because no public API accepts it.
func (p guardLaunchPlan) applyOpenCodeModelRoute(explicit, provider, model string, autodetected bool) (string, string, bool) {
	routeProvider, routeModel, ok := p.openCodeModelRoute()
	if !ok {
		return provider, model, autodetected
	}
	if strings.TrimSpace(explicit) == "" {
		provider, autodetected = routeProvider, true
	}
	if provider == routeProvider && strings.TrimSpace(model) == "" {
		model = routeModel
	}
	return provider, model, autodetected
}

func (p guardLaunchPlan) agentName() string {
	if len(p.semantic) == 0 {
		return ""
	}
	return p.semantic[0]
}

func (p guardLaunchPlan) agentBaseName() string { return p.baseName }

func (p guardLaunchPlan) isCodex() bool {
	return p.profile.HasRepoint(harnessprofile.RepointCLIConfig) || p.baseName == "codex"
}

func (p guardLaunchPlan) interactive() bool {
	if p.baseName == "codex" && guardCodexSemanticSubcommand(p.semantic) == "exec" {
		return false
	}
	return guardChildInteractive(p.semanticCommand())
}

func guardCodexSemanticSubcommand(command []string) string {
	for i := 1; i < len(command); i++ {
		arg := command[i]
		switch {
		case arg == "-c" || arg == "--config":
			i++
		case strings.HasPrefix(arg, "-c="):
		case strings.HasPrefix(arg, "--") && strings.Contains(arg, "="):
		case strings.HasPrefix(arg, "-"):
		default:
			return arg
		}
	}
	return ""
}
