package main

import (
	"errors"
	"flag"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

// serveConfigPath finds the bootstrap flag that must be known before the full
// serve FlagSet is parsed. Configuration is deliberately explicit: an ambient
// fak.toml never changes a command merely because its working directory moved.
func serveConfigPath(argv []string) (string, error) {
	var path string
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "--config" || arg == "-config":
			if i+1 >= len(argv) || strings.HasPrefix(argv[i+1], "-") {
				return "", errors.New("--config requires a file path")
			}
			i++
			if path != "" {
				return "", errors.New("--config may be specified only once")
			}
			path = argv[i]
		case strings.HasPrefix(arg, "--config=") || strings.HasPrefix(arg, "-config="):
			value := strings.TrimSpace(strings.SplitN(arg, "=", 2)[1])
			if value == "" {
				return "", errors.New("--config requires a file path")
			}
			if path != "" {
				return "", errors.New("--config may be specified only once")
			}
			path = value
		}
	}
	return path, nil
}

// applyServeManifestDefaults wires the deployment fields that have an exact
// serve counterpart. Pointer assignment changes the parse defaults; the normal
// flag parser then naturally gives every explicitly supplied flag final say.
func applyServeManifestDefaults(sf *serveFlags, m deploymanifest.Manifest) {
	if m.Present("policy", "floor") {
		*sf.policyPath = m.Policy.Floor
	}
	if m.Present("auth", "require_key_env") {
		*sf.requireKeyEnv = m.Auth.RequireKeyEnv
	}
	if m.Present("budgets", "default_tokens") {
		*sf.contextBudgetTokens = m.Budgets.DefaultTokens
	}
	if m.Present("observability", "bind") {
		*sf.addr = m.Observability.Bind
	}
}

func explicitFlagNames(fs *flag.FlagSet) map[string]bool {
	names := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { names[f.Name] = true })
	return names
}

type effectiveConfigValue struct {
	Value  any    `json:"value"`
	Source string `json:"source"`
}

type manifestOpinion struct {
	Value       any    `json:"value"`
	Declared    bool   `json:"declared"`
	Disposition string `json:"disposition"` // applied | reserved | refused
	Reason      string `json:"reason"`
	NextAction  string `json:"next_action,omitempty"`
}

type effectiveServeConfigReport struct {
	Schema          string                          `json:"schema"`
	Config          string                          `json:"config,omitempty"`
	Values          map[string]effectiveConfigValue `json:"values"`
	Opinions        map[string]manifestOpinion      `json:"opinions"`
	ToolPlugins     []toolplugin.Profile            `json:"tool_plugins,omitempty"`
	ToolPreferences toolplugin.ResolvedPreference   `json:"tool_preferences"`
}

type serveManifestSpec struct {
	flagName        string
	appliedValue    any
	hasAppliedValue bool
	reason          string
	nextAction      string
}

// serveManifestSpecs is intentionally exhaustive over deploymanifest.KnownKeys.
// The coverage test fails when the manifest vocabulary grows without this
// runtime deciding whether the new opinion is applied, reserved, or refused.
var serveManifestSpecs = buildServeManifestSpecs()

func buildServeManifestSpecs() map[string]serveManifestSpec {
	specs := map[string]serveManifestSpec{
		"agent_templates.dir":    {reason: "agent templates are started by the all-in-one orchestrator", nextAction: "use fak up when agent-template orchestration ships"},
		"audit.journal":          {reason: "audit journal lifecycle belongs to the all-in-one orchestrator", nextAction: "use the existing serve audit controls until fak up owns this field"},
		"audit.retention_days":   {reason: "retention requires the audit lifecycle manager", nextAction: "configure retention through the audit subsystem until fak up owns this field"},
		"auth.require_key_env":   {flagName: "require-key-env", reason: "mapped directly to serve authentication"},
		"budgets.default_tokens": {flagName: "context-budget-tokens", reason: "mapped directly to the default session token budget"},
		"observability.bind":     {flagName: "addr", reason: "mapped directly to the serve listener"},
		"observability.metrics":  {appliedValue: true, hasAppliedValue: true, reason: "the gateway serves its metrics surface whenever serve is running", nextAction: "use fak up when endpoint enablement becomes topology-selectable"},
		"policy.floor":           {flagName: "policy", reason: "mapped directly to the capability-floor path"},
		"policy.inline":          {reason: "serve accepts a policy path, not unmaterialized inline policy", nextAction: "write the policy to a reviewed file and set policy.floor"},
		"runtimes.agent_runtime": {reason: "serve starts only the gateway process", nextAction: "use fak up when multi-runtime orchestration ships"},
		"runtimes.gateway":       {appliedValue: true, hasAppliedValue: true, reason: "invoking fak serve necessarily starts the gateway", nextAction: "omit fak serve when gateway=false; use fak up for topology selection"},
		"runtimes.model":         {reason: "model source selection requires the full backend/provider flag set", nextAction: "use explicit serve backend/provider flags until fak up maps this field"},
		"tenants.enabled":        {reason: "tenant lifecycle belongs to the all-in-one orchestrator", nextAction: "leave tenants disabled until fak up owns tenant provisioning"},
	}
	for _, key := range deploymanifest.KnownKeys() {
		if strings.HasPrefix(key.Dotted(), "tool_plugins.") {
			specs[key.Dotted()] = serveManifestSpec{flagName: "tool-plugin-config", reason: "compiled into the gateway tool-plugin host and monotone preference layers"}
		}
	}
	return specs
}

func compileToolPluginConfig(m deploymanifest.Manifest) ([]toolplugin.Plugin, toolplugin.PreferenceLayers, error) {
	plugins := make([]toolplugin.Plugin, 0, len(m.ToolPlugins.Plugins))
	seen := map[string]bool{}
	for _, sel := range m.ToolPlugins.Plugins {
		if seen[sel.ID] {
			return nil, toolplugin.PreferenceLayers{}, fmt.Errorf("PLUGIN_DUPLICATE: %s", sel.ID)
		}
		seen[sel.ID] = true
		p, err := toolplugin.ResolvePinned(sel.ID, sel.Version, sel.Digest)
		if err != nil {
			return nil, toolplugin.PreferenceLayers{}, err
		}
		plugins = append(plugins, p)
	}
	toPref := func(p deploymanifest.PreferenceLayer) toolplugin.Preference {
		return toolplugin.Preference{RequireWitness: p.RequireWitness, WitnessRoute: p.WitnessRoute, WaitMode: p.WaitMode, TransformMode: p.TransformMode, Disclosure: p.Disclosure, Timeout: p.Timeout, ResumeNotification: p.ResumeNotification}
	}
	return plugins, toolplugin.PreferenceLayers{Organization: toPref(m.ToolPlugins.Organization), Project: toPref(m.ToolPlugins.Project), User: toPref(m.ToolPlugins.User)}, nil
}
func toolPluginProfiles(ps []toolplugin.Plugin) []toolplugin.Profile {
	out := make([]toolplugin.Profile, 0, len(ps))
	for _, p := range ps {
		out = append(out, p.Profile())
	}
	return out
}

func effectiveServeConfig(sf *serveFlags, m deploymanifest.Manifest, hasManifest bool, explicit map[string]bool) effectiveServeConfigReport {
	plugins, prefs, _ := compileToolPluginConfig(m)
	source := func(flagName, section, key string) string {
		if explicit[flagName] {
			return "flag"
		}
		if hasManifest && m.Present(section, key) {
			return "manifest"
		}
		return "built-in"
	}
	return effectiveServeConfigReport{
		Schema: "fak-serve-effective-config/2",
		Config: *sf.configPath,
		Values: map[string]effectiveConfigValue{
			"addr":                                 {Value: *sf.addr, Source: source("addr", "observability", "bind")},
			"policy":                               {Value: *sf.policyPath, Source: source("policy", "policy", "floor")},
			"require_key_env":                      {Value: *sf.requireKeyEnv, Source: source("require-key-env", "auth", "require_key_env")},
			"context_budget_tokens":                {Value: *sf.contextBudgetTokens, Source: source("context-budget-tokens", "budgets", "default_tokens")},
			"native_qwen_q4k_prefill_chunk_tokens": {Value: *sf.nativeQwenQ4KPrefillChunk, Source: source(nativeQwenQ4KPrefillChunkFlag, "", "")},
			"native_qwen35_metal_gdn_sequence":     {Value: *sf.nativeQwen35MetalGDNSequence, Source: source("native-qwen35-metal-gdn-sequence", "", "")},
			"native_q4k_gateup_slab":               {Value: *sf.nativeQ4KGateUpOutputSlab, Source: source("native-q4k-gateup-slab", "", "")},
			"native_prefix_profile":                {Value: *sf.nativePrefixProfile, Source: source("native-prefix-profile", "", "")},
			"vulkan_q4k_profile":                   {Value: *sf.vulkanQ4KProfile, Source: source("vulkan-q4k-profile", "", "")},
			"vulkan_stage_q4k":                     {Value: *sf.vulkanStageQ4K, Source: source("vulkan-stage-q4k", "", "")},
		},
		Opinions:        serveManifestOpinions(m),
		ToolPlugins:     toolPluginProfiles(plugins),
		ToolPreferences: toolplugin.ResolvePreferences(prefs),
	}
}

func serveManifestOpinions(m deploymanifest.Manifest) map[string]manifestOpinion {
	defaults := deploymanifest.Defaults()
	opinions := make(map[string]manifestOpinion, len(serveManifestSpecs))
	for _, key := range deploymanifest.KnownKeys() {
		dotted := key.Dotted()
		spec, ok := serveManifestSpecs[dotted]
		if !ok {
			panic("missing serve manifest disposition: " + dotted)
		}
		value := m.Value(key)
		opinion := manifestOpinion{
			Value:      value,
			Declared:   m.Present(key.Section, key.Name),
			Reason:     spec.reason,
			NextAction: spec.nextAction,
		}
		switch {
		case spec.flagName != "":
			opinion.Disposition = "applied"
		case spec.hasAppliedValue && reflect.DeepEqual(value, spec.appliedValue):
			opinion.Disposition = "applied"
		case spec.hasAppliedValue && opinion.Declared:
			opinion.Disposition = "refused"
		case opinion.Declared && !reflect.DeepEqual(value, defaults.Value(key)):
			opinion.Disposition = "refused"
		default:
			opinion.Disposition = "reserved"
		}
		opinions[dotted] = opinion
	}
	return opinions
}

// validateServeManifestOpinions prevents an operator's non-default opinion from
// being acknowledged and then ignored. Default-valued declarations emitted by
// `fak init` remain safe and are reported as reserved.
func validateServeManifestOpinions(m deploymanifest.Manifest) error {
	var refused []string
	for dotted, opinion := range serveManifestOpinions(m) {
		if opinion.Disposition == "refused" {
			refused = append(refused, fmt.Sprintf("%s=%v (%s; %s)", dotted, opinion.Value, opinion.Reason, opinion.NextAction))
		}
	}
	if len(refused) == 0 {
		return nil
	}
	sort.Strings(refused)
	return fmt.Errorf("CONFIG_OPINION_UNSUPPORTED: %s", strings.Join(refused, "; "))
}

func applyToolPluginConfig(cfg *gateway.Config, m deploymanifest.Manifest) error {
	plugins, prefs, err := compileToolPluginConfig(m)
	if err != nil {
		return err
	}
	cfg.ToolPlugins = plugins
	cfg.ToolPreferences = prefs
	return nil
}
