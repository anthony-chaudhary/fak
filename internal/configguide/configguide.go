// Package configguide turns user intent into minimal, reviewable fak.toml
// deltas. It deliberately depends on deploymanifest for validation so guided
// postures cannot become a parallel configuration system.
//
// Invariant: configuration posture resolution is fail-closed and bounded.
// Guard: all synthesized manifests are parsed and verified through deploymanifest before emission.
package configguide

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

// Schema identifies the versioned JSON contract emitted by configguide results.
const Schema = "fak-config-guide/1"

// Change describes an atomic mutation to a manifest field with operator rationale.
type Change struct {
	Field          string `json:"field"`
	Value          any    `json:"value"`
	Why            string `json:"why"`
	EquivalentFlag string `json:"equivalent_flag,omitempty"`
}

// Result captures the guided posture outcome, TOML manifest delta, and recommended command.
type Result struct {
	Schema      string   `json:"schema"`
	Posture     string   `json:"posture"`
	Summary     string   `json:"summary"`
	NeedsConfig bool     `json:"needs_config"`
	Manifest    string   `json:"manifest"`
	Changes     []Change `json:"changes"`
	Run         string   `json:"run"`
}

// Options contains operator inputs for parameterizing the requested posture.
type Options struct {
	Posture    string
	PolicyPath string
	KeyEnv     string
	Budget     int
	Bind       string
}

type posture struct {
	name    string
	summary string
	changes func(Options) []Change
}

var postures = []posture{
	{
		name:    "default",
		summary: "Use fak's tested local defaults; no configuration file is required.",
		changes: func(Options) []Change { return nil },
	},
	{
		name:    "long-session",
		summary: "Keep the local safe topology and add an explicit context budget for longer controlled sessions.",
		changes: func(o Options) []Change {
			budget := o.Budget
			if budget == 0 {
				budget = 160000
			}
			return []Change{{
				Field: "budgets.default_tokens", Value: budget,
				Why:            "Make the session's context ceiling reviewable instead of relying on an implicit harness limit.",
				EquivalentFlag: "--context-budget-tokens " + strconv.Itoa(budget),
			}}
		},
	},
	{
		name:    "team-gateway",
		summary: "Run a shared authenticated gateway while keeping secrets out of the manifest.",
		changes: func(o Options) []Change {
			keyEnv := configuredOr(o.KeyEnv, "FAK_GATEWAY_KEY")
			bind := configuredOr(o.Bind, "0.0.0.0:8080")
			return []Change{
				{Field: "auth.require_key_env", Value: keyEnv, Why: "Require every remote caller to present the team gateway token; only the variable name is reviewable here.", EquivalentFlag: "--require-key-env " + keyEnv},
				{Field: "observability.bind", Value: bind, Why: "Listen beyond loopback so approved teammates can reach the authenticated gateway.", EquivalentFlag: "--addr " + bind},
			}
		},
	},
	{
		name:    "hardened",
		summary: "Require both an explicit capability floor and authenticated ingress.",
		changes: func(o Options) []Change {
			policy := configuredOr(o.PolicyPath, "policy.json")
			keyEnv := configuredOr(o.KeyEnv, "FAK_GATEWAY_KEY")
			bind := configuredOr(o.Bind, "127.0.0.1:8080")
			return []Change{
				{Field: "policy.floor", Value: policy, Why: "Pin the capability floor to a reviewed policy instead of accepting a remembered launch convention.", EquivalentFlag: "--policy " + policy},
				{Field: "auth.require_key_env", Value: keyEnv, Why: "Require authenticated ingress while leaving the secret itself in the environment.", EquivalentFlag: "--require-key-env " + keyEnv},
				{Field: "observability.bind", Value: bind, Why: "Keep the hardened posture loopback-only unless the operator explicitly chooses another bind.", EquivalentFlag: "--addr " + bind},
			}
		},
	},
}

func configuredOr(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

// Names returns the canonical list of supported posture identifiers in stable order.
func Names() []string {
	names := make([]string, len(postures))
	for i, p := range postures {
		names[i] = p.name
	}
	return names
}

// Guide resolves requested options into a validated posture configuration result.
//
// Invariant: configuration posture resolution is fail-closed and bounded.
// Precondition: budget must be non-negative; unknown postures are rejected without side-effects.
// Postcondition: generated manifests strictly round-trip through deploymanifest.Parse.
func Guide(opts Options) (Result, error) {
	name := strings.TrimSpace(opts.Posture)
	if name == "" {
		name = "default"
	}
	var selected *posture
	for i := range postures {
		if postures[i].name == name {
			selected = &postures[i]
			break
		}
	}
	if selected == nil {
		return Result{}, fmt.Errorf("unknown posture %q (choose %s)", name, strings.Join(Names(), ", "))
	}
	if opts.Budget < 0 {
		return Result{}, fmt.Errorf("budget must be non-negative")
	}
	changes := selected.changes(opts)
	manifest, err := render(changes)
	if err != nil {
		return Result{}, err
	}
	result := Result{
		Schema: Schema, Posture: name, Summary: selected.summary,
		NeedsConfig: manifest != "", Manifest: manifest, Changes: changes,
		Run: "fak serve",
	}
	if result.NeedsConfig {
		result.Run = "fak serve --config fak.toml"
	}
	return result, nil
}

func render(changes []Change) (string, error) {
	if len(changes) == 0 {
		return "", nil
	}
	sections := []string{"runtimes", "policy", "auth", "budgets", "audit", "tenants", "agent_templates", "observability"}
	bySection := make(map[string][]Change)
	for _, change := range changes {
		parts := strings.SplitN(change.Field, ".", 2)
		if len(parts) != 2 {
			return "", fmt.Errorf("invalid guided field %q", change.Field)
		}
		bySection[parts[0]] = append(bySection[parts[0]], change)
	}
	var b strings.Builder
	b.WriteString("# fak.toml — minimal delta generated by `fak config guide`.\n")
	b.WriteString("# Explicit flags still override these values; omitted values keep fak defaults.\n")
	for _, section := range sections {
		group := bySection[section]
		if len(group) == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n[%s]\n", section)
		for _, change := range group {
			key := strings.SplitN(change.Field, ".", 2)[1]
			fmt.Fprintf(&b, "%s = %s\n", key, tomlValue(change.Value))
		}
	}
	manifest := b.String()
	if _, err := deploymanifest.Parse([]byte(manifest)); err != nil {
		return "", fmt.Errorf("guided manifest did not round-trip: %w", err)
	}
	return manifest, nil
}

func tomlValue(v any) string {
	switch value := v.(type) {
	case string:
		return strconv.Quote(value)
	case int:
		return strconv.Itoa(value)
	case bool:
		return strconv.FormatBool(value)
	default:
		panic(fmt.Sprintf("unsupported guided value %T", v))
	}
}
