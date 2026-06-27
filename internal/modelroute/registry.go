package modelroute

// Logical model-id registry + residency-aware resolution (#616, epic #595).
//
// A Plan.Member.Model is a free string today (`small`, `large`, `guard-a`). For
// LIVE dispatch it must resolve to a REGISTERED engine / provider+model. This
// file is the resolution + eligibility layer that makes that string valid and
// safe — genuinely net-new. #596 only WRITES Plan.Primary() into ToolCall.Engine
// and ASSUMES the string is already a valid registered engine id; r4 (this file)
// is the layer that makes it valid, and #596 consumes r4's output. The two must
// not both claim the resolution step.
//
// THREE things this layer adds, none of which a free string carries:
//
//   - VALIDATE AT LOAD. A Registry maps a logical alias (`small`) to a concrete
//     ResolvedModel (engine id `gpt-4o-mini`, provider `openai`, remote). At
//     manifest-bind time every Member.Model is checked against the registry AND
//     the registry's engine ids against the known-engine-id set — an unknown
//     target fails LOUD, exactly like the policy floor's closed-vocab check, so a
//     typo never silently mis-dispatches.
//   - CARRY PROVIDER / LOCALITY. The residency PDP and the cost layer need to
//     reason about a member: which provider serves it, and is it REMOTE or LOCAL.
//     ResolvedModel carries both.
//   - ENFORCE THE SENSITIVE-REMOTE FLOOR. A sensitive-labeled Subject must not
//     resolve to a remote member. Resolve refuses it — the eligibility half of
//     the wiring contract, designed in now so later wiring cannot regress the
//     default-deny floor.
//
// PURITY (lane rule): the registry/resolution lives in this wiring-facing tier
// but the tier-1 leaf stays stdlib-only. The known-engine-id set is taken as a
// []string PARAMETER (the caller passes abi.EngineIDs()) — this file never
// imports internal/abi. The "sensitive" signal is a Subject label key, not a new
// cross-package type.

import (
	"fmt"
	"sort"
	"strings"
)

// SensitiveLabel is the Subject.Labels key that marks a subject as carrying
// data that must NOT leave for a remote provider. A truthy value ("1", "true",
// "yes", any non-empty value other than the explicit falses) trips the
// sensitive-remote floor in Resolve. It mirrors the residency PDP's taint signal
// without importing the engine package.
const SensitiveLabel = "sensitive"

// ResolvedModel is what a logical model alias resolves to: a concrete engine id
// the dispatcher hands to abi.ToolCall.Engine, the provider that serves it, and
// whether reaching it leaves the box (Remote). The residency PDP (remoteRoute in
// internal/engine) and the cost layer read Provider + Remote to reason about a
// member; the dispatcher reads EngineID. A LOCAL (in-kernel) member sets Remote
// false and may leave Provider "".
type ResolvedModel struct {
	// Alias is the logical name as it appears in a Plan (`small`, `large`).
	Alias string `json:"alias"`
	// EngineID is the concrete registered engine id (`gpt-4o-mini`, an in-kernel
	// engine id) the dispatcher assigns to abi.ToolCall.Engine.
	EngineID string `json:"engine_id"`
	// Provider is the serving provider (`openai`, `anthropic`, `gemini`, `xai`)
	// for a remote member; "" for an in-kernel/local member.
	Provider string `json:"provider,omitempty"`
	// Remote is true iff reaching this model leaves the box — the bit the
	// sensitive-remote floor and the residency PDP gate on.
	Remote bool `json:"remote"`
}

// Registry is the alias -> ResolvedModel map an operator configures alongside a
// routing Manifest: it turns the Plan's logical model names into concrete,
// locality-tagged engine ids. It is data-in / decision-out and pure — the wiring
// layer holds one Registry and consults it after Route picks a Plan.
type Registry struct {
	models map[string]ResolvedModel
}

// NewRegistry builds a registry from a set of resolved models, keyed by Alias.
// A duplicate alias, an empty alias, or an empty engine id fails loud — a
// misconfigured registry must not silently shadow a model. A remote model with no
// Provider also fails: locality without a named provider is meaningless to the
// residency PDP.
func NewRegistry(models []ResolvedModel) (*Registry, error) {
	r := &Registry{models: make(map[string]ResolvedModel, len(models))}
	for i, m := range models {
		if m.Alias == "" {
			return nil, fmt.Errorf("modelroute: registry entry %d has an empty alias", i)
		}
		if m.EngineID == "" {
			return nil, fmt.Errorf("modelroute: registry alias %q has an empty engine id", m.Alias)
		}
		if _, dup := r.models[m.Alias]; dup {
			return nil, fmt.Errorf("modelroute: duplicate registry alias %q", m.Alias)
		}
		if m.Remote && m.Provider == "" {
			return nil, fmt.Errorf("modelroute: registry alias %q is remote but names no provider", m.Alias)
		}
		r.models[m.Alias] = m
	}
	return r, nil
}

// Aliases returns the registered aliases in sorted order (determinism helper).
func (r *Registry) Aliases() []string {
	out := make([]string, 0, len(r.models))
	for a := range r.models {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

// Lookup returns the ResolvedModel for an alias, or false if unregistered.
func (r *Registry) Lookup(alias string) (ResolvedModel, bool) {
	m, ok := r.models[alias]
	return m, ok
}

// ValidateEngines checks every registered engine id against the known-engine-id
// set (the caller passes abi.EngineIDs(); an empty/nil set disables the check so
// a deployment with no static engine registry still loads). An engine id absent
// from the known set fails LOUD — the closed-vocab check that stops a manifest
// naming a model no engine can serve. Returns the first offending alias.
func (r *Registry) ValidateEngines(knownEngineIDs []string) error {
	if len(knownEngineIDs) == 0 {
		return nil
	}
	known := make(map[string]bool, len(knownEngineIDs))
	for _, id := range knownEngineIDs {
		known[id] = true
	}
	for _, alias := range r.Aliases() { // sorted -> deterministic first offender
		m := r.models[alias]
		if !known[m.EngineID] {
			return fmt.Errorf("modelroute: registry alias %q -> engine %q is not a known engine", alias, m.EngineID)
		}
	}
	return nil
}

// ValidateManifest checks that every member model named anywhere in m (the
// default plan and every rule plan) resolves to a registered alias, AND that
// every registered engine id is known. An unknown model name, or an unknown
// engine id, fails LOUD at bind time — a route that names a model the registry
// cannot resolve never reaches dispatch. knownEngineIDs is the abi.EngineIDs()
// set (nil disables the engine-id check).
func (r *Registry) ValidateManifest(m Manifest, knownEngineIDs []string) error {
	if err := r.ValidateEngines(knownEngineIDs); err != nil {
		return err
	}
	check := func(where string, p Plan) error {
		for _, mem := range p.Members {
			if _, ok := r.models[mem.Model]; !ok {
				return fmt.Errorf("modelroute: %s names model %q which is not in the registry", where, mem.Model)
			}
		}
		return nil
	}
	if err := check("default plan", m.Default); err != nil {
		return err
	}
	for _, rule := range m.Rules {
		if err := check("rule "+rule.Name, rule.Plan); err != nil {
			return err
		}
	}
	return nil
}

// ResolveError is the typed failure of resolving a member for a subject. Reason
// distinguishes an UNREGISTERED alias from a SENSITIVE-REMOTE floor refusal so a
// caller (and a test) can branch on WHICH invariant tripped.
type ResolveError struct {
	Alias  string
	Reason string // "unregistered" | "sensitive_remote"
	msg    string
}

func (e *ResolveError) Error() string { return e.msg }

// Resolve turns one member model alias into its concrete ResolvedModel FOR a
// given subject, enforcing the sensitive-remote floor: if the subject is
// labeled sensitive and the alias resolves to a REMOTE member, resolution is
// REFUSED (a sensitive payload must never bind to a remote engine). An
// unregistered alias is also refused. This is the eligibility gate the wiring
// contract needs — run it BEFORE writing the engine id to abi.ToolCall.Engine, so
// the residency PDP never has to adjudicate a sensitive payload already bound for
// a remote model.
func (r *Registry) Resolve(s Subject, alias string) (ResolvedModel, error) {
	m, ok := r.models[alias]
	if !ok {
		return ResolvedModel{}, &ResolveError{
			Alias:  alias,
			Reason: "unregistered",
			msg:    fmt.Sprintf("modelroute: model %q is not registered", alias),
		}
	}
	if m.Remote && isSensitive(s) {
		return ResolvedModel{}, &ResolveError{
			Alias:  alias,
			Reason: "sensitive_remote",
			msg:    fmt.Sprintf("modelroute: sensitive subject must not resolve to remote model %q (provider %q)", alias, m.Provider),
		}
	}
	return m, nil
}

// ResolvePlan resolves every member of a Plan for a subject, in member order
// (the order the dispatcher must preserve into the fold). It fails on the FIRST
// member that does not resolve — an unregistered alias or a sensitive-remote
// refusal — so a plan with even one ineligible member never partially dispatches.
func (r *Registry) ResolvePlan(s Subject, p Plan) ([]ResolvedModel, error) {
	out := make([]ResolvedModel, 0, len(p.Members))
	for _, mem := range p.Members {
		rm, err := r.Resolve(s, mem.Model)
		if err != nil {
			return nil, err
		}
		out = append(out, rm)
	}
	return out, nil
}

// isSensitive reports whether a subject's labels mark it sensitive. Any non-empty
// value at SensitiveLabel other than an explicit false ("0"/"false"/"no") trips
// it — fail-toward-sensitive: an unrecognized value is treated as sensitive.
func isSensitive(s Subject) bool {
	v, ok := s.Labels[SensitiveLabel]
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "0", "false", "no", "off":
		return false
	}
	return true
}
