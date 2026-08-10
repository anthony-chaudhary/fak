// Package toolplugin defines a monotone, typed extension host around tool-call
// adjudication. Plugins can add restrictions or request transforms/witnesses;
// they never execute tools or override the kernel/organization floor.
package toolplugin

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

type Stage string

const (
	StageCanonicalize   Stage = "canonicalize"
	StageAdjudicate     Stage = "adjudicate"
	StageWitnessRequest Stage = "witness_request"
	StageAttest         Stage = "attest"
	StageResultAdmit    Stage = "result_admit"
	StageObserve        Stage = "observe"
)

type Action string

const (
	ActionDefer          Action = "defer"
	ActionNarrow         Action = "narrow"
	ActionTransform      Action = "transform"
	ActionRequireWitness Action = "require_witness"
	ActionQuarantine     Action = "quarantine"
	ActionDeny           Action = "deny"
)

type Profile struct {
	ID          string   `json:"id"`
	Version     string   `json:"version"`
	Digest      string   `json:"digest"`
	Stage       Stage    `json:"stage"`
	Permissions []string `json:"permissions"`
	DataEgress  bool     `json:"data_egress"`
	Timeout     Duration `json:"timeout"`
	Fallback    Action   `json:"fallback"`
	Precedence  int      `json:"precedence"`
}

type Duration time.Duration

func (d Duration) Duration() time.Duration { return time.Duration(d) }

type Proposal struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
}

func (p Proposal) Digest() string {
	h := sha256.Sum256(append(append([]byte(p.Tool), 0), p.Args...))
	return "sha256:" + hex.EncodeToString(h[:])
}

type Preference struct {
	RequireWitness     bool   `json:"require_witness"`
	WitnessRoute       string `json:"witness_route,omitempty"`
	WaitMode           string `json:"wait_mode,omitempty"`
	TransformMode      string `json:"transform_mode,omitempty"` // preview | auto
	Disclosure         string `json:"disclosure,omitempty"`
	Timeout            string `json:"timeout,omitempty"`
	ResumeNotification string `json:"resume_notification,omitempty"`
}

type PreferenceLayers struct {
	Kernel       Preference `json:"kernel"`
	Organization Preference `json:"organization"`
	Project      Preference `json:"project"`
	User         Preference `json:"user"`
	Call         Preference `json:"call"`
}

type ResolvedPreference struct {
	Preference
	Sources map[string]string `json:"sources"`
}

// ResolvePreferences folds least-authoritative convenience upward through the
// kernel floor. Scalars use the narrowest non-empty layer (call -> user ->
// project -> organization -> kernel); mandatory witness is monotone OR.
func ResolvePreferences(l PreferenceLayers) ResolvedPreference {
	layers := []struct {
		name string
		p    Preference
	}{{"kernel", l.Kernel}, {"organization", l.Organization}, {"project", l.Project}, {"user", l.User}, {"call", l.Call}}
	out := ResolvedPreference{Sources: make(map[string]string)}
	for _, layer := range layers {
		if layer.p.RequireWitness {
			out.RequireWitness = true
			out.Sources["require_witness"] = joinSource(out.Sources["require_witness"], layer.name)
		}
	}
	set := func(field string, value *string) {
		for i := len(layers) - 1; i >= 0; i-- {
			candidate := preferenceField(layers[i].p, field)
			if candidate != "" {
				*value = candidate
				out.Sources[field] = layers[i].name
				return
			}
		}
	}
	set("witness_route", &out.WitnessRoute)
	set("wait_mode", &out.WaitMode)
	set("transform_mode", &out.TransformMode)
	set("disclosure", &out.Disclosure)
	set("timeout", &out.Timeout)
	set("resume_notification", &out.ResumeNotification)
	return out
}

func preferenceField(p Preference, field string) string {
	switch field {
	case "witness_route":
		return p.WitnessRoute
	case "wait_mode":
		return p.WaitMode
	case "transform_mode":
		return p.TransformMode
	case "disclosure":
		return p.Disclosure
	case "timeout":
		return p.Timeout
	case "resume_notification":
		return p.ResumeNotification
	default:
		panic("unknown preference field: " + field)
	}
}

func joinSource(current, next string) string {
	if current == "" {
		return next
	}
	return current + "+" + next
}

type Decision struct {
	Action      Action          `json:"action"`
	Proposal    *Proposal       `json:"proposal,omitempty"`
	Attestation json.RawMessage `json:"attestation,omitempty"`
	Reason      string          `json:"reason,omitempty"`
}

type Input struct {
	Proposal    Proposal           `json:"proposal"`
	Preference  ResolvedPreference `json:"preference"`
	Result      json.RawMessage    `json:"result,omitempty"`
	Attestation json.RawMessage    `json:"attestation,omitempty"`
}

type Plugin interface {
	Profile() Profile
	Apply(context.Context, Input) (Decision, error)
}

type Floor interface {
	Adjudicate(context.Context, Proposal) Decision
}

type Executor interface {
	Execute(context.Context, Proposal) (json.RawMessage, error)
}

type TraceEvent struct {
	Stage          string `json:"stage"`
	Plugin         string `json:"plugin,omitempty"`
	Action         Action `json:"action,omitempty"`
	ProposalDigest string `json:"proposal_digest,omitempty"`
	Reason         string `json:"reason,omitempty"`
}

type Outcome struct {
	Decision   Decision           `json:"decision"`
	Proposal   Proposal           `json:"proposal"`
	Result     json.RawMessage    `json:"result,omitempty"`
	Preference ResolvedPreference `json:"preference"`
	Trace      []TraceEvent       `json:"trace"`
}

type Host struct {
	Floor         Floor
	Executor      Executor
	Plugins       []Plugin
	MaxTransforms int
}

func (h Host) Run(ctx context.Context, proposal Proposal, layers PreferenceLayers) Outcome {
	pref := ResolvePreferences(layers)
	out := Outcome{Proposal: proposal, Preference: pref}
	plugins := append([]Plugin(nil), h.Plugins...)
	sort.SliceStable(plugins, func(i, j int) bool { return plugins[i].Profile().Precedence < plugins[j].Profile().Precedence })
	byStage := func(stage Stage) []Plugin {
		var selected []Plugin
		for _, p := range plugins {
			if p.Profile().Stage == stage {
				selected = append(selected, p)
			}
		}
		return selected
	}
	transforms := 0
	seen := map[string]bool{proposal.Digest(): true}
	for _, plugin := range byStage(StageCanonicalize) {
		decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref}, &out)
		if terminal(decision.Action) {
			out.Decision = decision
			return out
		}
		if decision.Action == ActionTransform && decision.Proposal != nil {
			transforms++
			if h.MaxTransforms <= 0 {
				h.MaxTransforms = 4
			}
			digest := decision.Proposal.Digest()
			if transforms > h.MaxTransforms || seen[digest] {
				out.Decision = Decision{Action: ActionDeny, Reason: "TRANSFORM_LOOP"}
				out.Trace = append(out.Trace, TraceEvent{Stage: "transform_guard", Action: ActionDeny, Reason: "TRANSFORM_LOOP", ProposalDigest: digest})
				return out
			}
			seen[digest] = true
			proposal = *decision.Proposal
			out.Proposal = proposal
		}
	}
	floor := h.Floor.Adjudicate(ctx, proposal)
	out.Trace = append(out.Trace, TraceEvent{Stage: "floor", Action: floor.Action, Reason: floor.Reason, ProposalDigest: proposal.Digest()})
	if floor.Action == ActionDeny || floor.Action == ActionQuarantine {
		out.Decision = floor
		return out
	}
	for _, plugin := range byStage(StageAdjudicate) {
		decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref}, &out)
		if terminal(decision.Action) {
			out.Decision = decision
			return out
		}
		if decision.Action == ActionTransform {
			out.Decision = Decision{Action: ActionDeny, Reason: "TRANSFORM_REQUIRES_CANONICALIZE_STAGE"}
			return out
		}
		if decision.Action == ActionRequireWitness {
			pref.RequireWitness = true
			out.Preference = pref
		}
	}
	if pref.RequireWitness {
		var attestation json.RawMessage
		for _, plugin := range byStage(StageWitnessRequest) {
			decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref}, &out)
			if terminal(decision.Action) {
				out.Decision = decision
				return out
			}
			if len(decision.Attestation) != 0 {
				attestation = decision.Attestation
			}
		}
		verified := false
		for _, plugin := range byStage(StageAttest) {
			decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref, Attestation: attestation}, &out)
			if terminal(decision.Action) {
				out.Decision = decision
				return out
			}
			if decision.Action == ActionNarrow {
				verified = true
			}
		}
		if !verified {
			out.Decision = Decision{Action: ActionRequireWitness, Reason: "WITNESS_REQUIRED"}
			return out
		}
	}
	result, err := h.Executor.Execute(ctx, proposal)
	if err != nil {
		out.Decision = Decision{Action: ActionQuarantine, Reason: "EXECUTION_ERROR"}
		return out
	}
	out.Result = result
	out.Trace = append(out.Trace, TraceEvent{Stage: "execute", Action: ActionNarrow, ProposalDigest: proposal.Digest()})
	for _, plugin := range byStage(StageResultAdmit) {
		decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref, Result: result}, &out)
		if terminal(decision.Action) {
			out.Decision = decision
			return out
		}
	}
	out.Decision = Decision{Action: ActionNarrow, Reason: "ADMITTED"}
	for _, plugin := range byStage(StageObserve) {
		decision := h.apply(ctx, plugin, Input{Proposal: proposal, Preference: pref, Result: result}, &out)
		if decision.Action != ActionDefer && decision.Action != ActionNarrow {
			out.Trace = append(out.Trace, TraceEvent{Stage: "observer_guard", Plugin: plugin.Profile().ID, Action: ActionDefer, Reason: "OBSERVER_DECISION_IGNORED"})
		}
	}
	return out
}

func (h Host) apply(ctx context.Context, plugin Plugin, input Input, out *Outcome) Decision {
	profile := plugin.Profile()
	if err := validateProfile(profile); err != nil {
		decision := Decision{Action: ActionDeny, Reason: "INVALID_PROFILE: " + err.Error()}
		out.Trace = append(out.Trace, TraceEvent{Stage: string(profile.Stage), Plugin: profile.ID, Action: decision.Action, Reason: decision.Reason})
		return decision
	}
	deadline := profile.Timeout.Duration()
	if deadline <= 0 {
		deadline = 2 * time.Second
	}
	callCtx, cancel := context.WithTimeout(ctx, deadline)
	defer cancel()
	decision, err := plugin.Apply(callCtx, input)
	if err != nil {
		decision = Decision{Action: profile.Fallback, Reason: "PLUGIN_ERROR: " + err.Error()}
	}
	if decision.Action == "" {
		decision.Action = ActionDefer
	}
	if decision.Action == ActionNarrow && profile.Stage == StageCanonicalize && decision.Proposal != nil {
		decision = Decision{Action: ActionDeny, Reason: "AUTHORITY_WIDENING"}
	}
	out.Trace = append(out.Trace, TraceEvent{Stage: string(profile.Stage), Plugin: profile.ID, Action: decision.Action, Reason: decision.Reason, ProposalDigest: input.Proposal.Digest()})
	return decision
}

func validateProfile(p Profile) error {
	if p.ID == "" || p.Version == "" || !strings.HasPrefix(p.Digest, "sha256:") {
		return errors.New("identity/version/digest required")
	}
	if p.DataEgress && !contains(p.Permissions, "network") {
		return errors.New("data egress requires network permission")
	}
	switch p.Fallback {
	case ActionDeny, ActionQuarantine, ActionRequireWitness:
	default:
		return fmt.Errorf("fallback %q is not fail-closed", p.Fallback)
	}
	return nil
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
func terminal(a Action) bool { return a == ActionDeny || a == ActionQuarantine }
