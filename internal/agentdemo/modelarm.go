package agentdemo

// modelarm.go adds the *latest-model* arm as a clean UPGRADE on the deterministic
// spine, never a fork. The lowest-common-denominator demo path stays a rule-based
// Planner (no key, no network, the CI witness); the model arm is an adapter that
// wraps a caller-supplied live-model seam behind the SAME step→kernel.Fold path.
//
// Three properties fall out of this shape, and they are exactly the issue's
// acceptance criteria (#1740):
//
//   - The package holds NO credentials. The live provider call is a ProposeFunc the
//     caller supplies — an OpenAI/Anthropic/local-runtime adapter lives in the
//     credentialed main, not here — so internal/agentdemo builds and tests with no
//     key, network, or model, and the LCD stays green.
//   - The kernel is unchanged. A ModelArm only PROPOSES Steps; RunArm folds them
//     through the same Toolset.Run → kernel.Fold path the deterministic planner uses,
//     so a model-planned destructive call is refused at the capability floor exactly
//     as a keyword-planned one is — independent of what the model "wanted".
//   - The model choice is recorded at RUN TIME, not hard-coded. PlanMeta carries the
//     provider/model/rung the ProposeFunc actually reports for the run, plus the date
//     the caller stamped it — so the witness records the live choice instead of an
//     evergreen claim that goes stale (the exact model name is time-sensitive).

import (
	"context"
	"fmt"
)

// Source names which planner actually produced a run's steps — the honest label the
// witness records so a reader can tell the live arm from its fallback.
type Source string

const (
	// SourceModel means a live model (the ProposeFunc) planned the steps.
	SourceModel Source = "model"
	// SourceFallback means the deterministic planner planned the steps because the
	// model seam was unavailable, errored, or proposed nothing — the LCD path.
	SourceFallback Source = "deterministic-fallback"
)

// PlanMeta records what actually served a plan at run time. It is the anti-stale
// witness the issue requires: the exact Provider / Model / Rung the live seam
// reported for THIS run (not a hard-coded evergreen name), the AsOf date the caller
// stamped the choice, and Source — whether the model or the deterministic fallback
// planned the steps. A zero PlanMeta is the honest "no model was in the loop" record.
type PlanMeta struct {
	Source   Source `json:"source"`             // model | deterministic-fallback
	Provider string `json:"provider,omitempty"` // e.g. "anthropic", "openai", "local-sglang"
	Model    string `json:"model,omitempty"`    // the exact model id the seam reported this run
	Rung     string `json:"rung,omitempty"`     // the runtime/rung that served (e.g. "hosted", "on-box")
	AsOf     string `json:"as_of,omitempty"`    // date the caller verified the choice (YYYY-MM-DD), NOT evergreen
	Note     string `json:"note,omitempty"`     // free-form: why the fallback fired, etc.
}

// ProposeFunc is the live-model seam: given a prompt, it returns the tool-call Steps
// the model wants to attempt plus the PlanMeta describing the provider/model/rung it
// actually used. It is the ONLY place a credentialed provider call enters the arm —
// the caller (a credentialed main) supplies it; internal/agentdemo never does. A
// ProposeFunc that errors or returns no steps hands the turn to the deterministic
// fallback, so a missing key or a flaky provider degrades to the LCD path instead of
// breaking the demo.
type ProposeFunc func(ctx context.Context, prompt string) ([]Step, PlanMeta, error)

// ModelArm is the opt-in latest-model planner arm. Propose is the live seam;
// Fallback is the deterministic Planner that runs when Propose is nil, errors, or
// proposes nothing. Base carries the provider/model/rung the caller INTENDS to use
// and its AsOf date; the actual PlanMeta returned by Plan is the seam's live report
// (Source=SourceModel) or Base annotated as the fallback (Source=SourceFallback).
type ModelArm struct {
	// Propose is the live-model seam. Nil means "no model configured" — every plan
	// takes the deterministic fallback (the LCD path with no credentials).
	Propose ProposeFunc
	// Fallback is the deterministic planner. It MUST be non-nil: it is both the
	// default arm and the safety net when the model seam is unavailable.
	Fallback Planner
	// Base is the intended provider/model/rung + AsOf date, used to annotate a
	// fallback record and as the default when a ProposeFunc reports empty meta.
	Base PlanMeta
}

// Configured reports whether a live model seam is wired. When false, the arm is the
// deterministic planner in a trench coat — useful for a main to print "live arm: off
// (no credentials)" without duplicating the nil check.
func (a ModelArm) Configured() bool { return a.Propose != nil }

// Plan produces the steps for a prompt and the honest PlanMeta of what planned them.
// It tries the live seam first; on a nil seam, an error, or an empty proposal it
// falls back to the deterministic planner and records Source=SourceFallback (with the
// error, if any, in Note). The model's proposed steps are returned AS-IS — they are
// NOT trusted here; RunArm folds them through the kernel, which is where a destructive
// call is refused. Plan never returns an error: the fallback is always available.
func (a ModelArm) Plan(ctx context.Context, prompt string) ([]Step, PlanMeta) {
	if a.Propose != nil {
		steps, meta, err := a.Propose(ctx, prompt)
		if err == nil && len(steps) > 0 {
			meta.Source = SourceModel
			if meta.Provider == "" {
				meta.Provider = a.Base.Provider
			}
			if meta.Model == "" {
				meta.Model = a.Base.Model
			}
			if meta.Rung == "" {
				meta.Rung = a.Base.Rung
			}
			if meta.AsOf == "" {
				meta.AsOf = a.Base.AsOf
			}
			return steps, meta
		}
		// Model seam failed or proposed nothing — degrade to the LCD path.
		fb := a.fallbackMeta()
		if err != nil {
			fb.Note = "model seam error: " + err.Error()
		} else {
			fb.Note = "model proposed no steps"
		}
		return a.fallbackSteps(prompt), fb
	}
	return a.fallbackSteps(prompt), a.fallbackMeta()
}

func (a ModelArm) fallbackSteps(prompt string) []Step {
	if a.Fallback == nil {
		return nil
	}
	return a.Fallback(prompt)
}

func (a ModelArm) fallbackMeta() PlanMeta {
	m := a.Base
	m.Source = SourceFallback
	m.Note = ""
	return m
}

// RunArm plans a prompt with the arm and folds the planned steps through the REAL
// kernel via Toolset.Run, returning the transcript AND the PlanMeta of what planned
// it. This is the model arm's one-call entry point: whether the steps came from the
// live model or the deterministic fallback, they take the identical kernel.Fold path,
// so the safety floor is enforced the same way for both — a destructive model
// proposal is refused with its closed reason code, independent of model behavior.
func (ts *Toolset) RunArm(ctx context.Context, scenario, prompt string, arm ModelArm) (Transcript, PlanMeta, error) {
	steps, meta := arm.Plan(ctx, prompt)
	tr, err := ts.Run(ctx, scenario, prompt, steps)
	if err != nil {
		return Transcript{}, PlanMeta{}, fmt.Errorf("agentdemo: run %s arm: %w", meta.Source, err)
	}
	return tr, meta, nil
}
