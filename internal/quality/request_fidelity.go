package quality

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// This file closes the one acceptance criterion of the reference-runner adapter
// (#4518) that runner.go did not carry: "unsupported fields and request diffs
// are explicit". Differential testing's whole premise is that both sides ran the
// SAME normalized request. A runner that quietly ignores TopK, clamps MaxTokens,
// or rewrites the prompt breaks that premise silently — an equal-tokens verdict
// then proves nothing, and a divergence verdict indicts the decode for what the
// request layer already changed. Three properties make the record trustworthy:
//
//   - The declaration is OPTIONAL (RequestAdapter). ReferenceRunner,
//     ScriptedRunner, and every runner the child cohort already ships satisfy
//     Runner unedited — the same additive posture the pinned /1 schemas require.
//   - The harness DERIVES the diff by comparing the declaration against the
//     case, instead of trusting a runner's own summary of how it differed. That
//     is the posture the package already takes toward a runner's trace: the
//     spine never trusts a runner's self-description.
//   - A run whose request was not honored is explicit and is NOT an unqualified
//     pass, consistent with "missing or inconclusive evidence is never pass".

// requestFields is the closed, canonically ORDERED vocabulary of the normalized
// request a case hands both sides: the prompt plus every SamplingParams field.
// The order is fixed so the FIRST field a failing run names is a property of the
// request shape rather than of map iteration — a replayed result has to name the
// same first offender every time.
var requestFields = []string{
	"prompt",
	"params.temperature",
	"params.top_k",
	"params.top_p",
	"params.max_tokens",
	"params.seed",
}

// EffectiveRequest is a runner's declaration of the request it ACTUALLY
// executes for a case: the prompt and params it ran with, plus the field names
// it cannot honor at all. Unsupported is a static capability claim — a real
// adapter always has some field it does not implement — while Prompt/Params are
// what this specific case turned into on the way in.
type EffectiveRequest struct {
	Prompt string         `json:"prompt"`
	Params SamplingParams `json:"params"`
	// Unsupported names request fields (from the requestFields vocabulary) this
	// runner cannot honor. Naming a field here is not by itself a failure: it
	// costs a run nothing unless the case actually specifies that field.
	Unsupported []string `json:"unsupported,omitempty"`
}

// RequestAdapter is the optional half of the reference-runner adapter seam
// (#4518). A runner that implements it declares what it really ran, so the
// harness can prove both sides were handed the same request instead of assuming
// it. A runner that does NOT implement it asserts faithfulness by omission —
// RequestFidelity.Declared records which of the two a result rests on, so an
// audited faithful run is distinguishable from an unaudited one.
type RequestAdapter interface {
	Runner
	EffectiveRequest(c QualityCase) EffectiveRequest
}

// FieldDelta is one normalized request field whose executed value differed from
// the one the case declared — the request-level analogue of a token Divergence.
type FieldDelta struct {
	Field     string `json:"field"`
	Requested string `json:"requested"`
	Effective string `json:"effective"`
}

// RequestFidelity is the harness-derived record of how faithfully ONE runner
// executed the case's normalized request. Dropped and Diff are the two ways a
// request can be lost in translation: a field the runner silently ignores, and a
// field it substituted a different value for.
type RequestFidelity struct {
	Runner string `json:"runner"`
	// Declared reports whether this runner implements RequestAdapter. False
	// means the fidelity below is an assumption, not a measurement.
	Declared bool `json:"declared"`
	// Unsupported is every field the runner claims it cannot honor, in canonical
	// order — the capability claim, recorded whether or not this case exercises it.
	Unsupported []string `json:"unsupported,omitempty"`
	// Dropped is the subset of Unsupported that THIS case actually specifies:
	// the fields the run silently threw away.
	Dropped []string `json:"dropped,omitempty"`
	// Diff is the request the runner executed minus the request the case
	// declared, excluding fields already named in Dropped.
	Diff []FieldDelta `json:"diff,omitempty"`
}

// Faithful reports whether this runner executed the case's normalized request
// without loss. A runner that declares an unsupported field the case never set
// is still faithful for that case — nothing was translated away.
func (f RequestFidelity) Faithful() bool { return len(f.Dropped) == 0 && len(f.Diff) == 0 }

// offenses renders every way this runner departed from the normalized request,
// in canonical field order, as "field: what happened". Element zero is therefore
// the FIRST actionable request divergence — the localization contract the spine
// holds every verdict to. Values are echoed only for a substitution; a dropped
// field renders as its name alone, because the value the case asked for is
// already in the bundle's own embedded Case.
func (f RequestFidelity) offenses() []string {
	out := make([]string, 0, len(f.Dropped)+len(f.Diff))
	for _, name := range f.Dropped {
		out = append(out, name+": unsupported by this runner and silently dropped")
	}
	for _, d := range f.Diff {
		out = append(out, fmt.Sprintf("%s: case declared %s, runner ran %s", d.Field, d.Requested, d.Effective))
	}
	return out
}

// RequestRecord pairs both sides' fidelity records for one run. It is the
// request-level half of a result's provenance: the evidence that the comparison
// underneath it was between like and like.
type RequestRecord struct {
	Reference RequestFidelity `json:"reference"`
	Engine    RequestFidelity `json:"engine"`
}

// Faithful reports whether BOTH sides ran the case's normalized request.
func (r RequestRecord) Faithful() bool { return r.Reference.Faithful() && r.Engine.Faithful() }

// requestFidelity derives one runner's fidelity record. A runner that does not
// implement RequestAdapter returns an undeclared, empty record: it is treated as
// faithful, because refusing every pre-existing runner would be a breaking
// change dressed up as rigor — Declared is what tells a reader the difference.
func requestFidelity(r Runner, c QualityCase) RequestFidelity {
	f := RequestFidelity{Runner: r.Name()}
	a, ok := r.(RequestAdapter)
	if !ok {
		return f
	}
	f.Declared = true
	eff := a.EffectiveRequest(c)

	claimed := map[string]bool{}
	for _, name := range eff.Unsupported {
		if name = strings.TrimSpace(name); name != "" {
			claimed[name] = true
		}
	}
	for _, name := range requestFields {
		if !claimed[name] {
			continue
		}
		delete(claimed, name)
		f.Unsupported = append(f.Unsupported, name)
		if requestFieldSet(c, name) {
			f.Dropped = append(f.Dropped, name)
		}
	}
	// A claim naming a field this package does not define cannot be checked
	// against the case at all, so it is reported verbatim AND counted as
	// dropped: an unverifiable capability claim is never a pass, and a typo'd
	// field name is exactly the silent omission this record exists to surface.
	unknown := make([]string, 0, len(claimed))
	for name := range claimed {
		unknown = append(unknown, name)
	}
	sort.Strings(unknown)
	f.Unsupported = append(f.Unsupported, unknown...)
	f.Dropped = append(f.Dropped, unknown...)

	dropped := map[string]bool{}
	for _, name := range f.Dropped {
		dropped[name] = true
	}
	want, got := requestValues(c.Prompt, c.Params), requestValues(eff.Prompt, eff.Params)
	for _, name := range requestFields {
		if dropped[name] || want[name] == got[name] {
			continue
		}
		f.Diff = append(f.Diff, FieldDelta{Field: name, Requested: want[name], Effective: got[name]})
	}
	return f
}

// requestValues renders one request as field name → canonical string, so two
// requests compare by value without a per-field comparison ladder. Floats use
// the shortest exact representation, which keeps a rendered delta readable
// without introducing a rounding tolerance the request layer must not have.
func requestValues(prompt string, p SamplingParams) map[string]string {
	return map[string]string{
		"prompt":             prompt,
		"params.temperature": strconv.FormatFloat(p.Temperature, 'g', -1, 64),
		"params.top_k":       strconv.Itoa(p.TopK),
		"params.top_p":       strconv.FormatFloat(p.TopP, 'g', -1, 64),
		"params.max_tokens":  strconv.Itoa(p.MaxTokens),
		"params.seed":        strconv.FormatInt(p.Seed, 10),
	}
}

// requestFieldSet reports whether the case actually SPECIFIES a field. An
// unsupported field the case never set costs the run nothing, and failing every
// case on a static capability claim would make the declaration unusable — a
// llama.cpp reference or a fixed-seed engine mode always has something it does
// not implement. "Specified" follows the struct's own omitempty tags: an
// omitempty field left at its zero value was never written by the case author.
// Temperature and MaxTokens carry no omitempty and always count — temperature 0
// is the deliberate greedy setting, and ValidateCanonical requires max_tokens.
func requestFieldSet(c QualityCase, name string) bool {
	switch name {
	case "prompt":
		return c.Prompt != ""
	case "params.temperature", "params.max_tokens":
		return true
	case "params.top_k":
		return c.Params.TopK != 0
	case "params.top_p":
		return c.Params.TopP != 0
	case "params.seed":
		return c.Params.Seed != 0
	}
	return false
}

// requestFidelityVerdict folds both sides into the run's FIRST verdict when
// either did not execute the case's normalized request, and reports false when
// both did. The reference side is reported first: if the golden path itself ran
// a different request there is no baseline to judge anything against, and
// naming the engine's drift first would send an operator to the wrong side of
// the comparison. Its Kind is "request" — a closed kind the localizer routes on,
// distinct from "differential" and "rubric" because nothing was decoded yet.
func requestFidelityVerdict(rec RequestRecord) (Verdict, bool) {
	var first RequestFidelity
	switch {
	case !rec.Reference.Faithful():
		first = rec.Reference
	case !rec.Engine.Faithful():
		first = rec.Engine
	default:
		return Verdict{}, false
	}
	off := first.offenses()
	detail := fmt.Sprintf("runner %q did not execute the case's normalized request; first offending field — %s",
		first.Runner, off[0])
	if rest := off[1:]; len(rest) > 0 {
		detail += fmt.Sprintf(" (then: %s)", strings.Join(rest, "; "))
	}
	return Verdict{Oracle: "request-fidelity", Kind: "request", Pass: false, Detail: detail}, true
}
