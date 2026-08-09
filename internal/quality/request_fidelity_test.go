package quality

import (
	"encoding/json"
	"strings"
	"testing"
)

// This file is the captured witness for the reference-runner adapter (#4518).
// The defect it plants is the one differential testing is structurally blind to:
// an adapter that silently drops or rewrites a normalized request field. Both
// sides then emit IDENTICAL traces, every decode oracle agrees, and the run goes
// green while proving nothing — the two paths answered different questions. Each
// test below therefore pairs the planted defect with the fixed adapter on the
// SAME case and the SAME traces, so the only thing that moved is the request
// record.

// fidelityRunner is a test adapter that implements RequestAdapter: it replays a
// fixed trace like ScriptedRunner and, separately, declares the request it
// actually executed. The two are independent on purpose — the whole class of
// defect here is a runner whose OUTPUT looks right while its INPUT was rewritten.
type fidelityRunner struct {
	label       string
	trace       Trace
	unsupported []string
	// rewrite plants a substitution: it mutates the effective request after it is
	// seeded from the case, modelling a runner that clamps, coerces, or rewrites.
	rewrite func(*EffectiveRequest)
}

func (f fidelityRunner) Name() string { return f.label }

func (f fidelityRunner) Run(_ QualityCase) (Trace, error) {
	t := f.trace
	t.Runner = f.Name()
	return t, nil
}

func (f fidelityRunner) EffectiveRequest(c QualityCase) EffectiveRequest {
	eff := EffectiveRequest{Prompt: c.Prompt, Params: c.Params, Unsupported: f.unsupported}
	if f.rewrite != nil {
		f.rewrite(&eff)
	}
	return eff
}

// topKCase is the demo case with top_k actually SPECIFIED, so that dropping it is
// a real loss rather than a capability claim the case never exercises.
func topKCase() QualityCase {
	c := DemoCase()
	c.Params.TopK = 40
	return c
}

// requestVerdict returns the run's request-fidelity verdict, or nil.
func requestVerdict(res Result) *Verdict {
	for i := range res.Verdicts {
		if res.Verdicts[i].Oracle == "request-fidelity" {
			return &res.Verdicts[i]
		}
	}
	return nil
}

// TestRequestDropFailsAndFixPasses is the #4518 Witness: the planted defect (an
// engine adapter that silently drops the case's top_k) fails, and the fixed
// adapter — same case, same emitted trace — passes. It also proves the request
// record is LOAD-BEARING: in the failing run every decode/rubric oracle agrees,
// so nothing but the request check stood between this defect and a false green.
func TestRequestDropFailsAndFixPasses(t *testing.T) {
	c := topKCase()
	clean := c.Reference

	// BEFORE the adapter seam: a runner had no way to declare the request it ran,
	// so the same engine — decoding under the wrong top_k and therefore emitting a
	// trace it was never entitled to — is an unqualified green. This is the false
	// pass #4518 exists to close, reproduced rather than asserted.
	unwired, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Label: "engine-topk-dropped", Trace: clean}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("unwired RunCase: %v", err)
	}
	if !unwired.Pass {
		t.Fatalf("the pre-adapter baseline must reproduce the false pass; got %s", Explain(unwired))
	}

	defective := fidelityRunner{
		label:       "engine-topk-dropped",
		trace:       clean,
		unsupported: []string{"params.top_k"},
	}
	res, err := RunCase(c, ReferenceRunner{}, defective, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("an engine that silently dropped the case's top_k must not pass; got %s", Explain(res))
	}
	v := requestVerdict(res)
	if v == nil {
		t.Fatalf("failing run carries no request-fidelity verdict; got %s", Explain(res))
	}
	if res.Verdicts[0].Oracle != "request-fidelity" {
		t.Errorf("request fidelity must be judged first; verdict 0 = %q", res.Verdicts[0].Oracle)
	}
	if v.Kind != "request" {
		t.Errorf("request verdict kind = %q, want request", v.Kind)
	}
	if !strings.Contains(v.Detail, "params.top_k") {
		t.Errorf("detail must name the first offending field; got %q", v.Detail)
	}
	// Load-bearing: every OTHER oracle agreed. The traces are byte-identical, so
	// without this check the run would have been an unqualified pass.
	for _, other := range res.Verdicts {
		if other.Oracle != "request-fidelity" && !other.Pass {
			t.Fatalf("oracle %q also failed; the defect must be caught by the request record alone: %s",
				other.Oracle, other.Detail)
		}
	}
	eng := res.Provenance.Requests.Engine
	if !eng.Declared {
		t.Error("an adapter implementing RequestAdapter must be recorded as declared")
	}
	if len(eng.Dropped) != 1 || eng.Dropped[0] != "params.top_k" {
		t.Errorf("dropped = %v, want [params.top_k]", eng.Dropped)
	}
	if eng.Faithful() || res.Provenance.Requests.Faithful() {
		t.Error("a run that dropped a specified field is not faithful")
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("a failing run must emit a replay bundle")
	}
	if fb.FailingKind != "request" || fb.FailingOracle != "request-fidelity" {
		t.Errorf("bundle first failure = %q/%q, want request-fidelity/request", fb.FailingOracle, fb.FailingKind)
	}
	if !fb.Scrubbed {
		t.Error("emitted bundle must be scrubbed")
	}
	if len(fb.Requests.Engine.Dropped) != 1 {
		t.Errorf("bundle must carry the request record so the failure replays from the bundle alone: %+v", fb.Requests)
	}

	// The fix: the same adapter, honoring top_k. Nothing else changes.
	fixed := fidelityRunner{label: "engine-topk-honored", trace: clean}
	after, err := RunCase(c, ReferenceRunner{}, fixed, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("fixed RunCase: %v", err)
	}
	if !after.Pass {
		t.Fatalf("the fixed adapter must pass; got %s", Explain(after))
	}
	if after.FailureBundle != nil {
		t.Errorf("a passing run must not carry a bundle: %+v", after.FailureBundle)
	}
	if requestVerdict(after) != nil {
		t.Error("a faithful run must not append a request verdict")
	}
	if !after.Provenance.Requests.Engine.Declared {
		t.Error("the fixed adapter still declares its request; an audited pass must be distinguishable from an unaudited one")
	}
}

// TestRequestSubstitutionIsExplicit proves the second loss mode — a field the
// runner honors by NAME but executes with a different value — is recorded as an
// explicit requested-vs-effective delta rather than being invisible.
func TestRequestSubstitutionIsExplicit(t *testing.T) {
	c := DemoCase()
	clamped := fidelityRunner{
		label:   "engine-maxtokens-clamped",
		trace:   c.Reference,
		rewrite: func(e *EffectiveRequest) { e.Params.MaxTokens = 4 },
	}
	res, err := RunCase(c, ReferenceRunner{}, clamped, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("a clamped max_tokens must not pass silently; got %s", Explain(res))
	}
	diff := res.Provenance.Requests.Engine.Diff
	if len(diff) != 1 {
		t.Fatalf("diff = %+v, want exactly the clamped field", diff)
	}
	if diff[0].Field != "params.max_tokens" || diff[0].Requested != "8" || diff[0].Effective != "4" {
		t.Errorf("delta = %+v, want params.max_tokens 8 -> 4", diff[0])
	}
	if len(res.Provenance.Requests.Engine.Dropped) != 0 {
		t.Errorf("a substituted field is a diff, not a drop: %v", res.Provenance.Requests.Engine.Dropped)
	}
}

// TestRequestOffensesAreCanonicallyOrdered pins the localization contract: the
// FIRST field a failing run names is a property of the request shape, not of map
// iteration, so a replayed failure indicts the same field every time. Drops are
// named before substitutions, and each group follows the canonical field order.
func TestRequestOffensesAreCanonicallyOrdered(t *testing.T) {
	c := topKCase()
	c.Params.Seed = 7
	messy := fidelityRunner{
		label: "engine-multi-drift",
		trace: c.Reference,
		// Declared out of canonical order on purpose.
		unsupported: []string{"params.seed", "params.top_k"},
		rewrite:     func(e *EffectiveRequest) { e.Params.MaxTokens = 4; e.Params.Temperature = 0.7 },
	}
	res, err := RunCase(c, ReferenceRunner{}, messy, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	rec := res.Provenance.Requests.Engine
	if got, want := strings.Join(rec.Unsupported, ","), "params.top_k,params.seed"; got != want {
		t.Errorf("unsupported = %q, want canonical order %q", got, want)
	}
	if got, want := strings.Join(rec.Dropped, ","), "params.top_k,params.seed"; got != want {
		t.Errorf("dropped = %q, want canonical order %q", got, want)
	}
	if len(rec.Diff) != 2 || rec.Diff[0].Field != "params.temperature" || rec.Diff[1].Field != "params.max_tokens" {
		t.Fatalf("diff = %+v, want temperature then max_tokens", rec.Diff)
	}
	v := requestVerdict(res)
	if v == nil {
		t.Fatalf("no request verdict; got %s", Explain(res))
	}
	if !strings.Contains(v.Detail, "first offending field — params.top_k:") {
		t.Errorf("detail must lead with the canonically first offence; got %q", v.Detail)
	}
	// Replay identity: the same inputs must name the same first offender.
	again, err := RunCase(c, ReferenceRunner{}, messy, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("replay RunCase: %v", err)
	}
	if got := requestVerdict(again); got == nil || got.Detail != v.Detail {
		t.Errorf("replayed detail drifted: %+v vs %q", got, v.Detail)
	}
}

// TestRequestReferenceDriftReportedFirst proves that when BOTH sides drifted the
// reference is indicted first: if the golden path ran a different request there
// is no baseline at all, and naming the engine would send an operator to the
// wrong side of the comparison.
func TestRequestReferenceDriftReportedFirst(t *testing.T) {
	c := topKCase()
	ref := fidelityRunner{label: "reference-drifted", trace: c.Reference, unsupported: []string{"params.top_k"}}
	eng := fidelityRunner{
		label:   "engine-drifted",
		trace:   c.Reference,
		rewrite: func(e *EffectiveRequest) { e.Params.MaxTokens = 1 },
	}
	res, err := RunCase(c, ref, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	v := requestVerdict(res)
	if v == nil {
		t.Fatalf("no request verdict; got %s", Explain(res))
	}
	if !strings.Contains(v.Detail, `"reference-drifted"`) {
		t.Errorf("the reference side must be indicted first; got %q", v.Detail)
	}
	if res.Provenance.Requests.Engine.Faithful() {
		t.Error("the engine's own drift must still be recorded, not swallowed by the reference's")
	}
}

// TestRequestUnknownFieldNeverPasses proves an unverifiable capability claim — a
// runner naming a field this package does not define, e.g. a typo — is reported
// verbatim AND counted as a loss. "Missing or inconclusive evidence is never
// pass" applies to the claim itself, not only to the decode.
func TestRequestUnknownFieldNeverPasses(t *testing.T) {
	c := DemoCase()
	typo := fidelityRunner{
		label:       "engine-typo-claim",
		trace:       c.Reference,
		unsupported: []string{"params.topk"},
	}
	res, err := RunCase(c, ReferenceRunner{}, typo, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("an uncheckable capability claim must not pass; got %s", Explain(res))
	}
	rec := res.Provenance.Requests.Engine
	if len(rec.Dropped) != 1 || rec.Dropped[0] != "params.topk" {
		t.Errorf("dropped = %v, want the verbatim unknown claim", rec.Dropped)
	}
}

// TestRequestUnexercisedClaimIsFree proves the declaration stays usable: every
// real adapter has something it does not implement, so an unsupported field the
// case never SET costs the run nothing while still being recorded.
func TestRequestUnexercisedClaimIsFree(t *testing.T) {
	c := DemoCase() // no top_k, no top_p, no seed
	partial := fidelityRunner{
		label:       "engine-no-seed-support",
		trace:       c.Reference,
		unsupported: []string{"params.seed", "params.top_p"},
	}
	res, err := RunCase(c, ReferenceRunner{}, partial, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("an unexercised capability claim must not fail the case; got %s", Explain(res))
	}
	rec := res.Provenance.Requests.Engine
	if got, want := strings.Join(rec.Unsupported, ","), "params.top_p,params.seed"; got != want {
		t.Errorf("unsupported = %q, want %q recorded even when unexercised", got, want)
	}
	if len(rec.Dropped) != 0 || !rec.Faithful() {
		t.Errorf("nothing was translated away, so the run is faithful; got %+v", rec)
	}
}

// TestRequestUndeclaredRunnerIsAdditive proves the seam is OPTIONAL: the runners
// that shipped before it satisfy Runner unedited, keep the exact verdict list
// they had, and are recorded as UNdeclared so an audited faithful run stays
// distinguishable from an assumed one.
func TestRequestUndeclaredRunnerIsAdditive(t *testing.T) {
	c := topKCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("a pre-existing runner must keep passing; got %s", Explain(res))
	}
	if n := len(res.Verdicts); n != len(c.Oracles) {
		t.Errorf("verdicts = %d, want %d: a faithful run appends nothing", n, len(c.Oracles))
	}
	// ScriptedRunner predates the seam and still satisfies Runner unedited: it is
	// recorded as UNdeclared, and treated as faithful, so nothing it shipped before
	// the seam existed changed underneath it.
	eng := res.Provenance.Requests.Engine
	if eng.Declared {
		t.Errorf("runner %q does not implement RequestAdapter and must not be recorded as declared", eng.Runner)
	}
	if !eng.Faithful() {
		t.Errorf("an undeclared runner is treated as faithful; got %+v", eng)
	}
	if res.Provenance.Requests.Reference.Runner != "reference" {
		t.Errorf("record must name the runner; got %q", res.Provenance.Requests.Reference.Runner)
	}
}

// TestReferenceRunnerDeclaresItsRequest pins the concrete adapter: the golden
// path is AUDITED, not assumed. The reference is what every other verdict is
// measured against, so a result that audited only the engine would rest its
// baseline on an assumption — the "missing or inconclusive evidence is never
// pass" rule applied to the side that defines what passing means.
func TestReferenceRunnerDeclaresItsRequest(t *testing.T) {
	c := topKCase() // top_k SET, so an unsupported-field claim would be a real drop
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	ref := res.Provenance.Requests.Reference
	if !ref.Declared {
		t.Error("ReferenceRunner must implement RequestAdapter so the baseline is measured, not assumed")
	}
	if !ref.Faithful() {
		t.Errorf("the reference answers the case's own request by definition; got %+v", ref)
	}
	if len(ref.Unsupported) != 0 {
		t.Errorf("the reference supports every normalized field; got %v", ref.Unsupported)
	}
	// Declaring the baseline must not perturb a clean run: same verdicts, no bundle.
	if !res.Pass {
		t.Fatalf("a clean run must still pass; got %s", Explain(res))
	}
	if n := len(res.Verdicts); n != len(c.Oracles) {
		t.Errorf("verdicts = %d, want %d: a faithful reference appends nothing", n, len(c.Oracles))
	}
	if _, ok := any(ReferenceRunner{}).(RequestAdapter); !ok {
		t.Error("ReferenceRunner must satisfy RequestAdapter")
	}
}

// TestRequestDeltaScrubbedInBundle proves a request delta is scrubbed like every
// other quoted surface — a prompt rewrite echoes request text verbatim — and
// that scrubbing the bundle does not reach back through the shared backing array
// into the Result's own Provenance.
func TestRequestDeltaScrubbedInBundle(t *testing.T) {
	const secret = "api_key=sk-live-should-not-travel"
	c := DemoCase()
	leaky := fidelityRunner{
		label:   "engine-prompt-rewritten",
		trace:   c.Reference,
		rewrite: func(e *EffectiveRequest) { e.Prompt = c.Prompt + " " + secret },
	}
	res, err := RunCase(c, ReferenceRunner{}, leaky, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("a rewritten prompt must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("a failing run must emit a bundle")
	}
	blob, err := json.Marshal(fb)
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}
	if strings.Contains(string(blob), "sk-live-should-not-travel") {
		t.Errorf("the portable bundle leaked a secret through its request delta: %s", blob)
	}
	if len(fb.Requests.Engine.Diff) != 1 || !strings.Contains(fb.Requests.Engine.Diff[0].Effective, "[REDACTED]") {
		t.Errorf("bundle delta must be redacted, not dropped: %+v", fb.Requests.Engine.Diff)
	}
	// The aliasing guard: the in-memory provenance the caller still holds was not
	// rewritten underneath it by the bundle's scrub.
	prov := res.Provenance.Requests.Engine.Diff
	if len(prov) != 1 || !strings.Contains(prov[0].Effective, "sk-live-should-not-travel") {
		t.Errorf("scrubbing the bundle must not mutate the result's own provenance: %+v", prov)
	}
}
