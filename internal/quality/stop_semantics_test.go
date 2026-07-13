package quality

import (
	"strings"
	"testing"
)

// stopSemanticsCase is the hermetic stop-semantics fixture (#4528): a tiny greedy
// decode whose reference terminates on the EOS sentinel well inside the truncation
// cap, with one declared stop string that must never reach the engine text.
func stopSemanticsCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "stop-semantics-eos-truncation",
		Version: 1,
		Prompt:  "List the deploy steps, then stop.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 6},
		Reference: Trace{
			Tokens: []string{"Build", "test", "ship", StopTruncationEOS},
			Text:   "Build test ship",
		},
		Oracles: []string{"stop-truncation"},
		Rubric:  RubricSpec{Forbidden: []string{"STOP_MARKER"}},
	}
}

// stopSemanticsEngine mirrors the DemoEngine defect-injection pattern for the stop
// gates: "" replays the reference (clean pass); "past-eos" emits one token after
// the hard stop; "over-cap" ignores the truncation cap; "stop-string" lets a
// declared stop string leak into the assembled text; "early-eos" stops one step
// early on its own EOS (a valid earlier stop); "short" stops early with NO stop
// fired.
func stopSemanticsEngine(defect string) ScriptedRunner {
	ref := stopSemanticsCase().Reference
	switch defect {
	case "past-eos":
		toks := append(append([]string(nil), ref.Tokens...), "deploy")
		return ScriptedRunner{
			Label: "engine-past-eos",
			Trace: Trace{Tokens: toks, Text: "Build test ship deploy"},
		}
	case "over-cap":
		return ScriptedRunner{
			Label: "engine-over-cap",
			Trace: Trace{
				Tokens: []string{"Build", "test", "ship", "then", "keep", "on", "going", "forever"},
				Text:   "Build test ship then keep on going forever",
			},
		}
	case "stop-string":
		return ScriptedRunner{
			Label: "engine-stop-string",
			Trace: Trace{Tokens: ref.Tokens, Text: "Build test ship STOP_MARKER trailing prose"},
		}
	case "early-eos":
		return ScriptedRunner{
			Label: "engine-early-eos",
			Trace: Trace{Tokens: []string{"Build", "test", StopTruncationEOS}, Text: "Build test"},
		}
	case "short":
		return ScriptedRunner{
			Label: "engine-short",
			Trace: Trace{Tokens: []string{"Build", "test"}, Text: "Build test"},
		}
	default:
		return ScriptedRunner{Label: "engine-clean", Trace: ref}
	}
}

// TestStopTruncationCleanPasses proves a faithful engine — one that stops exactly
// at the EOS sentinel, within MaxTokens, emitting no stop string — passes the
// stop-truncation gate with no failure bundle.
func TestStopTruncationCleanPasses(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("clean stop-semantics case should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestStopTruncationPastEOSFails is the over-generation gate: one token emitted
// after the first EOS sentinel must fail AT that offending index.
func TestStopTruncationPastEOSFails(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine("past-eos"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("generation past EOS must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "stop-truncation" || fb.FailingKind != "differential" {
		t.Errorf("failing oracle = %q (%s), want stop-truncation (differential)", fb.FailingOracle, fb.FailingKind)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 4 {
		t.Fatalf("expected first divergence at token 4 (first token past EOS), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Engine != "deploy" {
		t.Errorf("offending engine token = %q, want %q", fb.FirstDivergence.Engine, "deploy")
	}
}

// TestStopTruncationOverCapFails is the truncation gate: an engine trace longer
// than Params.MaxTokens must fail with the divergence pinned at index MaxTokens.
func TestStopTruncationOverCapFails(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine("over-cap"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("trace exceeding max_tokens must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "stop-truncation" {
		t.Errorf("failing oracle = %q, want stop-truncation", fb.FailingOracle)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != c.Params.MaxTokens {
		t.Fatalf("expected first divergence at index MaxTokens=%d, got %+v", c.Params.MaxTokens, fb.FirstDivergence)
	}
}

// TestStopTruncationStopStringFails is the stop-string gate: a declared stop
// string (Rubric.Forbidden) present in the assembled engine text must fail with
// the string named in the detail.
func TestStopTruncationStopStringFails(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine("stop-string"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("emitted stop string must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "stop-truncation" {
		t.Errorf("failing oracle = %q, want stop-truncation", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, "STOP_MARKER") {
		t.Errorf("detail must name the offending stop string; got %q", fb.Detail)
	}
}

// TestStopTruncationEarlyEOSPasses proves the reference-consistency rule excuses
// an engine that terminated EARLIER than the reference on a valid stop of its own
// (its trace ends on the EOS sentinel).
func TestStopTruncationEarlyEOSPasses(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine("early-eos"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("early termination on the engine's own EOS is a valid stop; got %s", Explain(res))
	}
}

// TestStopTruncationShortNoStopFails is the reference-consistency gate: an engine
// that terminated before the reference with NO stop fired (no EOS, under the cap)
// must fail at the step where the reference kept going.
func TestStopTruncationShortNoStopFails(t *testing.T) {
	c := stopSemanticsCase()
	res, err := RunCase(c, ReferenceRunner{}, stopSemanticsEngine("short"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("early termination with no stop fired must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 2 {
		t.Fatalf("expected termination divergence at token 2, got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != "ship" {
		t.Errorf("reference token at divergence = %q, want %q", fb.FirstDivergence.Reference, "ship")
	}
}
