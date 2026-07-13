package quality

import (
	"strings"
	"testing"
)

// instrFaithfulReport obeys every instruction instrCase declares: 16 words
// (cap 30), every line bulleted, both required fields present, no forbidden
// content.
const instrFaithfulReport = "- Throughput increased 12% week over week.\n" +
	"- Cache hit rate reached 91% after the rebalance."

// instrCase builds a hermetic report case judged only by the
// instruction-following oracle: a word cap and a bullet-format constraint
// carried as the Prompt's INSTRUCTIONS spec, two required fields, and one
// forbidden phrase — five explicit instructions in total, with the default
// threshold (every instruction must be followed).
func instrCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "report-instruction-following",
		Version: 1,
		Prompt: "Write the weekly engineering rollup for the executive audience.\n" +
			`INSTRUCTIONS: {"max_words":30,"line_prefix":"- "}`,
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: instrFaithfulReport},
		Oracles:   []string{"instruction-following"},
		Rubric: RubricSpec{
			Required:  []string{"throughput", "cache hit rate"},
			Forbidden: []string{"confidential"},
		},
	}
}

// instrVerdict pulls the instruction-following verdict out of a result or
// fails the test.
func instrVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "instruction-following" {
			return v
		}
	}
	t.Fatalf("no instruction-following verdict in %s", Explain(res))
	return Verdict{}
}

// instrRun runs instrCase (optionally mutated) against a scripted engine text.
func instrRun(t *testing.T, c QualityCase, text string) Result {
	t.Helper()
	eng := ScriptedRunner{Label: "engine-under-test", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	return res
}

// TestInstructionFollowingFaithfulReportPasses is the faithful path: a report
// obeying the length cap, the bullet format, both required fields, and the
// forbidden-content ban passes with a full score and no failure bundle.
func TestInstructionFollowingFaithfulReportPasses(t *testing.T) {
	res := instrRun(t, instrCase(), instrFaithfulReport)
	if !res.Pass {
		t.Fatalf("obedient report should pass; got %s", Explain(res))
	}
	v := instrVerdict(t, res)
	if v.Score != 1 {
		t.Errorf("obedient report score = %v, want 1", v.Score)
	}
	if want := "all 5 explicit instruction(s) followed"; v.Detail != want {
		t.Errorf("Detail = %q, want %q", v.Detail, want)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestInstructionFollowingLengthCapViolationFails is the defect Witness for the
// length-cap instruction: a report that obeys everything else but runs past the
// 30-word cap must fail, and the Detail must name the cap with the actual word
// count.
func TestInstructionFollowingLengthCapViolationFails(t *testing.T) {
	text := instrFaithfulReport +
		"\n- The rebalance also cut tail latency, reduced hot-shard contention, simplified" +
		" the pager rotation, and unlocked the deferred capacity work planned for next quarter."
	res := instrRun(t, instrCase(), text)
	if res.Pass {
		t.Fatalf("report exceeding the word cap must not pass; got %s", Explain(res))
	}
	v := instrVerdict(t, res)
	if v.Pass {
		t.Fatal("instruction-following verdict should have failed")
	}
	if v.Score != 0.8 {
		t.Errorf("score = %v, want 0.8 (4 of 5 instructions obeyed)", v.Score)
	}
	if !strings.Contains(v.Detail, "length cap violated") || !strings.Contains(v.Detail, "cap is 30") {
		t.Errorf("Detail %q does not name the violated length cap", v.Detail)
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "instruction-following" {
		t.Errorf("first failing oracle = %q, want instruction-following", fb.FailingOracle)
	}
}

// TestInstructionFollowingOmissionAndForbiddenFail injects two violations at
// once — the report drops the required "cache hit rate" field and includes the
// forbidden "confidential" content — and asserts the score counts both while
// the Detail names the FIRST violated instruction per the documented order
// (required fields before forbidden content).
func TestInstructionFollowingOmissionAndForbiddenFail(t *testing.T) {
	text := "- Throughput increased 12% week over week.\n" +
		"- Confidential: the vendor contract is being renegotiated."
	res := instrRun(t, instrCase(), text)
	if res.Pass {
		t.Fatalf("report omitting a required field and leaking forbidden content must not pass; got %s", Explain(res))
	}
	v := instrVerdict(t, res)
	if v.Score != 0.6 {
		t.Errorf("score = %v, want 0.6 (3 of 5 instructions obeyed)", v.Score)
	}
	if want := `required field omitted: report does not include "cache hit rate"`; !strings.Contains(v.Detail, want) {
		t.Errorf("Detail %q missing first violation %q", v.Detail, want)
	}
	if strings.Contains(v.Detail, "confidential") {
		t.Errorf("Detail %q should name only the FIRST violated instruction", v.Detail)
	}
}

// TestInstructionFollowingFormatViolationFails is the format-constraint defect:
// one line missing the ordered "- " bullet prefix fails, and the Detail
// localizes the violation to that line.
func TestInstructionFollowingFormatViolationFails(t *testing.T) {
	text := "- Throughput increased 12% week over week.\n" +
		"Cache hit rate reached 91% after the rebalance."
	res := instrRun(t, instrCase(), text)
	if res.Pass {
		t.Fatalf("report breaking the bullet format must not pass; got %s", Explain(res))
	}
	v := instrVerdict(t, res)
	if v.Score != 0.8 {
		t.Errorf("score = %v, want 0.8 (4 of 5 instructions obeyed)", v.Score)
	}
	for _, want := range []string{"format violated", "line 2", `required prefix "- "`} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing %q", v.Detail, want)
		}
	}
}

// TestInstructionFollowingSpecEdges pins the spec-parsing edges: a case with no
// instructions at all passes with a note (nothing to disobey), and a malformed
// INSTRUCTIONS line is skipped rather than judged — the rubric instructions
// still apply.
func TestInstructionFollowingSpecEdges(t *testing.T) {
	none := instrCase()
	none.Prompt = "Write the weekly engineering rollup."
	none.Rubric = RubricSpec{}
	res := instrRun(t, none, "Anything at all.")
	if !res.Pass {
		t.Fatalf("case declaring no instructions should pass; got %s", Explain(res))
	}
	if v := instrVerdict(t, res); !strings.Contains(v.Detail, "no explicit instructions declared") {
		t.Errorf("Detail = %q, want the nothing-to-check note", v.Detail)
	}

	malformed := instrCase()
	malformed.Prompt = "Write the rollup.\nINSTRUCTIONS: {not json"
	// The unparseable spec is skipped, so only the 3 rubric instructions count:
	// an over-long, unbulleted — but complete and clean — report still passes.
	long := "Throughput increased 12% week over week and the cache hit rate reached 91%" +
		" after the rebalance, with tail latency down, contention down, the pager" +
		" rotation simplified, and the deferred capacity work unlocked for next quarter."
	res = instrRun(t, malformed, long)
	if !res.Pass {
		t.Fatalf("malformed spec must be skipped, not enforced; got %s", Explain(res))
	}
	if v := instrVerdict(t, res); !strings.Contains(v.Detail, "all 3 explicit instruction(s) followed") {
		t.Errorf("Detail = %q, want 3 rubric-only instructions followed", v.Detail)
	}
}
