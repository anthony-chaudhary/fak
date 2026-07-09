package livecodebench

import (
	"strings"
	"testing"
)

// Golden fixtures: one problem per scenario, with the EXACT rendered user
// message and the EXACT prompt hash pinned as literals. The pinned hex is the
// stability witness #2095 demands — any drift in assembly, framing, or system
// prompts breaks a literal below, so a silent workload-identity change cannot
// land.

func codegenGoldenProblem() Problem {
	return Problem{
		QuestionID:  "q-codegen",
		Scenario:    ScenarioCodeGeneration,
		Prompt:      "Given a list of integers, return the sum.",
		StarterCode: "class Solution:\n    def sumList(self, nums: List[int]) -> int:",
	}
}

const codegenGoldenUser = "### Question:\n" +
	"Given a list of integers, return the sum.\n\n" +
	"### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n" +
	"```python\n" +
	"class Solution:\n    def sumList(self, nums: List[int]) -> int:\n" +
	"```\n\n" +
	"### Answer: (use the provided format with backticks)\n"

const codegenGoldenHash = "90734f82dddfa1bb8a52714215a92e51a32186d91906022455843605436331b8"

func TestNormalizeProblemCodegenGolden(t *testing.T) {
	call, err := NormalizeProblem(codegenGoldenProblem(), ScenarioCodeGeneration)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	assertCallShape(t, call, ScenarioCodeGeneration, "q-codegen", codegenSystemPrompt, codegenGoldenUser, codegenGoldenHash)
}

// The stdin branch: a problem with no starter code gets the read-stdin /
// write-stdout format instructions instead of a starter-code block.
const codegenStdinGoldenUser = "### Question:\n" +
	"Read n, then print n doubled.\n\n" +
	"### Format: Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n" +
	"```python\n" +
	"# YOUR CODE HERE\n" +
	"```\n\n" +
	"### Answer: (use the provided format with backticks)\n"

const codegenStdinGoldenHash = "4c7bdfa6dd6362d065681eade8edabd80a0fa184e366f2b0a1541745864adbb4"

func TestNormalizeProblemCodegenStdinGolden(t *testing.T) {
	p := Problem{QuestionID: "q-stdin", Prompt: "Read n, then print n doubled."}
	call, err := NormalizeProblem(p, ScenarioCodeGeneration)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	assertCallShape(t, call, ScenarioCodeGeneration, "q-stdin", codegenSystemPrompt, codegenStdinGoldenUser, codegenStdinGoldenHash)
}

// Self-repair shares the codegen user body (the question + format base the
// runtime BuildRepairPrompt threads the wrong attempt and feedback onto) but
// carries its own system prompt — so its hash MUST differ from codegen's over
// the same problem.
const selfRepairGoldenHash = "a2816766affe648597ee86f379c13a08acdb7cd291124c2c6b1360d82c6f84b5"

func TestNormalizeProblemSelfRepairGolden(t *testing.T) {
	p := codegenGoldenProblem()
	p.QuestionID = "q-repair"
	p.Scenario = ScenarioSelfRepair
	call, err := NormalizeProblem(p, ScenarioSelfRepair)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	assertCallShape(t, call, ScenarioSelfRepair, "q-repair", selfRepairSystemPrompt, codegenGoldenUser, selfRepairGoldenHash)
	if call.PromptSHA256 == codegenGoldenHash {
		t.Fatalf("self-repair hash equals codegen hash %s; the system prompt must be part of the identity", call.PromptSHA256)
	}
}

const testOutputGoldenUser = "Predict the exact output this program produces for the given input. Return only the output, with no explanation.\n\n" +
	"## Problem\n" +
	"Sum two numbers from stdin.\n\n" +
	"## Code\n" +
	"a, b = map(int, input().split())\nprint(a + b)\n\n" +
	"## Input\n" +
	"2 3\n\n" +
	"## Output\n"

const testOutputGoldenHash = "6d716167bd3c5404d250a704229abc8cb074427acb52c8e78ed01ddca9b98da4"

func TestNormalizeProblemTestOutputGolden(t *testing.T) {
	p := Problem{
		QuestionID:  "q-testoutput",
		Scenario:    ScenarioTestOutputPrediction,
		Prompt:      "Sum two numbers from stdin.",
		StarterCode: "a, b = map(int, input().split())\nprint(a + b)",
		PublicTests: []TestCase{{Input: "2 3", Output: "5", TestType: "stdin"}},
	}
	call, err := NormalizeProblem(p, ScenarioTestOutputPrediction)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	assertCallShape(t, call, ScenarioTestOutputPrediction, "q-testoutput", testOutputSystemPrompt, testOutputGoldenUser, testOutputGoldenHash)
}

const codeExecutionGoldenUser = "Determine the exact output this program produces when run on the given input. Return only the output, with no explanation.\n\n" +
	"## Program\n" +
	"x = int(input())\nprint(x * x)\n\n" +
	"## Input\n" +
	"4\n\n" +
	"## Output\n"

const codeExecutionGoldenHash = "2695b1e43b8dd7f17f4c4a60753796b82624ff3d3f0aae507715548872667bcd"

func TestNormalizeProblemCodeExecutionGolden(t *testing.T) {
	p := Problem{
		QuestionID:  "q-codeexec",
		Scenario:    ScenarioCodeExecution,
		Prompt:      "What does this program print?",
		StarterCode: "x = int(input())\nprint(x * x)",
		PublicTests: []TestCase{{Input: "4", Output: "16", TestType: "stdin"}},
	}
	call, err := NormalizeProblem(p, ScenarioCodeExecution)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	assertCallShape(t, call, ScenarioCodeExecution, "q-codeexec", codeExecutionSystemPrompt, codeExecutionGoldenUser, codeExecutionGoldenHash)
}

// assertCallShape pins one normalized call against its golden: the two-message
// system+user shape, the exact rendered contents, the recomputed hash of the
// canonical rendering, and the pinned literal hex.
func assertCallShape(t *testing.T, call GenerationCall, scenario Scenario, qid, system, user, hash string) {
	t.Helper()
	if call.Scenario != scenario {
		t.Fatalf("scenario = %q, want %q", call.Scenario, scenario)
	}
	if call.QuestionID != qid {
		t.Fatalf("question_id = %q, want %q", call.QuestionID, qid)
	}
	if len(call.Messages) != 2 {
		t.Fatalf("messages = %d, want 2 (system + user)", len(call.Messages))
	}
	if call.Messages[0].Role != GenerationRoleSystem || call.Messages[1].Role != GenerationRoleUser {
		t.Fatalf("roles = %q,%q want %q,%q", call.Messages[0].Role, call.Messages[1].Role, GenerationRoleSystem, GenerationRoleUser)
	}
	if call.Messages[0].Content != system {
		t.Fatalf("system message drifted:\ngot:  %q\nwant: %q", call.Messages[0].Content, system)
	}
	if call.Messages[1].Content != user {
		t.Fatalf("user message drifted:\ngot:  %q\nwant: %q", call.Messages[1].Content, user)
	}
	if got := promptSHA256(call.Prompt()); call.PromptSHA256 != got {
		t.Fatalf("PromptSHA256 = %s does not match its own canonical rendering %s", call.PromptSHA256, got)
	}
	if call.PromptSHA256 != hash {
		t.Fatalf("PromptSHA256 = %s, want pinned %s (the workload identity moved — if the prompt change is intended, repin the golden hash)", call.PromptSHA256, hash)
	}
}

// The canonical rendering must be deterministic call-over-call, and the four
// scenarios must mint pairwise-distinct identities over comparable problems.
func TestNormalizeProblemStableAndDistinct(t *testing.T) {
	a, err := NormalizeProblem(codegenGoldenProblem(), ScenarioCodeGeneration)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	b, err := NormalizeProblem(codegenGoldenProblem(), ScenarioCodeGeneration)
	if err != nil {
		t.Fatalf("NormalizeProblem: %v", err)
	}
	if a.PromptSHA256 != b.PromptSHA256 || a.Prompt() != b.Prompt() {
		t.Fatalf("normalization is not deterministic: %s vs %s", a.PromptSHA256, b.PromptSHA256)
	}

	hashes := map[string]string{
		codegenGoldenHash:       "codegen",
		codegenStdinGoldenHash:  "codegen-stdin",
		selfRepairGoldenHash:    "selfrepair",
		testOutputGoldenHash:    "testoutput",
		codeExecutionGoldenHash: "codeexecution",
	}
	if len(hashes) != 5 {
		t.Fatalf("golden hashes collide across scenarios: %v", hashes)
	}
}

func TestNormalizeProblemRefusals(t *testing.T) {
	t.Run("unknown scenario", func(t *testing.T) {
		_, err := NormalizeProblem(codegenGoldenProblem(), Scenario("wat"))
		if err == nil || !strings.Contains(err.Error(), "not supported") {
			t.Fatalf("want unsupported-scenario refusal, got %v", err)
		}
	})
	t.Run("cross-scenario identity", func(t *testing.T) {
		p := codegenGoldenProblem() // tagged codegeneration
		_, err := NormalizeProblem(p, ScenarioSelfRepair)
		if err == nil || !strings.Contains(err.Error(), "cross-scenario") {
			t.Fatalf("want cross-scenario refusal, got %v", err)
		}
	})
	t.Run("empty prompt", func(t *testing.T) {
		_, err := NormalizeProblem(Problem{QuestionID: "q"}, ScenarioCodeGeneration)
		if err == nil || !strings.Contains(err.Error(), "prompt is empty") {
			t.Fatalf("want empty-prompt refusal, got %v", err)
		}
	})
	t.Run("test-output without a public test", func(t *testing.T) {
		_, err := NormalizeProblem(Problem{QuestionID: "q", Prompt: "p"}, ScenarioTestOutputPrediction)
		if err == nil || !strings.Contains(err.Error(), "no public test case") {
			t.Fatalf("want no-input refusal, got %v", err)
		}
	})
	t.Run("code-execution without a public test", func(t *testing.T) {
		_, err := NormalizeProblem(Problem{QuestionID: "q", StarterCode: "print(1)"}, ScenarioCodeExecution)
		if err == nil || !strings.Contains(err.Error(), "no public test case") {
			t.Fatalf("want no-input refusal, got %v", err)
		}
	})
}
