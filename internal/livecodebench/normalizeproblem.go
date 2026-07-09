package livecodebench

import (
	"fmt"
	"strings"
)

// normalizeproblem.go — #2095 (epic #2085, A5): NormalizeProblem turns one suite
// Problem into the exact generation tool-call fak sends through the gateway —
// the system + user messages plus the stable identity of the rendered prompt —
// mirroring upstream lcb_runner format_prompt_generation semantics (problem
// statement + starter code + IO-format instructions, python-fenced answer).
// Assembly is per-scenario pluggable via generationAssemblers, because the four
// scenarios ask for different things: codegen wants a solution, self-repair
// wants a fix, test-output and code-execution want a predicted output. The
// rendered prompt hashes to PromptSHA256 — the workload-identity input the
// cross-arm SamePromptHash assertion compares (#2105/#3060) — so a golden test
// pinning the hash catches ANY silent prompt drift.

// GenerationLanguage is the language every normalized call targets. Upstream
// code_generation_lite is Python-only, so the language hint is threaded as a
// constant into the code fences rather than carried per problem.
const GenerationLanguage = "python"

// Roles of the normalized tool-call messages. These are this package's own
// strings (not the agent client's) so the foundation tier keeps zero upward
// imports; the CLI maps them 1:1 onto the wire shape it POSTs.
const (
	GenerationRoleSystem = "system"
	GenerationRoleUser   = "user"
)

// GenerationMessage is one chat message of the normalized tool-call.
type GenerationMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GenerationCall is the normalized prompt + tool-call for one (problem,
// scenario): the messages the gateway request carries, and the stable identity
// of the exact rendered prompt. Two calls are the same workload if and only if
// their PromptSHA256 match — this is the input the A/B arms must agree on for
// SamePromptHash to hold.
type GenerationCall struct {
	Scenario     Scenario            `json:"scenario"`
	QuestionID   string              `json:"question_id"`
	Messages     []GenerationMessage `json:"messages"`
	PromptSHA256 string              `json:"prompt_sha256"`
}

// Prompt is the canonical rendering the hash covers: every message as
// role + NUL + content, messages joined by double-NUL. NUL framing cannot
// collide with message text, so moving bytes between messages (or between role
// and content) always changes the identity.
func (c GenerationCall) Prompt() string {
	parts := make([]string, len(c.Messages))
	for i, m := range c.Messages {
		parts[i] = m.Role + "\x00" + m.Content
	}
	return strings.Join(parts, "\x00\x00")
}

// Per-scenario system prompts. Codegen mirrors the upstream
// format_prompt_generation system message; the other three frame the same
// expert role around what their scenario actually asks for.
const (
	codegenSystemPrompt = "You are an expert Python programmer. You will be given a question " +
		"(a problem specification) and will generate a correct Python program that matches the " +
		"specification and passes all tests."
	selfRepairSystemPrompt = "You are an expert Python programmer. You will be given a question, " +
		"a previous incorrect solution, and the failing-test feedback, and will generate a corrected " +
		"Python program that matches the specification and passes all tests."
	testOutputSystemPrompt = "You are an expert Python programmer. You will be given a question, " +
		"a program, and an input, and will predict the exact output the program produces for that input."
	codeExecutionSystemPrompt = "You are an expert Python programmer. You will be given a program " +
		"and an input, and will determine the exact output of executing the program on that input."
)

// generationAssembler renders one scenario's user message from a Problem. Each
// assembler refuses (clear error) when the problem lacks what its scenario
// needs — a call with nothing to ask is never rendered.
type generationAssembler func(Problem) (user string, err error)

// generationAssemblers is the pluggable per-scenario assembly table
// NormalizeProblem dispatches on. Adding a scenario means adding one row plus
// its golden test — nothing else changes.
var generationAssemblers = map[Scenario]struct {
	system   string
	assemble generationAssembler
}{
	ScenarioCodeGeneration:       {codegenSystemPrompt, assembleCodegenUser},
	ScenarioSelfRepair:           {selfRepairSystemPrompt, assembleCodegenUser},
	ScenarioTestOutputPrediction: {testOutputSystemPrompt, assembleTestOutputUser},
	ScenarioCodeExecution:        {codeExecutionSystemPrompt, assembleCodeExecutionUser},
}

// assembleCodegenUser renders the format_prompt_generation-shaped user message:
// the question, then the format instructions — the starter-code branch when the
// problem carries starter code (functional problems), the stdin/stdout branch
// otherwise — then the answer cue. Self-repair shares this base: upstream
// self-repair shows the same question + format, and the per-generation wrong
// attempt + failing-test feedback are threaded at run time by BuildRepairPrompt
// (they are run outputs, so they can never be part of the problem's identity).
func assembleCodegenUser(p Problem) (string, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return "", fmt.Errorf("livecodebench normalize: problem prompt is empty (nothing to ask)")
	}
	var b strings.Builder
	b.WriteString("### Question:\n")
	b.WriteString(strings.TrimSpace(p.Prompt))
	b.WriteString("\n\n### Format: ")
	if starter := strings.TrimSpace(p.StarterCode); starter != "" {
		b.WriteString("You will use the following starter code to write the solution to the problem " +
			"and enclose your code within delimiters.\n")
		b.WriteString("```" + GenerationLanguage + "\n")
		b.WriteString(starter)
		b.WriteString("\n```\n\n")
	} else {
		b.WriteString("Read the inputs from stdin, solve the problem, and write the answer to stdout " +
			"(do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n")
		b.WriteString("```" + GenerationLanguage + "\n")
		b.WriteString("# YOUR CODE HERE\n")
		b.WriteString("```\n\n")
	}
	b.WriteString("### Answer: (use the provided format with backticks)\n")
	return b.String(), nil
}

// assembleTestOutputUser renders the test-output-prediction user message via
// BuildTestOutputPrompt (so the normalized call and the per-input runtime
// request cannot drift), pinned to the problem's FIRST public test input — the
// canonical prediction target the problem's identity is taken over. The
// per-input fan-out over the remaining tests stays a runtime concern.
func assembleTestOutputUser(p Problem) (string, error) {
	if len(p.PublicTests) == 0 {
		return "", fmt.Errorf("livecodebench normalize: problem %q has no public test case (test-output prediction needs an input to predict for)", p.QuestionID)
	}
	return BuildTestOutputPrompt(p, p.PublicTests[0].Input)
}

// assembleCodeExecutionUser renders the code-execution user message via
// BuildCodeExecutionPrompt (same no-drift reuse), pinned to the first public
// test input and to direct (non-CoT) mode: chain-of-thought is a run-time knob
// (the upstream --cot_code_execution flag), and the normalized identity pins
// the default so a CoT run is a visibly different workload.
func assembleCodeExecutionUser(p Problem) (string, error) {
	if len(p.PublicTests) == 0 {
		return "", fmt.Errorf("livecodebench normalize: problem %q has no public test case (code execution needs an input to run on)", p.QuestionID)
	}
	return BuildCodeExecutionPrompt(p, p.PublicTests[0].Input, false)
}

// NormalizeProblem turns one suite Problem into the generation tool-call fak
// sends through the gateway for the given scenario: the per-scenario system +
// user messages, and the SHA-256 identity of the exact rendered prompt. It
// refuses an unknown scenario, and refuses a problem whose own Scenario tag
// disagrees with the requested one — silently normalizing a problem under the
// wrong scenario would mint a plausible-but-wrong workload identity.
func NormalizeProblem(p Problem, scenario Scenario) (GenerationCall, error) {
	entry, ok := generationAssemblers[scenario]
	if !ok {
		return GenerationCall{}, fmt.Errorf("livecodebench normalize: scenario %q is not supported", scenario)
	}
	if p.Scenario != "" && p.Scenario != scenario {
		return GenerationCall{}, fmt.Errorf("livecodebench normalize: problem %q is tagged scenario %q, not %q (refusing a cross-scenario identity)", p.QuestionID, p.Scenario, scenario)
	}
	user, err := entry.assemble(p)
	if err != nil {
		return GenerationCall{}, err
	}
	call := GenerationCall{
		Scenario:   scenario,
		QuestionID: p.QuestionID,
		Messages: []GenerationMessage{
			{Role: GenerationRoleSystem, Content: entry.system},
			{Role: GenerationRoleUser, Content: user},
		},
	}
	call.PromptSHA256 = promptSHA256(call.Prompt())
	return call, nil
}
