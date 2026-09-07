package issueorchestrator

import (
	"strings"
	"testing"
)

func TestOpencode_PromptLeafWorkerDirectExecution(t *testing.T) {
	iss := Issue{
		Number: 12028,
		Key:    "issue-12028",
		Title:  "fix(opencode): prevent subagent depth limit recursion failure",
		Lane:   "issueorchestrator",
		Paths:  []string{"internal/issueorchestrator/opencode.go"},
	}

	prompt := FormatOpencodePrompt(iss)

	// 1. Must NOT contain the broken instruction encouraging subagent task calls / parallel subagents.
	if strings.Contains(prompt, "Coordinate substantive work through parallel subagents") {
		t.Fatalf("FormatOpencodePrompt still contains broken subagent coordination instruction:\n%s", prompt)
	}
	if strings.Contains(prompt, "parallel subagents") {
		t.Fatalf("FormatOpencodePrompt must not encourage subagents via task tool:\n%s", prompt)
	}

	// 2. Must contain direct execution instructions for leaf workers.
	if !strings.Contains(prompt, "leaf worker") {
		t.Fatalf("FormatOpencodePrompt missing direct execution instruction for leaf worker:\n%s", prompt)
	}
	if !strings.Contains(prompt, "directly") {
		t.Fatalf("FormatOpencodePrompt missing instruction to execute work directly:\n%s", prompt)
	}
	if !strings.Contains(prompt, "package boundaries") {
		t.Fatalf("FormatOpencodePrompt missing package boundaries instruction:\n%s", prompt)
	}

	// 3. Must prohibit nested task tool delegation (preventing subagent depth limit recursion).
	if !strings.Contains(prompt, "prohibit") && !strings.Contains(prompt, "Prohibit") && !strings.Contains(prompt, "Do not call") {
		t.Fatalf("FormatOpencodePrompt missing prohibition of task tool delegation:\n%s", prompt)
	}
	if !strings.Contains(prompt, "'task'") && !strings.Contains(prompt, "task tool") {
		t.Fatalf("FormatOpencodePrompt missing reference to task tool prohibition:\n%s", prompt)
	}
	if !strings.Contains(prompt, "nested subagent") {
		t.Fatalf("FormatOpencodePrompt missing prohibition of nested subagent delegation:\n%s", prompt)
	}
}

func TestFormatOpencodePrompt_LeafWorkerDirectExecution(t *testing.T) {
	TestOpencode_PromptLeafWorkerDirectExecution(t)
}

func TestOpencode_WorkerGitSyncLanding(t *testing.T) {
	iss := Issue{
		Number: 12032,
		Key:    "issue-12032",
		Title:  "feat(issueorchestrator): worker process git sync landing by default",
		Lane:   "issueorchestrator",
		Paths:  []string{"internal/issueorchestrator/opencode.go"},
	}

	prompt := FormatOpencodePrompt(iss)

	// 1. Must instruct worker process to perform autonomous safe git landing by default.
	if !strings.Contains(prompt, "fak sync") {
		t.Fatalf("FormatOpencodePrompt missing 'fak sync' instruction for autonomous landing:\n%s", prompt)
	}
	if !strings.Contains(prompt, "fak commit --path") {
		t.Fatalf("FormatOpencodePrompt missing 'fak commit --path' explicit path staging instruction:\n%s", prompt)
	}
	if !strings.Contains(prompt, "fak sync push") {
		t.Fatalf("FormatOpencodePrompt missing 'fak sync push' instruction:\n%s", prompt)
	}

	// 2. Must mandate landing directly within worker process by default.
	if !strings.Contains(strings.ToLower(prompt), "landing") {
		t.Fatalf("FormatOpencodePrompt missing git landing instruction:\n%s", prompt)
	}
	if !strings.Contains(strings.ToLower(prompt), "worker process") {
		t.Fatalf("FormatOpencodePrompt missing explicit reference to worker process landing:\n%s", prompt)
	}
}

func TestFormatOpencodePrompt_WorkerGitSyncLanding(t *testing.T) {
	TestOpencode_WorkerGitSyncLanding(t)
}
