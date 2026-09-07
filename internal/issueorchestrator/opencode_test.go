package issueorchestrator

import (
	"strings"
	"testing"
)

func TestFormatOpencodePrompt_LeafWorkerDirectExecution(t *testing.T) {
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

func TestFormatOpencodePrompt_WorkerGitSyncLanding(t *testing.T) {
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

func TestBuildOpencodeChat_SubagentDepth(t *testing.T) {
	iss := Issue{
		Number: 12030,
		Key:    "issue-12030",
		Title:  "feat(issueorchestrator): add subagent depth configuration to OpencodeChatOptions for multi-tier sessions",
		Lane:   "issueorchestrator",
	}

	// Case 1: SubagentDepth > 0 (e.g. 2)
	opts := OpencodeChatOptions{
		SubagentDepth: 2,
	}
	chat := BuildOpencodeChat(iss, opts)
	foundFlag := false
	for i, arg := range chat.Command {
		if arg == "--subagent-depth" {
			foundFlag = true
			if i+1 >= len(chat.Command) || chat.Command[i+1] != "2" {
				t.Fatalf("expected '--subagent-depth' followed by '2', got command: %v", chat.Command)
			}
			break
		}
	}
	if !foundFlag {
		t.Fatalf("expected command to contain '--subagent-depth', got: %v", chat.Command)
	}

	// Case 2: SubagentDepth == 0
	optsZero := OpencodeChatOptions{
		SubagentDepth: 0,
	}
	chatZero := BuildOpencodeChat(iss, optsZero)
	for _, arg := range chatZero.Command {
		if arg == "--subagent-depth" {
			t.Fatalf("expected command to omit '--subagent-depth' when SubagentDepth is 0, got: %v", chatZero.Command)
		}
	}
}

func TestFormatOpencodePrompt_HaloHardwareValidation(t *testing.T) {
	// 1. Relevant issue: lane amdgpu
	relevantLaneIssue := Issue{
		Number: 12033,
		Key:    "issue-12033",
		Title:  "feat(amdgpu): optimize kernel dispatch",
		Lane:   "amdgpu",
		Paths:  []string{"internal/amdgpu/dispatch.go"},
	}
	promptLane := FormatOpencodePrompt(relevantLaneIssue)
	if !strings.Contains(promptLane, "strix1") {
		t.Fatalf("relevant lane issue prompt missing leased Halo hardware guidance ('strix1'):\n%s", promptLane)
	}
	if !strings.Contains(promptLane, "early low-cost probe") {
		t.Fatalf("relevant lane issue prompt missing early low-cost probe requirement:\n%s", promptLane)
	}
	if !strings.Contains(promptLane, "fak-dev amd-strix-probe") {
		t.Fatalf("relevant lane issue prompt missing 'fak-dev amd-strix-probe':\n%s", promptLane)
	}
	if !strings.Contains(promptLane, "source-bound physical execution evidence") {
		t.Fatalf("relevant lane issue prompt missing source-bound physical execution evidence requirement:\n%s", promptLane)
	}
	if !strings.Contains(promptLane, "pending hardware status") {
		t.Fatalf("relevant lane issue prompt missing pending hardware status instruction:\n%s", promptLane)
	}

	// 2. Relevant issue: title mentioning Strix Halo
	relevantTitleIssue := Issue{
		Number: 12034,
		Key:    "issue-12034",
		Title:  "bench(engine): evaluate Strix Halo GEMM throughput",
		Lane:   "engine",
		Paths:  []string{"internal/engine/gemm.go"},
	}
	promptTitle := FormatOpencodePrompt(relevantTitleIssue)
	if !strings.Contains(promptTitle, "strix1") {
		t.Fatalf("relevant title issue prompt missing leased Halo hardware guidance ('strix1'):\n%s", promptTitle)
	}
	if !strings.Contains(promptTitle, "early low-cost probe") {
		t.Fatalf("relevant title issue prompt missing early low-cost probe requirement:\n%s", promptTitle)
	}
	if !strings.Contains(promptTitle, "source-bound physical execution evidence") {
		t.Fatalf("relevant title issue prompt missing source-bound physical execution evidence requirement:\n%s", promptTitle)
	}
	if !strings.Contains(promptTitle, "pending hardware status") {
		t.Fatalf("relevant title issue prompt missing pending hardware status instruction:\n%s", promptTitle)
	}

	// 3. Unrelated issue (e.g. lane issueorchestrator, docs, agent)
	unrelatedIssue := Issue{
		Number: 12028,
		Key:    "issue-12028",
		Title:  "fix(opencode): prevent subagent depth limit recursion failure",
		Lane:   "issueorchestrator",
		Paths:  []string{"internal/issueorchestrator/opencode.go"},
	}
	promptUnrelated := FormatOpencodePrompt(unrelatedIssue)
	if strings.Contains(promptUnrelated, "Leased Halo Hardware Validation") {
		t.Fatalf("unrelated issue prompt must NOT contain Halo Hardware Validation:\n%s", promptUnrelated)
	}
	if strings.Contains(promptUnrelated, "strix1") {
		t.Fatalf("unrelated issue prompt must NOT contain 'strix1':\n%s", promptUnrelated)
	}
	if strings.Contains(promptUnrelated, "amd-strix-probe") {
		t.Fatalf("unrelated issue prompt must NOT contain 'amd-strix-probe':\n%s", promptUnrelated)
	}
	if strings.Contains(promptUnrelated, "pending hardware status") {
		t.Fatalf("unrelated issue prompt must NOT contain 'pending hardware status':\n%s", promptUnrelated)
	}
}

func TestIsHaloHardwareRelevant(t *testing.T) {
	// Lanes: amdgpu, compute, modelperfobs, nativeperf
	lanes := []string{"amdgpu", "compute", "modelperfobs", "nativeperf", "AMDGPU", " Compute "}
	for _, lane := range lanes {
		iss := Issue{Lane: lane, Title: "Generic task", Paths: []string{"internal/pkg/foo.go"}}
		if !isHaloHardwareRelevant(iss) {
			t.Errorf("expected isHaloHardwareRelevant to be true for lane %q", lane)
		}
	}

	// Paths: amdgpu, compute, strix, halo, gfx115, vulkan
	pathKeywords := []string{"amdgpu", "compute", "strix", "halo", "gfx115", "vulkan"}
	for _, kw := range pathKeywords {
		iss := Issue{Lane: "other", Title: "Generic task", Paths: []string{"internal/sub/" + kw + "/file.go"}}
		if !isHaloHardwareRelevant(iss) {
			t.Errorf("expected isHaloHardwareRelevant to be true for path with keyword %q", kw)
		}
	}

	// Titles: strix, halo, gfx115, amdgpu, rocm, vulkan, rdna (case-insensitive)
	titleKeywords := []string{"strix", "halo", "gfx115", "amdgpu", "rocm", "vulkan", "rdna", "Strix", "HALO", "GFX115", "ROCm", "RDNA3"}
	for _, kw := range titleKeywords {
		iss := Issue{Lane: "other", Title: "Implement " + kw + " support", Paths: []string{"internal/other/foo.go"}}
		if !isHaloHardwareRelevant(iss) {
			t.Errorf("expected isHaloHardwareRelevant to be true for title with keyword %q", kw)
		}
	}

	// Unrelated issues
	unrelated := []Issue{
		{Lane: "issueorchestrator", Title: "fix(opencode): clean prompt", Paths: []string{"internal/issueorchestrator/opencode.go"}},
		{Lane: "docs", Title: "docs: update readme", Paths: []string{"docs/readme.md"}},
		{Lane: "agent", Title: "feat(agent): support streaming", Paths: []string{"internal/agent/stream.go"}},
	}
	for _, iss := range unrelated {
		if isHaloHardwareRelevant(iss) {
			t.Errorf("expected isHaloHardwareRelevant to be false for unrelated issue: %+v", iss)
		}
	}
}



