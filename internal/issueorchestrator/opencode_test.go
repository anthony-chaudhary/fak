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

func TestOpencode_MandatoryLandingExitGate(t *testing.T) {
	iss := Issue{
		Number: 12080,
		Key:    "issue-12080",
		Title:  "feat(opencode): enforce landing by default exit gate",
		Lane:   "issueorchestrator",
		Paths:  []string{"internal/issueorchestrator/opencode.go"},
	}

	prompt := FormatOpencodePrompt(iss)
	if !strings.Contains(prompt, "Mandatory 4-Phase Delivery and Landing Pipeline") {
		t.Fatalf("FormatOpencodePrompt missing mandatory 4-phase pipeline:\n%s", prompt)
	}
	if !strings.Contains(prompt, "MANDATORY EXIT GATE") {
		t.Fatalf("FormatOpencodePrompt missing mandatory exit gate declaration:\n%s", prompt)
	}
	if !strings.Contains(prompt, "fak sync reconcile --apply") {
		t.Fatalf("FormatOpencodePrompt missing divergence recovery instruction:\n%s", prompt)
	}
}

func TestBuildOpencodeChat_Flags(t *testing.T) {
	iss := Issue{
		Number: 12080,
		Title:  "Test issue",
		Lane:   "issueorchestrator",
	}

	chat := BuildOpencodeChat(iss, OpencodeChatOptions{
		AutoApprove: true,
		Agent:       "worker",
	})

	foundAgentWorker := false
	foundSkipPermissions := false
	for i, arg := range chat.Command {
		if arg == "--agent" && i+1 < len(chat.Command) && chat.Command[i+1] == "worker" {
			foundAgentWorker = true
		}
		if arg == "--dangerously-skip-permissions" {
			foundSkipPermissions = true
		}
	}

	if !foundAgentWorker {
		t.Fatalf("BuildOpencodeChat missing --agent worker: %v", chat.Command)
	}
	if !foundSkipPermissions {
		t.Fatalf("BuildOpencodeChat missing --dangerously-skip-permissions: %v", chat.Command)
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

func TestBuildOpencodeChat_DefaultAgentWorker(t *testing.T) {
	iss := Issue{
		Number: 12255,
		Title:  "Test default agent",
		Lane:   "issueorchestrator",
	}

	// When opts.Agent is empty, it must default to "worker".
	chat := BuildOpencodeChat(iss, OpencodeChatOptions{})

	foundAgent := false
	for i, arg := range chat.Command {
		if arg == "--agent" {
			foundAgent = true
			if i+1 >= len(chat.Command) || chat.Command[i+1] != "worker" {
				t.Fatalf("expected '--agent' followed by 'worker', got: %v", chat.Command)
			}
			break
		}
	}
	if !foundAgent {
		t.Fatalf("expected command to contain '--agent worker', got: %v", chat.Command)
	}
}

func TestBuildOpencodeChat_ExplicitAgentOverride(t *testing.T) {
	iss := Issue{
		Number: 12255,
		Title:  "Test explicit agent override",
		Lane:   "issueorchestrator",
	}

	// When opts.Agent is explicitly provided, it must preserve the override.
	chat := BuildOpencodeChat(iss, OpencodeChatOptions{
		Agent: "researcher",
	})

	foundAgent := false
	for i, arg := range chat.Command {
		if arg == "--agent" {
			foundAgent = true
			if i+1 >= len(chat.Command) || chat.Command[i+1] != "researcher" {
				t.Fatalf("expected '--agent' followed by 'researcher', got: %v", chat.Command)
			}
			break
		}
	}
	if !foundAgent {
		t.Fatalf("expected command to contain '--agent researcher', got: %v", chat.Command)
	}

	for _, arg := range chat.Command {
		if arg == "worker" {
			t.Fatalf("command should not contain default 'worker' when overridden, got: %v", chat.Command)
		}
	}
}

func TestFormatOpencodePrompt_ShiftLeftSoftwareFirst(t *testing.T) {
	iss := Issue{
		Number: 12253,
		Key:    "issue-12253",
		Title:  "feat(amdgpu): shift-left deterministic software witness before hardware validation",
		Lane:   "amdgpu",
		Paths:  []string{"internal/amdgpu/device.go"},
	}

	prompt := FormatOpencodePrompt(iss)

	// 1. Verifies that for a hardware-relevant issue, the software witness / landing pipeline index appears before Physical hardware validation: index in prompt.
	idxSoftwareWitness := strings.Index(prompt, "deterministic software witness")
	idxPipeline := strings.Index(prompt, "Mandatory 4-Phase Delivery and Landing Pipeline")
	idxHw := strings.Index(prompt, "Physical hardware validation:")

	if idxSoftwareWitness == -1 {
		t.Fatalf("prompt missing deterministic software witness instruction:\n%s", prompt)
	}
	if idxPipeline == -1 {
		t.Fatalf("prompt missing Mandatory 4-Phase Delivery and Landing Pipeline:\n%s", prompt)
	}
	if idxHw == -1 {
		t.Fatalf("prompt missing Physical hardware validation for hardware-relevant issue:\n%s", prompt)
	}
	if idxSoftwareWitness >= idxHw {
		t.Fatalf("expected software witness instruction (idx %d) before hardware validation (idx %d)", idxSoftwareWitness, idxHw)
	}
	if idxPipeline >= idxHw {
		t.Fatalf("expected 4-phase landing pipeline (idx %d) before hardware validation (idx %d)", idxPipeline, idxHw)
	}

	// 2. Verifies prompt contains "PENDING_HARDWARE" and "green software increment".
	if !strings.Contains(prompt, "PENDING_HARDWARE") {
		t.Fatalf("prompt missing 'PENDING_HARDWARE':\n%s", prompt)
	}
	if !strings.Contains(prompt, "green software increment") {
		t.Fatalf("prompt missing 'green software increment':\n%s", prompt)
	}

	// 3. Verifies prompt contains coordination edges distinction ("alignment assumptions", "typed pickup block").
	if !strings.Contains(prompt, "alignment assumptions") {
		t.Fatalf("prompt missing 'alignment assumptions':\n%s", prompt)
	}
	if !strings.Contains(prompt, "typed pickup block") {
		t.Fatalf("prompt missing 'typed pickup block':\n%s", prompt)
	}

	// 4. Verifies prompt contains collision narrowing ("disjoint slice", "BLOCKED").
	if !strings.Contains(prompt, "disjoint slice") {
		t.Fatalf("prompt missing 'disjoint slice':\n%s", prompt)
	}
	if !strings.Contains(prompt, "BLOCKED") {
		t.Fatalf("prompt missing 'BLOCKED':\n%s", prompt)
	}

	// 5. Verifies software-first sequence explicitly stated.
	seq := "failing deterministic contract -> smallest implementation -> focused validation -> guarded software landing -> physical qualification"
	if !strings.Contains(prompt, seq) {
		t.Fatalf("prompt missing software-first sequence %q:\n%s", seq, prompt)
	}
}
