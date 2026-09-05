package tb4bench

import (
	"bytes"
	"strings"
	"testing"
)

func TestTUI_RunInteractive_Keybindings(t *testing.T) {
	resA := &ArmExecutionResult{
		ArmID:      "fak_native",
		TaskID:     "task-repro-01",
		Status:     "COMPLETED",
		TotalTurns: 3,
		VDSOHits:   4,
		Turns: []TurnRecord{
			{
				Turn:                1,
				ModelText:           "Inspecting files and preparing patch.",
				AdjudicationVerdict: "ALLOWED",
				ToolCalls: []ToolCallProposal{
					{ID: "c1", Name: "edit_file", Arguments: `{"path":"main.go","newString":"+ fix"}`},
				},
				ToolResults: []ToolExecutionResult{
					{ToolCallID: "c1", Tool: "edit_file", Stdout: "--- main.go\n+++ main.go\n@@ -1 +1 @@\n+ fix\n- bug", ExitCode: 0},
				},
				PromptTokens:     100,
				CompletionTokens: 25,
				DurationMs:       350,
			},
			{
				Turn:                2,
				ModelText:           "Running tests to verify fix.",
				AdjudicationVerdict: "ALLOWED",
				ToolCalls: []ToolCallProposal{
					{ID: "c2", Name: "bash", Arguments: `{"cmd":"go test ./..."}`},
				},
				ToolResults: []ToolExecutionResult{
					{ToolCallID: "c2", Tool: "bash", Stdout: "PASS\nok", ExitCode: 0},
				},
				PromptTokens:     140,
				CompletionTokens: 20,
				DurationMs:       420,
			},
			{
				Turn:                3,
				ModelText:           "Done. TASK_COMPLETED",
				AdjudicationVerdict: "ALLOWED",
				PromptTokens:        160,
				CompletionTokens:    10,
				DurationMs:          120,
			},
		},
	}

	resB := &ArmExecutionResult{
		ArmID:      "opencode_ref",
		TaskID:     "task-repro-01",
		Status:     "COMPLETED",
		TotalTurns: 2,
		Turns: []TurnRecord{
			{
				Turn:                1,
				ModelText:           "Reading directory.",
				AdjudicationVerdict: "ALLOWED",
				ToolCalls: []ToolCallProposal{
					{ID: "cb1", Name: "bash", Arguments: `{"cmd":"ls -la"}`},
				},
				ToolResults: []ToolExecutionResult{
					{ToolCallID: "cb1", Tool: "bash", Stdout: "total 8\n-rw-r--r-- main.go", ExitCode: 0},
				},
				PromptTokens:     100,
				CompletionTokens: 15,
				DurationMs:       210,
			},
			{
				Turn:                2,
				ModelText:           "Done. TASK_COMPLETED",
				AdjudicationVerdict: "ALLOWED",
				PromptTokens:        150,
				CompletionTokens:    10,
				DurationMs:          190,
			},
		},
	}

	ctrl := NewTUIController(resA, resB)
	var out bytes.Buffer

	// Key sequence requirement: "j\tk\nq"
	// 'j': step turn forward
	// '\t': switch arm
	// 'k': step turn back
	// '\n': toggle tool expansion
	// 'q': quit cleanly
	input := strings.NewReader("j\tk\nq")

	err := ctrl.RunInteractive(input, &out, resA, resB)
	if err != nil {
		t.Fatalf("RunInteractive returned unexpected error: %v", err)
	}

	if len(ctrl.StateHistory) < 5 {
		t.Fatalf("expected at least 5 state transitions recorded, got %d", len(ctrl.StateHistory))
	}

	// 1. Initial state
	s0 := ctrl.StateHistory[0]
	if s0.CurrentTurn != 1 {
		t.Errorf("initial state: expected turn 1, got %d", s0.CurrentTurn)
	}
	if s0.ActiveArm != 0 {
		t.Errorf("initial state: expected active arm 0 (Arm A), got %d", s0.ActiveArm)
	}
	if s0.Expanded {
		t.Errorf("initial state: expected expanded false, got true")
	}

	// 2. Step forward on 'j'
	s1 := ctrl.StateHistory[1]
	if s1.CurrentTurn != 2 {
		t.Errorf("after 'j': expected turn to advance to 2, got %d", s1.CurrentTurn)
	}

	// 3. Switch arm on Tab ('\t')
	s2 := ctrl.StateHistory[2]
	if s2.ActiveArm != 1 {
		t.Errorf("after Tab: expected active arm to switch to 1 (Arm B), got %d", s2.ActiveArm)
	}
	if s2.CurrentTurn != 2 {
		t.Errorf("after Tab: expected turn to remain 2, got %d", s2.CurrentTurn)
	}

	// 4. Step backward on 'k'
	s3 := ctrl.StateHistory[3]
	if s3.CurrentTurn != 1 {
		t.Errorf("after 'k': expected turn to step back to 1, got %d", s3.CurrentTurn)
	}

	// 5. Toggle tool expansion on Enter ('\n')
	s4 := ctrl.StateHistory[4]
	if !s4.Expanded {
		t.Errorf("after Enter: expected expanded to become true, got false")
	}

	// 6. Output assertions
	rendered := out.String()
	if !strings.Contains(rendered, "TB4 Comparative Replay") {
		t.Errorf("missing comparative replay banner in rendered output")
	}
	if !strings.Contains(rendered, "Arm A: fak (In-Kernel)") {
		t.Errorf("missing Arm A label in output")
	}
	if !strings.Contains(rendered, "Arm B: OpenCode (Reference)") {
		t.Errorf("missing Arm B label in output")
	}
	if !strings.Contains(rendered, "Controls:") {
		t.Errorf("missing controls footer in output")
	}
}

func TestTUI_RunInteractive_ArrowKeysAndCarriageReturn(t *testing.T) {
	res := &ArmExecutionResult{
		ArmID:      "fak_native",
		TaskID:     "task-arrow-test",
		Status:     "COMPLETED",
		TotalTurns: 3,
		Turns: []TurnRecord{
			{Turn: 1, ModelText: "Turn 1 thoughts", PromptTokens: 100, CompletionTokens: 10},
			{Turn: 2, ModelText: "Turn 2 thoughts", PromptTokens: 120, CompletionTokens: 15},
			{Turn: 3, ModelText: "Turn 3 thoughts", PromptTokens: 140, CompletionTokens: 20},
		},
	}

	// \x1b[B = Down (Next), \x1b[A = Up (Prev), \r\n = Enter (Toggle), \x1b = Esc (Quit)
	input := strings.NewReader("\x1b[B\x1b[A\r\n\x1b")
	var out bytes.Buffer

	ctrl := NewTUIController(res, nil)
	err := ctrl.RunInteractive(input, &out, res, nil)
	if err != nil {
		t.Fatalf("RunInteractive failed: %v", err)
	}

	if len(ctrl.StateHistory) < 4 {
		t.Fatalf("expected at least 4 state transitions, got %d", len(ctrl.StateHistory))
	}

	// Initial
	if ctrl.StateHistory[0].CurrentTurn != 1 {
		t.Errorf("expected initial turn 1, got %d", ctrl.StateHistory[0].CurrentTurn)
	}
	// After Down arrow
	if ctrl.StateHistory[1].CurrentTurn != 2 {
		t.Errorf("after Down arrow: expected turn 2, got %d", ctrl.StateHistory[1].CurrentTurn)
	}
	// After Up arrow
	if ctrl.StateHistory[2].CurrentTurn != 1 {
		t.Errorf("after Up arrow: expected turn 1, got %d", ctrl.StateHistory[2].CurrentTurn)
	}
	// After \r\n Enter
	if !ctrl.StateHistory[3].Expanded {
		t.Errorf("after \\r\\n Enter: expected expanded true, got false")
	}
}

func TestTUI_SingleArmTrajectoryRender(t *testing.T) {
	res := &ArmExecutionResult{
		ArmID:      "fak_native",
		TaskID:     "task-single-arm",
		Status:     "COMPLETED",
		TotalTurns: 2,
		VDSOHits:   12,
		Turns: []TurnRecord{
			{
				Turn:                1,
				ModelText:           "Writing configuration file.",
				AdjudicationVerdict: "ALLOWED",
				ToolCalls: []ToolCallProposal{
					{ID: "w1", Name: "write_file", Arguments: `{"path":"config.json","content":"{\"debug\":true}"}`},
				},
				ToolResults: []ToolExecutionResult{
					{ToolCallID: "w1", Tool: "write_file", Stdout: "File written successfully", ExitCode: 0},
				},
				PromptTokens:     150,
				CompletionTokens: 30,
				DurationMs:       280,
			},
			{
				Turn:                2,
				ModelText:           "Verification complete. TASK_COMPLETED",
				AdjudicationVerdict: "ALLOWED",
				PromptTokens:        210,
				CompletionTokens:    12,
				DurationMs:          160,
			},
		},
	}

	state := &TUIState{
		ResA:        res,
		CurrentTurn: 1,
		ActiveArm:   0,
		Expanded:    false,
		Comparative: false,
	}

	var out bytes.Buffer
	err := DrawTurnFrame(&out, state)
	if err != nil {
		t.Fatalf("DrawTurnFrame failed: %v", err)
	}

	frame := out.String()

	// Assertions for required card elements
	// 1. Turn index, role, status
	if !strings.Contains(frame, "Turn 1 / 2") {
		t.Errorf("missing turn index in frame: %s", frame)
	}
	if !strings.Contains(frame, "Role: assistant") {
		t.Errorf("missing role in frame: %s", frame)
	}
	if !strings.Contains(frame, "Status: ALLOWED") {
		t.Errorf("missing status in frame: %s", frame)
	}

	// 2. Real-time token breakdown
	if !strings.Contains(frame, "Tokens: Prompt Cold:") || !strings.Contains(frame, "Prompt Cached:") {
		t.Errorf("missing prompt cold/cached token breakdown: %s", frame)
	}
	if !strings.Contains(frame, "Tokens: Completion:") || !strings.Contains(frame, "Thoughts:") {
		t.Errorf("missing completion/thoughts token breakdown: %s", frame)
	}

	// 3. vDSO / cache hit indicators
	if !strings.Contains(frame, "[vDSO: HIT") {
		t.Errorf("missing vDSO hit indicator: %s", frame)
	}

	// 4. Tool calls and diff cues
	if !strings.Contains(frame, "write_file") {
		t.Errorf("missing tool call name: %s", frame)
	}
	if !strings.Contains(frame, "[MOD/DIFF]") {
		t.Errorf("missing file modification diff cue: %s", frame)
	}

	// 5. Output collapsed cue
	if !strings.Contains(frame, "Output collapsed - press Enter to expand") {
		t.Errorf("missing collapsed output indicator: %s", frame)
	}
}

func TestTUI_ComparativeDualArmFrameGeneration(t *testing.T) {
	resA := &ArmExecutionResult{
		ArmID:      "fak_native",
		TaskID:     "task-compare-dual",
		Status:     "COMPLETED",
		TotalTurns: 2,
		VDSOHits:   8,
		Turns: []TurnRecord{
			{
				Turn:                1,
				ModelText:           "Refactoring logic in parser.",
				AdjudicationVerdict: "ALLOWED",
				ToolCalls: []ToolCallProposal{
					{ID: "e1", Name: "edit_file", Arguments: `{"path":"parser.go","newString":"return token"}`},
				},
				ToolResults: []ToolExecutionResult{
					{
						ToolCallID: "e1",
						Tool:       "edit_file",
						Stdout:     "@@ -10,3 +10,3 @@\n- panic(\"unimplemented\")\n+ return token\n  return nil",
						ExitCode:   0,
					},
				},
				PromptTokens:     180,
				CompletionTokens: 35,
			},
			{
				Turn:                2,
				ModelText:           "TASK_COMPLETED",
				AdjudicationVerdict: "ALLOWED",
				PromptTokens:        240,
				CompletionTokens:    10,
			},
		},
	}

	resB := &ArmExecutionResult{
		ArmID:      "opencode_ref",
		TaskID:     "task-compare-dual",
		Status:     "COMPLETED",
		TotalTurns: 1,
		Turns: []TurnRecord{
			{
				Turn:                1,
				ModelText:           "Checking directory.",
				AdjudicationVerdict: "ALLOWED",
				PromptTokens:        120,
				CompletionTokens:    15,
			},
		},
	}

	state := &TUIState{
		ResA:        resA,
		ResB:        resB,
		CurrentTurn: 1,
		ActiveArm:   0,
		Expanded:    true, // Expanded mode: renders stdout with diff highlighting
		Comparative: true,
	}

	var out bytes.Buffer
	err := DrawTurnFrame(&out, state)
	if err != nil {
		t.Fatalf("DrawTurnFrame failed: %v", err)
	}

	frame := out.String()

	// Comparative header check
	if !strings.Contains(frame, "TB4 Comparative Replay: Task task-compare-dual") {
		t.Errorf("missing comparative header: %s", frame)
	}

	// Check dual column separators
	if !strings.Contains(frame, "│") {
		t.Errorf("missing column border separator │")
	}

	// Check Arm A active indicator
	if !strings.Contains(frame, "Arm A: fak (In-Kernel) [ACTIVE]") {
		t.Errorf("missing Arm A active indicator: %s", frame)
	}

	// Check diff cues in expanded output
	// Added line "+" in green (\033[32m)
	if !strings.Contains(frame, "\033[32m+ return token\033[0m") {
		t.Errorf("missing green diff addition highlight for '+ return token': %s", frame)
	}
	// Deleted line "-" in red (\033[31m)
	if !strings.Contains(frame, "\033[31m- panic(\"unimplemented\")\033[0m") {
		t.Errorf("missing red diff deletion highlight for '- panic': %s", frame)
	}
	// Header "@@" in cyan (\033[36m)
	if !strings.Contains(frame, "\033[36m@@ -10,3 +10,3 @@\033[0m") {
		t.Errorf("missing cyan hunk header highlight: %s", frame)
	}

	// Toggle active arm to Arm B
	state.ToggleArm()
	if state.ActiveArm != 1 {
		t.Fatalf("expected ActiveArm to be 1, got %d", state.ActiveArm)
	}

	out.Reset()
	err = DrawTurnFrame(&out, state)
	if err != nil {
		t.Fatalf("DrawTurnFrame failed after arm switch: %v", err)
	}

	frameB := out.String()
	if !strings.Contains(frameB, "Arm B: OpenCode (Reference) [ACTIVE]") {
		t.Errorf("missing Arm B active indicator after toggle: %s", frameB)
	}

	// Advance turn to turn 2 (where Arm B is finished)
	state.NextTurn()
	out.Reset()
	err = DrawTurnFrame(&out, state)
	if err != nil {
		t.Fatalf("DrawTurnFrame failed at turn 2: %v", err)
	}

	frameTurn2 := out.String()
	if !strings.Contains(frameTurn2, "Finished at turn 1") {
		t.Errorf("expected Arm B finished indication at turn 2: %s", frameTurn2)
	}
}

func TestTUIState_BoundaryConditions(t *testing.T) {
	state := &TUIState{
		CurrentTurn: 1,
	}

	// PrevTurn at 1 should not go below 1
	state.PrevTurn()
	if state.CurrentTurn != 1 {
		t.Errorf("expected turn to remain 1 on underflow, got %d", state.CurrentTurn)
	}

	// NextTurn with 0 turns should not exceed 1
	state.NextTurn()
	if state.CurrentTurn != 1 {
		t.Errorf("expected turn to remain 1 with empty results, got %d", state.CurrentTurn)
	}

	// Nil state error handling
	err := DrawTurnFrame(&bytes.Buffer{}, nil)
	if err == nil {
		t.Errorf("expected error when passing nil state to DrawTurnFrame")
	}
}
