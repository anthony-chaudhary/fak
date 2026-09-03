package tb4bench

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReplayViewer(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-replay-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	transcriptPath := filepath.Join(tempDir, "transcript.jsonl")

	turn1 := TurnRecord{
		Turn:                1,
		ModelText:           "I will check the directory contents.",
		AdjudicationVerdict: "ALLOWED",
		ToolCalls: []ToolCallProposal{
			{ID: "c1", Name: "bash", Arguments: `{"cmd":"ls"}`},
		},
		ToolResults: []ToolExecutionResult{
			{ToolCallID: "c1", Tool: "bash", Stdout: "main.py\nREADME.md", ExitCode: 0},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
		DurationMs:       450,
	}

	turn2 := TurnRecord{
		Turn:                2,
		ModelText:           "Directory listed. TASK_COMPLETED",
		AdjudicationVerdict: "ALLOWED",
		PromptTokens:        140,
		CompletionTokens:    10,
		DurationMs:          220,
	}

	f, err := os.Create(transcriptPath)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	d1, _ := json.Marshal(turn1)
	d2, _ := json.Marshal(turn2)
	_, _ = f.Write(append(d1, '\n'))
	_, _ = f.Write(append(d2, '\n'))
	f.Close()

	// Load trajectory
	res, err := LoadTranscriptJSONL(transcriptPath)
	if err != nil {
		t.Fatalf("failed to load transcript: %v", err)
	}
	if len(res.Turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(res.Turns))
	}

	// Test headless render
	viewer := NewReplayViewer(false)
	var out bytes.Buffer
	viewer.RenderTrajectory(&out, res)

	rendered := out.String()
	if !strings.Contains(rendered, "TB4 Run Replay") {
		t.Errorf("missing replay header")
	}
	if !strings.Contains(rendered, "Turn 1 / 2") {
		t.Errorf("missing turn pagination: %s", rendered)
	}
	if !strings.Contains(rendered, "[ALLOWED] bash") {
		t.Errorf("missing tool call card: %s", rendered)
	}
	if !strings.Contains(rendered, "main.py") {
		t.Errorf("missing tool output: %s", rendered)
	}

	// Test comparative side-by-side render
	resB := &ArmExecutionResult{
		ArmID:      "opencode_llamacpp",
		TaskID:     res.TaskID,
		Status:     "COMPLETED",
		TotalTurns: 1,
		Turns: []TurnRecord{
			{Turn: 1, ModelText: "Quick fix"},
		},
	}
	var compOut bytes.Buffer
	viewer.RenderComparativeSideBySide(&compOut, res, resB)
	compRendered := compOut.String()

	if !strings.Contains(compRendered, "TB4 Comparative Replay") {
		t.Errorf("missing comparative header")
	}
	if !strings.Contains(compRendered, "Arm A (fak native)") {
		t.Errorf("missing Arm A column header")
	}
}
