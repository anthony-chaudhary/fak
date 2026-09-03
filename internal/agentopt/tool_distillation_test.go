package agentopt

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestToolDistillationPipeline(t *testing.T) {
	grepSchema := ToolSchema{
		Name:        "Grep",
		Description: "Search file contents by regex pattern",
		Properties: map[string]PropertySchema{
			"pattern": {Type: TypeString},
			"path":    {Type: TypeString},
		},
		Required:             []string{"pattern"},
		AdditionalProperties: false,
	}

	readSchema := ToolSchema{
		Name:        "Read",
		Description: "Read file contents",
		Properties: map[string]PropertySchema{
			"filePath": {Type: TypeString},
			"offset":   {Type: TypeInteger},
			"limit":    {Type: TypeInteger},
		},
		Required:             []string{"filePath"},
		AdditionalProperties: false,
	}

	validator := NewSchemaValidator(grepSchema, readSchema)

	t.Run("clean_demonstration_filtering", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		cleanTraj := FrontierTrajectory{
			ID:           "traj-001",
			SystemPrompt: "You are an autonomous coding agent.",
			UserQuery:    "Find and inspect auth token validation in src/auth.go",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "Search for token validation symbol in src/auth.go",
					ToolCall: ToolCall{
						Name:     "Grep",
						Args:     map[string]any{"pattern": "validateToken", "path": "src/auth.go"},
						ReadOnly: true,
					},
					ToolResult: "src/auth.go:42: func validateToken(token string) bool",
				},
				{
					StepIndex: 2,
					Thought:   "Read the function definition in src/auth.go",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "src/auth.go", "offset": 40.0, "limit": 10.0},
						ReadOnly: true,
					},
					ToolResult: "func validateToken(token string) bool {\n  return len(token) > 0\n}",
				},
			},
			FinalResponse: "The validateToken function is located at line 42 of src/auth.go.",
			Success:       true,
		}

		res := pipeline.FilterTrajectory(cleanTraj)
		if !res.Admitted {
			t.Fatalf("expected clean trajectory to be admitted, got rejection: %s", res.Reason)
		}

		example, err := pipeline.ProcessTrajectory(cleanTraj)
		if err != nil {
			t.Fatalf("unexpected error processing trajectory: %v", err)
		}

		if example.ID != "traj-001" {
			t.Errorf("expected ID 'traj-001', got %q", example.ID)
		}
		if example.SystemPrompt != cleanTraj.SystemPrompt {
			t.Errorf("expected system prompt %q, got %q", cleanTraj.SystemPrompt, example.SystemPrompt)
		}
		if example.UserQuery != cleanTraj.UserQuery {
			t.Errorf("expected user query %q, got %q", cleanTraj.UserQuery, example.UserQuery)
		}
		if len(example.ToolCalls) != 2 {
			t.Fatalf("expected 2 tool calls, got %d", len(example.ToolCalls))
		}
		if example.ToolCalls[0].Name != "Grep" || example.ToolCalls[1].Name != "Read" {
			t.Errorf("tool calls mismatch: %+v", example.ToolCalls)
		}
		if !strings.Contains(example.Thought, "Search for token validation") {
			t.Errorf("thought missing step 1 content: %q", example.Thought)
		}
		if example.FinalResponse != cleanTraj.FinalResponse {
			t.Errorf("expected final response %q, got %q", cleanTraj.FinalResponse, example.FinalResponse)
		}
	})

	t.Run("discard_failed_tool_calls", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		// Trajectory with step.Error populated
		trajWithError := FrontierTrajectory{
			ID:           "traj-err",
			SystemPrompt: "System",
			UserQuery:    "Read file",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "Read missing file",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "nonexistent.go"},
						ReadOnly: true,
					},
					Error: "file not found: nonexistent.go",
				},
			},
			FinalResponse: "Done",
			Success:       true,
		}

		res := pipeline.FilterTrajectory(trajWithError)
		if res.Admitted {
			t.Fatal("expected trajectory with failed tool call to be rejected")
		}
		if !strings.Contains(res.Reason, "failed tool call") {
			t.Errorf("expected failure reason to mention failed tool call, got %q", res.Reason)
		}

		// Trajectory with error prefix in ToolResult
		trajWithErrorOutput := FrontierTrajectory{
			ID:           "traj-err-out",
			SystemPrompt: "System",
			UserQuery:    "Read file",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "Attempt read",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "bad.go"},
						ReadOnly: true,
					},
					ToolResult: "FATAL: disk read failure",
				},
			},
			FinalResponse: "Done",
			Success:       true,
		}

		resOut := pipeline.FilterTrajectory(trajWithErrorOutput)
		if resOut.Admitted {
			t.Fatal("expected trajectory with fatal tool result to be rejected")
		}
	})

	t.Run("discard_loops_and_repeated_reads", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		// Repeated read-only call with identical arguments
		trajRepeatedRead := FrontierTrajectory{
			ID:           "traj-loop-read",
			SystemPrompt: "System",
			UserQuery:    "Read twice",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "First read",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "src/main.go"},
						ReadOnly: true,
					},
					ToolResult: "package main",
				},
				{
					StepIndex: 2,
					Thought:   "Read again identically",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "src/main.go"},
						ReadOnly: true,
					},
					ToolResult: "package main",
				},
			},
			FinalResponse: "Done",
			Success:       true,
		}

		resRead := pipeline.FilterTrajectory(trajRepeatedRead)
		if resRead.Admitted {
			t.Fatal("expected trajectory with repeated identical read to be rejected")
		}
		if !strings.Contains(resRead.Reason, "repeated identical") && !strings.Contains(resRead.Reason, "cycle") {
			t.Errorf("expected reason to mention cycle or repeated call, got %q", resRead.Reason)
		}

		// Loop with non-read tool
		editSchema := ToolSchema{
			Name: "Edit",
			Properties: map[string]PropertySchema{
				"file": {Type: TypeString},
			},
			Required: []string{"file"},
		}
		validator.RegisterSchema(editSchema)

		trajLoopEdit := FrontierTrajectory{
			ID:           "traj-loop-edit",
			SystemPrompt: "System",
			UserQuery:    "Edit twice",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall: ToolCall{
						Name: "Edit",
						Args: map[string]any{"file": "foo.txt"},
					},
					ToolResult: "ok",
				},
				{
					StepIndex: 2,
					ToolCall: ToolCall{
						Name: "Edit",
						Args: map[string]any{"file": "foo.txt"},
					},
					ToolResult: "ok",
				},
			},
			FinalResponse: "Done",
			Success:       true,
		}

		resEdit := pipeline.FilterTrajectory(trajLoopEdit)
		if resEdit.Admitted {
			t.Fatal("expected trajectory with repeated identical mutating call to be rejected")
		}
	})

	t.Run("discard_backtracking", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		trajBacktrack := FrontierTrajectory{
			ID:           "traj-backtrack",
			SystemPrompt: "System",
			UserQuery:    "Find bug",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "Let me inspect auth.go",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "auth.go"},
						ReadOnly: true,
					},
					ToolResult: "empty",
				},
				{
					StepIndex: 2,
					Thought:   "That failed to find the bug, let me backtrack and check handler.go instead",
					ToolCall: ToolCall{
						Name:     "Read",
						Args:     map[string]any{"filePath": "handler.go"},
						ReadOnly: true,
					},
					ToolResult: "found bug",
				},
			},
			FinalResponse: "Bug found in handler.go",
			Success:       true,
		}

		res := pipeline.FilterTrajectory(trajBacktrack)
		if res.Admitted {
			t.Fatal("expected backtracking trajectory to be rejected")
		}
		if !strings.Contains(res.Reason, "backtracking") {
			t.Errorf("expected reason to mention backtracking, got %q", res.Reason)
		}
	})

	t.Run("discard_unsuccessful_or_empty_response", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		// Success == false
		unsuccessfulTraj := FrontierTrajectory{
			ID:        "traj-unsuccessful",
			UserQuery: "Do something",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall:  ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.txt"}},
				},
			},
			FinalResponse: "Could not complete",
			Success:       false,
		}
		if res := pipeline.FilterTrajectory(unsuccessfulTraj); res.Admitted {
			t.Fatal("expected unsuccessful trajectory to be rejected")
		}

		// Empty final response
		emptyResponseTraj := FrontierTrajectory{
			ID:        "traj-empty-resp",
			UserQuery: "Do something",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall:  ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.txt"}},
				},
			},
			FinalResponse: "   ",
			Success:       true,
		}
		if res := pipeline.FilterTrajectory(emptyResponseTraj); res.Admitted {
			t.Fatal("expected empty final response trajectory to be rejected")
		}

		// Zero tool calls
		zeroCallsTraj := FrontierTrajectory{
			ID:            "traj-no-tools",
			UserQuery:     "Answer without tools",
			Steps:         nil,
			FinalResponse: "Direct answer",
			Success:       true,
		}
		if res := pipeline.FilterTrajectory(zeroCallsTraj); res.Admitted {
			t.Fatal("expected trajectory with no tool calls to be rejected")
		}
	})

	t.Run("schema_validation", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		cfg.ValidateSchema = true
		pipeline := NewToolDistillationPipeline(cfg, validator)

		// Conforming tool call passes
		conforming := FrontierTrajectory{
			ID:            "traj-schema-ok",
			UserQuery:     "Query",
			FinalResponse: "Result",
			Success:       true,
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall: ToolCall{
						Name: "Read",
						Args: map[string]any{"filePath": "foo.go", "offset": 10.0, "limit": 20.0},
					},
				},
			},
		}
		if res := pipeline.FilterTrajectory(conforming); !res.Admitted {
			t.Fatalf("expected schema conforming trajectory to be admitted, got: %s", res.Reason)
		}

		// Missing required property 'filePath'
		missingReq := FrontierTrajectory{
			ID:            "traj-schema-missing",
			UserQuery:     "Query",
			FinalResponse: "Result",
			Success:       true,
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall: ToolCall{
						Name: "Read",
						Args: map[string]any{"offset": 10.0},
					},
				},
			},
		}
		resMissing := pipeline.FilterTrajectory(missingReq)
		if resMissing.Admitted {
			t.Fatal("expected trajectory with missing required schema property to be rejected")
		}
		if !strings.Contains(resMissing.Reason, `missing required property "filePath"`) {
			t.Errorf("expected missing property error, got: %q", resMissing.Reason)
		}

		// Type violation: offset is string instead of integer
		typeViolation := FrontierTrajectory{
			ID:            "traj-schema-bad-type",
			UserQuery:     "Query",
			FinalResponse: "Result",
			Success:       true,
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall: ToolCall{
						Name: "Read",
						Args: map[string]any{"filePath": "foo.go", "offset": "ten"},
					},
				},
			},
		}
		resType := pipeline.FilterTrajectory(typeViolation)
		if resType.Admitted {
			t.Fatal("expected type violation to be rejected")
		}
		if !strings.Contains(resType.Reason, "expected integer") {
			t.Errorf("expected integer type violation error, got: %q", resType.Reason)
		}

		// Unregistered tool
		unknownTool := FrontierTrajectory{
			ID:            "traj-schema-unknown",
			UserQuery:     "Query",
			FinalResponse: "Result",
			Success:       true,
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					ToolCall: ToolCall{
						Name: "UnregisteredTool",
						Args: map[string]any{"arg": "val"},
					},
				},
			},
		}
		resUnknown := pipeline.FilterTrajectory(unknownTool)
		if resUnknown.Admitted {
			t.Fatal("expected unregistered tool to be rejected by schema validation")
		}
		if !strings.Contains(resUnknown.Reason, "no schema registered") {
			t.Errorf("expected unregistered tool error, got: %q", resUnknown.Reason)
		}
	})

	t.Run("batch_processing_and_statistics", func(t *testing.T) {
		cfg := DefaultDistillationFilterConfig()
		pipeline := NewToolDistillationPipeline(cfg, validator)

		trajectories := []FrontierTrajectory{
			// 1. Clean and valid
			{
				ID:            "t1-clean",
				UserQuery:     "Grep pattern",
				FinalResponse: "Found pattern",
				Success:       true,
				Steps: []TrajectoryStep{
					{
						StepIndex: 1,
						ToolCall: ToolCall{
							Name: "Grep",
							Args: map[string]any{"pattern": "main", "path": "main.go"},
						},
					},
				},
			},
			// 2. Failed call
			{
				ID:            "t2-failed",
				UserQuery:     "Read missing",
				FinalResponse: "Failed",
				Success:       true,
				Steps: []TrajectoryStep{
					{
						StepIndex: 1,
						ToolCall:  ToolCall{Name: "Read", Args: map[string]any{"filePath": "no.go"}},
						Error:     "file missing",
					},
				},
			},
			// 3. Loop / repeated read
			{
				ID:            "t3-loop",
				UserQuery:     "Loop read",
				FinalResponse: "Done",
				Success:       true,
				Steps: []TrajectoryStep{
					{StepIndex: 1, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.go"}}},
					{StepIndex: 2, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.go"}}},
				},
			},
			// 4. Schema violation
			{
				ID:            "t4-schema",
				UserQuery:     "Bad args",
				FinalResponse: "Done",
				Success:       true,
				Steps: []TrajectoryStep{
					{StepIndex: 1, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"offset": 0.0}}},
				},
			},
			// 5. Unsuccessful
			{
				ID:            "t5-unsuccessful",
				UserQuery:     "Failed task",
				FinalResponse: "Failed",
				Success:       false,
				Steps: []TrajectoryStep{
					{StepIndex: 1, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.go"}}},
				},
			},
			// 6. Backtracking
			{
				ID:            "t6-backtrack",
				UserQuery:     "Try and backtrack",
				FinalResponse: "Done",
				Success:       true,
				Steps: []TrajectoryStep{
					{StepIndex: 1, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"filePath": "a.go"}}, Thought: "Check a.go"},
					{StepIndex: 2, ToolCall: ToolCall{Name: "Read", Args: map[string]any{"filePath": "b.go"}}, Thought: "That didn't work, let me backtrack to b.go"},
				},
			},
		}

		dataset, stats := pipeline.Process(trajectories)
		if dataset.Len() != 1 {
			t.Fatalf("expected dataset length 1, got %d", dataset.Len())
		}
		if stats.TotalProcessed != 6 {
			t.Errorf("expected 6 processed, got %d", stats.TotalProcessed)
		}
		if stats.Admitted != 1 {
			t.Errorf("expected 1 admitted, got %d", stats.Admitted)
		}
		if stats.Discarded != 5 {
			t.Errorf("expected 5 discarded, got %d", stats.Discarded)
		}
		if stats.DiscardedErrors < 1 {
			t.Errorf("expected at least 1 error/backtrack discard, got %d", stats.DiscardedErrors)
		}
		if stats.DiscardedCycles < 1 {
			t.Errorf("expected at least 1 cycle discard, got %d", stats.DiscardedCycles)
		}
		if stats.DiscardedSchema < 1 {
			t.Errorf("expected at least 1 schema discard, got %d", stats.DiscardedSchema)
		}
		if stats.DiscardedStatus < 1 {
			t.Errorf("expected at least 1 status discard, got %d", stats.DiscardedStatus)
		}
	})

	t.Run("dataset_formatting_and_export", func(t *testing.T) {
		example := DistillationExample{
			ID:           "ex-01",
			SystemPrompt: "You are a coding assistant.",
			UserQuery:    "Read main.go",
			Thought:      "I will read main.go to inspect the entry point.",
			ToolCalls: []ToolCall{
				{
					ID:       "call-1",
					Name:     "Read",
					Args:     map[string]any{"filePath": "main.go"},
					ReadOnly: true,
				},
			},
			FinalResponse: "Entry point found.",
		}

		dataset := &DistillationDataset{
			Name:     "test-distillation-dataset",
			Examples: []DistillationExample{example},
		}

		// 1. JSON Export
		jsonBytes, err := dataset.ToJSON()
		if err != nil {
			t.Fatalf("failed to serialize dataset to JSON: %v", err)
		}
		if !strings.Contains(string(jsonBytes), "test-distillation-dataset") {
			t.Errorf("serialized JSON missing dataset name: %s", string(jsonBytes))
		}

		// 2. Standard JSONL Export
		var jsonlBuf bytes.Buffer
		if err := dataset.ToJSONL(&jsonlBuf); err != nil {
			t.Fatalf("failed to write JSONL: %v", err)
		}
		var parsedExample DistillationExample
		if err := json.Unmarshal(jsonlBuf.Bytes(), &parsedExample); err != nil {
			t.Fatalf("failed to unmarshal JSONL line: %v", err)
		}
		if parsedExample.UserQuery != example.UserQuery {
			t.Errorf("JSONL round-trip mismatch: got %q, want %q", parsedExample.UserQuery, example.UserQuery)
		}

		// 3. Chat / OpenAI JSONL Export
		var chatBuf bytes.Buffer
		if err := dataset.ToChatJSONL(&chatBuf); err != nil {
			t.Fatalf("failed to write Chat JSONL: %v", err)
		}
		var chatDemo ChatDemonstration
		if err := json.Unmarshal(chatBuf.Bytes(), &chatDemo); err != nil {
			t.Fatalf("failed to unmarshal Chat JSONL: %v", err)
		}
		if len(chatDemo.Messages) != 3 {
			t.Fatalf("expected 3 chat messages (system, user, assistant), got %d", len(chatDemo.Messages))
		}
		if chatDemo.Messages[0].Role != "system" || chatDemo.Messages[1].Role != "user" || chatDemo.Messages[2].Role != "assistant" {
			t.Errorf("unexpected message roles: %+v", chatDemo.Messages)
		}
		if len(chatDemo.Messages[2].ToolCalls) != 1 {
			t.Errorf("assistant message missing tool call: %+v", chatDemo.Messages[2])
		}

		// 4. Alpaca JSONL Export
		var alpacaBuf bytes.Buffer
		if err := dataset.ToAlpacaJSONL(&alpacaBuf); err != nil {
			t.Fatalf("failed to write Alpaca JSONL: %v", err)
		}
		var alpacaDemo AlpacaDemonstration
		if err := json.Unmarshal(alpacaBuf.Bytes(), &alpacaDemo); err != nil {
			t.Fatalf("failed to unmarshal Alpaca JSONL: %v", err)
		}
		if alpacaDemo.Instruction != example.SystemPrompt {
			t.Errorf("expected instruction %q, got %q", example.SystemPrompt, alpacaDemo.Instruction)
		}
		if alpacaDemo.Input != example.UserQuery {
			t.Errorf("expected input %q, got %q", example.UserQuery, alpacaDemo.Input)
		}
		if !strings.Contains(alpacaDemo.Output, "<thought>") || !strings.Contains(alpacaDemo.Output, "<tool_calls>") {
			t.Errorf("alpaca output missing thought or tool calls: %s", alpacaDemo.Output)
		}

		// 5. ShareGPT JSONL Export
		var shareGPTBuf bytes.Buffer
		if err := dataset.ToShareGPTJSONL(&shareGPTBuf); err != nil {
			t.Fatalf("failed to write ShareGPT JSONL: %v", err)
		}
		var shareGPTDemo ShareGPTDemonstration
		if err := json.Unmarshal(shareGPTBuf.Bytes(), &shareGPTDemo); err != nil {
			t.Fatalf("failed to unmarshal ShareGPT JSONL: %v", err)
		}
		if len(shareGPTDemo.Conversations) != 3 {
			t.Fatalf("expected 3 conversations (system, human, gpt), got %d", len(shareGPTDemo.Conversations))
		}
		if shareGPTDemo.Conversations[0].From != "system" || shareGPTDemo.Conversations[1].From != "human" || shareGPTDemo.Conversations[2].From != "gpt" {
			t.Errorf("unexpected ShareGPT from values: %+v", shareGPTDemo.Conversations)
		}
	})
}
