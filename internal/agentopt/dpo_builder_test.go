package agentopt

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestDPOPreferenceDatasetBuilder(t *testing.T) {
	t.Run("valid chosen/rejected pair extraction on same prompt", func(t *testing.T) {
		builder := NewDPOPreferenceDatasetBuilder()

		chosen := AgentTrajectory{
			ID:     "traj-chosen-1",
			Prompt: "Implement a thread-safe cache in Go",
			Steps: []ExecutionStep{
				{
					StepIndex: 1,
					Thought:   "I need to implement a cache struct with a sync.RWMutex.",
					ToolCall: &ToolCall{
						Name: "WriteFile",
						Args: map[string]any{"path": "cache.go"},
					},
					Receipt: &ToolReceipt{
						ToolName: "WriteFile",
						Output:   "File cache.go written successfully with 45 lines",
					},
				},
				{
					StepIndex: 2,
					Thought:   "Now running tests to verify concurrency safety.",
					ToolCall: &ToolCall{
						Name: "RunTests",
						Args: map[string]any{"package": "./cache"},
					},
					Receipt: &ToolReceipt{
						ToolName: "RunTests",
						Output:   "PASS: TestCacheConcurrentGetSet (0.05s)",
					},
				},
			},
			FinalAnswer: "Successfully implemented concurrent cache with sync.RWMutex.",
			Success:     true,
			Witnessed:   true,
			Metadata:    map[string]any{"model": "qwen38-27b", "latency_ms": 1200},
		}

		rejected := AgentTrajectory{
			ID:     "traj-rejected-1",
			Prompt: "Implement a thread-safe cache in Go",
			Steps: []ExecutionStep{
				{
					StepIndex: 1,
					Thought:   "I will create a map without locks.",
					ToolCall: &ToolCall{
						Name: "WriteFile",
						Args: map[string]any{"path": "cache.go"},
					},
					Receipt: &ToolReceipt{
						ToolName: "WriteFile",
						Output:   "File cache.go written",
					},
				},
				{
					StepIndex: 2,
					Thought:   "Running tests now.",
					ToolCall: &ToolCall{
						Name: "RunTests",
						Args: map[string]any{"package": "./cache"},
					},
					Receipt: &ToolReceipt{
						ToolName: "RunTests",
						Error:    "fatal error: concurrent map read and map write",
					},
				},
			},
			Error:       "fatal error: concurrent map read and map write",
			Success:     false,
			Witnessed:   false,
			FinalAnswer: "Tests failed with concurrency race.",
			Metadata:    map[string]any{"model": "baseline-7b"},
		}

		builder.AddTrajectory(chosen, rejected)

		dataset, err := builder.Build()
		if err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}

		if len(dataset) != 1 {
			t.Fatalf("expected exactly 1 DPOPair extracted, got %d", len(dataset))
		}

		pair := dataset[0]

		// Verify Prompt
		if pair.Prompt != "Implement a thread-safe cache in Go" {
			t.Errorf("unexpected prompt: got %q", pair.Prompt)
		}

		// Verify Chosen content
		if !strings.Contains(pair.Chosen, "Action: WriteFile") {
			t.Errorf("expected chosen to contain Action: WriteFile, got:\n%s", pair.Chosen)
		}
		if !strings.Contains(pair.Chosen, "PASS: TestCacheConcurrentGetSet") {
			t.Errorf("expected chosen to contain passing test receipt, got:\n%s", pair.Chosen)
		}
		if !strings.Contains(pair.Chosen, "Final Answer: Successfully implemented concurrent cache") {
			t.Errorf("expected chosen to contain final answer, got:\n%s", pair.Chosen)
		}

		// Verify Rejected content
		if !strings.Contains(pair.Rejected, "fatal error: concurrent map read and map write") {
			t.Errorf("expected rejected to contain error observation, got:\n%s", pair.Rejected)
		}

		// Verify Metadata provenance
		if pair.Metadata == nil {
			t.Fatal("expected non-nil metadata")
		}
		if pair.Metadata["chosen_id"] != "traj-chosen-1" {
			t.Errorf("expected chosen_id 'traj-chosen-1', got %v", pair.Metadata["chosen_id"])
		}
		if pair.Metadata["rejected_id"] != "traj-rejected-1" {
			t.Errorf("expected rejected_id 'traj-rejected-1', got %v", pair.Metadata["rejected_id"])
		}
		if pair.Metadata["witnessed"] != true {
			t.Errorf("expected witnessed true, got %v", pair.Metadata["witnessed"])
		}
		if pair.Metadata["family"] != "Family 15: Training & adaptation for agent optimization" {
			t.Errorf("expected family metadata, got %v", pair.Metadata["family"])
		}
	})

	t.Run("scrubbing prompts and tool receipts", func(t *testing.T) {
		builder := NewDPOPreferenceDatasetBuilder()

		// Trajectory with secrets in prompt and tool receipts
		rawPrompt := "Audit server at db.internal using api_key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
		chosen := AgentTrajectory{
			ID:     "traj-scrub-chosen",
			Prompt: rawPrompt,
			Steps: []ExecutionStep{
				{
					StepIndex: 1,
					Thought:   "Connecting with credentials to private host.",
					ToolCall: &ToolCall{
						Name: "ConnectDB",
						Args: map[string]any{"token": "secret_vault_token_123"},
					},
					Receipt: &ToolReceipt{
						ToolName: "ConnectDB",
						Output:   "Connected with Authorization: Bearer session_token_99999999 to 10.0.1.20",
					},
				},
			},
			FinalAnswer: "Audited host db.internal successfully. Access verified with password=super_secret_pw",
			Success:     true,
			Witnessed:   true,
		}

		rejected := AgentTrajectory{
			ID:     "traj-scrub-rejected",
			Prompt: rawPrompt,
			Steps: []ExecutionStep{
				{
					StepIndex: 1,
					Thought:   "Failed connection attempt.",
					ToolCall: &ToolCall{
						Name: "ConnectDB",
						Args: map[string]any{"token": "secret_vault_token_123"},
					},
					Receipt: &ToolReceipt{
						ToolName: "ConnectDB",
						Error:    "Connection refused from 192.168.1.50: invalid auth=bad_auth_token",
					},
				},
			},
			Error:   "Connection refused from 192.168.1.50: invalid auth=bad_auth_token",
			Success: false,
		}

		builder.AddTrajectory(chosen, rejected)
		dataset, err := builder.Build()
		if err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		if len(dataset) != 1 {
			t.Fatalf("expected 1 pair, got %d", len(dataset))
		}

		pair := dataset[0]

		// 1. Verify prompt was scrubbed
		if strings.Contains(pair.Prompt, "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456") {
			t.Errorf("prompt leaked raw api key: %s", pair.Prompt)
		}
		if strings.Contains(pair.Prompt, "db.internal") {
			t.Errorf("prompt leaked internal hostname: %s", pair.Prompt)
		}
		if !strings.Contains(pair.Prompt, "[REDACTED_API_KEY]") && !strings.Contains(pair.Prompt, "[REDACTED]") {
			t.Errorf("expected redacted api key token in prompt: %s", pair.Prompt)
		}
		if !strings.Contains(pair.Prompt, "[REDACTED_HOST]") {
			t.Errorf("expected redacted host token in prompt: %s", pair.Prompt)
		}

		// 2. Verify chosen receipt and answer were scrubbed
		if strings.Contains(pair.Chosen, "session_token_99999999") {
			t.Errorf("chosen leaked bearer token: %s", pair.Chosen)
		}
		if strings.Contains(pair.Chosen, "10.0.1.20") {
			t.Errorf("chosen leaked private IP: %s", pair.Chosen)
		}
		if strings.Contains(pair.Chosen, "super_secret_pw") {
			t.Errorf("chosen leaked password: %s", pair.Chosen)
		}
		if !strings.Contains(pair.Chosen, "Bearer [REDACTED]") {
			t.Errorf("expected Bearer [REDACTED] in chosen: %s", pair.Chosen)
		}
		if !strings.Contains(pair.Chosen, "[REDACTED_IP]") {
			t.Errorf("expected [REDACTED_IP] in chosen: %s", pair.Chosen)
		}

		// 3. Verify rejected receipt was scrubbed
		if strings.Contains(pair.Rejected, "192.168.1.50") {
			t.Errorf("rejected leaked private IP: %s", pair.Rejected)
		}
		if strings.Contains(pair.Rejected, "bad_auth_token") {
			t.Errorf("rejected leaked auth token: %s", pair.Rejected)
		}
		if !strings.Contains(pair.Rejected, "[REDACTED_IP]") {
			t.Errorf("expected [REDACTED_IP] in rejected: %s", pair.Rejected)
		}
	})

	t.Run("export standard JSONL DPO dataset", func(t *testing.T) {
		pair := DPOPair{
			Prompt:   "Write a hello world program in Go",
			Chosen:   "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello, World!\") }",
			Rejected: "echo 'hello'",
			Metadata: map[string]any{
				"family":    "Family 15: Training & adaptation for agent optimization",
				"chosen_id": "c1",
			},
		}

		// Single pair export
		raw, err := pair.ExportJSONL()
		if err != nil {
			t.Fatalf("pair.ExportJSONL failed: %v", err)
		}
		if !strings.HasSuffix(string(raw), "\n") {
			t.Errorf("JSONL must end with newline, got %q", string(raw))
		}

		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("failed to unmarshal exported JSONL: %v", err)
		}
		if decoded["prompt"] != "Write a hello world program in Go" {
			t.Errorf("unexpected decoded prompt: %v", decoded["prompt"])
		}
		if decoded["chosen"] != pair.Chosen {
			t.Errorf("unexpected decoded chosen: %v", decoded["chosen"])
		}
		if decoded["rejected"] != pair.Rejected {
			t.Errorf("unexpected decoded rejected: %v", decoded["rejected"])
		}

		// Multi-pair dataset export
		dataset := DPODataset{pair, pair}
		var buf bytes.Buffer
		if err := dataset.ExportJSONL(&buf); err != nil {
			t.Fatalf("dataset.ExportJSONL failed: %v", err)
		}

		lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 JSONL lines, got %d", len(lines))
		}

		for idx, line := range lines {
			var lineDecoded DPOPair
			if err := json.Unmarshal([]byte(line), &lineDecoded); err != nil {
				t.Fatalf("line %d invalid JSON: %v", idx, err)
			}
			if lineDecoded.Prompt != pair.Prompt {
				t.Errorf("line %d prompt mismatch: %q", idx, lineDecoded.Prompt)
			}
		}

		// Builder.ExportJSONL method
		builder := NewDPOPreferenceDatasetBuilder()
		var bBuf bytes.Buffer
		if err := builder.ExportJSONL(&bBuf, dataset); err != nil {
			t.Fatalf("builder.ExportJSONL failed: %v", err)
		}
		if bBuf.String() != buf.String() {
			t.Errorf("builder.ExportJSONL output differed from dataset.ExportJSONL")
		}
	})

	t.Run("skips prompts with only chosen or only rejected attempts", func(t *testing.T) {
		builder := NewDPOPreferenceDatasetBuilder()

		// Prompt 1: Only successes (no rejected to contrast against)
		builder.AddTrajectory(
			AgentTrajectory{
				ID:        "t1",
				Prompt:    "Prompt with only success",
				Success:   true,
				Witnessed: true,
			},
			AgentTrajectory{
				ID:        "t2",
				Prompt:    "Prompt with only success",
				Success:   true,
				Witnessed: true,
			},
		)

		// Prompt 2: Only failures (no chosen)
		builder.AddTrajectory(
			AgentTrajectory{
				ID:      "t3",
				Prompt:  "Prompt with only failure",
				Success: false,
				Error:   "failure 1",
			},
			AgentTrajectory{
				ID:      "t4",
				Prompt:  "Prompt with only failure",
				Success: false,
				Error:   "failure 2",
			},
		)

		dataset, err := builder.Build()
		if err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		if len(dataset) != 0 {
			t.Fatalf("expected 0 pairs from uncontrasted prompts, got %d", len(dataset))
		}
	})

	t.Run("pairing strategies and caps", func(t *testing.T) {
		trajs := []AgentTrajectory{
			{ID: "c1", Prompt: "Task A", Success: true, Witnessed: true, Completion: "chosen 1"},
			{ID: "c2", Prompt: "Task A", Success: true, Witnessed: true, Completion: "chosen 2"},
			{ID: "r1", Prompt: "Task A", Success: false, Error: "err1", Completion: "rejected 1"},
			{ID: "r2", Prompt: "Task A", Success: false, Error: "err2", Completion: "rejected 2"},
		}

		// 1. PairAll -> 2 chosen x 2 rejected = 4 pairs
		bAll := NewDPOPreferenceDatasetBuilder(WithPairingStrategy(PairAll))
		bAll.AddTrajectory(trajs...)
		dAll, _ := bAll.Build()
		if len(dAll) != 4 {
			t.Errorf("expected 4 pairs for PairAll, got %d", len(dAll))
		}

		// 2. PairOneToOne -> min(2, 2) = 2 pairs
		b1to1 := NewDPOPreferenceDatasetBuilder(WithPairingStrategy(PairOneToOne))
		b1to1.AddTrajectory(trajs...)
		d1to1, _ := b1to1.Build()
		if len(d1to1) != 2 {
			t.Errorf("expected 2 pairs for PairOneToOne, got %d", len(d1to1))
		}

		// 3. PairBestWorst -> 1 pair
		bBW := NewDPOPreferenceDatasetBuilder(WithPairingStrategy(PairBestWorst))
		bBW.AddTrajectory(trajs...)
		dBW, _ := bBW.Build()
		if len(dBW) != 1 {
			t.Errorf("expected 1 pair for PairBestWorst, got %d", len(dBW))
		}

		// 4. MaxPairsPerPrompt
		bCap := NewDPOPreferenceDatasetBuilder(WithMaxPairsPerPrompt(1))
		bCap.AddTrajectory(trajs...)
		dCap, _ := bCap.Build()
		if len(dCap) != 1 {
			t.Errorf("expected 1 pair for WithMaxPairsPerPrompt(1), got %d", len(dCap))
		}
	})

	t.Run("journal ingestion from JSON and JSONL", func(t *testing.T) {
		jsonlData := `{"id":"t-c","prompt":"Summarize file","success":true,"witnessed":true,"completion":"Summary of file."}
{"id":"t-r","prompt":"Summarize file","success":false,"error":"file not found","completion":"Error reading file."}
`
		bJSONL := NewDPOPreferenceDatasetBuilder()
		if err := bJSONL.IngestJournalJSONL(strings.NewReader(jsonlData)); err != nil {
			t.Fatalf("IngestJournalJSONL failed: %v", err)
		}
		dJSONL, err := bJSONL.Build()
		if err != nil || len(dJSONL) != 1 {
			t.Fatalf("expected 1 pair from JSONL ingestion, got %d (err: %v)", len(dJSONL), err)
		}

		jsonData := `[
			{"id":"t-c","prompt":"Format JSON","success":true,"witnessed":true,"completion":"{\"ok\":true}"},
			{"id":"t-r","prompt":"Format JSON","success":false,"error":"bad json","completion":"{invalid}"}
		]`
		bJSON := NewDPOPreferenceDatasetBuilder()
		if err := bJSON.IngestJournalJSON(strings.NewReader(jsonData)); err != nil {
			t.Fatalf("IngestJournalJSON failed: %v", err)
		}
		dJSON, err := bJSON.Build()
		if err != nil || len(dJSON) != 1 {
			t.Fatalf("expected 1 pair from JSON ingestion, got %d (err: %v)", len(dJSON), err)
		}
	})

	t.Run("adaptation from FrontierTrajectory and ReplayTrajectory", func(t *testing.T) {
		ft := FrontierTrajectory{
			ID:            "ft-1",
			UserQuery:     "Refactor code",
			Success:       true,
			FinalResponse: "Code refactored cleanly",
			Steps: []TrajectoryStep{
				{
					StepIndex:  1,
					Thought:    "Analyzing structure",
					ToolCall:   ToolCall{Name: "ReadFile", Args: map[string]any{"path": "main.go"}},
					ToolResult: "package main",
				},
			},
		}
		adaptedFT := AdaptFrontierTrajectory(ft)
		if adaptedFT.Prompt != "Refactor code" || !adaptedFT.Success || !adaptedFT.Witnessed {
			t.Errorf("unexpected adapted frontier trajectory: %+v", adaptedFT)
		}

		rt := Trajectory{
			ID:     "rt-1",
			Prompt: "Refactor code",
			Turns: []TrajectoryTurn{
				{
					TurnIndex: 1,
					ToolCalls: []ToolCall{{Name: "ReadFile"}},
					Results:   []ToolResult{{Output: "err", Error: "permission denied"}},
				},
			},
		}
		adaptedRT := AdaptReplayTrajectory(rt)
		if adaptedRT.Prompt != "Refactor code" || adaptedRT.Success {
			t.Errorf("unexpected adapted replay trajectory: %+v", adaptedRT)
		}

		b := NewDPOPreferenceDatasetBuilder()
		b.AddTrajectory(adaptedFT, adaptedRT)
		pairs, err := b.Build()
		if err != nil || len(pairs) != 1 {
			t.Fatalf("expected 1 adapted pair, got %d (err: %v)", len(pairs), err)
		}
	})
}
