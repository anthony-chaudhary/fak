package tb4bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RunCampaignConfig defines configuration for executing a TB4 benchmark campaign.
type RunCampaignConfig struct {
	Tasks       []TaskManifest       `json:"tasks"`
	Arm         string               `json:"arm"` // "fak", "opencode", or "both"
	ModelPath   string               `json:"model_path"`
	OutDir      string               `json:"out_dir"`
	MockMode    bool                 `json:"mock_mode"`
	Determinism DeterminismEnvelope  `json:"determinism"`
	Contract    *OfficialRunContract `json:"contract,omitempty"`
}

// RunCampaignResult summarizes the completed execution of tasks across arms.
type RunCampaignResult struct {
	OutDir      string                                    `json:"out_dir"`
	Contract    *OfficialRunContract                      `json:"contract"`
	ArmResults  map[string]map[string]*ArmExecutionResult `json:"arm_results"` // arm -> taskID -> result
	ArmExecuted []string                                  `json:"arm_executed"`
}

// EvaluateCampaignConfig defines configuration for evaluating completed task workspaces.
type EvaluateCampaignConfig struct {
	RunDir  string         `json:"run_dir"`
	Dataset string         `json:"dataset,omitempty"`
	Tasks   []TaskManifest `json:"tasks,omitempty"`
}

// EvaluateCampaignResult summarizes grading outcomes across evaluated arms.
type EvaluateCampaignResult struct {
	Receipts    map[string]map[string]*GradingReceipt `json:"receipts"` // arm -> taskID -> receipt
	SolvedCount map[string]int                        `json:"solved_count"`
	TotalCount  map[string]int                        `json:"total_count"`
	SolveRates  map[string]float64                    `json:"solve_rates"`
}

// CompareCampaignConfig defines configuration for synthesizing comparative analysis.
type CompareCampaignConfig struct {
	FakDir       string         `json:"fak_dir"`
	OpenCodeDir  string         `json:"opencode_dir"`
	Dataset      string         `json:"dataset,omitempty"`
	Tasks        []TaskManifest `json:"tasks,omitempty"`
	ContractPath string         `json:"contract_path,omitempty"`
	OutJSON      string         `json:"out_json,omitempty"`
	OutMD        string         `json:"out_md,omitempty"`
}

// RunCampaign executes the benchmark campaign across the requested arms.
func RunCampaign(ctx context.Context, cfg RunCampaignConfig) (*RunCampaignResult, error) {
	if len(cfg.Tasks) == 0 {
		return nil, errors.New("no tasks specified in campaign config")
	}
	if cfg.OutDir == "" {
		return nil, errors.New("outDir cannot be empty")
	}

	arm := strings.ToLower(strings.TrimSpace(cfg.Arm))
	if arm == "" {
		arm = "both"
	}

	var runFak, runOpenCode bool
	switch arm {
	case "fak", "fak_inkernel":
		runFak = true
	case "opencode", "opencode_llamacpp":
		runOpenCode = true
	case "both":
		runFak = true
		runOpenCode = true
	default:
		return nil, fmt.Errorf("unknown arm %q: expected 'fak', 'opencode', or 'both'", cfg.Arm)
	}

	if err := os.MkdirAll(cfg.OutDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}

	// Prepare and save official run contract if not already present
	contract := cfg.Contract
	if contract == nil {
		var taskIDs []string
		for _, t := range cfg.Tasks {
			taskIDs = append(taskIDs, t.TaskID)
		}
		contract = DefaultRunContract(cfg.ModelPath, "sha256:pinned", "Q4_K_M", taskIDs)
		contract.Determinism = cfg.Determinism
	}
	contractData, err := json.MarshalIndent(contract, "", "  ")
	if err == nil {
		_ = os.WriteFile(filepath.Join(cfg.OutDir, "contract.json"), contractData, 0644)
	}

	// Save task manifest suite for reproducibility and downstream evaluation
	suite := ManifestSuite{
		Benchmark: BenchmarkName,
		Version:   "1.0",
		Tasks:     cfg.Tasks,
	}
	if suiteData, err := json.MarshalIndent(suite, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(cfg.OutDir, "manifest.json"), suiteData, 0644)
	}

	result := &RunCampaignResult{
		OutDir:     cfg.OutDir,
		Contract:   contract,
		ArmResults: make(map[string]map[string]*ArmExecutionResult),
	}

	if cfg.MockMode {
		return runMockCampaign(ctx, cfg, contract, runFak, runOpenCode, result)
	}

	return runRealCampaign(ctx, cfg, contract, runFak, runOpenCode, result)
}

func runMockCampaign(
	ctx context.Context,
	cfg RunCampaignConfig,
	contract *OfficialRunContract,
	runFak, runOpenCode bool,
	result *RunCampaignResult,
) (*RunCampaignResult, error) {
	// 1. Arm A ("fak_inkernel")
	if runFak {
		result.ArmExecuted = append(result.ArmExecuted, "fak")
		fakResults := make(map[string]*ArmExecutionResult)

		mockEngine := NewMockContainerEngine()
		defer mockEngine.Close()

		adapter, err := NewInKernelModelAdapter("", "")
		if err != nil {
			return nil, fmt.Errorf("failed to init in-kernel adapter: %w", err)
		}
		registerSyntheticScriptedResponses(adapter)

		for _, task := range cfg.Tasks {
			taskDir := filepath.Join(cfg.OutDir, "fak", "tasks", task.TaskID)
			wsDir := filepath.Join(taskDir, "workspace")
			if err := os.MkdirAll(wsDir, 0755); err != nil {
				return nil, err
			}

			cConfig := ContainerConfig{
				ImageDigest: task.EnvironmentImageDigest,
				Name:        "tb4-fak-" + task.TaskID,
				NetworkMode: NetworkModeNone,
				WorkingDir:  "/workspace",
			}
			inst, err := mockEngine.CreateContainer(ctx, cConfig)
			if err != nil {
				return nil, fmt.Errorf("mock container creation failed for %s: %w", task.TaskID, err)
			}

			mockEngine.workspaces[inst.ID] = wsDir
			wsMgr := NewWorkspaceManager(mockEngine, inst.ID, task.TaskID, wsDir)
			if _, err := wsMgr.SeedWorkspace(ctx, task.Prompt, nil); err != nil {
				return nil, fmt.Errorf("failed to seed workspace for %s: %w", task.TaskID, err)
			}

			if task.SetupCommand != "" {
				_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", task.SetupCommand}, 30*time.Second)
			}

			harness := NewFakHarness(adapter, nil, wsMgr)
			det := cfg.Determinism
			if det.MaxTurns <= 0 {
				det = DefaultDeterminismEnvelope()
			}

			armExec, err := harness.ExecuteTask(ctx, task, det)
			if err != nil {
				return nil, fmt.Errorf("fak harness execution failed for %s: %w", task.TaskID, err)
			}

			// Write transcript.jsonl
			if err := writeTranscriptJSONL(filepath.Join(taskDir, "transcript.jsonl"), armExec.Turns); err != nil {
				return nil, err
			}

			// Write result.json
			resData, _ := json.MarshalIndent(armExec, "", "  ")
			if err := os.WriteFile(filepath.Join(taskDir, "result.json"), resData, 0644); err != nil {
				return nil, err
			}

			fakResults[task.TaskID] = armExec
		}

		result.ArmResults["fak"] = fakResults

		// Write telemetry.json for fak
		var totalPrompt, totalComp, vdsoHits, policyBlocks int64
		for _, r := range fakResults {
			totalPrompt += r.TotalPromptTokens
			totalComp += r.TotalCompletionTokens
			vdsoHits += r.VDSOHits
			policyBlocks += r.PolicyBlocks
		}
		if vdsoHits == 0 {
			vdsoHits = adapter.Telemetry().KVHits
		}
		telemetryA := TelemetryTierMetrics{
			TotalPromptTokens:     totalPrompt,
			TotalCompletionTokens: totalComp,
			VDSOHits:              vdsoHits,
			PolicyBlocks:          policyBlocks,
		}
		teleDataA, _ := json.MarshalIndent(telemetryA, "", "  ")
		_ = os.WriteFile(filepath.Join(cfg.OutDir, "fak", "telemetry.json"), teleDataA, 0644)
	}

	// 2. Arm B ("opencode_llamacpp")
	if runOpenCode {
		result.ArmExecuted = append(result.ArmExecuted, "opencode")
		opencodeResults := make(map[string]*ArmExecutionResult)

		engB := NewMockContainerEngine()
		defer engB.Close()

		for i, task := range cfg.Tasks {
			taskDir := filepath.Join(cfg.OutDir, "opencode", "tasks", task.TaskID)
			wsDir := filepath.Join(taskDir, "workspace")
			if err := os.MkdirAll(wsDir, 0755); err != nil {
				return nil, err
			}

			cConfig := ContainerConfig{
				ImageDigest: task.EnvironmentImageDigest,
				Name:        "tb4-opencode-" + task.TaskID,
				NetworkMode: NetworkModeNone,
				WorkingDir:  "/workspace",
			}
			inst, err := engB.CreateContainer(ctx, cConfig)
			if err != nil {
				return nil, err
			}

			engB.workspaces[inst.ID] = wsDir
			wsMgr := NewWorkspaceManager(engB, inst.ID, task.TaskID, wsDir)
			if _, err := wsMgr.SeedWorkspace(ctx, task.Prompt, nil); err != nil {
				return nil, fmt.Errorf("failed to seed workspace for %s: %w", task.TaskID, err)
			}

			if task.SetupCommand != "" {
				_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", task.SetupCommand}, 30*time.Second)
			}

			// In MockMode, simulate 3 solved and 2 failed (matching e2e_test.go)
			shouldSolve := (i < 3)
			var turns []TurnRecord

			if shouldSolve {
				applyMockSolution(ctx, wsMgr, task.TaskID)
				turns = []TurnRecord{
					{
						Turn:      1,
						ModelText: "Inspecting files and applying requested change.",
						ToolCalls: []ToolCallProposal{
							{ID: "call_opencode_1", Name: "bash", Arguments: `{"cmd":"make fix"}`},
						},
						ToolResults: []ToolExecutionResult{
							{ToolCallID: "call_opencode_1", Tool: "bash", Stdout: "applied", ExitCode: 0},
						},
						PromptTokens:     280,
						CompletionTokens: 60,
						DurationMs:       450,
					},
					{
						Turn:             2,
						ModelText:        "Done. TASK_COMPLETED",
						PromptTokens:     350,
						CompletionTokens: 20,
						DurationMs:       200,
					},
				}
			} else {
				// Failed task simulation
				turns = []TurnRecord{
					{
						Turn:      1,
						ModelText: "Checking logs and debugging.",
						ToolCalls: []ToolCallProposal{
							{ID: "call_opencode_1", Name: "bash", Arguments: `{"cmd":"test error"}`},
						},
						ToolResults: []ToolExecutionResult{
							{ToolCallID: "call_opencode_1", Tool: "bash", Stderr: "error", ExitCode: 1},
						},
						PromptTokens:     300,
						CompletionTokens: 40,
						DurationMs:       500,
					},
					{
						Turn:             2,
						ModelText:        "Could not solve. TASK_COMPLETED",
						PromptTokens:     360,
						CompletionTokens: 15,
						DurationMs:       250,
					},
				}
			}

			var pTokens, cTokens, durMs int64
			for _, tr := range turns {
				pTokens += tr.PromptTokens
				cTokens += tr.CompletionTokens
				durMs += tr.DurationMs
			}

			execRes := &ArmExecutionResult{
				ArmID:                 "opencode_llamacpp",
				TaskID:                task.TaskID,
				Status:                "COMPLETED",
				Turns:                 turns,
				TotalTurns:            len(turns),
				TotalPromptTokens:     pTokens,
				TotalCompletionTokens: cTokens,
				DurationMs:            durMs,
			}

			// Write transcript.jsonl
			if err := writeTranscriptJSONL(filepath.Join(taskDir, "transcript.jsonl"), turns); err != nil {
				return nil, err
			}

			// Write result.json
			resData, _ := json.MarshalIndent(execRes, "", "  ")
			if err := os.WriteFile(filepath.Join(taskDir, "result.json"), resData, 0644); err != nil {
				return nil, err
			}

			opencodeResults[task.TaskID] = execRes
		}

		result.ArmResults["opencode"] = opencodeResults

		// Write telemetry.json for opencode
		var totalPromptB, totalCompB int64
		for _, r := range opencodeResults {
			totalPromptB += r.TotalPromptTokens
			totalCompB += r.TotalCompletionTokens
		}
		telemetryB := TelemetryTierMetrics{
			TotalPromptTokens:     totalPromptB,
			TotalCompletionTokens: totalCompB,
		}
		teleDataB, _ := json.MarshalIndent(telemetryB, "", "  ")
		_ = os.WriteFile(filepath.Join(cfg.OutDir, "opencode", "telemetry.json"), teleDataB, 0644)
	}

	return result, nil
}

func runRealCampaign(
	ctx context.Context,
	cfg RunCampaignConfig,
	contract *OfficialRunContract,
	runFak, runOpenCode bool,
	result *RunCampaignResult,
) (*RunCampaignResult, error) {
	dockerEngine := NewDockerEngine("")
	if !dockerEngine.IsAvailable(ctx) {
		return nil, errors.New("docker container runtime is not accessible; pass --mock for synthetic execution or start container daemon")
	}

	if runFak {
		if cfg.ModelPath == "" {
			return nil, errors.New("model checkpoint path (--model) is required for real Arm A execution")
		}
		result.ArmExecuted = append(result.ArmExecuted, "fak")
		fakResults := make(map[string]*ArmExecutionResult)

		adapter, err := NewInKernelModelAdapter(cfg.ModelPath, contract.Model.Sha256)
		if err != nil {
			return nil, fmt.Errorf("failed to load in-kernel model adapter: %w", err)
		}

		for _, task := range cfg.Tasks {
			taskDir := filepath.Join(cfg.OutDir, "fak", "tasks", task.TaskID)
			_ = os.MkdirAll(taskDir, 0755)

			cConfig := ContainerConfig{
				ImageDigest: task.EnvironmentImageDigest,
				Name:        "tb4-fak-" + task.TaskID,
				NetworkMode: NetworkModeNone,
				WorkingDir:  "/workspace",
			}
			inst, err := dockerEngine.CreateContainer(ctx, cConfig)
			if err != nil {
				return nil, fmt.Errorf("container creation failed: %w", err)
			}
			_ = dockerEngine.StartContainer(ctx, inst.ID)
			defer func(id string) {
				_ = dockerEngine.StopContainer(context.Background(), id, 5*time.Second)
				_ = dockerEngine.RemoveContainer(context.Background(), id, true)
			}(inst.ID)

			wsMgr := NewWorkspaceManager(dockerEngine, inst.ID, task.TaskID, "")
			_, _ = wsMgr.SeedWorkspace(ctx, task.Prompt, nil)
			if task.SetupCommand != "" {
				_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", task.SetupCommand}, 60*time.Second)
			}

			harness := NewFakHarness(adapter, nil, wsMgr)
			det := cfg.Determinism
			if det.MaxTurns <= 0 {
				det = DefaultDeterminismEnvelope()
			}
			execRes, err := harness.ExecuteTask(ctx, task, det)
			if err != nil {
				return nil, fmt.Errorf("task execution failed for %s: %w", task.TaskID, err)
			}

			_ = writeTranscriptJSONL(filepath.Join(taskDir, "transcript.jsonl"), execRes.Turns)
			resData, _ := json.MarshalIndent(execRes, "", "  ")
			_ = os.WriteFile(filepath.Join(taskDir, "result.json"), resData, 0644)
			fakResults[task.TaskID] = execRes
		}
		result.ArmResults["fak"] = fakResults
	}

	if runOpenCode {
		result.ArmExecuted = append(result.ArmExecuted, "opencode")
		opencodeResults := make(map[string]*ArmExecutionResult)

		ocAdapter := NewOpenCodeAdapter(OpenCodeConfig{
			ModelID: contract.Model.Checkpoint,
		})

		for _, task := range cfg.Tasks {
			taskDir := filepath.Join(cfg.OutDir, "opencode", "tasks", task.TaskID)
			wsDir := filepath.Join(taskDir, "workspace")
			_ = os.MkdirAll(wsDir, 0755)

			execRes, err := ocAdapter.Execute(ctx, task, wsDir)
			if err != nil {
				return nil, fmt.Errorf("opencode execution failed for %s: %w", task.TaskID, err)
			}

			_ = writeTranscriptJSONL(filepath.Join(taskDir, "transcript.jsonl"), execRes.Turns)
			resData, _ := json.MarshalIndent(execRes, "", "  ")
			_ = os.WriteFile(filepath.Join(taskDir, "result.json"), resData, 0644)
			opencodeResults[task.TaskID] = execRes
		}
		result.ArmResults["opencode"] = opencodeResults
	}

	return result, nil
}

// EvaluateCampaign grades executed workspaces using task verification oracles.
func EvaluateCampaign(ctx context.Context, cfg EvaluateCampaignConfig) (*EvaluateCampaignResult, error) {
	if cfg.RunDir == "" {
		return nil, errors.New("runDir cannot be empty")
	}

	tasks := cfg.Tasks
	if len(tasks) == 0 && cfg.Dataset != "" {
		suite, err := LoadManifestFile(cfg.Dataset)
		if err == nil {
			tasks = suite.Tasks
		}
	}
	if len(tasks) == 0 {
		// Attempt auto-discovery from manifest.json in runDir
		manifestPath := filepath.Join(cfg.RunDir, "manifest.json")
		if suite, err := LoadManifestFile(manifestPath); err == nil {
			tasks = suite.Tasks
		}
	}
	if len(tasks) == 0 {
		// Fallback to testdata synthetic suite
		for _, candidate := range []string{
			filepath.Join("testdata", "tb4bench", "synthetic_suite.json"),
			filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json"),
		} {
			if suite, err := LoadManifestFile(candidate); err == nil {
				tasks = suite.Tasks
				break
			}
		}
	}
	if len(tasks) == 0 {
		return nil, errors.New("no task manifests found; pass --dataset or ensure manifest.json exists")
	}

	taskMap := make(map[string]TaskManifest, len(tasks))
	for _, t := range tasks {
		taskMap[t.TaskID] = t
	}

	// Discover evaluated arm subdirectories
	var armsToEval []string
	for _, cand := range []string{"fak", "opencode"} {
		tasksDir := filepath.Join(cfg.RunDir, cand, "tasks")
		if info, err := os.Stat(tasksDir); err == nil && info.IsDir() {
			armsToEval = append(armsToEval, cand)
		}
	}
	if len(armsToEval) == 0 {
		// Single arm run directly under runDir
		if info, err := os.Stat(filepath.Join(cfg.RunDir, "tasks")); err == nil && info.IsDir() {
			armsToEval = append(armsToEval, filepath.Base(cfg.RunDir))
		}
	}
	if len(armsToEval) == 0 {
		return nil, fmt.Errorf("no task runs found under %s", cfg.RunDir)
	}

	grader := NewGrader()
	res := &EvaluateCampaignResult{
		Receipts:    make(map[string]map[string]*GradingReceipt),
		SolvedCount: make(map[string]int),
		TotalCount:  make(map[string]int),
		SolveRates:  make(map[string]float64),
	}

	for _, arm := range armsToEval {
		armID := arm
		if arm == "fak" {
			armID = "fak_inkernel"
		} else if arm == "opencode" {
			armID = "opencode_llamacpp"
		}

		res.Receipts[arm] = make(map[string]*GradingReceipt)
		tasksBaseDir := filepath.Join(cfg.RunDir, arm, "tasks")
		if _, err := os.Stat(tasksBaseDir); err != nil {
			tasksBaseDir = filepath.Join(cfg.RunDir, "tasks")
		}

		mockEngine := NewMockContainerEngine()

		for _, task := range tasks {
			taskDir := filepath.Join(tasksBaseDir, task.TaskID)
			if _, err := os.Stat(taskDir); err != nil {
				continue
			}

			wsDir := filepath.Join(taskDir, "workspace")
			if _, err := os.Stat(wsDir); err != nil {
				wsDir = taskDir
			}

			// Load result.json if available
			var armExec *ArmExecutionResult
			resPath := filepath.Join(taskDir, "result.json")
			if data, err := os.ReadFile(resPath); err == nil {
				var r ArmExecutionResult
				if err := json.Unmarshal(data, &r); err == nil {
					armExec = &r
				}
			}

			cConfig := ContainerConfig{
				ImageDigest: task.EnvironmentImageDigest,
				Name:        fmt.Sprintf("eval-%s-%s", arm, task.TaskID),
				NetworkMode: NetworkModeNone,
				WorkingDir:  "/workspace",
			}
			inst, err := mockEngine.CreateContainer(ctx, cConfig)
			if err != nil {
				mockEngine.Close()
				return nil, fmt.Errorf("failed to create eval container: %w", err)
			}

			mockEngine.workspaces[inst.ID] = wsDir
			wsMgr := NewWorkspaceManager(mockEngine, inst.ID, task.TaskID, wsDir)

			receipt, err := grader.Grade(ctx, armID, task, wsMgr, armExec)
			if err != nil {
				mockEngine.Close()
				return nil, fmt.Errorf("grading failed for task %s arm %s: %w", task.TaskID, arm, err)
			}

			// Persist receipt.json
			receiptPath := filepath.Join(taskDir, "receipt.json")
			if err := SaveReceipt(receipt, receiptPath); err != nil {
				mockEngine.Close()
				return nil, fmt.Errorf("failed to save receipt %s: %w", receiptPath, err)
			}

			res.Receipts[arm][task.TaskID] = receipt
			res.TotalCount[arm]++
			if receipt.Verdict == "SOLVED" {
				res.SolvedCount[arm]++
			}
		}

		mockEngine.Close()

		if res.TotalCount[arm] > 0 {
			res.SolveRates[arm] = float64(res.SolvedCount[arm]) / float64(res.TotalCount[arm])
		}
	}

	return res, nil
}

// CompareCampaign synthesizes the comparative analysis between Arm A and Arm B.
func CompareCampaign(ctx context.Context, cfg CompareCampaignConfig) (*CompareReport, error) {
	fakDir := cfg.FakDir
	opencodeDir := cfg.OpenCodeDir

	if fakDir == "" || opencodeDir == "" {
		return nil, errors.New("both FakDir and OpenCodeDir must be specified")
	}

	// Adjust paths if user pointed to the root runDir
	if _, err := os.Stat(filepath.Join(fakDir, "tasks")); os.IsNotExist(err) {
		if _, err := os.Stat(filepath.Join(fakDir, "fak", "tasks")); err == nil {
			fakDir = filepath.Join(fakDir, "fak")
		}
	}
	if _, err := os.Stat(filepath.Join(opencodeDir, "tasks")); os.IsNotExist(err) {
		if _, err := os.Stat(filepath.Join(opencodeDir, "opencode", "tasks")); err == nil {
			opencodeDir = filepath.Join(opencodeDir, "opencode")
		}
	}

	// 1. Load grading receipts
	receiptsA := make(map[string]*GradingReceipt)
	receiptsB := make(map[string]*GradingReceipt)

	filesA, err := filepath.Glob(filepath.Join(fakDir, "tasks", "*", "receipt.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to glob fak receipts: %w", err)
	}
	for _, f := range filesA {
		r, err := LoadReceipt(f)
		if err != nil {
			return nil, fmt.Errorf("failed to load receipt %s: %w", f, err)
		}
		receiptsA[r.TaskID] = r
	}

	filesB, err := filepath.Glob(filepath.Join(opencodeDir, "tasks", "*", "receipt.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to glob opencode receipts: %w", err)
	}
	for _, f := range filesB {
		r, err := LoadReceipt(f)
		if err != nil {
			return nil, fmt.Errorf("failed to load receipt %s: %w", f, err)
		}
		receiptsB[r.TaskID] = r
	}

	if len(receiptsA) == 0 && len(receiptsB) == 0 {
		return nil, fmt.Errorf("no receipts found under %s or %s; run eval first", fakDir, opencodeDir)
	}

	// 2. Load telemetry
	var teleA, teleB TelemetryTierMetrics
	teleAPath := filepath.Join(fakDir, "telemetry.json")
	if data, err := os.ReadFile(teleAPath); err == nil {
		_ = json.Unmarshal(data, &teleA)
	}
	teleBPath := filepath.Join(opencodeDir, "telemetry.json")
	if data, err := os.ReadFile(teleBPath); err == nil {
		_ = json.Unmarshal(data, &teleB)
	}

	// 3. Load contract.json
	contractPath := cfg.ContractPath
	if contractPath == "" {
		for _, cand := range []string{
			filepath.Join(fakDir, "contract.json"),
			filepath.Join(filepath.Dir(fakDir), "contract.json"),
			filepath.Join(opencodeDir, "contract.json"),
			filepath.Join(filepath.Dir(opencodeDir), "contract.json"),
		} {
			if _, err := os.Stat(cand); err == nil {
				contractPath = cand
				break
			}
		}
	}

	var contract *OfficialRunContract
	if contractPath != "" {
		contract, _ = LoadContractFile(contractPath, false)
	}
	if contract == nil {
		var taskIDs []string
		for id := range receiptsA {
			taskIDs = append(taskIDs, id)
		}
		for id := range receiptsB {
			if _, ok := receiptsA[id]; !ok {
				taskIDs = append(taskIDs, id)
			}
		}
		sort.Strings(taskIDs)
		contract = DefaultRunContract("qwen3.8-reference.gguf", "sha256:pinned", "Q4_K_M", taskIDs)
	}

	// 4. Load tasks
	tasks := cfg.Tasks
	if len(tasks) == 0 && cfg.Dataset != "" {
		if suite, err := LoadManifestFile(cfg.Dataset); err == nil {
			tasks = suite.Tasks
		}
	}
	if len(tasks) == 0 {
		for _, cand := range []string{
			filepath.Join(fakDir, "manifest.json"),
			filepath.Join(filepath.Dir(fakDir), "manifest.json"),
			filepath.Join("testdata", "tb4bench", "synthetic_suite.json"),
			filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json"),
		} {
			if suite, err := LoadManifestFile(cand); err == nil {
				tasks = suite.Tasks
				break
			}
		}
	}
	if len(tasks) == 0 {
		// Synthesize minimal manifests from available receipts
		taskIDSet := make(map[string]bool)
		for id := range receiptsA {
			taskIDSet[id] = true
		}
		for id := range receiptsB {
			taskIDSet[id] = true
		}
		var sortedIDs []string
		for id := range taskIDSet {
			sortedIDs = append(sortedIDs, id)
		}
		sort.Strings(sortedIDs)
		for _, id := range sortedIDs {
			tasks = append(tasks, TaskManifest{
				TaskID:                 id,
				Category:               CategoryRefactor,
				Prompt:                 "Task " + id,
				EnvironmentImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
				VerificationOracle:     "#!/bin/bash\nexit 0\n",
				VerificationOracleHash: "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
				TimeoutSeconds:         30,
				BudgetTurns:            5,
			})
		}
	}

	// 5. Build CompareReport
	report, err := BuildCompareReport(contract, receiptsA, receiptsB, teleA, teleB, tasks)
	if err != nil {
		return nil, fmt.Errorf("failed to build compare report: %w", err)
	}

	// 6. Save reports if paths provided
	if cfg.OutJSON != "" || cfg.OutMD != "" {
		if err := report.Save(cfg.OutJSON, cfg.OutMD); err != nil {
			return nil, fmt.Errorf("failed to save report: %w", err)
		}
	}

	return report, nil
}

func writeTranscriptJSONL(path string, turns []TurnRecord) error {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, tr := range turns {
		data, err := json.Marshal(tr)
		if err != nil {
			return err
		}
		if _, err := f.Write(append(data, '\n')); err != nil {
			return err
		}
	}
	return nil
}

func registerSyntheticScriptedResponses(adapter *InKernelModelAdapter) {
	adapter.RegisterScriptedResponse("Fix the syntax error in main.py", &CompletionResponse{
		Text: "Fixed syntax.",
		ToolCalls: []ToolCallProposal{
			{ID: "c1", Name: "write_file", Arguments: `{"path":"main.py","content":"print('fixed')\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 15 bytes to main.py", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})

	adapter.RegisterScriptedResponse("Rebase feature branch", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 100, CompletionTokens: 10,
	})

	adapter.RegisterScriptedResponse("Process log files into output.log", &CompletionResponse{
		Text: "Created log.",
		ToolCalls: []ToolCallProposal{
			{ID: "c3", Name: "write_file", Arguments: `{"path":"output.log","content":"processed logs\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 15 bytes to output.log", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})

	adapter.RegisterScriptedResponse("Configure missing PORT", &CompletionResponse{
		Text: "Configured PORT.",
		ToolCalls: []ToolCallProposal{
			{ID: "c4", Name: "write_file", Arguments: `{"path":".env","content":"PORT=8080\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 10 bytes to .env", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})

	adapter.RegisterScriptedResponse("Refactor Go package", &CompletionResponse{
		Text: "Refactored pkg.",
		ToolCalls: []ToolCallProposal{
			{ID: "c5", Name: "write_file", Arguments: `{"path":"pkg/refactored.go","content":"package pkg\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 12 bytes to pkg/refactored.go", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})
}

func applyMockSolution(ctx context.Context, wsMgr *WorkspaceManager, taskID string) {
	if strings.Contains(taskID, "syntax") {
		_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", "echo \"print('fixed')\" > main.py"}, 10*time.Second)
	} else if strings.Contains(taskID, "rebase") || strings.Contains(taskID, "git") {
		_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", "git init -q && git commit --allow-empty -m 'initial'"}, 10*time.Second)
	} else if strings.Contains(taskID, "pipeline") || strings.Contains(taskID, "log") {
		_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", "echo 'processed logs' > output.log"}, 10*time.Second)
	} else if strings.Contains(taskID, "env") {
		_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", "echo 'PORT=8080' > .env"}, 10*time.Second)
	} else if strings.Contains(taskID, "refactor") {
		_, _ = wsMgr.Exec(ctx, []string{"sh", "-c", "mkdir -p pkg && echo 'package pkg' > pkg/refactored.go"}, 10*time.Second)
	}
}
