package swebench

import (
	"fmt"
	"strings"
)

const DeepSWERawFakContractSchema = "fak.swebench-deepswe-raw-fak-contract.v1"

type DeepSWERawFakContractInput struct {
	GeneratedAt    string
	Dataset        *Dataset
	Source         string
	Filter         string
	Limit          int
	Model          string
	RawBaseURL     string
	FakBaseURL     string
	Adapter        string
	AdapterArgs    string
	RawCommand     string
	FakCommand     string
	RawOutputDir   string
	FakOutputDir   string
	MaxSteps       int
	Timeout        string
	MaxWorkers     int
	EvalCapability EvalCapability
}

type DeepSWERawFakContract struct {
	Schema              string             `json:"schema"`
	GeneratedAt         string             `json:"generated_at"`
	Benchmark           string             `json:"benchmark"`
	Runner              string             `json:"runner"`
	Status              string             `json:"status"`
	ClaimBoundary       string             `json:"claim_boundary"`
	TaskSelection       SmokeTaskSelection `json:"task_selection"`
	Adapter             DeepSWEAdapterSpec `json:"adapter"`
	Model               DeepSWEModelSpec   `json:"model"`
	Budget              DeepSWEBudgetSpec  `json:"budget"`
	Arms                []SmokeArm         `json:"arms"`
	Gates               []SmokeGate        `json:"gates"`
	OfficialGrader      EvalCapability     `json:"official_grader"`
	CompareMetrics      []string           `json:"compare_metrics"`
	RequiredBeforeClaim []string           `json:"required_before_claim"`
	ResultClaimAllowed  bool               `json:"result_claim_allowed"`
}

type DeepSWEAdapterSpec struct {
	Command             string   `json:"command"`
	Args                string   `json:"args,omitempty"`
	SameAdapter         bool     `json:"same_adapter"`
	RequiredEnvironment []string `json:"required_environment,omitempty"`
}

type DeepSWEModelSpec struct {
	ModelID    string `json:"model_id"`
	RawBaseURL string `json:"raw_base_url"`
	FakBaseURL string `json:"fak_base_url"`
	SameModel  bool   `json:"same_model"`
}

type DeepSWEBudgetSpec struct {
	MaxSteps   int    `json:"max_steps"`
	Timeout    string `json:"timeout"`
	SameBudget bool   `json:"same_budget"`
}

func BuildDeepSWERawFakContract(in DeepSWERawFakContractInput) DeepSWERawFakContract {
	if in.MaxWorkers <= 0 {
		in.MaxWorkers = 4
	}
	if in.MaxSteps <= 0 {
		in.MaxSteps = 50
	}
	if strings.TrimSpace(in.Timeout) == "" {
		in.Timeout = "30m"
	}
	taskIDs, dist := smokeTaskSelection(in.Dataset)
	rawPreds := joinPath(in.RawOutputDir, "predictions.json")
	fakPreds := joinPath(in.FakOutputDir, "predictions.json")
	arms := []SmokeArm{
		{
			Name:            "raw-deepswe",
			Harness:         "deepswe-r2e-gym-raw",
			Model:           in.Model,
			Command:         in.RawCommand,
			OutputDir:       in.RawOutputDir,
			PredictionsPath: rawPreds,
			EvalRunID:       "deepswe-raw-smoke",
			EvalCommand:     EvalCommandHint(rawPreds, "deepswe-raw-smoke", in.MaxWorkers),
		},
		{
			Name:            "fak-deepswe",
			Harness:         "deepswe-r2e-gym-through-fak-gateway",
			Model:           in.Model,
			Command:         in.FakCommand,
			OutputDir:       in.FakOutputDir,
			PredictionsPath: fakPreds,
			EvalRunID:       "deepswe-fak-smoke",
			EvalCommand:     EvalCommandHint(fakPreds, "deepswe-fak-smoke", in.MaxWorkers),
		},
	}
	gates := []SmokeGate{
		{Name: "fixed_task_ids", OK: len(taskIDs) > 0, Detail: fmt.Sprintf("%d task ids selected", len(taskIDs))},
		{Name: "same_task_ids", OK: true, Detail: "raw and fak arms consume the same selected task id list"},
		{Name: "same_adapter", OK: strings.TrimSpace(in.Adapter) != "", Detail: strings.TrimSpace(in.Adapter)},
		{Name: "same_model_id", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "raw_model_endpoint", OK: strings.TrimSpace(in.RawBaseURL) != "", Detail: strings.TrimSpace(in.RawBaseURL)},
		{Name: "fak_model_endpoint", OK: strings.TrimSpace(in.FakBaseURL) != "", Detail: strings.TrimSpace(in.FakBaseURL)},
		{Name: "same_budget", OK: in.MaxSteps > 0 && strings.TrimSpace(in.Timeout) != "", Detail: fmt.Sprintf("max_steps=%d timeout=%s", in.MaxSteps, in.Timeout)},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "official_grader_local", OK: in.EvalCapability.Runnable, Detail: in.EvalCapability.Reason},
	}
	requiredEnv := []string{"FAK_DEEPSWE_RUNNER", "FAK_DEEPSWE_BASE_URL", "FAK_DEEPSWE_MODEL"}
	if strings.Contains(in.RawBaseURL, "RAW_DEEPSWE_BASE_URL") {
		requiredEnv = append(requiredEnv, "RAW_DEEPSWE_BASE_URL")
	}
	return DeepSWERawFakContract{
		Schema:        DeepSWERawFakContractSchema,
		GeneratedAt:   in.GeneratedAt,
		Benchmark:     "SWE-bench Verified",
		Runner:        string(RunnerDeepSWE),
		Status:        deepSWERawFakStatus(gates),
		ClaimBoundary: "Pre-run contract only: fixes task ids, DeepSWE/R2E-Gym adapter, model id, raw/fak endpoint routing, budget, and official-grader commands. It is not a solve-rate result until both arms produce predictions and the official SWE-bench harness grades them.",
		TaskSelection: SmokeTaskSelection{
			Source:         in.Source,
			Filter:         in.Filter,
			Limit:          in.Limit,
			TaskIDs:        taskIDs,
			DifficultyDist: dist,
			SameTaskIDs:    true,
		},
		Adapter: DeepSWEAdapterSpec{
			Command:             strings.TrimSpace(in.Adapter),
			Args:                strings.TrimSpace(in.AdapterArgs),
			SameAdapter:         strings.TrimSpace(in.Adapter) != "",
			RequiredEnvironment: requiredEnv,
		},
		Model: DeepSWEModelSpec{
			ModelID:    strings.TrimSpace(in.Model),
			RawBaseURL: strings.TrimSpace(in.RawBaseURL),
			FakBaseURL: strings.TrimSpace(in.FakBaseURL),
			SameModel:  strings.TrimSpace(in.Model) != "",
		},
		Budget: DeepSWEBudgetSpec{
			MaxSteps:   in.MaxSteps,
			Timeout:    in.Timeout,
			SameBudget: in.MaxSteps > 0 && strings.TrimSpace(in.Timeout) != "",
		},
		Arms:           arms,
		Gates:          gates,
		OfficialGrader: in.EvalCapability,
		CompareMetrics: []string{
			"solve_rate",
			"safe_completion",
			"cost_or_token_budget",
			"latency",
			"tool_call_count",
			"adapter_failures",
			"policy_blocks",
			"evidence_completeness",
		},
		RequiredBeforeClaim: []string{
			"raw-deepswe predictions.json generated by a real DeepSWE/R2E-Gym adapter over the selected task ids",
			"fak-deepswe predictions.json generated by the same adapter over the same task ids through the fak gateway",
			"official SWE-bench harness report.json for both arms",
			"adapter logs or metadata showing the same model id, max steps, timeout, and task id list for both arms",
			"raw/fak compare artifact that folds solve rate, safe completion, cost or token budget, latency, adapter failures, policy blocks, and evidence completeness",
		},
		ResultClaimAllowed: false,
	}
}

func RenderDeepSWERawFakContractMarkdown(c DeepSWERawFakContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# DeepSWE Raw-vs-fak SWE-bench Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Runner: `%s`\n", c.Runner)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Task Selection\n\n")
	fmt.Fprintf(&b, "- Source: `%s`\n", c.TaskSelection.Source)
	if c.TaskSelection.Filter != "" {
		fmt.Fprintf(&b, "- Filter: `%s`\n", c.TaskSelection.Filter)
	}
	if c.TaskSelection.Limit > 0 {
		fmt.Fprintf(&b, "- Limit: `%d`\n", c.TaskSelection.Limit)
	}
	fmt.Fprintf(&b, "- Tasks: `%d`\n", len(c.TaskSelection.TaskIDs))
	fmt.Fprintf(&b, "- Same task ids: `%t`\n\n", c.TaskSelection.SameTaskIDs)

	fmt.Fprintf(&b, "## Adapter And Budget\n\n")
	fmt.Fprintf(&b, "- Adapter: `%s`\n", c.Adapter.Command)
	if c.Adapter.Args != "" {
		fmt.Fprintf(&b, "- Adapter args: `%s`\n", c.Adapter.Args)
	}
	fmt.Fprintf(&b, "- Model id: `%s`\n", c.Model.ModelID)
	fmt.Fprintf(&b, "- Raw base URL: `%s`\n", c.Model.RawBaseURL)
	fmt.Fprintf(&b, "- fak base URL: `%s`\n", c.Model.FakBaseURL)
	fmt.Fprintf(&b, "- Max steps: `%d`\n", c.Budget.MaxSteps)
	fmt.Fprintf(&b, "- Timeout: `%s`\n\n", c.Budget.Timeout)

	fmt.Fprintf(&b, "| Arm | Harness | Model | Predictions | Eval run id |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|\n")
	for _, arm := range c.Arms {
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | `%s` | `%s` |\n",
			mdCell(arm.Name), mdCell(arm.Harness), mdCell(arm.Model), mdCell(arm.PredictionsPath), mdCell(arm.EvalRunID))
	}
	fmt.Fprintf(&b, "\n## Gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|:---:|---|\n")
	for _, gate := range c.Gates {
		mark := "no"
		if gate.OK {
			mark = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", mdCell(gate.Name), mark, mdCell(gate.Detail))
	}
	fmt.Fprintf(&b, "\n## Required Before Any Result Claim\n\n")
	for _, req := range c.RequiredBeforeClaim {
		fmt.Fprintf(&b, "- %s\n", req)
	}
	return b.String()
}

func deepSWERawFakStatus(gates []SmokeGate) string {
	for _, gate := range gates {
		if gate.Name == "official_grader_local" {
			continue
		}
		if !gate.OK {
			return "INCOMPLETE_CONTRACT"
		}
	}
	return "READY_FOR_EXTERNAL_RUN"
}
