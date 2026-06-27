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
	Schema              string                     `json:"schema"`
	GeneratedAt         string                     `json:"generated_at"`
	Benchmark           string                     `json:"benchmark"`
	Runner              string                     `json:"runner"`
	Status              string                     `json:"status"`
	EvidenceClass       string                     `json:"evidence_class"`
	ClaimBoundary       string                     `json:"claim_boundary"`
	TaskSelection       SmokeTaskSelection         `json:"task_selection"`
	Adapter             DeepSWEAdapterSpec         `json:"adapter"`
	Model               DeepSWEModelSpec           `json:"model"`
	Budget              DeepSWEBudgetSpec          `json:"budget"`
	Arms                []SmokeArm                 `json:"arms"`
	CompareEvidenceLink DeepSWECompareEvidenceLink `json:"compare_evidence_link"`
	Gates               []SmokeGate                `json:"gates"`
	OfficialGrader      EvalCapability             `json:"official_grader"`
	CompareMetrics      []string                   `json:"compare_metrics"`
	RequiredBeforeClaim []string                   `json:"required_before_claim"`
	ResultClaimAllowed  bool                       `json:"result_claim_allowed"`
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

type DeepSWECompareEvidenceLink struct {
	Required     bool     `json:"required"`
	Predictions  []string `json:"predictions"`
	Metadata     []string `json:"metadata"`
	OfficialEval []string `json:"official_eval"`
	FakEvidence  []string `json:"fak_evidence"`
	JoinKeys     []string `json:"join_keys"`
	Detail       string   `json:"detail"`
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
	arms := buildTwoArms(in.Model, in.RawCommand, in.FakCommand, in.RawOutputDir, in.FakOutputDir, rawPreds, fakPreds, in.MaxWorkers, twoArmNames{
		RawName:    "raw-deepswe",
		RawHarness: "deepswe-r2e-gym-raw",
		RawEvalID:  "deepswe-raw-smoke",
		FakName:    "fak-deepswe",
		FakHarness: "deepswe-r2e-gym-through-fak-gateway",
		FakEvalID:  "deepswe-fak-smoke",
	})
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
		EvidenceClass: "EXTERNAL_RUN_CONTRACT",
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
		Arms:                arms,
		CompareEvidenceLink: deepSWECompareEvidenceLink(in.RawOutputDir, in.FakOutputDir, rawPreds, fakPreds),
		Gates:               gates,
		OfficialGrader:      in.EvalCapability,
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
	if c.EvidenceClass != "" {
		fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	}
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

	renderSmokeArmsTable(&b, c.Arms)
	fmt.Fprintf(&b, "\n## Compare Evidence Link\n\n")
	fmt.Fprintf(&b, "- Required: `%t`\n", c.CompareEvidenceLink.Required)
	fmt.Fprintf(&b, "- Predictions: `%s`\n", strings.Join(c.CompareEvidenceLink.Predictions, "`, `"))
	fmt.Fprintf(&b, "- Metadata: `%s`\n", strings.Join(c.CompareEvidenceLink.Metadata, "`, `"))
	fmt.Fprintf(&b, "- Official eval: `%s`\n", strings.Join(c.CompareEvidenceLink.OfficialEval, "`, `"))
	fmt.Fprintf(&b, "- fak evidence: `%s`\n", strings.Join(c.CompareEvidenceLink.FakEvidence, "`, `"))
	fmt.Fprintf(&b, "- Join keys: `%s`\n", strings.Join(c.CompareEvidenceLink.JoinKeys, "`, `"))
	fmt.Fprintf(&b, "- Detail: %s\n", c.CompareEvidenceLink.Detail)
	fmt.Fprintf(&b, "\n")
	renderSmokeGatesTable(&b, c.Gates)
	renderRequiredBeforeClaim(&b, c.RequiredBeforeClaim)
	return b.String()
}

func deepSWECompareEvidenceLink(rawOutputDir, fakOutputDir, rawPreds, fakPreds string) DeepSWECompareEvidenceLink {
	return DeepSWECompareEvidenceLink{
		Required: true,
		Predictions: []string{
			rawPreds,
			fakPreds,
		},
		Metadata: []string{
			joinPath(rawOutputDir, "meta.json"),
			joinPath(fakOutputDir, "meta.json"),
		},
		OfficialEval: []string{
			joinPath(rawOutputDir, "eval.json"),
			joinPath(fakOutputDir, "eval.json"),
		},
		FakEvidence: []string{
			joinPath(fakOutputDir, "fak-adjudication-evidence.jsonl"),
			joinPath(fakOutputDir, "raw-fak-deepswe-compare.json"),
		},
		JoinKeys: []string{
			"instance_id",
			"runner",
			"model",
			"prediction_sha256",
			"evidence_id",
		},
		Detail: "The raw/fak compare artifact must join each SWE-bench prediction and official grader row to the fak-arm adjudication evidence for the same instance id.",
	}
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
