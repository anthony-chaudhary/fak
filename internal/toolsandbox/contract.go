package toolsandbox

import (
	"fmt"
	"sort"
	"strings"
)

const OfficialRunContractSchema = "fak.toolsandbox-official-run-contract.v1"

type OfficialRunContractInput struct {
	GeneratedAt          string
	Suite                Suite
	SuitePath            string
	LocalFixtureArtifact string
	OfficialHarness      string
	Domain               string
	Model                string
	UserModel            string
	Trials               int
	RawCommand           string
	FakCommand           string
	RawOutputDir         string
	FakOutputDir         string
	FakGateway           string
}

type OfficialRunContract struct {
	Schema               string                `json:"schema"`
	GeneratedAt          string                `json:"generated_at"`
	Benchmark            string                `json:"benchmark"`
	Status               string                `json:"status"`
	ClaimBoundary        string                `json:"claim_boundary"`
	LocalFixtureArtifact string                `json:"local_fixture_artifact,omitempty"`
	TaskSelection        ContractTaskSelection `json:"task_selection"`
	Model                ContractModel         `json:"model"`
	Arms                 []ContractArm         `json:"arms"`
	UpstreamRefs         []UpstreamRef         `json:"upstream_refs"`
	Gates                []ContractGate        `json:"gates"`
	CompareMetrics       []string              `json:"compare_metrics"`
	RequiredBeforeClaim  []string              `json:"required_before_claim"`
	ResultClaimAllowed   bool                  `json:"result_claim_allowed"`
}

type ContractTaskSelection struct {
	CandidateSuite          string   `json:"candidate_suite,omitempty"`
	CandidateTaskIDs        []string `json:"candidate_task_ids,omitempty"`
	OfficialHarness         string   `json:"official_harness"`
	OfficialDomain          string   `json:"official_domain,omitempty"`
	OfficialTaskIDsRequired bool     `json:"official_task_ids_required"`
	SameTaskIDsRequired     bool     `json:"same_task_ids_required"`
	SameSimulatorRequired   bool     `json:"same_simulator_required"`
	SameBudgetRequired      bool     `json:"same_budget_required"`
	Trials                  int      `json:"trials"`
}

type ContractModel struct {
	AgentModel        string `json:"agent_model"`
	UserSimulator     string `json:"user_simulator"`
	FakGateway        string `json:"fak_gateway"`
	SameModelRequired bool   `json:"same_model_required"`
}

type ContractArm struct {
	Name              string   `json:"name"`
	Harness           string   `json:"harness"`
	Command           string   `json:"command"`
	OutputDir         string   `json:"output_dir"`
	RequiredArtifacts []string `json:"required_artifacts"`
}

type UpstreamRef struct {
	Name  string `json:"name"`
	URL   string `json:"url"`
	Notes string `json:"notes"`
}

type ContractGate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

func BuildOfficialRunContract(in OfficialRunContractInput) OfficialRunContract {
	if in.OfficialHarness == "" {
		in.OfficialHarness = "tau3"
	}
	if in.Domain == "" {
		in.Domain = "retail"
	}
	if in.Model == "" {
		in.Model = "shared-agent-model"
	}
	if in.UserModel == "" {
		in.UserModel = in.Model
	}
	if in.Trials <= 0 {
		in.Trials = 1
	}
	taskIDs := contractTaskIDs(in.Suite, in.Domain)
	gates := []ContractGate{
		{Name: "candidate_task_ids", OK: len(taskIDs) > 0, Detail: candidateTaskDetail(len(taskIDs))},
		{Name: "same_task_ids_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-native task ids"},
		{Name: "same_model_required", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "same_user_simulator_required", OK: strings.TrimSpace(in.UserModel) != "", Detail: strings.TrimSpace(in.UserModel)},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "official_harness_required", OK: true, Detail: "external tau3 or Apple ToolSandbox task/grader output is required before promotion"},
	}
	return OfficialRunContract{
		Schema:               OfficialRunContractSchema,
		GeneratedAt:          in.GeneratedAt,
		Benchmark:            "ToolSandbox/tau3 policy-state official-run contract",
		Status:               contractStatus(gates),
		LocalFixtureArtifact: strings.TrimSpace(in.LocalFixtureArtifact),
		ClaimBoundary:        "External-run contract only: fixes the raw/fak command shape, shared model and simulator requirements, evidence paths, and promotion gates for a benchmark-native tau3 or ToolSandbox run. It is not an official result until external task definitions, raw/fak outputs, and native grader summaries are checked in.",
		TaskSelection: ContractTaskSelection{
			CandidateSuite:          strings.TrimSpace(in.SuitePath),
			CandidateTaskIDs:        taskIDs,
			OfficialHarness:         in.OfficialHarness,
			OfficialDomain:          in.Domain,
			OfficialTaskIDsRequired: true,
			SameTaskIDsRequired:     true,
			SameSimulatorRequired:   true,
			SameBudgetRequired:      true,
			Trials:                  in.Trials,
		},
		Model: ContractModel{
			AgentModel:        strings.TrimSpace(in.Model),
			UserSimulator:     strings.TrimSpace(in.UserModel),
			FakGateway:        strings.TrimSpace(in.FakGateway),
			SameModelRequired: true,
		},
		Arms: []ContractArm{
			{
				Name:      "raw-toolsandbox",
				Harness:   "benchmark-native",
				Command:   strings.TrimSpace(in.RawCommand),
				OutputDir: strings.TrimSpace(in.RawOutputDir),
				RequiredArtifacts: []string{
					"benchmark-native raw result summary",
					"raw trajectory or simulation log",
				},
			},
			{
				Name:      "fak-toolsandbox",
				Harness:   "benchmark-native-through-fak-gateway",
				Command:   strings.TrimSpace(in.FakCommand),
				OutputDir: strings.TrimSpace(in.FakOutputDir),
				RequiredArtifacts: []string{
					"benchmark-native fak result summary",
					"fak trajectory or simulation log",
					"fak verdict/evidence log linked to mediated tool calls",
				},
			},
		},
		UpstreamRefs: []UpstreamRef{
			{
				Name:  "Apple ToolSandbox",
				URL:   "https://github.com/apple/ToolSandbox",
				Notes: "Official ToolSandbox CLI stores result_summary.json and trajectories under the repository data directory.",
			},
			{
				Name:  "tau3/tau2-bench",
				URL:   "https://github.com/sierra-research/tau2-bench",
				Notes: "Current tau benchmark CLI documents tau2 run with domain, agent/user model, trials, task count, and outputs under data/simulations.",
			},
		},
		Gates: gates,
		CompareMetrics: []string{
			"benchmark_native_success_or_pass_k",
			"safe_completion",
			"policy_breach_rate",
			"minefield_or_mutating_action_avoidance",
			"tool_call_count",
			"latency",
			"cost_or_token_budget",
			"fak_denied_calls",
			"evidence_completeness",
		},
		RequiredBeforeClaim: []string{
			"benchmark-native task manifest or scenario ids for the selected tau3 or ToolSandbox subset",
			"raw-arm benchmark-native output over those exact task ids",
			"fak-arm benchmark-native output over those exact task ids",
			"benchmark-native grader or result summary for both arms",
			"proof that raw and fak arms used the same model, user simulator, task ids, budget, and retry policy",
			"fak verdict/evidence log linked to each mediated tool call",
			"raw/fak compare artifact reporting task success separately from policy compliance and benign utility preservation",
		},
		ResultClaimAllowed: false,
	}
}

func RenderOfficialRunContractMarkdown(c OfficialRunContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# ToolSandbox/tau3 Official-Run Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", c.ResultClaimAllowed)
	if c.LocalFixtureArtifact != "" {
		fmt.Fprintf(&b, "- Local fixture artifact: `%s`\n", c.LocalFixtureArtifact)
	}
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Task Selection\n\n")
	fmt.Fprintf(&b, "- Candidate suite: `%s`\n", c.TaskSelection.CandidateSuite)
	fmt.Fprintf(&b, "- Candidate task ids: `%s`\n", strings.Join(c.TaskSelection.CandidateTaskIDs, ", "))
	fmt.Fprintf(&b, "- Official harness: `%s`\n", c.TaskSelection.OfficialHarness)
	fmt.Fprintf(&b, "- Official domain: `%s`\n", c.TaskSelection.OfficialDomain)
	fmt.Fprintf(&b, "- Trials: `%d`\n\n", c.TaskSelection.Trials)

	fmt.Fprintf(&b, "## Arms\n\n")
	fmt.Fprintf(&b, "| Arm | Harness | Output | Command |\n")
	fmt.Fprintf(&b, "|---|---|---|---|\n")
	for _, arm := range c.Arms {
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | %s |\n", arm.Name, arm.Harness, arm.OutputDir, mdCell(arm.Command))
	}

	fmt.Fprintf(&b, "\n## Gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|:---:|---|\n")
	for _, gate := range c.Gates {
		mark := "no"
		if gate.OK {
			mark = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", gate.Name, mark, mdCell(gate.Detail))
	}

	fmt.Fprintf(&b, "\n## Required Before Any Result Claim\n\n")
	for _, req := range c.RequiredBeforeClaim {
		fmt.Fprintf(&b, "- %s\n", req)
	}
	return b.String()
}

func contractTaskIDs(s Suite, domain string) []string {
	ids := make([]string, 0, len(s.Tasks))
	domain = strings.TrimSpace(domain)
	for _, task := range s.Tasks {
		if domain != "" && task.Domain != "" && !strings.EqualFold(task.Domain, domain) {
			continue
		}
		if task.ID != "" {
			ids = append(ids, task.ID)
		}
	}
	sort.Strings(ids)
	return ids
}

func candidateTaskDetail(n int) string {
	label := "ids"
	if n == 1 {
		label = "id"
	}
	return fmt.Sprintf("%d candidate %s from local smoke suite", n, label)
}

func contractStatus(gates []ContractGate) string {
	for _, gate := range gates {
		if !gate.OK {
			return "INCOMPLETE_CONTRACT"
		}
	}
	return "READY_FOR_EXTERNAL_HARNESS"
}

func mdCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
