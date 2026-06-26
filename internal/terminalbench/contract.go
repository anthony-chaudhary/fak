package terminalbench

import (
	"fmt"
	"sort"
	"strings"
)

const OfficialRunContractSchema = "fak.terminalbench-official-run-contract.v1"

type OfficialRunContractInput struct {
	GeneratedAt          string
	Suite                Suite
	SuitePath            string
	LocalFixtureArtifact string
	DatasetName          string
	DatasetVersion       string
	Model                string
	Agent                string
	FakAgent             string
	NConcurrent          int
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
	EvidenceClass        string                `json:"evidence_class"`
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
	CandidateSuite          string                  `json:"candidate_suite,omitempty"`
	CandidateTaskIDs        []string                `json:"candidate_task_ids,omitempty"`
	CandidateTasks          []ContractTaskCandidate `json:"candidate_tasks,omitempty"`
	OfficialDataset         string                  `json:"official_dataset"`
	OfficialDatasetVersion  string                  `json:"official_dataset_version"`
	OfficialTaskIDsRequired bool                    `json:"official_task_ids_required"`
	SameTaskIDsRequired     bool                    `json:"same_task_ids_required"`
	SameImageRequired       bool                    `json:"same_image_required"`
	SameBudgetRequired      bool                    `json:"same_budget_required"`
	NConcurrent             int                     `json:"n_concurrent"`
}

type ContractTaskCandidate struct {
	ID          string `json:"id"`
	Image       string `json:"image,omitempty"`
	TestOracle  string `json:"test_oracle,omitempty"`
	BudgetTurns int    `json:"budget_turns,omitempty"`
}

type ContractModel struct {
	Agent             string `json:"agent"`
	FakAgent          string `json:"fak_agent"`
	Model             string `json:"model"`
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
	if in.DatasetName == "" {
		in.DatasetName = "terminal-bench-core"
	}
	if in.DatasetVersion == "" {
		in.DatasetVersion = "0.1.1"
	}
	if in.Model == "" {
		in.Model = "shared-agent-model"
	}
	if in.Agent == "" {
		in.Agent = "terminus"
	}
	if in.FakAgent == "" {
		in.FakAgent = in.Agent + "-through-fak"
	}
	if in.NConcurrent <= 0 {
		in.NConcurrent = 1
	}
	tasks := contractTaskCandidates(in.Suite)
	taskIDs := contractTaskIDs(tasks)
	gates := []ContractGate{
		{Name: "candidate_task_ids", OK: len(taskIDs) > 0, Detail: candidateTaskDetail(len(taskIDs))},
		{Name: "official_dataset_pin", OK: strings.TrimSpace(in.DatasetName) != "" && strings.TrimSpace(in.DatasetVersion) != "", Detail: datasetPin(in.DatasetName, in.DatasetVersion)},
		{Name: "same_task_ids_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-native Terminal-Bench task ids"},
		{Name: "same_image_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-provided image or environment setup for each task"},
		{Name: "same_budget_required", OK: true, Detail: "raw and fak official runs must use the same task budget and retry policy"},
		{Name: "same_model_required", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "official_harness_required", OK: true, Detail: "external tb run output with benchmark-native test results is required before promotion"},
	}
	return OfficialRunContract{
		Schema:               OfficialRunContractSchema,
		GeneratedAt:          in.GeneratedAt,
		Benchmark:            "Terminal-Bench command-boundary official-run contract",
		Status:               contractStatus(gates),
		EvidenceClass:        "EXTERNAL_RUN_CONTRACT",
		LocalFixtureArtifact: strings.TrimSpace(in.LocalFixtureArtifact),
		ClaimBoundary:        "External-run contract only: fixes the raw/fak Terminal-Bench command shape, shared task/model/image/budget requirements, evidence paths, and promotion gates. It is not an official result until benchmark-native tb run task logs, test output, and a raw-vs-fak compare artifact are checked in.",
		TaskSelection: ContractTaskSelection{
			CandidateSuite:          strings.TrimSpace(in.SuitePath),
			CandidateTaskIDs:        taskIDs,
			CandidateTasks:          tasks,
			OfficialDataset:         strings.TrimSpace(in.DatasetName),
			OfficialDatasetVersion:  strings.TrimSpace(in.DatasetVersion),
			OfficialTaskIDsRequired: true,
			SameTaskIDsRequired:     true,
			SameImageRequired:       true,
			SameBudgetRequired:      true,
			NConcurrent:             in.NConcurrent,
		},
		Model: ContractModel{
			Agent:             strings.TrimSpace(in.Agent),
			FakAgent:          strings.TrimSpace(in.FakAgent),
			Model:             strings.TrimSpace(in.Model),
			FakGateway:        strings.TrimSpace(in.FakGateway),
			SameModelRequired: true,
		},
		Arms: []ContractArm{
			{
				Name:      "raw-terminalbench",
				Harness:   "benchmark-native",
				Command:   strings.TrimSpace(in.RawCommand),
				OutputDir: strings.TrimSpace(in.RawOutputDir),
				RequiredArtifacts: []string{
					"tb run directory for each selected task",
					"benchmark-native command log",
					"benchmark-native test output or result summary",
				},
			},
			{
				Name:      "fak-terminalbench",
				Harness:   "benchmark-native-through-fak-gateway",
				Command:   strings.TrimSpace(in.FakCommand),
				OutputDir: strings.TrimSpace(in.FakOutputDir),
				RequiredArtifacts: []string{
					"tb run directory for each selected task",
					"benchmark-native command log",
					"benchmark-native test output or result summary",
					"fak verdict/evidence log linked to mediated terminal commands",
				},
			},
		},
		UpstreamRefs: []UpstreamRef{
			{
				Name:  "Terminal-Bench first steps",
				URL:   "https://www.tbench.ai/docs/first-steps",
				Notes: "Documents tb run with dataset, agent, model, and task-id selection.",
			},
			{
				Name:  "Terminal-Bench adapters",
				URL:   "https://www.tbench.ai/docs/adapters",
				Notes: "Documents benchmark-native task subsets and parity experiments through the Terminal-Bench harness.",
			},
			{
				Name:  "Terminal-Bench leaderboard logs",
				URL:   "https://github.com/abacusai/abacusai-terminal-bench-leaderboard",
				Notes: "Shows the expected runs directory convention for submitted Terminal-Bench results.",
			},
			{
				Name:  "Terminal-Bench 3 Harbor preview",
				URL:   "https://github.com/harbor-framework/terminal-bench-3",
				Notes: "Tracks the Harbor-based next-generation harness shape for future migration.",
			},
		},
		Gates: gates,
		CompareMetrics: []string{
			"benchmark_native_test_success_or_pass_1",
			"safe_resolve",
			"blocked_dangerous_actions",
			"unnecessary_blocks",
			"command_count",
			"runtime",
			"cost_or_token_budget",
			"fak_denied_commands",
			"fak_verdict_evidence_completeness",
		},
		RequiredBeforeClaim: []string{
			"benchmark-native Terminal-Bench task ids for the selected fixed subset",
			"Terminal-Bench image or environment setup manifest for each selected task",
			"raw-arm tb run directory with command log and official test output over those exact task ids",
			"fak-arm tb run directory with command log and official test output over those exact task ids",
			"proof that raw and fak arms used the same model, task ids, image or environment, budget, concurrency, and retry policy",
			"fak per-command verdict/evidence log linked to the corresponding tb run command log and test output",
			"raw/fak compare artifact reporting benchmark-native solve separately from safe resolve, blocked dangerous actions, unnecessary blocks, runtime, cost or token budget, and evidence completeness",
		},
		ResultClaimAllowed: false,
	}
}

func RenderOfficialRunContractMarkdown(c OfficialRunContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Terminal-Bench Official-Run Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", c.ResultClaimAllowed)
	if c.LocalFixtureArtifact != "" {
		fmt.Fprintf(&b, "- Local fixture artifact: `%s`\n", c.LocalFixtureArtifact)
	}
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Task Selection\n\n")
	fmt.Fprintf(&b, "- Candidate suite: `%s`\n", c.TaskSelection.CandidateSuite)
	fmt.Fprintf(&b, "- Candidate task ids: `%s`\n", strings.Join(c.TaskSelection.CandidateTaskIDs, ", "))
	fmt.Fprintf(&b, "- Official dataset: `%s==%s`\n", c.TaskSelection.OfficialDataset, c.TaskSelection.OfficialDatasetVersion)
	fmt.Fprintf(&b, "- Concurrent tasks: `%d`\n\n", c.TaskSelection.NConcurrent)
	if len(c.TaskSelection.CandidateTasks) > 0 {
		fmt.Fprintf(&b, "| Candidate | Image | Budget turns | Test oracle |\n")
		fmt.Fprintf(&b, "|---|---|---:|---|\n")
		for _, task := range c.TaskSelection.CandidateTasks {
			fmt.Fprintf(&b, "| `%s` | `%s` | %d | `%s` |\n", task.ID, task.Image, task.BudgetTurns, task.TestOracle)
		}
		fmt.Fprintf(&b, "\n")
	}

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

func contractTaskCandidates(s Suite) []ContractTaskCandidate {
	out := make([]ContractTaskCandidate, 0, len(s.Tasks))
	for _, task := range s.Tasks {
		if strings.TrimSpace(task.ID) == "" {
			continue
		}
		out = append(out, ContractTaskCandidate{
			ID:          strings.TrimSpace(task.ID),
			Image:       strings.TrimSpace(task.Image),
			TestOracle:  strings.TrimSpace(task.TestOracle),
			BudgetTurns: task.BudgetTurns,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func contractTaskIDs(tasks []ContractTaskCandidate) []string {
	ids := make([]string, 0, len(tasks))
	for _, task := range tasks {
		ids = append(ids, task.ID)
	}
	return ids
}

func candidateTaskDetail(n int) string {
	label := "ids"
	if n == 1 {
		label = "id"
	}
	return fmt.Sprintf("%d candidate %s from local Terminal-Bench-shaped smoke suite", n, label)
}

func datasetPin(name, version string) string {
	name = strings.TrimSpace(name)
	version = strings.TrimSpace(version)
	if version == "" {
		return name
	}
	return name + "==" + version
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
