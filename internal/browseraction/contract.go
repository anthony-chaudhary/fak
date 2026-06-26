package browseraction

import (
	"fmt"
	"sort"
	"strings"
)

const OfficialRunContractSchema = "fak.browseraction-official-run-contract.v1"

type OfficialRunContractInput struct {
	GeneratedAt          string
	Suite                ActionMediationSuite
	SuitePath            string
	LocalFixtureArtifact string
	Harness              string
	Benchmark            string
	Model                string
	Agent                string
	FakAgent             string
	MaxSteps             int
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
	CandidateSuite           string                  `json:"candidate_suite,omitempty"`
	CandidateTaskIDs         []string                `json:"candidate_task_ids,omitempty"`
	CandidateTasks           []ContractTaskCandidate `json:"candidate_tasks,omitempty"`
	OfficialHarness          string                  `json:"official_harness"`
	OfficialBenchmark        string                  `json:"official_benchmark"`
	OfficialTaskIDsRequired  bool                    `json:"official_task_ids_required"`
	SameTaskIDsRequired      bool                    `json:"same_task_ids_required"`
	SameBrowserStateRequired bool                    `json:"same_browser_state_required"`
	SameBudgetRequired       bool                    `json:"same_budget_required"`
	MaxSteps                 int                     `json:"max_steps"`
}

type ContractTaskCandidate struct {
	ID          string `json:"id"`
	Benchmark   string `json:"benchmark,omitempty"`
	Domain      string `json:"domain,omitempty"`
	SourceURL   string `json:"source_url,omitempty"`
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
	if in.Harness == "" {
		in.Harness = "BrowserGym/AgentLab"
	}
	if in.Benchmark == "" {
		in.Benchmark = "WebArena"
	}
	if in.Model == "" {
		in.Model = "shared-agent-model"
	}
	if in.Agent == "" {
		in.Agent = "browsergym-agent"
	}
	if in.FakAgent == "" {
		in.FakAgent = in.Agent + "-through-fak"
	}
	if in.MaxSteps <= 0 {
		in.MaxSteps = 30
	}
	tasks := contractTaskCandidates(in.Suite)
	taskIDs := contractTaskIDs(tasks)
	gates := []ContractGate{
		{Name: "candidate_task_ids", OK: len(taskIDs) > 0, Detail: candidateTaskDetail(len(taskIDs))},
		{Name: "official_harness_pin", OK: strings.TrimSpace(in.Harness) != "" && strings.TrimSpace(in.Benchmark) != "", Detail: strings.TrimSpace(in.Harness) + " / " + strings.TrimSpace(in.Benchmark)},
		{Name: "same_task_ids_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-native BrowserGym task ids"},
		{Name: "same_browser_state_required", OK: true, Detail: "raw and fak official runs must use the same browser profile, website snapshot, credentials, and reset policy"},
		{Name: "same_budget_required", OK: true, Detail: "raw and fak official runs must use the same max steps, retry policy, and timeout"},
		{Name: "same_model_required", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "official_harness_required", OK: true, Detail: "external BrowserGym or AgentLab run output with benchmark-native score and trace logs is required before promotion"},
	}
	return OfficialRunContract{
		Schema:               OfficialRunContractSchema,
		GeneratedAt:          in.GeneratedAt,
		Benchmark:            "Browser/computer-use action official-run contract",
		Status:               contractStatus(gates),
		EvidenceClass:        "EXTERNAL_RUN_CONTRACT",
		LocalFixtureArtifact: strings.TrimSpace(in.LocalFixtureArtifact),
		ClaimBoundary:        "External-run contract only: fixes the raw/fak BrowserGym/WebArena-style command shape, shared model/task/browser-state/budget requirements, evidence paths, and promotion gates. It is not an official browser or computer-use benchmark result until benchmark-native task traces, score reports, and a raw-vs-fak compare artifact are checked in.",
		TaskSelection: ContractTaskSelection{
			CandidateSuite:           strings.TrimSpace(in.SuitePath),
			CandidateTaskIDs:         taskIDs,
			CandidateTasks:           tasks,
			OfficialHarness:          strings.TrimSpace(in.Harness),
			OfficialBenchmark:        strings.TrimSpace(in.Benchmark),
			OfficialTaskIDsRequired:  true,
			SameTaskIDsRequired:      true,
			SameBrowserStateRequired: true,
			SameBudgetRequired:       true,
			MaxSteps:                 in.MaxSteps,
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
				Name:      "raw-browseraction",
				Harness:   "benchmark-native",
				Command:   strings.TrimSpace(in.RawCommand),
				OutputDir: strings.TrimSpace(in.RawOutputDir),
				RequiredArtifacts: []string{
					"benchmark-native task trace or study directory",
					"benchmark-native score report",
					"browser state reset or environment manifest",
				},
			},
			{
				Name:      "fak-browseraction",
				Harness:   "benchmark-native-through-fak-gateway",
				Command:   strings.TrimSpace(in.FakCommand),
				OutputDir: strings.TrimSpace(in.FakOutputDir),
				RequiredArtifacts: []string{
					"benchmark-native task trace or study directory",
					"benchmark-native score report",
					"browser state reset or environment manifest",
					"fak action verdict/evidence log linked to mediated browser actions",
				},
			},
		},
		UpstreamRefs: []UpstreamRef{
			{
				Name:  "BrowserGym",
				URL:   "https://github.com/ServiceNow/BrowserGym",
				Notes: "Documents BrowserGym setup, benchmark packages, Gym environment ids, and WebArena/WorkArena task registration.",
			},
			{
				Name:  "WorkArena",
				URL:   "https://github.com/ServiceNow/WorkArena",
				Notes: "Documents the BrowserGym-based WorkArena setup and AgentLab as the preferred benchmark execution path.",
			},
			{
				Name:  "AgentLab",
				URL:   "https://github.com/ServiceNow/AgentLab",
				Notes: "Documents parallel BrowserGym benchmark experiments and study reload/run workflows.",
			},
			{
				Name:  "BrowserGym leaderboard",
				URL:   "https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard/blob/main/README.md",
				Notes: "Documents benchmark result JSON files for WebArena and WorkArena leaderboard submission.",
			},
		},
		Gates: gates,
		CompareMetrics: []string{
			"benchmark_native_task_success_or_score",
			"safe_success",
			"action_count",
			"invalid_action_rate",
			"policy_blocks",
			"minefield_or_mutating_action_avoidance",
			"runtime",
			"cost_or_token_budget",
			"fak_action_evidence_completeness",
		},
		RequiredBeforeClaim: []string{
			"benchmark-native BrowserGym/WebArena, WorkArena, OSWorld, or BrowseComp task ids for the selected fixed subset",
			"browser state, website snapshot, credentials, reset policy, and environment manifest for each selected task",
			"raw-arm benchmark-native trace or study directory with score report over those exact task ids",
			"fak-arm benchmark-native trace or study directory with score report over those exact task ids",
			"proof that raw and fak arms used the same model, task ids, browser state, budget, max steps, timeout, and retry policy",
			"fak action verdict/evidence log linked to the corresponding benchmark-native action trace and score report",
			"writeup separating model perception or grounding failures from harness/tool-boundary failures",
			"raw/fak compare artifact reporting task success separately from safe success, action count, invalid actions, policy blocks, runtime, cost or token budget, and evidence completeness",
		},
		ResultClaimAllowed: false,
	}
}

func RenderOfficialRunContractMarkdown(c OfficialRunContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Browser Action Official-Run Contract\n\n")
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
	fmt.Fprintf(&b, "- Official harness: `%s`\n", c.TaskSelection.OfficialHarness)
	fmt.Fprintf(&b, "- Official benchmark: `%s`\n", c.TaskSelection.OfficialBenchmark)
	fmt.Fprintf(&b, "- Max steps: `%d`\n\n", c.TaskSelection.MaxSteps)
	if len(c.TaskSelection.CandidateTasks) > 0 {
		fmt.Fprintf(&b, "| Candidate | Benchmark | Domain | Budget turns | Source |\n")
		fmt.Fprintf(&b, "|---|---|---|---:|---|\n")
		for _, task := range c.TaskSelection.CandidateTasks {
			fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | %d | `%s` |\n", task.ID, task.Benchmark, task.Domain, task.BudgetTurns, task.SourceURL)
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

func contractTaskCandidates(s ActionMediationSuite) []ContractTaskCandidate {
	out := make([]ContractTaskCandidate, 0, len(s.Tasks))
	for _, task := range s.Tasks {
		if strings.TrimSpace(task.ID) == "" {
			continue
		}
		out = append(out, ContractTaskCandidate{
			ID:          strings.TrimSpace(task.ID),
			Benchmark:   strings.TrimSpace(task.Benchmark),
			Domain:      strings.TrimSpace(task.Domain),
			SourceURL:   strings.TrimSpace(task.SourceURL),
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
	return fmt.Sprintf("%d candidate %s from local browser-action smoke suite", n, label)
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
