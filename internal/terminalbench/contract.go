package terminalbench

import (
	"fmt"
	"sort"
	"strings"
)

const (
	OfficialRunContractSchema = "fak.terminalbench-official-run-contract.v1"

	OfficialTerminalBench21Dataset       = "terminal-bench/terminal-bench-2-1"
	OfficialTerminalBench21TopAgentModel = "gpt-5.5"
)

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
	PublicAgentLabel     string
	NConcurrent          int
	RawCommand           string
	FakCommand           string
	RawOutputDir         string
	FakOutputDir         string
	FakGateway           string
}

type OfficialRunContract struct {
	Schema               string                    `json:"schema"`
	GeneratedAt          string                    `json:"generated_at"`
	Benchmark            string                    `json:"benchmark"`
	Status               string                    `json:"status"`
	EvidenceClass        string                    `json:"evidence_class"`
	ClaimBoundary        string                    `json:"claim_boundary"`
	LocalFixtureArtifact string                    `json:"local_fixture_artifact,omitempty"`
	TaskSelection        ContractTaskSelection     `json:"task_selection"`
	Model                ContractModel             `json:"model"`
	Arms                 []ContractArm             `json:"arms"`
	ScoreEvidenceLink    ContractScoreEvidenceLink `json:"score_evidence_link"`
	UpstreamRefs         []UpstreamRef             `json:"upstream_refs"`
	Gates                []ContractGate            `json:"gates"`
	CompareMetrics       []string                  `json:"compare_metrics"`
	RequiredBeforeClaim  []string                  `json:"required_before_claim"`
	ResultClaimAllowed   bool                      `json:"result_claim_allowed"`
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
	Agent               string `json:"agent"`
	FakAgent            string `json:"fak_agent"`
	PublicAgentLabel    string `json:"public_agent_label,omitempty"`
	Model               string `json:"model"`
	TopAgentModel       string `json:"top_agent_model,omitempty"`
	FakGateway          string `json:"fak_gateway"`
	SameAgentRequired   bool   `json:"same_agent_required"`
	SameModelRequired   bool   `json:"same_model_required"`
	HarborCodexRequired bool   `json:"harbor_codex_required"`
}

type ContractArm struct {
	Name              string   `json:"name"`
	Harness           string   `json:"harness"`
	Command           string   `json:"command"`
	OutputDir         string   `json:"output_dir"`
	RequiredArtifacts []string `json:"required_artifacts"`
}

type ContractScoreEvidenceLink struct {
	Required                bool     `json:"required"`
	OfficialTestArtifacts   []string `json:"official_test_artifacts"`
	FakCommandEvidenceFiles []string `json:"fak_command_evidence_files"`
	JoinKeys                []string `json:"join_keys"`
	Detail                  string   `json:"detail"`
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
		in.DatasetName = OfficialTerminalBench21Dataset
	}
	if in.Model == "" {
		in.Model = OfficialTerminalBench21TopAgentModel
	}
	if in.Agent == "" {
		in.Agent = "codex"
	}
	if in.FakAgent == "" {
		in.FakAgent = in.Agent
	}
	if in.PublicAgentLabel == "" && strings.TrimSpace(in.Agent) == "codex" {
		in.PublicAgentLabel = "codex-cli"
	}
	if in.NConcurrent <= 0 {
		in.NConcurrent = 1
	}
	tasks := contractTaskCandidates(in.Suite)
	taskIDs := contractTaskIDs(tasks)
	gates := []ContractGate{
		{Name: "candidate_task_ids", OK: len(taskIDs) > 0, Detail: candidateTaskDetail(len(taskIDs))},
		{Name: "official_dataset_pin", OK: officialDatasetPinReady(in.DatasetName, in.DatasetVersion), Detail: datasetPin(in.DatasetName, in.DatasetVersion)},
		{Name: "same_task_ids_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-native Terminal-Bench task ids"},
		{Name: "same_image_required", OK: true, Detail: "raw and fak official runs must use the same benchmark-provided image or environment setup for each task"},
		{Name: "same_budget_required", OK: true, Detail: "raw and fak official runs must use the same task budget and retry policy"},
		{Name: "same_agent_required", OK: strings.TrimSpace(in.Agent) == strings.TrimSpace(in.FakAgent), Detail: fmt.Sprintf("raw=%s fak=%s", strings.TrimSpace(in.Agent), strings.TrimSpace(in.FakAgent))},
		{Name: "same_model_required", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "top_agent_model_current", OK: strings.TrimSpace(in.Model) == OfficialTerminalBench21TopAgentModel, Detail: OfficialTerminalBench21TopAgentModel},
		{Name: "harbor_codex_adapter", OK: strings.TrimSpace(in.Agent) == "codex" && strings.TrimSpace(in.FakAgent) == "codex", Detail: "Harbor adapter name must be codex; codex-cli is only the public leaderboard label"},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "raw_arm_official_dataset", OK: harborCommandDatasetReady(in.RawCommand), Detail: "raw Harbor arm must pass -d/--dataset terminal-bench/terminal-bench-2-1 with no dataset version"},
		{Name: "fak_arm_official_dataset", OK: harborCommandDatasetReady(in.FakCommand), Detail: "fak Harbor arm must pass -d/--dataset terminal-bench/terminal-bench-2-1 with no dataset version"},
		{Name: "fak_gateway_agent_env", OK: fakGatewayAgentEnvReady(in.FakCommand), Detail: "fak Harbor arm must pass OPENAI_BASE_URL, OPENAI_API_BASE, and OPENAI_API_KEY through --agent-env"},
		{Name: "fak_gateway_host_allowlist", OK: strings.Contains(in.FakCommand, "--allow-agent-host"), Detail: "fak Harbor arm must allow the Docker agent to reach the host fak gateway"},
		{Name: "official_harness_required", OK: true, Detail: "external Harbor run output with benchmark-native test results is required before promotion"},
	}
	return OfficialRunContract{
		Schema:               OfficialRunContractSchema,
		GeneratedAt:          in.GeneratedAt,
		Benchmark:            "Terminal-Bench Harbor/Codex official-run contract",
		Status:               contractStatus(gates),
		EvidenceClass:        "EXTERNAL_RUN_CONTRACT",
		LocalFixtureArtifact: strings.TrimSpace(in.LocalFixtureArtifact),
		ClaimBoundary:        "External-run contract only: fixes the raw/fak Harbor command shape, shared task/model/image/budget requirements, fak gateway adapter wiring, evidence paths, and promotion gates. It is not an official result until benchmark-native Harbor task logs, test output, gateway witness, and a raw-vs-fak compare artifact are checked in.",
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
			Agent:               strings.TrimSpace(in.Agent),
			FakAgent:            strings.TrimSpace(in.FakAgent),
			PublicAgentLabel:    strings.TrimSpace(in.PublicAgentLabel),
			Model:               strings.TrimSpace(in.Model),
			TopAgentModel:       OfficialTerminalBench21TopAgentModel,
			FakGateway:          strings.TrimSpace(in.FakGateway),
			SameAgentRequired:   true,
			SameModelRequired:   true,
			HarborCodexRequired: true,
		},
		Arms: []ContractArm{
			{
				Name:      "raw-terminalbench",
				Harness:   "benchmark-native",
				Command:   strings.TrimSpace(in.RawCommand),
				OutputDir: strings.TrimSpace(in.RawOutputDir),
				RequiredArtifacts: []string{
					"Harbor run directory for each selected task",
					"Harbor command log",
					"Harbor test output or result summary",
				},
			},
			{
				Name:      "fak-terminalbench",
				Harness:   "benchmark-native-through-fak-gateway",
				Command:   strings.TrimSpace(in.FakCommand),
				OutputDir: strings.TrimSpace(in.FakOutputDir),
				RequiredArtifacts: []string{
					"Harbor run directory for each selected task",
					"Harbor command log",
					"Harbor test output or result summary",
					"fak verdict/evidence log linked to mediated terminal commands",
					"fak gateway log witness proving Dockerized Codex traffic reached fak",
				},
			},
		},
		ScoreEvidenceLink: terminalBenchScoreEvidenceLink(in.RawOutputDir, in.FakOutputDir),
		UpstreamRefs: []UpstreamRef{
			{
				Name:  "Harbor job CLI",
				URL:   "https://github.com/harbor-framework/harbor/blob/main/src/harbor/cli/jobs.py",
				Notes: "Defines harbor run/job start flags including dataset, agent, model, --agent-env, --allow-agent-host, and --allow-environment-host.",
			},
			{
				Name:  "Harbor agent names",
				URL:   "https://github.com/harbor-framework/harbor/blob/main/src/harbor/models/agent/name.py",
				Notes: "Defines codex as the valid Harbor adapter name; codex-cli is the public leaderboard label only.",
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
			"fak_gateway_model_http_success",
			"fak_gateway_inference_turn_observed",
		},
		RequiredBeforeClaim: []string{
			"benchmark-native Terminal-Bench task ids for the selected fixed subset",
			"Terminal-Bench image or environment setup manifest for each selected task",
			"raw-arm Harbor run directory with command log and official test output over those exact task ids",
			"fak-arm Harbor run directory with command log and official test output over those exact task ids",
			"proof that raw and fak arms used the same Harbor codex adapter, model, task ids, image or environment, budget, concurrency, and retry policy",
			"fak per-command verdict/evidence log linked to the corresponding Harbor command log and test output",
			"gateway witness proving at least one structured model HTTP success and one gateway inference-turn event from the Dockerized Codex agent",
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
	fmt.Fprintf(&b, "- Official dataset: `%s`\n", datasetPin(c.TaskSelection.OfficialDataset, c.TaskSelection.OfficialDatasetVersion))
	fmt.Fprintf(&b, "- Concurrent tasks: `%d`\n", c.TaskSelection.NConcurrent)
	fmt.Fprintf(&b, "- Shared model: `%s`\n", c.Model.Model)
	if c.Model.TopAgentModel != "" {
		fmt.Fprintf(&b, "- Terminal-Bench 2.1 top-agent model: `%s`\n", c.Model.TopAgentModel)
	}
	if c.Model.PublicAgentLabel != "" {
		fmt.Fprintf(&b, "- Public agent label: `%s`\n", c.Model.PublicAgentLabel)
	}
	fmt.Fprintf(&b, "\n")
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

	fmt.Fprintf(&b, "\n## Score Evidence Link\n\n")
	fmt.Fprintf(&b, "- Required: `%t`\n", c.ScoreEvidenceLink.Required)
	fmt.Fprintf(&b, "- Official test artifacts: `%s`\n", strings.Join(c.ScoreEvidenceLink.OfficialTestArtifacts, "`, `"))
	fmt.Fprintf(&b, "- fak command evidence files: `%s`\n", strings.Join(c.ScoreEvidenceLink.FakCommandEvidenceFiles, "`, `"))
	fmt.Fprintf(&b, "- Join keys: `%s`\n", strings.Join(c.ScoreEvidenceLink.JoinKeys, "`, `"))
	fmt.Fprintf(&b, "- Detail: %s\n", c.ScoreEvidenceLink.Detail)

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

func terminalBenchScoreEvidenceLink(rawOutputDir, fakOutputDir string) ContractScoreEvidenceLink {
	return ContractScoreEvidenceLink{
		Required: true,
		OfficialTestArtifacts: []string{
			joinArtifactPath(rawOutputDir, "harbor-test-results.json"),
			joinArtifactPath(rawOutputDir, "harbor-command-log.jsonl"),
			joinArtifactPath(fakOutputDir, "harbor-test-results.json"),
			joinArtifactPath(fakOutputDir, "harbor-command-log.jsonl"),
		},
		FakCommandEvidenceFiles: []string{
			joinArtifactPath(fakOutputDir, "fak-command-evidence.jsonl"),
			joinArtifactPath(fakOutputDir, "raw-fak-command-join.json"),
			joinArtifactPath(fakOutputDir, "fak-gateway-witness.json"),
		},
		JoinKeys: []string{
			"task_id",
			"command_index",
			"command",
			"cwd",
			"evidence_id",
			"state_hash",
		},
		Detail: "The official compare artifact must join Harbor test/pass rows and command logs to the mediated fak command verdict and evidence checkpoint for the same task command.",
	}
}

func fakGatewayAgentEnvReady(command string) bool {
	return strings.Contains(command, "--agent-env") &&
		strings.Contains(command, "OPENAI_BASE_URL=") &&
		strings.Contains(command, "OPENAI_API_BASE=") &&
		strings.Contains(command, "OPENAI_API_KEY=")
}

func joinArtifactPath(dir, leaf string) string {
	dir = strings.TrimRight(strings.TrimSpace(dir), "/\\")
	if dir == "" {
		return leaf
	}
	return dir + "/" + leaf
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

func officialDatasetPinReady(name, version string) bool {
	return strings.TrimSpace(name) == OfficialTerminalBench21Dataset && strings.TrimSpace(version) == ""
}

func harborCommandDatasetReady(command string) bool {
	fields := strings.Fields(command)
	sawDataset := false
	for i, field := range fields {
		field = strings.Trim(field, `"'`)
		switch {
		case field == "-d" || field == "--dataset":
			sawDataset = true
			if i+1 >= len(fields) || strings.Trim(fields[i+1], `"'`) != OfficialTerminalBench21Dataset {
				return false
			}
		case strings.HasPrefix(field, "-d="):
			sawDataset = true
			if strings.Trim(strings.TrimPrefix(field, "-d="), `"'`) != OfficialTerminalBench21Dataset {
				return false
			}
		case strings.HasPrefix(field, "--dataset="):
			sawDataset = true
			if strings.Trim(strings.TrimPrefix(field, "--dataset="), `"'`) != OfficialTerminalBench21Dataset {
				return false
			}
		}
	}
	return sawDataset
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
