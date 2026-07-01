package frontierswe

import (
	"fmt"
	"os/exec"
	"strings"
)

const RawFakContractSchema = "fak.frontierswe-raw-fak-contract.v1"

type RawFakContractInput struct {
	GeneratedAt    string
	Task           *Task
	Source         string
	Model          string
	Agent          string
	RawBaseURL     string
	FakBaseURL     string
	RawCommand     string
	FakCommand     string
	RawOutputDir   string
	FakOutputDir   string
	Trials         int
	EvalCapability FrontierEvalCapability
}

type RawFakContract struct {
	Schema              string                 `json:"schema"`
	GeneratedAt         string                 `json:"generated_at"`
	Benchmark           string                 `json:"benchmark"`
	Status              string                 `json:"status"`
	EvidenceClass       string                 `json:"evidence_class"`
	ClaimBoundary       string                 `json:"claim_boundary"`
	TaskSelection       ContractTaskSelection  `json:"task_selection"`
	Model               ContractModelSpec      `json:"model"`
	Budget              ContractBudgetSpec     `json:"budget"`
	Arms                []ContractArm          `json:"arms"`
	CompareEvidenceLink ContractEvidenceLink   `json:"compare_evidence_link"`
	Gates               []ContractGate         `json:"gates"`
	OfficialGrader      FrontierEvalCapability `json:"official_grader"`
	CompareMetrics      []string               `json:"compare_metrics"`
	RequiredBeforeClaim []string               `json:"required_before_claim"`
	ResultClaimAllowed  bool                   `json:"result_claim_allowed"`
}

type ContractTaskSelection struct {
	Source          string `json:"source"`
	Task            string `json:"task"`
	ScoringCategory string `json:"scoring_category"`
	DockerImage     string `json:"docker_image"`
	SameTask        bool   `json:"same_task"`
}

type ContractModelSpec struct {
	ModelID    string `json:"model_id"`
	Agent      string `json:"agent"`
	RawBaseURL string `json:"raw_base_url"`
	FakBaseURL string `json:"fak_base_url"`
	SameModel  bool   `json:"same_model"`
	SameAgent  bool   `json:"same_agent"`
}

type ContractBudgetSpec struct {
	AgentTimeoutSec int64 `json:"agent_timeout_sec"`
	Trials          int   `json:"trials"`
	SameBudget      bool  `json:"same_budget"`
}

type ContractArm struct {
	Name       string `json:"name"`
	Harness    string `json:"harness"`
	Agent      string `json:"agent"`
	Model      string `json:"model"`
	Command    string `json:"command"`
	OutputDir  string `json:"output_dir"`
	Submission string `json:"submission_artifact"`
	Meta       string `json:"meta_path"`
	TTSTrace   string `json:"tts_trace_path"`
	Reward     string `json:"reward_path"`
}

type ContractEvidenceLink struct {
	Required bool     `json:"required"`
	Artifact string   `json:"artifact"`
	JoinKeys []string `json:"join_keys"`
	Detail   string   `json:"detail"`
}

type ContractGate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

type FrontierEvalCapability struct {
	DockerPresent bool   `json:"docker_present"`
	ModalPresent  bool   `json:"modal_present"`
	Runnable      bool   `json:"runnable"`
	Reason        string `json:"reason,omitempty"`
}

func DetectFrontierEvalCapability() FrontierEvalCapability {
	cap := FrontierEvalCapability{}
	if _, err := exec.LookPath("docker"); err == nil {
		cap.DockerPresent = true
	}
	if _, err := exec.LookPath("modal"); err == nil {
		cap.ModalPresent = true
	}
	cap.Runnable = cap.DockerPresent && cap.ModalPresent
	switch {
	case !cap.DockerPresent && !cap.ModalPresent:
		cap.Reason = "no Docker and no Modal CLI; FrontierSWE official grading is a Docker/Modal-capable-box metric"
	case !cap.DockerPresent:
		cap.Reason = "Docker not found; FrontierSWE task images cannot be run locally"
	case !cap.ModalPresent:
		cap.Reason = "Modal CLI not found; official FrontierSWE sandbox/grader is not runnable here"
	}
	return cap
}

func BuildRawFakContract(in RawFakContractInput) RawFakContract {
	task := in.Task
	if task == nil {
		task = &Task{Name: ""}
	}
	model := strings.TrimSpace(in.Model)
	agent := strings.TrimSpace(in.Agent)
	if agent == "" {
		agent = "claude-code"
	}
	trials := in.Trials
	if trials <= 0 {
		trials = task.Job.NAttempts
	}
	if trials <= 0 {
		trials = 1
	}
	rawOut := defaultString(in.RawOutputDir, "experiments/frontierswe/raw-smoke")
	fakOut := defaultString(in.FakOutputDir, "experiments/frontierswe/fak-smoke")
	rawBase := strings.TrimSpace(in.RawBaseURL)
	fakBase := strings.TrimSpace(in.FakBaseURL)
	rawCommand := strings.TrimSpace(in.RawCommand)
	if rawCommand == "" {
		rawCommand = buildRawFrontierCommand(task.Name, agent, model, rawBase, rawOut, trials)
	}
	fakCommand := strings.TrimSpace(in.FakCommand)
	if fakCommand == "" {
		fakCommand = buildFakFrontierCommand(task.Name, agent, model, fakBase, fakOut, trials)
	}

	arms := []ContractArm{
		contractArm("raw-frontierswe", "frontierswe-harness-as-shipped", agent, model, rawCommand, rawOut),
		contractArm("fak-frontierswe", "frontierswe-harness-through-fak-gateway", agent, model, fakCommand, fakOut),
	}
	gates := []ContractGate{
		{Name: "fixed_task", OK: strings.TrimSpace(task.Name) != "", Detail: task.Name},
		{Name: "same_task", OK: true, Detail: "raw and fak arms run the same FrontierSWE task"},
		{Name: "same_agent", OK: agent != "", Detail: agent},
		{Name: "same_model", OK: model != "", Detail: model},
		{Name: "raw_model_endpoint", OK: rawBase != "", Detail: rawBase},
		{Name: "fak_model_endpoint", OK: fakBase != "", Detail: fakBase},
		{Name: "same_budget", OK: task.AgentTimeoutSec() > 0 && trials > 0, Detail: fmt.Sprintf("agent_timeout_sec=%d trials=%d", int64(task.AgentTimeoutSec()), trials)},
		{Name: "raw_arm_command", OK: rawCommand != "", Detail: rawCommand},
		{Name: "fak_arm_command", OK: fakCommand != "", Detail: fakCommand},
		{Name: "score_parity_gate_declared", OK: true, Detail: "fak correctness/speedup must be >= raw before any TTS claim"},
		{Name: "tts_metric_declared", OK: true, Detail: "wall-clock and turns-to-correctness plus C8 reuse trace"},
		{Name: "official_grader_local", OK: in.EvalCapability.Runnable, Detail: in.EvalCapability.Reason},
	}

	return RawFakContract{
		Schema:        RawFakContractSchema,
		GeneratedAt:   in.GeneratedAt,
		Benchmark:     "FrontierSWE",
		Status:        rawFakStatus(gates),
		EvidenceClass: "EXTERNAL_RUN_CONTRACT",
		ClaimBoundary: "Pre-run contract only: fixes one FrontierSWE task, same agent, same model, same budget, raw/fak routing, score-parity gate, TTS metric, and official grader requirements. It is not a leaderboard score or TTS result until both arms run and the official FrontierSWE scorer/grader confirms reward.json parity.",
		TaskSelection: ContractTaskSelection{
			Source: in.Source, Task: task.Name, ScoringCategory: task.ScoringCategory.String(),
			DockerImage: task.Environment.DockerImage, SameTask: true,
		},
		Model: ContractModelSpec{
			ModelID: model, Agent: agent, RawBaseURL: rawBase, FakBaseURL: fakBase,
			SameModel: model != "", SameAgent: agent != "",
		},
		Budget: ContractBudgetSpec{
			AgentTimeoutSec: int64(task.AgentTimeoutSec()), Trials: trials,
			SameBudget: task.AgentTimeoutSec() > 0 && trials > 0,
		},
		Arms:                arms,
		CompareEvidenceLink: frontierEvidenceLink(rawOut, fakOut),
		Gates:               gates,
		OfficialGrader:      in.EvalCapability,
		CompareMetrics: []string{
			"correctness",
			"speedup",
			"leaderboard_score",
			"score_parity",
			"wall_clock_to_correctness_1",
			"turns_to_correctness_1",
			"realized_reuse_rate",
			"artifact_completeness",
		},
		RequiredBeforeClaim: []string{
			"raw FrontierSWE arm writes reward.json, submission artifact, and run metadata for the fixed task",
			"fak FrontierSWE arm writes reward.json, submission artifact, run metadata, and C8 TTS/cache-witness trace for the same task",
			"official FrontierSWE scorer/grader runs over both reward.json files",
			"score-parity gate passes: fak correctness and speedup are greater than or equal to raw",
			"TTS metric is measured: wall-clock and turn count to correctness==1.0 per arm, with C8 realized reuse rate joined to the fak arm",
			"raw/fak compare artifact joins both arms by task, agent, model, trial, reward.json digest, submission digest, and TTS trace digest",
		},
		ResultClaimAllowed: false,
	}
}

func RenderRawFakContractMarkdown(c RawFakContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# FrontierSWE Raw-vs-fak Smoke Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)
	fmt.Fprintf(&b, "## Task And Budget\n\n")
	fmt.Fprintf(&b, "- Task: `%s`\n", c.TaskSelection.Task)
	fmt.Fprintf(&b, "- Scoring category: `%s`\n", c.TaskSelection.ScoringCategory)
	fmt.Fprintf(&b, "- Docker image: `%s`\n", c.TaskSelection.DockerImage)
	fmt.Fprintf(&b, "- Agent: `%s`\n", c.Model.Agent)
	fmt.Fprintf(&b, "- Model: `%s`\n", c.Model.ModelID)
	fmt.Fprintf(&b, "- Agent timeout: `%d` seconds\n", c.Budget.AgentTimeoutSec)
	fmt.Fprintf(&b, "- Trials: `%d`\n\n", c.Budget.Trials)
	renderContractArms(&b, c.Arms)
	renderContractEvidence(&b, c.CompareEvidenceLink)
	renderContractGates(&b, c.Gates)
	fmt.Fprintf(&b, "## Required Before Any Result Claim\n\n")
	for _, req := range c.RequiredBeforeClaim {
		fmt.Fprintf(&b, "- %s\n", req)
	}
	return b.String()
}

func buildRawFrontierCommand(task, agent, model, baseURL, out string, trials int) string {
	return fmt.Sprintf("frontierswe run --task %s --agent %s --model %s --base-url %s --output %s --trials %d --preds-only",
		shWord(task), shWord(agent), shWord(model), shWord(baseURL), shWord(out), trials)
}

func buildFakFrontierCommand(task, agent, model, baseURL, out string, trials int) string {
	return fmt.Sprintf("fak frontierswe run --task %s --agent %s --model %s --gateway %s --output %s --trials %d --preds-only",
		shWord(task), shWord(agent), shWord(model), shWord(baseURL), shWord(out), trials)
}

func contractArm(name, harness, agent, model, command, outputDir string) ContractArm {
	return ContractArm{
		Name: name, Harness: harness, Agent: agent, Model: model, Command: command, OutputDir: outputDir,
		Submission: joinContractPath(outputDir, "submission"), Meta: joinContractPath(outputDir, "meta.json"),
		TTSTrace: joinContractPath(outputDir, "tts-trace.jsonl"), Reward: joinContractPath(outputDir, "reward.json"),
	}
}

func frontierEvidenceLink(rawOut, fakOut string) ContractEvidenceLink {
	return ContractEvidenceLink{
		Required: true,
		Artifact: "raw-fak-frontierswe-compare.json",
		JoinKeys: []string{"task", "agent", "model", "trial", "reward_sha256", "submission_sha256", "tts_trace_sha256"},
		Detail:   "The compare artifact must join raw/fak reward.json, submission artifact, run metadata, official scorer output, and fak-arm TTS/cache-witness trace before any result claim.",
	}
}

func rawFakStatus(gates []ContractGate) string {
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

func renderContractArms(b *strings.Builder, arms []ContractArm) {
	fmt.Fprintf(b, "## Arms\n\n")
	fmt.Fprintf(b, "| Arm | Harness | Command | Output |\n")
	fmt.Fprintf(b, "|---|---|---|---|\n")
	for _, arm := range arms {
		fmt.Fprintf(b, "| `%s` | `%s` | `%s` | `%s` |\n", mdCell(arm.Name), mdCell(arm.Harness), mdCell(arm.Command), mdCell(arm.OutputDir))
	}
	fmt.Fprintln(b)
}

func renderContractEvidence(b *strings.Builder, e ContractEvidenceLink) {
	fmt.Fprintf(b, "## Compare Evidence Link\n\n")
	fmt.Fprintf(b, "- Required: `%t`\n", e.Required)
	fmt.Fprintf(b, "- Artifact: `%s`\n", e.Artifact)
	fmt.Fprintf(b, "- Join keys: `%s`\n", strings.Join(e.JoinKeys, "`, `"))
	fmt.Fprintf(b, "- Detail: %s\n\n", e.Detail)
}

func renderContractGates(b *strings.Builder, gates []ContractGate) {
	fmt.Fprintf(b, "## Gates\n\n")
	fmt.Fprintf(b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(b, "|---|---:|---|\n")
	for _, gate := range gates {
		fmt.Fprintf(b, "| `%s` | `%t` | %s |\n", mdCell(gate.Name), gate.OK, mdCell(gate.Detail))
	}
	fmt.Fprintln(b)
}

func mdCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}

func joinContractPath(dir, leaf string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return leaf
	}
	return strings.TrimRight(strings.ReplaceAll(dir, "\\", "/"), "/") + "/" + leaf
}
