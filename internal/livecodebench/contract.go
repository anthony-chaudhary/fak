package livecodebench

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// OfficialRunContractSchema identifies the LiveCodeBench official-run contract:
// a machine-readable statement of the two-arm run shape (raw lcb_runner vs
// fak-native), the constants pinned before generating anything, the exact
// official grading handoff, and the gates that must be green before any
// pass-rate is claimable. It performs no run and carries no score — its whole
// job is to make the run reproducible and to keep result_claim_allowed=false
// until the official evaluator has graded the exact saved generations. Part of
// the LiveCodeBench epic #2085 (#2110), gating the DGX GLM-5.2 run #3060.
const OfficialRunContractSchema = "fak.livecodebench-official-run-contract.v1"

// Closed-vocabulary contract status. It describes whether the run is fully
// pinned, never whether it passed.
const (
	ContractIncomplete = "INCOMPLETE_CONTRACT"
	ContractReady      = "READY_FOR_OFFICIAL_RUN"
)

// OfficialRunContractInput carries the campaign coordinates the contract pins.
// Suite is optional: when supplied, its question_ids for the chosen scenario
// become the candidate problem selection so raw and fak arms provably score the
// same problems. No host/channel/token/URL belongs in here — the gateway is a
// local base URL only; secrets stay in gitignored config (#3059).
type OfficialRunContractInput struct {
	GeneratedAt     string
	Issue           string
	Suite           *Suite
	SuitePath       string
	ReleaseSelector string
	Scenario        Scenario
	StartDate       string
	EndDate         string
	Model           string
	ServingBackend  string
	Gateway         string
	RunDir          string
}

// OfficialRunContract is the result-claim-gated run contract. It mirrors the
// terminalbench official-run contract shape but is LiveCodeBench-native: two
// arms plus a single official grading authority.
type OfficialRunContract struct {
	Schema              string                   `json:"schema"`
	GeneratedAt         string                   `json:"generated_at,omitempty"`
	Issue               string                   `json:"issue,omitempty"`
	Benchmark           string                   `json:"benchmark"`
	Status              string                   `json:"status"`
	EvidenceClass       string                   `json:"evidence_class"`
	ClaimBoundary       string                   `json:"claim_boundary"`
	Constants           ContractConstants        `json:"constants"`
	ProblemSelection    ContractProblemSelection `json:"problem_selection"`
	Arms                []ContractArm            `json:"arms"`
	Grading             ContractGrading          `json:"grading"`
	UpstreamRefs        []ContractUpstreamRef    `json:"upstream_refs"`
	Gates               []ContractGate           `json:"gates"`
	CompareMetrics      []string                 `json:"compare_metrics"`
	RequiredBeforeClaim []string                 `json:"required_before_claim"`
	ResultClaimAllowed  bool                     `json:"result_claim_allowed"`
}

// ContractConstants are the run identities that must be pinned before any
// generation and kept identical across both arms (#3060 "constants pinned").
type ContractConstants struct {
	ReleaseSelector string `json:"release_selector"`
	ReleaseVersion  string `json:"release_version"`
	Scenario        string `json:"scenario"`
	StartDate       string `json:"start_date"`
	EndDate         string `json:"end_date"`
	Model           string `json:"model"`
	ServingBackend  string `json:"serving_backend,omitempty"`
	Gateway         string `json:"gateway,omitempty"`
	RunDir          string `json:"run_dir"`
}

// ContractProblemSelection fixes which problems both arms score and the
// cross-arm identity requirements (#3060 SameProblemIDs / SamePromptHash).
type ContractProblemSelection struct {
	CandidateSuite         string   `json:"candidate_suite,omitempty"`
	CandidateProblemIDs    []string `json:"candidate_problem_ids,omitempty"`
	SameProblemIDsRequired bool     `json:"same_problem_ids_required"`
	SamePromptHashRequired bool     `json:"same_prompt_hash_required"`
	SameReleaseRequired    bool     `json:"same_release_required"`
	SameModelRequired      bool     `json:"same_model_required"`
	SameScenarioRequired   bool     `json:"same_scenario_required"`
}

// ContractArm is one generation arm: the raw official lcb_runner arm or the
// fak-native arm. Neither arm's command grades — grading is the single
// official authority in ContractGrading.
type ContractArm struct {
	Name              string   `json:"name"`
	Harness           string   `json:"harness"`
	Commands          []string `json:"commands"`
	OutputDir         string   `json:"output_dir"`
	RequiredArtifacts []string `json:"required_artifacts"`
	Notes             string   `json:"notes,omitempty"`
}

// ContractGrading is the sole result-bearing authority: the official
// LiveCodeBench evaluator over the exact saved generations from each arm.
type ContractGrading struct {
	Authority              string   `json:"authority"`
	RawEvaluateCommand     string   `json:"raw_evaluate_command"`
	CustomEvaluatorCommand string   `json:"custom_evaluator_command"`
	ComputeScoresCommand   string   `json:"compute_scores_command"`
	RequiredArtifacts      []string `json:"required_artifacts"`
	Detail                 string   `json:"detail"`
}

// ContractUpstreamRef is a pin to the upstream harness command a reader can
// verify the contract against.
type ContractUpstreamRef struct {
	Name  string `json:"name"`
	URL   string `json:"url"`
	Notes string `json:"notes,omitempty"`
}

// ContractGate is one readiness condition. All gates OK => READY_FOR_OFFICIAL_RUN.
type ContractGate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// BuildOfficialRunContract builds the result-claim-gated contract. It is pure:
// it performs no run, touches no network, and always sets
// ResultClaimAllowed=false — only the official evaluator can ever back a score.
func BuildOfficialRunContract(in OfficialRunContractInput) OfficialRunContract {
	scenario := Scenario(strings.TrimSpace(string(in.Scenario)))
	if scenario == "" {
		scenario = ScenarioCodeGeneration
	}
	model := strings.TrimSpace(in.Model)
	gateway := strings.TrimSpace(in.Gateway)
	runDir := strings.TrimSpace(in.RunDir)
	if runDir == "" {
		runDir = "experiments/livecodebench/<run-id>"
	}

	// Resolve the release selector so the concrete release is recorded and an
	// implicit release_latest can be caught by a gate.
	sel, relErr := ResolveRelease(in.ReleaseSelector)
	resolvedRelease := sel.Resolved
	releaseExplicit := relErr == nil &&
		strings.TrimSpace(in.ReleaseSelector) != "" &&
		sel.Selector != ReleaseLatest

	start := strings.TrimSpace(in.StartDate)
	end := strings.TrimSpace(in.EndDate)
	windowOK, windowDetail := dateWindowGate(start, end)

	problemIDs := contractProblemIDs(in.Suite, scenario)

	arms := []ContractArm{
		{
			Name:              "raw-lcb_runner",
			Harness:           "official-lcb_runner-through-fak-gateway",
			Commands:          []string{rawGenerateCommand(model, scenario, resolvedRelease)},
			OutputDir:         armOutputDir(runDir, "raw", scenario),
			RequiredArtifacts: []string{"saved generations", "generation artifact SHA256 digest"},
			Notes:             "requires an lcb_runner model registration pointing at the fak gateway; --evaluate produces the official pass@1 directly.",
		},
		{
			Name:    "fak-native",
			Harness: "fak-livecodebench-generate-through-fak-gateway",
			Commands: []string{
				fakGenerateCommand(gateway, model, resolvedRelease, scenario, start, end, runDir),
				fakExportCommand(runDir, scenario),
			},
			OutputDir:         armOutputDir(runDir, "fak", scenario),
			RequiredArtifacts: []string{"saved generations", "generation artifact SHA256 digest", "custom-evaluator export digest"},
			Notes:             "generates only; the SAME problem_ids/prompt_hash/model/release/scenario as the raw arm, exported to the official custom-evaluator shape.",
		},
	}

	grading := ContractGrading{
		Authority:              "official LiveCodeBench lcb_runner",
		RawEvaluateCommand:     rawEvaluateCommand(model, scenario, resolvedRelease),
		CustomEvaluatorCommand: customEvaluatorCommand(runDir, scenario, resolvedRelease),
		ComputeScoresCommand:   computeScoresCommand(runDir, start, end),
		RequiredArtifacts: []string{
			"official eval_all artifact + SHA256 digest",
			"pass@1 / pass@5 for the recorded contest-date window",
		},
		Detail: "Grading is the sole result-bearing authority. The exact saved generations from each arm are graded by the official evaluator; only then may LIVECODEBENCH-RESULTS.md be filled and result_claim_allowed flip true (#2113).",
	}

	gates := []ContractGate{
		{Name: "release_pinned_explicit", OK: releaseExplicit, Detail: releaseGateDetail(in.ReleaseSelector, resolvedRelease, relErr)},
		{Name: "scenario_known", OK: KnownScenario(scenario), Detail: string(scenario)},
		{Name: "date_window_recorded", OK: windowOK, Detail: windowDetail},
		{Name: "model_recorded", OK: model != "", Detail: modelGateDetail(model)},
		{Name: "serving_backend_recorded", OK: strings.TrimSpace(in.ServingBackend) != "", Detail: servingBackendDetail(in.ServingBackend)},
		{Name: "fak_gateway_recorded", OK: gateway != "", Detail: gatewayGateDetail(gateway)},
		{Name: "candidate_problem_ids", OK: len(problemIDs) > 0, Detail: problemIDsDetail(len(problemIDs))},
		{Name: "same_problem_ids_required", OK: true, Detail: "raw and fak arms must score the identical question_ids"},
		{Name: "same_prompt_hash_required", OK: true, Detail: "raw and fak arms must send the identical rendered prompt per problem (SamePromptHash)"},
		{Name: "same_release_required", OK: true, Detail: "both arms must use the identical release_version"},
		{Name: "official_grading_required", OK: true, Detail: "the exact saved generations must be graded by the official lcb_runner evaluator before any claim"},
	}

	return OfficialRunContract{
		Schema:        OfficialRunContractSchema,
		GeneratedAt:   strings.TrimSpace(in.GeneratedAt),
		Issue:         strings.TrimSpace(in.Issue),
		Benchmark:     "LiveCodeBench official-run contract (raw lcb_runner vs fak-native)",
		Status:        contractStatus(gates),
		EvidenceClass: "OFFICIAL_RUN_CONTRACT",
		ClaimBoundary: "Official-run contract only: it pins the run constants, the raw and fak generation commands, the shared problem/prompt/model/release requirements, and the official grading handoff. It performs no run and claims no pass-rate; result_claim_allowed stays false until the official evaluator grades the exact saved generations (#2113).",
		Constants: ContractConstants{
			ReleaseSelector: strings.TrimSpace(in.ReleaseSelector),
			ReleaseVersion:  resolvedRelease,
			Scenario:        string(scenario),
			StartDate:       start,
			EndDate:         end,
			Model:           model,
			ServingBackend:  strings.TrimSpace(in.ServingBackend),
			Gateway:         gateway,
			RunDir:          runDir,
		},
		ProblemSelection: ContractProblemSelection{
			CandidateSuite:         strings.TrimSpace(in.SuitePath),
			CandidateProblemIDs:    problemIDs,
			SameProblemIDsRequired: true,
			SamePromptHashRequired: true,
			SameReleaseRequired:    true,
			SameModelRequired:      true,
			SameScenarioRequired:   true,
		},
		Arms:    arms,
		Grading: grading,
		UpstreamRefs: []ContractUpstreamRef{
			{Name: "LiveCodeBench harness", URL: "https://github.com/LiveCodeBench/LiveCodeBench", Notes: "Official runner; the scoring authority."},
			{Name: "lcb_runner custom_evaluator", URL: "https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/runner/custom_evaluator.py", Notes: "Grades externally-generated code_list rows (the fak arm handoff)."},
			{Name: "lcb_runner compute_scores", URL: "https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_scores.py", Notes: "Computes the date-windowed score from a saved eval_all artifact."},
		},
		Gates: gates,
		CompareMetrics: []string{
			"pass_1",
			"pass_5",
			"window_pass_1",
			"graded_count",
			"same_problem_ids",
			"same_prompt_hash",
			"fak_gateway_model_http_success",
		},
		RequiredBeforeClaim: []string{
			"release_version, scenario, start_date, and end_date recorded",
			"model identity, serving backend, and model training-cutoff statement or explicit residual",
			"raw-arm saved generations + SHA256 digest",
			"fak-arm saved generations + SHA256 digest",
			"SameProblemIDs and SamePromptHash asserted between raw and fak arms over the same release",
			"official grading command + eval_all artifact digest for each arm",
			"result_claim_allowed flips true only after official grading (#2113); LIVECODEBENCH-RESULTS.md cells filled for both arms",
		},
		ResultClaimAllowed: false,
	}
}

// contractProblemIDs returns the sorted, de-duplicated question_ids from a
// suite for the chosen scenario. A nil suite yields no candidates (the
// candidate_problem_ids gate then asks the operator to pass a fetched suite).
func contractProblemIDs(s *Suite, scenario Scenario) []string {
	if s == nil {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(s.Problems))
	for _, p := range s.Problems {
		if scenario != "" && p.Scenario != scenario {
			continue
		}
		id := strings.TrimSpace(p.QuestionID)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func rawGenerateCommand(model string, scenario Scenario, release string) string {
	return fmt.Sprintf("python -m lcb_runner.runner.main --model %s --scenario %s --release_version %s",
		orPlaceholder(model, "<model>"), scenario, release)
}

func rawEvaluateCommand(model string, scenario Scenario, release string) string {
	return fmt.Sprintf("python -m lcb_runner.runner.main --model %s --scenario %s --evaluate --release_version %s",
		orPlaceholder(model, "<model>"), scenario, release)
}

func fakGenerateCommand(gateway, model, release string, scenario Scenario, start, end, runDir string) string {
	return fmt.Sprintf("fak livecodebench generate --gateway %s --model %s --release-version %s --scenario %s --start-date %s --end-date %s --out %s",
		orPlaceholder(gateway, "<gateway>"), orPlaceholder(model, "<model>"), release, scenario,
		orPlaceholder(start, "<start-date>"), orPlaceholder(end, "<end-date>"),
		armOutputDir(runDir, "fak", scenario))
}

func fakExportCommand(runDir string, scenario Scenario) string {
	return fmt.Sprintf("go run ./cmd/livecodebench export --format custom-evaluator --fixture %s --out %s",
		fixturePath(runDir, scenario), customPath(runDir, scenario))
}

func customEvaluatorCommand(runDir string, scenario Scenario, release string) string {
	return fmt.Sprintf("python -m lcb_runner.runner.custom_evaluator --custom_output_file %s --release_version %s",
		customPath(runDir, scenario), release)
}

func computeScoresCommand(runDir, start, end string) string {
	return fmt.Sprintf("python -m lcb_runner.evaluation.compute_scores --eval_all_file %s --start_date %s --end_date %s",
		joinRunPath(runDir, "<official-eval-all-file>"), orPlaceholder(start, "<start-date>"), orPlaceholder(end, "<end-date>"))
}

func armOutputDir(runDir, arm string, scenario Scenario) string {
	return joinRunPath(runDir, arm+"/"+string(scenario))
}

func fixturePath(runDir string, scenario Scenario) string {
	return joinRunPath(runDir, fmt.Sprintf("fak-%s-fixture.json", scenario))
}

func customPath(runDir string, scenario Scenario) string {
	return joinRunPath(runDir, fmt.Sprintf("fak-%s-custom.json", scenario))
}

func joinRunPath(runDir, leaf string) string {
	runDir = strings.TrimRight(strings.TrimSpace(runDir), "/\\")
	if runDir == "" {
		return leaf
	}
	return runDir + "/" + leaf
}

func orPlaceholder(v, placeholder string) string {
	if strings.TrimSpace(v) == "" {
		return placeholder
	}
	return strings.TrimSpace(v)
}

func dateWindowGate(start, end string) (bool, string) {
	if start == "" || end == "" {
		return false, "start_date/end_date required (the contamination window; never implicit)"
	}
	s, err := parseContractDate(start)
	if err != nil {
		return false, fmt.Sprintf("start_date %q is not YYYY-MM-DD", start)
	}
	e, err := parseContractDate(end)
	if err != nil {
		return false, fmt.Sprintf("end_date %q is not YYYY-MM-DD", end)
	}
	if e.Before(s) {
		return false, fmt.Sprintf("end_date %s precedes start_date %s", end, start)
	}
	return true, start + " .. " + end
}

func parseContractDate(s string) (time.Time, error) {
	return time.Parse(dateLayout, strings.TrimSpace(s))
}

func releaseGateDetail(selector, resolved string, err error) string {
	selector = strings.TrimSpace(selector)
	if err != nil {
		return err.Error()
	}
	if selector == "" || selector == ReleaseLatest {
		return "release must be pinned explicitly, never release_latest in a published result"
	}
	return "resolved " + resolved
}

func modelGateDetail(model string) string {
	if model == "" {
		return "model is required (raw and fak arms must share it)"
	}
	return model
}

func servingBackendDetail(backend string) string {
	if strings.TrimSpace(backend) == "" {
		return "record the serving engine + quantization (e.g. SGLang W4AFP8) and a training-cutoff statement"
	}
	return strings.TrimSpace(backend)
}

func gatewayGateDetail(gateway string) string {
	if gateway == "" {
		return "record the local fak gateway base URL (no host/channel/token/URL secrets in tracked files)"
	}
	return gateway
}

func problemIDsDetail(n int) string {
	if n == 0 {
		return "pass a fetched suite (--suite) to pin the exact question_ids both arms score"
	}
	label := "ids"
	if n == 1 {
		label = "id"
	}
	return fmt.Sprintf("%d candidate question %s from the pinned suite", n, label)
}

func contractStatus(gates []ContractGate) string {
	for _, g := range gates {
		if !g.OK {
			return ContractIncomplete
		}
	}
	return ContractReady
}

// RenderOfficialRunContractMarkdown renders the contract as human-readable
// markdown, mirroring the terminalbench contract layout.
func RenderOfficialRunContractMarkdown(c OfficialRunContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# LiveCodeBench Official-Run Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	if c.Issue != "" {
		fmt.Fprintf(&b, "- Issue: `%s`\n", c.Issue)
	}
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", c.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Constants\n\n")
	fmt.Fprintf(&b, "| Constant | Value |\n|---|---|\n")
	fmt.Fprintf(&b, "| Release | `%s` (selector `%s`) |\n", c.Constants.ReleaseVersion, orPlaceholder(c.Constants.ReleaseSelector, "-"))
	fmt.Fprintf(&b, "| Scenario | `%s` |\n", c.Constants.Scenario)
	fmt.Fprintf(&b, "| Date window | `%s .. %s` |\n", orPlaceholder(c.Constants.StartDate, "-"), orPlaceholder(c.Constants.EndDate, "-"))
	fmt.Fprintf(&b, "| Model | `%s` |\n", orPlaceholder(c.Constants.Model, "-"))
	fmt.Fprintf(&b, "| Serving backend | `%s` |\n", orPlaceholder(c.Constants.ServingBackend, "-"))
	fmt.Fprintf(&b, "| Gateway | `%s` |\n", orPlaceholder(c.Constants.Gateway, "-"))
	fmt.Fprintf(&b, "| Run dir | `%s` |\n\n", c.Constants.RunDir)

	if len(c.ProblemSelection.CandidateProblemIDs) > 0 {
		fmt.Fprintf(&b, "## Problem Selection\n\n")
		fmt.Fprintf(&b, "- Candidate suite: `%s`\n", c.ProblemSelection.CandidateSuite)
		fmt.Fprintf(&b, "- Candidate question ids: `%s`\n\n", strings.Join(c.ProblemSelection.CandidateProblemIDs, "`, `"))
	}

	fmt.Fprintf(&b, "## Arms\n\n")
	for _, arm := range c.Arms {
		fmt.Fprintf(&b, "### `%s` (%s)\n\n", arm.Name, arm.Harness)
		fmt.Fprintf(&b, "- Output: `%s`\n", arm.OutputDir)
		if arm.Notes != "" {
			fmt.Fprintf(&b, "- Notes: %s\n", arm.Notes)
		}
		for _, cmd := range arm.Commands {
			fmt.Fprintf(&b, "\n```bash\n%s\n```\n", cmd)
		}
		fmt.Fprintf(&b, "\n")
	}

	fmt.Fprintf(&b, "## Grading (sole result authority: %s)\n\n", c.Grading.Authority)
	fmt.Fprintf(&b, "```bash\n# raw arm direct grade\n%s\n\n# fak arm: grade the exported generations\n%s\n\n# date-windowed score\n%s\n```\n\n", c.Grading.RawEvaluateCommand, c.Grading.CustomEvaluatorCommand, c.Grading.ComputeScoresCommand)
	fmt.Fprintf(&b, "%s\n\n", c.Grading.Detail)

	fmt.Fprintf(&b, "## Gates\n\n| Gate | OK | Detail |\n|---|:---:|---|\n")
	for _, g := range c.Gates {
		mark := "no"
		if g.OK {
			mark = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", g.Name, mark, mdContractCell(g.Detail))
	}

	fmt.Fprintf(&b, "\n## Required Before Any Result Claim\n\n")
	for _, req := range c.RequiredBeforeClaim {
		fmt.Fprintf(&b, "- %s\n", req)
	}
	return b.String()
}

func mdContractCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
