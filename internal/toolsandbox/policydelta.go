package toolsandbox

import (
	"fmt"
	"strings"
)

// Tau2PolicyDeltaContractSchema is the #1069 extension of the #873 Packet E
// official-run contract. It fixes the raw-vs-(model+fak-gateway) two-arm shape
// for a tau2-bench (Sierra) self-submit, and — the part the generic official-run
// contract does not carry — it labels every reported number by provenance and
// pins the honesty fence that keeps fak from claiming the fronted model's score.
const Tau2PolicyDeltaContractSchema = "fak.tau2-policy-delta-contract.v1"

// Provenance separates what fak controls from what it merely relays. This is the
// conflation fence of #1069: the tau2 Pass^k task-resolve rate is the fronted
// MODEL's property (graded by the tau2 harness, the scoring authority) and is
// only OBSERVED; the under-policy delta, gateway tax, and per-call evidence are
// authored by fak and are WITNESSED.
const (
	ProvenanceWitnessed = "WITNESSED" // fak authors and controls this value
	ProvenanceObserved  = "OBSERVED"  // relayed; the fronted model's property, graded by the tau2 harness
)

// Metric kinds let the fence assert that task success is never folded into a
// policy-compliance number.
const (
	MetricKindTaskResolve      = "task_resolve"
	MetricKindPolicyCompliance = "policy_compliance"
	MetricKindCost             = "cost"
	MetricKindEvidence         = "evidence"
)

type Tau2PolicyDeltaContractInput struct {
	GeneratedAt      string
	Suite            Suite
	SuitePath        string
	Domain           string   // primary tau2 domain (retail|airline|telecom|banking_knowledge)
	Domains          []string // every domain in the paired run; banking_knowledge is required for the Overall column
	Model            string
	UserModel        string
	Trials           int // >= 4 so Pass^k for k in {1,2,3,4} is defined
	RawCommand       string
	FakCommand       string
	RawOutputDir     string
	FakOutputDir     string
	FakGateway       string
	SubmissionFile   string // single committed leaderboard file
	TrajectoriesLink string // external host (HF/Drive); trajectories are NOT committed
}

// DeltaMetric is one reported number with its provenance label and a flag that
// must stay false for every policy/cost/evidence metric: task success is
// reported on its own task_resolve rows, never mixed into a compliance figure.
type DeltaMetric struct {
	Name             string `json:"name"`
	Provenance       string `json:"provenance"`
	Kind             string `json:"kind"`
	FoldsTaskSuccess bool   `json:"folds_task_success"`
	Detail           string `json:"detail"`
}

// Tau2HonestyFence is the machine-readable form of the #1069 honesty fence. Each
// bool is an invariant the compare artifact must hold before a claim is allowed.
type Tau2HonestyFence struct {
	TaskSuccessNeverFoldedIntoPolicy           bool   `json:"task_success_never_folded_into_policy"`
	RawResolveMustBeStatisticallyFlat          bool   `json:"raw_resolve_must_be_statistically_flat"`
	ComplianceGainByDepressingSuccessIsNotAWin bool   `json:"compliance_gain_by_depressing_success_is_not_a_win"`
	NoVsNaiveCacheMultiple                     bool   `json:"no_vs_naive_cache_multiple"`
	CacheFigureMarginalOverTunedWarmKV         bool   `json:"cache_figure_marginal_over_tuned_warm_kv"`
	HeadlineNumberIsModelsNotFak               bool   `json:"headline_number_is_models_not_fak"`
	Detail                                     string `json:"detail"`
}

// Tau2SubmissionShape pins the tau2-bench self-submit mechanics (open GitHub PR,
// submission_type:'custom').
type Tau2SubmissionShape struct {
	SubmissionType                      string `json:"submission_type"`
	SubmissionFile                      string `json:"submission_file"`
	MethodologyNotesRequired            bool   `json:"methodology_notes_required"`
	VerificationModifiedPromptsRequired bool   `json:"verification_modified_prompts_required"`
	ReferencesRequired                  bool   `json:"references_required"`
	TrajectoriesHostedExternally        bool   `json:"trajectories_hosted_externally"`
	TrajectoriesLink                    string `json:"trajectories_link,omitempty"`
	ManifestEntryRequired               bool   `json:"manifest_entry_required"`
	ValidateCommand                     string `json:"validate_command"`
}

type Tau2PolicyDeltaContract struct {
	Schema              string                    `json:"schema"`
	GeneratedAt         string                    `json:"generated_at"`
	Benchmark           string                    `json:"benchmark"`
	ParentEpic          int                       `json:"parent_epic"`
	Issue               int                       `json:"issue"`
	Status              string                    `json:"status"`
	EvidenceClass       string                    `json:"evidence_class"`
	ClaimBoundary       string                    `json:"claim_boundary"`
	TaskSelection       ContractTaskSelection     `json:"task_selection"`
	Model               ContractModel             `json:"model"`
	Arms                []ContractArm             `json:"arms"`
	DeltaMetrics        []DeltaMetric             `json:"delta_metrics"`
	ScoreEvidenceLink   ContractScoreEvidenceLink `json:"score_evidence_link"`
	HonestyFence        Tau2HonestyFence          `json:"honesty_fence"`
	Submission          Tau2SubmissionShape       `json:"submission"`
	UpstreamRefs        []UpstreamRef             `json:"upstream_refs"`
	Gates               []ContractGate            `json:"gates"`
	RequiredBeforeClaim []string                  `json:"required_before_claim"`
	ResultClaimAllowed  bool                      `json:"result_claim_allowed"`
}

// BuildTau2PolicyDeltaContract assembles the gated tau2 policy-delta contract.
// It NEVER allows a result claim: result_claim_allowed flips to true only after
// both arms' grader summaries are checked in and parity is proven on disk, which
// this builder cannot see — so the structural artifact is permanently gated.
func BuildTau2PolicyDeltaContract(in Tau2PolicyDeltaContractInput) Tau2PolicyDeltaContract {
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
		in.Trials = 4
	}
	if len(in.Domains) == 0 {
		in.Domains = []string{in.Domain}
	}
	submissionFile := strings.TrimSpace(in.SubmissionFile)
	if submissionFile == "" {
		submissionFile = "web/leaderboard/public/submissions/{model}_{org}_{date}/submission.json"
	}

	taskIDs := contractTaskIDs(in.Suite, in.Domain)
	gates := []ContractGate{
		{Name: "candidate_task_ids", OK: len(taskIDs) > 0, Detail: candidateTaskDetail(len(taskIDs))},
		{Name: "same_task_ids_required", OK: true, Detail: "raw and fak arms must run the identical tau2 task ids"},
		{Name: "same_model_required", OK: strings.TrimSpace(in.Model) != "", Detail: strings.TrimSpace(in.Model)},
		{Name: "same_user_simulator_required", OK: strings.TrimSpace(in.UserModel) != "", Detail: strings.TrimSpace(in.UserModel)},
		{Name: "same_budget_required", OK: true, Detail: "identical per-task budget and retry policy across arms"},
		{Name: "trials_ge_4_for_pass_k", OK: in.Trials >= 4, Detail: fmt.Sprintf("trials=%d; Pass^k for k in {1,2,3,4} needs >= 4", in.Trials)},
		{Name: "banking_knowledge_domain_for_overall", OK: domainsInclude(in.Domains, "banking_knowledge"), Detail: "banking_knowledge (with retrieval_config) is required for the Overall column"},
		{Name: "raw_arm_command", OK: strings.TrimSpace(in.RawCommand) != "", Detail: strings.TrimSpace(in.RawCommand)},
		{Name: "fak_arm_command", OK: strings.TrimSpace(in.FakCommand) != "", Detail: strings.TrimSpace(in.FakCommand)},
		{Name: "solve_rate_neutrality_required", OK: true, Detail: "raw resolve-% delta between arms must be reported with trial count and shown statistically flat"},
		{Name: "official_grader_required", OK: true, Detail: "tau2-native result_summary.json + trajectories.jsonl per arm are required before promotion"},
	}

	return Tau2PolicyDeltaContract{
		Schema:        Tau2PolicyDeltaContractSchema,
		GeneratedAt:   in.GeneratedAt,
		Benchmark:     "tau2-bench (Sierra) under-policy delta — bare model vs model+fak-gateway",
		ParentEpic:    1063,
		Issue:         1069,
		Status:        contractStatus(gates),
		EvidenceClass: "EXTERNAL_RUN_CONTRACT",
		ClaimBoundary: "Policy-delta contract only: fixes the two-arm command shape, the same-task/model/simulator/budget parity gates, the provenance label on every reported number, and the tau2 self-submit mechanics. It is NOT a result until both arms' tau2 grader summaries and the fak per-call evidence log are checked in and parity is proven on disk. fak claims ONLY the under-policy delta; the headline Pass^k on the board is and remains the fronted model's.",
		TaskSelection: ContractTaskSelection{
			CandidateSuite:          strings.TrimSpace(in.SuitePath),
			CandidateTaskIDs:        taskIDs,
			OfficialHarness:         "tau2",
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
				Name:      "raw",
				Harness:   "tau2-native",
				Command:   strings.TrimSpace(in.RawCommand),
				OutputDir: strings.TrimSpace(in.RawOutputDir),
				RequiredArtifacts: []string{
					"result_summary.json",
					"trajectories.jsonl",
				},
			},
			{
				Name:      "fak",
				Harness:   "tau2-native-through-fak-gateway",
				Command:   strings.TrimSpace(in.FakCommand),
				OutputDir: strings.TrimSpace(in.FakOutputDir),
				RequiredArtifacts: []string{
					"result_summary.json",
					"trajectories.jsonl",
					"fak-toolcall-evidence.jsonl",
				},
			},
		},
		DeltaMetrics:      tau2DeltaMetrics(),
		ScoreEvidenceLink: toolsandboxScoreEvidenceLink(in.RawOutputDir, in.FakOutputDir),
		HonestyFence: Tau2HonestyFence{
			TaskSuccessNeverFoldedIntoPolicy:           true,
			RawResolveMustBeStatisticallyFlat:          true,
			ComplianceGainByDepressingSuccessIsNotAWin: true,
			NoVsNaiveCacheMultiple:                     true,
			CacheFigureMarginalOverTunedWarmKV:         true,
			HeadlineNumberIsModelsNotFak:               true,
			Detail:                                     "tau2 Pass^k is the fronted model's task-resolve number (OBSERVED, graded by the tau2 harness), never fak's. fak claims only the under-policy delta (blocked-bad-tool-call / policy-adherence), labeled WITNESSED. Hold raw resolve-% statistically flat across arms or the comparison is invalid; a compliance gain bought by depressing task success is reported as the utility cost, not a win. Any cache/cost figure is marginal-over-tuned-warm-KV; no vs-baseline cache multiple is quoted.",
		},
		Submission: Tau2SubmissionShape{
			SubmissionType:                      "custom",
			SubmissionFile:                      submissionFile,
			MethodologyNotesRequired:            true,
			VerificationModifiedPromptsRequired: true,
			ReferencesRequired:                  true,
			TrajectoriesHostedExternally:        true,
			TrajectoriesLink:                    strings.TrimSpace(in.TrajectoriesLink),
			ManifestEntryRequired:               true,
			ValidateCommand:                     "tau2 submit validate ./my_submission",
		},
		UpstreamRefs: []UpstreamRef{
			{
				Name:  "tau2-bench (Sierra)",
				URL:   "https://github.com/sierra-research/tau2-bench",
				Notes: "Open self-submit leaderboard (taubench.com); submission_type:'custom' is the non-default-scaffold shape a gateway run files under.",
			},
		},
		Gates: gates,
		RequiredBeforeClaim: []string{
			"tau2-native task ids for retail, airline, telecom, and banking_knowledge (banking_knowledge with retrieval_config is required for the Overall column)",
			"raw-arm result_summary.json + trajectories.jsonl over those exact task ids, >= 4 trials per domain",
			"fak-arm result_summary.json + trajectories.jsonl over the SAME task ids, model, user simulator, budget, and retry policy",
			"fak-arm fak-toolcall-evidence.jsonl joined to the fak trajectory on task_id/turn/tool/args_hash",
			"proof the raw resolve-% delta between arms is statistically flat, reported with its trial count (gateway is solve-rate-neutral)",
			"a raw/fak compare artifact reporting raw Pass^k (OBSERVED) separately from the WITNESSED policy-compliance delta — task success never folded into a policy metric",
			"a submission.json validated by `tau2 submit validate` with submission_type=custom, methodology.verification populated, and trajectories linked externally",
		},
		ResultClaimAllowed: false,
	}
}

// tau2DeltaMetrics is the provenance-labeled compare ledger: every task-resolve
// row is OBSERVED (the model's, relayed) and every policy/cost/evidence row is
// WITNESSED (fak's). No row folds task success into a policy number.
func tau2DeltaMetrics() []DeltaMetric {
	return []DeltaMetric{
		{Name: "raw_pass_k", Provenance: ProvenanceObserved, Kind: MetricKindTaskResolve, Detail: "raw-arm Pass^k for k in {1,2,3,4}; the fronted model's task-resolve rate graded by the tau2 harness — relayed, not a fak score."},
		{Name: "fak_arm_pass_k", Provenance: ProvenanceObserved, Kind: MetricKindTaskResolve, Detail: "model+gateway-arm Pass^k; still the model's resolve, measured through the gateway."},
		{Name: "raw_resolve_delta", Provenance: ProvenanceObserved, Kind: MetricKindTaskResolve, Detail: "fak-arm minus raw resolve-%, reported with trial count; must be statistically flat — any non-flat delta is explained, never claimed as fak skill."},
		{Name: "blocked_disallowed_tool_calls", Provenance: ProvenanceWitnessed, Kind: MetricKindPolicyCompliance, Detail: "default-deny adjudications fak made on disallowed tool calls — the structural admission floor fak authors and controls."},
		{Name: "policy_adherence_events", Provenance: ProvenanceWitnessed, Kind: MetricKindPolicyCompliance, Detail: "per-turn policy-adherence events fak recorded across the mediated trajectory."},
		{Name: "unnecessary_blocks", Provenance: ProvenanceWitnessed, Kind: MetricKindPolicyCompliance, Detail: "benign tool calls fak wrongly blocked — the utility cost, reported honestly and never hidden."},
		{Name: "gateway_tax_pct", Provenance: ProvenanceWitnessed, Kind: MetricKindCost, Detail: "the ~3% gateway latency/throughput tax at saturation; a marginal-over-tuned-warm-KV cost, never a vs-baseline cache multiple."},
		{Name: "evidence_completeness", Provenance: ProvenanceWitnessed, Kind: MetricKindEvidence, Detail: "fraction of mediated tool calls with a complete per-call fak verdict/evidence row joined on task_id/turn/tool/args_hash."},
	}
}

func domainsInclude(domains []string, want string) bool {
	for _, d := range domains {
		if strings.EqualFold(strings.TrimSpace(d), want) {
			return true
		}
	}
	return false
}

// RenderTau2PolicyDeltaContractMarkdown renders the gated contract for human and
// agent inspection, mirroring the official-run contract's markdown shape.
func RenderTau2PolicyDeltaContractMarkdown(c Tau2PolicyDeltaContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# tau2-bench Policy-Delta Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Parent epic / issue: `#%d` / `#%d`\n", c.ParentEpic, c.Issue)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", c.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Arms\n\n")
	fmt.Fprintf(&b, "| Arm | Harness | Output | Command |\n")
	fmt.Fprintf(&b, "|---|---|---|---|\n")
	for _, arm := range c.Arms {
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | %s |\n", arm.Name, arm.Harness, arm.OutputDir, mdCell(arm.Command))
	}

	fmt.Fprintf(&b, "\n## Delta Metrics (provenance-labeled)\n\n")
	fmt.Fprintf(&b, "| Metric | Provenance | Kind | Detail |\n")
	fmt.Fprintf(&b, "|---|:---:|---|---|\n")
	for _, m := range c.DeltaMetrics {
		fmt.Fprintf(&b, "| `%s` | %s | `%s` | %s |\n", m.Name, m.Provenance, m.Kind, mdCell(m.Detail))
	}

	fmt.Fprintf(&b, "\n## Honesty Fence\n\n%s\n", c.HonestyFence.Detail)

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
