package main

// cachevalue_status.go holds the `fak cachevalue status` command core: options,
// the schema, and the top-level report assembly + rendering dispatch. The optional
// per-digest builders (session, ablation, headroom-bench, and the vcache score/
// actions/join/observe/witness views) live in sibling files split from cachevalue_status.go,
// split out along those concern seams so this dispatch surface stays steerable as
// new digests land (steerability dispatch_god_file). Behavior-preserving code
// motion -- same package, no logic change.
import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

const cachevalueStatusSchema = "fak.cachevalue.status.v1"

type cachevalueStatusOptions struct {
	Ledger                   string
	SavingsLedger            string
	UsageLedger              string
	Since                    string
	ContextBudgetTokens      uint64
	HeadroomTimeout          time.Duration
	ArtifactDir              string
	SessionPath              string
	AblationReportPath       string
	HeadroomBenchPath        string
	VCacheScorePath          string
	VCacheActionsPath        string
	VCacheContextJoinPath    string
	VCacheObservePath        string
	VCacheContextWitnessPath string
}

type cachevalueStatusReport struct {
	Schema               string                                `json:"schema"`
	GeneratedAt          string                                `json:"generated_at"`
	Since                string                                `json:"since,omitempty"`
	Verdict              string                                `json:"verdict"`
	Summary              string                                `json:"summary"`
	Sources              cachevalueStatusSources               `json:"sources"`
	Counts               map[string]int                        `json:"counts"`
	Attribution          cachevalueStatusAttribution           `json:"attribution"`
	Value                cachevalueValueDigest                 `json:"value"`
	Headroom             cachevalueHeadroomDigest              `json:"headroom"`
	VCache               cachevalueVCacheDigest                `json:"vcache"`
	Session              *cachevalueSessionDigest              `json:"session,omitempty"`
	Ablation             *cachevalueAblationDigest             `json:"ablation,omitempty"`
	HeadroomBench        *cachevalueHeadroomBenchDigest        `json:"headroom_bench,omitempty"`
	VCacheScore          *cachevalueVCacheScoreDigest          `json:"vcache_score,omitempty"`
	VCacheActions        *cachevalueVCacheActionsDigest        `json:"vcache_actions,omitempty"`
	VCacheContextJoin    *cachevalueVCacheContextJoinDigest    `json:"vcache_context_join,omitempty"`
	VCacheObserve        *cachevalueVCacheObserveDigest        `json:"vcache_observe,omitempty"`
	VCacheContextWitness *cachevalueVCacheContextWitnessDigest `json:"vcache_context_witness,omitempty"`
	Rows                 []cachevalueStatusRow                 `json:"rows"`
	NextActions          []string                              `json:"next_actions,omitempty"`
}

type cachevalueStatusSources struct {
	KernelLedger               string `json:"kernel_ledger"`
	SavingsLedger              string `json:"savings_ledger"`
	UsageLedger                string `json:"usage_ledger"`
	ArtifactDir                string `json:"artifact_dir,omitempty"`
	VCacheSnapshot             string `json:"vcache_snapshot"`
	ContextSnapshot            string `json:"context_snapshot"`
	HeadroomURL                string `json:"headroom_url"`
	AblationReport             string `json:"ablation_report,omitempty"`
	HeadroomBenchReport        string `json:"headroom_bench_report,omitempty"`
	VCacheScoreReport          string `json:"vcache_score_report,omitempty"`
	VCacheActionsReport        string `json:"vcache_actions_report,omitempty"`
	VCacheContextJoinReport    string `json:"vcache_context_join_report,omitempty"`
	VCacheObserveReport        string `json:"vcache_observe_report,omitempty"`
	VCacheContextWitnessReport string `json:"vcache_context_witness_report,omitempty"`
}

type cachevalueValueDigest struct {
	Verdict              string `json:"verdict"`
	Finding              string `json:"finding,omitempty"`
	NextAction           string `json:"next_action,omitempty"`
	Track1Verdict        string `json:"track1_verdict"`
	RejectedTierAccesses uint64 `json:"rejected_tier_accesses"`
	Track2Buckets        int    `json:"track2_buckets"`
	FleetUsageRows       int    `json:"fleet_usage_rows"`
	DollarBlindRows      int    `json:"dollar_blind_rows,omitempty"`
}

type cachevalueHeadroomDigest struct {
	Selected          string         `json:"selected"`
	HeadroomReachable bool           `json:"headroom_reachable"`
	GateStats         headroom.Stats `json:"gate_stats"`
}

type cachevalueVCacheDigest struct {
	Status                  string `json:"status"`
	LiveProvider            string `json:"live_provider"`
	ContextAPI              string `json:"context_api"`
	ProviderCalibration     string `json:"provider_calibration"`
	ProviderActions         string `json:"provider_actions"`
	ProviderActionTransport string `json:"provider_action_transport"`
	CodexOpenAI             string `json:"codex_openai"`
	RecentProviderStatus    string `json:"recent_provider_status,omitempty"`
	RecentContextStatus     string `json:"recent_context_status,omitempty"`
	RecentObservationError  string `json:"recent_observation_error,omitempty"`
}

type cachevalueSessionDigest struct {
	Path               string   `json:"path"`
	Session            string   `json:"session"`
	Status             string   `json:"status"`
	LikelyDomain       string   `json:"likely_domain"`
	Finding            string   `json:"finding"`
	AssistantTurns     int64    `json:"assistant_turns"`
	ToolCalls          int64    `json:"tool_calls"`
	ReadOnlyToolFrac   *float64 `json:"read_only_tool_frac,omitempty"`
	InputTokens        int64    `json:"input_tokens"`
	OutputTokens       int64    `json:"output_tokens"`
	CacheReadTokens    int64    `json:"cache_read_tokens"`
	CacheCreateTokens  int64    `json:"cache_create_tokens"`
	TotalContextTokens int64    `json:"total_context_tokens"`
	CacheHitFrac       *float64 `json:"cache_hit_frac,omitempty"`
	IORatio            *float64 `json:"io_ratio,omitempty"`
	CostUSD            float64  `json:"cost_usd"`
	Error              string   `json:"error,omitempty"`
}

type cachevalueAblationDigest struct {
	Path                      string `json:"path"`
	Status                    string `json:"status"`
	Runs                      int    `json:"runs"`
	DroppedArms               int    `json:"dropped_arms"`
	DroppedWithDiagnostics    int    `json:"dropped_with_diagnostics,omitempty"`
	DroppedWorkloadMismatches int    `json:"dropped_workload_mismatches,omitempty"`
	DroppedChildExits         int    `json:"dropped_child_exits,omitempty"`
	CacheEffects              int    `json:"cache_effects"`
	ActiveEffects             int    `json:"active_effects"`
	UnavailableEffects        int    `json:"unavailable_effects"`
	Error                     string `json:"error,omitempty"`
}

type cachevalueHeadroomBenchDigest struct {
	Path       string  `json:"path"`
	Status     string  `json:"status"`
	Compressor string  `json:"compressor"`
	Owner      string  `json:"owner,omitempty"`
	Dependency string  `json:"dependency,omitempty"`
	Fidelity   string  `json:"fidelity,omitempty"`
	Evidence   string  `json:"evidence,omitempty"`
	Samples    int     `json:"samples"`
	OrigTotal  int     `json:"orig_total"`
	NewTotal   int     `json:"new_total"`
	SavedRatio float64 `json:"saved_ratio"`
	Reason     string  `json:"reason,omitempty"`
	Error      string  `json:"error,omitempty"`
}

type cachevalueVCacheScoreDigest struct {
	Path                   string  `json:"path"`
	Status                 string  `json:"status"`
	Grade                  string  `json:"grade,omitempty"`
	Score                  int     `json:"score,omitempty"`
	ActiveSource           string  `json:"active_source,omitempty"`
	ActiveMultiplier       float64 `json:"active_multiplier,omitempty"`
	TwoXBetter             bool    `json:"two_x_better"`
	DefaultUsefulness      string  `json:"default_usefulness,omitempty"`
	AgenticActivation      bool    `json:"agentic_activation"`
	ProviderObserved       string  `json:"provider_observed"`
	KernelWitnessed        string  `json:"kernel_witnessed"`
	ContextWitnessed       string  `json:"context_witnessed"`
	ExternalEngineObserved string  `json:"external_engine_observed"`
	Forecast               string  `json:"forecast"`
	Error                  string  `json:"error,omitempty"`
}

type cachevalueVCacheActionsDigest struct {
	Path            string `json:"path"`
	Status          string `json:"status"`
	Turns           int    `json:"turns"`
	FamilyCount     int    `json:"family_count"`
	Noop            int    `json:"noop"`
	Ready           int    `json:"ready"`
	Gated           int    `json:"gated"`
	TransportMode   string `json:"transport_mode"`
	TransportReady  bool   `json:"transport_ready"`
	TransportReason string `json:"transport_reason,omitempty"`
	Error           string `json:"error,omitempty"`
}

type cachevalueVCacheContextJoinDigest struct {
	Path               string `json:"path"`
	Status             string `json:"status"`
	FailureDomain      string `json:"failure_domain,omitempty"`
	Turns              int    `json:"turns"`
	Events             int    `json:"events"`
	TotalChanges       int    `json:"total_changes"`
	PlanningAttributed int    `json:"planning_attributed"`
	ProviderAttributed int    `json:"provider_attributed"`
	Error              string `json:"error,omitempty"`
}

type cachevalueVCacheObserveDigest struct {
	Path                string  `json:"path"`
	Status              string  `json:"status"`
	FailureDomain       string  `json:"failure_domain,omitempty"`
	Turns               int     `json:"turns"`
	FamilyCount         int     `json:"family_count"`
	TurnsReordered      bool    `json:"turns_reordered,omitempty"`
	OutOfOrderTurns     int     `json:"out_of_order_turns,omitempty"`
	HitRate             float64 `json:"hit_rate"`
	Multiplier          float64 `json:"multiplier"`
	SavedTokenEquiv     float64 `json:"saved_token_equiv"`
	CacheReadTokens     float64 `json:"cache_read_tokens"`
	CacheCreationTokens float64 `json:"cache_creation_tokens"`
	FalseWarm           int     `json:"false_warm"`
	FalseWarmRate       float64 `json:"false_warm_rate"`
	Error               string  `json:"error,omitempty"`
}

type cachevalueVCacheContextWitnessDigest struct {
	Path              string  `json:"path"`
	Status            string  `json:"status"`
	FailureDomain     string  `json:"failure_domain,omitempty"`
	Fixture           string  `json:"fixture,omitempty"`
	Wire              string  `json:"wire,omitempty"`
	Snapshot          string  `json:"snapshot,omitempty"`
	ReplayExit        int     `json:"replay_exit"`
	ScoreExit         int     `json:"score_exit"`
	ScoreStatus       string  `json:"score_status,omitempty"`
	ContextWitnessed  string  `json:"context_witnessed"`
	ContextEvents     int     `json:"context_events"`
	ContextShedTokens float64 `json:"context_shed_tokens"`
	Error             string  `json:"error,omitempty"`
}

type cachevalueStatusRow struct {
	Plane         string `json:"plane"`
	Component     string `json:"component"`
	Owner         string `json:"owner"`
	Dependency    string `json:"dependency"`
	Fidelity      string `json:"fidelity"`
	Evidence      string `json:"evidence"`
	Status        string `json:"status"`
	Selected      bool   `json:"selected,omitempty"`
	FailureDomain string `json:"failure_domain"`
	SessionImpact string `json:"session_impact"`
	Reason        string `json:"reason"`
	NextAction    string `json:"next_action,omitempty"`
}

type cachevalueStatusAttribution struct {
	Owners          map[string]cachevalueStatusBucket `json:"owners"`
	Fidelities      map[string]int                    `json:"fidelities"`
	Evidence        map[string]int                    `json:"evidence"`
	FailureDomains  map[string]cachevalueStatusBucket `json:"failure_domains"`
	ProblemOwners   []cachevalueStatusFinding         `json:"problem_owners,omitempty"`
	ProblemDomains  []cachevalueStatusFinding         `json:"problem_domains,omitempty"`
	ProblemFidelity []cachevalueStatusFinding         `json:"problem_fidelity,omitempty"`
}

type cachevalueStatusBucket struct {
	Total   int `json:"total"`
	Working int `json:"working,omitempty"`
	Problem int `json:"problem,omitempty"`
}

type cachevalueStatusFinding struct {
	Key        string `json:"key"`
	Problems   int    `json:"problems"`
	Working    int    `json:"working,omitempty"`
	Example    string `json:"example,omitempty"`
	NextAction string `json:"next_action,omitempty"`
}

func runCachevalueStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ ledger")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	contextBudget := fs.Uint64("context-budget-tokens", 0, "optional session context budget denominator")
	sessionPath := fs.String("session", "", "optional Claude Code transcript JSONL to diagnose against the cache rollup")
	artifactDir := fs.String("artifact-dir", "", "optional diagnostics bundle directory; loads known cache artifacts from conventional filenames when per-report flags are omitted")
	ablationReport := fs.String("ablation-report", "", "optional fak ablate --json report to fold cache effects and dropped subprocess arms into this status")
	headroomBench := fs.String("headroom-bench-report", "", "optional fak headroom bench --json report to fold compressor proof rows into this status")
	vcacheScore := fs.String("vcache-score-report", "", "optional fak vcache score --json report to fold provider/kernel/context/external/forecast cache planes into this status")
	vcacheActions := fs.String("vcache-actions-report", "", "optional fak vcache actions --json report to fold provider-cache action transport gating into this status")
	vcacheContextJoin := fs.String("vcache-context-join-report", "", "optional fak vcache context-join --json report to fold fak context-planning vs provider-cache attribution into this status")
	vcacheObserve := fs.String("vcache-observe-report", "", "optional fak vcache observe --json report to fold direct provider-cache telemetry and false-warm attribution into this status")
	vcacheContextWitness := fs.String("vcache-context-witness-report", "", "optional fak vcache context-witness --json report to fold fak-owned lossy context proof and replay status into this status")
	asJSON := fs.Bool("json", false, "emit machine-readable status")
	gate := fs.Bool("gate", false, "exit 1 when the folded verdict is at or worse than the --fail-on floor (opt-in; default exit stays 0)")
	failOn := fs.String("fail-on", "", "verdict floor for the gate: OK, PARTIAL, or INSUFFICIENT (implies --gate; defaults to PARTIAL when --gate is set)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue status: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}
	failOnSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "fail-on" {
			failOnSet = true
		}
	})
	gateFloor := strings.ToUpper(strings.TrimSpace(*failOn))
	gateActive := *gate || failOnSet
	if gateActive {
		if gateFloor == "" && !failOnSet {
			gateFloor = "PARTIAL"
		}
		if _, ok := cachevalueVerdictRank(gateFloor); !ok {
			fmt.Fprintf(stderr, "fak cachevalue status: --fail-on must be OK, PARTIAL, or INSUFFICIENT (got %q)\n", *failOn)
			return 2
		}
	}

	rep := buildCachevalueStatus(cachevalueStatusOptions{
		Ledger:                   *ledger,
		SavingsLedger:            *savingsLedger,
		UsageLedger:              *usageLedger,
		Since:                    *since,
		ContextBudgetTokens:      *contextBudget,
		HeadroomTimeout:          2 * time.Second,
		ArtifactDir:              *artifactDir,
		SessionPath:              *sessionPath,
		AblationReportPath:       *ablationReport,
		HeadroomBenchPath:        *headroomBench,
		VCacheScorePath:          *vcacheScore,
		VCacheActionsPath:        *vcacheActions,
		VCacheContextJoinPath:    *vcacheContextJoin,
		VCacheObservePath:        *vcacheObserve,
		VCacheContextWitnessPath: *vcacheContextWitness,
	}, time.Now().UTC())
	rc := 0
	if gateActive {
		if rc = cachevalueStatusGateExit(rep.Verdict, gateFloor); rc != 0 {
			fmt.Fprintf(stderr, "fak cachevalue status: gate: verdict %s is at or worse than --fail-on floor %s\n", rep.Verdict, gateFloor)
		}
	}
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue status: marshal: %v\n", err)
			return 1
		}
		return rc
	}
	renderCachevalueStatus(stdout, rep)
	return rc
}

func buildCachevalueStatus(opt cachevalueStatusOptions, now time.Time) cachevalueStatusReport {
	opt = resolveCachevalueStatusArtifactDir(opt)
	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(opt.Ledger), opt.Since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(opt.SavingsLedger), opt.Since)
	usage := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(opt.UsageLedger), opt.Since)
	value := cachevaluereport.FoldTwoTrackWithUsage(track1, track2, usage, now, cachevaluereport.FleetBenefitOptions{
		ContextBudgetTokens: opt.ContextBudgetTokens,
	})
	value.Since = opt.Since

	timeout := opt.HeadroomTimeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	head := headroom.BuildStatus(ctx)
	vc := defaultVCacheStatus()

	var rows []cachevalueStatusRow
	rows = append(rows, rowsFromCacheValueComponents(value.ComponentHealth)...)
	rows = append(rows, rowsFromHeadroomStatus(head)...)
	rows = append(rows, rowsFromVCacheStatus(vc)...)
	rows = append(rows, cacheAblationStatusRow())
	var sessionDigest *cachevalueSessionDigest
	if strings.TrimSpace(opt.SessionPath) != "" {
		session := sessionaudit.Analyze(opt.SessionPath)
		digest := cachevalueSessionDigestFromSession(session)
		sessionDigest = &digest
		rows = append(rows, rowsFromSessionDiagnosis(session)...)
	}
	var ablationDigest *cachevalueAblationDigest
	if strings.TrimSpace(opt.AblationReportPath) != "" {
		digest, ablationRows := loadCachevalueAblationStatus(opt.AblationReportPath)
		ablationDigest = &digest
		rows = append(rows, ablationRows...)
	}
	var headroomBenchDigest *cachevalueHeadroomBenchDigest
	if strings.TrimSpace(opt.HeadroomBenchPath) != "" {
		digest, benchRows := loadCachevalueHeadroomBenchStatus(opt.HeadroomBenchPath)
		headroomBenchDigest = &digest
		rows = append(rows, benchRows...)
	}
	var vcacheScoreDigest *cachevalueVCacheScoreDigest
	if strings.TrimSpace(opt.VCacheScorePath) != "" {
		digest, scoreRows := loadCachevalueVCacheScoreStatus(opt.VCacheScorePath)
		vcacheScoreDigest = &digest
		rows = append(rows, scoreRows...)
	}
	var vcacheActionsDigest *cachevalueVCacheActionsDigest
	if strings.TrimSpace(opt.VCacheActionsPath) != "" {
		digest, actionRows := loadCachevalueVCacheActionsStatus(opt.VCacheActionsPath)
		vcacheActionsDigest = &digest
		rows = append(rows, actionRows...)
	}
	var vcacheContextJoinDigest *cachevalueVCacheContextJoinDigest
	if strings.TrimSpace(opt.VCacheContextJoinPath) != "" {
		digest, contextRows := loadCachevalueVCacheContextJoinStatus(opt.VCacheContextJoinPath)
		vcacheContextJoinDigest = &digest
		rows = append(rows, contextRows...)
	}
	var vcacheObserveDigest *cachevalueVCacheObserveDigest
	if strings.TrimSpace(opt.VCacheObservePath) != "" {
		digest, observeRows := loadCachevalueVCacheObserveStatus(opt.VCacheObservePath)
		vcacheObserveDigest = &digest
		rows = append(rows, observeRows...)
	} else if strings.TrimSpace(opt.SessionPath) != "" {
		digest, observeRows := loadCachevalueVCacheObserveFromSession(opt.SessionPath)
		vcacheObserveDigest = &digest
		rows = append(rows, observeRows...)
	}
	var vcacheContextWitnessDigest *cachevalueVCacheContextWitnessDigest
	if strings.TrimSpace(opt.VCacheContextWitnessPath) != "" {
		digest, witnessRows := loadCachevalueVCacheContextWitnessStatus(opt.VCacheContextWitnessPath)
		vcacheContextWitnessDigest = &digest
		rows = append(rows, witnessRows...)
	}

	counts := cachevalueStatusCounts(rows)
	verdict, summary := cachevalueStatusVerdict(rows)
	attribution := cachevalueStatusAttributionFromRows(rows)
	return cachevalueStatusReport{
		Schema:      cachevalueStatusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Since:       opt.Since,
		Verdict:     verdict,
		Summary:     summary,
		Sources: cachevalueStatusSources{
			KernelLedger:               opt.Ledger,
			SavingsLedger:              opt.SavingsLedger,
			UsageLedger:                opt.UsageLedger,
			ArtifactDir:                opt.ArtifactDir,
			VCacheSnapshot:             statusVCacheSnapshotPath(),
			ContextSnapshot:            statusVCacheContextSnapshotPath(),
			HeadroomURL:                head.HeadroomURL,
			AblationReport:             opt.AblationReportPath,
			HeadroomBenchReport:        opt.HeadroomBenchPath,
			VCacheScoreReport:          opt.VCacheScorePath,
			VCacheActionsReport:        opt.VCacheActionsPath,
			VCacheContextJoinReport:    opt.VCacheContextJoinPath,
			VCacheObserveReport:        opt.VCacheObservePath,
			VCacheContextWitnessReport: opt.VCacheContextWitnessPath,
		},
		Counts:      counts,
		Attribution: attribution,
		Value: cachevalueValueDigest{
			Verdict:              value.Verdict,
			Finding:              value.Finding,
			NextAction:           value.NextAction,
			Track1Verdict:        value.Track1.Verdict,
			RejectedTierAccesses: value.Track1.RejectedTierAccesses,
			Track2Buckets:        len(value.Track2),
			FleetUsageRows:       value.FleetBenefit.UsageRows,
			DollarBlindRows:      value.DollarBlindRows,
		},
		Headroom: cachevalueHeadroomDigest{
			Selected:          head.Selected,
			HeadroomReachable: head.HeadroomReachable,
			GateStats:         head.GateStats,
		},
		VCache: cachevalueVCacheDigest{
			Status:                  vc.Status,
			LiveProvider:            vc.LiveProvider,
			ContextAPI:              vc.ContextAPI.Verifier,
			ProviderCalibration:     vc.ProviderCalibration.Verifier,
			ProviderActions:         vc.ProviderActions.Verifier,
			ProviderActionTransport: vc.ProviderActions.Transport,
			CodexOpenAI:             vc.CodexOpenAI.Verifier,
			RecentObservationError:  vc.RecentObservationError,
			RecentProviderStatus:    vcacheRecentProviderStatus(vc),
			RecentContextStatus:     vcacheRecentContextStatus(vc),
		},
		Session:              sessionDigest,
		Ablation:             ablationDigest,
		HeadroomBench:        headroomBenchDigest,
		VCacheScore:          vcacheScoreDigest,
		VCacheActions:        vcacheActionsDigest,
		VCacheContextJoin:    vcacheContextJoinDigest,
		VCacheObserve:        vcacheObserveDigest,
		VCacheContextWitness: vcacheContextWitnessDigest,
		Rows:                 rows,
		NextActions:          cachevalueStatusNextActions(rows),
	}
}

func resolveCachevalueStatusArtifactDir(opt cachevalueStatusOptions) cachevalueStatusOptions {
	dir := strings.TrimSpace(opt.ArtifactDir)
	if dir == "" {
		return opt
	}
	if strings.TrimSpace(opt.AblationReportPath) == "" {
		opt.AblationReportPath = firstExistingArtifact(dir, "ablation.json", "ablate.json", "fak-ablate.json")
	}
	if strings.TrimSpace(opt.HeadroomBenchPath) == "" {
		opt.HeadroomBenchPath = firstExistingArtifact(dir, "headroom-bench.json", "headroom.json", "fak-headroom-bench.json")
	}
	if strings.TrimSpace(opt.VCacheScorePath) == "" {
		opt.VCacheScorePath = firstExistingArtifact(dir, "vcache-score.json", "score.json", "fak-vcache-score.json")
	}
	if strings.TrimSpace(opt.VCacheActionsPath) == "" {
		opt.VCacheActionsPath = firstExistingArtifact(dir, "vcache-actions.json", "actions.json", "fak-vcache-actions.json")
	}
	if strings.TrimSpace(opt.VCacheContextJoinPath) == "" {
		opt.VCacheContextJoinPath = firstExistingArtifact(dir, "vcache-context-join.json", "context-join.json", "fak-vcache-context-join.json")
	}
	if strings.TrimSpace(opt.VCacheObservePath) == "" {
		opt.VCacheObservePath = firstExistingArtifact(dir, "vcache-observe.json", "observe.json", "fak-vcache-observe.json")
	}
	if strings.TrimSpace(opt.VCacheContextWitnessPath) == "" {
		opt.VCacheContextWitnessPath = firstExistingArtifact(dir, "vcache-context-witness.json", "context-witness.json", "fak-vcache-context-witness.json")
	}
	return opt
}

func firstExistingArtifact(dir string, names ...string) string {
	for _, name := range names {
		path := filepath.Join(dir, name)
		st, err := os.Stat(path)
		if err == nil && !st.IsDir() {
			return path
		}
	}
	return ""
}

func rowsFromCacheValueComponents(in []cachevaluereport.ComponentHealth) []cachevalueStatusRow {
	out := make([]cachevalueStatusRow, 0, len(in))
	for _, c := range in {
		out = append(out, cachevalueStatusRow{
			Plane:         c.Plane,
			Component:     c.Component,
			Owner:         cachevalueNonEmpty(c.Owner, "unknown"),
			Dependency:    cachevalueComponentDependency(c.Component),
			Fidelity:      cachevalueNonEmpty(c.Fidelity, "unknown"),
			Evidence:      cachevalueNonEmpty(c.Evidence, "unknown"),
			Status:        cachevalueNonEmpty(c.Status, "unknown"),
			FailureDomain: cachevalueFailureDomain(c.Owner, c.Component),
			SessionImpact: cachevalueComponentImpact(c.Component),
			Reason:        c.Reason,
			NextAction:    c.NextAction,
		})
	}
	return out
}

func rowsFromHeadroomStatus(rep headroom.StatusReport) []cachevalueStatusRow {
	out := make([]cachevalueStatusRow, 0, len(rep.Plugins))
	for _, p := range rep.Plugins {
		out = append(out, cachevalueStatusRow{
			Plane:         "context_compression",
			Component:     "headroom_plugin:" + p.Name,
			Owner:         cachevalueNonEmpty(p.Owner, "unknown"),
			Dependency:    cachevalueNonEmpty(p.Dependency, "unknown"),
			Fidelity:      cachevalueNonEmpty(p.Fidelity, "unknown"),
			Evidence:      cachevalueNonEmpty(p.Evidence, "unknown"),
			Status:        cachevalueNonEmpty(p.Status, "unknown"),
			Selected:      p.Selected,
			FailureDomain: cachevalueFailureDomain(p.Owner, p.Name),
			SessionImpact: headroomSessionImpact(p),
			Reason:        p.Reason,
			NextAction:    headroomNextAction(p),
		})
	}
	return out
}

func rowsFromVCacheStatus(rep vcacheStatusReport) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{
		{
			Plane:         "managed_context",
			Component:     "vcache_context_api",
			Owner:         "fak",
			Dependency:    "gateway_http_mcp",
			Fidelity:      "passive",
			Evidence:      "configured",
			Status:        statusReady(rep.ContextAPI.Verifier),
			FailureDomain: "fak",
			SessionImpact: "context sizing is fak-owned; inspect guard/serve context events before blaming provider cache",
			Reason:        rep.ContextAPI.Reason,
			NextAction:    rep.ContextAPI.NoKeyWitnessCommand,
		},
		{
			Plane:         "provider_prompt_cache",
			Component:     "provider_calibration",
			Owner:         "fak",
			Dependency:    "operator_provider_probe_samples",
			Fidelity:      "passive",
			Evidence:      "configured",
			Status:        statusReady(rep.ProviderCalibration.Verifier),
			FailureDomain: "fak_or_provider_samples",
			SessionImpact: "bad calibration points to stale/missing probe samples, not fak-native KV",
			Reason:        rep.ProviderCalibration.Reason,
			NextAction:    rep.ProviderCalibration.CLI,
		},
		{
			Plane:         "provider_prompt_cache",
			Component:     "provider_actions",
			Owner:         "fak",
			Dependency:    "provider_action_transport",
			Fidelity:      "lossless",
			Evidence:      "DECISION",
			Status:        providerActionStatus(rep.ProviderActions),
			FailureDomain: "provider_transport",
			SessionImpact: "warm/miss actions are planned by fak, but spendful provider transport must be witnessed",
			Reason:        rep.ProviderActions.Reason,
			NextAction:    rep.ProviderActions.CLI,
		},
		{
			Plane:         "provider_prompt_cache",
			Component:     "codex_openai_telemetry",
			Owner:         "provider",
			Dependency:    "provider_usage_fields",
			Fidelity:      "passive",
			Evidence:      "OBSERVED",
			Status:        statusReady(rep.CodexOpenAI.Verifier),
			FailureDomain: "provider_or_cli_telemetry",
			SessionImpact: "missing cached-token fields means provider caching may be working but unobservable",
			Reason:        rep.CodexOpenAI.Reason,
			NextAction:    "fak vcache prove-telemetry --file FILE --json",
		},
	}
	if rep.RecentObservationError != "" {
		return append(rows, cachevalueStatusRow{
			Plane:         "provider_prompt_cache",
			Component:     "recent_vcache_snapshot",
			Owner:         "fak",
			Dependency:    "local_snapshot",
			Fidelity:      "passive",
			Evidence:      "OBSERVED",
			Status:        "unavailable",
			FailureDomain: "snapshot",
			SessionImpact: "recent provider-cache attribution is unavailable because the local snapshot could not be read",
			Reason:        rep.RecentObservationError,
			NextAction:    "fak vcache observe --telemetry FILE --json",
		})
	}
	if rep.RecentObservation == nil {
		return append(rows, cachevalueStatusRow{
			Plane:         "provider_prompt_cache",
			Component:     "recent_vcache_snapshot",
			Owner:         "provider",
			Dependency:    "local_snapshot",
			Fidelity:      "passive",
			Evidence:      "OBSERVED",
			Status:        "missing",
			FailureDomain: "provider_telemetry",
			SessionImpact: "no recent provider-cache snapshot is present, so a bad session cannot be attributed to provider hit/miss behavior yet",
			Reason:        "no persisted vCache provider/context snapshot found",
			NextAction:    "fak vcache observe --transcript FILE --json",
		})
	}
	recent := rep.RecentObservation
	rows = append(rows, cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "recent_provider_cache_observation",
		Owner:         "provider",
		Dependency:    "local_snapshot",
		Fidelity:      "passive",
		Evidence:      "OBSERVED",
		Status:        recentProviderObservationStatus(recent.ProviderStatus),
		FailureDomain: "provider_telemetry",
		SessionImpact: "provider-cache miss or false-warm behavior is provider telemetry, not fak-native KV",
		Reason:        recentProviderObservationReason(*recent),
		NextAction:    "fak vcache context-join --events FILE",
	})
	rows = append(rows, cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "recent_context_observation",
		Owner:         "fak",
		Dependency:    "local_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      "WITNESSED",
		Status:        recentContextObservationStatus(recent.ContextStatus),
		FailureDomain: "fak_context_planner",
		SessionImpact: "managed-context drops/compaction are fak-authored and separate from natural provider cache misses",
		Reason:        recent.ContextReason,
		NextAction:    "fak vcache context-witness --json",
	})
	return rows
}
