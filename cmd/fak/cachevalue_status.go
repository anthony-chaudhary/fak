package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
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
	Verdict         string `json:"verdict"`
	Finding         string `json:"finding,omitempty"`
	NextAction      string `json:"next_action,omitempty"`
	Track1Verdict   string `json:"track1_verdict"`
	Track2Buckets   int    `json:"track2_buckets"`
	FleetUsageRows  int    `json:"fleet_usage_rows"`
	DollarBlindRows int    `json:"dollar_blind_rows,omitempty"`
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
			Verdict:         value.Verdict,
			Finding:         value.Finding,
			NextAction:      value.NextAction,
			Track1Verdict:   value.Track1.Verdict,
			Track2Buckets:   len(value.Track2),
			FleetUsageRows:  value.FleetBenefit.UsageRows,
			DollarBlindRows: value.DollarBlindRows,
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

func cachevalueSessionDigestFromSession(s sessionaudit.Session) cachevalueSessionDigest {
	total := s.TotalInputTokens
	if total == 0 {
		total = s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
	}
	status, domain, finding := cachevalueSessionHeadline(s, total)
	return cachevalueSessionDigest{
		Path:               s.Path,
		Session:            nonEmptySessionName(s),
		Status:             status,
		LikelyDomain:       domain,
		Finding:            finding,
		AssistantTurns:     s.AssistantTurns,
		ToolCalls:          s.NToolUse,
		ReadOnlyToolFrac:   s.ReadOnlyFrac,
		InputTokens:        s.Tokens.Input,
		OutputTokens:       s.Tokens.Output,
		CacheReadTokens:    s.Tokens.CacheRead,
		CacheCreateTokens:  s.Tokens.CacheCreate,
		TotalContextTokens: total,
		CacheHitFrac:       s.CacheHitFrac,
		IORatio:            s.IORatio,
		CostUSD:            s.CostUSD,
		Error:              s.Error,
	}
}

func rowsFromSessionDiagnosis(s sessionaudit.Session) []cachevalueStatusRow {
	if s.Error != "" {
		return []cachevalueStatusRow{{
			Plane:         "session",
			Component:     "session_transcript",
			Owner:         "unknown",
			Dependency:    "session_file",
			Fidelity:      "passive",
			Evidence:      "MISSING",
			Status:        "unavailable",
			FailureDomain: "transcript",
			SessionImpact: "session diagnosis cannot attribute cache behavior because the transcript could not be read",
			Reason:        s.Error,
			NextAction:    "pass a readable Claude Code transcript JSONL to --session",
		}}
	}
	total := s.TotalInputTokens
	if total == 0 {
		total = s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
	}
	rows := []cachevalueStatusRow{
		sessionProviderCacheRow(s, total),
		sessionWorkloadRow(s, total),
		sessionFakContextVisibilityRow(s),
	}
	return rows
}

func sessionProviderCacheRow(s sessionaudit.Session, total int64) cachevalueStatusRow {
	status := "no_provider_cache_evidence"
	reason := fmt.Sprintf("session has %d input, %d cache-read, and %d cache-create token(s)", s.Tokens.Input, s.Tokens.CacheRead, s.Tokens.CacheCreate)
	impact := "provider prompt-cache counters are absent, so a bad session may be prompt churn, an uncached provider path, or missing telemetry"
	next := "fak vcache observe --transcript " + quotePathForHint(s.Path) + " --json"
	if total == 0 || s.AssistantTurns == 0 {
		status = "no_usage"
		impact = "transcript has no assistant usage rows, so cache behavior is not attributable from this file"
		next = "verify the session transcript contains assistant message usage blocks"
	} else if s.Tokens.CacheRead > 0 {
		status = "observed"
		impact = "provider cache-read tokens are present; if the session still went badly, inspect workload pressure and fak context rows next"
		next = ""
	} else if s.Tokens.CacheCreate > 0 {
		status = "cold_write_only"
		impact = "provider cache writes happened without later reads; this points at cold start, prefix churn, or TTL/window behavior outside fak-native KV"
		next = "fak vcache context-join --events FILE"
	}
	return cachevalueStatusRow{
		Plane:         "session_provider_prompt_cache",
		Component:     "session_provider_cache",
		Owner:         "provider",
		Dependency:    "session_transcript_usage",
		Fidelity:      "lossless",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: sessionProviderFailureDomain(status),
		SessionImpact: impact,
		Reason:        reason,
		NextAction:    next,
	}
}

func sessionWorkloadRow(s sessionaudit.Session, total int64) cachevalueStatusRow {
	status := "measured"
	if total == 0 && s.Tokens.Output == 0 {
		status = "no_usage"
	} else if sessionHighContextPressure(s, total) {
		status = "high_pressure"
	}
	reason := fmt.Sprintf("total_context=%d output=%d io_ratio=%s tool_result_chars=%d read_only_tool_frac=%s",
		total, s.Tokens.Output, fmtFloatPtr(s.IORatio), s.ToolResultChars, fmtFloatPtr(s.ReadOnlyFrac))
	next := ""
	if status == "high_pressure" {
		next = "fak headroom bench --via native"
	}
	return cachevalueStatusRow{
		Plane:         "session_workload",
		Component:     "session_context_pressure",
		Owner:         "workload",
		Dependency:    "session_transcript_usage",
		Fidelity:      "passive",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: "workload",
		SessionImpact: "large tool/prompt context can make a session bad even when cache machinery is working; this is not by itself a fak cache fault",
		Reason:        reason,
		NextAction:    next,
	}
}

func sessionFakContextVisibilityRow(s sessionaudit.Session) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "session_fak_context",
		Component:     "session_fak_context_events",
		Owner:         "fak",
		Dependency:    "vcache_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      "MISSING",
		Status:        "not_observed_from_transcript",
		FailureDomain: "evidence_gap",
		SessionImpact: "Claude transcripts show provider usage counters but do not prove fak context drops/compaction; use a fak context snapshot before blaming fak context planning",
		Reason:        "transcript " + nonEmptySessionName(s) + " has no fak_context_* counters in the sessionaudit shape",
		NextAction:    "fak vcache context-witness --json",
	}
}

func cachevalueSessionHeadline(s sessionaudit.Session, total int64) (status, domain, finding string) {
	if s.Error != "" {
		return "unavailable", "transcript", s.Error
	}
	if total == 0 || s.AssistantTurns == 0 {
		return "no_usage", "transcript", "no assistant usage rows were found, so cache behavior is not attributable"
	}
	if s.Tokens.CacheRead == 0 && s.Tokens.CacheCreate > 0 {
		return "partial", "provider_prompt_cache", "provider cache writes occurred but no cache reads were observed; suspect cold start, prefix churn, or TTL/window behavior"
	}
	if s.Tokens.CacheRead == 0 {
		return "partial", "provider_telemetry", "no provider cache-read evidence was observed in the transcript"
	}
	if sessionHighContextPressure(s, total) {
		return "partial", "workload", "provider cache reads are present, but context/workload pressure is high"
	}
	return "observed", "provider_prompt_cache", "provider cache-read evidence is present in the session transcript"
}

func sessionHighContextPressure(s sessionaudit.Session, total int64) bool {
	if total >= 200_000 {
		return true
	}
	return s.IORatio != nil && *s.IORatio >= 100
}

func sessionProviderFailureDomain(status string) string {
	switch status {
	case "observed":
		return "provider"
	case "cold_write_only":
		return "provider_cache_window"
	case "no_usage":
		return "transcript"
	default:
		return "provider_telemetry_or_prompt_churn"
	}
}

func loadCachevalueAblationStatus(path string) (cachevalueAblationDigest, []cachevalueStatusRow) {
	digest := cachevalueAblationDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{ablationReportUnavailableRow(path, err.Error())}
	}
	var rep ablate.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{ablationReportUnavailableRow(path, "decode: "+err.Error())}
	}
	rows := rowsFromAblationReport(&rep, path)
	digest.Runs = len(rep.Runs)
	digest.DroppedArms = len(rep.Dropped)
	for _, drop := range rep.Dropped {
		if strings.TrimSpace(drop.Stage) != "" || drop.ExitCode != nil ||
			strings.TrimSpace(drop.StderrTail) != "" || strings.TrimSpace(drop.StdoutTail) != "" ||
			strings.TrimSpace(drop.ExpectedWorkloadHash) != "" || strings.TrimSpace(drop.ActualWorkloadHash) != "" ||
			drop.DurationSeconds > 0 {
			digest.DroppedWithDiagnostics++
		}
		switch strings.TrimSpace(drop.Stage) {
		case "workload_hash":
			digest.DroppedWorkloadMismatches++
		case "child_exit":
			digest.DroppedChildExits++
		}
	}
	for _, run := range rep.Runs {
		for _, effect := range run.CacheEffects {
			digest.CacheEffects++
			switch strings.ToLower(strings.TrimSpace(effect.Status)) {
			case "active":
				digest.ActiveEffects++
			case "unavailable":
				digest.UnavailableEffects++
			}
		}
	}
	switch {
	case len(rep.Runs) == 0:
		digest.Status = "missing"
	case digest.DroppedArms > 0 || digest.UnavailableEffects > 0:
		digest.Status = "partial"
	default:
		digest.Status = "measured"
	}
	return digest, rows
}

func ablationReportUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "ablation_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "ablation_artifact",
		SessionImpact: "cache ablation evidence could not be folded, so subprocess/cache-effect attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak ablate --sweep " + strings.Join(cacheAblationFeatures(), ",") + " --json",
	}
}

func rowsFromAblationReport(rep *ablate.Report, path string) []cachevalueStatusRow {
	if rep == nil {
		return []cachevalueStatusRow{ablationReportUnavailableRow(path, "nil report")}
	}
	rows := []cachevalueStatusRow{{
		Plane:         "diagnostics",
		Component:     "ablation_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "WITNESSED",
		Status:        ablationReportStatus(rep),
		FailureDomain: "fak_diagnostics",
		SessionImpact: "ablation artifact is folded into this status so cache-effect and subprocess-arm holes are visible",
		Reason:        fmt.Sprintf("%d arm(s), %d dropped arm(s), workload=%s", len(rep.Runs), len(rep.Dropped), rep.WorkloadHash),
		NextAction:    ablationReportNextAction(rep),
	}}
	for _, run := range rep.Runs {
		for _, effect := range run.CacheEffects {
			rows = append(rows, rowFromAblationEffect(run.ArmID, effect))
		}
	}
	for _, drop := range rep.Dropped {
		rows = append(rows, rowFromDroppedAblationArm(drop))
	}
	return rows
}

func rowFromAblationEffect(armID string, effect ablate.CacheEffect) cachevalueStatusRow {
	status := strings.ToLower(strings.TrimSpace(effect.Status))
	if status == "" {
		status = "unknown"
	}
	component := "ablation_effect:" + nonEmpty(armID, "unknown_arm") + ":" + nonEmpty(effect.Feature, effect.Component)
	return cachevalueStatusRow{
		Plane:         nonEmpty(effect.Plane, "diagnostics"),
		Component:     component,
		Owner:         nonEmpty(effect.Owner, "unknown"),
		Dependency:    nonEmpty(effect.Dependency, "unknown"),
		Fidelity:      nonEmpty(effect.Fidelity, "unknown"),
		Evidence:      nonEmpty(effect.Evidence, "unknown"),
		Status:        status,
		FailureDomain: cachevalueFailureDomain(effect.Owner, effect.Component),
		SessionImpact: ablationEffectImpact(effect),
		Reason:        fmt.Sprintf("arm=%s feature=%s component=%s: %s", armID, effect.Feature, effect.Component, effect.Reason),
		NextAction:    ablationEffectNextAction(effect),
	}
}

func rowFromDroppedAblationArm(drop ablate.DroppedArm) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "ablation_dropped_arm:" + nonEmpty(drop.ArmID, "unknown"),
		Owner:         "fak",
		Dependency:    "subprocess_reexec",
		Fidelity:      "diagnostic",
		Evidence:      "WITNESSED",
		Status:        "dropped",
		FailureDomain: droppedAblationFailureDomain(drop),
		SessionImpact: "one ablation child failed, so that cache/fak feature has no measured arm in this report",
		Reason:        droppedAblationReason(drop),
		NextAction:    droppedAblationNextAction(drop),
	}
}

func droppedAblationFailureDomain(drop ablate.DroppedArm) string {
	switch strings.TrimSpace(drop.Stage) {
	case "workload_hash":
		return "fak_diagnostics_workload_guard"
	case "child_exit":
		return "fak_diagnostics_subprocess_exit"
	case "decode_stdout":
		return "fak_diagnostics_subprocess_codec"
	default:
		return "fak_diagnostics_subprocess"
	}
}

func droppedAblationReason(drop ablate.DroppedArm) string {
	parts := []string{}
	if strings.TrimSpace(drop.Stage) != "" {
		parts = append(parts, "stage="+drop.Stage)
	}
	if drop.ExitCode != nil {
		parts = append(parts, fmt.Sprintf("exit=%d", *drop.ExitCode))
	}
	if drop.DurationSeconds > 0 {
		parts = append(parts, fmt.Sprintf("duration=%.3fs", drop.DurationSeconds))
	}
	if strings.TrimSpace(drop.StderrTail) != "" {
		parts = append(parts, "stderr="+drop.StderrTail)
	}
	if strings.TrimSpace(drop.StdoutTail) != "" {
		parts = append(parts, "stdout="+drop.StdoutTail)
	}
	if strings.TrimSpace(drop.ExpectedWorkloadHash) != "" || strings.TrimSpace(drop.ActualWorkloadHash) != "" {
		parts = append(parts, fmt.Sprintf("hash child=%s parent=%s", cachevalueEmptyDash(drop.ActualWorkloadHash), cachevalueEmptyDash(drop.ExpectedWorkloadHash)))
	}
	if strings.TrimSpace(drop.Reason) != "" {
		parts = append(parts, drop.Reason)
	}
	return strings.Join(parts, " ")
}

func droppedAblationNextAction(drop ablate.DroppedArm) string {
	switch strings.TrimSpace(drop.Stage) {
	case "workload_hash":
		return "inspect trace/session replay drift before comparing arm " + nonEmpty(drop.ArmID, "unknown")
	case "decode_stdout":
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown") + " and inspect child stdout/stderr"
	case "child_exit":
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown") + " after fixing the child process failure"
	default:
		return "rerun fak ablate for arm " + nonEmpty(drop.ArmID, "unknown")
	}
}

func ablationReportStatus(rep *ablate.Report) string {
	if rep == nil || len(rep.Runs) == 0 {
		return "missing"
	}
	if len(rep.Dropped) > 0 {
		return "partial"
	}
	return "measured"
}

func ablationReportNextAction(rep *ablate.Report) string {
	if rep == nil || len(rep.Runs) == 0 {
		return "fak ablate --sweep " + strings.Join(cacheAblationFeatures(), ",") + " --json"
	}
	if len(rep.Dropped) > 0 {
		return "rerun dropped ablation arms before treating the sweep as complete"
	}
	return ""
}

func ablationEffectImpact(effect ablate.CacheEffect) string {
	switch strings.ToLower(strings.TrimSpace(effect.Owner)) {
	case "provider":
		return "ablation effect belongs to provider prompt-cache behavior, not fak-native cache machinery"
	case "external":
		return "ablation effect depends on an external cache/compression sidecar"
	case "fak":
		return "ablation effect is fak-owned; compare this arm against the baseline before blaming provider cache"
	default:
		return "ablation effect owner is unknown"
	}
}

func ablationEffectNextAction(effect ablate.CacheEffect) string {
	switch strings.ToLower(strings.TrimSpace(effect.Status)) {
	case "unavailable":
		return "check dependency " + nonEmpty(effect.Dependency, "unknown")
	case "no-op":
		return "confirm the selected engine/component can exercise " + nonEmpty(effect.Feature, effect.Component)
	default:
		return ""
	}
}

func loadCachevalueHeadroomBenchStatus(path string) (cachevalueHeadroomBenchDigest, []cachevalueStatusRow) {
	digest := cachevalueHeadroomBenchDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{headroomBenchUnavailableRow(path, err.Error())}
	}
	var rep headroom.BenchReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{headroomBenchUnavailableRow(path, "decode: "+err.Error())}
	}
	status := headroomBenchStatus(rep)
	owner, dependency, fidelity, evidence := headroomBenchReportAttribution(rep)
	digest.Status = status
	digest.Compressor = rep.Compressor
	digest.Owner = owner
	digest.Dependency = dependency
	digest.Fidelity = fidelity
	digest.Evidence = evidence
	digest.Samples = len(rep.Samples)
	digest.OrigTotal = rep.OrigTotal
	digest.NewTotal = rep.NewTotal
	digest.SavedRatio = rep.Saved
	digest.Reason = rep.Reason
	return digest, rowsFromHeadroomBenchReport(rep, path)
}

func headroomBenchUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "context_compression",
		Component:     "headroom_bench_report",
		Owner:         "unknown",
		Dependency:    "local_json_report",
		Fidelity:      "recoverable",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "headroom_bench_artifact",
		SessionImpact: "headroom compression proof could not be folded, so compressor behavior is not evidenced in this rollup",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak headroom bench --via native --json",
	}
}

func rowsFromHeadroomBenchReport(rep headroom.BenchReport, path string) []cachevalueStatusRow {
	owner, dependency, fidelity, evidence := headroomBenchReportAttribution(rep)
	rows := []cachevalueStatusRow{{
		Plane:         "context_compression",
		Component:     "headroom_bench_report",
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        headroomBenchStatus(rep),
		FailureDomain: headroomBenchFailureDomain(rep),
		SessionImpact: "headroom bench proves realized compressor savings on a representative or supplied corpus; use it to separate compressor value from provider cache behavior",
		Reason:        headroomBenchReportReason(rep, path),
		NextAction:    headroomBenchNextAction(rep),
	}}
	for _, sample := range rep.Samples {
		rows = append(rows, rowFromHeadroomBenchSample(rep.Compressor, sample))
	}
	return rows
}

func rowFromHeadroomBenchSample(compressor string, sample headroom.BenchSample) cachevalueStatusRow {
	owner, dependency, fidelity, evidence := headroomBenchAttribution(compressor)
	status := headroomBenchSampleStatus(sample)
	return cachevalueStatusRow{
		Plane:         "context_compression",
		Component:     "headroom_bench_sample:" + nonEmpty(sample.Name, "unnamed"),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        status,
		FailureDomain: headroomBenchOwnerDomain(owner, compressor),
		SessionImpact: "per-sample compressor evidence; no_effect on one sample can be normal, aggregate status decides whether the compressor proof worked",
		Reason:        headroomBenchSampleReason(sample),
	}
}

func headroomBenchAttribution(compressor string) (owner, dependency, fidelity, evidence string) {
	switch strings.ToLower(strings.TrimSpace(compressor)) {
	case headroom.HeadroomName:
		return "external", "external_http_sidecar", "recoverable", "observed"
	case headroom.NoopName:
		return "fak", "none", "no-op", "configured"
	case headroom.NativeName:
		return "fak", "in_process", "recoverable", "witnessed"
	default:
		return "unknown", "registered_plugin", "unknown", "configured"
	}
}

func headroomBenchReportAttribution(rep headroom.BenchReport) (owner, dependency, fidelity, evidence string) {
	owner, dependency, fidelity, evidence = headroomBenchAttribution(rep.Compressor)
	if strings.TrimSpace(rep.Owner) != "" {
		owner = rep.Owner
	}
	if strings.TrimSpace(rep.Dependency) != "" {
		dependency = rep.Dependency
	}
	if strings.TrimSpace(rep.Fidelity) != "" {
		fidelity = rep.Fidelity
	}
	if strings.TrimSpace(rep.Evidence) != "" {
		evidence = rep.Evidence
	}
	return owner, dependency, fidelity, evidence
}

func headroomBenchStatus(rep headroom.BenchReport) string {
	if strings.TrimSpace(rep.Status) != "" {
		return rep.Status
	}
	switch {
	case rep.Compressor == "" || len(rep.Samples) == 0 || rep.OrigTotal <= 0:
		return "missing"
	case strings.EqualFold(rep.Compressor, headroom.NoopName):
		return "no-op"
	case rep.Saved <= 0:
		return "no_saving"
	default:
		return "measured"
	}
}

func headroomBenchFailureDomain(rep headroom.BenchReport) string {
	owner, _, _, _ := headroomBenchReportAttribution(rep)
	switch headroomBenchStatus(rep) {
	case "unavailable":
		return headroomBenchOwnerDomain(owner, rep.Compressor) + "_unavailable"
	case "error":
		return headroomBenchOwnerDomain(owner, rep.Compressor) + "_error"
	}
	if rep.Saved > 0 || strings.EqualFold(rep.Compressor, headroom.NoopName) {
		return headroomBenchOwnerDomain(owner, rep.Compressor)
	}
	return headroomBenchOwnerDomain(owner, rep.Compressor) + "_or_corpus"
}

func headroomBenchOwnerDomain(owner, compressor string) string {
	switch owner {
	case "external":
		return "external:" + nonEmpty(compressor, "headroom")
	case "fak":
		return "fak"
	default:
		return "unknown"
	}
}

func headroomBenchNextAction(rep headroom.BenchReport) string {
	switch headroomBenchStatus(rep) {
	case "missing":
		return "rerun fak headroom bench --via native --json"
	case "unavailable":
		return "start headroom proxy or select FAK_COMPRESSOR=native/noop"
	case "error":
		return "inspect the compressor error and rerun fak headroom bench"
	case "no_saving":
		return "try fak headroom bench --via native on a representative captured tool-output corpus"
	default:
		return ""
	}
}

func headroomBenchReportReason(rep headroom.BenchReport, path string) string {
	reason := strings.TrimSpace(rep.Reason)
	if reason == "" {
		reason = fmt.Sprintf("compressor=%s samples=%d orig=%d new=%d saved=%.2f%%", rep.Compressor, len(rep.Samples), rep.OrigTotal, rep.NewTotal, rep.Saved*100)
	}
	return fmt.Sprintf("%s source=%s", reason, path)
}

func headroomBenchSampleStatus(sample headroom.BenchSample) string {
	if strings.TrimSpace(sample.Status) != "" {
		return sample.Status
	}
	if sample.Saved > 0 {
		return "saved"
	}
	return "no_effect"
}

func headroomBenchSampleReason(sample headroom.BenchSample) string {
	reason := strings.TrimSpace(sample.Reason)
	if reason == "" {
		reason = "sample compressor outcome"
	}
	return fmt.Sprintf("kind=%s codec=%s orig=%d new=%d saved=%.2f%% detail=%s", sample.Kind, sample.Codec, sample.OrigLen, sample.NewLen, sample.Saved*100, reason)
}

func loadCachevalueVCacheScoreStatus(path string) (cachevalueVCacheScoreDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheScoreDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheScoreUnavailableRow(path, err.Error())}
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheScoreUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = nonEmpty(rep.Status, vcacheScoreReportStatus(rep))
	digest.Grade = rep.Grade
	digest.Score = rep.Score
	digest.ActiveSource = rep.ActiveSource
	digest.ActiveMultiplier = rep.ActiveMultiplier
	digest.TwoXBetter = rep.TwoXBetter
	digest.DefaultUsefulness = rep.DefaultUsefulness.Verdict
	digest.AgenticActivation = rep.AgenticActivation.Active
	digest.ProviderObserved = vcacheScorePlaneLabel(rep.Planes.ProviderObserved)
	digest.KernelWitnessed = vcacheScorePlaneLabel(rep.Planes.KernelWitnessed)
	digest.ContextWitnessed = vcacheScorePlaneLabel(rep.Planes.ContextWitnessed)
	digest.ExternalEngineObserved = vcacheScorePlaneLabel(rep.Planes.ExternalEngineObserved)
	digest.Forecast = vcacheScorePlaneLabel(rep.Planes.Forecast)
	return digest, rowsFromVCacheScoreReport(rep, path)
}

func vcacheScoreUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "vcache_score",
		Component:     "vcache_score_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_score_artifact",
		SessionImpact: "vCache score evidence could not be folded, so provider/fak/external/forecast attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache score --json",
	}
}

func rowsFromVCacheScoreReport(rep vcachescore.Report, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "vcache_score",
		Component:     "vcache_score_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      vcacheScoreReportEvidence(rep),
		Status:        vcacheScoreReportStatus(rep),
		FailureDomain: "vcache_score",
		SessionImpact: "vCache score artifact separates provider rebates, fak-owned KV/context value, external engine cache, and forecasted value",
		Reason:        vcacheScoreReportReason(rep, path),
		NextAction:    vcacheScoreReportNextAction(rep),
	}}
	rows = append(rows,
		rowFromVCacheScorePlane("provider_prompt_cache", "vcache_score_provider_observed", "provider", "provider_usage_snapshot", "lossless", "provider_telemetry", rep.Planes.ProviderObserved, "provider-reported cache economics; missing here points at telemetry/snapshot gaps, not fak-native KV", "fak vcache observe --transcript FILE --json"),
		rowFromVCacheScorePlane("kernel_tool_cache", "vcache_score_kernel_witnessed", "fak", "cachevalue_ledger", "lossless", "fak", rep.Planes.KernelWitnessed, "pure-fak KV reuse witness; missing here means no fak-owned reuse value was supplied to the scorecard", "fak vcache score --kernel-ledger default --json"),
		rowFromVCacheScorePlane("managed_context", "vcache_score_context_witnessed", "fak", "vcache_context_snapshot", "lossy", "fak_context_planner", rep.Planes.ContextWitnessed, "O(1) context/drop witness; separates lossy context shrink from provider cache hit/miss behavior", "fak vcache context-witness --json"),
		rowFromVCacheScorePlane("external_engine_cache", "vcache_score_external_engine_observed", "external", "external_engine_snapshot", "passive", "external_engine", rep.Planes.ExternalEngineObserved, "external serving-engine prefix-cache evidence; failures here belong to the driver or sidecar, not fak-native KV", "fak vcache score --external-engine-events N --external-engine-hit-rate F --json"),
		rowFromVCacheScorePlane("forecast", "vcache_score_forecast", "fak", "score_model", "forecast", "forecast", rep.Planes.Forecast, "deterministic score forecast; useful for planning but not a realized provider/fak/external witness", "fak vcache score --telemetry FILE --json"),
	)
	return rows
}

func rowFromVCacheScorePlane(plane, component, owner, dependency, fidelity, failureDomain string, p vcachescore.PlaneValueReport, impact, missingAction string) cachevalueStatusRow {
	status := vcacheScorePlaneStatus(p)
	next := ""
	if status == "missing" {
		next = missingAction
	}
	return cachevalueStatusRow{
		Plane:         plane,
		Component:     component,
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      vcacheScorePlaneEvidence(p),
		Status:        status,
		FailureDomain: failureDomain,
		SessionImpact: impact,
		Reason:        vcacheScorePlaneReason(p),
		NextAction:    next,
	}
}

func vcacheScoreReportEvidence(rep vcachescore.Report) string {
	if hasVCacheScoreRealizedPlane(rep) {
		return "WITNESSED"
	}
	if rep.Planes.Forecast.Available {
		return "FORECAST"
	}
	return "MISSING"
}

func vcacheScoreReportStatus(rep vcachescore.Report) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && strings.TrimSpace(rep.Status) == "" &&
		rep.Planes == (vcachescore.PlaneReport{}):
		return "missing"
	case !hasVCacheScoreRealizedPlane(rep) && rep.Planes.Forecast.Available:
		return "forecast_only"
	case strings.EqualFold(rep.DefaultUsefulness.Verdict, "not_ready"):
		return "not_ready"
	case strings.EqualFold(rep.DefaultUsefulness.Verdict, "partial"):
		return "partial"
	case !rep.TwoXBetter:
		return "not_2x"
	default:
		return "measured"
	}
}

func vcacheScoreReportReason(rep vcachescore.Report, path string) string {
	return fmt.Sprintf("status=%s grade=%s score=%d active=%s multiplier=%.2fx two_x=%v default=%s activation=%v source=%s",
		nonEmpty(rep.Status, "-"),
		nonEmpty(rep.Grade, "-"),
		rep.Score,
		nonEmpty(rep.ActiveSource, "-"),
		rep.ActiveMultiplier,
		rep.TwoXBetter,
		nonEmpty(rep.DefaultUsefulness.Verdict, "-"),
		rep.AgenticActivation.Active,
		path)
}

func vcacheScoreReportNextAction(rep vcachescore.Report) string {
	switch vcacheScoreReportStatus(rep) {
	case "missing":
		return "fak vcache score --json"
	case "forecast_only":
		return "supply provider telemetry, cachevalue ledger, context snapshot, or external engine evidence to fak vcache score"
	case "not_2x":
		return strings.Join(nonEmptyActions(rep.Actions), "; ")
	case "not_ready", "partial":
		if len(rep.Actions) > 0 {
			return strings.Join(nonEmptyActions(rep.Actions), "; ")
		}
		return "add realized fak/provider/external plane evidence before treating vCache as default-ready"
	default:
		return ""
	}
}

func hasVCacheScoreRealizedPlane(rep vcachescore.Report) bool {
	return rep.Planes.ProviderObserved.Available ||
		rep.Planes.KernelWitnessed.Available ||
		rep.Planes.ContextWitnessed.Available ||
		rep.Planes.ExternalEngineObserved.Available
}

func vcacheScorePlaneStatus(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "missing"
	}
	switch strings.ToUpper(strings.TrimSpace(p.Provenance)) {
	case "OBSERVED":
		return "observed"
	case "WITNESSED":
		return "measured"
	case "FORECAST":
		return "forecast"
	default:
		return "measured"
	}
}

func vcacheScorePlaneEvidence(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Provenance) != "" {
		return p.Provenance
	}
	if !p.Available {
		return "MISSING"
	}
	return "unknown"
}

func vcacheScorePlaneLabel(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "MISSING"
	}
	return nonEmpty(p.Provenance, "AVAILABLE")
}

func vcacheScorePlaneReason(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Reason) == "" {
		return fmt.Sprintf("available=%v provenance=%s multiplier=%.2fx saved=%.1f baseline=%.1f hit=%.2f%% cost=%.1f",
			p.Available,
			vcacheScorePlaneEvidence(p),
			p.Multiplier,
			p.SavedTokenEquiv,
			p.BaselineTokenEquiv,
			100*p.HitRate,
			p.CostTokenEquiv)
	}
	return fmt.Sprintf("%s (multiplier=%.2fx saved=%.1f baseline=%.1f hit=%.2f%% cost=%.1f)",
		p.Reason,
		p.Multiplier,
		p.SavedTokenEquiv,
		p.BaselineTokenEquiv,
		100*p.HitRate,
		p.CostTokenEquiv)
}

func nonEmptyActions(actions []string) []string {
	out := make([]string, 0, len(actions))
	for _, action := range actions {
		if strings.TrimSpace(action) != "" {
			out = append(out, strings.TrimSpace(action))
		}
	}
	if len(out) == 0 {
		return []string{"rerun fak vcache score with realized plane evidence"}
	}
	if len(out) > 3 {
		return out[:3]
	}
	return out
}

func loadCachevalueVCacheActionsStatus(path string) (cachevalueVCacheActionsDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheActionsDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheActionsUnavailableRow(path, err.Error())}
	}
	var plan vcacheobserve.ProviderActionPlan
	if err := json.Unmarshal(b, &plan); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheActionsUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheActionsStatus(plan)
	digest.Turns = plan.Turns
	digest.FamilyCount = plan.FamilyCount
	digest.Noop = plan.Counts.Noop
	digest.Ready = plan.Counts.Ready
	digest.Gated = plan.Counts.Gated
	digest.TransportMode = plan.Transport.Mode
	digest.TransportReady = plan.Transport.Ready
	digest.TransportReason = plan.Transport.Reason
	return digest, rowsFromVCacheActionsPlan(plan, path)
}

func vcacheActionsUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache_control",
		Component:     "vcache_actions_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_actions_artifact",
		SessionImpact: "provider-cache action-plan evidence could not be folded, so transport gating attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache actions --json --out FILE",
	}
}

func rowsFromVCacheActionsPlan(plan vcacheobserve.ProviderActionPlan, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{
		{
			Plane:         "provider_prompt_cache_control",
			Component:     "vcache_actions_report",
			Owner:         "fak",
			Dependency:    "provider_action_transport",
			Fidelity:      "lossless",
			Evidence:      "DECISION",
			Status:        vcacheActionsStatus(plan),
			FailureDomain: vcacheActionsFailureDomain(plan),
			SessionImpact: "provider-cache actions are fak-authored decisions over provider telemetry; ready/gated rows are not proof that a provider warm already executed",
			Reason:        vcacheActionsReportReason(plan, path),
			NextAction:    vcacheActionsNextAction(plan),
		},
		{
			Plane:         "provider_prompt_cache_control",
			Component:     "vcache_actions_transport",
			Owner:         "provider",
			Dependency:    "provider_action_transport",
			Fidelity:      "lossless",
			Evidence:      vcacheActionsTransportEvidence(plan),
			Status:        vcacheActionsTransportStatus(plan),
			FailureDomain: "provider_transport",
			SessionImpact: "spendful provider-cache actions require provider transport plus byte-identical prefix evidence; gated transport is provider/API boundary, not fak-native KV",
			Reason:        plan.Transport.Reason,
			NextAction:    vcacheActionsTransportNextAction(plan),
		},
	}
	for _, action := range plan.Actions {
		rows = append(rows, rowFromVCacheAction(action))
	}
	return rows
}

func rowFromVCacheAction(action vcacheobserve.ProviderAction) cachevalueStatusRow {
	owner, dependency, failureDomain := vcacheActionAttribution(action)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache_control",
		Component:     "vcache_action:" + nonEmpty(action.Family, "unknown") + ":" + nonEmpty(action.Action, "unknown"),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      "lossless",
		Evidence:      "DECISION",
		Status:        string(action.State),
		FailureDomain: failureDomain,
		SessionImpact: "one provider-cache family action candidate; state names whether this is a no-op, locally/provider-ready, or still gated by missing transport evidence",
		Reason:        vcacheActionReason(action),
		NextAction:    vcacheActionNextAction(action),
	}
}

func vcacheActionsStatus(plan vcacheobserve.ProviderActionPlan) string {
	switch {
	case strings.TrimSpace(plan.Schema) == "" && len(plan.Actions) == 0:
		return "missing"
	case plan.Counts.Gated > 0:
		return "gated"
	case plan.Counts.Ready > 0:
		return "ready"
	default:
		return "no-op"
	}
}

func vcacheActionsFailureDomain(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Counts.Gated > 0 {
		return "provider_transport"
	}
	return "provider_prompt_cache_control"
}

func vcacheActionsReportReason(plan vcacheobserve.ProviderActionPlan, path string) string {
	return fmt.Sprintf("turns=%d families=%d noop=%d ready=%d gated=%d transport=%s ready=%v source=%s law=%s",
		plan.Turns,
		plan.FamilyCount,
		plan.Counts.Noop,
		plan.Counts.Ready,
		plan.Counts.Gated,
		nonEmpty(plan.Transport.Mode, "-"),
		plan.Transport.Ready,
		path,
		nonEmpty(plan.CorrectnessLaw, "-"))
}

func vcacheActionsNextAction(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Counts.Gated == 0 {
		return ""
	}
	return "rerun fak vcache actions with required provider transport witnesses, e.g. --heartbeat-transport/--explicit-cache-transport --prefix-witness"
}

func vcacheActionsTransportStatus(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Transport.Ready {
		return "ready"
	}
	if plan.Counts.Gated > 0 {
		return "gated"
	}
	return "no-op"
}

func vcacheActionsTransportEvidence(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Transport.Ready || plan.Transport.Witness != nil {
		return "WITNESSED"
	}
	return "MISSING"
}

func vcacheActionsTransportNextAction(plan vcacheobserve.ProviderActionPlan) string {
	if vcacheActionsTransportStatus(plan) != "gated" {
		return ""
	}
	return vcacheActionsNextAction(plan)
}

func vcacheActionAttribution(action vcacheobserve.ProviderAction) (owner, dependency, failureDomain string) {
	switch action.Action {
	case "evict_manifest", "no_cache":
		return "fak", "local_provider_manifest", "fak_provider_manifest"
	case "heartbeat_pin", "explicit_cache":
		return "provider", "provider_action_transport", "provider_transport"
	case "ride_natural", "lazy_rebuild":
		return "provider", "natural_provider_cache_window", "provider_prompt_cache_control"
	default:
		return "unknown", "provider_action_transport", "provider_transport"
	}
}

func vcacheActionReason(action vcacheobserve.ProviderAction) string {
	missing := missingVCacheActionRequires(action.Requires, action.Witnessed)
	parts := []string{
		fmt.Sprintf("decision=%s action=%s state=%s turns=%d saved=%.1f", action.Decision, action.Action, action.State, action.Turns, action.SavedTokenEquiv),
	}
	if len(action.Requires) > 0 {
		parts = append(parts, "requires="+strings.Join(action.Requires, ","))
	}
	if len(action.Witnessed) > 0 {
		parts = append(parts, "witnessed="+strings.Join(action.Witnessed, ","))
	}
	if len(missing) > 0 {
		parts = append(parts, "missing="+strings.Join(missing, ","))
	}
	if strings.TrimSpace(action.Reason) != "" {
		parts = append(parts, action.Reason)
	}
	return strings.Join(parts, " ")
}

func vcacheActionNextAction(action vcacheobserve.ProviderAction) string {
	if action.State != vcacheobserve.ActionGated {
		return ""
	}
	missing := missingVCacheActionRequires(action.Requires, action.Witnessed)
	if len(missing) == 0 {
		return "inspect provider action transport witness"
	}
	return "supply provider action witness: " + strings.Join(missing, ",")
}

func missingVCacheActionRequires(requires, witnessed []string) []string {
	if len(requires) == 0 {
		return nil
	}
	seen := map[string]bool{}
	for _, w := range witnessed {
		seen[w] = true
	}
	var missing []string
	for _, req := range requires {
		if !seen[req] {
			missing = append(missing, req)
		}
	}
	return missing
}

func loadCachevalueVCacheContextJoinStatus(path string) (cachevalueVCacheContextJoinDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheContextJoinDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextJoinUnavailableRow(path, err.Error())}
	}
	var rep vcacheobserve.JoinReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextJoinUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheContextJoinStatus(rep)
	digest.FailureDomain = vcacheContextJoinFailureDomain(rep)
	digest.Turns = rep.Turns
	digest.Events = rep.Events
	digest.TotalChanges = vcacheContextJoinTotalChanges(rep)
	digest.PlanningAttributed = rep.Summary.PlanningAttributed
	digest.ProviderAttributed = rep.Summary.ProviderAttributed
	return digest, rowsFromVCacheContextJoinReport(rep, path)
}

func vcacheContextJoinUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_join_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_context_join_artifact",
		SessionImpact: "context-join evidence could not be folded, so fak context-planning and provider cache behavior remain ambiguous",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache context-join --telemetry FILE --events FILE --json",
	}
}

func rowsFromVCacheContextJoinReport(rep vcacheobserve.JoinReport, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "managed_context",
		Component:     "vcache_context_join_report",
		Owner:         "fak",
		Dependency:    "context_lifecycle_events",
		Fidelity:      "diagnostic",
		Evidence:      vcacheContextJoinEvidence(rep),
		Status:        vcacheContextJoinStatus(rep),
		FailureDomain: vcacheContextJoinFailureDomain(rep),
		SessionImpact: "context-join separates fak managed-context events from natural provider-cache behavior for bad-session attribution",
		Reason:        vcacheContextJoinReportReason(rep, path),
		NextAction:    vcacheContextJoinNextAction(rep),
	}}
	for i, change := range rep.Changes {
		rows = append(rows, rowFromVCacheContextJoinChange(change, i))
	}
	return rows
}

func rowFromVCacheContextJoinChange(change vcacheobserve.AttributedChange, idx int) cachevalueStatusRow {
	owner, dependency, fidelity, evidence, status, failureDomain := vcacheContextJoinChangeAttribution(change)
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     fmt.Sprintf("context_join:%s:%s:%d", nonEmpty(change.Family, "unknown"), nonEmpty(string(change.Change), "change"), idx+1),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        status,
		FailureDomain: failureDomain,
		SessionImpact: vcacheContextJoinChangeImpact(change),
		Reason:        vcacheContextJoinChangeReason(change),
		NextAction:    vcacheContextJoinChangeNextAction(change),
	}
}

func vcacheContextJoinChangeAttribution(change vcacheobserve.AttributedChange) (owner, dependency, fidelity, evidence, status, failureDomain string) {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		return "fak", "context_lifecycle_events", vcacheContextPlanningFidelity(change.MatchedEvent), "WITNESSED", string(vcacheobserve.CausePlanning), "fak_context_planner"
	case vcacheobserve.CauseProviderBehavior:
		return "provider", "provider_cache_telemetry", "passive", "OBSERVED", string(vcacheobserve.CauseProviderBehavior), "provider_cache_behavior"
	default:
		return "unknown", "provider_cache_telemetry", "diagnostic", "OBSERVED", "unattributed", "evidence_gap"
	}
}

func vcacheContextPlanningFidelity(ev *vcacheobserve.LifecycleEvent) string {
	if ev == nil {
		return "diagnostic"
	}
	switch ev.Kind {
	case vcacheobserve.EventCompaction, vcacheobserve.EventPrefixMutation:
		return "lossy"
	case vcacheobserve.EventContextReset:
		return "recoverable"
	case vcacheobserve.EventPageFault:
		return "passive"
	default:
		return "diagnostic"
	}
}

func vcacheContextJoinStatus(rep vcacheobserve.JoinReport) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.Events == 0 && len(rep.Changes) == 0:
		return "missing"
	case vcacheContextJoinTotalChanges(rep) == 0:
		return "no-op"
	default:
		return "measured"
	}
}

func vcacheContextJoinEvidence(rep vcacheobserve.JoinReport) string {
	if strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.Events == 0 && len(rep.Changes) == 0 {
		return "MISSING"
	}
	if rep.Summary.PlanningAttributed > 0 || rep.Events > 0 {
		return "WITNESSED"
	}
	if rep.Summary.ProviderAttributed > 0 || len(rep.Changes) > 0 {
		return "OBSERVED"
	}
	return "WITNESSED"
}

func vcacheContextJoinFailureDomain(rep vcacheobserve.JoinReport) string {
	planning := rep.Summary.PlanningAttributed
	provider := rep.Summary.ProviderAttributed
	switch {
	case planning > 0 && provider > 0:
		return "mixed_context_provider"
	case planning > 0:
		return "fak_context_planner"
	case provider > 0:
		return "provider_cache_behavior"
	case vcacheContextJoinTotalChanges(rep) == 0:
		return "none"
	default:
		return "evidence_gap"
	}
}

func vcacheContextJoinTotalChanges(rep vcacheobserve.JoinReport) int {
	if rep.Summary.TotalChanges > 0 {
		return rep.Summary.TotalChanges
	}
	return len(rep.Changes)
}

func vcacheContextJoinReportReason(rep vcacheobserve.JoinReport, path string) string {
	return fmt.Sprintf("turns=%d events=%d changes=%d planning=%d provider=%d source=%s",
		rep.Turns,
		rep.Events,
		vcacheContextJoinTotalChanges(rep),
		rep.Summary.PlanningAttributed,
		rep.Summary.ProviderAttributed,
		path)
}

func vcacheContextJoinNextAction(rep vcacheobserve.JoinReport) string {
	planning := rep.Summary.PlanningAttributed
	provider := rep.Summary.ProviderAttributed
	switch {
	case vcacheContextJoinStatus(rep) == "missing":
		return "fak vcache context-join --telemetry FILE --events FILE --json"
	case planning > 0 && provider > 0:
		return "inspect matched fak lifecycle events and provider TTL/prefix behavior"
	case planning > 0:
		return "inspect matched fak lifecycle events with fak vcache context-witness --json"
	case provider > 0:
		return "inspect provider TTL/prefix churn or run fak vcache actions --json"
	default:
		return ""
	}
}

func vcacheContextJoinChangeImpact(change vcacheobserve.AttributedChange) string {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		return "a fak managed-context lifecycle event explains the provider cost change; inspect reset/compaction/page-fault handling before blaming natural provider cache behavior"
	case vcacheobserve.CauseProviderBehavior:
		return "the cost change had no nearby fak lifecycle event; suspect provider TTL, prefix churn, or provider cache-window behavior"
	default:
		return "the cost change is not attributable from the supplied lifecycle and provider telemetry"
	}
}

func vcacheContextJoinChangeReason(change vcacheobserve.AttributedChange) string {
	parts := []string{
		fmt.Sprintf("family=%s change=%s cause=%s", nonEmpty(change.Family, "unknown"), nonEmpty(string(change.Change), "change"), nonEmpty(string(change.Cause), "unknown")),
	}
	if strings.TrimSpace(change.Detail) != "" {
		parts = append(parts, change.Detail)
	}
	if change.MatchedEvent != nil {
		parts = append(parts, fmt.Sprintf("matched=%s outcome=%s detail=%s",
			change.MatchedEvent.Kind,
			nonEmpty(change.MatchedEvent.Outcome, "-"),
			nonEmpty(change.MatchedEvent.Detail, "-")))
	}
	return strings.Join(parts, " ")
}

func vcacheContextJoinChangeNextAction(change vcacheobserve.AttributedChange) string {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		if change.MatchedEvent != nil {
			return "inspect fak lifecycle event " + string(change.MatchedEvent.Kind)
		}
		return "rerun fak vcache context-join with lifecycle events"
	case vcacheobserve.CauseProviderBehavior:
		return "inspect provider TTL/prefix churn or run fak vcache actions --json"
	default:
		return "rerun fak vcache context-join with a complete lifecycle-event stream"
	}
}

func loadCachevalueVCacheObserveStatus(path string) (cachevalueVCacheObserveDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheObserveDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(path, err.Error())}
	}
	var rep vcacheobserve.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(path, "decode: "+err.Error())}
	}
	digest = cachevalueVCacheObserveDigestFromReport(path, rep)
	return digest, rowsFromVCacheObserveReport(rep, path)
}

func loadCachevalueVCacheObserveFromSession(path string) (cachevalueVCacheObserveDigest, []cachevalueStatusRow) {
	source := "session:" + path
	digest := cachevalueVCacheObserveDigest{Path: source, Status: "unavailable"}
	turns, err := readObserveTranscript(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(source, err.Error())}
	}
	rep := vcacheobserve.ObserveWithOptions(turns, vcacheobserve.DefaultOptions())
	digest = cachevalueVCacheObserveDigestFromReport(source, rep)
	return digest, rowsFromVCacheObserveReport(rep, source)
}

func cachevalueVCacheObserveDigestFromReport(path string, rep vcacheobserve.Report) cachevalueVCacheObserveDigest {
	digest := cachevalueVCacheObserveDigest{Path: path}
	digest.Status = vcacheObserveStatus(rep)
	digest.FailureDomain = vcacheObserveFailureDomain(rep)
	digest.Turns = rep.Turns
	digest.FamilyCount = rep.FamilyCount
	digest.TurnsReordered = rep.TurnsReordered
	digest.OutOfOrderTurns = rep.OutOfOrderTurns
	digest.HitRate = rep.HitRate
	digest.Multiplier = rep.Multiplier
	digest.SavedTokenEquiv = rep.Aggregate.SavedTokenEquiv
	digest.CacheReadTokens = rep.Aggregate.CacheReadTokens
	digest.CacheCreationTokens = rep.Aggregate.CacheCreationTokens
	digest.FalseWarm = rep.Prediction.FalseWarm
	digest.FalseWarmRate = rep.Prediction.FalseWarmRate()
	return digest
}

func vcacheObserveUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_report",
		Owner:         "provider",
		Dependency:    "local_json_report",
		Fidelity:      "lossless",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_observe_artifact",
		SessionImpact: "provider-cache observation evidence could not be folded, so provider hit/miss behavior remains ambiguous",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache observe --transcript FILE --json",
	}
}

func rowsFromVCacheObserveReport(rep vcacheobserve.Report, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_report",
		Owner:         "provider",
		Dependency:    "provider_usage_fields",
		Fidelity:      "lossless",
		Evidence:      vcacheObserveEvidence(rep),
		Status:        vcacheObserveStatus(rep),
		FailureDomain: vcacheObserveFailureDomain(rep),
		SessionImpact: "direct provider-cache telemetry separates realized provider rebates and misses from fak-owned KV/context behavior",
		Reason:        vcacheObserveReportReason(rep, path),
		NextAction:    vcacheObserveNextAction(rep),
	}}
	for _, slice := range rep.OwnerSlices {
		rows = append(rows, rowFromVCacheObserveOwnerSlice(slice))
	}
	for _, family := range rep.Families {
		rows = append(rows, rowFromVCacheObserveFamily(family))
	}
	return rows
}

func rowFromVCacheObserveOwnerSlice(slice vcacheobserve.OwnerSlice) cachevalueStatusRow {
	owner := nonEmpty(slice.Owner, "unknown")
	mechanism := nonEmpty(slice.Mechanism, "unknown")
	status := vcacheObserveOwnerSliceStatus(slice)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_owner:" + componentKey(owner) + ":" + componentKey(mechanism),
		Owner:         owner,
		Dependency:    vcacheObserveOwnerDependency(owner),
		Fidelity:      vcacheObserveOwnerFidelity(slice),
		Evidence:      string(slice.Provenance),
		Status:        status,
		FailureDomain: vcacheObserveOwnerFailureDomain(slice),
		SessionImpact: vcacheObserveOwnerImpact(slice),
		Reason:        vcacheObserveOwnerReason(slice),
		NextAction:    vcacheObserveOwnerNextAction(slice),
	}
}

func rowFromVCacheObserveFamily(family vcacheobserve.Family) cachevalueStatusRow {
	status := vcacheObserveFamilyStatus(family)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_family:" + componentKey(nonEmpty(family.Key, "unknown")),
		Owner:         "provider",
		Dependency:    "provider_usage_fields",
		Fidelity:      "lossless",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: vcacheObserveFamilyFailureDomain(family),
		SessionImpact: vcacheObserveFamilyImpact(family),
		Reason:        vcacheObserveFamilyReason(family),
		NextAction:    vcacheObserveFamilyNextAction(family),
	}
}

func vcacheObserveStatus(rep vcacheobserve.Report) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.FamilyCount == 0:
		return "missing"
	case rep.Prediction.FalseWarm > 0:
		return "false_warm"
	case rep.TurnsReordered:
		return "turns_reordered"
	case rep.Aggregate.CacheReadTokens > 0:
		return "observed"
	case rep.Aggregate.CacheCreationTokens > 0:
		return "cold_write_only"
	default:
		return "no_usage"
	}
}

func vcacheObserveEvidence(rep vcacheobserve.Report) string {
	if strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.FamilyCount == 0 {
		return "MISSING"
	}
	return "OBSERVED"
}

func vcacheObserveFailureDomain(rep vcacheobserve.Report) string {
	switch vcacheObserveStatus(rep) {
	case "false_warm":
		return "provider_cache_prediction"
	case "turns_reordered":
		return "provider_telemetry_ordering"
	case "cold_write_only":
		return "provider_cache_window"
	case "missing", "no_usage":
		return "provider_telemetry"
	default:
		return "provider_prompt_cache"
	}
}

func vcacheObserveReportReason(rep vcacheobserve.Report, path string) string {
	return fmt.Sprintf("turns=%d families=%d hit=%.2f%% multiplier=%.2fx saved=%.1f cache_read=%.0f cache_create=%.0f false_warm=%d false_warm_rate=%.2f%% reordered=%v/%d source=%s",
		rep.Turns,
		rep.FamilyCount,
		100*rep.HitRate,
		rep.Multiplier,
		rep.Aggregate.SavedTokenEquiv,
		rep.Aggregate.CacheReadTokens,
		rep.Aggregate.CacheCreationTokens,
		rep.Prediction.FalseWarm,
		100*rep.Prediction.FalseWarmRate(),
		rep.TurnsReordered,
		rep.OutOfOrderTurns,
		path)
}

func vcacheObserveNextAction(rep vcacheobserve.Report) string {
	switch vcacheObserveStatus(rep) {
	case "missing", "no_usage":
		return "fak vcache observe --transcript FILE --json"
	case "false_warm":
		return "run fak vcache context-join --events FILE and inspect provider TTL/prefix calibration"
	case "turns_reordered":
		return "sort or timestamp provider telemetry before comparing TTL-sensitive cache behavior"
	case "cold_write_only":
		return "run fak vcache context-join --events FILE or fak vcache actions --json"
	default:
		return ""
	}
}

func vcacheObserveOwnerSliceStatus(slice vcacheobserve.OwnerSlice) string {
	switch slice.Provenance {
	case vcacheobserve.Observed:
		if slice.SavedTokenEquiv > 0 || slice.CacheReadTokens > 0 {
			return "observed"
		}
		if slice.CacheCreationTokens > 0 {
			return "cold_write_only"
		}
		return "no_effect"
	case vcacheobserve.NotObserved:
		return "not_observed"
	case vcacheobserve.Decision:
		return "ready"
	default:
		return "unknown"
	}
}

func vcacheObserveOwnerDependency(owner string) string {
	switch strings.ToLower(strings.TrimSpace(owner)) {
	case "provider":
		return "provider_usage_fields"
	case "fak":
		return "fak_cache_witness"
	default:
		return "observe_report"
	}
}

func vcacheObserveOwnerFidelity(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "diagnostic"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "lossless"
	}
	return "diagnostic"
}

func vcacheObserveOwnerFailureDomain(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "evidence_gap"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "provider_prompt_cache"
	}
	return cachevalueFailureDomain(slice.Owner, slice.Mechanism)
}

func vcacheObserveOwnerImpact(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "this observe source cannot prove fak-authored KV/context effects; use fak witnesses before attributing a bad session to fak"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "provider-reported prompt-cache economics; these are external rebates/misses, not fak-native KV hits"
	}
	return "owner slice from the vcache observe report"
}

func vcacheObserveOwnerReason(slice vcacheobserve.OwnerSlice) string {
	return fmt.Sprintf("mechanism=%s provenance=%s saved=%.1f cache_read=%.0f cache_create=%.0f evidence=%s",
		nonEmpty(slice.Mechanism, "unknown"),
		slice.Provenance,
		slice.SavedTokenEquiv,
		slice.CacheReadTokens,
		slice.CacheCreationTokens,
		nonEmpty(slice.Evidence, "-"))
}

func vcacheObserveOwnerNextAction(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance != vcacheobserve.NotObserved {
		return ""
	}
	return "add fak-owned witnesses with fak vcache score --kernel-ledger default --json or fak vcache context-witness --json"
}

func vcacheObserveFamilyStatus(family vcacheobserve.Family) string {
	switch {
	case family.Prediction.FalseWarm > 0:
		return "false_warm"
	case family.TurnsReordered:
		return "turns_reordered"
	case family.CacheReadTokens > 0:
		return "observed"
	case family.CacheCreationTokens > 0:
		return "cold_write_only"
	default:
		return "no_usage"
	}
}

func vcacheObserveFamilyFailureDomain(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "provider_cache_prediction"
	case "turns_reordered":
		return "provider_telemetry_ordering"
	case "cold_write_only":
		return "provider_cache_window"
	case "no_usage":
		return "provider_telemetry"
	default:
		return "provider_prompt_cache"
	}
}

func vcacheObserveFamilyImpact(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "fak's warmth belief expected a provider cache hit but provider telemetry missed; inspect TTL, prefix churn, and context events"
	case "turns_reordered":
		return "same-family telemetry arrived out of order; TTL-sensitive attribution was repaired by sorting but source ordering should be fixed"
	case "cold_write_only":
		return "provider cache writes happened without reads for this family; suspect cold start, TTL expiry, or prefix churn"
	default:
		return "per-family provider prompt-cache observation"
	}
}

func vcacheObserveFamilyReason(family vcacheobserve.Family) string {
	return fmt.Sprintf("turns=%d hit=%.2f%% cache_read=%d cache_create=%d saved=%.1f governor=%s false_warm=%d false_warm_rate=%.2f%% reordered=%v/%d",
		family.Turns,
		100*family.HitRate,
		family.CacheReadTokens,
		family.CacheCreationTokens,
		family.Economics.SavedTokenEquiv,
		family.GovernorDecision,
		family.Prediction.FalseWarm,
		100*family.Prediction.FalseWarmRate(),
		family.TurnsReordered,
		family.OutOfOrderTurns)
}

func vcacheObserveFamilyNextAction(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "run fak vcache context-join --events FILE for family " + nonEmpty(family.Key, "unknown")
	case "turns_reordered":
		return "sort or timestamp telemetry for family " + nonEmpty(family.Key, "unknown")
	case "cold_write_only":
		return "inspect TTL/prefix churn for family " + nonEmpty(family.Key, "unknown")
	default:
		return ""
	}
}

func componentKey(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}
	replacer := strings.NewReplacer(" ", "_", "\t", "_", "\n", "_", ":", "_", "/", "_", "\\", "_")
	return replacer.Replace(s)
}

func loadCachevalueVCacheContextWitnessStatus(path string) (cachevalueVCacheContextWitnessDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheContextWitnessDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextWitnessUnavailableRow(path, err.Error())}
	}
	var rep vcacheContextWitnessReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextWitnessUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheContextWitnessStatus(rep)
	digest.FailureDomain = vcacheContextWitnessFailureDomain(rep)
	digest.Fixture = rep.Fixture
	digest.Wire = rep.Wire
	digest.Snapshot = rep.Snapshot
	digest.ReplayExit = rep.ReplayExit
	digest.ScoreExit = rep.ScoreExit
	digest.ScoreStatus = rep.ScoreStatus
	digest.ContextWitnessed = vcacheContextWitnessPlaneLabel(rep.ContextWitnessed)
	digest.ContextEvents = rep.ContextEvents
	digest.ContextShedTokens = rep.ContextShedTokens
	return digest, rowsFromVCacheContextWitnessReport(rep, path)
}

func vcacheContextWitnessUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "lossy",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_context_witness_artifact",
		SessionImpact: "fak context-witness evidence could not be folded, so lossy context behavior remains unevidenced",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache context-witness --json",
	}
}

func rowsFromVCacheContextWitnessReport(rep vcacheContextWitnessReport, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_report",
		Owner:         "fak",
		Dependency:    "guard_replay_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      vcacheContextWitnessEvidence(rep),
		Status:        vcacheContextWitnessStatus(rep),
		FailureDomain: vcacheContextWitnessFailureDomain(rep),
		SessionImpact: "fak-owned context replay proves lossy context shedding separately from provider prompt-cache hit/miss behavior",
		Reason:        vcacheContextWitnessReason(rep, path),
		NextAction:    vcacheContextWitnessNextAction(rep),
	}}
	rows = append(rows, cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_plane",
		Owner:         "fak",
		Dependency:    "vcache_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      vcacheContextWitnessPlaneEvidence(rep.ContextWitnessed),
		Status:        vcacheContextWitnessPlaneStatus(rep.ContextWitnessed),
		FailureDomain: "fak_context_planner",
		SessionImpact: "context-witness plane carries the shed-token economics used by fak vcache score",
		Reason:        vcacheScorePlaneReason(rep.ContextWitnessed),
		NextAction:    vcacheContextWitnessPlaneNextAction(rep.ContextWitnessed),
	})
	return rows
}

func vcacheContextWitnessStatus(rep vcacheContextWitnessReport) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && strings.TrimSpace(rep.Snapshot) == "":
		return "missing"
	case rep.ReplayExit != 0:
		return "replay_failed"
	case rep.ScoreExit != 0:
		return "score_failed"
	case !rep.ContextWitnessed.Available || rep.ContextEvents == 0:
		return "missing"
	default:
		return "measured"
	}
}

func vcacheContextWitnessEvidence(rep vcacheContextWitnessReport) string {
	if vcacheContextWitnessStatus(rep) == "measured" {
		return "WITNESSED"
	}
	if rep.ContextWitnessed.Available {
		return vcacheContextWitnessPlaneEvidence(rep.ContextWitnessed)
	}
	return "MISSING"
}

func vcacheContextWitnessFailureDomain(rep vcacheContextWitnessReport) string {
	switch vcacheContextWitnessStatus(rep) {
	case "replay_failed":
		return "fak_context_replay"
	case "score_failed":
		return "fak_context_score"
	case "missing":
		return "fak_context_evidence_gap"
	default:
		return "fak_context_planner"
	}
}

func vcacheContextWitnessReason(rep vcacheContextWitnessReport, path string) string {
	return fmt.Sprintf("fixture=%s wire=%s snapshot=%s replay_exit=%d score_exit=%d score_status=%s context=%s events=%d shed=%.1f source=%s",
		nonEmpty(rep.Fixture, "-"),
		nonEmpty(rep.Wire, "-"),
		nonEmpty(rep.Snapshot, "-"),
		rep.ReplayExit,
		rep.ScoreExit,
		nonEmpty(rep.ScoreStatus, "-"),
		vcacheContextWitnessPlaneLabel(rep.ContextWitnessed),
		rep.ContextEvents,
		rep.ContextShedTokens,
		path)
}

func vcacheContextWitnessNextAction(rep vcacheContextWitnessReport) string {
	switch vcacheContextWitnessStatus(rep) {
	case "replay_failed":
		return "rerun fak vcache context-witness and inspect guard replay failure"
	case "score_failed":
		return "rerun fak vcache score --context-snapshot " + nonEmpty(rep.Snapshot, "FILE") + " --json"
	case "missing":
		return "rerun fak vcache context-witness --json and ensure the replay emits fak_context_* counters"
	default:
		return ""
	}
}

func vcacheContextWitnessPlaneStatus(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "missing"
	}
	return "measured"
}

func vcacheContextWitnessPlaneEvidence(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Provenance) != "" {
		return p.Provenance
	}
	if p.Available {
		return "WITNESSED"
	}
	return "MISSING"
}

func vcacheContextWitnessPlaneLabel(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "MISSING"
	}
	return nonEmpty(p.Provenance, "WITNESSED")
}

func vcacheContextWitnessPlaneNextAction(p vcachescore.PlaneValueReport) string {
	if p.Available {
		return ""
	}
	return "fak vcache context-witness --json"
}

func cacheAblationStatusRow() cachevalueStatusRow {
	features := cacheAblationFeatures()
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "cache_ablation_runner",
		Owner:         "fak",
		Dependency:    "subprocess_reexec",
		Fidelity:      "diagnostic",
		Evidence:      "configured",
		Status:        "available",
		FailureDomain: "fak_diagnostics",
		SessionImpact: "use this when a bad session needs a controlled fak-native versus provider-cache comparison",
		Reason:        "can sweep cache/cache-adjacent runtime features: " + strings.Join(features, ","),
		NextAction:    "fak ablate --sweep " + strings.Join(features, ",") + " --json",
	}
}

func cacheAblationFeatures() []string {
	cacheFeatures := map[string]bool{
		"bp_plan":       true,
		"compressor":    true,
		"ctxplan_seam":  true,
		"prefix_guard":  true,
		"radix":         true,
		"ttl_1h":        true,
		"uncached_trim": true,
		"vdso":          true,
	}
	var out []string
	for _, feature := range ablate.KnownFeatures {
		if cacheFeatures[feature] {
			out = append(out, feature)
		}
	}
	if len(out) == 0 {
		out = []string{"vdso"}
	}
	return out
}

func renderCachevalueStatus(w io.Writer, rep cachevalueStatusReport) {
	fmt.Fprintf(w, "cachevalue status: %s - %s\n", rep.Verdict, rep.Summary)
	fmt.Fprintf(w, "sources: kernel=%s savings=%s usage=%s\n", rep.Sources.KernelLedger, rep.Sources.SavingsLedger, rep.Sources.UsageLedger)
	if strings.TrimSpace(rep.Sources.ArtifactDir) != "" {
		fmt.Fprintf(w, "artifacts: dir=%s\n", rep.Sources.ArtifactDir)
	}
	fmt.Fprintf(w, "headroom: selected=%s reachable=%v url=%s\n", rep.Headroom.Selected, rep.Headroom.HeadroomReachable, rep.Sources.HeadroomURL)
	fmt.Fprintf(w, "vcache: provider_actions=%s transport=%s recent_provider=%s recent_context=%s\n",
		rep.VCache.ProviderActions, rep.VCache.ProviderActionTransport, cachevalueEmptyDash(rep.VCache.RecentProviderStatus), cachevalueEmptyDash(rep.VCache.RecentContextStatus))
	if rep.Session != nil {
		s := rep.Session
		fmt.Fprintf(w, "session: %s status=%s likely=%s turns=%d cache_read=%d cache_create=%d total_context=%d io=%s finding=%s\n",
			s.Session,
			s.Status,
			s.LikelyDomain,
			s.AssistantTurns,
			s.CacheReadTokens,
			s.CacheCreateTokens,
			s.TotalContextTokens,
			fmtFloatPtr(s.IORatio),
			s.Finding,
		)
	}
	if rep.Ablation != nil {
		a := rep.Ablation
		fmt.Fprintf(w, "ablation: status=%s runs=%d dropped=%d diag=%d child_exit=%d workload_mismatch=%d cache_effects=%d active=%d unavailable=%d path=%s\n",
			a.Status, a.Runs, a.DroppedArms, a.DroppedWithDiagnostics, a.DroppedChildExits, a.DroppedWorkloadMismatches,
			a.CacheEffects, a.ActiveEffects, a.UnavailableEffects, a.Path)
	}
	if rep.HeadroomBench != nil {
		h := rep.HeadroomBench
		fmt.Fprintf(w, "headroom bench: status=%s compressor=%s samples=%d saved=%.2f%% path=%s\n",
			h.Status, h.Compressor, h.Samples, h.SavedRatio*100, h.Path)
	}
	if rep.VCacheScore != nil {
		v := rep.VCacheScore
		fmt.Fprintf(w, "vcache score: status=%s grade=%s score=%d active=%s multiplier=%.2fx two_x=%v default=%s activation=%v path=%s\n",
			v.Status, cachevalueEmptyDash(v.Grade), v.Score, cachevalueEmptyDash(v.ActiveSource),
			v.ActiveMultiplier, v.TwoXBetter, cachevalueEmptyDash(v.DefaultUsefulness), v.AgenticActivation, v.Path)
	}
	if rep.VCacheActions != nil {
		a := rep.VCacheActions
		fmt.Fprintf(w, "vcache actions: status=%s families=%d noop=%d ready=%d gated=%d transport=%s ready=%v path=%s\n",
			a.Status, a.FamilyCount, a.Noop, a.Ready, a.Gated, cachevalueEmptyDash(a.TransportMode), a.TransportReady, a.Path)
	}
	if rep.VCacheObserve != nil {
		o := rep.VCacheObserve
		fmt.Fprintf(w, "vcache observe: status=%s domain=%s turns=%d families=%d hit=%.2f%% multiplier=%.2fx saved=%.1f false_warm=%d rate=%.2f%% path=%s\n",
			o.Status, cachevalueEmptyDash(o.FailureDomain), o.Turns, o.FamilyCount, 100*o.HitRate, o.Multiplier,
			o.SavedTokenEquiv, o.FalseWarm, 100*o.FalseWarmRate, o.Path)
	}
	if rep.VCacheContextJoin != nil {
		c := rep.VCacheContextJoin
		fmt.Fprintf(w, "vcache context-join: status=%s domain=%s changes=%d planning=%d provider=%d turns=%d events=%d path=%s\n",
			c.Status, cachevalueEmptyDash(c.FailureDomain), c.TotalChanges, c.PlanningAttributed, c.ProviderAttributed, c.Turns, c.Events, c.Path)
	}
	if rep.VCacheContextWitness != nil {
		c := rep.VCacheContextWitness
		fmt.Fprintf(w, "vcache context-witness: status=%s domain=%s context=%s events=%d shed=%.1f replay=%d score=%d path=%s\n",
			c.Status, cachevalueEmptyDash(c.FailureDomain), cachevalueEmptyDash(c.ContextWitnessed),
			c.ContextEvents, c.ContextShedTokens, c.ReplayExit, c.ScoreExit, c.Path)
	}
	renderCachevalueAttribution(w, rep.Attribution)
	fmt.Fprintln(w, "\ncache-plane rollup:")
	fmt.Fprintf(w, "  %-30s %-42s %-9s %-30s %-11s %-10s %-24s %s\n",
		"plane", "component", "owner", "status", "fidelity", "evidence", "dependency", "impact / next")
	for _, row := range rep.Rows {
		component := row.Component
		if row.Selected {
			component = "*" + component
		}
		fmt.Fprintf(w, "  %-30s %-42s %-9s %-30s %-11s %-10s %-24s %s\n",
			truncStatus(row.Plane, 30),
			truncStatus(component, 42),
			truncStatus(row.Owner, 9),
			truncStatus(row.Status, 30),
			truncStatus(row.Fidelity, 11),
			truncStatus(row.Evidence, 10),
			truncStatus(row.Dependency, 24),
			rowImpactNext(row),
		)
	}
	if len(rep.NextActions) > 0 {
		fmt.Fprintln(w, "\nnext actions:")
		for _, action := range rep.NextActions {
			fmt.Fprintf(w, "- %s\n", action)
		}
	}
}

func renderCachevalueAttribution(w io.Writer, a cachevalueStatusAttribution) {
	fmt.Fprintf(w, "attribution: problem owners=%s\n", cachevalueFormatFindings(a.ProblemOwners))
	fmt.Fprintf(w, "             problem domains=%s\n", cachevalueFormatFindings(a.ProblemDomains))
	fmt.Fprintf(w, "             problem fidelity=%s\n", cachevalueFormatFindings(a.ProblemFidelity))
	fmt.Fprintf(w, "             fidelity=%s evidence=%s\n", cachevalueFormatIntMap(a.Fidelities), cachevalueFormatIntMap(a.Evidence))
}

func cachevalueFormatFindings(findings []cachevalueStatusFinding) string {
	if len(findings) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(findings))
	for _, f := range findings {
		part := fmt.Sprintf("%s=%d", f.Key, f.Problems)
		if strings.TrimSpace(f.Example) != "" {
			part += " [" + f.Example + "]"
		}
		parts = append(parts, part)
	}
	return strings.Join(parts, "; ")
}

func cachevalueFormatIntMap(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] != m[keys[j]] {
			return m[keys[i]] > m[keys[j]]
		}
		return keys[i] < keys[j]
	})
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, ",")
}

func cachevalueComponentDependency(component string) string {
	switch component {
	case "kernel_prefix_reuse":
		return "cachevalue_ledger"
	case "provider_prompt_cache":
		return "provider_usage_ledger"
	case "compaction_shed", "guard_serve_usage_ledger":
		return "gateway_usage_ledger"
	default:
		return "ledger"
	}
}

func cachevalueComponentImpact(component string) string {
	switch component {
	case "kernel_prefix_reuse":
		return "fak-native KV reuse; inspect fak guard/serve cache admission when insufficient"
	case "provider_prompt_cache":
		return "provider-reported cache counters; missing/dollar-blind evidence is provider telemetry or pricing"
	case "compaction_shed":
		return "fak-authored context compaction; lossy token shedding is separate from lossless cache hits"
	case "guard_serve_usage_ledger":
		return "fak gateway counters; missing rows make session-level attribution incomplete"
	default:
		return "cache plane status row"
	}
}

func cachevalueFailureDomain(owner, component string) string {
	switch strings.ToLower(strings.TrimSpace(owner)) {
	case "provider":
		return "provider"
	case "external":
		return "external:" + strings.ToLower(strings.TrimSpace(component))
	case "fak":
		return "fak"
	default:
		return "unknown"
	}
}

func headroomSessionImpact(p headroom.PluginStatus) string {
	if p.Name == headroom.HeadroomName && p.Status == "unavailable" {
		if p.Selected {
			return "selected external sidecar is down; compression passes original bytes through, so bad compression value is external headroom, not fak core"
		}
		return "external sidecar is down but not selected; current sessions do not depend on it"
	}
	if p.Name == headroom.NoopName && p.Selected {
		return "compression is intentionally off; large tool outputs are not a headroom failure"
	}
	if p.Name == headroom.NativeName && p.Selected {
		return "in-process fak compressor is active; compression behavior is fak-owned"
	}
	return "registered compressor capability"
}

func headroomNextAction(p headroom.PluginStatus) string {
	switch {
	case p.Name == headroom.HeadroomName && p.Status == "unavailable":
		return "start headroom proxy or select FAK_COMPRESSOR=native/noop"
	case p.Name == headroom.NoopName && p.Selected:
		return "set FAK_COMPRESSOR=native or FAK_COMPRESSOR=headroom to enable context compression"
	case p.Name == headroom.NativeName && (p.Status == "active" || p.Status == "available"):
		return "fak headroom bench --via native"
	case p.Name == headroom.HeadroomName && (p.Status == "active" || p.Status == "available"):
		return "fak headroom bench --via headroom"
	default:
		return ""
	}
}

func statusReady(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return "missing"
	}
	if v == "ready" || v == "proven" {
		return "ready"
	}
	return v
}

func providerActionStatus(v vcacheProviderActionStatus) string {
	if strings.EqualFold(strings.TrimSpace(v.Transport), "decision_only") {
		return "gated"
	}
	return statusReady(v.Verifier)
}

func vcacheRecentProviderStatus(rep vcacheStatusReport) string {
	if rep.RecentObservation == nil {
		return ""
	}
	return rep.RecentObservation.ProviderStatus
}

func vcacheRecentContextStatus(rep vcacheStatusReport) string {
	if rep.RecentObservation == nil {
		return ""
	}
	return rep.RecentObservation.ContextStatus
}

func recentProviderObservationStatus(status string) string {
	if strings.EqualFold(strings.TrimSpace(status), "MISSING") || strings.TrimSpace(status) == "" {
		return "missing"
	}
	return "observed"
}

func recentContextObservationStatus(status string) string {
	if strings.EqualFold(strings.TrimSpace(status), "WITNESSED") {
		return "measured"
	}
	return "missing"
}

func recentProviderObservationReason(recent vcacheRecentObservation) string {
	if strings.EqualFold(strings.TrimSpace(recent.ProviderStatus), "MISSING") {
		return fmt.Sprintf("%d turn(s), no provider-cache telemetry in snapshot", recent.Turns)
	}
	return fmt.Sprintf("%d turn(s), provider %s, multiplier %.2fx, saved %.1f token-equiv, false-warm %.2f%%",
		recent.Turns, recent.ProviderStatus, recent.Multiplier, recent.SavedTokenEquiv, 100*recent.FalseWarmRate)
}

func cachevalueStatusCounts(rows []cachevalueStatusRow) map[string]int {
	counts := map[string]int{}
	for _, row := range rows {
		counts[row.Status]++
	}
	return counts
}

func cachevalueStatusAttributionFromRows(rows []cachevalueStatusRow) cachevalueStatusAttribution {
	owners := map[string]cachevalueStatusBucket{}
	domains := map[string]cachevalueStatusBucket{}
	fidelityBuckets := map[string]cachevalueStatusBucket{}
	fidelities := map[string]int{}
	evidence := map[string]int{}
	ownerExamples := map[string]cachevalueStatusRow{}
	domainExamples := map[string]cachevalueStatusRow{}
	fidelityExamples := map[string]cachevalueStatusRow{}

	for _, row := range rows {
		owner := cachevalueAttributionKey(row.Owner, "unknown")
		domain := cachevalueAttributionKey(row.FailureDomain, "unknown")
		fidelity := cachevalueAttributionKey(row.Fidelity, "unknown")
		ev := cachevalueAttributionKey(row.Evidence, "unknown")
		working := cachevalueRowWorking(row)
		problem := cachevalueRowProblem(row)

		cachevalueBumpBucket(owners, owner, working, problem)
		cachevalueBumpBucket(domains, domain, working, problem)
		cachevalueBumpBucket(fidelityBuckets, fidelity, working, problem)
		fidelities[fidelity]++
		evidence[ev]++
		if problem {
			cachevalueRememberExample(ownerExamples, owner, row)
			cachevalueRememberExample(domainExamples, domain, row)
			cachevalueRememberExample(fidelityExamples, fidelity, row)
		}
	}

	return cachevalueStatusAttribution{
		Owners:          owners,
		Fidelities:      fidelities,
		Evidence:        evidence,
		FailureDomains:  domains,
		ProblemOwners:   cachevalueProblemFindings(owners, ownerExamples),
		ProblemDomains:  cachevalueProblemFindings(domains, domainExamples),
		ProblemFidelity: cachevalueProblemFindings(fidelityBuckets, fidelityExamples),
	}
}

func cachevalueBumpBucket(m map[string]cachevalueStatusBucket, key string, working, problem bool) {
	b := m[key]
	b.Total++
	if working {
		b.Working++
	}
	if problem {
		b.Problem++
	}
	m[key] = b
}

func cachevalueRememberExample(m map[string]cachevalueStatusRow, key string, row cachevalueStatusRow) {
	if _, ok := m[key]; ok {
		return
	}
	m[key] = row
}

func cachevalueProblemFindings(buckets map[string]cachevalueStatusBucket, examples map[string]cachevalueStatusRow) []cachevalueStatusFinding {
	keys := make([]string, 0, len(buckets))
	for key, b := range buckets {
		if b.Problem > 0 {
			keys = append(keys, key)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := buckets[keys[i]], buckets[keys[j]]
		if a.Problem != b.Problem {
			return a.Problem > b.Problem
		}
		if a.Working != b.Working {
			return a.Working > b.Working
		}
		return keys[i] < keys[j]
	})
	if len(keys) > 8 {
		keys = keys[:8]
	}
	out := make([]cachevalueStatusFinding, 0, len(keys))
	for _, key := range keys {
		b := buckets[key]
		row := examples[key]
		out = append(out, cachevalueStatusFinding{
			Key:        key,
			Problems:   b.Problem,
			Working:    b.Working,
			Example:    cachevalueFindingExample(row),
			NextAction: row.NextAction,
		})
	}
	return out
}

func cachevalueFindingExample(row cachevalueStatusRow) string {
	if row.Component == "" {
		return ""
	}
	detail := strings.TrimSpace(row.Reason)
	if detail == "" {
		detail = strings.TrimSpace(row.SessionImpact)
	}
	if detail == "" {
		return row.Component + " status=" + row.Status
	}
	return row.Component + " status=" + row.Status + ": " + truncStatus(detail, 180)
}

func cachevalueAttributionKey(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func cachevalueStatusVerdict(rows []cachevalueStatusRow) (string, string) {
	working, problems := 0, 0
	for _, row := range rows {
		if cachevalueRowWorking(row) {
			working++
		}
		if cachevalueRowProblem(row) {
			problems++
		}
	}
	if problems == 0 {
		return "OK", fmt.Sprintf("%d cache component(s) report working or intentionally inactive", working)
	}
	if working == 0 {
		return "INSUFFICIENT", fmt.Sprintf("%d cache component(s) lack evidence or are unavailable", problems)
	}
	return "PARTIAL", fmt.Sprintf("%d working/available component(s), %d component(s) missing, gated, unavailable, or dollar-blind", working, problems)
}

func cachevalueVerdictRank(v string) (int, bool) {
	switch strings.ToUpper(strings.TrimSpace(v)) {
	case "OK":
		return 0, true
	case "PARTIAL":
		return 1, true
	case "INSUFFICIENT":
		return 2, true
	}
	return 0, false
}

// cachevalueStatusGateExit maps the folded verdict onto the opt-in gate exit:
// 1 when the verdict is at or worse than the caller-chosen floor, else 0. An
// unrecognized verdict or floor never gates (the floor is validated at flag
// parse; the verdict set is closed by cachevalueStatusVerdict).
func cachevalueStatusGateExit(verdict, floor string) int {
	floorRank, ok := cachevalueVerdictRank(floor)
	if !ok {
		return 0
	}
	rank, ok := cachevalueVerdictRank(verdict)
	if !ok || rank < floorRank {
		return 0
	}
	return 1
}

func cachevalueRowWorking(row cachevalueStatusRow) bool {
	switch row.Status {
	case "measured", "active", "available", "ready", "observed", "no-op", "saved", "no_effect", "forecast":
		return true
	default:
		return false
	}
}

func cachevalueRowProblem(row cachevalueStatusRow) bool {
	switch row.Status {
	case "missing", "insufficient", "dollar_blind", "gated", "partial", "dropped", "no_saving",
		"cold_write_only", "high_pressure", "no_provider_cache_evidence",
		"no_usage", "not_observed_from_transcript", "forecast_only", "not_2x", "not_ready", "error",
		"context_planning", "provider_cache_behavior", "unattributed", "false_warm", "turns_reordered", "not_observed",
		"replay_failed", "score_failed":
		return true
	case "unavailable":
		return row.Selected || row.Component == "recent_vcache_snapshot" ||
			row.Component == "ablation_report" || row.Component == "vcache_score_report" ||
			row.Component == "vcache_context_join_report" || row.Component == "vcache_observe_report" ||
			row.Component == "vcache_context_witness_report" ||
			strings.HasPrefix(row.Component, "headroom_bench")
	default:
		return false
	}
}

func cachevalueStatusNextActions(rows []cachevalueStatusRow) []string {
	seen := map[string]bool{}
	var actions []string
	for _, row := range rows {
		if !cachevalueRowProblem(row) || strings.TrimSpace(row.NextAction) == "" {
			continue
		}
		action := row.Component + ": " + row.NextAction
		if seen[action] {
			continue
		}
		seen[action] = true
		actions = append(actions, action)
	}
	sort.Strings(actions)
	if len(actions) > 8 {
		return actions[:8]
	}
	return actions
}

func cachevalueNonEmpty(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return strings.TrimSpace(v)
}

func rowImpactNext(row cachevalueStatusRow) string {
	impact := row.SessionImpact
	if cachevalueShowRowReason(row) && strings.TrimSpace(row.Reason) != "" {
		impact += " reason=" + row.Reason
	}
	if strings.TrimSpace(row.NextAction) == "" {
		return impact
	}
	return impact + " next=" + row.NextAction
}

func cachevalueShowRowReason(row cachevalueStatusRow) bool {
	if row.Component == "vcache_context_witness_report" || row.Component == "vcache_context_witness_plane" {
		return true
	}
	switch row.Status {
	case "dropped", "unavailable", "error", "no_saving", "not_2x", "forecast_only", "gated",
		"context_planning", "provider_cache_behavior", "unattributed", "false_warm", "turns_reordered", "not_observed",
		"replay_failed", "score_failed":
		return true
	default:
		return false
	}
}

func truncStatus(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "~"
}

func cachevalueEmptyDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func nonEmptySessionName(s sessionaudit.Session) string {
	if strings.TrimSpace(s.Session) != "" {
		return s.Session
	}
	if strings.TrimSpace(s.Path) != "" {
		base := filepath.Base(s.Path)
		return strings.TrimSuffix(base, filepath.Ext(base))
	}
	return "unknown_session"
}

func quotePathForHint(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "FILE"
	}
	if strings.ContainsAny(path, " \t\"'") {
		return fmt.Sprintf("%q", path)
	}
	return path
}

func fmtFloatPtr(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.3f", *v)
}
