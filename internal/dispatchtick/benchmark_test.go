package dispatchtick

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

var (
	benchPreflightSink     PreflightResult
	benchVelocityPriorSink VelocityPrior
	benchHostCapSink       HostCapacityInfo
	benchSetpointSink      SetpointPlan
	benchRouteSink         AccountRouteResult
	benchTierRouteSink     TierRouteResult
	benchWaveSink          AccountWaveResult
	benchSeatPoolSink      SeatPoolResult
	benchIssueRouteSink    IssueRoute
	benchRouterPayloadSink RouterPayload
	benchOrderSink         []int
	benchEDFSink           EDFPlan
	benchCmdSink           []string
	benchPromptSink        IssuePromptRecord
)

// BenchmarkEvaluatePreflight_Roomy exercises unconstrained tick evaluation where
// host resources and kernel targets provide ample headroom.
func BenchmarkEvaluatePreflight_Roomy(b *testing.B) {
	in := preflightInput()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := EvaluatePreflight(in)
		if res.Verdict != PreflightOKVerdict {
			b.Fatalf("unexpected verdict: %s", res.Verdict)
		}
		benchPreflightSink = res
	}
}

// BenchmarkEvaluatePreflight_Constrained exercises preflight tick evaluation when
// capacity is bound by multiple active constraints: core/RAM/thread budgets,
// forecast worker floors, kernel targets, and seat pools.
func BenchmarkEvaluatePreflight_Constrained(b *testing.B) {
	cores := 4
	ram := 4096
	threads := 120
	target := 5
	in := PreflightInput{
		Workspace:   "repo",
		MaxWorkers:  20,
		WorkerFloor: 3,
		Host:        HostCheck{Safe: true},
		Account:     AccountCheck{Available: true, Tag: "worker-tight", Tier: 1, Model: "claude"},
		Kernel:      KernelCheck{Alive: IntPtr(2), Target: &target, Verdict: "FILLING"},
		Seat:        SeatCheck{Total: IntPtr(10)},
		Resources: HostResources{
			Cores:        &cores,
			FreeRAMMB:    &ram,
			TotalThreads: &threads,
		},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := EvaluatePreflight(in)
		if res.Cap <= 0 {
			b.Fatalf("unexpected cap: %d", res.Cap)
		}
		benchPreflightSink = res
	}
}

// BenchmarkApplyGateBackpressure measures folding observed gate latency rollups
// (hook p99 vs budget) into the preflight admission decision.
func BenchmarkApplyGateBackpressure(b *testing.B) {
	in := preflightInput()
	in.MaxWorkers = 8
	in.Kernel = KernelCheck{Alive: IntPtr(3), Target: IntPtr(10), Verdict: "FILLING"}
	base := EvaluatePreflight(in)
	check := GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 15, P99MS: 380},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ApplyGateBackpressure(base, check)
		if res.Verdict != PreflightRefuseGate {
			b.Fatalf("unexpected verdict: %s", res.Verdict)
		}
		benchPreflightSink = res
	}
}

// BenchmarkApplyRateLimitBackpressure measures folding transient concurrency
// 429 overload bursts into the preflight admission cap.
func BenchmarkApplyRateLimitBackpressure(b *testing.B) {
	in := preflightInput()
	in.MaxWorkers = 8
	in.Kernel = KernelCheck{Alive: IntPtr(3), Target: IntPtr(10), Verdict: "FILLING"}
	base := EvaluatePreflight(in)
	check := RateLimitCheck{
		Recent:    5,
		Threshold: DefaultRateLimitMin429,
		Window:    15 * time.Minute,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ApplyRateLimitBackpressure(base, check)
		if res.Verdict != PreflightRefuseRateLimit {
			b.Fatalf("unexpected verdict: %s", res.Verdict)
		}
		benchPreflightSink = res
	}
}

// BenchmarkApplyChurnBackpressure measures folding whole-host spawn storm
// bursts into the preflight admission cap.
func BenchmarkApplyChurnBackpressure(b *testing.B) {
	in := preflightInput()
	in.MaxWorkers = 8
	in.Kernel = KernelCheck{Alive: IntPtr(3), Target: IntPtr(10), Verdict: "FILLING"}
	base := EvaluatePreflight(in)
	check := ChurnCheck{
		Recent:        25,
		WindowSeconds: 0,
		Threshold:     DefaultChurnBurstThreshold,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ApplyChurnBackpressure(base, check)
		if res.Verdict != PreflightRefuseChurn {
			b.Fatalf("unexpected verdict: %s", res.Verdict)
		}
		benchPreflightSink = res
	}
}

// BenchmarkApplyWIPLimit measures flow-limit enforcement against the started WIP census.
func BenchmarkApplyWIPLimit(b *testing.B) {
	in := preflightInput()
	in.MaxWorkers = 10
	base := EvaluatePreflight(in)
	census := WIPCensus{
		Measured: true,
		Started:  6,
		Limit:    8,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ApplyWIPLimit(base, census)
		if res.Cap <= 0 {
			b.Fatalf("unexpected cap: %d", res.Cap)
		}
		benchPreflightSink = res
	}
}

// BenchmarkEvaluateVelocityPrior measures calculating the raised collision prior
// from git module revision velocity.
func BenchmarkEvaluateVelocityPrior(b *testing.B) {
	check := VelocityCheck{
		Lane:      "gateway",
		RevDelta:  15,
		Weeks:     1.0,
		Threshold: DefaultHotRevsPerWeek,
		Present:   true,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		prior := EvaluateVelocityPrior(check)
		if !prior.Hot {
			b.Fatalf("expected hot velocity prior")
		}
		benchVelocityPriorSink = prior
	}
}

// BenchmarkHostCapacityWith_Cadence measures deriving host worker capacity and
// identifying the binding hardware dimension under explicit budgets.
func BenchmarkHostCapacityWith_Cadence(b *testing.B) {
	cores := 32
	ram := 65536
	threads := 800
	res := HostResources{
		Cores:        &cores,
		FreeRAMMB:    &ram,
		TotalThreads: &threads,
	}
	budgets := HostBudgets{
		CoresPerWorker:   2,
		RAMMBPerWorker:   2048,
		ThreadsPerCore:   32,
		ThreadsPerWorker: 16,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		info := HostCapacityWith(res, budgets)
		if info.HostCap == nil || *info.HostCap <= 0 {
			b.Fatalf("invalid host cap")
		}
		benchHostCapSink = info
	}
}

// BenchmarkReconcileSetpoint_Cadence measures level-triggered autoscaling setpoint
// reconciliation across grow, steady, and drain branches.
func BenchmarkReconcileSetpoint_Cadence(b *testing.B) {
	cases := []struct {
		live     int
		setpoint int
	}{
		{live: 4, setpoint: 8},  // grow
		{live: 8, setpoint: 8},  // steady
		{live: 12, setpoint: 6}, // drain
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c := cases[i%len(cases)]
		plan := ReconcileSetpoint(c.live, c.setpoint)
		if !plan.Active {
			b.Fatalf("plan must be active")
		}
		benchSetpointSink = plan
	}
}

// BenchmarkRouteAccount_StateAssessment measures evaluating available account pools,
// session caps, and live load to select the primary worker account.
func BenchmarkRouteAccount_StateAssessment(b *testing.B) {
	rows := accountRowsFixture()
	in := AccountRouteInput{
		Rows:     rows,
		Product:  "claude",
		WorkKind: "engineering",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := RouteAccount(in)
		if !res.OK {
			b.Fatalf("route account failed: %s", res.Reason)
		}
		benchRouteSink = res
	}
}

// BenchmarkRouteAccountForTier_StateAssessment measures tier-aware account routing
// matching work complexity against model capability floors.
func BenchmarkRouteAccountForTier_StateAssessment(b *testing.B) {
	rows := accountRowsFixture()
	issue := IssueTier{
		Required: modelroute.TierT1,
		Optimal:  modelroute.TierT0,
		HasTier:  true,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := RouteAccountForTier(rows, "claude", issue)
		if !res.OK {
			b.Fatalf("tier routing failed: %s", res.FallbackReason)
		}
		benchTierRouteSink = res
	}
}

// BenchmarkAllocateWave_StateAssessment measures allocating multi-worker waves across
// distinct account pools while respecting session caps and live leases.
func BenchmarkAllocateWave_StateAssessment(b *testing.B) {
	rows := accountRowsFixture()
	in := AccountWaveInput{
		Rows:     rows,
		Count:    6,
		Product:  "claude",
		WorkKind: "engineering",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := AllocateWave(in)
		if !res.OK {
			b.Fatalf("allocate wave failed: %s", res.Reason)
		}
		benchWaveSink = res
	}
}

// BenchmarkBuildSeatPool_StateAssessment measures building the partitioned inventory
// of free, leased, and blocked account seats.
func BenchmarkBuildSeatPool_StateAssessment(b *testing.B) {
	rows := accountRowsFixture()
	leases := []SeatLease{
		{Worker: "resolve-1", Tag: "day26", Dir: "C:/Users/u/.claude-day26"},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool := BuildSeatPool(rows, leases, "claude")
		if pool.TotalSeats <= 0 {
			b.Fatalf("invalid seat pool capacity")
		}
		benchSeatPoolSink = pool
	}
}

// BenchmarkRouteIssue measures single issue routing against lane taxonomy and file trees.
func BenchmarkRouteIssue(b *testing.B) {
	issue := routerIssue(101, "fix(gateway): resolve deadlock in stream buffer", []string{"gateway"}, "Touches internal/gateway/buffer.go to fix race condition.")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		route := RouteIssue(issue, routerTestTaxonomy, RouteOptions{})
		if route.Lane == "" {
			b.Fatalf("expected routed lane")
		}
		benchIssueRouteSink = route
	}
}

func benchmarkScopedIssueBody(lane, path string) string {
	return strings.Join([]string{
		"## Parent context",
		"benchmark dispatch fixture",
		"## Current state",
		"Routing benchmark issues.",
		"## Why this is next",
		"Measuring production issue routing performance.",
		"## Working spine",
		"Route issues against lane taxonomy.",
		"## Work unit",
		"leaf",
		"## Expected steps",
		"2",
		"## Trigger",
		"Benchmark execution.",
		"## Batch policy",
		"One issue per benchmark batch.",
		"## In scope",
		"Route the benchmark issues.",
		"## Out of scope",
		"Do not spawn workers.",
		"## Done condition",
		"Issues are routed to lanes.",
		"## Witness",
		"go test ./internal/dispatchtick",
		"## Acceptance gate",
		"go test ./internal/dispatchtick",
		"## Lane",
		lane,
		"## Path hints",
		"- `" + path + "`",
		"## Boundary notes",
		"Public issue only.",
		"## Closure binding",
		"Resolving commit cites #N.",
	}, "\n\n")
}

// BenchmarkRouteIssues_Backlog measures multi-issue backlog routing across lane taxonomies.
func BenchmarkRouteIssues_Backlog(b *testing.B) {
	issues := []Issue{
		routerIssue(1, "fix(gateway): buffer stall", []string{"gateway"}, benchmarkScopedIssueBody("gateway", "internal/gateway/stream.go")),
		routerIssue(2, "feat(compute): add batching", []string{"compute"}, benchmarkScopedIssueBody("compute", "internal/compute/batch.go")),
		routerIssue(3, "docs: update install guide", []string{"docs"}, benchmarkScopedIssueBody("docs", "docs/install.md")),
		routerIssue(4, "fix(tools): test doctor fix", []string{"tools"}, benchmarkScopedIssueBody("tools", "tools/doctor.py")),
		routerIssue(5, "perf(model): quantize weights", []string{"model"}, benchmarkScopedIssueBody("model", "internal/model/weights.go")),
		routerIssue(6, "refactor(abi): update wire format", []string{"abi"}, benchmarkScopedIssueBody("abi", "internal/abi/wire.go")),
		routerIssue(7, "ci: bump actions version", []string{"ci"}, benchmarkScopedIssueBody("ci", ".github/workflows/ci.yml")),
		routerIssue(8, "bench: add latency bench", []string{"bench"}, benchmarkScopedIssueBody("bench", "internal/bench/lat.go")),
	}
	in := RouterInput{
		Workspace: "repo",
		Taxonomy:  routerTestTaxonomy,
		Issues:    issues,
		Injected:  true,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		payload := RouteIssues(in)
		if len(payload.Lanes) == 0 {
			b.Fatalf("expected non-empty lanes in payload")
		}
		benchRouterPayloadSink = payload
	}
}

// BenchmarkOrderLaneCandidates measures priority weight and recency ordering of lane candidates.
func BenchmarkOrderLaneCandidates(b *testing.B) {
	cands := []LaneCandidate{
		{Number: 101, Weight: PriorityWeightP0, ReadySince: 1700000000},
		{Number: 102, Weight: PriorityWeightP2, ReadySince: 1700000100},
		{Number: 103, Weight: PriorityWeightP1, ReadySince: 1700000050},
		{Number: 104, Weight: PriorityWeightDefault, ReadySince: 1700000200},
		{Number: 105, Weight: PriorityWeightP1, ReadySince: 1700000030},
		{Number: 106, Weight: PriorityWeightP0, ReadySince: 1700000080},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ordered := OrderLaneCandidates(cands, false)
		if len(ordered) != len(cands) {
			b.Fatalf("order mismatch: got %d, want %d", len(ordered), len(cands))
		}
		benchOrderSink = ordered
	}
}

// BenchmarkPlanEDF_Admission measures earliest-deadline-first admission scheduling
// and simulated load-shedding of degradable units.
func BenchmarkPlanEDF_Admission(b *testing.B) {
	planner := EDFPlanner{}
	items := []DeadlineItem{
		{ID: "task-1", DeadlineTick: 100, CostHintTicks: 20, Degradable: true},
		{ID: "task-2", DeadlineTick: 50, CostHintTicks: 30, Degradable: false},
		{ID: "task-3", DeadlineTick: 80, CostHintTicks: 40, Degradable: true},
		{ID: "task-4", DeadlineTick: 120, CostHintTicks: 15, Degradable: true},
		{ID: "task-5", DeadlineTick: 60, CostHintTicks: 25, Degradable: false},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan := planner.Plan(items)
		if len(plan.Order)+len(plan.Shed) != len(items) {
			b.Fatalf("plan items mismatch")
		}
		benchEDFSink = plan
	}
}

// BenchmarkBuildWorkerCommand measures worker CLI argv generation with model flags,
// fallback chains, and execution settings.
func BenchmarkBuildWorkerCommand(b *testing.B) {
	launch := WorkerLaunch{
		Model:      "claude-opus-4-8",
		Fallback:   "claude-sonnet-5",
		Effort:     "xhigh",
		AccountTag: "seat-1",
		AccountDir: "C:/seats/1",
	}
	prompt := "Resolve issue #1234: race condition in worker heartbeat"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cmd, err := BuildWorkerCommand("claude", prompt, launch)
		if err != nil {
			b.Fatalf("build command failed: %v", err)
		}
		benchCmdSink = cmd
	}
}

// BenchmarkLaunchCommandShape measures status-safe argv redaction of workspace paths,
// accounts, and sensitive flag/URL values.
func BenchmarkLaunchCommandShape(b *testing.B) {
	cmd := []string{
		"claude", "-p",
		"--account", "secret-account-tag",
		"--config-dir", "C:/Users/test/.claude-secret",
		"--base-url", "https://user:pass@api.anthropic.com/v1/messages?key=secret",
		"task prompt",
	}
	acct := Account{
		Tag: "secret-account-tag",
		Dir: "C:/Users/test/.claude-secret",
	}
	workspace := "C:/Users/test/workspace/fak"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		shaped := LaunchCommandShape(cmd, workspace, acct)
		if len(shaped) == 0 {
			b.Fatalf("empty shaped command")
		}
		benchCmdSink = shaped
	}
}

// BenchmarkBuildIssuePrompt measures issue prompt construction and project-work brief extraction.
func BenchmarkBuildIssuePrompt(b *testing.B) {
	in := IssuePromptInput{
		Number:    42,
		Lane:      "gateway",
		Title:     "fix(gateway): resolve stream buffer race",
		Body:      "Fix the race in internal/gateway/stream.go.\n\n## Estimated Work\n2 hours.\n\n## Completion Standard\nPass all gateway unit tests.",
		Workspace: "repo",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		record := BuildIssuePrompt(in)
		if record.PromptChars <= 0 {
			b.Fatalf("empty prompt generated")
		}
		benchPromptSink = record
	}
}

// TestBenchmarkDispatchtickSanity verifies all Benchmark* functions execute cleanly
// and complete at least one iteration.
func TestBenchmarkDispatchtickSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 20-benchmark sweep in short mode")
	}
	benchmarks := []struct {
		name string
		fn   func(*testing.B)
	}{
		{"BenchmarkEvaluatePreflight_Roomy", BenchmarkEvaluatePreflight_Roomy},
		{"BenchmarkEvaluatePreflight_Constrained", BenchmarkEvaluatePreflight_Constrained},
		{"BenchmarkApplyGateBackpressure", BenchmarkApplyGateBackpressure},
		{"BenchmarkApplyRateLimitBackpressure", BenchmarkApplyRateLimitBackpressure},
		{"BenchmarkApplyChurnBackpressure", BenchmarkApplyChurnBackpressure},
		{"BenchmarkApplyWIPLimit", BenchmarkApplyWIPLimit},
		{"BenchmarkEvaluateVelocityPrior", BenchmarkEvaluateVelocityPrior},
		{"BenchmarkHostCapacityWith_Cadence", BenchmarkHostCapacityWith_Cadence},
		{"BenchmarkReconcileSetpoint_Cadence", BenchmarkReconcileSetpoint_Cadence},
		{"BenchmarkRouteAccount_StateAssessment", BenchmarkRouteAccount_StateAssessment},
		{"BenchmarkRouteAccountForTier_StateAssessment", BenchmarkRouteAccountForTier_StateAssessment},
		{"BenchmarkAllocateWave_StateAssessment", BenchmarkAllocateWave_StateAssessment},
		{"BenchmarkBuildSeatPool_StateAssessment", BenchmarkBuildSeatPool_StateAssessment},
		{"BenchmarkRouteIssue", BenchmarkRouteIssue},
		{"BenchmarkRouteIssues_Backlog", BenchmarkRouteIssues_Backlog},
		{"BenchmarkOrderLaneCandidates", BenchmarkOrderLaneCandidates},
		{"BenchmarkPlanEDF_Admission", BenchmarkPlanEDF_Admission},
		{"BenchmarkBuildWorkerCommand", BenchmarkBuildWorkerCommand},
		{"BenchmarkLaunchCommandShape", BenchmarkLaunchCommandShape},
		{"BenchmarkBuildIssuePrompt", BenchmarkBuildIssuePrompt},
	}

	for _, tc := range benchmarks {
		t.Run(tc.name, func(t *testing.T) {
			res := testing.Benchmark(tc.fn)
			if res.N <= 0 {
				t.Fatalf("%s failed to execute any iterations: %+v", tc.name, res)
			}
		})
	}
}
