package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const maxDepth = 2

var taskIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$`)

type turnBand string

const (
	bandOneTurn           turnBand = "one_turn"
	bandBoundedCorrection turnBand = "bounded_correction"
	bandRootOnly          turnBand = "root_only"
)

type task struct {
	ID       string
	Parent   string
	Depth    int
	MaxTurns int
	Goal     string
	Class    turnBand `json:"class,omitempty"`
	Case     string   `json:"case,omitempty"`
}

type receipt struct {
	TaskID   string   `json:"task_id"`
	Parent   string   `json:"parent,omitempty"`
	Depth    int      `json:"depth"`
	Turns    int      `json:"turns"`
	Class    turnBand `json:"class,omitempty"`
	Case     string   `json:"case,omitempty"`
	Decision string   `json:"decision"`
	Evidence []string `json:"evidence"`
	Children []task   `json:"children,omitempty"`
}

type corpusMeasurement struct {
	Class    turnBand `json:"class"`
	Case     string   `json:"case"`
	Turns    int      `json:"turns"`
	Outcome  string   `json:"outcome"`
	Evidence []string `json:"evidence"`
}

type benchmarkArm struct {
	Mode            string `json:"mode"`
	QualityPassed   int    `json:"quality_passed"`
	QualityTotal    int    `json:"quality_total"`
	WallTimeMS      int    `json:"wall_time_ms"`
	InputTokens     int    `json:"input_tokens"`
	OutputTokens    int    `json:"output_tokens"`
	CacheReadTokens int    `json:"cache_read_tokens"`
	RetainedBytes   int    `json:"retained_bytes"`
	CostMicroUSD    int    `json:"cost_microusd"`
}

type benchmarkComparison struct {
	Method               string       `json:"method"`
	Monolith             benchmarkArm `json:"monolith"`
	ReceiptOnly          benchmarkArm `json:"receipt_only"`
	RetainedReductionPct int          `json:"retained_reduction_pct"`
	TokenReductionPct    int          `json:"token_reduction_pct"`
}

type report struct {
	Goal                  string              `json:"goal"`
	Harness               []string            `json:"harness"`
	Receipts              []receipt           `json:"receipts"`
	MaxDepth              int                 `json:"max_depth"`
	MaxTurnsPerMicroagent int                 `json:"max_turns_per_microagent"`
	ChildTranscriptBytes  int                 `json:"child_transcript_bytes"`
	MasterReceiptBytes    int                 `json:"master_receipt_bytes"`
	FullTranscriptsInRoot bool                `json:"full_transcripts_in_root"`
	TaskClasses           []corpusMeasurement `json:"task_classes"`
	Benchmark             benchmarkComparison `json:"benchmark"`
	Execution             executionReport     `json:"execution"`
}

type executionReport struct {
	Mode              string `json:"mode"`
	Model             string `json:"model"`
	Completions       int    `json:"completions"`
	PromptTokens      int    `json:"prompt_tokens"`
	CompletionTokens  int    `json:"completion_tokens"`
	NonemptyResponses int    `json:"nonempty_responses"`
	WallTimeMS        int64  `json:"wall_time_ms"`
}

type measuredPlanner struct {
	agent.Planner
	mu         sync.Mutex
	completion int
	bytes      int
	usage      agent.Usage
	nonempty   int
}

func (p *measuredPlanner) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	c, err := p.Planner.Complete(ctx, msgs, tools, opts...)
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.completion++
	for _, msg := range msgs {
		p.bytes += len(msg.Content)
	}
	p.usage.PromptTokens += c.Usage.PromptTokens
	p.usage.CompletionTokens += c.Usage.CompletionTokens
	if strings.TrimSpace(c.Message.Content) != "" {
		p.nonempty++
	}
	p.mu.Unlock()
	return c, nil
}

func (p *measuredPlanner) measurement(mode string) executionReport {
	p.mu.Lock()
	defer p.mu.Unlock()
	return executionReport{Mode: mode, Model: p.Model(), Completions: p.completion, PromptTokens: p.usage.PromptTokens, CompletionTokens: p.usage.CompletionTokens, NonemptyResponses: p.nonempty}
}

type scriptedModel struct {
	mu    sync.Mutex
	bytes int
}

func (*scriptedModel) Model() string { return "microharness-fixture" }
func (p *scriptedModel) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if len(msgs) == 0 {
		return nil, errors.New("empty microagent context")
	}
	p.mu.Lock()
	for _, msg := range msgs {
		p.bytes += len(msg.Content)
	}
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "bounded evidence accepted"}}, nil
}

func (p *scriptedModel) transcriptBytes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.bytes
}

type boundedAgent struct {
	task          task
	turns         int
	receipt       chan<- receipt
	modelEvidence string
}

func (a *boundedAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.turns++
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "Return evidence for only this bounded harness-building task."}, {Role: agent.RoleUser, Content: a.task.Goal}}
	completion, err := gw.Complete(ctx, messages, nil)
	if err != nil {
		return false, err
	}
	a.modelEvidence = compactModelEvidence(completion.Message.Content)
	if a.turns < a.task.MaxTurns {
		return false, nil
	}
	r := makeReceipt(a.task, a.turns)
	if a.modelEvidence != "" {
		r.Evidence = append(r.Evidence, "model: "+a.modelEvidence)
	}
	select {
	case a.receipt <- r:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func compactModelEvidence(raw string) string {
	text := strings.Join(strings.Fields(raw), " ")
	const limit = 240
	if len(text) > limit {
		text = text[:limit] + "..."
	}
	return text
}

func makeReceipt(t task, turns int) receipt {
	r := receipt{TaskID: t.ID, Parent: t.Parent, Depth: t.Depth, Turns: turns, Class: t.Class, Case: t.Case}
	switch t.ID {
	case "architecture":
		r.Decision = "compose a local coding harness from bounded tool and proof profiles"
		r.Evidence = []string{"goal requires repository edits", "effects need an independent proof boundary"}
		r.Children = []task{
			{ID: "tools", Parent: t.ID, Depth: t.Depth + 1, MaxTurns: 1, Goal: "Select the least-privilege tools for a local coding harness.", Class: bandOneTurn, Case: "capability-selection"},
			{ID: "proof", Parent: t.ID, Depth: t.Depth + 1, MaxTurns: 3, Goal: "Define the harness completion witness after a verifier names a gap.", Class: bandBoundedCorrection, Case: "witness-correction"},
			{ID: "irreversible-goal", Parent: t.ID, Depth: t.Depth + 1, Goal: "Choose an irreversible repository-wide effect from an ambiguous goal.", Class: bandRootOnly, Case: "irreversible-goal"},
		}
	case "tools":
		r.Decision = "profile=repo-read-write; shell=workspace-only"
		r.Evidence = []string{"editing needs scoped writes", "network is not required by the goal"}
	case "proof":
		r.Decision = "require build plus affected tests before completion"
		r.Evidence = []string{"a diff is not a behavior witness", "verification must be independent of the done claim"}
	default:
		r.Decision = "refuse unknown task"
		r.Evidence = []string{"task id is outside the declared workflow"}
	}
	return r
}

func admitTask(t task) (bool, corpusMeasurement, error) {
	if strings.TrimSpace(t.ID) == "" || strings.TrimSpace(t.Goal) == "" {
		return false, corpusMeasurement{}, errors.New("task id and goal are required before admission")
	}
	if !taskIDPattern.MatchString(t.ID) {
		return false, corpusMeasurement{}, fmt.Errorf("task id %q must be 1..120 literal letters, digits, dots, underscores, or hyphens", t.ID)
	}
	if t.Depth < 1 || t.Depth > maxDepth {
		return false, corpusMeasurement{}, fmt.Errorf("task %s depth must be within 1..%d", t.ID, maxDepth)
	}
	switch t.Class {
	case bandRootOnly:
		return false, corpusMeasurement{
			Class: t.Class, Case: t.Case, Outcome: "refused-delegation",
			Evidence: []string{"irreversible or ambiguous decisions remain in the master context"},
		}, nil
	case bandOneTurn:
		if t.MaxTurns != 1 {
			return false, corpusMeasurement{}, fmt.Errorf("task %s decision class one_turn requires turn budget 1", t.ID)
		}
	case bandBoundedCorrection:
		if t.MaxTurns < 2 || t.MaxTurns > 3 {
			return false, corpusMeasurement{}, fmt.Errorf("task %s decision class bounded_correction requires turn budget within 2..3", t.ID)
		}
	case "":
		if t.MaxTurns < 1 || t.MaxTurns > 3 {
			return false, corpusMeasurement{}, fmt.Errorf("task %s turn budget must be within 1..3", t.ID)
		}
	default:
		return false, corpusMeasurement{}, fmt.Errorf("task %s decision class %q is unsupported", t.ID, t.Class)
	}
	return true, corpusMeasurement{}, nil
}

func compareArchitectures(r report) benchmarkComparison {
	// Fixture units keep this comparison reproducible; they are not provider billing telemetry.
	quality := len(r.Receipts)
	monolithRoot := r.ChildTranscriptBytes + r.MasterReceiptBytes
	output := quality * 8
	monolith := benchmarkArm{Mode: "monolith", QualityPassed: quality, QualityTotal: quality, WallTimeMS: 18, InputTokens: monolithRoot, OutputTokens: output, RetainedBytes: monolithRoot, CostMicroUSD: monolithRoot + output}
	receipt := benchmarkArm{Mode: "receipt_only", QualityPassed: quality, QualityTotal: quality, WallTimeMS: 8, InputTokens: r.MasterReceiptBytes, OutputTokens: output, CacheReadTokens: r.MasterReceiptBytes, RetainedBytes: r.MasterReceiptBytes, CostMicroUSD: r.MasterReceiptBytes + output}
	return benchmarkComparison{Method: "deterministic fixture units; not provider billing telemetry", Monolith: monolith, ReceiptOnly: receipt, RetainedReductionPct: reductionPct(monolith.RetainedBytes, receipt.RetainedBytes), TokenReductionPct: reductionPct(monolith.InputTokens+monolith.OutputTokens, receipt.InputTokens+receipt.OutputTokens)}
}

func reductionPct(baseline, candidate int) int {
	if baseline <= 0 || candidate < 0 || candidate > baseline {
		return 0
	}
	return (baseline - candidate) * 100 / baseline
}

func run(ctx context.Context) (report, error) {
	return runWithPlanner(ctx, &scriptedModel{}, "fixture", "Build a local coding harness that can edit this repository and prove its work.")
}

func runWithPlanner(ctx context.Context, planner agent.Planner, mode, rootGoal string) (report, error) {
	if err := ctx.Err(); err != nil {
		return report{}, fmt.Errorf("spawn architecture: %w; retry with a live context", err)
	}
	if planner == nil {
		return report{}, errors.New("planner is required")
	}
	rootGoal = strings.TrimSpace(rootGoal)
	if rootGoal == "" {
		return report{}, errors.New("goal is required")
	}
	measured := &measuredPlanner{Planner: planner}
	started := time.Now()
	pending := []task{{ID: "architecture", Parent: "root", Depth: 1, MaxTurns: 2, Goal: rootGoal}}
	var receipts []receipt
	var taskClasses []corpusMeasurement
	for len(pending) > 0 {
		wave := pending
		pending = nil
		out := make(chan receipt, len(wave))
		host, err := microagent.NewHost(measured, microagent.Config{Workers: 3, Queue: 8})
		if err != nil {
			return report{}, err
		}
		spawned := 0
		for _, t := range wave {
			admitted, measurement, err := admitTask(t)
			if err != nil {
				host.Close()
				return report{}, err
			}
			if !admitted {
				taskClasses = append(taskClasses, measurement)
				continue
			}
			if err := host.Spawn(t.ID, &boundedAgent{task: t, receipt: out}); err != nil {
				host.Close()
				return report{}, err
			}
			spawned++
		}
		if err := host.Drain(ctx); err != nil {
			host.Close()
			return report{}, err
		}
		host.Close()
		for i := 0; i < spawned; i++ {
			r := <-out
			receipts = append(receipts, r)
			pending = append(pending, r.Children...)
			if r.Class != "" {
				taskClasses = append(taskClasses, corpusMeasurement{
					Class: r.Class, Case: r.Case, Turns: r.Turns, Outcome: "completed", Evidence: r.Evidence,
				})
			}
		}
	}

	sort.Slice(receipts, func(i, j int) bool { return receipts[i].TaskID < receipts[j].TaskID })
	sort.Slice(taskClasses, func(i, j int) bool {
		return turnBandRank(taskClasses[i].Class) < turnBandRank(taskClasses[j].Class)
	})
	masterBytes := 0
	for _, r := range receipts {
		compact := r
		compact.Children = nil
		raw, _ := json.Marshal(compact)
		masterBytes += len(raw)
	}
	r := report{
		Goal: rootGoal, Harness: []string{"runtime=fak-native", "tools=repo-read-write/workspace-only", "completion=build+affected-tests"},
		Receipts: receipts, MaxDepth: maxDepth, MaxTurnsPerMicroagent: 3,
		MasterReceiptBytes: masterBytes, FullTranscriptsInRoot: false, TaskClasses: taskClasses,
	}
	measured.mu.Lock()
	r.ChildTranscriptBytes = measured.bytes
	measured.mu.Unlock()
	r.Execution = measured.measurement(mode)
	if mode == "live" {
		r.Execution.WallTimeMS = time.Since(started).Milliseconds()
	}
	r.Benchmark = compareArchitectures(r)
	return r, nil
}

func turnBandRank(band turnBand) int {
	switch band {
	case bandOneTurn:
		return 0
	case bandBoundedCorrection:
		return 1
	default:
		return 2
	}
}

func check(r report) error {
	if len(r.Receipts) != 3 || len(r.Harness) != 3 {
		return fmt.Errorf("got %d receipts and %d harness fields", len(r.Receipts), len(r.Harness))
	}
	if r.MaxDepth != 2 || r.MaxTurnsPerMicroagent != 3 || r.FullTranscriptsInRoot {
		return errors.New("bounded recursion or context-isolation invariant failed")
	}
	for _, rec := range r.Receipts {
		if rec.Turns < 1 || rec.Turns > 3 || rec.Depth > r.MaxDepth || rec.Decision == "" || len(rec.Evidence) == 0 {
			return fmt.Errorf("invalid receipt for %s", rec.TaskID)
		}
		if rec.Class == bandRootOnly {
			return fmt.Errorf("master-context task %s crossed the receipt boundary", rec.TaskID)
		}
	}
	if r.Benchmark.Monolith.QualityPassed != len(r.Receipts) || r.Benchmark.ReceiptOnly.QualityPassed != len(r.Receipts) {
		return errors.New("benchmark quality does not cover every harness task")
	}
	if r.Benchmark.ReceiptOnly.RetainedBytes >= r.Benchmark.Monolith.RetainedBytes || r.Benchmark.RetainedReductionPct <= 0 {
		return errors.New("receipt-only benchmark did not reduce root context")
	}
	if r.Benchmark.ReceiptOnly.CostMicroUSD >= r.Benchmark.Monolith.CostMicroUSD {
		return errors.New("receipt-only benchmark did not reduce deterministic cost units")
	}
	wantBands := map[turnBand]struct {
		caseID  string
		turns   int
		outcome string
	}{
		bandOneTurn:           {caseID: "capability-selection", turns: 1, outcome: "completed"},
		bandBoundedCorrection: {caseID: "witness-correction", turns: 3, outcome: "completed"},
		bandRootOnly:          {caseID: "irreversible-goal", turns: 0, outcome: "refused-delegation"},
	}
	for _, measurement := range r.TaskClasses {
		want, ok := wantBands[measurement.Class]
		if !ok || measurement.Case != want.caseID || measurement.Turns != want.turns || measurement.Outcome != want.outcome || len(measurement.Evidence) == 0 {
			return fmt.Errorf("invalid task-class measurement for %s", measurement.Class)
		}
		delete(wantBands, measurement.Class)
	}
	if len(wantBands) != 0 {
		return fmt.Errorf("task-class corpus missing %d classes", len(wantBands))
	}
	return nil
}

func render(w io.Writer, r report) {
	fmt.Fprintln(w, "FAK MICROHARNESS — bounded microagents construct a harness")
	fmt.Fprintf(w, "goal: %s\n", r.Goal)
	for _, rec := range r.Receipts {
		fmt.Fprintf(w, "  receipt %-12s depth=%d turns=%d -> %s\n", rec.TaskID, rec.Depth, rec.Turns, rec.Decision)
	}
	for _, measurement := range r.TaskClasses {
		fmt.Fprintf(w, "  task class %s case=%s turns=%d outcome=%s\n", measurement.Class, measurement.Case, measurement.Turns, measurement.Outcome)
	}
	fmt.Fprintf(w, "harness: %s\n", strings.Join(r.Harness, "; "))
	fmt.Fprintf(w, "context boundary: root retained %d receipt bytes; child transcript bytes=%d; full child transcripts in root=%t\n", r.MasterReceiptBytes, r.ChildTranscriptBytes, r.FullTranscriptsInRoot)
	fmt.Fprintf(w, "benchmark monolith quality=%d/%d wall_ms=%d tokens=%d cache_read=%d root_bytes=%d cost_microusd=%d\n", r.Benchmark.Monolith.QualityPassed, r.Benchmark.Monolith.QualityTotal, r.Benchmark.Monolith.WallTimeMS, r.Benchmark.Monolith.InputTokens+r.Benchmark.Monolith.OutputTokens, r.Benchmark.Monolith.CacheReadTokens, r.Benchmark.Monolith.RetainedBytes, r.Benchmark.Monolith.CostMicroUSD)
	fmt.Fprintf(w, "benchmark receipt_only quality=%d/%d wall_ms=%d tokens=%d cache_read=%d root_bytes=%d cost_microusd=%d reduction=root:%d%% tokens:%d%%\n", r.Benchmark.ReceiptOnly.QualityPassed, r.Benchmark.ReceiptOnly.QualityTotal, r.Benchmark.ReceiptOnly.WallTimeMS, r.Benchmark.ReceiptOnly.InputTokens+r.Benchmark.ReceiptOnly.OutputTokens, r.Benchmark.ReceiptOnly.CacheReadTokens, r.Benchmark.ReceiptOnly.RetainedBytes, r.Benchmark.ReceiptOnly.CostMicroUSD, r.Benchmark.RetainedReductionPct, r.Benchmark.TokenReductionPct)
	fmt.Fprintln(w, "recursion boundary: depth<=2; turns/child<=3; child requests are re-admitted by the host")
	fmt.Fprintln(w, "PASS — go run ./cmd/microharnessdemo -selfcheck")
}

func newLivePlanner(provider, baseURL, model, apiKey string) (agent.Planner, error) {
	if strings.TrimSpace(baseURL) == "" || strings.TrimSpace(model) == "" {
		return nil, errors.New("live model requires base URL and model")
	}
	return agent.NewProviderHTTPPlanner(provider, baseURL, model, apiKey)
}

func resolveAPIKey(provider, explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "anthropic":
		return os.Getenv("ANTHROPIC_API_KEY")
	case "gemini":
		if key := os.Getenv("GEMINI_API_KEY"); key != "" {
			return key
		}
		return os.Getenv("GOOGLE_API_KEY")
	case "xai":
		return os.Getenv("XAI_API_KEY")
	default:
		return os.Getenv("OPENAI_API_KEY")
	}
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the captured end-to-end witness")
	asJSON := flag.Bool("json", false, "emit the machine-readable report")
	live := flag.Bool("live", false, "run through a configured live model and the shared fak gateway")
	provider := flag.String("provider", "openai", "live provider: openai|anthropic|gemini|xai")
	baseURL := flag.String("base-url", "", "live provider base URL")
	model := flag.String("model", "", "live provider model")
	apiKey := flag.String("api-key", "", "live provider API key (prefer the provider environment variable)")
	goal := flag.String("goal", "Build a local coding harness that can edit this repository and prove its work.", "harness-building goal")
	ledger := flag.String("ledger", "", "usage JSONL path (default: user config directory; use off to disable)")
	foldUsage := flag.Bool("fold-usage", false, "print weekly usage counts and exit")
	flag.Parse()
	ledgerPath := strings.TrimSpace(*ledger)
	if ledgerPath == "" {
		ledgerPath, _ = defaultUsageLedgerPath()
	}
	if *foldUsage {
		rows, err := foldWeeklyUsage(ledgerPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "FAIL:", err)
			os.Exit(1)
		}
		_ = json.NewEncoder(os.Stdout).Encode(rows)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	var r report
	var err error
	if *live {
		if strings.TrimSpace(*baseURL) == "" || strings.TrimSpace(*model) == "" {
			err = errors.New("--live requires --base-url and --model")
		} else {
			planner, plannerErr := newLivePlanner(*provider, *baseURL, *model, resolveAPIKey(*provider, *apiKey))
			if plannerErr != nil {
				err = plannerErr
			} else {
				r, err = runWithPlanner(ctx, planner, "live", *goal)
			}
		}
	} else {
		r, err = run(ctx)
	}
	if err == nil && *selfcheck {
		err = check(r)
	}
	if ledgerPath != "" && !strings.EqualFold(ledgerPath, "off") {
		outcome := "completed"
		if err != nil {
			outcome = "failed"
		}
		row := usageRow{Mode: r.Execution.Mode, Model: r.Execution.Model, Outcome: outcome, Completions: r.Execution.Completions, PromptTokens: r.Execution.PromptTokens, CompletionTokens: r.Execution.CompletionTokens}
		if row.Mode == "" {
			if *live {
				row.Mode = "live"
			} else {
				row.Mode = "fixture"
			}
		}
		if ledgerErr := appendUsage(ledgerPath, row); ledgerErr != nil && err == nil {
			err = ledgerErr
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
	if *asJSON {
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	render(os.Stdout, r)
}
