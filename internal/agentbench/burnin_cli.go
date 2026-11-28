package agentbench

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

var burnInCLITaskRunner = taskrun.Run

func runBurnInCLICommand(ctx context.Context, stdout, stderr io.Writer, opts options) int {
	repo, err := resolveRepository(opts.repo)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: repository: %v\n", err)
		return 2
	}
	if opts.endpoint == "" || opts.out == "" {
		fmt.Fprintln(stderr, "agentbench: --endpoint and --out are required")
		return 2
	}
	if opts.burnInMaxGoroutineDelta < 0 || opts.burnInMaxHeapGrowth < 0 || opts.burnInMaxFDDelta < 0 || opts.burnInEpochs < 0 || opts.duration < 0 {
		fmt.Fprintln(stderr, "agentbench: burn-in bounds, epochs, and duration must be non-negative")
		return 2
	}
	endpoint, err := normalizeEndpoint(opts.endpoint)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: endpoint: %v\n", err)
		return 2
	}
	out, err := resolveOutput(repo, opts.out)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: output: %v\n", err)
		return 2
	}
	conc := opts.concurrency
	if conc == 0 {
		conc = 2
	}
	planDuration := opts.duration
	if planDuration != 2*time.Hour && planDuration != 8*time.Hour {
		planDuration = 0
	}
	plan, err := profileplan.Build(profileplan.Options{
		Profile:              "burn-in",
		PreferredConcurrency: conc,
		Duration:             planDuration,
	})
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: burn-in plan: %v\n", err)
		return 2
	}
	allowShort := false
	if opts.duration > 0 && opts.duration != 2*time.Hour && opts.duration != 8*time.Hour {
		allowShort = true
		plan.Deadline = opts.duration
		plan.DeadlineText = opts.duration.String()
		plan.DrainGrace = min(90*time.Second, max(50*time.Millisecond, opts.duration/4))
		plan.DrainGraceText = plan.DrainGrace.String()
	}
	if opts.burnInEpochs > 0 {
		allowShort = true
		plan.MinimumEpochs = opts.burnInEpochs
		if opts.duration == 0 {
			plan.Deadline = 2 * time.Minute
			plan.DeadlineText = plan.Deadline.String()
			plan.DrainGrace = time.Second
			plan.DrainGraceText = plan.DrainGrace.String()
		}
	}

	receipt, runErr := runBurnInCLI(ctx, stderr, repo, endpoint, opts.model, out, plan, allowShort, opts)
	if printErr := printValue(stdout, receipt, opts.json); printErr != nil && runErr == nil {
		runErr = printErr
	}
	if runErr != nil {
		fmt.Fprintf(stderr, "agentbench: burn-in incomplete: %v\n", runErr)
		return 1
	}
	return 0
}

func runBurnInCLI(ctx context.Context, stderr io.Writer, repo, endpoint, requestedModel, out string, plan profileplan.Plan, allowShort bool, opts options) (burnInReceipt, error) {
	receipt := burnInReceipt{Status: "INCOMPLETE"}
	if err := os.MkdirAll(out, 0700); err != nil {
		return receipt, err
	}
	model, err := resolveNormalModel(ctx, endpoint, requestedModel)
	if err != nil {
		return receipt, err
	}
	base, err := referenceBaseURL(endpoint)
	if err != nil {
		return receipt, err
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxConnsPerHost = max(8, plan.PreferredConcurrency)
	transport.MaxIdleConnsPerHost = max(8, plan.PreferredConcurrency)
	defer transport.CloseIdleConnections()
	client := &http.Client{
		Transport:     transport,
		Timeout:       requestTimeout,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	reference, err := discoverNativeReference(ctx, client, base.String(), model)
	if err != nil {
		return receipt, err
	}
	if reference.ContextWindowTokens < plan.MinimumModelContextTokens {
		return receipt, fmt.Errorf("burn-in context %d below %d", reference.ContextWindowTokens, plan.MinimumModelContextTokens)
	}
	repoInput, err := freezeRepositoryInput(repo)
	if err != nil {
		return receipt, err
	}
	inputs := newNormalInputs(repo, repoInput)
	normalCorpus, err := buildNormalCorpus(ctx, reference, inputs.sourceRoot)
	if err != nil {
		return receipt, fmt.Errorf("build normal corpus: %w", err)
	}
	burnCorpus, err := buildBurnInCorpus(ctx, reference, normalCorpus, inputs.sourceRoot)
	if err != nil {
		return receipt, fmt.Errorf("build burn-in corpus: %w", err)
	}
	identity := burnInIdentity{
		ModelID:        model,
		ConfigDigest:   burnDigest(fmt.Sprintf("%s:%d:%s", reference.ModelID, reference.ContextWindowTokens, plan.SeedHex)),
		WorkloadDigest: burnDigest(fmt.Sprintf("%s:%s:%d", burnCorpus.Schema, burnCorpus.Revision, len(burnCorpus.Sessions))),
	}
	exe, err := os.Executable()
	if err != nil {
		return receipt, err
	}
	bounds := burnInLeakBounds{
		MaxGoroutineDelta:  opts.burnInMaxGoroutineDelta,
		MaxHeapGrowthBytes: opts.burnInMaxHeapGrowth,
		MaxFDDelta:         opts.burnInMaxFDDelta,
	}
	cfg := burnInControllerConfig{
		Plan:               plan,
		Identity:           identity,
		Now:                time.Now,
		AllowShortSchedule: allowShort,
		MaxEpochs:          opts.burnInEpochs,
		LeakBounds:         &bounds,
		RunEpoch: func(epochCtx context.Context, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
			epochOut := filepath.Join(out, fmt.Sprintf("epoch-%03d", spec.Ordinal))
			return runBurnInEpoch(epochCtx, burnInEpochConfig{
				Client:    client,
				Endpoint:  endpoint,
				Model:     model,
				OutDir:    epochOut,
				Corpus:    burnCorpus,
				Reference: reference,
				Now:       time.Now,
				RunTasks:  burnInCLITaskRunner,
				TaskOptions: taskrun.Options{
					Executable:  exe,
					Endpoint:    base.String(),
					Model:       model,
					OutDir:      filepath.Join(epochOut, "tasks"),
					Concurrency: spec.Concurrency,
					TaskLimit:   spec.Tasks,
				},
			}, spec, cutoff)
		},
		Drain: func(_ context.Context, _ time.Duration) error {
			transport.CloseIdleConnections()
			return nil
		},
	}
	if allowShort {
		cfg.Normal = &burnInNormalAdmission{
			Qualified:   true,
			Concurrency: plan.PreferredConcurrency,
			Identity:    identity,
		}
	} else {
		cfg.RunNormalPreflight = func(pCtx context.Context) (burnInNormalAdmission, error) {
			normalReceipt, normalErr := runNormal(pCtx, stderr, repo, endpoint, model, filepath.Join(out, "normal-preflight"))
			if normalErr != nil {
				return burnInNormalAdmission{}, normalErr
			}
			return burnInNormalAdmission{
				Qualified:   normalReceipt.OverallVerdict == "NORMAL_PASSED",
				Concurrency: normalReceipt.CapacitySelection.Selected,
				Identity:    identity,
			}, nil
		}
	}
	receipt, runErr := runBurnInController(ctx, cfg)
	if writeErr := writeNormalProgressJSON(filepath.Join(out, "receipt.json"), receipt); writeErr != nil && runErr == nil {
		runErr = writeErr
	}
	return receipt, runErr
}
