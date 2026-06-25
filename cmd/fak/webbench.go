// fak webbench — frontier web/browser agent benchmarks as a fak-native benchmark.
// Measures the value of fak's session value stack on multi-turn web automation tasks.
// Subcommands:
//
//	describe — load the web task set + derived geometry; print the deterministic
//	           prefill-token work-elimination (the value-stack floor) at a worker
//	           sweep. Fully offline; needs no model, GPU, or network. RUNNABLE NOW.
//
//	eval     — grade predictions into the task success-rate via the official harness
//	           (when available). Gated when this box lacks the runtime.
//
//	compare  — the full comparison: fak's metric families keyed to external benchmarks,
//	           with optional side-by-side against a benchmark results file.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/webbench"
)

func cmdWebbench(argv []string) {
	if len(argv) == 0 {
		webbenchUsage()
		os.Exit(2)
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "describe":
		cmdWebbenchDescribe(rest)
	case "eval":
		cmdWebbenchEval(rest)
	case "compare":
		cmdWebbenchCompare(rest)
	case "serving":
		cmdWebbenchServing(rest)
	case "parity-gate":
		cmdWebbenchParityGate(rest)
	case "-h", "--help", "help":
		webbenchUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak webbench: unknown subcommand %q\n", sub)
		webbenchUsage()
		os.Exit(2)
	}
}

func webbenchUsage() {
	fmt.Fprint(os.Stderr, `fak webbench — Frontier Web Agent Benchmarks as a fak-native benchmark

usage:
  fak webbench describe [--dataset FILE] [--workers 1,2,4,8] [--out FILE]
        Load the web task instance set and its derived agent-workload geometry,
        then print the DETERMINISTIC prefill-token work-elimination (the value-stack
        floor) across a worker sweep. Fully offline; needs no model, GPU, or network.

  fak webbench eval --predictions preds.json [--run-id ID] [--max-workers N] [--out FILE]
        Grade a predictions file into the task success-rate via the official harness
        (when available). Gated when this box lacks the browser runtime.

  fak webbench compare [--dataset FILE] [--workers 1,2,4,8] [--predictions preds.json]
        [--bench-result FILE] [--out FILE] [--md FILE]
        THE comparison: fak's headline metric families keyed to external benchmarks,
        with optional side-by-side against a benchmark results file.

  fak webbench serving --dataset FILE [--tracks ours,sglang,vllm,fak-fronts-fleet]
        [--endpoints track=http://host/v1,...] [--metrics track=http://host/metrics,...]
        [--model MODEL] [--agents 100] [--concurrency 16] [--out FILE]
        Client-measured serving parity harness: same workload, same JSON schema,
        measured TTFT/ITL/TPOT/latency/throughput/goodput where an endpoint is
        configured; missing endpoints are emitted as "not_measured".

  fak webbench parity-gate --claim-file FILE --artifact FILE
        Reject a "parity or better" serving claim unless the artifact contains
        measured vllm, sglang, and fak-fronts-fleet tracks.

The metrics most relevant to web agents:
  A/C  net prefill work-elimination vs the naive re-prefill-every-turn harness
  B/C  cross-worker prefix reuse vs a tuned single-tenant per-worker KV
  A/B  the turn-tax (re-prefill vs KV persistence), independent of workers
`)
}

func cmdWebbenchDescribe(argv []string) {
	fs := flag.NewFlagSet("webbench describe", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to web task dataset (JSONL or JSON array)")
	workersArg := fs.String("workers", "1,2,4,8", "comma-separated worker counts to sweep")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	out := fs.String("out", "", "write the Summary JSON here (default: stdout JSON + human table on stderr)")
	_ = fs.Parse(argv)

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "fak webbench describe: --dataset is required")
		os.Exit(2)
	}

	d, err := webbench.LoadDataset(*dataset)
	must(err)

	if *limit > 0 && *limit < d.Len() {
		d = d.Limit(*limit)
	}

	workers := parseIntList(*workersArg)
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}

	gm := webbench.DefaultGeometryModel()
	s := webbench.Describe(d, gm, workers)

	if *out != "" {
		data, _ := json.MarshalIndent(s, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(s, "", "  ")
		fmt.Println(string(data))
	}

	printWebbenchSummary(os.Stderr, s, *dataset, *out)
}

func printWebbenchSummary(w *os.File, s webbench.Summary, src, out string) {
	fmt.Fprintf(w, "\n== fak webbench describe ==\n")
	fmt.Fprintf(w, "source        : %s\n", src)
	fmt.Fprintf(w, "instances     : %d\n", s.Instances)
	fmt.Fprintf(w, "difficulty    : %s\n", sortedCounts(s.DifficultyDist))
	fmt.Fprintf(w, "category      : %s\n", sortedCounts(s.CategoryDist))
	fmt.Fprintf(w, "geometry src  : %s\n", sortedCounts(s.GeometrySources))
	fmt.Fprintf(w, "turns         : min %d  median %d  max %d  (total %d navigation turns)\n",
		s.TurnsMin, s.TurnsMedian, s.TurnsMax, s.TotalTurns)
	fmt.Fprintf(w, "\nprefill-token work-elimination (deterministic floor, no model):\n")
	fmt.Fprintf(w, "  %-8s %16s %16s %16s   %8s %8s %8s\n", "workers", "A naive", "B per-agent", "C fak", "A/C", "B/C", "A/B")
	for _, p := range s.Prefill {
		fmt.Fprintf(w, "  %-8d %16s %16s %16s   %7.1fx %7.2fx %7.1fx\n",
			p.Workers,
			formatTokens(p.ANaive),
			formatTokens(p.BAgent),
			formatTokens(p.CFak),
			p.AOverC,
			p.BOverC,
			p.AOverB,
		)
	}
	fmt.Fprintf(w, "\n  A/C = net prefill work-elimination vs the naive re-prefill-every-turn harness\n")
	fmt.Fprintf(w, "  B/C = cross-worker prefix reuse (the value stack; bites at workers>1)\n")
	fmt.Fprintf(w, "  A/B = the turn-tax (re-prefill vs KV persistence), worker-independent\n")
	if out != "" {
		fmt.Fprintf(w, "\nSummary JSON written: %s\n", out)
	}
}

func cmdWebbenchEval(argv []string) {
	fs := flag.NewFlagSet("webbench eval", flag.ExitOnError)
	preds := fs.String("predictions", "", "path to predictions JSON (required)")
	benchmark := fs.String("benchmark", "browser-agent", "benchmark name (browser-agent, webvoyager, etc.)")
	runID := fs.String("run-id", "fak-webbench", "harness run id")
	maxWorkers := fs.Int("max-workers", 4, "harness parallelism")
	python := fs.String("python", "", "python interpreter (default: detected)")
	out := fs.String("out", "", "write the EvalResult JSON here (default: stdout)")
	_ = fs.Parse(argv)

	if *preds == "" {
		fmt.Fprintln(os.Stderr, "fak webbench eval: --predictions is required")
		os.Exit(2)
	}

	res, err := webbench.RunEval(webbench.EvalConfig{
		PredictionsPath: *preds,
		Benchmark:       *benchmark,
		RunID:           *runID,
		MaxWorkers:      *maxWorkers,
		Python:          *python,
	})
	must(err)

	if *out != "" {
		data, _ := json.MarshalIndent(res, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(res, "", "  ")
		fmt.Println(string(data))
	}

	fmt.Fprintf(os.Stderr, "\n== fak webbench eval ==\n")
	if res.Available {
		fmt.Fprintf(os.Stderr, "PASSED %d / %d  (%.1f%% success rate)\n", res.Passed, res.Total, res.SuccessRatePct)
		if res.ReportPath != "" {
			fmt.Fprintf(os.Stderr, "report: %s\n", res.ReportPath)
		}
	} else {
		fmt.Fprintf(os.Stderr, "GATED on this box: %s\n", res.Reason)
		fmt.Fprintf(os.Stderr, "run on a box with the harness:\n  %s\n", res.Command)
	}
}

func cmdWebbenchCompare(argv []string) {
	fs := flag.NewFlagSet("webbench compare", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to web task dataset (required)")
	workersArg := fs.String("workers", "1,2,4,8", "worker sweep")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	preds := fs.String("predictions", "", "predictions JSON to fold the success rate (optional)")
	benchResult := fs.String("bench-result", "", "external benchmark results for side-by-side (optional)")
	out := fs.String("out", "", "write the Comparison JSON here (default: stdout)")
	md := fs.String("md", "", "write the markdown report here (optional)")
	_ = fs.Parse(argv)
	_ = preds // TODO: use predictions to fold success rate

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "fak webbench compare: --dataset is required")
		os.Exit(2)
	}

	d, err := webbench.LoadDataset(*dataset)
	must(err)

	if *limit > 0 && *limit < d.Len() {
		d = d.Limit(*limit)
	}

	workers := parseIntList(*workersArg)
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}

	in := webbench.CompareInputs{
		Dataset:     d,
		Geometry:    webbench.DefaultGeometryModel(),
		Workers:     workers,
		BenchResult: *benchResult,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
	}

	c := webbench.BuildComparison(in)

	if *out != "" {
		data, _ := json.MarshalIndent(c, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(c, "", "  ")
		fmt.Println(string(data))
	}

	if *md != "" {
		must(os.WriteFile(*md, []byte(webbench.RenderMarkdown(c)), 0644))
	}

	fmt.Fprintf(os.Stderr, "\n== fak webbench compare ==\n")
	fmt.Fprintf(os.Stderr, "source        : %s\n", *dataset)
	fmt.Fprintf(os.Stderr, "instances     : %d\n", c.Summary.Instances)
	for _, f := range c.Families {
		fmt.Fprintf(os.Stderr, "  %-30s %-11s %s\n", f.Name, "["+f.Kind+"]", f.Provenance)
	}
	if *md != "" {
		fmt.Fprintf(os.Stderr, "markdown: %s\n", *md)
	}
}

func cmdWebbenchServing(argv []string) {
	fs := flag.NewFlagSet("webbench serving", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to web task dataset (required)")
	tracksArg := fs.String("tracks", "ours,sglang,vllm,fak-fronts-fleet", "comma-separated tracks")
	endpointsArg := fs.String("endpoints", "", "comma-separated track=OpenAI-compatible /v1 base URLs")
	metricsArg := fs.String("metrics", "", "comma-separated track=Prometheus metrics URLs for prefix-cache hit rate")
	model := fs.String("model", "unknown", "model id sent to each OpenAI-compatible endpoint")
	machine := fs.String("machine", "", "machine id for artifact path (default: hostname)")
	agents := fs.Int("agents", 100, "number of agent requests to replay; repeats dataset tasks if needed")
	concurrency := fs.Int("concurrency", 16, "parallel request workers")
	limit := fs.Int("limit", 0, "cap source dataset tasks before agent replay (0 = all)")
	maxOutput := fs.Int("max-output-tokens", 64, "max output tokens per request")
	sloMS := fs.Int("slo-ms", 2000, "goodput SLO threshold in milliseconds")
	timeoutSec := fs.Int("timeout-sec", 60, "per-request timeout in seconds")
	apiKeyEnv := fs.String("api-key-env", "", "optional env var containing a bearer token for all endpoints")
	replicas := fs.Int("replicas", 1, "replica count described by the fak-fronts-fleet plan script")
	sharedPrefix := fs.String("shared-prefix", "", "override the shared prefix used across all requests")
	out := fs.String("out", "", "write artifact JSON here (default: by-machine dated run dir)")
	outDir := fs.String("out-dir", "", "artifact root (default: experiments/benchmark/runs/by-machine)")
	_ = fs.Parse(argv)

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "fak webbench serving: --dataset is required")
		os.Exit(2)
	}
	tracks, err := webbench.ParseServingTracks(*tracksArg)
	must(err)
	endpoints, err := parseServingTrackMap(*endpointsArg)
	must(err)
	metrics, err := parseServingTrackMap(*metricsArg)
	must(err)

	d, err := webbench.LoadDataset(*dataset)
	must(err)
	workload := webbench.BuildServingWorkload(d, webbench.DefaultGeometryModel(), *limit, *agents, *maxOutput, *sharedPrefix)
	if len(workload) == 0 {
		fmt.Fprintln(os.Stderr, "fak webbench serving: workload is empty")
		os.Exit(2)
	}

	machineID := *machine
	if machineID == "" {
		if host, err := os.Hostname(); err == nil && host != "" {
			machineID = host
		} else {
			machineID = "unknown"
		}
	}
	var cfgTracks []webbench.ServingTrackConfig
	for _, tr := range tracks {
		cfgTracks = append(cfgTracks, webbench.ServingTrackConfig{
			Track:      tr,
			BaseURL:    endpoints[tr],
			MetricsURL: metrics[tr],
			Model:      *model,
			APIKeyEnv:  *apiKeyEnv,
			Replicas:   *replicas,
		})
	}

	rep, err := webbench.RunServingParity(ctx(), webbench.ServingParityConfig{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		MachineID:   machineID,
		Model:       *model,
		Tracks:      cfgTracks,
		Workload:    workload,
		Concurrency: *concurrency,
		SLO:         time.Duration(*sloMS) * time.Millisecond,
		Timeout:     time.Duration(*timeoutSec) * time.Second,
	})
	must(err)
	outPath := *out
	if outPath == "" {
		outPath = webbench.DefaultServingArtifactPath(*outDir, machineID, rep.GeneratedAt, tracks)
	}
	must(webbench.WriteServingParityReport(rep, outPath))
	printServingSummary(os.Stderr, rep, outPath)
}

func cmdWebbenchParityGate(argv []string) {
	fs := flag.NewFlagSet("webbench parity-gate", flag.ExitOnError)
	claimFile := fs.String("claim-file", "", "file containing claim text")
	claim := fs.String("claim", "", "inline claim text")
	artifact := fs.String("artifact", "", "serving parity artifact JSON")
	_ = fs.Parse(argv)

	claimText := *claim
	if *claimFile != "" {
		b, err := os.ReadFile(*claimFile)
		must(err)
		claimText = string(b)
	}
	if claimText == "" {
		fmt.Fprintln(os.Stderr, "fak webbench parity-gate: --claim-file or --claim is required")
		os.Exit(2)
	}
	var rep *webbench.ServingParityReport
	if *artifact != "" {
		var err error
		rep, err = webbench.LoadServingParityReport(*artifact)
		must(err)
	}
	if err := webbench.ValidateParityClaim(claimText, rep); err != nil {
		fmt.Fprintf(os.Stderr, "serving parity gate: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "serving parity gate: OK")
}

func parseServingTrackMap(arg string) (map[webbench.ServingTrack]string, error) {
	out := make(map[webbench.ServingTrack]string)
	if strings.TrimSpace(arg) == "" {
		return out, nil
	}
	for _, part := range strings.Split(arg, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		k, v, ok := strings.Cut(part, "=")
		if !ok || strings.TrimSpace(k) == "" || strings.TrimSpace(v) == "" {
			return nil, fmt.Errorf("track map entry %q must be track=value", part)
		}
		tr, err := webbench.ParseServingTrack(k)
		if err != nil {
			return nil, err
		}
		out[tr] = strings.TrimSpace(v)
	}
	return out, nil
}

func printServingSummary(w *os.File, rep *webbench.ServingParityReport, out string) {
	fmt.Fprintf(w, "\n== fak webbench serving ==\n")
	fmt.Fprintf(w, "artifact      : %s\n", out)
	fmt.Fprintf(w, "requests      : %d  concurrency: %d  slo: %dms\n", rep.Workload.Requests, rep.Workload.Concurrency, rep.Workload.SLOMillis)
	for _, tr := range rep.Tracks {
		fmt.Fprintf(w, "  %-16s %-12s", tr.Track, tr.Status)
		if tr.Status != "measured" {
			fmt.Fprintf(w, " %s\n", tr.Reason)
			continue
		}
		fmt.Fprintf(w, " ok=%d/%d", tr.Stats.OK, tr.Stats.Requests)
		if tr.Stats.TTFTMillis.P50 != nil {
			fmt.Fprintf(w, " ttft.p50=%.1fms", *tr.Stats.TTFTMillis.P50)
		}
		if tr.Stats.GoodputRPS.Value != nil {
			fmt.Fprintf(w, " goodput=%.3f req/s", *tr.Stats.GoodputRPS.Value)
		}
		if tr.Stats.PrefixCacheHitRate.Value != nil {
			fmt.Fprintf(w, " prefix-hit=%.3f", *tr.Stats.PrefixCacheHitRate.Value)
		} else {
			fmt.Fprintf(w, " prefix-hit=%s", tr.Stats.PrefixCacheHitRate.Status)
		}
		fmt.Fprintln(w)
	}
}

func formatTokens(n int64) string {
	if n < 1_000_000 {
		return fmt.Sprintf("%d", n)
	}
	m := float64(n) / 1_000_000
	if m < 1000 {
		return fmt.Sprintf("%.1f M", m)
	}
	g := m / 1000
	return fmt.Sprintf("%.2f G", g)
}

func sortedCounts(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, "  ")
}
