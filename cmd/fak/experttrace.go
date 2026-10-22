package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deepseekv4moe"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const expertTraceUsage = "usage: fak expert-trace analyze <trace.jsonl> [--json] [--phase prefill|decode] [--keep-repeats] [--coverage 0.8] [--windows 1,8,32,128] [--top-n 8,16,32,64] [--capacities 1,2,...] [--ranks 2,4] [--usage-out <hist.json>] [--strix-out <trace.json> --expert-bytes N]\n  policy: fak expert-trace policy <trace.jsonl> --capacity N [--json] (selected route weights; global uniform slots)\n  overlap: fak expert-trace overlap <a.jsonl> <b.jsonl> [more.jsonl...] [--phase decode] [--ranks 2,4] [--json] (streams recorded separately, aligned by routed step)\n  record: " + model.ExpertRouteTraceEnv + "=<trace.jsonl> fak serve ..."

func runExpertTrace(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 && args[0] == "policy" {
		return runExpertTraceReplay(stdout, stderr, args[1:])
	}
	if len(args) > 0 && args[0] == "overlap" {
		return runExpertTraceOverlap(stdout, stderr, args[1:])
	}
	if len(args) == 0 || args[0] != "analyze" {
		fmt.Fprintln(stderr, expertTraceUsage)
		return 2
	}
	fs := flag.NewFlagSet("expert-trace analyze", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the full analysis as JSON")
	phase := fs.String("phase", "", "keep only routes from this phase (prefill|decode)")
	keep := fs.Bool("keep-repeats", false, "keep re-routes of an already-seen (token, layer)")
	coverage := fs.Float64("coverage", 0.8, "per-layer pick share the hot set must cover")
	windows := fs.String("windows", "", "comma-separated token windows (default 1,8,32,128)")
	topN := fs.String("top-n", "", "comma-separated top-N cut-offs (default 8,16,32,64)")
	caps := fs.String("capacities", "", "comma-separated per-layer cache capacities in experts")
	ranks := fs.String("ranks", "", "comma-separated expert-parallel rank counts for per-step balance (moe_lb)")
	usageOut := fs.String("usage-out", "", "write the hot set as a "+model.ExpertUsageHistogramSchema+" warm-start file")
	strixOut := fs.String("strix-out", "", "write a "+model.StrixMoECacheTraceSchema+" trace for fak-strix moe-cache-bench --trace")
	expertBytes := fs.Int64("expert-bytes", 0, "bytes per expert for --strix-out")
	path, rest := "", args[1:]
	if len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
		path, rest = rest[0], rest[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	if path == "" && fs.NArg() > 0 {
		path = fs.Arg(0)
	}
	if path == "" {
		fmt.Fprintln(stderr, expertTraceUsage)
		return 2
	}
	opts := model.ExpertRouteAnalyzeOptions{Phase: *phase, KeepRepeat: *keep, Coverage: *coverage}
	var err error
	if opts.Windows, err = parseExpertTraceInts(*windows); err == nil {
		if opts.TopN, err = parseExpertTraceInts(*topN); err == nil {
			if opts.Capacities, err = parseExpertTraceInts(*caps); err == nil {
				opts.Ranks, err = parseExpertTraceInts(*ranks)
			}
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	trace, err := model.ReadExpertRouteTrace(f)
	f.Close()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	rep, err := model.AnalyzeExpertRouteTrace(trace, opts)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *usageOut != "" {
		h := model.NewExpertUsageHistogram()
		for _, c := range rep.HotSet {
			h.Observe(c.Layer, c.Expert, c.Count)
		}
		if err := h.Persist(*usageOut); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *strixOut != "" {
		st, err := model.ExpertRouteTraceToStrix(trace, opts, *expertBytes)
		if err == nil {
			var b []byte
			if b, err = json.Marshal(st); err == nil {
				err = os.WriteFile(*strixOut, append(b, '\n'), 0o644)
			}
		}
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "routes=%d picks=%d layers=%d top_k=%d repeats_dropped=%d phases=%v\n", rep.Routes, rep.Picks, rep.Layers, rep.TopK, rep.RepeatsDropped, rep.PhaseRoutes)
	fmt.Fprintf(stdout, "unique experts: total=%d mean/layer=%.1f max/layer=%d\n", rep.UniqueExperts, rep.MeanUniquePerLayer, rep.MaxUniquePerLayer)
	for _, w := range rep.Windows {
		fmt.Fprintf(stdout, "window %4d tok: mean unique/layer=%.1f max=%d\n", w.Window, w.MeanUnique, w.MaxUnique)
	}
	for _, s := range rep.TopNShare {
		fmt.Fprintf(stdout, "top-%d/layer share=%.3f\n", s.N, s.Share)
	}
	for _, c := range rep.Cache {
		fmt.Fprintf(stdout, "cache %4d/layer (%d total): lru=%.3f belady=%.3f\n", c.CapacityPerLayer, c.TotalExperts, c.LRUHitRate, c.BeladyHitRate)
	}
	fmt.Fprintf(stdout, "hot set @%.0f%% coverage: %d experts (%.1f/layer); cold picks=%d\n", rep.Coverage*100, rep.HotSetSize, float64(rep.HotSetSize)/float64(rep.Layers), rep.ColdPicks)
	printExpertRankBalance(stdout, rep.RankBalance)
	return 0
}

func printExpertRankBalance(stdout io.Writer, rows []model.ExpertRouteRankBalance) {
	for _, b := range rows {
		fmt.Fprintf(stdout, "ranks %d held-out moe_lb (%d steps): count-even=%.3f load-banded=%.3f floor=%.3f; p99 step count-even=%.3f load-banded=%.3f; in-sample load-banded=%.3f\n",
			b.Ranks, b.LoadBanded.Steps, b.CountEven.MoeLB, b.LoadBanded.MoeLB, b.Floor.MoeLB, b.CountEven.P99StepRatio, b.LoadBanded.P99StepRatio, b.LoadBandedInSample.MoeLB)
	}
}

func runExpertTraceOverlap(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("expert-trace overlap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the overlap report as JSON")
	phase := fs.String("phase", "", "keep only routes from this phase (prefill|decode)")
	ranks := fs.String("ranks", "", "comma-separated expert-parallel rank counts for merged-step balance")
	var paths []string
	rest := args
	for len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
		paths, rest = append(paths, rest[0]), rest[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	paths = append(paths, fs.Args()...)
	if len(paths) < 2 {
		fmt.Fprintln(stderr, expertTraceUsage)
		return 2
	}
	opts := model.ExpertRouteAnalyzeOptions{Phase: *phase}
	var err error
	if opts.Ranks, err = parseExpertTraceInts(*ranks); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	traces := make([]model.ExpertRouteTrace, 0, len(paths))
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		tr, err := model.ReadExpertRouteTrace(f)
		f.Close()
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", path, err)
			return 1
		}
		traces = append(traces, tr)
	}
	rep, err := model.AnalyzeExpertRouteStreams(traces, opts)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "streams=%d layers=%d aligned steps=%d picks=%d unique reads=%d shared fraction=%.3f mean union/step=%.2f\n",
		rep.Streams, rep.Layers, rep.Steps, rep.Picks, rep.UniqueReads, rep.SharedFraction, rep.MeanUnion)
	printExpertRankBalance(stdout, rep.RankBalance)
	return 0
}

func parseExpertTraceInts(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid positive integer %q in list %q", part, s)
		}
		out = append(out, n)
	}
	return out, nil
}

func runExpertTraceReplay(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("expert-trace policy", flag.ContinueOnError)
	fs.SetOutput(stderr)
	capacity := fs.Int("capacity", 0, "global cache capacity in uniform expert slots")
	asJSON := fs.Bool("json", false, "emit simulated policy receipt")
	armNames := fs.String("policies", "lru,lfru,scoreema,opt", "comma-separated eviction policy arms")
	path := ""
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		path, args = args[0], args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if path == "" && fs.NArg() == 1 {
		path = fs.Arg(0)
	} else if fs.NArg() != 0 {
		fmt.Fprintln(stderr, expertTraceUsage)
		return 2
	}
	if path == "" || *capacity <= 0 {
		fmt.Fprintln(stderr, expertTraceUsage)
		return 2
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	trace, err := model.ReadExpertRouteTrace(f)
	f.Close()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	stream := deepseekv4moe.CachePolicyStream{Name: path, Scores: []float64{}}
	for _, event := range trace.Events {
		if len(event.Weights) != len(event.Experts) {
			fmt.Fprintln(stderr, "policy replay requires one selected routing weight per expert")
			return 1
		}
		seen := make(map[int]bool, len(event.Experts))
		for i, expert := range event.Experts {
			if seen[expert] {
				fmt.Fprintln(stderr, "duplicate expert in router event")
				return 1
			}
			seen[expert] = true
			stream.Groups = append(stream.Groups, deepseekv4moe.ExpertGroup{Layer: event.Layer, Expert: expert})
			if event.Weights[i] < 0 {
				fmt.Fprintln(stderr, "policy replay supports nonnegative selected weights; signed negative weights are unsupported")
				return 1
			}
			stream.Scores = append(stream.Scores, float64(event.Weights[i]))
		}
	}
	var policies []deepseekv4moe.CachePolicy
	for _, name := range strings.Split(*armNames, ",") {
		policy, ok := deepseekv4moe.ParseReplayArm(strings.TrimSpace(name))
		if !ok {
			fmt.Fprintf(stderr, "unknown cache policy %q\n", name)
			return 2
		}
		policies = append(policies, policy)
	}
	receipt, err := deepseekv4moe.CachePolicyReplayWith(stream, *capacity, policies)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *asJSON {
		report := struct {
			deepseekv4moe.CachePolicyBenchReceipt
			ScoreInput     string `json:"score_input"`
			CapacityScope  string `json:"capacity_scope"`
			RepeatHandling string `json:"repeat_handling"`
		}{receipt, "selected-route-weights", "global-uniform-expert-slots", "all-recorded-events"}
		if err := json.NewEncoder(stdout).Encode(report); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "simulated selected-route-weights capacity=%d global uniform expert slots; all recorded events\n", *capacity)
		for _, row := range receipt.Rows {
			fmt.Fprintf(stdout, "%s hits=%d misses=%d hit_rate=%.4f\n", row.Policy, row.Hits, row.Misses, row.HitRate)
		}
	}
	return 0
}
