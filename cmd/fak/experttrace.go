package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const expertTraceUsage = "usage: fak expert-trace analyze <trace.jsonl> [--json] [--phase prefill|decode] [--keep-repeats] [--coverage 0.8] [--windows 1,8,32,128] [--top-n 8,16,32,64] [--capacities 1,2,...] [--usage-out <hist.json>] [--strix-out <trace.json> --expert-bytes N]\n  record: " + model.ExpertRouteTraceEnv + "=<trace.jsonl> fak serve ..."

func runExpertTrace(stdout, stderr io.Writer, args []string) int {
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
			opts.Capacities, err = parseExpertTraceInts(*caps)
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
