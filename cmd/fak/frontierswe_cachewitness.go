package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachewitness"
	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// frontierswe_cachewitness.go is the integrator glue for `fak frontierswe
// cache-witness` (epic #1706, C8 #1714): the per-turn cache-reuse witness over a
// long FrontierSWE trial. It reuses internal/cachewitness.Parse (the same
// Prometheus /metrics folder `fak swebench cache-witness` uses) to turn each scrape
// into three cumulative counters, then folds the SEQUENCE with
// frontierswe.FoldCacheWitness into the per-turn reused-prefill series + the
// realized reuse rate C14 measures. cmd/fak is the only layer that knows both
// packages, so frontierswe stays a tier-1 leaf that imports nothing internal.
//
// Two input modes mirror `fak swebench cache-witness`:
//
//	--metrics-dir DIR  — offline: fold a directory of captured /metrics bodies
//	    (name-sorted = scrape order), the artifact a co-resident gateway (C7) writes
//	    periodically during a trial. RUNNABLE NOW with no gateway.
//	--gateway URL --interval SEC --samples N — live: scrape a running fak serve
//	    /metrics N times at the interval and fold in real time (needs the co-resident
//	    gateway from C7).
func runFrontiersweCacheWitness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe cache-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	metricsDir := fs.String("metrics-dir", "", "fold a directory of captured /metrics bodies (name-sorted = trajectory order); offline, no gateway")
	metricsFiles := fs.String("metrics-files", "", "comma-separated captured /metrics files in trajectory order (alternative to --metrics-dir)")
	gateway := fs.String("gateway", "", "live: scrape this fak serve gateway's /metrics (URL or HOST:PORT)")
	interval := fs.Float64("interval", 30, "live: seconds between scrapes")
	samples := fs.Int("samples", 0, "live: number of scrapes to take (>0 required with --gateway)")
	timeout := fs.Float64("timeout-seconds", 15, "live: HTTP fetch timeout per scrape")
	out := fs.String("out", "", "write the compact JSONL trace ("+frontierswe.CacheWitnessSchema+", one line per turn) here")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	modes := 0
	for _, on := range []bool{*metricsDir != "", *metricsFiles != "", *gateway != ""} {
		if on {
			modes++
		}
	}
	if modes == 0 {
		fmt.Fprintln(stderr, "fak frontierswe cache-witness: need --metrics-dir, --metrics-files, or --gateway")
		return 2
	}
	if modes > 1 {
		fmt.Fprintln(stderr, "fak frontierswe cache-witness: --metrics-dir / --metrics-files / --gateway are mutually exclusive")
		return 2
	}

	var samplesIn []frontierswe.CacheSample
	var err error
	switch {
	case *metricsDir != "":
		samplesIn, err = foldMetricsDir(*metricsDir)
	case *metricsFiles != "":
		samplesIn, err = foldMetricsFiles(splitCSV(*metricsFiles))
	default:
		if *samples <= 0 {
			fmt.Fprintln(stderr, "fak frontierswe cache-witness: --gateway needs --samples > 0")
			return 2
		}
		samplesIn, err = scrapeGatewaySeries(stderr, *gateway, *samples, time.Duration(*interval*float64(time.Second)), time.Duration(*timeout*float64(time.Second)))
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe cache-witness: %v\n", err)
		return 1
	}
	if len(samplesIn) == 0 {
		fmt.Fprintln(stderr, "fak frontierswe cache-witness: no /metrics scrapes to fold")
		return 1
	}

	series := frontierswe.FoldCacheWitness(samplesIn)

	jb, err := json.MarshalIndent(series, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe cache-witness: marshal: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(jb))

	if *out != "" {
		if err := writeCacheWitnessJSONL(*out, series); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe cache-witness: write %s: %v\n", *out, err)
			return 1
		}
	}

	printCacheWitnessSummary(stderr, series, *out)
	return 0
}

// (splitCSV lives in debug.go — reused here for the --metrics-files list.)

// foldMetricsDir reads every regular file in dir (name-sorted, so a zero-padded
// scrape naming like 0001.metrics preserves trajectory order) and parses each into
// a cumulative CacheSample.
func foldMetricsDir(dir string) ([]frontierswe.CacheSample, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read metrics dir: %w", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	paths := make([]string, len(names))
	for i, n := range names {
		paths[i] = filepath.Join(dir, n)
	}
	return foldMetricsFiles(paths)
}

// foldMetricsFiles parses each captured /metrics body (in the given order) into a
// cumulative CacheSample via cachewitness.Parse — the identical folder used for
// swebench, so the numbers are directly comparable across benchmarks.
func foldMetricsFiles(paths []string) ([]frontierswe.CacheSample, error) {
	out := make([]frontierswe.CacheSample, 0, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		rec, err := cachewitness.Parse("file://"+p, string(b))
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", p, err)
		}
		out = append(out, sampleFromRecord(rec))
	}
	return out, nil
}

// scrapeGatewaySeries live-scrapes the gateway /metrics n times at interval,
// folding each into a CacheSample. The first scrape is immediate; the remaining
// n-1 wait interval between them (the C7 live path).
func scrapeGatewaySeries(stderr io.Writer, gateway string, n int, interval, timeout time.Duration) ([]frontierswe.CacheSample, error) {
	url := metricsURL(gateway)
	out := make([]frontierswe.CacheSample, 0, n)
	for i := 0; i < n; i++ {
		if i > 0 {
			time.Sleep(interval)
		}
		body, err := fetchMetrics(url, timeout)
		if err != nil {
			return out, fmt.Errorf("scrape %d/%d %s: %w", i+1, n, url, err)
		}
		rec, err := cachewitness.Parse(url, body)
		if err != nil {
			return out, fmt.Errorf("parse scrape %d/%d: %w", i+1, n, err)
		}
		out = append(out, sampleFromRecord(rec))
		fmt.Fprintf(stderr, "scrape %d/%d: cum reused %d/%d prefill tokens (%.1f%%)\n",
			i+1, n, rec.KVPrefix.ReusedTokens, rec.KVPrefix.PromptTokens, 100*rec.KVPrefix.ReuseRatio())
	}
	return out, nil
}

// sampleFromRecord projects a cachewitness.Record (one scrape) onto the three
// cumulative counters the fold consumes, keeping the WITNESSED reuse separate from
// the OBSERVED provider cache_read — the fold never sums them.
func sampleFromRecord(rec cachewitness.Record) frontierswe.CacheSample {
	return frontierswe.CacheSample{
		Turn:                    int(rec.KVPrefix.Turns),
		PromptTokens:            rec.KVPrefix.PromptTokens,
		ReusedTokens:            rec.KVPrefix.ReusedTokens,
		ProviderCacheReadTokens: rec.ProviderCacheReadTokens,
	}
}

// writeCacheWitnessJSONL emits the compact per-turn trace: one JSON object per
// point (each stamped with the schema so a line is self-describing) followed by one
// summary object carrying the realized reuse rate + provenance.
func writeCacheWitnessJSONL(path string, s frontierswe.CacheWitnessSeries) error {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	for _, p := range s.Points {
		line := struct {
			Schema string `json:"schema"`
			Kind   string `json:"kind"`
			frontierswe.CacheWitnessPoint
		}{Schema: s.Schema, Kind: "point", CacheWitnessPoint: p}
		if err := enc.Encode(line); err != nil {
			return err
		}
	}
	summary := struct {
		Schema                  string            `json:"schema"`
		Kind                    string            `json:"kind"`
		RealizedReuseRate       float64           `json:"realized_reuse_rate"`
		FinalPromptTokens       uint64            `json:"final_prompt_tokens"`
		FinalReusedTokens       uint64            `json:"final_reused_tokens"`
		ProviderCacheReadTokens uint64            `json:"provider_cache_read_tokens"`
		CacheBit                bool              `json:"cache_bit"`
		Provenance              map[string]string `json:"provenance"`
	}{
		Schema: s.Schema, Kind: "summary", RealizedReuseRate: s.RealizedReuseRate,
		FinalPromptTokens: s.FinalPromptTokens, FinalReusedTokens: s.FinalReusedTokens,
		ProviderCacheReadTokens: s.ProviderCacheReadTokens, CacheBit: s.CacheBit, Provenance: s.Provenance,
	}
	if err := enc.Encode(summary); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// printCacheWitnessSummary writes the human-readable witness summary on stderr,
// leading with whether fak's own cache bit and keeping the WITNESSED/OBSERVED split
// explicit (the provenance is the point, not just the number).
func printCacheWitnessSummary(w io.Writer, s frontierswe.CacheWitnessSeries, out string) {
	fmt.Fprintf(w, "\n== fak frontierswe cache-witness (%s) ==\n", s.Schema)
	fmt.Fprintf(w, "scrapes       : %d\n", len(s.Points))
	if s.CacheBit {
		fmt.Fprintf(w, "fak KV-prefix cache (WITNESSED): BIT — realized reuse rate r=%.4f (reused %d/%d cumulative prefill tokens)\n",
			s.RealizedReuseRate, s.FinalReusedTokens, s.FinalPromptTokens)
	} else {
		fmt.Fprintf(w, "fak KV-prefix cache (WITNESSED): did NOT bite (all-cold; realized reuse rate 0)\n")
	}
	fmt.Fprintf(w, "provider cache_read (OBSERVED, relayed): %d tokens — echoed separately, never summed with fak's reuse\n", s.ProviderCacheReadTokens)
	fmt.Fprintf(w, "\n  realized reuse rate r is the MEASURED input C14 plugs into the C4 TTS projection (TTSRatio(r)).\n")
	if out != "" {
		fmt.Fprintf(w, "\nJSONL trace written: %s\n", out)
	}
}
