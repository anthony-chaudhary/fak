package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachewitness"
)

// cmdSwebenchCacheWitness is `fak swebench cache-witness`: scrape a live fak
// serve gateway's /metrics and fold the in-kernel KV-prefix cache family into ONE
// provenance-labeled evidence record — the cache VALUE a fak-served model (e.g.
// GLM-5.2 on the pure kernel) realized across an agentic run.
//
// This is the observation seam for epic #1010 / child #1011: after the Claude
// harness drives `fak swebench run --agent fleet` (or `fak guard --base-url`)
// against the gateway, the repeated system+tools+repo prefix can be served from
// cached KV. This command reads the aggregate reused-token count off /metrics
// and emits it as WITNESSED (fak's own RadixAttention), beside the provider
// cache_read as OBSERVED — the trust split the DOS / conflation discipline owes.
//
// Two input modes, because the GLM-5.2 box is often reachable only over the Slack
// control bridge (no direct HTTP from the dev box): --gateway fetches /metrics
// live; --metrics-file parses a body captured on the box (curl localhost/metrics
// > f) and relayed back. Either way the record is identical.
func cmdSwebenchCacheWitness(argv []string) {
	os.Exit(runSwebenchCacheWitness(os.Stdout, os.Stderr, argv))
}

func runSwebenchCacheWitness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("swebench cache-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", "", "fak serve gateway base URL or HOST:PORT (its /metrics is scraped); empty with --metrics-file reads a captured body")
	metricsFile := fs.String("metrics-file", "", "read a captured /metrics body from this file instead of fetching (for a box reachable only over the bridge)")
	baselineFile := fs.String("baseline", "", "optional run-start /metrics body to subtract from the end scrape, producing a per-run delta")
	out := fs.String("out", "", "write the Record JSON here (default: stdout)")
	timeout := fs.Float64("timeout-seconds", 15, "HTTP fetch timeout in seconds")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *gateway == "" && *metricsFile == "" {
		fmt.Fprintln(stderr, "fak swebench cache-witness: need --gateway URL or --metrics-file PATH")
		return 2
	}

	var body, srcLabel string
	if *metricsFile != "" {
		b, err := os.ReadFile(*metricsFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak swebench cache-witness: read %s: %v\n", *metricsFile, err)
			return 2
		}
		body = string(b)
		srcLabel = "file://" + *metricsFile
	} else {
		url := metricsURL(*gateway)
		b, err := fetchMetrics(url, time.Duration(*timeout*float64(time.Second)))
		if err != nil {
			fmt.Fprintf(stderr, "fak swebench cache-witness: fetch %s: %v\n", url, err)
			return 2
		}
		body = b
		srcLabel = url
	}

	rec, err := cachewitness.Parse(srcLabel, body)
	if err != nil {
		fmt.Fprintf(stderr, "fak swebench cache-witness: %v\n", err)
		return 2
	}
	if *baselineFile != "" {
		b, err := os.ReadFile(*baselineFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak swebench cache-witness: read baseline %s: %v\n", *baselineFile, err)
			return 2
		}
		base, err := cachewitness.Parse("file://"+*baselineFile, string(b))
		if err != nil {
			fmt.Fprintf(stderr, "fak swebench cache-witness: baseline %s: %v\n", *baselineFile, err)
			return 2
		}
		rec = rec.Sub(base)
	}

	enc, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak swebench cache-witness: marshal: %v\n", err)
		return 1
	}
	if *out != "" {
		if err := os.WriteFile(*out, append(enc, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak swebench cache-witness: write %s: %v\n", *out, err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, string(enc))
	}

	// Human summary on stderr — leads with whether fak's own cache actually bit,
	// and keeps the WITNESSED/OBSERVED line explicit so the operator reads the
	// provenance, not just the number.
	k := rec.KVPrefix
	scope := rec.CacheBitScope
	if scope == "" {
		scope = cachewitness.CacheBitScopeAggregateRun
	}
	bit := "did NOT bite (all-cold; reuse 0)"
	if rec.CacheBit() {
		bit = fmt.Sprintf("BIT — reused %d/%d prefill tokens (%.1f%%) across %d turns (frozen=%d partial=%d cold=%d)",
			k.ReusedTokens, k.PromptTokens, 100*k.ReuseRatio(), k.Turns, k.FrozenTurns, k.PartialTurns, k.ColdTurns)
	}
	fmt.Fprintf(stderr, "fak in-kernel KV-prefix cache (WITNESSED, %s): %s\n", scope, bit)
	fmt.Fprintf(stderr, "provider cache_read (OBSERVED, relayed): %d tokens\n", rec.ProviderCacheReadTokens)
	if rec.WitnessWindow != nil {
		fmt.Fprintf(stderr, "witness window: %s -> %s (gateway uptime turns %d)\n",
			rec.WitnessWindow.StartScrape, rec.WitnessWindow.EndScrape, rec.GatewayUptimeTurns)
	}
	// The #1066 cache-value fence: publish the marginal-over-tuned-warm-KV family,
	// never the vs-naive multiple a long trajectory's high reuse tempts.
	cv := rec.CacheValue
	fmt.Fprintf(stderr, "publishable cache-value: %s = %.2fx single-session (fleet >1.0x via %s); vs-naive multiple excluded per #1066\n",
		cv.PublishableValueFamily, cv.SingleSessionMarginalX, cv.FleetMarginalSource)
	return 0
}

// metricsURL normalizes a --gateway value (HOST:PORT, a base URL, or a full
// /metrics URL) into the /metrics endpoint to scrape.
func metricsURL(gw string) string {
	u := gw
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		u = "http://" + u
	}
	u = strings.TrimRight(u, "/")
	if strings.HasSuffix(u, "/metrics") {
		return u
	}
	return u + "/metrics"
}

// fetchMetrics GETs the /metrics body with a bounded timeout.
func fetchMetrics(url string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", err
	}
	return string(b), nil
}
