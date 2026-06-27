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
// against the gateway, the repeated system+tools+repo prefix is served from the
// cached KV on turns 2..N. This command reads that off /metrics and emits the
// reused-token count as WITNESSED (fak's own RadixAttention), beside the provider
// cache_read as OBSERVED — the trust split the DOS / conflation discipline owes.
//
// Two input modes, because the GLM-5.2 box is often reachable only over the Slack
// control bridge (no direct HTTP from the dev box): --gateway fetches /metrics
// live; --metrics-file parses a body captured on the box (curl localhost/metrics
// > f) and relayed back. Either way the record is identical.
func cmdSwebenchCacheWitness(argv []string) {
	fs := flag.NewFlagSet("swebench cache-witness", flag.ExitOnError)
	gateway := fs.String("gateway", "", "fak serve gateway base URL or HOST:PORT (its /metrics is scraped); empty with --metrics-file reads a captured body")
	metricsFile := fs.String("metrics-file", "", "read a captured /metrics body from this file instead of fetching (for a box reachable only over the bridge)")
	out := fs.String("out", "", "write the Record JSON here (default: stdout)")
	timeout := fs.Float64("timeout-seconds", 15, "HTTP fetch timeout in seconds")
	_ = fs.Parse(argv)

	if *gateway == "" && *metricsFile == "" {
		fmt.Fprintln(os.Stderr, "fak swebench cache-witness: need --gateway URL or --metrics-file PATH")
		os.Exit(2)
	}

	var body, srcLabel string
	if *metricsFile != "" {
		b, err := os.ReadFile(*metricsFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak swebench cache-witness: read %s: %v\n", *metricsFile, err)
			os.Exit(2)
		}
		body = string(b)
		srcLabel = "file://" + *metricsFile
	} else {
		url := metricsURL(*gateway)
		b, err := fetchMetrics(url, time.Duration(*timeout*float64(time.Second)))
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak swebench cache-witness: fetch %s: %v\n", url, err)
			os.Exit(2)
		}
		body = b
		srcLabel = url
	}

	rec, err := cachewitness.Parse(srcLabel, body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak swebench cache-witness: %v\n", err)
		os.Exit(2)
	}

	enc, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak swebench cache-witness: marshal: %v\n", err)
		os.Exit(1)
	}
	if *out != "" {
		if err := os.WriteFile(*out, append(enc, '\n'), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "fak swebench cache-witness: write %s: %v\n", *out, err)
			os.Exit(1)
		}
	} else {
		fmt.Println(string(enc))
	}

	// Human summary on stderr — leads with whether fak's own cache actually bit,
	// and keeps the WITNESSED/OBSERVED line explicit so the operator reads the
	// provenance, not just the number.
	k := rec.KVPrefix
	bit := "did NOT bite (all-cold; reuse 0)"
	if rec.CacheBit() {
		bit = fmt.Sprintf("BIT — reused %d/%d prefill tokens (%.1f%%) across %d turns (frozen=%d partial=%d cold=%d)",
			k.ReusedTokens, k.PromptTokens, 100*k.ReuseRatio(), k.Turns, k.FrozenTurns, k.PartialTurns, k.ColdTurns)
	}
	fmt.Fprintf(os.Stderr, "fak in-kernel KV-prefix cache (WITNESSED): %s\n", bit)
	fmt.Fprintf(os.Stderr, "provider cache_read (OBSERVED, relayed): %d tokens\n", rec.ProviderCacheReadTokens)
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
	resp, err := http.DefaultClient.Do(req)
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
