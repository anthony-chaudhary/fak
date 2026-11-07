package main

// perf.go — `fak perf`: the per-request serving-performance read (TTFT,
// prefill/decode rates, e2e, cache share) over the last N served turns. By
// default it reads the durable perf ledger `fak serve` appends to, so it answers
// with the server down or after a restart; --url reads a live server's
// /v1/fak/perf/recent instead. The fold and renderer live in internal/perfledger,
// shared with the gateway endpoint so both reads print the same line.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func cmdPerf(argv []string) { os.Exit(runPerf(os.Stdout, os.Stderr, argv)) }

func runPerf(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("perf", flag.ContinueOnError)
	fs.SetOutput(stderr)
	n := fs.Int("n", perfledger.DefaultRecent, "fold the last N served turns (max 1024)")
	asJSON := fs.Bool("json", false, "emit fak.gateway.perf-report.v1 JSON (summary + records), or fak.bench-token-comparison.v1 with comparison flags")
	ledger := fs.String("ledger", "", "perf ledger path (default: .fak/nightrun/gateway-perf.jsonl under the nightrun ledger root)")
	baseURL := fs.String("url", "", "read a live server's /v1/fak/perf/recent instead of the ledger, e.g. http://127.0.0.1:8080")
	compareLeft := fs.String("compare-left", "", "compare complete token digests for this request ID and --compare-right within the selected window")
	compareRight := fs.String("compare-right", "", "second request ID for explicit token digest comparison")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *n <= 0 {
		fmt.Fprintln(stderr, "fak perf: --n must be positive")
		return 2
	}
	compare := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "compare-left" || f.Name == "compare-right" {
			compare = true
		}
	})
	if compare && (strings.TrimSpace(*compareLeft) == "" || strings.TrimSpace(*compareRight) == "" || *compareLeft == *compareRight || fs.NArg() != 0) {
		fmt.Fprintln(stderr, "fak perf: comparison requires two distinct nonempty request IDs and no positional arguments")
		return 2
	}

	var rep perfledger.Report
	if strings.TrimSpace(*baseURL) != "" {
		got, err := fetchPerfReport(*baseURL, *n)
		if err != nil {
			fmt.Fprintf(stderr, "fak perf: %v\n", err)
			return 1
		}
		rep = got
	} else {
		path := *ledger
		if path == "" {
			path = perfLedgerDefaultPath()
		}
		recs, truncated := perfledger.ReadTail(path)
		rep = perfledger.BuildReport(recs, *n, truncated, 0)
		rep.Source = path
		if len(recs) == 0 && !*asJSON {
			fmt.Fprintf(stderr, "fak perf: no perf records yet in %s (serve some turns with `fak serve` first)\n", path)
		}
	}

	if compare {
		if strings.TrimSpace(*baseURL) != "" {
			sourceURL, err := url.Parse(strings.TrimSpace(*baseURL))
			if err != nil {
				fmt.Fprintf(stderr, "fak perf: %v\n", err)
				return 1
			}
			sourceURL.User, sourceURL.RawQuery, sourceURL.Fragment = nil, "", ""
			sourceURL.Path = strings.TrimRight(sourceURL.Path, "/") + "/v1/fak/perf/recent"
			rep.Source = sourceURL.String()
		}
		source := rep.Source
		rep = perfledger.BuildReport(rep.Records, *n, rep.Window.Capped, rep.Window.DroppedWrites)
		rep.Source = source
		result := perfledger.CompareBenchTokens(rep, *compareLeft, *compareRight)
		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(result); err != nil {
				fmt.Fprintf(stderr, "fak perf: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprintf(stdout, "token_identity=%s reason=%s left=%q right=%q source=%q window=%d capped=%t\n", result.Outcome, result.Reason, result.LeftRequestID, result.RightRequestID, result.Source, result.Window.Requests, result.Window.Capped)
		}
		if result.Outcome != perfledger.BenchTokensEqual {
			return 1
		}
		return 0
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak perf: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, perfledger.RenderCompact(rep))
	return 0
}

func fetchPerfReport(base string, n int) (perfledger.Report, error) {
	u, err := url.Parse(strings.TrimRight(strings.TrimSpace(base), "/") + "/v1/fak/perf/recent")
	if err != nil {
		return perfledger.Report{}, err
	}
	q := u.Query()
	q.Set("n", strconv.Itoa(n))
	u.RawQuery = q.Encode()
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Get(u.String())
	if err != nil {
		return perfledger.Report{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return perfledger.Report{}, fmt.Errorf("GET %s: %s: %s", u.Redacted(), resp.Status, strings.TrimSpace(string(body)))
	}
	var rep perfledger.Report
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&rep); err != nil {
		return perfledger.Report{}, fmt.Errorf("decode %s: %w", u.Redacted(), err)
	}
	return rep, nil
}

// perfLedgerDefaultPath resolves the same file `fak serve` writes, without the
// directory creation nightrunLedgerPath does for writers.
func perfLedgerDefaultPath() string {
	return filepath.Join(nightrunLedgerRoot(repoRoot()), filepath.FromSlash(perfledger.DefaultLedgerRel))
}

// installServePerfLedger wires the durable perf ledger into a serving gateway:
// it seeds the in-memory ring from the ledger tail, then starts the async
// writer. "off" keeps the ring memory-only. The returned func flushes and
// closes the writer and is always safe to call.
func installServePerfLedger(srv *gateway.Server, flagPath string, stderr io.Writer) func() {
	path := strings.TrimSpace(flagPath)
	if strings.EqualFold(path, "off") {
		return func() {}
	}
	if path == "" {
		path = nightrunLedgerPath(perfledger.DefaultLedgerRel)
	} else if dir := filepath.Dir(path); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	seed, truncated := perfledger.ReadTail(path)
	w := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	srv.SetPerfLedger(w, seed, truncated)
	return func() {
		if err := w.Close(); err != nil {
			fmt.Fprintf(stderr, "fak: perf ledger close (non-fatal): %v (dropped=%d)\n", err, w.Dropped())
		}
		if err := srv.ClosePhaseLedger(); err != nil {
			fmt.Fprintf(stderr, "fak: phase ledger close (non-fatal): %v\n", err)
		}
	}
}
