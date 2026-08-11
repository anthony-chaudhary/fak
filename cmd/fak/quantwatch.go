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

	"github.com/anthony-chaudhary/fak/internal/quantwatch"
)

func cmdQuantwatch(argv []string) {
	os.Exit(runQuantwatch(os.Stdout, os.Stderr, argv, http.DefaultClient, time.Now))
}

func runQuantwatch(stdout, stderr io.Writer, argv []string, client *http.Client, now func() time.Time) int {
	fs := flag.NewFlagSet("quantwatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	snapshot := fs.String("snapshot", "", "offline fak.quantwatch.snapshot/v1 JSON fixture")
	live := fs.Bool("live", false, "query bounded public arXiv/GitHub release metadata")
	query := fs.String("query", "quantization", "research metadata query")
	repos := fs.String("github-repos", "", "comma-separated explicit owner/repo release sources")
	limit := fs.Int("limit", 10, "maximum records per source (1..100)")
	timeout := fs.Duration("timeout", 20*time.Second, "live request deadline")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if (*snapshot == "") == !*live {
		fmt.Fprintln(stderr, "fak quantwatch: choose exactly one of --snapshot FILE or --live")
		return 2
	}

	var result quantwatch.Result
	if *snapshot != "" {
		raw, err := os.ReadFile(*snapshot)
		if err != nil {
			fmt.Fprintf(stderr, "fak quantwatch: read snapshot: %v\n", err)
			return 2
		}
		result = quantwatch.IngestSnapshot(raw)
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		defer cancel()
		var repositories []string
		for _, repo := range strings.Split(*repos, ",") {
			if repo = strings.TrimSpace(repo); repo != "" {
				repositories = append(repositories, repo)
			}
		}
		result = quantwatch.FetchLive(ctx, client, quantwatch.LiveRequest{Query: *query, QueryTime: now().UTC(), MaxPerSource: *limit, GitHubRepositories: repositories})
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak quantwatch: encode result: %v\n", err)
		return 1
	}
	switch result.Outcome {
	case quantwatch.OutcomeRanked:
		return 0
	case quantwatch.OutcomeAbstain, quantwatch.OutcomeUnsupported:
		return 3
	default:
		return 2
	}
}
