package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func loadQuoteCorpus(path string) ([]trajctl.QuoteObservation, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return trajctl.ReadQuoteCorpus(f)
}

func runTrajctlQuote(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("trajctl quote", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpus := fs.String("corpus", "", "chronological JSONL outcome corpus")
	ledger := fs.String("ledger", "", "append-only quote ledger")
	atText := fs.String("at", "", "quote timestamp (RFC3339; defaults now)")
	capVersion := fs.String("capability-version", "", "capability snapshot version")
	policyVersion := fs.String("policy-version", "", "tool-policy snapshot version")
	indexVersion := fs.String("index-version", "", "index snapshot version")
	coverage := fs.Float64("index-coverage", 0, "index coverage [0,1]")
	qualityMetric := fs.String("quality-metric", "answer_score", "quality metric")
	qualityMin := fs.Float64("quality-min", 0, "minimum accepted quality")
	witnessRef := fs.String("quality-witness", "", "independent witness method")
	modelClass := fs.String("model-class", "standard", "route model class")
	maxTurns := fs.Int("max-turns", 4, "route maximum turns")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *corpus == "" || *capVersion == "" || *policyVersion == "" || *indexVersion == "" || *qualityMin <= 0 || *witnessRef == "" {
		fmt.Fprintln(stderr, "fak trajctl quote: --corpus, snapshot versions, --quality-min, and --quality-witness are required")
		return 2
	}
	obs, err := loadQuoteCorpus(*corpus)
	if err != nil {
		fmt.Fprintf(stderr, "fak trajctl quote: %v\n", err)
		return 1
	}
	at := time.Now().UTC()
	if *atText != "" {
		at, err = time.Parse(time.RFC3339, *atText)
		if err != nil {
			fmt.Fprintf(stderr, "fak trajctl quote: --at: %v\n", err)
			return 2
		}
	}
	cap := trajctl.CapabilitySnapshot{Version: *capVersion, RepoRead: true, Search: true, ToolPolicyVersion: *policyVersion}
	idx := trajctl.IndexSnapshot{Version: *indexVersion, Coverage: *coverage}
	qual := trajctl.QualityContract{Metric: *qualityMetric, Minimum: *qualityMin, Witness: *witnessRef}
	route := trajctl.RouteTemplate{ModelClass: *modelClass, Tools: []string{"repo_read", "search"}, MaxTurns: *maxTurns}
	q, err := trajctl.NewRepoQuestionQuote(at, cap, idx, qual, route, obs)
	if err != nil {
		if errors.Is(err, trajctl.ErrUnsupportedColdStart) {
			fmt.Fprintf(stderr, "fak trajctl quote: REFUSED: %v\n", err)
			return 3
		}
		fmt.Fprintf(stderr, "fak trajctl quote: %v\n", err)
		return 1
	}
	if *ledger != "" {
		if err := trajctl.AppendQuoteRecord(*ledger, trajctl.QuoteLedgerRecord{Kind: "quote", Quote: &q}); err != nil {
			fmt.Fprintf(stderr, "fak trajctl quote: %v\n", err)
			return 1
		}
	}
	return trajctlEmitJSON(stdout, stderr, q)
}

func runTrajctlQuoteRevise(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("trajctl quote-revise", flag.ContinueOnError)
	fs.SetOutput(stderr)
	quoteFile := fs.String("quote", "", "initial quote JSON")
	ledger := fs.String("ledger", "", "append-only quote ledger")
	reason := fs.String("reason", "capability_failure", "typed revision reason")
	revision := fs.Int("revision", 1, "monotonic revision number")
	atText := fs.String("at", "", "revision timestamp (RFC3339; defaults now)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *quoteFile == "" {
		fmt.Fprintln(stderr, "fak trajctl quote-revise: --quote is required")
		return 2
	}
	b, err := os.ReadFile(*quoteFile)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var q trajctl.Quote
	if err = json.Unmarshal(b, &q); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	at := time.Now().UTC()
	if *atText != "" {
		at, err = time.Parse(time.RFC3339, *atText)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
	}
	rev := trajctl.ReviseForCapabilityFailure(q, *revision, at, *reason)
	if *ledger != "" {
		if err := trajctl.AppendQuoteRecord(*ledger, trajctl.QuoteLedgerRecord{Kind: "revision", Revision: &rev}); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	return trajctlEmitJSON(stdout, stderr, rev)
}

func runTrajctlQuoteBacktest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("trajctl quote-backtest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpus := fs.String("corpus", "", "chronological JSONL outcome corpus")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *corpus == "" {
		fmt.Fprintln(stderr, "fak trajctl quote-backtest: --corpus is required")
		return 2
	}
	obs, err := loadQuoteCorpus(*corpus)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return trajctlEmitJSON(stdout, stderr, trajctl.BacktestRepoQuestion(obs))
}

func runTrajctlQuoteComplete(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("trajctl quote-complete", flag.ContinueOnError)
	fs.SetOutput(stderr)
	quoteFile := fs.String("quote", "", "initial quote JSON")
	ledger := fs.String("ledger", "", "append-only quote ledger")
	witness := fs.String("quality-witness", "", "independently checkable witness reference")
	score := fs.Float64("quality-score", -1, "witnessed quality score")
	cost := fs.Float64("raw-cost", -1, "raw realized cost")
	unit := fs.String("cost-unit", "", "raw cost unit")
	atText := fs.String("at", "", "completion timestamp (RFC3339; defaults now)")
	censored := fs.Bool("censored", false, "cost is right-censored")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *quoteFile == "" || *ledger == "" || *witness == "" || *score < 0 || *cost < 0 || *unit == "" {
		fmt.Fprintln(stderr, "fak trajctl quote-complete: --quote, --ledger, --quality-witness, --quality-score, --raw-cost, and --cost-unit are required")
		return 2
	}
	b, err := os.ReadFile(*quoteFile)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var q trajctl.Quote
	if err := json.Unmarshal(b, &q); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	at := time.Now().UTC()
	if *atText != "" {
		at, err = time.Parse(time.RFC3339, *atText)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
	}
	completion := trajctl.QuoteCompletion{Schema: trajctl.QuoteSchema, QuoteID: q.QuoteID, CreatedAt: at.Format(time.RFC3339Nano), QualityScore: *score, QualityWitness: *witness, Quality: q.Quality, RawRealizedCost: *cost, CostUnit: *unit, Censored: *censored}
	if err := trajctl.AppendQuoteRecord(*ledger, trajctl.QuoteLedgerRecord{Kind: "completion", Completion: &completion}); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return trajctlEmitJSON(stdout, stderr, completion)
}
