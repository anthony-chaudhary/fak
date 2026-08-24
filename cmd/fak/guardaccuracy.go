package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/guardaccuracy"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdGuardAccuracy(argv []string) { os.Exit(runGuardAccuracy(os.Stdout, os.Stderr, argv)) }

// runGuardAccuracy scores the guard reversibility classifier's accuracy against
// a labeled command corpus: the false-positive rate (benign calls escalated) and
// false-negative rate (dangerous calls let through). It is the accuracy dual of
// `fak guard-verdict-rsi` -- honesty asks whether verdicts are explained, this
// asks whether the escalate/don't boundary is correct -- emitted as the same
// control-pane scorecard the garden/control-pane ratchet folds worst-first.
func runGuardAccuracy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-accuracy", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	corpusPath := fs.String("corpus", "", "override corpus file (default: the embedded seed corpus)")
	complaintsPath := fs.String("complaints", "", "fold agent-authored guard complaints (JSON) as an advisory field false-positive intake")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-accuracy: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	rows := guardaccuracy.SeedCorpus()
	if *corpusPath != "" {
		data, err := os.ReadFile(*corpusPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-accuracy: read corpus: %v\n", err)
			return 2
		}
		rows, err = guardaccuracy.LoadCorpus(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-accuracy: parse corpus: %v\n", err)
			return 2
		}
	}

	// The complaint intake is advisory (Soft, never debt): folding agent-authored
	// over-block appeals surfaces a field false-positive triage queue without ever
	// reding the gate. Absent --complaints, this is BuildScorecard's exact payload.
	var complaints []guardaccuracy.FieldComplaint
	if *complaintsPath != "" {
		data, err := os.ReadFile(*complaintsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-accuracy: read complaints: %v\n", err)
			return 2
		}
		complaints, err = guardaccuracy.LoadComplaints(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-accuracy: parse complaints: %v\n", err)
			return 2
		}
	}

	payload := guardaccuracy.BuildScorecardWithComplaints(root, rows, complaints)

	return emitScorecard(stdout, stderr, "fak guard-accuracy", guardaccuracy.DebtKey, payload,
		*comparePath, *asJSON, *asMarkdown, scorecard.MarkdownDoc{
			Title:       "fak guard accuracy scorecard",
			Description: "How well-tuned the guard reversibility classifier is, scored by folding a labeled command corpus through the real classifier: false-positive rate (benign escalated) and false-negative rate (dangerous let through).",
			Heading:     "fak guard accuracy scorecard",
			DebtKey:     guardaccuracy.DebtKey,
			HeaderExtra: fmt.Sprintf(" - fp rate %v - fn rate %v - %v corpus row(s)",
				payload.Corpus["fp_rate"], payload.Corpus["fn_rate"], payload.Corpus["total_rows"]),
		})
}
