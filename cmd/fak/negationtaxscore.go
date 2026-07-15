package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdNegationTaxScore(argv []string) { os.Exit(runNegationTaxScore(os.Stdout, os.Stderr, argv)) }

func runNegationTaxScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score negation-tax", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON")
	compare := fs.String("ratchet", "", "refuse debt above baseline scorecard JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	payload := negframe.BuildNegationTax(negframe.GuardRuntimeCorpus())
	if *compare != "" {
		b, err := os.ReadFile(*compare)
		if err != nil {
			fmt.Fprintf(stderr, "fak score negation-tax: ratchet: %v\n", err)
			return 2
		}
		var prior scorecard.Payload
		if json.Unmarshal(b, &prior) != nil {
			fmt.Fprintln(stderr, "fak score negation-tax: invalid ratchet JSON")
			return 2
		}
		cur := payload.Corpus[negframe.NegationTaxDebtKey].(int)
		old, ok := scoreInt(prior.Corpus[negframe.NegationTaxDebtKey])
		if !ok {
			return 2
		}
		if cur > old {
			fmt.Fprintf(stderr, "fak score negation-tax: ratchet regression %d > %d\n", cur, old)
			return 1
		}
	}
	if *asJSON {
		b, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "negation_tax_debt=%v verdict=%s\n", payload.Corpus[negframe.NegationTaxDebtKey], payload.Verdict)
	}
	if payload.OK {
		return 0
	}
	return 1
}
func scoreInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	}
	return 0, false
}
