package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devexmeter"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("devexmeter", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuePath := fs.String("issue", "", "GitHub issue JSON file (gh issue view --json number,title,body,labels)")
	ledgerPath := fs.String("ledger", "", "devexmeter JSONL ledger")
	issueNumber := fs.Int("issue-number", 0, "issue number when --issue is not supplied")
	class := fs.String("class", "", "friction class when --issue is not supplied")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *ledgerPath == "" {
		usage(stderr)
		return 2
	}

	ledgerBytes, err := readFileOrStdin(*ledgerPath)
	if err != nil {
		fmt.Fprintf(stderr, "devexmeter: read ledger: %v\n", err)
		return 2
	}
	rows, err := devexmeter.ParseLedger(ledgerBytes)
	if err != nil {
		fmt.Fprintf(stderr, "devexmeter: %v\n", err)
		return 2
	}

	issue, err := loadIssue(*issuePath, *issueNumber, *class)
	if err != nil {
		fmt.Fprintf(stderr, "devexmeter: %v\n", err)
		return 2
	}
	result := devexmeter.GateIssue(issue, rows)
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintf(stderr, "devexmeter: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, render(result))
	}
	if result.OK {
		return 0
	}
	return 3
}

func loadIssue(path string, number int, class string) (devexmeter.Issue, error) {
	if strings.TrimSpace(path) != "" {
		b, err := readFileOrStdin(path)
		if err != nil {
			return devexmeter.Issue{}, fmt.Errorf("read issue: %w", err)
		}
		return devexmeter.ParseIssue(b)
	}
	class = strings.TrimSpace(class)
	if number <= 0 || class == "" {
		return devexmeter.Issue{}, fmt.Errorf("pass --issue FILE or both --issue-number N --class CLASS")
	}
	return devexmeter.Issue{Number: number, Labels: []string{"dev-ex", "friction/" + class}}, nil
}

func readFileOrStdin(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}

func render(result devexmeter.GateResult) string {
	line := fmt.Sprintf("devexmeter gate: %s issue=#%d", result.Verdict, result.Issue)
	if result.Class != "" {
		line += " class=" + result.Class
	}
	if result.Before != nil && result.After != nil && result.Delta != nil {
		line += fmt.Sprintf(" before=%.4g after=%.4g delta=%.4g", *result.Before, *result.After, *result.Delta)
	}
	line += " - " + result.Reason
	if len(result.MissingWitness) > 0 {
		line += " missing_witness=[" + strings.Join(result.MissingWitness, "; ") + "]"
	}
	return line
}

func usage(w io.Writer) {
	fmt.Fprint(w, `devexmeter - dev-ex friction close gate

  devexmeter --issue ISSUE.json --ledger meter.jsonl [--json]
  devexmeter --issue-number N --class CLASS --ledger meter.jsonl [--json]

Rows are JSONL with schema fak.devexmeter.row.v1:
  {"issue":2166,"class":"retry-after-refusal","window":"before","value":12}
  {"issue":2166,"class":"retry-after-refusal","window":"after","value":7}

The gate returns exit 0 for PASS/SKIP and exit 3 for NOT_YET.
`)
}
