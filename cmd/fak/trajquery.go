package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/trajquery"
)

// cmdTrajQuery handles `fak trajquery <subcommand>` — query your OWN trajectory corpus
// with a small SQL SELECT, confined to an operator-published scope by view rewrite. The
// validator refuses any query that would escape the scope. See internal/trajquery.
func cmdTrajQuery(args []string) {
	if len(args) == 0 {
		trajQueryUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "run":
		cmdTrajQueryRun(args[1:])
	case "validate":
		cmdTrajQueryValidate(args[1:])
	case "-h", "--help", "help":
		trajQueryUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak trajquery: unknown subcommand %q\n", args[0])
		trajQueryUsage()
		os.Exit(2)
	}
}

func trajQueryUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak trajquery <run|validate> --view <view.json> --sql <SELECT ...> [--corpus <rows.jsonl>]")
	fmt.Fprintln(os.Stderr, "       fak trajquery validate --view v.json --sql \"SELECT id FROM myturns WHERE role='agent'\"")
	fmt.Fprintln(os.Stderr, "            (report whether the query is scope-safe; exit 1 if it escapes)")
	fmt.Fprintln(os.Stderr, "       fak trajquery run --view v.json --corpus rows.jsonl --sql \"SELECT * FROM myturns\"")
	fmt.Fprintln(os.Stderr, "            (validate, rewrite to enforce scope, then execute over the corpus)")
	fmt.Fprintln(os.Stderr, "  The query must target the view name; querying the base relation is refused as a scope escape.")
}

func loadView(path string) trajquery.View {
	return loadJSONFileOrExit[trajquery.View](path, "fak trajquery")
}

func loadRowCorpus(path string) []trajquery.Row {
	return readJSONLCorpus[trajquery.Row](path, "fak trajquery")
}

func cmdTrajQueryValidate(args []string) {
	fs := flag.NewFlagSet("trajquery validate", flag.ExitOnError)
	viewPath := fs.String("view", "", "view definition (JSON)")
	sql := fs.String("sql", "", "the SELECT to validate against the view")
	corpusPath := fs.String("corpus", "", "optional corpus JSONL for the dynamic no-leak check")
	_ = fs.Parse(args)
	v := loadView(*viewPath)
	q, err := trajquery.Parse(*sql)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse error: %v\n", err)
		os.Exit(1)
	}
	rep := v.Validate(q, loadRowCorpus(*corpusPath))
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(rep)
	if !rep.Valid {
		os.Exit(1)
	}
}

func cmdTrajQueryRun(args []string) {
	fs := flag.NewFlagSet("trajquery run", flag.ExitOnError)
	viewPath := fs.String("view", "", "view definition (JSON)")
	sql := fs.String("sql", "", "the SELECT to run against the view")
	corpusPath := fs.String("corpus", "", "corpus JSONL to execute over")
	_ = fs.Parse(args)
	v := loadView(*viewPath)
	q, err := trajquery.Parse(*sql)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse error: %v\n", err)
		os.Exit(1)
	}
	corpus := loadRowCorpus(*corpusPath)
	// Validate first: a query that escapes the scope must never execute.
	if rep := v.Validate(q, corpus); !rep.Valid {
		fmt.Fprintf(os.Stderr, "refused (scope violation): %v\n", rep.Violations)
		os.Exit(1)
	}
	rewritten, err := v.Rewrite(q)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rewrite error: %v\n", err)
		os.Exit(1)
	}
	rows, err := rewritten.Execute(corpus)
	if err != nil {
		fmt.Fprintf(os.Stderr, "execute error: %v\n", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	for _, r := range rows {
		enc.Encode(r)
	}
}
