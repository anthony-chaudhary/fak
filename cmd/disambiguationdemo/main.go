package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

type receipt struct {
	Schema        string                       `json:"schema"`
	Query         string                       `json:"query"`
	SearchVerdict disambiguation.SearchVerdict `json:"search_verdict"`
	Choices       []string                     `json:"choices"`
	SelectedScope disambiguation.Scope         `json:"selected_scope"`
	CanonicalTerm string                       `json:"canonical_term"`
	Contrast      string                       `json:"contrast"`
	Offline       bool                         `json:"offline"`
}

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }
func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguationdemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run deterministic overloaded-term lookup")
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfcheck || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: disambiguationdemo -selfcheck [-json]")
		return 2
	}
	r, err := selfcheckReceipt()
	if err != nil {
		fmt.Fprintf(stderr, "SELFCHECK FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encode(stdout, r)
	}
	fmt.Fprintf(stdout, "QUERY %s: %s choices=%v\n", r.Query, r.SearchVerdict, r.Choices)
	fmt.Fprintf(stdout, "SCOPE %s=%s -> %s\n", r.SelectedScope.Kind, r.SelectedScope.Value, r.CanonicalTerm)
	fmt.Fprintf(stdout, "CONTRAST %s\n", r.Contrast)
	fmt.Fprintln(stdout, "SELFCHECK PASS: public local index only; no model, key, GPU, network, or private data")
	return 0
}
func selfcheckReceipt() (receipt, error) {
	search := disambiguation.Search("runtime")
	choices := []string{}
	for _, match := range search.Groups.Exact {
		choices = append(choices, match.Entry.Scope.Value)
	}
	scope := disambiguation.Scope{Kind: "runtime", Value: "gateway-serving"}
	selected, err := disambiguation.QueryScoped("runtime", scope)
	if err != nil {
		return receipt{}, err
	}
	if len(selected.Entry.Contrasts) == 0 {
		return receipt{}, fmt.Errorf("selected term has no contrast")
	}
	return receipt{Schema: "fak-disambiguation-demo/1", Query: "runtime", SearchVerdict: search.Verdict, Choices: choices, SelectedScope: scope, CanonicalTerm: selected.Entry.Identity.CanonicalTerm, Contrast: selected.Entry.Contrasts[0].Explanation, Offline: true}, nil
}
func encode(w io.Writer, v any) int {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return 1
	}
	return 0
}
