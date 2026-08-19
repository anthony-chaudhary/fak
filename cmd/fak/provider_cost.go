package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/providercost"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"io"
	"os"
)

func cmdProviderCost(args []string) { runProviderCost(os.Stdout, os.Stderr, args) }
func runProviderCost(stdout, stderr io.Writer, args []string) {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak provider-cost import|report|reconcile ...")
		return
	}
	fs := flag.NewFlagSet("provider-cost "+args[0], flag.ExitOnError)
	ledger := fs.String("ledger", "", "provider cost JSONL path")
	registry := fs.String("registry", sessionregistry.DefaultPath(), "session registry path")
	input := fs.String("input", "", "provider export JSONL input")
	provider := fs.String("provider", "", "provider for reconciliation")
	expectedRows := fs.Int("expected-rows", -1, "provider-export row count")
	expectedAmount := fs.Int64("expected-micro-usd", -1, "provider-export billed micro-USD; omit when export has no total")
	_ = fs.Parse(args[1:])
	if *ledger == "" {
		fmt.Fprintln(stderr, "fak provider-cost: --ledger is required")
		return
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	switch args[0] {
	case "import":
		if *input == "" {
			fmt.Fprintln(stderr, "fak provider-cost import: --input is required")
			return
		}
		f, err := os.Open(*input)
		if err != nil {
			fatalProviderCost(stderr, err)
			return
		}
		defer f.Close()
		r, err := providercost.Import(*ledger, f)
		if err != nil {
			fatalProviderCost(stderr, err)
			return
		}
		_ = enc.Encode(r)
	case "report":
		rows, err := providercost.Read(*ledger)
		if err != nil {
			fatalProviderCost(stderr, err)
			return
		}
		regs, err := (sessionregistry.Store{Path: *registry}).ReadAll()
		if err != nil {
			fatalProviderCost(stderr, err)
			return
		}
		_ = enc.Encode(providercost.Fold(rows, regs))
	case "reconcile":
		if *provider == "" || *expectedRows < 0 {
			fmt.Fprintln(stderr, "fak provider-cost reconcile: --provider and --expected-rows are required")
			return
		}
		rows, err := providercost.Read(*ledger)
		if err != nil {
			fatalProviderCost(stderr, err)
			return
		}
		var amount *providercost.MicroUSD
		if *expectedAmount >= 0 {
			v := providercost.MicroUSD(*expectedAmount)
			amount = &v
		}
		_ = enc.Encode(providercost.Reconcile(rows, *provider, *expectedRows, amount))
	default:
		fmt.Fprintf(stderr, "fak provider-cost: unknown subcommand %q\n", args[0])
	}
}
func fatalProviderCost(stderr io.Writer, err error) {
	fmt.Fprintf(stderr, "fak provider-cost: %v\n", err)
}
