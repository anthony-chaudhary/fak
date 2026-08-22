package main

import (
	"flag"
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func runModelProviderContracts(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model provider-contracts", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the complete canonical contracts as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak model provider-contracts [--json]")
		return 2
	}
	if *asJSON {
		raw, err := modelroute.ProviderContractsJSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak model provider-contracts: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	contracts := modelroute.DefaultProviderContracts()
	if err := modelroute.ValidateProviderContracts(contracts); err != nil {
		fmt.Fprintf(stderr, "fak model provider-contracts: %v\n", err)
		return 1
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tFAMILY\tMODEL SCOPE\tENDPOINT\tCACHE\tMATURITY")
	for _, contract := range contracts {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", contract.Provider, contract.Family, contract.ModelScope, displayContractFact(contract.Endpoint), displayContractFact(contract.PromptCaching), displayContractFact(contract.SupportMaturity))
	}
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "fak model provider-contracts: %v\n", err)
		return 1
	}
	return 0
}

func displayContractFact[T any](fact modelroute.ContractFact[T]) string {
	if fact.State != modelroute.KnowledgeKnown {
		return string(fact.State)
	}
	return fmt.Sprint(fact.Value)
}
