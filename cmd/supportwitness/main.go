package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/supportgraph"
)

func main() { os.Exit(run(os.Args[1:])) }
func run(args []string) int {
	fs := flag.NewFlagSet("supportwitness", flag.ContinueOnError)
	graphPath := fs.String("graph", "", "support graph JSON")
	witnessPath := fs.String("witness", "", "support witness JSON")
	outputPath := fs.String("out", "", "write updated graph (stdout when empty)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *graphPath == "" || *witnessPath == "" {
		fmt.Fprintln(os.Stderr, "usage: supportwitness -graph G -witness W [-out G2]")
		return 2
	}
	graphRaw, err := os.ReadFile(*graphPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	graph, err := supportgraph.Parse(graphRaw)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	witnessRaw, err := os.ReadFile(*witnessPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	witness, err := supportgraph.ParseWitness(witnessRaw)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	result, err := supportgraph.Ingest(graph, witness)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	raw, _ := json.MarshalIndent(result.Graph, "", "  ")
	raw = append(raw, '\n')
	if *outputPath != "" {
		if err := os.WriteFile(*outputPath, raw, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	} else {
		_, _ = os.Stdout.Write(raw)
	}
	fmt.Fprintf(os.Stderr, "INGEST %s inserted=%t stale_edges=%d\n", result.WitnessID, result.Inserted, result.StaleEdges)
	return 0
}
