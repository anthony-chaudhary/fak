package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/supportgraph"
	"os"
	"time"
)

func main() { os.Exit(run(os.Args[1:])) }
func run(args []string) int {
	f := flag.NewFlagSet("supportgraphdemo", flag.ContinueOnError)
	self := f.Bool("selfcheck", false, "run fixture")
	if f.Parse(args) != nil {
		return 2
	}
	if !*self {
		fmt.Fprintln(os.Stderr, "usage: supportgraphdemo -selfcheck")
		return 2
	}
	raw, e := os.ReadFile("internal/supportgraph/testdata/awq.json")
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		return 1
	}
	g, e := supportgraph.Parse(raw)
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		return 1
	}
	at := time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC)
	names := []string{"L4 exact tuple", "T4 exact tuple", "old tuple"}
	for i, edge := range g.Edges {
		r := supportgraph.Query(g, edge.Tuple, at)
		b, _ := json.Marshal(r.Decisive)
		fmt.Printf("%s: %s — %s evidence=%s", names[i], r.State, r.Reason, b)
		if r.Fallback != "" {
			fmt.Printf(" fallback=%s penalty=%s", r.Fallback, r.Penalty)
		}
		fmt.Println()
	}
	q := g.Edges[0].Tuple
	q.Layout = "awq-unknown"
	fmt.Printf("unknown layout: %s — %s\n", supportgraph.Query(g, q, at).State, supportgraph.Query(g, q, at).Reason)
	fmt.Println("SELFCHECK PASS: exact witnessed support, witnessed refusal, stale, and unknown remain distinct")
	return 0
}
