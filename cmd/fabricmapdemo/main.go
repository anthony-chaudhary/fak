// fabricmapdemo is a runnable proof of direction-agnostic, composable data movement.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fabricmapdemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run the built-in arbitrary-direction mapping proof")
	manifest := fs.String("manifest", "", "JSON file containing graph and request")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	var input struct {
		Graph   fabricmap.Graph   `json:"graph"`
		Request fabricmap.Request `json:"request"`
	}
	switch {
	case *selfcheck:
		input = proofInput()
	case *manifest != "":
		data, err := os.ReadFile(*manifest)
		if err != nil {
			return fail(stderr, err)
		}
		if err := json.Unmarshal(data, &input); err != nil {
			return fail(stderr, err)
		}
	default:
		fmt.Fprintln(stderr, "usage: fabricmapdemo -selfcheck | -manifest input.json")
		return 2
	}
	route, err := input.Graph.Plan(input.Request)
	if err != nil {
		return fail(stderr, err)
	}
	data, err := json.MarshalIndent(route, "", "  ")
	if err != nil {
		return fail(stderr, err)
	}
	fmt.Fprintln(stdout, string(data))
	if *selfcheck {
		if len(route.Links) != 1 || route.Links[0].ID != "ssd-to-gpu-direct" {
			return fail(stderr, fmt.Errorf("unexpected proof route"))
		}
		fmt.Fprintln(stderr, "PASS: L3 is a user label, and the directed L3 -> L1 CPU-bypass link was selected")
	}
	return 0
}
func proofInput() (in struct {
	Graph   fabricmap.Graph   `json:"graph"`
	Request fabricmap.Request `json:"request"`
}) {
	in.Graph = fabricmap.Graph{Endpoints: []fabricmap.Endpoint{{ID: "L1", Kind: "gpu-memory"}, {ID: "L2", Kind: "host-memory"}, {ID: "L3", Kind: "ssd"}}, Links: []fabricmap.Link{
		{ID: "ssd-to-host", From: "L3", To: "L2", Transport: "nvme-copy", Cost: 2, CPUPath: "copy", Labels: map[string]string{"gpu-direct": "no"}},
		{ID: "host-to-gpu", From: "L2", To: "L1", Transport: "pcie-copy", Cost: 2, CPUPath: "copy", Labels: map[string]string{"gpu-direct": "no"}},
		{ID: "ssd-to-gpu-direct", From: "L3", To: "L1", Transport: "gds-rdma", Cost: 1, CPUPath: "bypass", Labels: map[string]string{"gpu-direct": "yes"}},
	}}
	in.Request = fabricmap.Request{From: "L3", To: "L1", AllowedCPUPaths: []string{"bypass"}, RequiredLinkLabels: map[string]string{"gpu-direct": "yes"}}
	return in
}
func fail(stderr io.Writer, err error) int {
	fmt.Fprintln(stderr, "fabricmapdemo:", err)
	return 1
}
