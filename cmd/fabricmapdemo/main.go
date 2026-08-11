// fabricmapdemo is a runnable proof of direction-agnostic, composable data movement.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the built-in arbitrary-direction mapping proof")
	manifest := flag.String("manifest", "", "JSON file containing graph and request")
	flag.Parse()
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
			fatal(err)
		}
		if err := json.Unmarshal(data, &input); err != nil {
			fatal(err)
		}
	default:
		fmt.Fprintln(os.Stderr, "usage: fabricmapdemo -selfcheck | -manifest input.json")
		os.Exit(2)
	}
	route, err := input.Graph.Plan(input.Request)
	if err != nil {
		fatal(err)
	}
	data, _ := json.MarshalIndent(route, "", "  ")
	fmt.Println(string(data))
	if *selfcheck {
		if len(route.Links) != 1 || route.Links[0].ID != "ssd-to-gpu-direct" {
			fatal(fmt.Errorf("unexpected proof route"))
		}
		fmt.Fprintln(os.Stderr, "PASS: L3 is a user label, and the directed L3 -> L1 CPU-bypass link was selected")
	}
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
func fatal(err error) { fmt.Fprintln(os.Stderr, "fabricmapdemo:", err); os.Exit(1) }
