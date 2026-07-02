package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/nodecompare"
)

func cmdNodeCompare(argv []string) { os.Exit(runNodeCompare(os.Stdout, os.Stderr, argv)) }

func runNodeCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("node-compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("workspace", "", "workspace root (default: repo root)")
	nodesDir := fs.String("nodes-dir", "", "fleet node result directory")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	workspace := *root
	if workspace == "" {
		workspace = resolveRoot("")
		if workspace == "" {
			workspace = "."
		}
	}
	dir := *nodesDir
	if dir == "" {
		dir = filepath.Join(workspace, "fak", "experiments", "fleet-nodes")
	}
	nodes, err := nodecompare.LoadNodes(dir)
	if err != nil {
		fmt.Fprintf(stderr, "node-compare: %v\n", err)
		return 2
	}
	if len(nodes) == 0 {
		fmt.Fprintf(stderr, "no node results under %s\nrun `bash tools/fak_node_bench.sh` on each node first.\n", dir)
		return 1
	}
	if *asJSON {
		data, err := json.MarshalIndent(nodes, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "node-compare: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, nodecompare.Render(nodes))
	}
	return 0
}
