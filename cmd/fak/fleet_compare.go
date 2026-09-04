package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/fleetcompare"
)

// `fak fleet compare` or `fak fleet-compare`: slices fleet sweep columns across
// a fixed key ("agents" or "turns") and decomposes shared vs isolated/cross savings.
func runFleetCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	key := fs.String("key", "agents", "fixed parameter key to slice on ('agents' or 'turns')")
	valStr := fs.String("val", "", "fixed parameter value (e.g. 50)")
	file := fs.String("file", "", "JSON input file path containing simulation columns (or - for stdin)")
	asJSON := fs.Bool("json", false, "emit output as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *valStr == "" {
		fmt.Fprintln(stderr, "fleet compare: --val is required")
		return 2
	}
	val, err := strconv.ParseFloat(*valStr, 64)
	if err != nil {
		fmt.Fprintf(stderr, "fleet compare: invalid --val: %v\n", err)
		return 2
	}

	var data []byte
	if *file == "" || *file == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(*file)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fleet compare: read file: %v\n", err)
		return 2
	}

	var cols map[string][]float64
	if err := json.Unmarshal(data, &cols); err != nil {
		fmt.Fprintf(stderr, "fleet compare: parse json columns: %v\n", err)
		return 2
	}

	slice := fleetcompare.SliceFixed(cols, *key, val)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(slice); err != nil {
			fmt.Fprintf(stderr, "fleet compare: encode json: %v\n", err)
			return 2
		}
		return 0
	}

	fmt.Fprintf(stdout, "fixed %s=%.2f: %d points\n", *key, val, len(slice.Xs))
	for i, x := range slice.Xs {
		fmt.Fprintf(stdout, "  x=%.2f shared=%.2f isolated=%.2f cross=%.2f\n",
			x, slice.Shared[i], slice.Isolated[i], slice.Cross[i])
	}
	return 0
}
