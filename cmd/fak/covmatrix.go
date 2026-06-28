package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdCoverageMatrix(argv []string) {
	fs := flag.NewFlagSet("coverage-matrix", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit as JSON (default: human-readable)")
	out := fs.String("out", "", "write output to file (default: stdout)")
	_ = fs.Parse(argv)

	// Build the coverage matrix from the internal package
	payload := covmatrix.Build()
	cells := covmatrix.Grid()

	var output []byte
	if *asJSON {
		// Emit the full control-pane payload
		jsonPayload, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak coverage-matrix: %v\n", err)
			os.Exit(1)
		}
		output = jsonPayload
	} else {
		output = []byte(renderHumanReadable(payload, cells))
	}

	if *out == "" {
		os.Stdout.Write(output)
	} else {
		if err := os.WriteFile(*out, output, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "fak coverage-matrix: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Coverage matrix written to: %s\n", *out)
		
		// Extract growth_debt from corpus (int in-process, float64 after a JSON round-trip).
		fmt.Fprintf(os.Stderr, "Growth debt: %d\n", corpusInt(payload.Corpus, covmatrix.DebtKey))
	}
}

func renderHumanReadable(payload scorecard.Payload, cells []covmatrix.Cell) string {
	var b strings.Builder

	// Extract corpus data. corpusInt tolerates int (the in-process Build() value) AND
	// float64 (a JSON round-trip), so the same reader works whether the payload came
	// straight from the fold or was decoded from --json — a plain .(float64) assertion
	// would read 0 for every count in the in-process case.
	families := corpusInt(payload.Corpus, "families")
	backends := corpusInt(payload.Corpus, "backends")
	supported := corpusInt(payload.Corpus, "supported")
	fenced := corpusInt(payload.Corpus, "fenced")
	proofPathOnly := corpusInt(payload.Corpus, "proof_path_only")
	undefined := corpusInt(payload.Corpus, "undefined")

	fmt.Fprintf(&b, "== fak coverage-matrix: %d families × %d backends ==\n", families, backends)
	fmt.Fprintf(&b, "Schema: %s\n", payload.Schema)
	fmt.Fprintf(&b, "Growth debt (silently undefined cells): %d\n\n", undefined)

	// Print summary
	fmt.Fprintf(&b, "Status summary:\n")
	fmt.Fprintf(&b, "  SUPPORTED:       %d\n", supported)
	fmt.Fprintf(&b, "  PROOF-PATH-ONLY: %d\n", proofPathOnly)
	fmt.Fprintf(&b, "  FENCED:          %d\n", fenced)
	fmt.Fprintf(&b, "  UNDEFINED:       %d  <- growth_debt\n\n", undefined)

	// Print matrix as a table
	fmt.Fprintf(&b, "%-18s", "")
	
	// Get sorted backend list
	backendSet := make(map[string]bool)
	for _, c := range cells {
		backendSet[c.Backend] = true
	}
	backendList := make([]string, 0, len(backendSet))
	for b := range backendSet {
		backendList = append(backendList, b)
	}
	sort.Strings(backendList)
	
	for _, backend := range backendList {
		fmt.Fprintf(&b, " %-12s", backend)
	}
	fmt.Fprintln(&b)

	// Get sorted family list
	familySet := make(map[string]bool)
	for _, c := range cells {
		familySet[c.Family] = true
	}
	familyList := make([]string, 0, len(familySet))
	for f := range familySet {
		familyList = append(familyList, f)
	}
	sort.Strings(familyList)

	for _, family := range familyList {
		fmt.Fprintf(&b, "%-18s", family)
		for _, backend := range backendList {
			cell := findCell(cells, family, backend)
			symbol := supportSymbol(cell.Support)
			fmt.Fprintf(&b, " %-12s", symbol)
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Legend:")
	fmt.Fprintln(&b, "  •  SUPPORTED")
	fmt.Fprintln(&b, "  □  PROOF-PATH-ONLY")
	fmt.Fprintln(&b, "  ✗  FENCED")
	fmt.Fprintln(&b, "  ○  UNDEFINED")
	fmt.Fprintln(&b)

	// Print topologies
	fmt.Fprintln(&b, "Family topologies:")
	topologyMap := make(map[string]string)
	for _, f := range covmatrix.Families {
		topologyMap[f.Name] = string(f.Topology)
	}
	for _, family := range familyList {
		if topo, ok := topologyMap[family]; ok {
			fmt.Fprintf(&b, "  %-18s %s\n", family, topo)
		}
	}

	fmt.Fprintln(&b)
	// Print oracle presence
	fmt.Fprintln(&b, "CI-runnable oracle presence:")
	for _, f := range covmatrix.Families {
		oracleStatus := "ABSENT"
		if f.OracleInCI {
			oracleStatus = "PRESENT"
		}
		fmt.Fprintf(&b, "  %-18s %s\n", f.Name, oracleStatus)
	}

	return b.String()
}

// corpusInt reads an integer count from a control-pane corpus, tolerating both the Go
// int the fold writes in-process and the float64 a JSON round-trip produces. Missing or
// non-numeric keys read 0.
func corpusInt(corpus map[string]any, key string) int {
	switch v := corpus[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func findCell(cells []covmatrix.Cell, family, backend string) covmatrix.Cell {
	for _, c := range cells {
		if c.Family == family && c.Backend == backend {
			return c
		}
	}
	return covmatrix.Cell{Support: covmatrix.Undefined}
}

func supportSymbol(s covmatrix.Support) string {
	switch s {
	case covmatrix.Supported:
		return "•"
	case covmatrix.ProofPathOnly:
		return "□"
	case covmatrix.Fenced:
		return "✗"
	case covmatrix.Undefined:
		return "○"
	default:
		return "?"
	}
}