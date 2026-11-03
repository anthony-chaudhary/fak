package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/godfileceiling"
)

// cmdGodFileCeiling is the thin shell over internal/godfileceiling — the ratcheting god-file LOC
// ceiling gate. This seam makes internal/godfileceiling reachable from the production command graph
// and mirrors TestLiveTreeUnderCeiling as an operator-runnable check. Exit 0 OK, 1 violations, 2 measure/usage error.
func cmdGodFileCeiling(argv []string) {
	os.Exit(runGodFileCeiling(os.Stdout, os.Stderr, argv))
}

// godfileCeilingReport mirrors godfileceiling.Verdict (and its Violation/Shrunk rows) in wire form for --json.
type godfileCeilingViolation struct {
	Path  string `json:"path"`
	Lines int    `json:"lines"`
	Cap   int    `json:"cap"`
	Over  int    `json:"over"`
	Kind  string `json:"kind"`
}

type godfileCeilingShrunk struct {
	Path  string `json:"path"`
	Lines int    `json:"lines"`
	Cap   int    `json:"cap"`
	Under int    `json:"under"`
}

type godfileCeilingReport struct {
	OK         bool                      `json:"ok"`
	NFiles     int                       `json:"n_files"`
	NPinned    int                       `json:"n_pinned"`
	Violations []godfileCeilingViolation `json:"violations"`
	Shrunk     []godfileCeilingShrunk    `json:"shrunk"`
	StalePins  []string                  `json:"stale_pins"`
}

func runGodFileCeiling(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("godfile-ceiling", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repo root to measure (git ls-files scope)")
	asJSON := fs.Bool("json", false, "emit the verdict as strict JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	measured, err := godfileceiling.MeasureTree(*root)
	if err != nil {
		fmt.Fprintf(stderr, "GODFILE_CEILING_MEASURE_FAILED: %v\n", err)
		return 2
	}
	verdict := godfileceiling.Evaluate(measured, godfileceiling.Baseline)
	if *asJSON {
		rep := godfileCeilingReport{
			OK: verdict.OK, NFiles: verdict.NFiles, NPinned: verdict.NPinned,
			Violations: make([]godfileCeilingViolation, 0, len(verdict.Violations)),
			Shrunk:     make([]godfileCeilingShrunk, 0, len(verdict.Shrunk)),
			StalePins:  make([]string, 0, len(verdict.StalePins)),
		}
		for _, v := range verdict.Violations {
			rep.Violations = append(rep.Violations, godfileCeilingViolation{v.Path, v.Lines, v.Cap, v.Over, v.Kind})
		}
		for _, s := range verdict.Shrunk {
			rep.Shrunk = append(rep.Shrunk, godfileCeilingShrunk{s.Path, s.Lines, s.Cap, s.Under})
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(&rep); err != nil {
			fmt.Fprintf(stderr, "godfile-ceiling: encode JSON: %v\n", err)
			return 2
		}
	} else {
		if verdict.OK {
			fmt.Fprintf(stdout, "godfile-ceiling: OK — %d files measured, %d pinned, no violations\n", verdict.NFiles, verdict.NPinned)
		} else {
			for _, v := range verdict.Violations {
				fmt.Fprintf(stdout, "%s: %s — %d lines > cap %d (+%d)\n", v.Path, v.Kind, v.Lines, v.Cap, v.Over)
			}
			fmt.Fprintf(stdout, "godfile-ceiling: FAIL — %d violation(s)\n", len(verdict.Violations))
		}
		for _, s := range verdict.Shrunk {
			fmt.Fprintf(stdout, "re-pin opportunity: %s shrank to %d (cap %d)\n", s.Path, s.Lines, s.Cap)
		}
		for _, p := range verdict.StalePins {
			fmt.Fprintf(stdout, "stale pin: %s\n", p)
		}
	}
	if verdict.OK {
		return 0
	}
	return 1
}
