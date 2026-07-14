package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// guardIntegrityLever measures the Stop guard's zero-leak floor from the
// durable decision ledger. A would-be-bad stop is one the guard either blocked
// (continue) or allowed only because its bounded ladder stood down / the guard
// failed open. Clean, shadow, and disabled observations are not labeled bad and
// therefore are not part of this denominator.
type guardIntegrityLever struct{ root string }

func (guardIntegrityLever) Name() string { return "guard-integrity" }

type guardIntegrityRow struct {
	Kind        string `json:"kind"`
	Disposition string `json:"disposition"`
	Blocked     bool   `json:"blocked"`
}

func (l guardIntegrityLever) Episodes(_ dojo.Scenario) ([]dojo.ScoredInput, error) {
	path := filepath.Join(l.root, "docs", "nightrun", "guard-stops.jsonl")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []dojo.ScoredInput{guardIntegrityInput(path, 0, 0)}, nil
		}
		return nil, err
	}
	defer f.Close()

	bad, leaked := 0, 0
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64*1024), 1024*1024)
	for scan.Scan() {
		var row guardIntegrityRow
		if json.Unmarshal(scan.Bytes(), &row) != nil {
			continue
		}
		switch row.Kind {
		case "continue":
			bad++
		case "standdown", "failopen":
			bad++
			leaked++
		default:
			// Backward-compatible classification for rows written before kind.
			if row.Blocked {
				bad++
			}
		}
	}
	if err := scan.Err(); err != nil {
		return nil, fmt.Errorf("read guard stops ledger: %w", err)
	}
	return []dojo.ScoredInput{guardIntegrityInput(path, leaked, bad)}, nil
}

func guardIntegrityInput(path string, leaked, bad int) dojo.ScoredInput {
	realized := 0.0
	if bad > 0 {
		realized = float64(leaked) / float64(bad)
	}
	return dojo.ScoredInput{
		Prediction: dojo.Registry.MustPredict("guard-integrity", "bad_stop_leak_rate", "fraction"),
		Outcome: dojo.Outcome{
			Realized:   realized,
			Provenance: dojo.Observed,
			Source:     path,
			Measured:   bad > 0,
			Sample:     bad,
		},
	}
}
