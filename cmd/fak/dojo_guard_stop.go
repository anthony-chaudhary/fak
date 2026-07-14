package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// guardStopLever measures the complement of bad_stop_leak_rate: blocked
// would-be-bad stops divided by all labeled would-be-bad decisions.
type guardStopLever struct{ root string }

func (guardStopLever) Name() string { return "guard-stop" }

func (l guardStopLever) Episodes(_ dojo.Scenario) ([]dojo.ScoredInput, error) {
	path := filepath.Join(l.root, "docs", "nightrun", "guard-stops.jsonl")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []dojo.ScoredInput{guardStopInput(path, 0, 0)}, nil
		}
		return nil, err
	}
	defer f.Close()

	bad, blocked := 0, 0
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
			blocked++
		case "standdown", "failopen":
			bad++
		default:
			if row.Blocked {
				bad++
				blocked++
			}
		}
	}
	if err := scan.Err(); err != nil {
		return nil, fmt.Errorf("read guard stops ledger: %w", err)
	}
	return []dojo.ScoredInput{guardStopInput(path, blocked, bad)}, nil
}

func guardStopInput(path string, blocked, bad int) dojo.ScoredInput {
	realized := 0.0
	if bad > 0 {
		realized = float64(blocked) / float64(bad)
	}
	return dojo.ScoredInput{
		Prediction: dojo.Registry.MustPredict("guard-stop", "bad_stop_block_rate", "fraction"),
		Outcome: dojo.Outcome{
			Realized: realized, Provenance: dojo.Observed, Source: path,
			Measured: bad > 0, Sample: bad,
		},
	}
}
