package experiments

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func ParseLedger(content string) []Experiment {
	var rows []Experiment
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row Experiment
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.ID == "" || row.Owner == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

func ReadLedgerFile(path string) []Experiment {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(b))
}

func ReadAllLedgers(root string) []Experiment {
	var all []Experiment
	
	// First read the default registry.jsonl if it exists
	defaultLedger := filepath.Join(root, "experiments", "registry.jsonl")
	if b, err := os.ReadFile(defaultLedger); err == nil {
		all = append(all, ParseLedger(string(b))...)
	}
	
	// Then read per-host registries under experiments/
	root = filepath.Join(root, "experiments")
	entries, err := os.ReadDir(root)
	if err != nil {
		return all
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		hostDir := filepath.Join(root, e.Name())
		ledgerPath := filepath.Join(hostDir, "registry.jsonl")
		if _, err := os.Stat(ledgerPath); err != nil {
			continue
		}
		all = append(all, ReadLedgerFile(ledgerPath)...)
	}
	return all
}

func FindOverlaps(experiments []Experiment, models, backends []string) []Experiment {
	var overlaps []Experiment
	targetCells := make(map[Cell]bool)
	for _, m := range models {
		for _, b := range backends {
			targetCells[Cell{Model: m, Backend: b}] = true
		}
	}
	for _, e := range experiments {
		for _, c := range e.Cells() {
			if targetCells[c] {
				overlaps = append(overlaps, e)
				break
			}
		}
	}
	return overlaps
}

func AppendLedgerLine(exp Experiment) (string, error) {
	b, err := json.Marshal(exp)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func WriteLedgerLine(path string, exp Experiment) error {
	line, err := AppendLedgerLine(exp)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintf(f, "%s\n", line)
	return err
}