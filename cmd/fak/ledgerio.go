package main

// ledgerio.go is the one cmd-side durable-ledger file-IO seam the trend-report
// subcommands (cadence, dojo, milestone) share. Each report package owns its
// row type, tolerant parser, and line renderer; this file owns only the disk
// plumbing around them — read-if-present and append-with-mkdir — so a fourth
// report subcommand wires its ledger without re-declaring either helper
// (#1437).

import (
	"os"
	"path/filepath"
)

// readLedgerFile reads the durable ledger if present and parses it with the
// report package's own tolerant parser (absent ledger -> no prior rows, the
// first tick establishes the series).
func readLedgerFile[T any](path string, parse func(string) []T) []T {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return parse(string(raw))
}

// appendLedgerFile appends one JSONL row to the ledger via the report
// package's own line renderer, creating the parent directory on first write.
func appendLedgerFile[T any](path string, row T, render func(T) (string, error)) error {
	line, err := render(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}
