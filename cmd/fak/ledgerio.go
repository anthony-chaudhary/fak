package main

// ledgerio.go is the one cmd-side durable-ledger file-IO seam the trend-report
// subcommands (cadence, dojo, milestone) share. Each report package owns its
// row type, tolerant parser, and line renderer; this file owns only the disk
// plumbing around them — read-if-present and append-with-mkdir — so a fourth
// report subcommand wires its ledger without re-declaring either helper
// (#1437).

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// readLedgerText reads a durable ledger as raw text. An absent file is a valid EMPTY view
// (nothing has been witnessed yet), not an error; any other read error is still returned.
// Unlike readLedgerFile it parses nothing, for the ledgers whose callers fold the raw text
// themselves.
func readLedgerText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(b), nil
}

// appendJSONLRows appends one JSON line per row to logName under runsDir, creating runsDir
// on first write. Append-only. An EMPTY row set writes nothing at all — not even the
// directory — which is what keeps a no-op pass from materialising an empty ledger.
func appendJSONLRows[T any](runsDir, logName string, rows []T) error {
	if len(rows) == 0 {
		return nil
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(runsDir, logName), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

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

// appendReportHistory owns the common operator contract around the report ledgers:
// append only when requested, preserve the command-specific error prefix, and announce
// the repository-relative ledger only for human non-gating output.
func appendReportHistory[T any](stdout, stderr io.Writer, enabled, announce bool, root, ledgerPath, command, rowName string, row T, render func(T) (string, error)) int {
	if !enabled {
		return 0
	}
	if err := appendLedgerFile(ledgerPath, row, render); err != nil {
		fmt.Fprintf(stderr, "fak %s: append ledger: %v\n", command, err)
		return 1
	}
	if announce {
		rel, err := filepath.Rel(root, ledgerPath)
		if err != nil || rel == "" {
			rel = ledgerPath
		}
		fmt.Fprintf(stdout, "appended %s row -> %s\n", rowName, filepath.ToSlash(rel))
	}
	return 0
}
