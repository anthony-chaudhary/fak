// Package jsonlledger parses a JSONL ledger (one JSON object per line) into a
// slice of typed rows, skipping blank lines, unparseable lines, and rows the
// keep predicate rejects. It replaces the byte-identical ParseLedger scan body
// that was copy-pasted across the report packages (cadencereport,
// milestonereport, programreport, …): each keeps its own row type and delegates
// the scan here, so the duplicated loop lives in exactly one place.
package jsonlledger

import (
	"bufio"
	"encoding/json"
	"strings"
)

// Parse scans content as JSONL, unmarshaling each non-blank line into a T and
// appending it when keep(row) reports true. Blank and malformed lines are
// skipped. A nil keep accepts every well-formed row. The 1 MiB line buffer
// matches the copies this consolidates, so long ledger lines still parse.
func Parse[T any](content string, keep func(T) bool) []T {
	var rows []T
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if keep != nil && !keep(row) {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}
